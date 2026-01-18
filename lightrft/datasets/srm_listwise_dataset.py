import os
import copy
import random
import torch
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor
from loguru import logger

from .utils import load_multimodal_content, zero_pad_sequences
from .image_reward_db import ImageRewardDBListwiseHandler

class RankDatasetListwiseVL(Dataset):
    """
    Listwise Preference ranking dataset used for vision-language scalar reward model (SRM) training.
    """
    def __init__(
        self,
        dataset_paths: List[str],
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        strategy=None,
        max_length: int = 4096,
        list_size: int = 4,
        config: Dict[str, Any] = None
    ):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.list_size = list_size
        self.config = config if config else {}
        
        self.media_content_loader = load_multimodal_content
        
        # Processor config setup (similar to RankDatasetVL)
        if "qwen" in self.processor.__class__.__name__.lower():
            self.image_processor = self.processor.image_processor
            self.tokenizer = self.processor.tokenizer
        else:
            self.image_processor = self.processor.image_processor

        # Use Listwise Handlers
        self.handlers = {
            "imagerewarddb": ImageRewardDBListwiseHandler(),
            # Add other handlers if they support listwise
        }

        self.data = []
        for item in dataset_paths:
            if ':' in item:
                source, path = item.split(':', 1)
            else:
                # Default fallback or error
                logger.warning(f"Invalid dataset path format: {item}. Skipping.")
                continue

            if source not in self.handlers:
                logger.warning(f"Handler for source {source} not found. Skipping.")
                continue
                
            handler = self.handlers[source]
            
            # Use standard load_data with optional list_size param?
            # Or assume handlers for listwise dataset have list_size
            try:
                data = handler.load_data(path, list_size=self.list_size)
            except TypeError:
                 # Fallback if load_data doesn't support list_size (should not happen for ListwiseHandler)
                 logger.warning(f"Handler {source} might not support list_size parameter.")
                 data = handler.load_data(path)
                 
            self.data.extend(data)

        logger.info(f"Loaded {len(self.data)} listwise samples in total.")
        random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        source = item.get('source', 'imagerewarddb')
        handler = self.handlers[source]
        
        # Load media content using standard interface
        media_info = handler.get_media_info(item)
        
        # Only ImageRewardDB supported for now, which is text-to-image
        # media_info is dict: {"image_0": path, "image_1": path ...}
        
        media_content = self.media_content_loader(media_info)
        
        # Prepare inputs for K candidates
        # Dynamic Sampling: Pick K items from the full candidate list available for this prompt
        full_candidates = item['candidates']
        full_ranks = item['ranks']

        # Determine target size
        if self.list_size is not None and self.list_size > 0:
            target_size = self.list_size
        else:
            target_size = len(full_candidates)
        
        if len(full_candidates) > target_size:
            # Randomly sample Indices
            sampled_indices = random.sample(range(len(full_candidates)), target_size)
            candidates = [full_candidates[i] for i in sampled_indices]
            ranks = [full_ranks[i] for i in sampled_indices]
            
            sampled_images = []
            for i in sampled_indices:
                key = f"image_{i}"
                sampled_images.append(media_content[key])
        else:
            candidates = full_candidates
            ranks = full_ranks
            # If len < list_size, we shouldn't be here due to load_data filtering, but just in case:
            # We take all available.
            sampled_images = [media_content[f"image_{i}"] for i in range(len(candidates))]

        # Shuffle the sampled K items to avoid positional bias
        combined = list(zip(sampled_images, ranks))
        random.shuffle(combined)
        sampled_images, ranks = zip(*combined)
        
        prompt_text = item['prompt']
        
        task_instruction_template = self.config.get("task_instruction", "")
        task_instruction = task_instruction_template.format(prompt=prompt_text)
        max_pixels = self.config.get("max_pixels", 224*224)
        
        # We need to construct K separate inputs (Prompt + Image_k)
        # Because SRM is Point-wise inference model.
        
        all_input_ids = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []
        
        for i in range(len(sampled_images)):
            image = sampled_images[i]
            
            # Message structure for Qwen-VL or similar
            messages = [
                {"role": "system", "content": task_instruction},
                {"role": "user", "content": [
                    {"type": "image", "image": image, "max_pixels": max_pixels}
                ]}
            ]
            
            # Prepare text using chat template
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=False, 
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Extract
            input_ids = inputs.input_ids[0]
            attention_mask = inputs.attention_mask[0]
            pixel_values = inputs.pixel_values
            image_grid_thw = inputs.image_grid_thw
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_pixel_values.append(pixel_values)
            all_image_grid_thw.append(image_grid_thw)

        return {
            "input_ids_list": all_input_ids,
            "attention_mask_list": all_attention_mask,
            "pixel_values_list": all_pixel_values,
            "image_grid_thw_list": all_image_grid_thw,
            "ranks": torch.tensor(ranks, dtype=torch.float), # [K]
            "id": idx
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        # Batch is a list of N samples. Each sample contains lists of K items.
        # We need to flatten everything to [N*K, ...] and also pad.
        
        ks = [len(item["ranks"]) for item in batch]
        max_k = max(ks)

        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        ranks = []
        candidate_masks = []
        
        # Helper lists to manage Flattening
        
        # Flatten all valid inputs into a single list
        for item in batch:
            input_ids.extend(item["input_ids_list"])
            attention_mask.extend(item["attention_mask_list"])
            pixel_values.extend(item["pixel_values_list"])
            image_grid_thw.extend(item["image_grid_thw_list"])
            
            # Prepare ranks and mask
            curr_ranks = item["ranks"] # [K]
            curr_k = len(curr_ranks)
            
            if curr_k < max_k:
                pad_len = max_k - curr_k
                
                # Pad ranks with inf (so they appear last in sort)
                ranks_padded = torch.cat([
                    curr_ranks, 
                    torch.full((pad_len,), float('inf'), dtype=curr_ranks.dtype)
                ], dim=0)
                ranks.append(ranks_padded)
                
                # Mask: 1 for valid, 0 for padded
                mask = torch.cat([torch.ones(curr_k, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)], dim=0)
                candidate_masks.append(mask)
            else:
                ranks.append(curr_ranks)
                candidate_masks.append(torch.ones(curr_k, dtype=torch.bool))

        # Pad sequences (Tokens)
        padding_side = "left"
        input_ids_padded = zero_pad_sequences(input_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        attention_mask_padded = zero_pad_sequences(attention_mask, side=padding_side)

        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0)
            image_grid_thw = torch.cat(image_grid_thw, dim=0)
        else:
            pixel_values = None
            image_grid_thw = None

        ranks = torch.stack(ranks) # [B, max_K]
        candidate_masks = torch.stack(candidate_masks) # [B, max_K]
        
        return {
            "input_ids": input_ids_padded, # [Total_Valid, L]
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ranks": ranks, # [B, max_K]
            "candidate_masks": candidate_masks # [B, max_K]
        }
