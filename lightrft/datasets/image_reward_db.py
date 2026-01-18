import os
import copy
import random
import pandas as pd
from typing import List, Dict, Any, Tuple
from itertools import combinations
from loguru import logger
from tqdm import tqdm

from .utils import BaseDataHandler


class ImageRewardDBPairwiseHandler(BaseDataHandler):
    """
    Data Handler for ImageRewardDB dataset for pairwise ranking.

    Paper: https://arxiv.org/abs/2304.05977
    Dataset Repo: https://huggingface.co/datasets/zai-org/ImageRewardDB
    """
    task_type = "text-to-image"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Load ImageRewardDB shards and build preference pairs.

        This method scans the given dataset root for ImageRewardDB JSON shards and
        aggregates image entries by ``prompt_id``. For each prompt group, it
        constructs all unordered pairs of images and determines the preferred
        image based on the ``rank`` field (smaller is better, i.e., ``1`` is best).

        :param path: Path to the dataset root directory of ImageRewardDB.
        :type path: str

        :return: List of preference pair dictionaries.
        :rtype: List[Dict[str, Any]]

        **Example:**

        .. code-block:: python

            data = handler.load_data("path/to/ImageRewardDB")
        """

        if not path.endswith(".parquet"):
            logger.error(f"ImageRewardDBHandler expects a .parquet file, got: {path}")
            raise ValueError(f"Invalid file format: {path}")

        if not os.path.exists(path):
            logger.error(f"Metadata file not found: {path}")
            raise FileNotFoundError(f"Metadata file not found: {path}")

        dataset_root = os.path.dirname(path)
        logger.info(f"Loading data from: {path}. Root: {dataset_root}")

        df = pd.read_parquet(path)
        grouped_data = df.groupby('prompt_id')

        print(f"Aggregated {len(grouped_data)} unique prompt groups. Generating pairs...")

        # Construct pairs
        preference_pairs = []
        skipped_files_count = 0
        for pid, group in tqdm(grouped_data, desc="Generating pairs"):
            items = group.to_dict('records')
            # Filter out items with missing or empty image files
            valid_items = []
            # dataset_root is defined above
            for item in items:
                full_img_path = os.path.join(dataset_root, item['image_path'])
                if os.path.exists(full_img_path) and os.path.getsize(full_img_path) > 0:
                    valid_items.append(item)
                else:
                    # print(f"[Warning] Skipping missing or empty file: {full_img_path}")
                    skipped_files_count += 1

            items = valid_items
            # If a prompt group has fewer than 2 images, skip
            if len(items) < 2:
                continue

            # Use combinations to generate all unordered pairs (e.g., AB and BA appear once)
            for item_a, item_b in combinations(items, 2):

                rank_a = item_a['rank']
                rank_b = item_b['rank']

                # Defensive: ensure rank fields are present
                if rank_a is None or rank_b is None:
                    continue

                # Decision rule: smaller rank value is better (1 is best)
                if rank_a == rank_b:
                    continue

                if rank_a < rank_b:
                    chosen = item_a
                    reject = item_b
                else:
                    chosen = item_b
                    reject = item_a

                # Build pair entry
                pair_entry = {
                    "prompt_id": pid,
                    "prompt": chosen['prompt'],
                    "classification": chosen.get('classification', 'Unknown'),
                    "data_root": dataset_root,

                    # Chosen Image Info
                    "chosen_img": chosen['image_path'],
                    "rank_chosen": chosen['rank'],
                    "overall_rating_chosen": chosen.get('overall_rating'),

                    # Rejected Image Info
                    "rejected_img": reject['image_path'],
                    "rank_rejected": reject['rank'],
                    "overall_rating_rejected": reject.get('overall_rating'),
                }
                preference_pairs.append(pair_entry)

        logger.info(f"Loaded {len(preference_pairs)} samples from ImageRewardDB.")
        return preference_pairs

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for chosen and rejected images.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing local paths for 'preferred_image' and 'rejected_image'
        :rtype: Dict[str, Dict[str, str]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
        data_root = item['data_root']

        # Build full local paths
        full_path1 = os.path.join(data_root, item['chosen_img'])
        full_path2 = os.path.join(data_root, item['rejected_img'])

        # Make sure files exist
        if not os.path.exists(full_path1) or not os.path.exists(full_path2):
            return None
        else:
            return {
                'preferred_image': {
                    'image_local_path': full_path1
                },
                'rejected_image': {
                    'image_local_path': full_path2
                }
            }

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse a single ImageRewardDB item into message pairs for ranking.

        :param item: Raw data item from ImageRewardDB dataset.
        :type item: Dict[str, Any]
        :param media_content: Loaded media content with 'preferred_image' and 'rejected_image' keys.
        :type media_content: Dict[str, Any]
        :param config: Configuration dict with task instructions and max_pixels
        :type config: Dict[str, Any]

        :return: A tuple of (messages0, messages1, metadata)
        :rtype: Tuple[List[Dict], List[Dict], Dict]

        **Example:**

        .. code-block:: python

            msg0, msg1, other = handler.parse_item(item, media_content, config)
        """
        # Get loaded visual content
        preferred_image = media_content['preferred_image']
        rejected_image = media_content['rejected_image']

        if not all([preferred_image, rejected_image]):
            raise ValueError("Missing visual content for 'preferred_image' or 'rejected_image'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]
        if not prompt_text:
            raise ValueError(f"Missing generation prompt in item: {item}")

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

        # Random pick from "A" or "B" to avoid positional bias
        preference = random.choice(["A", "B"])
        if preference == "A":  # "A" means image0 is preferred
            image0, image1 = preferred_image, rejected_image
        else:
            image0, image1 = rejected_image, preferred_image

        # Build messages
        messages0 = [
            {
                "role": "system",
                "content": copy.deepcopy(task_instruction)
            },
            {
                "role": "user",
                "content": [{
                    "type": "image",
                    "image": image0,
                    "max_pixels": max_pixels
                }
                            ]
            }
        ]

        messages1 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [{
                "type": "image",
                "image": image1,
                "max_pixels": max_pixels
            }]
        }]

        other = {
            "preference": preference,  # used for reward head labeling
            "task_type": self.task_type,
            "source": item["source"],
            "prompt_id": item["prompt_id"],
            "prompt": prompt_text,
            "chosen_img": item["chosen_img"],
            "rejected_img": item["rejected_img"],
            "rank_chosen": item["rank_chosen"],
            "rank_rejected": item["rank_rejected"],
            "overall_rating_chosen": item["overall_rating_chosen"],
            "overall_rating_rejected": item["overall_rating_rejected"],
            "classification": item["classification"],
        }
        return messages0, messages1, other


class ImageRewardDBListwiseHandler(BaseDataHandler):
    """
    Listwise Data Handler for ImageRewardDB dataset.
    """
    task_type = "text-to-image-listwise-ranking"

    def load_data(self, path: str, list_size: int = 4) -> List[Dict[str, Any]]:
        """
        Load ImageRewardDB from a specific parquet metadata file and build listwise ranking samples.

        :param path: Full path to a .parquet metadata file.
        :type path: str
        """
        if not path.endswith(".parquet"):
            logger.error(f"ImageRewardDBListwiseHandler expects a .parquet file, got: {path}")
            raise ValueError(f"Invalid file format: {path}")

        if not os.path.exists(path):
            logger.error(f"Metadata file not found: {path}")
            raise FileNotFoundError(f"Metadata file not found: {path}")

        dataset_root = os.path.dirname(path)
        logger.info(f"Loading listwise data from: {path}. Root: {dataset_root}")

        df = pd.read_parquet(path)

        # Aggregate entries by prompt_id
        grouped = df.groupby('prompt_id')
        
        listwise_samples = []
        for pid, group in tqdm(grouped, desc="Generating listwise samples"):
            valid_items = []
            for _, row in group.iterrows():
                # Rel path in parquet is usually like 'images/train_01/1.png'
                rel_path = row['image_path']
                full_img_path = os.path.join(dataset_root, rel_path)
                
                if os.path.exists(full_img_path) and os.path.getsize(full_img_path) > 0:
                    valid_items.append({
                        'image_path': rel_path,
                        'rank': row['rank'],
                        'prompt': row['prompt'],
                        'classification': row.get('classification', 'Unknown')
                    })
            
            # Require at least 2 items to form a ranking list
            if len(valid_items) < 2:
                continue

            candidates_paths = [item['image_path'] for item in valid_items]
            candidates_ranks = [item['rank'] for item in valid_items]
            
            # Construct entry
            entry = {
                "source": "imagerewarddb",
                "prompt_id": pid,
                "prompt": valid_items[0]['prompt'],
                "candidates": candidates_paths,
                "ranks": candidates_ranks,
                "data_root": dataset_root,
                "classification": valid_items[0].get('classification', 'Unknown')
            }
            listwise_samples.append(entry)

        logger.info(f"Loaded {len(listwise_samples)} listwise samples from ImageRewardDB.")
        return listwise_samples

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        data_root = item['data_root']
        candidates = item['candidates']
        
        media_info = {}
        for i, rel_path in enumerate(candidates):
            full_path = os.path.join(data_root, rel_path)
            media_info[f"image_{i}"] = {
                "image_local_path": full_path
            }
        return media_info
    
    def parse_item(self, item, media_content, config):
        # Dummy method to satisfy interface
        pass