import os
import json
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler


class SynthScarsHandler(BaseDataHandler):
    """
    Data handler for SynthScars dataset.

    Home: https://opendatalab.github.io/LEGION/
    Dataset repo: https://huggingface.co/datasets/khr0516/SynthScars
    """
    task_type = "artifact-detection"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from json file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        data_root = os.path.dirname(path)
        for item in raw_data:
            item['data_root'] = data_root

        logger.info(f"Loaded {len(raw_data)} samples from {path}")
        return raw_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the image.
        """
        data_root = item.get('data_root')
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve image paths.")

        # Images are in ../images/ relative to the annotations folder
        image_name = item['image']
        image_path = os.path.join(data_root, "..", "images", image_name)

        return {
            'image': {
                'image_local_path': image_path
            }
        }

    def parse_item(
        self,
        item: Dict[str, Any],
        media_content: Dict[str, Any],
        config: Dict[str, Any] | None,
    ) -> Tuple[List[Dict], Dict]:

        image = media_content.get('image')
        if not image:
            raise ValueError("Missing visual content for 'image'.")

        # Get image size for normalization
        width, height = image.size

        # Get assistant response
        response = item["conversations"][0]['value']
        if not response:
            raise ValueError(f"No assistant response found in item {item.get('id')}")

        # Replace bbox placeholders {bbox_i}
        # bbox format in json: {"0": {"bbox": [ymin, xmin, ymax, xmax], ...}}
        bbox_data = item["bbox"]
        for i, info in bbox_data.items():
            coords = info["bbox"]
            # Expects normalized coordinates in [0, 1000] (Qwen2.5-VL)
            # Formula: normalized_coord = actual_coord * 1000 / image_dimension
            ymin, xmin, ymax, xmax = coords
            
            ny1 = max(0, min(1000, int(ymin * 1000 / height)))
            nx1 = max(0, min(1000, int(xmin * 1000 / width)))
            ny2 = max(0, min(1000, int(ymax * 1000 / height)))
            nx2 = max(0, min(1000, int(xmax * 1000 / width)))

            bbox_str = f"[{ny1}, {nx1}, {ny2}, {nx2}]"
            response = response.replace(f"{{bbox_{i}}}", bbox_str)

        # Get system prompt
        system_prompt = config["task_instruction"]

        # Get max pixels
        max_pixels = config.get("max_pixels", 720 * 480)

        # Build messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "max_pixels": max_pixels
                    },
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response
                    }
                ]
            }
        ]

        other = {
            "id": item.get('id'),
            "source": "synthscars",
            "response": response,            
            "task_type": self.task_type,
            "box_data": bbox_data,
            "image_path": item["image"]
        }
        return messages, other
