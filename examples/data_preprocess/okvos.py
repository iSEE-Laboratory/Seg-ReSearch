# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import re
import random
import json
from pathlib import Path
from datasets import load_dataset
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2

from prompt import search_prompt, non_search_prompt


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def scale_box_coordinates(bbox, w, h, new_w, new_h, model_type):
    x1, y1, x2, y2 = bbox
    
    if model_type.lower() == 'qwen3vl':
        # Qwen3VL: Normalize to 0-1000 range
        x1_scaled = (x1 / w) * 1000
        y1_scaled = (y1 / h) * 1000
        x2_scaled = (x2 / w) * 1000
        y2_scaled = (y2 / h) * 1000
        return [round(x1_scaled), round(y1_scaled), round(x2_scaled), round(y2_scaled)]
    
    elif model_type.lower() == 'qwen2.5vl':
        # Qwen2.5VL: Convert to absolute coordinates in resized image
        x1_scaled = (x1 / w) * new_w
        y1_scaled = (y1 / h) * new_h
        x2_scaled = (x2 / w) * new_w
        y2_scaled = (y2 / h) * new_h
        return [round(x1_scaled), round(y1_scaled), round(x2_scaled), round(y2_scaled)]

def scale_point_coordinates(point, w, h, new_w, new_h, model_type):
    x, y = point
    
    if model_type.lower() == 'qwen3vl':
        # Qwen3VL: Normalize to 0-1000 range
        x_scaled = (x / w) * 1000
        y_scaled = (y / h) * 1000
        return [round(x_scaled), round(y_scaled)]
    
    elif model_type.lower() == 'qwen2.5vl':
        # Qwen2.5VL: Convert to absolute coordinates in resized image
        x_scaled = (x / w) * new_w
        y_scaled = (y / h) * new_h
        return [round(x_scaled), round(y_scaled)]

def bounding_box(m):
    coords = np.where(m > 0)
    y_coords, x_coords = coords[0], coords[1]
    return np.array([x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]).tolist()

def get_point(m):
    y_indices, x_indices = np.where(m > 0)
    dist_transform = ndimage.distance_transform_edt(m)
    y1, x1 = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
    if m[y1, x1] == 0:
        distances = (x_indices - x1) ** 2 + (y_indices - y1) ** 2
        nearest_idx = np.argmin(distances)
        x1, y1 = x_indices[nearest_idx], y_indices[nearest_idx]
    return [int(x1), int(y1)]


def load_annotations_from_masks(anno_dir, image_names):
    """
    Load annotations from mask files.

    Args:
        anno_dir: Directory containing mask files
        image_names: List of frame names

    Returns:
        answers_dict: Dictionary mapping frame_id to annotation data
        h, w: Image height and width
    """
    answers = {}
    h = w = 0

    for frame_name in image_names:
        mask_path = os.path.join(anno_dir, f"{frame_name}.png")
        if not os.path.exists(mask_path):
            continue

        mask = np.array(Image.open(mask_path).convert('P'))
        h, w = mask.shape
        mask = (mask == 255).astype(np.float32)

        if mask.any():
            fid = int(frame_name)
            answers[fid] = {
                "bbox_2d": bounding_box(mask),
                "point_2d": get_point(mask),
                "area": float(mask.sum())
            }

            mask_uint8 = (mask * 255).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            answers[fid]['max_conn_area'] = float(np.max(areas))

    return answers, h, w

def make_map_fn(split, num_frames, max_size, min_size, data_dir, model_type):
    def process_fn(example, idx):
        # print(example)
        prompt = example.pop('question')
        image_names = sorted(example.pop('images'))
        ability = example.pop('ability')
        video_dir = example.pop('video')
        id = example.get('id', str(idx))
        oid = example.pop('oid')

        system_prompt = non_search_prompt if ability == "in-video" else search_prompt

        video_path = os.path.join(data_dir, "JPEGImages", video_dir)
        image_paths = [os.path.join(video_path, image+'.jpg') for image in image_names]
        assert os.path.exists(image_paths[0]), f"Image file {image_paths[0]} does not exist"

        anno_dir = os.path.join(data_dir, "Annotations", video_dir, str(oid))
        answers_dict, h, w = load_annotations_from_masks(anno_dir, image_names)

        total_frames = len(image_names)
        if num_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames-1, num_frames, dtype=int).tolist()
            if split == 'train' and num_frames > 1:
                # replace the closest index with the anchor index
                anchor_fid = random.choice(list(answers_dict.keys()))
                anchor_idx = next(i for i, name in enumerate(image_names) if int(name.split('.')[0]) == anchor_fid)
                indices[min(range(len(indices)), key=lambda i: abs(indices[i] - anchor_idx))] = anchor_idx 
            
        indices = sorted(indices)

        new_w = max_size if w > h else min_size
        new_h = max_size if h > w else min_size

        images = []
        solution = {}
        mm_content = ''
        for i in indices:
            fid = int(image_names[i])
            images.append(image_paths[i])
            mm_content += f"frame {fid} <image>\n"
            if fid in answers_dict:
                solution[fid] = answers_dict[fid]
        answers_dict = solution

        if answers_dict:
            max_area = max(answers_dict.values(), key=lambda x: x.get('area', 0))['area']
            max_conn_area = max(answers_dict.values(), key=lambda x: x.get('max_conn_area', 0))['max_conn_area']
            for k, v in answers_dict.items():
                v['bbox_2d'] = scale_box_coordinates(v['bbox_2d'], w, h, new_w, new_h, model_type)
                v['point_2d'] = scale_point_coordinates(v['point_2d'], w, h, new_w, new_h, model_type)
                v['area'] = v['area'] / max_area
                v['max_conn_area'] = v['max_conn_area'] / max_conn_area

        answers_str = json.dumps(answers_dict)

        mm_content += f'\n{prompt}'
        data = {
            "data_source": f"Search_RVOS_{ability}",
            'video_path': video_path,
            "id": id,
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": mm_content,
                }
            ],
            "images": [{"image": image, "resized_height": new_h, "resized_width": new_w} for image in images],
            "ability": ability,
            "reward_model": {
                "style": "rule",
                "ground_truth": answers_str
            },
            "extra_info": {
                'split': split,
                'height': new_h,
                'width': new_w,
                'image_paths': images
            }
        }

        steps = example.get('steps', [])
        if steps:
            data['reward_model']['steps'] = steps
        
        if split == 'test':
            data['mask_path'] = anno_dir
        return data
    return process_fn


def main(local_parquet_filepath, local_save_dir, split="train", num_frames=6, max_size=640, min_size=352, model_type="Qwen3VL"):
    print(local_parquet_filepath)
    df_raw = load_dataset('parquet', data_files={split: local_parquet_filepath}, split=split)
    logger.info(f"Loaded {len(df_raw)} samples from {local_parquet_filepath}")

    data_dir = Path(local_parquet_filepath).parent
    data_dir = data_dir / split
    dataset = df_raw.map(
        function=make_map_fn(split, num_frames, max_size, min_size, data_dir, model_type),
        with_indices=True,
        num_proc=32
    )

    output_filename = f'{split}_{num_frames}_{max_size}_{min_size}.parquet'
    output_path = os.path.join(local_save_dir, output_filename)

    if split == 'test':
        meta = list(zip(dataset['id'], dataset['mask_path'], dataset['ability']))
        map_output_path = os.path.join(local_save_dir, f'meta_mapping.json')
        json.dump(meta, open(map_output_path, 'w'))
        dataset = dataset.remove_columns(["mask_path"])
        print(f"Saved meta mapping for {len(meta)} items to {map_output_path}")

    os.makedirs(local_save_dir, exist_ok=True)
    dataset.to_parquet(output_path)
    print(f"Saved {len(dataset)} {split} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RVOS dataset with configurable parameters')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Path to input parquet file')
    parser.add_argument('--num_frames', type=int, default=6,
                        help='Number of frames to sample')
    parser.add_argument('--max_size', type=int, default=448,
                        help='Maximum size for image resizing')
    parser.add_argument('--min_size', type=int, default=448,
                        help='Minimum size for image resizing')
    parser.add_argument('--model_type', type=str, default='Qwen3VL',
                        choices=['Qwen3VL', 'Qwen2.5VL'])

    args = parser.parse_args()

    if args.model_type == 'Qwen3VL':
        args.patch_factor = 32
    elif args.model_type == 'Qwen2.5VL':
        args.patch_factor = 28

    # Adjust sizes to be divisible by patch_factor (round up)
    original_max, original_min = args.max_size, args.min_size
    args.max_size = math.ceil(args.max_size / args.patch_factor) * args.patch_factor
    args.min_size = math.ceil(args.min_size / args.patch_factor) * args.patch_factor
    
    if original_max != args.max_size:
        logger.info(f"Adjusted max_size from {original_max} to {args.max_size}")
    if original_min != args.min_size:
        logger.info(f"Adjusted min_size from {original_min} to {args.min_size}")

    main(
        local_parquet_filepath=f'data/OK_VOS/{args.split}.parquet',
        local_save_dir="data/OK_VOS",
        split=args.split,
        num_frames=args.num_frames,
        max_size=args.max_size,
        min_size=args.min_size,
        model_type=args.model_type
    )
