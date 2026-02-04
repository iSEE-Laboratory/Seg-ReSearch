import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'sam2'))

import tqdm
_original_tqdm = tqdm.tqdm
def _disabled_tqdm(*args, **kwargs):
    kwargs['disable'] = True
    return _original_tqdm(*args, **kwargs)
tqdm.tqdm = _disabled_tqdm

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm
import imageio
import imageio.v3 as iio
from sam2.build_sam import build_sam2_video_predictor
import re
from collections import defaultdict
import json
import cv2
import random

DEBUG = False
USE_POINT = True
USE_BOX = True

def vis_add_mask(img_pil, mask, color=(0, 255, 0), alpha=0.5):
    img = np.array(img_pil)
    mask = mask > 0.5
    mask_rgb = np.zeros_like(img)
    mask_rgb[mask] = color
    vis_img = img.copy()
    vis_img[mask] = img[mask] * (1 - alpha) + mask_rgb[mask] * alpha
    return Image.fromarray(vis_img.astype(np.uint8))

def process_video(args, video_data, predictor, device):
    video_path = video_data[0]['video_path']

    frame_names = sorted(os.listdir(video_path))
    
    video_len = len(frame_names)
    state = predictor.init_state(video_path=video_path)
    
    first_frame_path = os.path.join(video_path, frame_names[0])
    first_frame = Image.open(first_frame_path)
    img_width, img_height = first_frame.size
    
    # for Qwen3VL 
    ratio_w, ratio_h = img_width/1000, img_height/1000
    box_scale = np.array([ratio_w, ratio_h, ratio_w, ratio_h], dtype=np.float32)
    point_scale = np.array([ratio_w, ratio_h], dtype=np.float32)

    for sample in video_data:
        predictor.reset_state(state)
        
        id = sample['id']
        output = sample['output']
        box = np.zeros(4, dtype=np.float32)
        points = np.zeros((1, 2), dtype=np.float32)
        fid = -1

        try:
            keyframe_match = re.search(r'<keyframe>\s*(\d+)\s*</keyframe>', output, re.S)
            fid = int(keyframe_match.group(1).strip())
            answer_match = re.search(r'<answer>(.*?)</answer>', output)
            pred = answer_match.group(1).strip()
            pred = json.loads(pred)
            box = np.array(pred["bbox_2d"], dtype=np.float32) * box_scale
            points = np.array([pred["point_2d"]], dtype=np.float32) * point_scale
        except Exception as e:
            if DEBUG:
                print(f"Incorrect format: {e} in ID: {id}")

        prompted = False
        if 0 <= fid < video_len:
            if box.any() and USE_BOX:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=fid,
                    obj_id=1,
                    box=box,
                )
                prompted = True

            if points.any() and USE_POINT:
                labels = np.array([1], np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=fid,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )
                prompted = True

        video_masks = np.zeros((video_len, img_height, img_width), dtype=bool)
        if prompted:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                video_masks[out_frame_idx] = (out_mask_logits > 0).squeeze(1).cpu().numpy()
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, reverse=True):
                video_masks[out_frame_idx] = (out_mask_logits > 0).squeeze(1).cpu().numpy()

        save_path = os.path.join(args.save_path, id)
        os.makedirs(save_path, exist_ok=True)

        vis_writer = None
        if args.vis:
            mask_color = np.array([0, 255, 0], dtype=np.uint8)
            vis_video_path = os.path.join(args.vis_path, f"{id}.mp4")
            os.makedirs(os.path.dirname(vis_video_path), exist_ok=True)
            vis_writer = imageio.get_writer(
                vis_video_path, 
                fps=15, 
                codec='libx264',
                macro_block_size=None,
                format='FFMPEG'
            )

        for t in range(video_len):
            mask = video_masks[t]
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                os.path.join(save_path, f"{frame_names[t].split('.')[0]}.png")
            )
            if vis_writer is not None:
                img_path = os.path.join(video_path, frame_names[t])
                img_arr = iio.imread(img_path) 
                if mask.any():
                    img_arr[mask] = (img_arr[mask].astype(np.uint16) + mask_color) // 2
                if t == fid:
                    x1, y1, x2, y2 = box.astype(np.int32)
                    cv2.rectangle(img_arr, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    px, py = points[0].astype(np.int32)
                    cv2.circle(img_arr, (px, py), 8, (0, 0, 255), -1)
                    cv2.circle(img_arr, (px, py), 8, (255, 255, 255), 2)
                vis_writer.append_data(img_arr)

        if vis_writer is not None:
            vis_writer.close()


def process_worker(args, data_list, device_id):
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    with _original_tqdm(total=len(data_list), desc=f"GPU {device_id}", position=device_id, leave=True) as pbar:
        for data in data_list:
            process_video(args, data, predictor, device)
            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='?', default="reuslts/ReasonVOS_test_6_448_448_Qwen3-VL-8B-Instruct_eval/0.jsonl", help='Path to the preprocessed data')
    parser.add_argument('--vis', action='store_true', help='Visualize masks on original images')
    args = parser.parse_args()

    with open(args.file, 'r', encoding='utf-8') as f:
        data = [item for line in f if (item := json.loads(line)).get('data_source') == 'ReasonVOS']

    mp.set_start_method('spawn')

    args.save_path = os.path.join(os.path.dirname(args.file), "seg")
    args.vis_path = os.path.join(os.path.dirname(args.file), "vis")
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.vis_path, exist_ok=True)

    video_data = defaultdict(list)
    for item in data:
        video_data[item['video_path']].append(item)
    
    video_data = list(video_data.values())
    random.shuffle(video_data)

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    videos_per_gpu = (len(video_data) + num_gpus - 1) // num_gpus

    processes = []
    for i in range(num_gpus):
        start_idx = i * videos_per_gpu
        end_idx = min((i + 1) * videos_per_gpu, len(video_data))
        gpu_video_list = video_data[start_idx:end_idx]

        if gpu_video_list:
            p = mp.Process(target=process_worker, args=(args, gpu_video_list, i))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()