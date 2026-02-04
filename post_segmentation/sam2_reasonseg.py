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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import re
import json
import cv2

USE_POINT = False
USE_BOX = True

def visualize(image, mask, box=None, points=None, color=(0, 255, 0), alpha=0.5):
    mask = mask > 0.5 
    img_arr = np.array(image)
    vis_img = img_arr.copy()
    for c in range(3):
        vis_img[:, :, c] = np.where(
            mask, 
            vis_img[:, :, c] * (1 - alpha) + color[c] * alpha, 
            vis_img[:, :, c]
        )
    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Red box, thickness 2

    if points is not None:
        for pt in points:
            x, y = pt.astype(int)
            cv2.circle(vis_img, (x, y), 5, (0, 0, 255), -1) # Blue filled circle, radius 5

    return Image.fromarray(vis_img.astype(np.uint8))

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence

def process_group(args, data, predictor, device):
    image_path = data['image_path']
    image_source = Image.open(image_path).convert("RGB")

    predictor.set_image(image_source)
    img_width, img_height = image_source.size
    
    # for Qwen3VL 
    ratio_w, ratio_h = img_width/1000, img_height/1000
    box_scale = np.array([ratio_w, ratio_h, ratio_w, ratio_h], dtype=np.float32)
    point_scale = np.array([ratio_w, ratio_h], dtype=np.float32)

    id = data['id']
    output = data['output']

    meta_file = image_path.replace(".jpg", ".json")
    gt_mask = get_mask_from_json(meta_file, np.array(image_source))[0]

    box = np.zeros(4, dtype=np.float32)
    points = np.zeros((1, 2), dtype=np.float32)

    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', output)
        pred = answer_match.group(1).strip()
        pred = json.loads(pred)
        if USE_BOX:
            box = np.array(pred["bbox_2d"], dtype=np.float32) * box_scale
        if USE_POINT:
            points = np.array([pred["point_2d"]], dtype=np.float32) * point_scale
    except Exception as e:
        print(f"Incorrect format: {e} in ID: {id}")
        
    if box.any() or points.any():
        input_box, input_points, input_point_labels = None, None, None
        if USE_BOX and box.any():
            input_box = box
        if USE_POINT and points.any():
            input_points = points
            input_point_labels = np.ones(len(points), dtype=np.uint8)

        mask, score, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_point_labels,
            box=input_box,
            multimask_output=False,
        )
        mask = mask.sum(0).clip(0, 1)
    else:
        mask = np.zeros_like(gt_mask)

    gt_mask = torch.from_numpy(gt_mask).to(device).int()
    pred_mask = torch.from_numpy(mask).to(device).int()
    intersection, union, _ = intersectionAndUnionGPU(
        pred_mask.contiguous().clone(), gt_mask.contiguous(), 2, ignore_index=255
    )
    inter_val = intersection[1].item()
    union_val = union[1].item()
    if union_val == 0:
        iou = 1.0
    else:
        iou = inter_val / (union_val + 1e-10)

    if args.vis:
        save_name = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        vis_path = os.path.join(args.vis_path, save_name)
        image_source = visualize(image_source, mask, box=box, points=points, color=(0, 255, 0))
        image_source.save(vis_path)

    return iou, inter_val, union_val

def process_worker(args, data_list, device_id, result_dict):
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device=device))
    results = []

    with _original_tqdm(total=len(data_list), desc=f"GPU {device_id}", position=device_id, leave=True) as pbar:
        for data in data_list:
            res = process_group(args, data, predictor, device)
            if res is not None:
                results.append(res)
            pbar.update(1)

    result_dict[device_id] = results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='?', default="reuslts/8B-ReasonSeg__data1_tianming_qwen_qwen3-vl-8b-instruct_eval/0.jsonl", help='Path to the preprocessed data')
    parser.add_argument('--vis', action='store_true', help='Visualize masks on original images')
    args = parser.parse_args()

    with open(args.file, 'r', encoding='utf-8') as f:
        data = [item for line in f if (item := json.loads(line)).get('data_source') == 'reasonseg']

    mp.set_start_method('spawn')

    args.vis_path = os.path.join(os.path.dirname(args.file), "vis")
    os.makedirs(args.vis_path, exist_ok=True)


    num_gpus = torch.cuda.device_count()
    groups_per_gpu = (len(data) + num_gpus - 1) // num_gpus

    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for i in range(num_gpus):
        start_idx = i * groups_per_gpu
        end_idx = min((i + 1) * groups_per_gpu, len(data))
        gpu_data_list = data[start_idx:end_idx]

        if gpu_data_list:
            p = mp.Process(target=process_worker, args=(args, gpu_data_list, i, result_dict))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    total_inter = 0.0
    total_union = 0.0 
    total_iou_sum = 0.0 
    total_count = 0

    for gpu_id, results_list in result_dict.items():
        for iou, inter, union in results_list:
            total_inter += inter
            total_union += union
            total_iou_sum += iou
            total_count += 1
    
    print("\n"*num_gpus + "="*40)
    print(f" gIoU:  {total_iou_sum / total_count:.4f}")
    print(f" cIoU: {total_inter / (total_union + 1e-10):.4f}")

if __name__ == "__main__":
    main()