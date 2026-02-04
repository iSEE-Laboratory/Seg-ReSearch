###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

import os
import time
import argparse
import cv2
import json
import glob
import numpy as np
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp

NUM_WOEKERS = 64


def eval_queue(q, rank, out_dict, pred_path):
    while not q.empty():
        id, mask_path_vid, ability = q.get()

        pred_path_vid_exp = os.path.join(pred_path, id)

        if not os.path.exists(pred_path_vid_exp):
            print(f'{pred_path_vid_exp} not found, not take into metric computation')
            continue

        gt_mask_list = [x for x in sorted(os.listdir(mask_path_vid)) if str(x).endswith('png')]
        gt_0_path = os.path.join(mask_path_vid, gt_mask_list[0])
        gt_0 = cv2.imread(gt_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt_0.shape

        vid_len = len(gt_mask_list)
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        for frame_idx, frame_name in enumerate(gt_mask_list):
            gt_masks[frame_idx] = cv2.imread(os.path.join(mask_path_vid, frame_name), cv2.IMREAD_GRAYSCALE)
            pred_masks[frame_idx] = cv2.imread(os.path.join(pred_path_vid_exp, frame_name), cv2.IMREAD_GRAYSCALE)

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[id] = [j, f, ability]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default="")
    parser.add_argument("--meta_path", type=str, default="data/RVOS_Search_Test/meta_mapping.json")
    args = parser.parse_args()
    args.save_name = os.path.join(os.path.dirname(args.pred_dir), "result.json")

    queue = mp.Queue()
    meta = json.load(open(args.meta_path, 'r'))
    output_dict = mp.Manager().dict()

    for (id, mask_path, ablity) in meta:
        queue.put([id, mask_path, ablity])

    print("Q-Size:", queue.qsize())

    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.pred_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(args.save_name, 'w') as f:
        json.dump(dict(output_dict), f)

    for ab in ['one-hop', 'multi-hop', 'relational']:
        print("==== {} ====".format(ab))
        j = [output_dict[x][0] for x in output_dict if output_dict[x][2] == ab]
        f = [output_dict[x][1] for x in output_dict if output_dict[x][2] == ab]
        print(f'J: {np.mean(j)}')
        print(f'F: {np.mean(f)}')
        print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')

    print("==== Overall ====")
    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    print(f'J: {np.mean(j)}')
    print(f'F: {np.mean(f)}')
    print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % total_time)