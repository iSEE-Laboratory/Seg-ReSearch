"""
Seg-zero style Reward Manager
"""
import torch
import random
import regex as re
import json
import time
import os
import numpy as np
from verl import DataProto
from verl.workers.reward_manager.registry import register
from collections import defaultdict


def accuracy_reward(response_str, ground_truth, for_video=True):
    def compute_iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        else:
            inter = 0
        area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union = area1 + area2 - inter
        return float(inter) / union

    ground_truth = json.loads(ground_truth)
    iou = 0.0
    try:
        if for_video:
            keyframe_matches = re.findall(r'<keyframe>\s*(\d+)\s*</keyframe>', response_str, re.S)
            pred_frame = keyframe_matches[-1].strip()
            ground_truth = ground_truth[pred_frame]
        
        answer_match = re.search(r'<answer>(.*?)</answer>', response_str)
        pred = answer_match.group(1).strip()
        pred = json.loads(pred)
        iou = compute_iou(pred["bbox_2d"], ground_truth["bbox_2d"])
    except Exception:
        pass

    return iou

@register("search_rvos_val")
class SearchRvosValManager:
    """
    Reward Manager for Search-R1 style QA tasks with Exact Match scoring.
    """
    name = "search_rvos_val"
    
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards for Search-R1 style responses."""
        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch['rm_scores'], "reward_extra_info": reward_extra_info}
            else:
                return data.batch['rm_scores']

        scores = [{} for _ in range(len(data))]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Get ground truth
            if 'reward_model' in data_item.non_tensor_batch:
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            else:
                # Fallback to direct ground truth or golden_answers
                ground_truth = data_item.non_tensor_batch.get('ground_truth', 
                              data_item.non_tensor_batch.get('golden_answers', []))

            # Print examples for debugging
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # Compute score
            for_video = data_source != 'reasonseg'
                
            score = accuracy_reward(
                response_str=response_str, 
                ground_truth=ground_truth, 
                for_video=for_video,
            )

            # update this score to the scores
            scores[i] = {"acc": score, 'num_search': float(response_str.count('<search>'))}

            reward_tensor[i, valid_response_length - 1] = score
            
            # Store information for validation metrics grouping
            ability = data_item.non_tensor_batch.get('ability', '')
            reward_extra_info['ability'].append(ability)
            
            id = data_item.non_tensor_batch.get('id', '')
            reward_extra_info['id'].append(id)

            image_path = data_item.non_tensor_batch.get('image_path', '')
            if image_path:
                reward_extra_info['image_path'].append(image_path)

            video_path = data_item.non_tensor_batch.get('video_path', '')
            if video_path:
                reward_extra_info['video_path'].append(video_path)

            # Save the records
            tool_interact_info = data_item.non_tensor_batch.get('tool_interact_info', None)
            if tool_interact_info is not None:
                # crop the image
                for info in tool_interact_info:
                    if "image" in info:
                        if isinstance(info['image'], list):
                            info['image'] = [x[:50] for x in info['image']]  # crop the image to first 50 characters
                        elif isinstance(info['image'], str):
                            info['image'] = info['image'][:50]  # for debug

        for i, score in enumerate(scores):
            if isinstance(score, dict):
                # convert the length to a Python int
                length_i = data[i].batch['attention_mask'][data[i].batch['prompts'].shape[-1]:].sum().item()
                # subtract 1 because you want the last *valid* token
                reward_tensor[i, length_i - 1] = score['acc']

                for k, v in score.items():
                    reward_extra_info[k].append(v)
            else:
                length_i = data[i].batch['attention_mask'][data[i].batch['prompts'].shape[-1]:].sum().item()
                reward_tensor[i, length_i - 1] = score

        if return_dict: 
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        else:
            return reward_tensor