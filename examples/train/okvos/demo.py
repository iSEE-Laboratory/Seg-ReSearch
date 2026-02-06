#!/usr/bin/env python3
"""
Simple demo for running single sample inference with RVOS agent.
Given a video directory, sample K frames and run inference.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import subprocess
import time
import requests
import signal
import regex as re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

system_prompt = """Your role is a video target identification assistant capable of web search. 
You will be given an object query and a sequence of video frames. Each frame is preceded by its index.
Your task is to locate the target with a bounding box and a point in your selected frame.

# Tools 
You have access to the following tools:

<tools>
{"name": "text_search", "description": " Search for textual information from the internet.", "parameters": {"query": {"type": "string", "description": "Search keywords or phrases"}}
{"name": "image_search", "description": " Search for images from the internet.", "parameters": {"query": {"type": "string", "description": "Search keywords or phrases"}}
</tools>

Each tool will return the top searched results between <information> and </information>.

# Output Format:
Depending on the situation, output one of the following:

## 1. If you need web search (zero to multiple times):
<thinking> [Your current findings and why a search is needed.] </thinking>
<search> {"name": <tool-name>, "query": <string>} </search>

## 2. If you confirm the target:
<thinking> [Combine all information, scan the video frames to find the frame where the target is most clearly visible.] </thinking>
<keyframe> [integer frame_index] </keyframe>

## 3. Once you receive the high-res keyframe:
<thinking> [Describe the unique visual features of the target to prove you captured it.] </thinking>
<answer> {"bbox_2d": [x1,y1,x2,y2], "point_2d": [x,y]} </answer>

"""


def load_frames_from_directory(video_dir, num_frames=6, max_size=640, min_size=384):
    """Load and uniformly sample K frames from a video directory."""
    video_path = Path(video_dir)
    
    if not video_path.exists():
        raise ValueError(f"Video directory {video_dir} does not exist")
    
    # Get all image files
    image_files = sorted([f for f in video_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {video_dir}")
    
    print(f"Found {len(image_files)} frames in {video_dir}")
    
    # Uniformly sample frames
    total_frames = len(image_files)
    if num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int).tolist()
    
    # Load first image to get dimensions
    first_img = Image.open(image_files[0])
    w, h = first_img.size
    
    # Calculate resize dimensions
    new_w = max_size if w > h else min_size
    new_h = max_size if h > w else min_size
    
    # Load and resize frames
    frames = []
    frame_ids = []
    for idx in indices:
        img_path = image_files[idx]
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        frames.append(img_resized)
        # Use frame index from filename (without extension)
        frame_id = int(img_path.stem) if img_path.stem.isdigit() else idx
        frame_ids.append(frame_id)
    
    return frames, frame_ids, [str(image_files[i]) for i in indices], (new_w, new_h)


def construct_prompt(frames, frame_ids, query):
    """Construct the prompt with frames."""
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": []
        }
    ]
    
    # Add frames to content
    for frame_id, frame in zip(frame_ids, frames):
        messages[1]["content"].append({
            "type": "text",
            "text": f"frame {frame_id} "
        })
        messages[1]["content"].append({
            "type": "image",
            "image": frame
        })
        messages[1]["content"].append({
            "type": "text",
            "text": "\n"
        })
    
    # Add query
    messages[1]["content"].append({
        "type": "text",
        "text": f"\n{query}"
    })
    
    return messages


def parse_tool_call(text):
    """Parse tool calls from generated text."""
    # Check for search
    search_match = re.search(r'<search>\s*(\{.*?\})\s*</search>', text, re.DOTALL)
    if search_match:
        try:
            search_json = json.loads(search_match.group(1))
            return 'search', search_json
        except:
            pass
    
    # Check for keyframe
    keyframe_match = re.search(r'<keyframe>\s*(\d+)\s*</keyframe>', text, re.DOTALL)
    if keyframe_match:
        frame_id = int(keyframe_match.group(1))
        return 'keyframe', frame_id
    
    # Check for answer
    answer_match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', text, re.DOTALL)
    if answer_match:
        try:
            answer_json = json.loads(answer_match.group(1))
            return 'answer', answer_json
        except:
            pass
    
    return None, None


def call_tool_server(tool_type, params, tool_server_url, trajectory_id, image_paths):
    """Call the tool server to get observation."""
    try:
        if tool_type == 'search':
            action = f"<search> {json.dumps(params)} </search>"
        elif tool_type == 'keyframe':
            action = f"<keyframe> {params} </keyframe>"
        else:
            return None
        
        payload = {
            'trajectory_ids': [trajectory_id],
            'actions': [action],
            'extra_fields': [{
                'image_paths': image_paths
            }]
        }
        
        response = requests.post(tool_server_url, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            # Extract the first observation from the batch response
            observations = result.get('observations', [])
            if observations:
                return observations[0]
            return None
        else:
            print(f"Tool server error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error calling tool server: {e}")
        return None


def run_inference(model, processor, messages, max_new_tokens=1024):
    """Run inference with the model."""
    # Prepare for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    
    # Decode only the generated part
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text


def main():
    parser = argparse.ArgumentParser(description='Demo for single sample RVOS inference')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to video directory containing frames')
    parser.add_argument('--query', type=str, required=True, help='Object query (e.g., "the red car")')
    parser.add_argument('--model_path', type=str, default='/data1/tianming/Qwen/Qwen3-VL-4B-Instruct', 
                        help='Path to model checkpoint')
    parser.add_argument('--num_frames', type=int, default=6, help='Number of frames to sample')
    parser.add_argument('--max_size', type=int, default=640, help='Max image dimension')
    parser.add_argument('--min_size', type=int, default=384, help='Min image dimension')
    parser.add_argument('--max_turns', type=int, default=6, help='Maximum agent turns')
    parser.add_argument('--tool_server_port', type=int, default=30754, help='Tool server port')
    parser.add_argument('--no_tool_server', action='store_true', help='Skip starting tool server')
    
    args = parser.parse_args()
    
    # Start tool server
    server_process = None
    import socket
    host = socket.gethostbyname(socket.gethostname())
    port = args.tool_server_port
    tool_server_url = f"http://{host}:{port}/get_observation"
    
    if not args.no_tool_server:
        print(f"Starting tool server at {tool_server_url}...")
        server_cmd = [
            'python', '-m', 'verl_tool.servers.serve',
            '--host', host,
            '--port', str(port),
            '--tool_type', 'local_retrieval,keyframe',
            '--workers_per_tool', '4'
        ]
        server_process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("Waiting for tool server to start...")
        time.sleep(5)
        
        # Check if server is ready
        server_ready = False
        for i in range(60):
            try:
                response = requests.get(f"http://{host}:{port}/health", timeout=2)
                if response.status_code == 200:
                    print(f"Tool server is ready! (took {i+5} seconds)")
                    server_ready = True
                    break
            except Exception as e:
                if i % 10 == 0:
                    print(f"Waiting for server... ({i}/60)")
            time.sleep(1)
        
        if not server_ready:
            print("Warning: Tool server may not be ready after 60 seconds")
    
    try:
        # Load frames
        print(f"\nLoading frames from {args.video_dir}...")
        frames, frame_ids, image_paths, (width, height) = load_frames_from_directory(
            args.video_dir, args.num_frames, args.max_size, args.min_size
        )
        print(f"Loaded {len(frames)} frames with IDs: {frame_ids}")
        print(f"Image dimensions: {width}x{height}")
        
        # Load model
        print(f"\nLoading model from {args.model_path}...")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        print("Model loaded!")
        
        # Construct initial prompt
        messages = construct_prompt(frames, frame_ids, args.query)
        
        print(f"\n{'='*80}")
        print(f"Query: {args.query}")
        print(f"{'='*80}\n")
        
        trajectory_id = f"demo_{os.getpid()}"
        
        # Agent loop
        for turn in range(args.max_turns):
            print(f"\n--- Turn {turn + 1} ---")
            
            # Generate response
            output_text = run_inference(model, processor, messages, max_new_tokens=1024)
            print(f"Generated: {output_text}")
            
            # Add assistant response to messages
            messages.append({
                "role": "assistant",
                "content": output_text
            })
            
            # Parse tool call
            tool_type, tool_params = parse_tool_call(output_text)
            
            if tool_type == 'answer':
                print(f"\n{'='*80}")
                print(f"Final Answer: {tool_params}")
                print(f"{'='*80}")
                break
            elif tool_type in ['search', 'keyframe'] and tool_server_url:
                # Call tool
                print(f"Calling {tool_type} tool with params: {tool_params}")
                observation = call_tool_server(tool_type, tool_params, tool_server_url, trajectory_id, image_paths)
                
                if observation:
                    if isinstance(observation, dict) and 'obs' in observation:
                        obs_text = observation['obs']
                        # Handle image in observation (for keyframe)
                        if 'image' in observation and observation['image']:
                            from verl_tool.agent_loop.vision_utils import decode_image_url
                            # Add observation with image
                            obs_content = []
                            obs_content.append({"type": "text", "text": "<information>\n"})
                            obs_content.append({"type": "text", "text": obs_text})
                            if isinstance(observation['image'], list):
                                for img_url in observation['image']:
                                    img = decode_image_url(img_url)
                                    obs_content.append({"type": "image", "image": img})
                            obs_content.append({"type": "text", "text": "\n</information>"})
                            
                            messages.append({
                                "role": "user",
                                "content": obs_content
                            })
                        else:
                            messages.append({
                                "role": "user",
                                "content": f"<information>\n{obs_text}\n</information>"
                            })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"<information>\n{observation}\n</information>"
                        })
                    print(f"Observation: {observation if isinstance(observation, str) else observation.get('obs', observation)[:200]}")
                else:
                    print("No observation received from tool")
                    break
            elif tool_type is None:
                print("No tool call detected, continuing...")
            else:
                print(f"Unknown tool type or tool server not available: {tool_type}")
                break
        else:
            print("\nReached maximum turns without final answer")
        
        print(f"\n{'='*80}")
        print("Demo completed!")
        print(f"{'='*80}")
        
    finally:
        # Cleanup: kill server
        if server_process:
            print("\nStopping tool server...")
            server_process.send_signal(signal.SIGTERM)
            server_process.wait(timeout=5)
            print("Tool server stopped")


if __name__ == '__main__':
    main()
