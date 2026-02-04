"""
Keyframe Tool for verl-tool
Extracts and processes a single keyframe from a sequence of images.
"""
from .base import BaseTool, register_tool
import regex as re
import asyncio
import concurrent.futures
from typing import Dict, Any, Tuple
import os
import base64
import io
from PIL import Image
from pathlib import Path
from verl_tool.agent_loop.vision_utils import process_image, encode_image_url


@register_tool
class KeyframeTool(BaseTool):
    tool_type = "keyframe"
    
    stop_tokens = ["</keyframe>"]
    
    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        # Create a thread pool for CPU-intensive image processing
        self.image_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="keyframe_processor"
        )
        self.img_width = 864
        self.img_height = 864
    
    def get_usage_inst(self):
        return "You can select a keyframe by putting a frame index between <keyframe> and </keyframe> tags. Format: <keyframe> 3 </keyframe>"
    
    def parse_action(self, action: str) -> Tuple[int, bool]:
        """
        Parse the raw action string to extract keyframe index.
        
        Args:
            action: Raw action string containing keyframe selection
            
        Returns:
            Tuple containing the frame index and a validity flag
        """
        if "</keyframe>" not in action:
            return -1, False
        
        try:
            # Extract content between <keyframe> and </keyframe>
            match = re.search(r"<keyframe>\s*(.*?)\s*</keyframe>", action, re.DOTALL)
            if not match:
                return -1, False
            
            content = match.group(1).strip()
            
            # Parse as integer
            frame_index = int(content)
            
            return frame_index, True
            
        except (ValueError, AttributeError):
            return -1, False
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """
        Get priority for handling this action.
        
        Args:
            action: The raw action string
            extra_field: Extra fields associated with the action
            
        Returns:
            priority: Integer priority (-1 means cannot handle, higher numbers = higher priority)
        """
        if "</keyframe>" in action:
            _, valid = self.parse_action(action)
            if valid:
                return 100  # High priority for keyframe selection actions
        
        return -1
    
    def load_env(self, trajectory_id):
        """Load the environment for the given trajectory_id"""
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "image_paths": None
            }
        return env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation):
        """Update the environment for the given trajectory_id"""
        # Minimal update - just save the environment back
        pass
    
    def delete_env(self, trajectory_id):
        """Delete the environment for the given trajectory_id"""
        env = self.env_cache.pop(trajectory_id, None)
    
    async def _process_single_image(self, img_path):
        """Process a single image asynchronously."""
        def _process():
            return Image.open(img_path).resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.image_executor, _process)
    
    async def _conduct_keyframe_selection_async(self, frame_index: int, env):
        """
        Execute the keyframe selection action asynchronously.
        
        Args:
            frame_index: Frame index to select (1-indexed)
            env: Environment containing available images
            
        Returns:
            Tuple containing observation and validity flag
        """
        valid = False
        
        if env['image_paths'] is None or len(env['image_paths']) == 0:
            observation = "No images available in the environment."
            return observation, valid
        
        available_frames = [int(Path(img).stem) for img in env['image_paths']]
        
        # Check index is valid
        if frame_index not in available_frames:
            observation = f"Frame index not in {available_frames}."
            return observation, valid
        
        try:
            i = available_frames.index(frame_index)
            target_frame_source = env['image_paths'][i]
            
            # Process the frame
            keyframe = await self._process_single_image(target_frame_source)
            
            observation = {
                'obs': f"Here is the high-res version of your selected frame: <image>",
                'image': [encode_image_url(keyframe)]
            }
            valid = True
            
        except Exception as e:
            observation = f"Error processing keyframe: {str(e)}"
            valid = False
        
        return observation, valid
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute keyframe selection action.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string containing keyframe selection
            extra_field: Additional parameters including 'image_paths' list
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_index, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        # Initialize images from extra_field if not already set
        if env['image_paths'] is None:
            env['image_paths'] = extra_field.get('image_paths', [])
        
        if not is_valid:
            observation = "Invalid keyframe selection format. Please use: <keyframe> integer </keyframe>"
            done = False
            valid = False
        else:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                observation, valid = loop.run_until_complete(
                    self._conduct_keyframe_selection_async(parsed_index, env)
                )
                done = False
            finally:
                loop.close()
        
        self.update_env(trajectory_id, env, parsed_index, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def __del__(self):
        """Cleanup when tool is destroyed."""
        if hasattr(self, 'image_executor'):
            self.image_executor.shutdown(wait=False)