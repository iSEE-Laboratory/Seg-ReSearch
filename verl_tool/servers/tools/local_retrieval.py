"""
Multi-modal Search Retrieval Tool for verl-tool
"""
import json

from .base import BaseTool, register_tool
import regex as re
import requests
from typing import Tuple, Dict, Any, List
import logging
from PIL import Image
from verl_tool.agent_loop.vision_utils import encode_image_url


logger = logging.getLogger(__name__)

@register_tool
class MMSearchRetrievalTool(BaseTool):
    tool_type = "mm_search_retrieval"
    
    def __init__(self, num_workers=1, retriever_url="http://127.0.0.1:8000/retrieve", topk=2,
                 img_retriever_url="http://127.0.0.1:8001/retrieve", topk_img=1, **kwargs):
        super().__init__(num_workers)
        # Allow configuration from environment or kwargs
        import os
        self.retriever_url = kwargs.get('retriever_url', os.getenv('RETRIEVER_URL', retriever_url))
        self.image_retriever_url = kwargs.get('img_retriever_url', os.getenv('IMG_RETRIEVER_URL', img_retriever_url))
        self.topk = kwargs.get('topk', int(os.getenv('RETRIEVER_TOPK', str(topk))))
        self.topk_img = str(topk_img)
        self.img_width = 256
        self.img_height = 256
        self.max_length = 320
        logger.info(f"SearchRetrievalTool initialized with URL: {self.retriever_url}, topk: {self.topk}")
        logger.info(f"ImageSearchRetrievalTool initialized with URL: {self.image_retriever_url}, topk: {self.topk_img}")
    
    def get_usage_inst(self):
        return "You can search for information by putting your query between <search> and </search> tags."
    
    def _parse_search_query(self, action: str) -> str:
        """
        Extract the search query from the action string.
        This is a helper function to parse the <search> tags.
        
        Args:
            action: Raw action string containing search query
            
        Returns:
            Extracted search query
        """
        # Priority logic moved from serve.py: prioritize search tool for <search> tags
        # This implements the original logic: if "</search>" in action and "search_retrieval" in self.tools
        if "</search>" in action:
            # Extract search query from <search>query</search> tags
            search_matches = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)
            
            if len(search_matches) > 0:
                # Use the last search query if multiple are found
                query = search_matches[-1].strip()
                return query, True
        return "", False
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string to extract search queries.
        Implements the prioritization logic that was originally in serve.py lines 112-115.
        
        Args:
            action: Raw action string containing search query
            
        Returns:
            Tuple containing the extracted query and a validity flag
        """
        # Check for <search> tags first
        search_query, is_valid = self._parse_search_query(action)
        if is_valid:
            return search_query, True
        
        # Default case - no valid action found
        return "", False
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """
        Get priority for handling this action. SearchRetrieval has high priority for <search> tags.
        This moves the tool identification logic from serve.py to the tool itself.
        
        Args:
            action: The raw action string
            extra_field: Extra fields associated with the action
        Returns:
            priority: Integer priority (-1 means cannot handle, higher numbers = higher priority)
        """
        # High priority for actions with </search> tags (original logic from serve.py line 112-115)
        if "</search>" in action:
            _, valid = self.parse_action(action)
            if valid:
                return 100  # High priority for search actions
        
        # Standard priority check
        _, valid = self.parse_action(action)
        return 0 if valid else -1
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute search query via retrieval service.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string containing search query
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_query, is_valid = self._parse_search_query(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            done = False
            valid = False
            try:
                query_dict = json.loads(parsed_query)
            except json.JSONDecodeError as e:
                observation = f"Invalid JSON format in search query: {str(e)}"
            else:
                # check parameters
                missing_parameters = []
                if 'query' not in query_dict:
                    missing_parameters.append('query')
                if 'name' not in query_dict:
                    missing_parameters.append('name')

                if missing_parameters:
                    observation = f"Missing parameters: {', '.join(missing_parameters)}"

                else:
                    query_string = query_dict['query']
                    tool_name = query_dict['name']
                    valid_tools = ['text_search', 'image_search']
                    if tool_name not in valid_tools:
                        observation = f"Invalid 'name' parameter. Expected one of {valid_tools}, but got '{tool_name}'."
                        valid = False
                    else:
                        if tool_name == 'image_search':
                            observation, valid = self._conduct_image_search(query_string)
                        else:
                            observation, valid = self._conduct_text_search(query_string)
        
        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, done, valid

    def _conduct_image_search(self, query: str):
        try:
            payload = {
                "queries": [query],
                "topk": self.topk_img,
                "return_scores": True
            }
            response = requests.post(self.image_retriever_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()['result'][0]

            format_reference = ''
            images = []
            image_paths = []
            for idx, doc_item in enumerate(result):
                name = doc_item['document'].get('name', '')  # unused
                img_path = doc_item['document'].get('image_path', '')
                image = Image.open(img_path).resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
                images.append(image)
                image_paths.append(img_path)
                format_reference += f"{idx + 1}. (Title: {name}) <image>\n"

            observation = {
                'obs': f"<information>{format_reference}</information>\n\n",
                'image': [encode_image_url(img) for img in images],
                'image_path': image_paths
            }
            valid = True

        except Exception as e:
            observation = f"Error searching images: {str(e)}"
            valid = False
            with open('test.json', 'w') as f:
                json.dump(query, f, indent=4)
            print(f"Error searching images: {str(e)}; query: {query}")

        return observation, valid

    def _conduct_text_search(self, query: str):
        try:
            payload = {
                "queries": [query],
                "topk": self.topk,
                "return_scores": True
            }
            response = requests.post(self.retriever_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()['result'][0]

            format_reference = ''
            for idx, doc_item in enumerate(result):
                if 'document' in doc_item:
                    content = doc_item['document']['contents']
                else:
                    content = doc_item.get('contents', '')

                title = content.split("\n")[0] if content else "No title"
                text = "\n".join(content.split("\n")[1:]) if content else "No content"
                if len(text) > self.max_length:
                    text = text[:self.max_length-1] + "…"
                format_reference += f"{idx + 1}. (Title: {title}) {text}\n"

            observation = f"<information>{format_reference}</information>\n\n"
            valid = True

        except Exception as e:
            observation = f"Error searching text: {str(e)}"
            valid = False
            with open('test.json', 'w') as f:
                json.dump(query, f, indent=4)
            print(f"Error searching images: {str(e)}; query: {query}")

        return observation, valid
