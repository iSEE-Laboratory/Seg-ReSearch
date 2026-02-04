"""
Multi-modal Google Search Tool for verl-tool
Combines Google Search capabilities with multi-modal support
"""
import os
import json
import time
import pathlib
import asyncio
import aiofiles
import aiohttp
import regex as re
from typing import Optional, Union, Dict, List, Any, Tuple
import langid
from collections import OrderedDict
from PIL import Image
import io

from .base import BaseTool, register_tool
from .utils.deepsearch_utils import extract_relevant_info_serper, extract_text_from_url, extract_snippet_with_context
from .utils.web_agent_utils import generate_webpage_to_reasonchain, get_prev_reasoning_chain
from verl_tool.agent_loop.vision_utils import encode_image_url

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

DEBUG = False


class AsyncLRUCache:
    """Thread-safe LRU cache for async operations"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                # Check TTL
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()


class MMGoogleSearchEngine:
    """
    Multi-modal async Google search engine supporting text and image search.
    """

    def __init__(
        self,
        api_key: str,
        topk: int = 3,
        topk_img: int = 1,
        max_length: int = 320,
        location: str = "us",
        language: str = "en",
        cache_file: Optional[str] = None,
        process_snippets: bool = False,
        summ_model_url: str = None,
        summ_model_path: str = None,
        max_doc_len: int = 3000,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        img_width: int = 448,
        img_height: int = 448
    ):
        """Initialize the multi-modal search engine."""
        # API configuration
        self._api_key = api_key
        self.topk = topk
        self.topk_img = topk_img
        self.max_length = max_length
        self._location = location
        self._language = language
        self.process_snippets = process_snippets
        self.summ_model_url = summ_model_url
        self.summ_model_path = summ_model_path
        self._max_doc_len = max_doc_len
        self.img_width = img_width
        self.img_height = img_height
        
        # Async-safe caching
        self._memory_cache = AsyncLRUCache(cache_size, cache_ttl)
        self._setup_cache_file(cache_file)
        
        # Performance tracking
        self._search_count = 0
    
    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            suffix = "with_summ" if self.process_snippets else "basic"
            self._cache_file = cache_dir / f"mm_google_search_{suffix}_cache.jsonl"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def _load_persistent_cache(self) -> None:
        """Load cache from file asynchronously."""
        if not self._cache_file.exists():
            return
            
        try:
            async with aiofiles.open(self._cache_file, "r", encoding="utf-8") as f:
                cache_entries = 0
                async for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            cache_key = f"{item.get('search_type', 'text')}:{item['query']}"
                            await self._memory_cache.set(cache_key, item['result'])
                            cache_entries += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                print(f"Loaded {cache_entries} cache entries from {self._cache_file}")
                
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    async def _append_to_persistent_cache(self, query: str, result: Any, search_type: str = "text") -> None:
        """Append to persistent cache asynchronously."""
        try:
            entry = {
                "query": query,
                "result": result,
                "search_type": search_type,
                "timestamp": time.time()
            }
            
            async with aiofiles.open(self._cache_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Cache write failed: {e}")
    
    async def _detect_language(self, query: str) -> Tuple[str, str]:
        """Detect language for the query."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            lang_code = await loop.run_in_executor(
                None, lambda: langid.classify(query)[0]
            )
            
            if lang_code == 'zh':
                return "zh-cn", "cn"
            else:
                return self._language, self._location
                
        except Exception as e:
            print(f"Language detection failed: {e}")
            return self._language, self._location
    
    async def _make_search_request(self, query: str, search_type: str = "search", timeout: int = 30) -> Dict:
        """
        Make search request to Serper API.
        
        Args:
            query: Search query string
            search_type: Type of search - "search" for text, "images" for image search
            timeout: Request timeout in seconds
        """
        hl, gl = await self._detect_language(query)
        
        # Add buffer for image search to handle bad URLs
        if search_type == "images":
            num_results = min(self.topk_img + 5, 100)  # +5 buffer to avoid bad urls
        else:
            num_results = min(self.topk, 100)
        
        payload = {
            "q": query,
            "hl": hl,
            "gl": gl,
            "num": num_results
        }

        headers = {
            'X-API-KEY': self._api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'MMAsyncSearchEngine/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        }

        # Determine API endpoint
        if search_type == "images":
            url = "https://google.serper.dev/images"
        else:
            url = "https://google.serper.dev/search"

        # Create a new session for each request
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        
        # Retry logic for transient failures
        max_retries = 2
        for attempt in range(max_retries + 1):
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                try:
                    async with session.post(url, headers=headers, json=payload) as response:
                        
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limited
                            if attempt < max_retries:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            else:
                                raise Exception(f"Rate limited after {max_retries} retries")
                        else:
                            text = await response.text()
                            raise Exception(f"API error {response.status}: {text[:200]}")
                            
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        timeout = min(timeout * 1.5, 60)
                        timeout_config = aiohttp.ClientTimeout(total=timeout)
                        continue
                    else:
                        raise Exception(f"Request timed out after {max_retries} retries")
                except Exception as e:
                    if attempt < max_retries and "timeout" in str(e).lower():
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise
    
    async def execute_text_search(self, query: str, timeout: int = None, prev_steps: Union[List[str], str] = None) -> Tuple[str, bool]:
        """Execute text search."""
        query = query.strip().replace('"', '')
        if not query:
            return "<information>Empty search query provided.</information>\n\n", False
        
        if len(query) > 500:
            return "<information>Search query too long (maximum 500 characters).</information>\n\n", False
        
        try:
            # Check memory cache
            cache_key = f"text:{query}"
            cached_result = await self._memory_cache.get(cache_key)
            if cached_result is not None:
                if not self.process_snippets:
                    return cached_result, True
                else:
                    data = json.loads(cached_result) if isinstance(cached_result, str) else cached_result
                    result = await self._process_cached_data(query, data, prev_steps)
                    return result, True
            
            # Make API request
            data = await self._make_search_request(query, "search", timeout or 30)
            
            # Process results
            result = await self._extract_and_format_results(query, data, prev_steps)
            
            # Cache results
            await self._cache_results(query, data if self.process_snippets else result, "text")
            
            return result, True
            
        except Exception as e:
            if DEBUG:
                raise e
            observation = f"<information>Error searching text: {str(e)}</information>\n\n"
            valid = False
            print(f"Error searching text: {str(e)}; query: {query}")
            return observation, valid
    
    async def execute_image_search(self, query: str, timeout: int = None) -> Tuple[Dict[str, Any], bool]:
        """
        Execute image search and return image URLs and metadata.
        
        Returns:
            Tuple containing (Dict with 'obs', 'image', 'image_path', validity flag)
        """
        query = query.strip().replace('"', '')
        if not query:
            return {"obs": "<information>Empty search query provided.</information>\n\n", "image": [], "image_path": []}, False
        
        if len(query) > 500:
            return {"obs": "<information>Search query too long (maximum 500 characters).</information>\n\n", "image": [], "image_path": []}, False
        
        try:
            # Check memory cache
            # cache_key = f"{self.topk_img}_image:{query}"
            cache_key = f"image:{query}"
            cached_result = await self._memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result, True
            
            # Make API request
            data = await self._make_search_request(query, "images", timeout or 30)
            
            # Process image results
            result = await self._process_image_results(query, data)
            
            # Cache results
            await self._cache_results(query, result, "image")
            
            return result, True
            
        except Exception as e:
            if DEBUG:
                raise e
            observation = {
                "obs": f"<information>Error searching images: {str(e)}</information>\n\n",
                "image": [],
                "image_path": []
            }
            valid = False
            print(f"Error searching images: {str(e)}; query: {query}")
            return observation, valid
    
    async def _process_image_results(self, query: str, data: Dict) -> Dict[str, Any]:
        """Process image search results."""
        if 'images' not in data or not data['images']:
            return {"obs": "<information>No image results found.</information>\n\n", "image": [], "image_path": []}
        
        images = []
        image_urls = []
        format_reference = ""
        
        # Process images with buffer - iterate through all results until we get topk_img valid images
        for img_result in data['images']:
            image_url = img_result.get('imageUrl', '')
            title = img_result.get('title', 'No title')
            source = img_result.get('source', 'Unknown source')
            
            if not image_url:
                continue
            
            try:
                # Download and encode image
                encoded_img = await self._download_and_encode_image(image_url)
                if encoded_img:
                    images.append(encoded_img)
                    image_urls.append(image_url)
                    # Format: {idx}. (Title: {title}) <image>
                    format_reference += f"{len(images)}. (Title: {title}) <image>\n"
            except Exception as e:
                continue
            
            # Stop once we have enough valid images
            if len(images) >= self.topk_img:
                break
        
        observation = f"<information>{format_reference}</information>\n\n"
        
        return {
            "obs": observation,
            "image": images,
            "image_path": image_urls
        }
    
    async def _download_and_encode_image(self, image_url: str) -> Optional[str]:
        """Download image from URL and encode it."""
        try:
            timeout_config = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        # Open and resize image
                        image = Image.open(io.BytesIO(image_data))
                        image = image.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
                        
                        # Encode to base64
                        return encode_image_url(image)
                    else:
                        return None
        except Exception as e:
            return None
    
    async def _process_cached_data(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        """Process cached data for snippet processing mode."""
        return await self._extract_and_format_results(query, data, prev_steps)
    
    async def _cache_results(self, query: str, data: Any, search_type: str = "text") -> None:
        """Cache results in both memory and persistent storage."""
        try:
            cache_key = f"{search_type}:{query}"
            # if search_type == "image":
            #     cache_key = f"{self.topk_img}_{cache_key}"
            # elif search_type == "text":
            #     cache_key = f"{self.topk}_{cache_key}"

            await self._memory_cache.set(cache_key, data)
            
            cache_item = data if isinstance(data, (str, dict)) else json.dumps(data, ensure_ascii=False)
            await self._append_to_persistent_cache(query, cache_item, search_type)
            
            self._search_count += 1
            
        except Exception as e:
            print(f"Caching failed: {e}")
    
    async def _extract_and_format_results(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        """Extract and format search results with async processing."""
        if 'organic' not in data or not data['organic']:
            return "<information>No search results found.</information>\n\n"
        
        if not self.process_snippets:
            format_reference = await self._format_basic_results(data)
        else:
            format_reference = await self._process_snippets_async(query, data, prev_steps)

        return f"<information>{format_reference}</information>\n\n"
    
    async def _format_basic_results(self, data: Dict) -> str:
        """Format basic search results without snippet processing."""
        format_reference = ''
        seen_snippets = set()

        for idx, result in enumerate(data['organic'][:self.topk]):
            title = result.get('title', 'No title').strip()
            snippet = result.get('snippet', result.get('description', '')).strip()

            if snippet and snippet not in seen_snippets:
                if len(snippet) > self.max_length:
                    snippet = snippet[:self.max_length-1] + "…"

                format_reference += f"{idx + 1}. (Title: {title}) {snippet}\n"
                seen_snippets.add(snippet)

        if format_reference:
            return format_reference
        else:
            return "No search results found."
    
    async def _process_snippets_async(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        """Process snippets with full content extraction asynchronously."""
        max_doc_len = self._max_doc_len if self.summ_model_url else self.max_length
        do_summarization = self.summ_model_url is not None and self.summ_model_path is not None
        
        # Extract info in thread pool
        loop = asyncio.get_event_loop()
        extracted_info = await loop.run_in_executor(
            None, extract_relevant_info_serper, data
        )
        
        # Process each URL concurrently
        processing_tasks = []
        for info in extracted_info:
            task = self._process_single_url(info, max_doc_len)
            processing_tasks.append(task)
        
        processed_info = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_info = []
        for i, result in enumerate(processed_info):
            if isinstance(result, Exception):
                print(f"URL processing failed: {result}")
                valid_info.append(extracted_info[i])
            else:
                valid_info.append(result)
        
        # Format document
        formatted_document = ""
        for i, doc_info in enumerate(valid_info):
            formatted_document += f"**Web Page {i + 1}:**\n"
            formatted_document += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

        if do_summarization and formatted_document:
            summary = await loop.run_in_executor(
                None, self._run_summarization, query, formatted_document, prev_steps
            )
            # Apply length constraint: max_length * topk (same as DDG)
            max_summary_length = self.max_length * self.topk
            if len(summary) > max_summary_length:
                summary = summary[:max_summary_length-1] + "…"
            return summary
        else:
            # Apply length constraint for formatted document
            max_doc_length = self.max_length * self.topk
            if len(formatted_document) > max_doc_length:
                formatted_document = formatted_document[:max_doc_length-1] + "…"
            return formatted_document if formatted_document else "<information>No relevant information found.</information>\n\n"
    
    async def _process_single_url(self, info: Dict, max_doc_len: int) -> Dict:
        """Process a single URL to extract context."""
        try:
            loop = asyncio.get_event_loop()
            full_text = await loop.run_in_executor(
                None, lambda: extract_text_from_url(info['url'], use_jina=False)
            )
            
            if full_text and not full_text.startswith("Error"):
                success, context = extract_snippet_with_context(
                    full_text, info['snippet'], context_chars=max_doc_len
                )
                if success:
                    info['context'] = context
                else:
                    info['context'] = f"Could not extract context. First {max_doc_len} chars: {full_text[:max_doc_len]}"
            else:
                info['context'] = f"Failed to fetch content: {full_text or 'Unknown error'}"
                
        except Exception as e:
            info['context'] = f"Error processing URL: {str(e)}"
        
        return info
    
    def _run_summarization(self, query: str, formatted_document: str, prev_steps: Union[List[str], str] = None) -> str:
        """Run summarization in sync context."""
        try:
            prev_reasoning_chain = get_prev_reasoning_chain(
                prev_steps, 
                begin_search_tag="<search>", 
                begin_search_result_tag="<information>"
            )
            return generate_webpage_to_reasonchain(
                prev_reasoning_chain,
                query,
                formatted_document,
                summ_model_url=self.summ_model_url,
                summ_model_path=self.summ_model_path
            )
        except Exception as e:
            if DEBUG:
                raise e
            print(f"Summarization failed: {e}")
            return formatted_document


@register_tool
class MMGoogleSearchTool(BaseTool):
    """
    Multi-modal Google search tool supporting both text and image search.
    """
    
    tool_type = "mm_google_search"
    
    def __init__(
        self,
        num_workers=1,
        api_key: str = None,
        topk: int = 3,
        topk_img: int = 1,
        max_length: int = 320,
        location: str = "us",
        language: str = "en",
        cache_file: Optional[str] = None,
        default_timeout: int = None,
        process_snippets: bool = True,
        summ_model_url: Optional[str] = None,
        summ_model_path: Optional[str] = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        img_width: int = 256,
        img_height: int = 256,
        **kwargs
    ):
        """Initialize the multi-modal search tool."""
        super().__init__(num_workers)
        
        # Validate API key
        if api_key is None:
            api_key = os.getenv('SERPER_API_KEY')
            if api_key is None:
                raise ValueError(
                    "API key required: set SERPER_API_KEY environment variable or pass api_key parameter"
                )
        
        # Read summarization model config from environment variables if not provided
        if summ_model_url is None:
            summ_model_url = os.getenv('SUMM_MODEL_URL')
        if summ_model_path is None:
            summ_model_path = os.getenv('SUMM_MODEL_PATH')
        
        # Initialize search engine
        self.search_engine = MMGoogleSearchEngine(
            api_key=api_key,
            topk=topk,
            topk_img=topk_img,
            max_length=max_length,
            location=location,
            language=language,
            cache_file=cache_file,
            process_snippets=process_snippets,
            summ_model_url=summ_model_url,
            summ_model_path=summ_model_path,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            img_width=img_width,
            img_height=img_height
        )
        
        self.default_timeout = default_timeout
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(16)
    
    async def _ensure_initialized(self):
        """Ensure search engine is initialized."""
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    await self.search_engine._load_persistent_cache()
                    self._initialized = True
    
    def get_usage_inst(self):
        """Get usage instructions."""
        return (
            "Search the web using Google (text or images). "
            "Use <search>{\"query\": \"your query\", \"name\": \"text_search\"}</search> for text search, "
            "or <search>{\"query\": \"your query\", \"name\": \"image_search\"}</search> for image search."
        )

    def _parse_search_query(self, action: str) -> Tuple[str, bool]:
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

    async def aget_observations(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Union[str, dict]], List[bool], List[bool]]:
        """Process multiple search actions concurrently."""
        await self._ensure_initialized()
        
        async def process_single_action(trajectory_id, action, extra_field):
            async with self.semaphore:
                try:
                    return await self._conduct_action_async(trajectory_id, action, extra_field)
                except Exception as e:
                    return f"Search error: {str(e)}", False, False
        
        tasks = [
            process_single_action(trajectory_id, action, extra_field)
            for trajectory_id, action, extra_field in zip(trajectory_ids, actions, extra_fields)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
                
        observations, dones, valids = [], [], []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if DEBUG:
                    raise result
                obs = f"Search error: {str(result)}"
                done, valid = False, False
            else:
                obs, done, valid = result
            
            observations.append(obs)
            dones.append(done)
            valids.append(valid)
        
        self.maybe_cleanup_env(trajectory_ids, actions, extra_fields)
        
        return observations, dones, valids
    
    async def _conduct_action_async(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[Union[str, Dict], bool, bool]:
        """Conduct single search action asynchronously."""
        parsed_query, is_valid = self._parse_search_query(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = ""
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
                    timeout = extra_field.get('timeout', self.default_timeout)
                    query_string = query_dict['query']
                    tool_name = query_dict['name']
                    valid_tools = ['text_search', 'image_search']
                    if tool_name not in valid_tools:
                        observation = f"Invalid 'name' parameter. Expected one of {valid_tools}, but got '{tool_name}'."
                        valid = False
                    else:
                        # Extract previous actions for snippet processing
                        prev_actions = []
                        if self.search_engine.process_snippets and env.get('previous_obs'):
                            prev_actions = [x.get('action') for x in env['previous_obs']]
                        prev_actions += [action]
                        
                        if tool_name == 'image_search':
                            observation, valid = await self.search_engine.execute_image_search(query_string, timeout)
                        else:
                            observation, valid = await self.search_engine.execute_text_search(query_string, timeout, prev_actions)

        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, done, valid
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[Union[str, Dict], bool, bool]:
        """Synchronous wrapper for async code."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import threading
                
                result = [None]
                exception = [None]
                
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result[0] = new_loop.run_until_complete(
                            self._conduct_action_async(trajectory_id, action, extra_field)
                        )
                    except Exception as e:
                        if DEBUG:
                            raise e
                        exception[0] = e
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_new_loop)
                thread.start()
                thread.join(timeout=60)
                
                if exception[0]:
                    raise exception[0]
                if result[0] is None:
                    return "Search timed out", False, False
                return result[0]
            else:
                return loop.run_until_complete(
                    self._conduct_action_async(trajectory_id, action, extra_field)
                )
        except RuntimeError:
            return asyncio.run(self._conduct_action_async(trajectory_id, action, extra_field))
        except Exception as e:
            if DEBUG:
                raise e
            return f"Search failed: {str(e)}", False, False

