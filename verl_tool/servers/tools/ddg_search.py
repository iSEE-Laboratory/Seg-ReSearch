"""
Multi-modal DuckDuckGo Search Tool with aiohttp
Uses aiohttp for better async performance and SSL handling
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
from collections import OrderedDict
from PIL import Image
import io
import tempfile
import ssl

from ddgs import DDGS
from firecrawl import FirecrawlApp
from .base import BaseTool, register_tool
from verl_tool.agent_loop.vision_utils import encode_image_url

DEBUG = False
SAVE_IMAGES_TO_DISK = True

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
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            self._cache[key] = value
            self._timestamps[key] = time.time()


class MMDDGSearchEngine:
    """Multi-modal async DuckDuckGo search engine with aiohttp."""

    def __init__(
        self,
        topk: int = 3,
        topk_img: int = 1,
        max_length: int = 320,
        cache_file: Optional[str] = None,
        search_mode: str = "fast",
        firecrawl_api_key: str = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        img_width: int = 448,
        img_height: int = 448,
        temp_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        self.topk = topk
        self.topk_img = topk_img
        self.max_length = max_length
        self.search_mode = search_mode
        self.img_width = img_width
        self.img_height = img_height
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup Firecrawl
        self.firecrawl_app = None
        if search_mode == "pro":
            if FirecrawlApp is None:
                raise ImportError("firecrawl-py required for pro mode")
            
            if firecrawl_api_key is None:
                firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
            
            if firecrawl_api_key:
                self.firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
            else:
                print("Warning: FIRECRAWL_API_KEY not set. Falling back to fast mode.")
                self.search_mode = "fast"
        
        # Temporary directory
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="ddg_images_")
        else:
            self.temp_dir = temp_dir
            os.makedirs(self.temp_dir, exist_ok=True)
        
        # Caching
        self._memory_cache = AsyncLRUCache(cache_size, cache_ttl)
        self._setup_cache_file(cache_file)
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.0
        
        # aiohttp session (will be created on first use)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
    
    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path and image cache directory."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            suffix = f"{self.search_mode}"
            self._cache_file = cache_dir / f"mm_ddg_search_{suffix}_cache.jsonl"
            self._image_cache_dir = cache_dir / "images"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._image_cache_dir = self._cache_file.parent / "images"
        
        if SAVE_IMAGES_TO_DISK:
            self._image_cache_dir.mkdir(exist_ok=True)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper SSL configuration."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    # Create SSL context that's more permissive
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    
                    # Configure connector with SSL context
                    connector = aiohttp.TCPConnector(
                        ssl=ssl_context,
                        limit=100,  # Connection pool limit
                        limit_per_host=30,
                        ttl_dns_cache=300,
                        enable_cleanup_closed=True
                    )
                    
                    # Create timeout configuration
                    timeout = aiohttp.ClientTimeout(
                        total=30,
                        connect=10,
                        sock_read=15
                    )
                    
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        }
                    )
        
        return self._session
    
    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        self._last_request_time = time.time()
    
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
                            search_type = item.get('search_type', 'text')
                            query = item['query']
                            result = item['result']
                            
                            if search_type == 'text':
                                mode = item.get('search_mode', 'fast')
                                topk = item.get('topk', 3)
                                length = item.get('max_length', 320)
                                cache_key = f"text:{mode}:{topk}:{length}:{query}"
                            elif search_type == 'img':
                                w = item.get('img_width', 256)
                                h = item.get('img_height', 256)
                                cache_key = f"img:{w}x{h}:{query}"
                            else:
                                continue
                            
                            await self._memory_cache.set(cache_key, result)
                            cache_entries += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                print(f"Loaded {cache_entries} cache entries from {self._cache_file}")
                
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    async def _append_to_persistent_cache(self, query: str, result: Any, search_type: str = "text") -> None:
        """Append cache entry to persistent storage."""
        try:
            entry = {
                "query": query,
                "result": result,
                "search_type": search_type,
                "timestamp": time.time()
            }
            
            if search_type == 'text':
                entry["search_mode"] = self.search_mode
                entry["topk"] = self.topk
                entry["max_length"] = self.max_length
            elif search_type == 'img':
                entry["img_width"] = self.img_width
                entry["img_height"] = self.img_height
            
            async with aiofiles.open(self._cache_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Cache write failed: {e}")
    
    async def _execute_ddgs_search_with_retry(self, search_func, query: str, max_results: int):
        """
        Execute DDGS search with retry logic and proper error handling.
        
        Args:
            search_func: The DDGS search function (text or images)
            query: Search query string
            max_results: Maximum number of results
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                await self._rate_limit()
                
                # Execute search in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self._do_ddgs_search(search_func, query, max_results)
                )
                
                return results, None
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                if DEBUG:
                    print(f"DDGS search attempt {attempt + 1}/{self.max_retries} failed: {error_msg}")
                
                # Check if it's an SSL error
                if "SSL" in error_msg or "EOF" in error_msg:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        wait_time = self.retry_delay * (2 ** attempt)
                        if DEBUG:
                            print(f"SSL error detected. Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                
                # For other errors, also retry but with shorter delay
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
        
        # All retries failed
        return None, last_error
    
    def _do_ddgs_search(self, search_func, query: str, max_results: int):
        """
        Perform the actual DDGS search (runs in thread pool).
        This creates a fresh DDGS instance for each search.
        """
        try:
            # Create new DDGS instance with explicit timeout
            with DDGS(timeout=20) as searcher:
                if search_func == 'text':
                    return list(searcher.text(query, max_results=max_results))
                elif search_func == 'images':
                    return list(searcher.images(query, max_results=max_results))
                else:
                    raise ValueError(f"Unknown search function: {search_func}")
        except Exception as e:
            # Re-raise to be caught by retry logic
            raise e

    async def execute_text_search(self, query: str):
        """Execute text search with robust error handling."""
        try:
            # Check cache
            cache_key = f"text:{self.search_mode}:{self.topk}:{self.max_length}:{query}"
            cached_result = await self._memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result, True

            # Execute search with retry
            results, error = await self._execute_ddgs_search_with_retry('text', query, self.topk)
            
            if results is None:
                error_msg = f"Search failed after {self.max_retries} attempts: {error}"
                observation = f"<information>{error_msg}</information>\n\n"
                return observation, False

            # Process results
            if self.search_mode == "pro" and self.firecrawl_app:
                for result in results:
                    try:
                        web_url = result.get('href', result.get('link', ''))
                        if web_url:
                            web_content = self.firecrawl_app.scrape_url(web_url)
                            result['web_content_markdown'] = web_content.get('markdown', '')
                            result['web_content_metadata'] = web_content.get('metadata', {})
                    except Exception as e:
                        if DEBUG:
                            print(f"Firecrawl failed for {web_url}: {e}")
                        result['web_content_markdown'] = ''
                        result['web_content_metadata'] = {}

            # Format results
            format_reference = ''
            for idx, result in enumerate(results):
                title = result.get('title', '')
                snippet = result.get('body', result.get('snippet', ''))
                
                if 'web_content_markdown' in result and result['web_content_markdown']:
                    text = result['web_content_markdown']
                else:
                    text = snippet
                
                if len(text) > self.max_length:
                    text = text[:self.max_length - 1] + "…"
                
                format_reference += f"{idx + 1}. (Title: {title}) {text}\n"

            observation = f"<information>{format_reference}</information>\n\n"
            
            # Cache result
            await self._memory_cache.set(cache_key, observation)
            await self._append_to_persistent_cache(query, observation, "text")
            
            return observation, True

        except Exception as e:
            observation = f"<information>Error searching text: {str(e)}</information>\n\n"
            print(f"Error searching text: {str(e)}; query: {query}")
            return observation, False

    async def execute_image_search(self, query: str):
        """Execute image search with robust error handling."""
        try:
            # Execute search with retry
            results, error = await self._execute_ddgs_search_with_retry(
                'images', query, self.topk_img + 5
            )
            
            if results is None:
                error_msg = f"Image search failed after {self.max_retries} attempts: {error}"
                observation = {
                    "obs": f"<information>{error_msg}</information>\n\n",
                    "image": [],
                    "image_path": []
                }
                return observation, False

            images, image_urls, format_reference = [], [], ""
            failed_count = 0

            for img_result in results:
                if len(images) >= self.topk_img:
                    break
                    
                image_url = img_result.get('image', img_result.get('url', ''))
                image_title = img_result.get('title', 'Unknown')
                
                if not image_url:
                    continue

                # Check cache
                img_cache_key = f"img:{self.img_width}x{self.img_height}:{image_url}"
                cached_image = await self._memory_cache.get(img_cache_key)

                if cached_image is not None:
                    encoded_img = cached_image
                else:
                    # Download image
                    encoded_img = await self._download_and_encode_image(image_url)

                    if encoded_img and not isinstance(encoded_img, Exception):
                        await self._memory_cache.set(img_cache_key, encoded_img)
                        await self._append_to_persistent_cache(image_url, encoded_img, "img")

                        if SAVE_IMAGES_TO_DISK:
                            await self._save_image_to_disk(image_url, encoded_img)
                    else:
                        failed_count += 1
                        continue

                if encoded_img and not isinstance(encoded_img, Exception):
                    images.append(encoded_img)
                    image_urls.append(image_url)
                    format_reference += f"{len(images)}. (Title: {image_title}) <image>\n"

            if DEBUG and failed_count > 0:
                print(f"Failed to download {failed_count} images for query: {query}")

            observation = {
                "obs": f"<information>{format_reference}</information>\n\n",
                "image": images,
                "image_path": image_urls
            }
            valid = len(images) > 0

            return observation, valid

        except Exception as e:
            observation = {
                "obs": f"<information>Error searching images: {str(e)}</information>\n\n",
                "image": [],
                "image_path": []
            }
            print(f"Error searching images: {str(e)}; query: {query}")
            return observation, False

    async def _download_and_encode_image(self, image_url: str) -> Optional[str]:
        """Download image from URL using aiohttp and encode it."""
        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                
                headers = {
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Referer": "https://www.google.com/"
                }
                
                async with session.get(image_url, headers=headers) as response:
                    response.raise_for_status()
                    content = await response.read()
                
                # Process image in thread pool (PIL is blocking)
                loop = asyncio.get_event_loop()
                encoded_img = await loop.run_in_executor(
                    None,
                    self._process_image,
                    content
                )
                
                return encoded_img
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    if DEBUG:
                        print(f"Image download retry {attempt + 1}/{self.max_retries} for {image_url}: {type(e).__name__}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    if DEBUG:
                        print(f"Failed to download image after {self.max_retries} attempts: {image_url}")
                    return None
                    
            except Exception as e:
                if DEBUG:
                    print(f"Unexpected error downloading image {image_url}: {e}")
                return None
        
        return None
    
    def _process_image(self, content: bytes) -> Optional[str]:
        """Process image data (resize and encode). Runs in thread pool."""
        try:
            image = Image.open(io.BytesIO(content))
            image = image.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
            return encode_image_url(image)
        except Exception as e:
            if DEBUG:
                print(f"Image processing error: {e}")
            return None
    
    async def _save_image_to_disk(self, image_url: str, encoded_img: str) -> None:
        """Save image to disk for visualization."""
        try:
            import hashlib
            import base64
            
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:16]
            filename = f"{url_hash}_{self.img_width}x{self.img_height}.png"
            filepath = self._image_cache_dir / filename
            
            if ',' in encoded_img:
                encoded_img = encoded_img.split(',', 1)[1]
            
            image_data = base64.b64decode(encoded_img)
            
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(image_data)
            
            if DEBUG:
                print(f"Saved image to: {filepath}")
            
        except Exception as e:
            if DEBUG:
                print(f"Failed to save image to disk: {e}")


@register_tool
class MMDDGSearchTool(BaseTool):
    """
    Multi-modal DuckDuckGo search tool supporting both text and image search.
    Uses free DuckDuckGo API with aiohttp for better async performance.
    """
    
    tool_type = "mm_ddg_search"
    
    def __init__(
        self,
        num_workers=1,
        topk: int = 3,
        topk_img: int = 1,
        max_length: int = 320,
        cache_file: Optional[str] = None,
        search_mode: str = "fast",
        firecrawl_api_key: str = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        img_width: int = 256,
        img_height: int = 256,
        temp_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs
    ):
        """Initialize the multi-modal DuckDuckGo search tool."""
        super().__init__(num_workers)
        
        # Initialize search engine
        self.search_engine = MMDDGSearchEngine(
            topk=topk,
            topk_img=topk_img,
            max_length=max_length,
            cache_file=cache_file,
            search_mode=search_mode,
            firecrawl_api_key=firecrawl_api_key,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            img_width=img_width,
            img_height=img_height,
            temp_dir=temp_dir,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

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
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'search_engine') and self.search_engine._session:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.search_engine.close())
                else:
                    loop.run_until_complete(self.search_engine.close())
        except:
            pass
    
    def get_usage_inst(self):
        """Get usage instructions."""
        mode_desc = "with full web content" if self.search_engine.search_mode == "pro" else "fast mode"
        return (
            f"Search the web using DuckDuckGo ({mode_desc}). "
            "Use <search>{\"query\": \"your query\", \"name\": \"text_search\"}</search> for text search, "
            "or <search>{\"query\": \"your query\", \"name\": \"image_search\"}</search> for image search."
        )

    def _parse_search_query(self, action: str):
        """Extract the search query from the action string."""
        if "</search>" in action:
            search_matches = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)
            if len(search_matches) > 0:
                query = search_matches[-1].strip()
                return query, True
        return "", False

    def parse_action(self, action: str) -> Tuple[str, bool]:
        """Parse the raw action string to extract search queries."""
        search_query, is_valid = self._parse_search_query(action)
        if is_valid:
            return search_query, True
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
                    query_string = query_dict['query']
                    tool_name = query_dict['name']
                    valid_tools = ['text_search', 'image_search']
                    if tool_name not in valid_tools:
                        observation = f"Invalid 'name' parameter. Expected one of {valid_tools}, but got '{tool_name}'."
                        valid = False
                    else:
                        if tool_name == 'image_search':
                            observation, valid = await self.search_engine.execute_image_search(query_string)
                        else:
                            observation, valid = await self.search_engine.execute_text_search(query_string)

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