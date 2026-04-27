"""Reverse-image search tool for the DeepEyesV2 pipeline.

Supports two backends:
  * **Cache-based** (default) -- loads pre-built JSON cache files keyed by
    ``data_idx``, compatible with the DeepEyesV2 search-cache format from
    https://huggingface.co/datasets/honglyhly/DeepEyesV2_Search_Cache
  * **SerpAPI-based** (optional) -- performs live reverse-image search when
    ``serpapi_key`` is set in the tool config.
"""

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ImageSearchExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = (
            self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        )

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(
            name="image-search-rate-limiter", get_if_exists=True
        ).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error during image search execution: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_image_search_pool(
    num_workers: int,
    enable_global_rate_limit=True,
    rate_limit=10,
    mode: PoolMode = PoolMode.ThreadMode,
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ImageSearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(
                enable_global_rate_limit=enable_global_rate_limit,
                rate_limit=rate_limit,
            )
        )
    raise NotImplementedError("Process mode is not implemented yet")


def _load_cache_files(paths: list[str]) -> dict[str, Any]:
    """Load and merge one or more JSON cache files."""
    merged: dict[str, Any] = {}
    for path in paths:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            logger.warning(f"Image search cache file not found: {path}")
            continue
        with open(path, "r") as f:
            merged.update(json.load(f))
    logger.info(f"Loaded image search cache with {len(merged)} entries from {len(paths)} file(s)")
    return merged


class ImageSearchTool(BaseTool):
    """Reverse-image search tool returning visually matched web results.

    The tool is **parameterless** at call time -- it operates on the image
    stored during ``create()``.  Results are returned as text (titles) and
    thumbnail images.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 20)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_image_search_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
        )

        # Cache backend (default)
        cache_paths = config.get("cache_json_paths", [])
        if isinstance(cache_paths, str):
            cache_paths = [cache_paths]
        self._cache: dict[str, Any] = _load_cache_files(cache_paths) if cache_paths else {}

        # Optional SerpAPI backend
        self._serpapi_key: Optional[str] = config.get("serpapi_key") or os.getenv("SERPAPI_KEY")
        self._max_results: int = config.get("max_results", 5)

        logger.info(
            f"Initialized ImageSearchTool: cache_entries={len(self._cache)}, "
            f"serpapi={'enabled' if self._serpapi_key else 'disabled'}"
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image = kwargs.get("image")
        data_idx = kwargs.get("data_idx")

        self._instance_dict[instance_id] = {
            "image": image,
            "data_idx": data_idx,
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        instance_data = self._instance_dict.get(instance_id)
        if instance_data is None:
            return (
                ToolResponse(text="Error: instance not found. Call create() first."),
                0.0,
                {"success": False},
            )

        data_idx = instance_data.get("data_idx")
        image = instance_data.get("image")

        # Try cache first
        if data_idx and data_idx in self._cache:
            try:
                result_text, result_images = await self.execution_pool.execute.remote(
                    self._search_from_cache, data_idx
                )
                resp_kwargs = {"text": result_text}
                if result_images:
                    resp_kwargs["image"] = result_images
                return (
                    ToolResponse(**resp_kwargs),
                    0.0,
                    {"success": True, "backend": "cache"},
                )
            except Exception as e:
                logger.warning(f"Cache lookup failed for {data_idx}: {e}")

        # Try SerpAPI if available
        if self._serpapi_key and image is not None:
            try:
                result_text, result_images = await self.execution_pool.execute.remote(
                    self._search_via_serpapi, image
                )
                resp_kwargs = {"text": result_text}
                if result_images:
                    resp_kwargs["image"] = result_images
                return (
                    ToolResponse(**resp_kwargs),
                    0.0,
                    {"success": True, "backend": "serpapi"},
                )
            except Exception as e:
                logger.warning(f"SerpAPI search failed: {e}")

        return (
            ToolResponse(text="Image search returned no results."),
            0.0,
            {"success": False, "backend": "none"},
        )

    def _search_from_cache(self, data_idx: str) -> tuple[str, list]:
        """Look up pre-cached image search results."""
        from PIL import Image

        cached = self._cache[data_idx]
        titles = cached.get("tool_returned_web_title", [])
        image_paths = cached.get("cached_images_path", [])

        result_lines = [f"A reverse image search found {len(titles)} results:", "## Web Results"]
        thumbnail_images = []

        for i, (title, img_path) in enumerate(zip(titles, image_paths), 1):
            if img_path and os.path.exists(img_path):
                try:
                    thumb = Image.open(img_path).convert("RGB")
                    thumbnail_images.append(thumb)
                    result_lines.append(f"{i}. [Image] {title}")
                except Exception:
                    result_lines.append(f"{i}. {title}")
            else:
                result_lines.append(f"{i}. {title}")

        return "\n".join(result_lines), thumbnail_images

    def _search_via_serpapi(self, image: Any) -> tuple[str, list]:
        """Live reverse-image search via SerpAPI (Google Lens)."""
        import io
        import base64
        import requests
        from PIL import Image as PILImage

        if isinstance(image, str):
            image_url = image
        else:
            buf = io.BytesIO()
            if hasattr(image, "save"):
                image.save(buf, format="JPEG")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_url = f"data:image/jpeg;base64,{b64}"

        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "engine": "google_lens",
                "url": image_url,
                "api_key": self._serpapi_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        visual_matches = data.get("visual_matches", [])[:self._max_results]
        result_lines = [
            f"A reverse image search found {len(visual_matches)} results:",
            "## Web Results",
        ]
        thumbnail_images = []

        for i, match in enumerate(visual_matches, 1):
            title = match.get("title", "No title")
            thumb_url = match.get("thumbnail")
            result_lines.append(f"{i}. {title}")
            if thumb_url:
                try:
                    img_resp = requests.get(thumb_url, timeout=10)
                    img_resp.raise_for_status()
                    thumb = PILImage.open(io.BytesIO(img_resp.content)).convert("RGB")
                    thumbnail_images.append(thumb)
                except Exception:
                    pass

        return "\n".join(result_lines), thumbnail_images

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


if __name__ == "__main__":
    import asyncio
    import tempfile

    # --- Mocks (same pattern as code_tool.py) ---
    class MockFunction:
        def __init__(self, name: str):
            self.name = name

    class MockOpenAIFunctionToolSchema:
        def __init__(self, name: str, description: str, parameters: dict):
            self.function = MockFunction(name=name)
            self.description = description
            self.parameters = parameters
            self.name = name

        def dict(self):
            return {
                "name": self.function.name,
                "description": self.description,
                "parameters": self.parameters,
            }

        def model_dump(self, **kwargs):
            return self.dict()

    class MockBaseTool:
        def __init__(self, config: dict, tool_schema=None):
            self.config = config
            self.tool_schema = tool_schema
            self.name = tool_schema.function.name if tool_schema else "MockTool"

    BaseTool = MockBaseTool
    OpenAIFunctionToolSchema = MockOpenAIFunctionToolSchema
    # --- End Mocks ---

    async def main_test():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        schema = MockOpenAIFunctionToolSchema(
            name="image_search",
            description="Reverse image search.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        # --- Test 1: Cache hit (titles only, no real images on disk) ---
        print("\n=== Test 1: Cache hit ===")
        cache_data = {
            "idx_0": {
                "tool_returned_web_title": [
                    "Eiffel Tower - Wikipedia",
                    "Paris landmarks guide",
                    "Famous monuments of France",
                ],
                "cached_images_path": [
                    "/nonexistent/a.jpg",
                    "/nonexistent/b.jpg",
                    "/nonexistent/c.jpg",
                ],
            }
        }
        cache_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(cache_data, cache_file)
        cache_file.close()

        tool = ImageSearchTool(
            config={
                "num_workers": 2,
                "enable_global_rate_limit": False,
                "cache_json_paths": [cache_file.name],
            },
            tool_schema=schema,
        )

        iid, create_resp = await tool.create(create_kwargs={"data_idx": "idx_0"})
        print(f"  Created instance: {iid}")
        assert isinstance(iid, str), "instance_id should be a string"

        resp, reward, metrics = await tool.execute(iid, {})
        print(f"  Response text: {resp.text}")
        print(f"  Reward: {reward}, Metrics: {metrics}")
        assert resp.text is not None, "Response text should not be None"
        assert "Eiffel Tower" in resp.text, "Cache result should contain title"
        assert resp.image is None, "No images on disk, so image should be None"
        assert metrics["success"] is True
        assert metrics["backend"] == "cache"
        await tool.release(iid)
        os.unlink(cache_file.name)
        print("  PASSED")

        # --- Test 2: Cache miss, no SerpAPI → fallback ---
        print("\n=== Test 2: Cache miss (no SerpAPI) ===")
        tool2 = ImageSearchTool(
            config={
                "num_workers": 2,
                "enable_global_rate_limit": False,
                "cache_json_paths": [],
            },
            tool_schema=schema,
        )

        iid2, _ = await tool2.create(create_kwargs={"data_idx": "nonexistent"})
        resp2, reward2, metrics2 = await tool2.execute(iid2, {})
        print(f"  Response text: {resp2.text}")
        print(f"  Metrics: {metrics2}")
        assert "no results" in resp2.text.lower(), "Should indicate no results"
        assert metrics2["success"] is False
        await tool2.release(iid2)
        print("  PASSED")

        # --- Test 3: Unknown instance_id ---
        print("\n=== Test 3: Unknown instance_id ===")
        resp3, reward3, metrics3 = await tool2.execute("bogus-id", {})
        print(f"  Response text: {resp3.text}")
        assert "instance not found" in resp3.text.lower()
        assert metrics3["success"] is False
        print("  PASSED")

        # --- Test 4: ToolResponse construction (regression for validator) ---
        print("\n=== Test 4: ToolResponse construction ===")
        r1 = ToolResponse(text="hello")
        assert r1.image is None
        r2 = ToolResponse(text="ok", image=[1, 2, 3])
        assert r2.image == [1, 2, 3]
        print("  PASSED")

        print("\n=== All tests passed! ===")

    ray.init(ignore_reinit_error=True)
    asyncio.run(main_test())
    ray.shutdown()
