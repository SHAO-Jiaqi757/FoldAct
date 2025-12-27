# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return ray.remote(SearchExecutionWorker).options(max_concurrency=num_workers).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SearchTool(BaseTool):
    """Search tool for retrieving information using external retrieval services.

    This tool provides search functionality with rate limiting and concurrent execution
    support through Ray. It integrates with external retrieval services to perform
    semantic search operations.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.retriever_num_workers
        self.rate_limit = config.retriever_rate_limit
        self.timeout = config.retriever_timeout

        self.enable_global_rate_limit = config.retriever_enable_global_rate_limit
        self.execution_pool = init_search_execution_pool(num_workers=self.num_workers, enable_global_rate_limit=self.enable_global_rate_limit, rate_limit=self.rate_limit, mode=PoolMode.ThreadMode)

        # Retrieval service configuration
        self.retrieval_service_url = config.search_url
        assert self.retrieval_service_url, "Configuration must include 'retriever.url'"
        self.topk = config.topk
        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")

        # Safety/validation thresholds (with sane defaults)
        # Skip triggering searches when queries are clearly malformed or too long
        self.max_query_words = getattr(config, "retriever_max_query_words", 48)

        logger.info(f"Initialized SearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        """Execute search operation using retrieval service.

        Args:
            instance_id: Tool instance ID
            query_list: List of search queries
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list") or parameters.get("query")
        
        # Handle single string query (convert to list)
        if isinstance(query_list_from_params, str):
            query_list_from_params = [query_list_from_params]

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' (or 'query') is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return json.dumps({"result": error_msg}), 0.0, {}

        # Validation: do not trigger search if queries are too long or too many
        def _too_long(q: str) -> bool:
            if not isinstance(q, str):
                return True
            if len(q.split()) > self.max_query_words:
                return True
            return False

        # Validate per-query instead of skipping the whole batch.
        valid_mask = [not _too_long(q) for q in query_list_from_params]
        any_valid = any(valid_mask)
        all_valid = all(valid_mask) if query_list_from_params else False

        if not any_valid:
            # All queries invalid: preserve previous behavior but be explicit and aligned
            placeholders = [f"[Search skipped] Query too long: '{str(q)[:128]}'" for q in query_list_from_params]
            logger.info("[SearchTool] All queries skipped due to validation")
            return json.dumps({"result": placeholders}), 0.0, {"skipped": True, "reason": {"all_queries_too_long": True}}

        # Execute search using Ray execution pool
        try:
            # If some queries are invalid, execute the search only for valid ones
            if all_valid:
                # Fast path: execute with full list
                result_text, metadata = await self.execution_pool.execute.remote(
                    self.execute_search, instance_id, query_list_from_params, self.retrieval_service_url, self.topk, timeout
                )
                # Store results in instance dictionary
                self._instance_dict[instance_id]["reward"].append(result_text.strip())
                # Convert metadata to metrics
                metrics = {
                    "query_count": metadata.get("query_count", 0),
                    "status": metadata.get("status", "unknown"),
                    "total_results": metadata.get("total_results", 0),
                    "api_request_error": metadata.get("api_request_error"),
                    "partially_skipped": False,
                }
                return result_text, 0.0, metrics

            # Partial valid: run on the subset, then merge back with placeholders
            valid_queries = [q for q, ok in zip(query_list_from_params, valid_mask) if ok]
            result_text_valid, metadata = await self.execution_pool.execute.remote(
                self.execute_search, instance_id, valid_queries, self.retrieval_service_url, self.topk, timeout
            )
            # Store subset results
            self._instance_dict[instance_id]["reward"].append(result_text_valid.strip())

            # Parse valid results into a list
            try:
                parsed = json.loads(result_text_valid)
                result_obj = parsed.get("result", parsed)
                if isinstance(result_obj, list):
                    valid_results = result_obj
                else:
                    valid_results = [str(result_obj)]
            except Exception:
                valid_results = [result_text_valid]

            # Reconstruct full results aligned with the original input order
            full_results = []
            it = iter(valid_results)
            for q, ok in zip(query_list_from_params, valid_mask):
                if ok:
                    full_results.append(next(it, "[No results returned]"))
                else:
                    full_results.append(f"[Search skipped] Query too long: '{str(q)[:128]}'")

            metrics = {
                "query_count": len(query_list_from_params),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
                "partially_skipped": True,
                "skipped_indices": [i for i, ok in enumerate(valid_mask) if not ok],
            }
            return json.dumps({"result": full_results}), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
