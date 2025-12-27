# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import requests
from typing import Any, Tuple, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class VisitTool(BaseTool):
    """Visit tool for accessing webpage content.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # Configuration
        self.visit_url = config.get("visit_url", config.get("search_url", "").replace("/retrieve", "/access"))
        self.timeout = config.get("retriever_timeout", 30)
        
        logger.info(f"Initialized VisitTool with config: {config}")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the visit tool.
        
        Args:
            instance_id: The instance ID
            parameters: Tool parameters containing 'url' or 'urls'
            
        Returns:
            (response_text, reward, metrics)
        """
        urls = parameters.get("url") or parameters.get("urls")
        
        if not urls:
            return json.dumps({"result": "Error: No URL provided"}), 0.0, {}
            
        if isinstance(urls, str):
            urls = [urls]
            
        if not self.visit_url:
             return json.dumps({"result": ["Error: Visit URL not configured"] * len(urls)}), 0.0, {}

        try:
            # Call the access endpoint
            payload = {"urls": urls}
            response = requests.post(self.visit_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result_data = response.json().get("result", [])
            
            # Format results
            formatted_results = []
            for item in result_data:
                if item:
                    # Depending on the format returned by PageAccess
                    content = item.get("contents", "") or item.get("text", "") or str(item)
                    formatted_results.append(content)
                else:
                    formatted_results.append("Page content not available.")
            
            return json.dumps({"result": formatted_results}), 0.0, {"success": True}
            
        except Exception as e:
            logger.warning(f"Visit tool execution failed: {e}")
            return json.dumps({"result": [f"Error visiting page: {str(e)}"] * len(urls)}), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

