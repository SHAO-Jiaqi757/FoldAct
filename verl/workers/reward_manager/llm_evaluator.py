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

"""
LLM Evaluator for Reward Manager
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LLMEvaluator:
    """LLM-based evaluator for information quality and reasoning grounding"""
    
    def __init__(self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.url = os.getenv("OPENAI_URL")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # client is created per-call to bind to the active event loop
        
        # Prompts from llm_judge.py
        self.SEARCH_RESULT_ANALYSIS_PROMPT = """
You are an expert data analyst. Your task is to evaluate the quality of a search result based on the query that produced it.

**Search Query:**
{query}

**Search Result Documents:**
```json
{documents_json}
```

---
**Your Task:**
Analyze the search result's sufficiency.

 **Sufficiency:** Does the result contain enough information to likely answer the user's implicit question in the query? Choose one:
    *   `Sufficient`: The answer seems to be present.
    *   `Insufficient`: The answer is likely not here.

---
**Your Final Output:**
Your response **MUST** be a single, valid JSON object with the attributes `information_quality`.

```json
{{
  "information_quality": "<'Sufficient'/'Insufficient'>",
}}
```
"""

        self.PREMISE_EVALUATION_PROMPT = """
You are a ruthless, evidence-based critical thinking expert. Your task is to evaluate the factual premise of an AI agent's reasoning based *only* on the evidence it had at the time.

**Context: The Agent's Goal (Original Question):**
{question}

**Evidence: The Search Results the Agent Had Access To:**
```json
{search_evidence_json}
```

**Agent's Reasoning Text to Analyze:**
"{reasoning_text}"

---
Task:
1) Extract the atomic factual premises from the step (skip meta/plan-only wording that contains no factual claim).
2) For each premise, find a direct supporting span in the provided evidence. If no exact or near-verbatim support exists, mark that premise as unmatched.
3) Decide the label with STRICT rules:
   - Directly Grounded: ALL atomic premises are supported by explicit evidence spans.
   - Not Grounded: ANY atomic premise lacks a supporting span; OR the step contains only meta/plan text without factual premises.

Additional rules:
- QUESTION anchor alone is NOT sufficient for Directly Grounded; do not require restating the task/intent.
- Superlatives/temporal/quantitative claims (e.g., last/first/only, years, counts) require explicit evidence spans.

---
Return a single JSON object:
{{
  "premise_grounding": "<Directly Grounded|Not Grounded>",
  "anchor_type": "<EVIDENCE|QUESTION|TRACE|NONE>",
  "evidence_citations": [
    {{"premise": "...", "evidence_snippet": "..."}}
  ],
  "unmatched_premises": ["..."],
  "premise_justification": "<brief explanation referencing the citations or explaining unmatched>"
}}
"""
    
    async def _call_llm(self, prompt: str, max_retries: int = 3, delay: float = 1.0) -> Optional[Dict]:
        """Call OpenAI API with retry logic and robust JSON parsing"""
        for attempt in range(max_retries):
            client = None
            try:
                client = AsyncOpenAI(api_key=self.api_key, base_url=self.url)
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                    continue
                    
            except Exception as e:
                logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                continue
            finally:
                try:
                    if client is not None:
                        await client.close()
                except Exception:
                    pass
        
        return None
    
    async def evaluate_information_quality(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate the quality of search results based on the query
        
        Args:
            query: The search query
            documents: List of search result documents
            
        Returns:
            Dict with evaluation results
        """
        try:
            prompt = self.SEARCH_RESULT_ANALYSIS_PROMPT.format(
                query=query,
                documents_json=json.dumps(documents, indent=2)
            )
            
            response = await self._call_llm(prompt)
            
            if response:
                return {
                    "information_quality": response.get("information_quality", "Unspecified"),
                    "evaluation_success": True
                }
            else:
                return {
                    "information_quality": "Unspecified",
                    "evaluation_success": False
                }
                
        except Exception as e:
            logger.error(f"Error evaluating information quality: {e}")
            return {
                "information_quality": "Unspecified",
                "evaluation_success": False
            }
    
    async def evaluate_reasoning_grounding(self, reasoning_text: str, search_evidence: List[Dict], 
                                        question: str) -> Dict[str, Any]:
        """
        Evaluate whether reasoning is grounded in evidence
        
        Args:
            reasoning_text: The reasoning text to evaluate
            search_evidence: List of search result documents as evidence
            question: The original question/goal
            
        Returns:
            Dict with grounding evaluation results
        """
        try:
            prompt = self.PREMISE_EVALUATION_PROMPT.format(
                question=question,
                search_evidence_json=json.dumps(search_evidence, indent=2),
                reasoning_text=reasoning_text
            )
            
            response = await self._call_llm(prompt)
            
            if response:
                return {
                    "premise_grounding": response.get("premise_grounding", "Unspecified"),
                    "anchor_type": response.get("anchor_type", "NONE"),
                    "evidence_citations": response.get("evidence_citations", []),
                    "unmatched_premises": response.get("unmatched_premises", []),
                    "premise_justification": response.get("premise_justification", "Unspecified"),
                    "evaluation_success": True
                }
            else:
                return {
                    "premise_grounding": "Unspecified",
                    "anchor_type": "NONE",
                    "evidence_citations": [],
                    "unmatched_premises": [],
                    "premise_justification": "LLM evaluation failed",
                    "evaluation_success": False
                }
                
        except Exception as e:
            logger.error(f"Error evaluating reasoning grounding: {e}")
            return {
                "premise_grounding": "Unspecified",
                "anchor_type": "NONE",
                "evidence_citations": [],
                "unmatched_premises": [],
                "premise_justification": f"Evaluation error: {str(e)}",
                "evaluation_success": False
            }
    
    def is_information_sufficient(self, evaluation_result: Dict[str, Any]) -> bool:
        """Determine if information is sufficient based on evaluation result"""
        if not evaluation_result.get("evaluation_success", False):
            return False
        
        quality = evaluation_result.get("information_quality", "Unspecified")
        return quality == "Sufficient"
    
    def is_reasoning_grounded(self, evaluation_result: Dict[str, Any]) -> bool:
        """Determine if reasoning is grounded based on evaluation result"""
        if not evaluation_result.get("evaluation_success", False):
            return False
        
        grounding = evaluation_result.get("premise_grounding", "Unspecified")
        return grounding == "Directly Grounded"
    
    async def batch_evaluate(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch evaluate multiple items for efficiency
        
        Args:
            evaluations: List of evaluation requests
            
        Returns:
            List of evaluation results
        """
        tasks = []
        
        for eval_request in evaluations:
            if eval_request["type"] == "information_quality":
                task = self.evaluate_information_quality(
                    eval_request["query"], 
                    eval_request["documents"]
                )
            elif eval_request["type"] == "reasoning_grounding":
                task = self.evaluate_reasoning_grounding(
                    eval_request["reasoning_text"],
                    eval_request["search_evidence"],
                    eval_request["question"]
                )
            else:
                continue
                
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        else:
            return []


# Convenience functions for synchronous usage
def create_evaluator(model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> LLMEvaluator:
    """Create an LLM evaluator instance"""
    return LLMEvaluator(model=model, api_key=api_key)


async def evaluate_single_information_quality(query: str, documents: List[Dict], 
                                           model: str = "gpt-4o-mini", 
                                           api_key: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate information quality for a single search result"""
    evaluator = LLMEvaluator(model=model, api_key=api_key)
    return await evaluator.evaluate_information_quality(query, documents)


async def evaluate_single_reasoning_grounding(reasoning_text: str, search_evidence: List[Dict], 
                                            question: str, 
                                            model: str = "gpt-4o-mini", 
                                            api_key: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate reasoning grounding for a single reasoning step"""
    evaluator = LLMEvaluator(model=model, api_key=api_key)
    return await evaluator.evaluate_reasoning_grounding(reasoning_text, search_evidence, question) 