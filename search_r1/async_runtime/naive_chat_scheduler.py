import asyncio
from typing import Any, Dict, List

import numpy as np
import torch

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler


class NaiveChatCompletionScheduler(ChatCompletionScheduler):
    """
    A minimal async chat scheduler that:
    - Accepts DataProto with either `non_tensor_batch['raw_prompt']` (OpenAI chat format)
      or tokenized `batch['input_ids']`.
    - Converts inputs into OpenAI messages and submits requests concurrently.
    - Returns a DataProto with `responses` token IDs only (generation manager and
      trainer will merge it back into the full batch and compute logprobs later).
    """

    async def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        # Build messages per sample
        messages_list: List[List[Dict[str, str]]] = []

        if "raw_prompt" in prompts.non_tensor_batch:
            # Expect raw_prompt as list of role/content dicts per sample
            raw_prompts = prompts.non_tensor_batch["raw_prompt"]
            if isinstance(raw_prompts, np.ndarray):
                raw_prompts = raw_prompts.tolist()
            for conv in raw_prompts:
                # Ensure it's a list of dicts
                messages_list.append(list(conv))
        else:
            # Decode tokenized inputs into a single user message
            input_ids = prompts.batch.get("input_ids", None)
            assert input_ids is not None, "prompts must contain either raw_prompt or input_ids"
            if isinstance(input_ids, torch.Tensor):
                decoded = [self.tokenizer.decode(x, skip_special_tokens=True) for x in input_ids]
            else:
                # Fallback for non-tensor inputs
                decoded = [self.tokenizer.decode(torch.as_tensor(x), skip_special_tokens=True) for x in input_ids]
            for text in decoded:
                messages_list.append([{"role": "user", "content": text}])

        # Default sampling params from rollout config
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        # Allow overrides
        kwargs.update(sampling_params)

        responses: List[str] = [None] * len(messages_list)

        async def _callback(completions, info: Dict[str, Any], exception: Exception):
            idx = info["idx"]
            if exception is not None:
                # On failure, return empty string to keep shapes consistent
                responses[idx] = ""
                return
            # Non-stream response; take first choice content
            try:
                content = completions.choices[0].message.content or ""
            except Exception:
                content = ""
            responses[idx] = content

        # Fire off all requests concurrently
        tasks = []
        for i, messages in enumerate(messages_list):
            tasks.append(
                self.submit_chat_completions(
                    callback=_callback,
                    callback_additional_info={"idx": i},
                    model=self.model_name,
                    messages=messages,
                    **kwargs,
                )
            )

        await asyncio.gather(*tasks)

        # Tokenize responses into tensor
        tok = self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )
        resp_ids: torch.Tensor = tok["input_ids"]

        return DataProto.from_dict({"responses": resp_ids})

