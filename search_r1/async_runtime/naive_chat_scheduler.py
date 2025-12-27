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

        # Helper to estimate prompt tokens and clamp max tokens to fit max_model_len
        def _estimate_prompt_tokens(msgs: List[Dict[str, str]]) -> int:
            # Simple heuristic: concatenate role/content and tokenize with HF tokenizer.
            # This underestimates vLLM chat-template tokens, so add a small margin below.
            text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs])
            tok = self.tokenizer([text], add_special_tokens=False, return_tensors='pt', padding='longest')
            return int(tok['input_ids'].shape[1])

        max_model_len = int(self.config.max_model_len) if getattr(self.config, 'max_model_len', None) else int(self.config.prompt_length + self.config.response_length)
        # Leave generous headroom for chat template/system tokens
        safety_margin = 192
        # Ensure at least this many tokens are available for generation
        min_gen_reserve = min(256, int(self.config.response_length))

        # Proactively truncate overlong prompts by token budget to avoid vLLM input overflow.
        # Keep the most recent tokens (right-side) to preserve latest context.
        truncated_messages_list: List[List[Dict[str, str]]] = []
        for msgs in messages_list:
            text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs])
            tok = self.tokenizer([text], add_special_tokens=False, return_tensors='pt', padding='longest')
            ids = tok['input_ids'][0]
            allowed_prompt_tokens = max(1, max_model_len - safety_margin - min_gen_reserve)
            if ids.shape[0] > allowed_prompt_tokens:
                trimmed_ids = ids[-allowed_prompt_tokens:]
                trimmed_text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
                truncated_messages_list.append([{ 'role': 'user', 'content': trimmed_text }])
            else:
                truncated_messages_list.append(msgs)
        messages_list = truncated_messages_list

        # Fire off all requests concurrently
        tasks = []
        for i, messages in enumerate(messages_list):
            est_prompt = _estimate_prompt_tokens(messages) + safety_margin
            avail = max(1, max_model_len - est_prompt)
            max_tok = max(1, min(int(kwargs.get('max_completion_tokens', self.config.response_length)), avail))
            req_kwargs = dict(kwargs)
            # Set both fields for compatibility
            req_kwargs['max_completion_tokens'] = max_tok
            req_kwargs['max_tokens'] = max_tok
            tasks.append(
                self.submit_chat_completions(
                    callback=_callback,
                    callback_additional_info={"idx": i},
                    model=self.model_name,
                    messages=messages,
                    **req_kwargs,
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
