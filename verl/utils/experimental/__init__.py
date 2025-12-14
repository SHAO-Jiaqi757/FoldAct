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

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

__all__ = ["collect_response_texts"]


def _as_sequence(value) -> Sequence:
    """Convert nested containers (np.ndarray, list, tuple) into python lists."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def collect_response_texts(
    data_proto,
    tokenizer,
    max_samples: int = 100,
    include_all_turns: bool = False,
) -> List[str]:
    """
    Extract decoded response texts from a rollout DataProto.

    This helper is intentionally lightweight: it falls back to decoding the full
    response sequence when the specialized summary tags are missing so that
    experiment logging always has data to aggregate.
    """
    texts: List[str] = []

    def _append_text(text: str) -> bool:
        if text is None:
            return False

        stripped = text.strip()
        if stripped:
            texts.append(stripped)
        return len(texts) >= max_samples

    # Prefer already-decoded responses when available
    non_tensor = getattr(data_proto, "non_tensor_batch", None)
    raw_responses = None
    if non_tensor is not None:
        getter = getattr(non_tensor, "get", None)
        if callable(getter):
            raw_responses = getter("responses_text", None)
        elif isinstance(non_tensor, dict):
            raw_responses = non_tensor.get("responses_text")

    raw_responses = _as_sequence(raw_responses)
    if raw_responses:
        for entry in raw_responses:
            if isinstance(entry, (list, tuple)):
                if entry:
                    if _append_text(str(entry[0])):
                        return texts
                    if include_all_turns:
                        for turn_text in entry[1:]:
                            if _append_text(str(turn_text)):
                                return texts
            else:
                if _append_text(str(entry)):
                    return texts
        return texts

    # Fall back to decoding token tensors
    batch = getattr(data_proto, "batch", None)
    responses = None
    if batch is not None:
        if isinstance(batch, dict):
            responses = batch.get("responses")
        else:
            getter = getattr(batch, "get", None)
            if callable(getter):
                responses = getter("responses")

    if isinstance(responses, torch.Tensor):
        if responses.dim() == 3:
            batch_size, num_turns, _ = responses.shape
            for b in range(batch_size):
                first_turn = responses[b, 0]
                decoded = tokenizer.decode(first_turn.tolist(), skip_special_tokens=False)
                if _append_text(decoded):
                    return texts
                if include_all_turns and num_turns > 1:
                    for t in range(1, num_turns):
                        decoded = tokenizer.decode(responses[b, t].tolist(), skip_special_tokens=False)
                        if _append_text(decoded):
                            return texts
        elif responses.dim() == 2:
            for row in responses:
                decoded = tokenizer.decode(row.tolist(), skip_special_tokens=False)
                if _append_text(decoded):
                    return texts

    return texts
