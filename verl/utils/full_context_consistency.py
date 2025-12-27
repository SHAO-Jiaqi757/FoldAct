from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.event_ledger import EventLedger
from verl.utils.full_context_builder import FullContextBuilder


def build_full_context_batch_from_ledgers(
    batch: DataProto,
    event_ledgers_array,
    indices: List[int],
    builder: FullContextBuilder,
    pad_token_id: int,
    tokenizer,
    prompts_tensor: Optional[torch.Tensor] = None,
) -> Optional[DataProto]:
    """
    Reconstruct a batch containing full-context input_ids/attention_mask/position_ids/responses
    for the provided sample indices using stored event ledgers.
    """
    if builder is None or not indices:
        return None
    required_keys = {"responses", "input_ids", "attention_mask"}
    batch_keys = set(batch.batch.keys())
    if any(key not in batch_keys for key in required_keys):
        return None

    responses = batch.batch["responses"].cpu()
    input_ids = batch.batch["input_ids"].cpu()
    attention_mask = batch.batch["attention_mask"].cpu()
    try:
        step_ids_tensor = batch.batch["step_ids"].cpu()
    except KeyError:
        step_ids_tensor = None

    full_inputs, full_masks, full_positions, resp_list = [], [], [], []
    for idx in indices:
        try:
            ledger_entry = event_ledgers_array[idx]
            if isinstance(ledger_entry, np.ndarray):
                ledger_entry = ledger_entry.item()
            ledger = EventLedger.from_dict(ledger_entry) if isinstance(ledger_entry, dict) else ledger_entry
            if ledger is None:
                continue

            compressed_ids = input_ids[idx : idx + 1]
            compressed_mask = attention_mask[idx : idx + 1]
            step_slice = step_ids_tensor[idx : idx + 1] if step_ids_tensor is not None else None
            original_prompt = None
            if prompts_tensor is not None:
                prompt_ids = prompts_tensor[idx]
                prompt_tokens = prompt_ids[prompt_ids != pad_token_id]
                if prompt_tokens.numel() > 0:
                    original_prompt = tokenizer.decode(prompt_tokens.tolist())
            full_ctx_ids, _, _ = builder.reconstruct_full_context(
                compressed_ids,
                compressed_mask,
                ledger,
                step_slice,
                original_prompt=original_prompt,
            )
            ctx_ids = full_ctx_ids.squeeze(0).to(torch.long)
            resp_ids = responses[idx].to(torch.long)
            combined_input = torch.cat([ctx_ids, resp_ids], dim=0)
            full_inputs.append(combined_input)
            full_masks.append(torch.ones_like(combined_input, dtype=torch.long))
            full_positions.append(torch.arange(combined_input.size(0), dtype=torch.long))
            resp_list.append(resp_ids)
        except Exception:
            continue

    if not full_inputs:
        return None

    input_ids_tensor = pad_sequence(full_inputs, batch_first=True, padding_value=pad_token_id)
    attention_mask_tensor = pad_sequence(full_masks, batch_first=True, padding_value=0)
    position_ids_tensor = pad_sequence(full_positions, batch_first=True, padding_value=0)
    responses_tensor = pad_sequence(resp_list, batch_first=True, padding_value=pad_token_id)

    data = DataProto.from_dict(
        {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "position_ids": position_ids_tensor,
            "responses": responses_tensor,
        }
    )
    data.meta_info = {
        "temperature": 1.0,
        "micro_batch_size": len(full_inputs),
        "use_dynamic_bsz": False,
        "max_token_len": input_ids_tensor.size(1),
    }
    return data
