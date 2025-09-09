import torch
from transformers import AutoTokenizer
from verl.tools.schemas import TrajectoryComponent
from search_r1.llm_agent.generation import ResponseType
from typing import Union, Tuple

def _split_by_group(t: torch.Tensor, group: torch.Tensor, with_group_id: bool = False) \
    -> Union[list[torch.Tensor], list[Tuple[int, torch.Tensor]]]:
    """
    Split tensor according to the group that each item belongs to.
    e.g. t = [0, 1, 2, 3, 4], group = [0, 0, 1, 2, 2] => [[0, 1], [2], [3, 4]]
    """

    # find boundaries where group id changes
    boundaries = torch.nonzero(group[1:] != group[:-1], as_tuple=False).flatten() + 1
    # add start=0 and end=len(ids)
    boundaries = torch.cat([torch.tensor([0]), boundaries, torch.tensor([len(group)])])

    if with_group_id:
        return [(group[boundaries[i]].item(), t[boundaries[i]:boundaries[i+1]]) for i in range(len(boundaries)-1)]
    else:
        return [t[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]


def split_steps(responses: torch.Tensor, step_ids: torch.Tensor, attention_mask: torch.Tensor) \
    -> Union[list[torch.Tensor], list[list[torch.Tensor]]]:
    """
    Split responses into a list of tensors based on step ids.
    - If inputs are 1D tensors: returns list[Tensor]
    - If inputs are 2D tensors: returns list[list[Tensor]]
    """
    assert responses.shape == step_ids.shape, \
        f"First two inputs must have the same shape, got {responses.shape}, {step_ids.shape}"
    assert responses.dim() in (1, 2), "Inputs must be 1D or 2D tensors"

    def process_single(resp: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor):
        resp_mask = mask[-len(resp):].bool()
        resp = resp[resp_mask]
        ids = ids[resp_mask]
        return _split_by_group(resp, ids)

    if responses.dim() == 1:
        return process_single(responses, step_ids, attention_mask)
    else:
        return [process_single(r, i, m) for r, i, m in zip(responses, step_ids, attention_mask)]

def get_components(
        responses: torch.Tensor,
        step_ids: torch.Tensor,
        responses_types: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer: AutoTokenizer) \
    -> Union[list[TrajectoryComponent], list[list[TrajectoryComponent]]]:
    """
    Get trajectory components of a response by cutting at each step and each response type.
    - If inputs are 1D tensors: returns list[TrajectoryComponent]
    - If inputs are 2D tensors: returns list[list[TrajectoryComponents]]
    """
    assert responses.shape == responses_types.shape == step_ids.shape, \
        f"First three inputs must have the same shape, got {responses.shape}, {responses_types.shape}, {step_ids.shape}"
    assert responses.dim() in (1, 2), "Inputs must be 1D or 2D tensors"

    steps = split_steps(responses, step_ids, attention_mask)
    step_types = split_steps(responses_types, step_ids, attention_mask)

    def process_single(steps, step_types):
        components = []
        token_offset = 0
        step_number = 0
        for s, t in zip(steps, step_types):
            for ct, tokens in _split_by_group(s, t, with_group_id=True):
                components.append(TrajectoryComponent(
                    component_type=ResponseType(ct).name,
                    content=tokenizer.decode(tokens, skip_special_tokens=True),
                    start_token_idx = token_offset,
                    end_token_idx = token_offset + len(tokens),
                    step_number = step_number,
                ))
                token_offset += len(tokens)
                step_number += 1
        return components
    
    if responses.dim() == 1:
        return process_single(steps, step_types)
    else:
        return [process_single(s, t) for s, t in zip(steps, step_types)]
