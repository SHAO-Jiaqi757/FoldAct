# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import random
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

from verl import DataProto
from verl.protocol import DataProtoConfig
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch
from verl.utils.full_context_builder import FullContextBuilder
from verl.utils.full_context_consistency import build_full_context_batch_from_ledgers
from verl.utils.experimental import collect_response_texts

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"


logger = logging.getLogger(__name__)

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        nodes=ray.nodes()
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else  data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.
    
    IMPORTANT: 
    1. Uses 'responses_types' if available to identify all assistant tokens across all turns
       (think=0, search=1, answer=2, information_summary=4, think_summary=5) 
       while excluding information blocks (information=3, which is raw environment data)
    2. Falls back to 'info_mask' if responses_types is not available
    3. Falls back to 'attention_mask' as last resort
    
    This ensures that:
    - All assistant-generated tokens (across all turns) are included
    - Raw information blocks from environment are excluded
    - information_summary is included (it's generated by assistant, not environment)
    - Multi-turn training works correctly

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens (1 for valid tokens, 0 for masked tokens).
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    
    # Strategy 1: Use responses_types to identify all assistant tokens (preferred for multi-turn)
    if 'responses_types' in data.batch:
        responses_types = data.batch['responses_types']
        
        # Ensure shape matches
        if responses_types.shape != responses.shape:
            logger.warning(
                f"[compute_response_mask] responses_types shape {responses_types.shape} != responses shape {responses.shape}. "
                f"Attempting to align shapes..."
            )
            # Try to align shapes: truncate or pad if needed
            if responses_types.size(1) > responses.size(1):
                responses_types = responses_types[:, -responses.size(1):]
                logger.debug(f"[compute_response_mask] Truncated responses_types to match responses length")
            elif responses_types.size(1) < responses.size(1):
                pad_size = responses.size(1) - responses_types.size(1)
                responses_types = torch.nn.functional.pad(responses_types, (0, pad_size), value=0)
                logger.debug(f"[compute_response_mask] Padded responses_types to match responses length")
        
        # ResponseType values:
        # think = 0, search = 1, answer = 2 (assistant tokens - include)
        # information = 3 (environment token - exclude)
        # information_summary = 4 (assistant token - include, generated by model to summarize information)
        # think_summary = 5 (assistant token - include)
        
        # Create mask: 1 for assistant tokens (think, search, answer, information_summary, think_summary), 0 for information
        # Assistant tokens are: 0, 1, 2, 4, 5
        # Information tokens are: 3 (only raw information from environment)
        response_mask = (
            (responses_types == 0) |  # think
            (responses_types == 1) |  # search
            (responses_types == 2) |  # answer
            (responses_types == 4) |  # information_summary (assistant-generated summary)
            (responses_types == 5)    # think_summary
        ).long()
        
        # DEBUG: Log mask statistics for multi-turn debugging
        if logger.isEnabledFor(logging.DEBUG):
            num_assistant_tokens = response_mask.sum().item()
            num_total_tokens = response_mask.numel()
            assistant_ratio = num_assistant_tokens / num_total_tokens if num_total_tokens > 0 else 0.0
            print(
                f"[compute_response_mask] Assistant tokens: {num_assistant_tokens}/{num_total_tokens} "
                f"({assistant_ratio*100:.1f}%) from responses_types"
            )
            
            # Check for potential issues: if ratio is very low, might indicate missing turns
            if assistant_ratio < 0.1 and num_total_tokens > 100:
                logger.warning(
                    f"[compute_response_mask] Very low assistant token ratio ({assistant_ratio*100:.1f}%). "
                    f"This may indicate that only the last turn's tokens are marked. "
                    f"Check responses_types generation logic."
                )
        
        # Also need to mask out padding tokens using attention_mask
        # CRITICAL FIX: Ensure attention_mask_response matches responses structure
        # responses contains all turns: [turn0, turn1, ..., turnN]
        # attention_mask_full structure: [prompt, turn0, turn1, ..., turnN]
        # So we need to extract the response portion correctly
        if 'attention_mask' in data.batch:
            attention_mask_full = data.batch['attention_mask']
            # Check if attention_mask includes prompt (length > response_length)
            # or if it's already just responses (length == response_length)
            if attention_mask_full.shape[1] > response_length:
                # attention_mask includes prompt, extract response portion
                attention_mask_response = attention_mask_full[:, -response_length:]
            elif attention_mask_full.shape[1] == response_length:
                # attention_mask is already just responses (shouldn't happen but handle it)
                attention_mask_response = attention_mask_full
            else:
                # attention_mask is shorter than responses, this is an error
                logger.warning(
                    f"attention_mask length ({attention_mask_full.shape[1]}) < response_length ({response_length}). "
                    f"Using responses_types mask only (may include padding tokens)."
                )
                # Use responses_types mask only, but still need to handle padding
                # Create a simple mask based on non-zero tokens in responses
                if 'responses' in data.batch:
                    responses = data.batch['responses']
                    pad_token_id = getattr(data, 'pad_token_id', 0)
                    attention_mask_response = (responses != pad_token_id).long()
                else:
                    attention_mask_response = torch.ones_like(response_mask)
            
            # Mask out padding tokens (where attention_mask == 0)
            # This ensures that padding tokens are excluded even if responses_types marks them
            response_mask = response_mask * attention_mask_response.long()
        else:
            # No attention_mask available, use responses to identify padding
            if 'responses' in data.batch:
                responses = data.batch['responses']
                pad_token_id = getattr(data, 'pad_token_id', 0)
                padding_mask = (responses != pad_token_id).long()
                response_mask = response_mask * padding_mask
        
        return response_mask
    
    # Strategy 2: Use info_mask if available (masks out information blocks)
    elif 'info_mask' in data.batch:
        mask_to_use = data.batch['info_mask']
        mask_name = 'info_mask'
        
        # 验证: 确保 mask 长度足够
        if mask_to_use.size(1) < response_length:
            raise ValueError(
                f"{mask_name} length ({mask_to_use.size(1)}) < response_length ({response_length})"
            )
        
        response_mask = mask_to_use[:, -response_length:]
        
        # 验证: 确保形状匹配
        assert response_mask.shape == responses.shape, \
            f"response_mask shape {response_mask.shape} != responses shape {responses.shape}"
        
        return response_mask
    
    # Strategy 3: Fall back to attention_mask (includes everything, not ideal)
    else:
        mask_to_use = data.batch["attention_mask"]
        mask_name = 'attention_mask'
        
        # 验证: 确保 mask 长度足够
        if mask_to_use.size(1) < response_length:
            raise ValueError(
                f"{mask_name} length ({mask_to_use.size(1)}) < response_length ({response_length})"
            )
        
        response_mask = mask_to_use[:, -response_length:]
        
        # 验证: 确保形状匹配
        assert response_mask.shape == responses.shape, \
            f"response_mask shape {response_mask.shape} != responses shape {responses.shape}"
        
        return response_mask


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:

        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # Synchronize use_summary to actor config for FoldAct
        # If use_summary is enabled (generating summary tokens), we should default to using 
        # separated log_prob computation (use_separated_loss) in the actor, unless explicitly disabled.
        if self.config.get('use_summary', False):
            if 'actor' in self.config.actor_rollout_ref:
                # We check if it is explicitly set to False by user, otherwise default to True
                # Note: config is a DictConfig, so we can check existence
                if self.config.actor_rollout_ref.actor.get('use_separated_loss') is None:
                     print("[RayTrainer] Auto-enabling use_separated_loss because use_summary=True")
                     # Use open_dict to temporarily disable struct mode for setting new keys
                     with open_dict(self.config.actor_rollout_ref.actor):
                         self.config.actor_rollout_ref.actor.use_separated_loss = True

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # CRITICAL FIX: Read from actor config, not top-level config
        # Training script sets: +actor_rollout_ref.actor.use_full_context_supervision=true
        self.use_full_context_supervision = bool(
            self.config.actor_rollout_ref.actor.get("use_full_context_supervision", False) or
            self.config.get("use_full_context_supervision", False)  # Fallback for backward compatibility
        )
        self.full_context_builder = None
        self.full_context_monitor_sample_size = 0
        self.full_context_monitor_interval = 0
        if self.use_full_context_supervision:
            max_ctx_len = int(self.config.data.get("max_prompt_length", 4096)) + int(self.config.data.get("max_response_length", 512))
            self.full_context_builder = FullContextBuilder(tokenizer=self.tokenizer, max_context_length=max_ctx_len)
            interval = int(self.config.trainer.get("full_context_consistency_interval", 100))
            sample_size = int(self.config.trainer.get("full_context_consistency_sample_size", 4))
            self.full_context_monitor_sample_size = sample_size
            self.full_context_monitor_interval = interval
            # Use consistency_loss_weight from actor config to determine if training should be enabled
            # If weight > 0, enable training; if weight = 0, monitoring only
            consistency_loss_weight = float(self.config.actor_rollout_ref.actor.get("consistency_loss_weight", 0.0))
            self.full_context_monitor_apply_grad = (consistency_loss_weight > 0)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO, AdvantageEstimator.GAE], "only GRPO and GAE are tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

        # Initialize experiment logger if enabled
        self.experiment_logger = None
        if self.config.trainer.get('enable_experiment_logging', False):
            try:
                from verl.utils.experiment_logger import ExperimentLogger
                
                self.experiment_logger = ExperimentLogger(
                    log_dir=self.config.trainer.get('experiment_log_dir', 'logs/paper_experiments'),
                    experiment_name=self.config.trainer.get('experiment_name', 'default')
                )
                print(
                    f"[Trainer] Experiment logging enabled: "
                    f"{self.config.trainer.get('experiment_name')} -> "
                    f"{self.config.trainer.get('experiment_log_dir')}"
                )
            except Exception as e:
                print(f"[Trainer] Failed to initialize experiment logger: {e}")
                import traceback
                traceback.print_exc()
                self.experiment_logger = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        """ 
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """    
        import torch

        print(f"\n{'='*80}")
        print(f"[VALIDATION] Starting validation at step {self.global_steps}")
        print(f"{'='*80}\n")

        reward_tensor_lst = []
        data_source_lst = []
        success_rate_dict = {}

        gen_config=GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            retriever_num_workers= self.config.retriever.num_workers ,
            retriever_rate_limit= self.config.retriever.rate_limit ,
            retriever_timeout = self.config.retriever.timeout ,
            retriever_enable_global_rate_limit = self.config.retriever.enable_global_rate_limit,
            use_summary = self.config.use_summary,
            enable_debug_logs = self.config.enable_debug_logs,
            max_model_len = self.config.get("max_model_len", 16384)
        )

        #Agent config preparation
        generation_manager=LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=True
        )

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        if not self.config.do_search:


            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # repeat test batch
                test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}

                # Store original inputs
                input_ids = test_batch.batch["input_ids"]
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                }
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

                # # pad to be divisible by dp_size

                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

                # # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                ################ agent-environment loop ###############
                # test_output_gen_batch = self.traj_collector.multi_turn_loop(
                #                                         gen_batch=test_gen_batch,
                #                                         actor_rollout_wg=self.actor_rollout_wg,
                #                                         envs=self.val_envs,
                #                                         is_train=False,
                #                                         )
                print('validation generation end')
                # Store generated outputs
                # output_ids = test_output_gen_batch.batch["responses"]
                # output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                # sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                result = self.val_reward_fn(test_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

                # success rate
                for k in test_batch.non_tensor_batch.keys():
                    if 'success_rate' in k:
                        if k not in success_rate_dict:
                            success_rate_dict[k] = []
                        success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                        # all success_rate should be the same
                        for i in range(1, len(test_batch.non_tensor_batch[k])):
                            assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                # success rate
                for k in test_batch.non_tensor_batch.keys():
                    if 'success_rate' in k:
                        if k not in success_rate_dict:
                            success_rate_dict[k] = []
                        success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                        # all success_rate should be the same
                        for i in range(1, len(test_batch.non_tensor_batch[k])):
                            assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # 确保所有reward_tensor的维度一致
        if len(reward_tensor_lst) > 0:
            # 获取所有张量的形状
            shapes = [tensor.shape for tensor in reward_tensor_lst]
            print(f"DEBUG: Reward tensor shapes before processing: {shapes}")
            
            # 找到最小的序列长度
            min_seq_len = min(tensor.shape[1] for tensor in reward_tensor_lst)
            print(f"DEBUG: Using minimum sequence length: {min_seq_len}")
            
            # 截断所有张量到相同的序列长度
            processed_tensors = [tensor[:, :min_seq_len] for tensor in reward_tensor_lst]
            
            # 拼接张量
            reward_tensor = torch.cat(processed_tensors, dim=0).sum(-1).cpu()  # (batch_size,)
            print(f"DEBUG: Final reward tensor shape: {reward_tensor.shape}")
        else:
            reward_tensor = torch.tensor([])
            
        # 处理data_source_lst可能为空的情况
        if data_source_lst:
            data_sources = np.concatenate(data_source_lst, axis=0)
        else:
            data_sources = np.array([])
            
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        
        # 确保reward_tensor不为空
        if reward_tensor.numel() > 0 and len(data_sources) > 0:
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_reward:
                    data_source_reward[data_source] = []
                data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v
            
        # Add debug logging to check metrics
        print(f"DEBUG: Validation metrics: {metric_dict}")
        print(f"DEBUG: Validation metrics keys: {list(metric_dict.keys())}")
        print(f"DEBUG: Validation metrics values: {list(metric_dict.values())}")

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator=='gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
            self.use_critic=True

        elif self.config.algorithm.adv_estimator=='grpo':
            self.use_critic=False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            # Provide a default scheduler if not specified
            if OmegaConf.select(self.config.actor_rollout_ref.rollout, "chat_scheduler") in [None, "null", "None", ""]:
                # Use minimal scheduler that decodes tokenized prompts into a single user message
                self.config.actor_rollout_ref.rollout.chat_scheduler = "search_r1.async_runtime.naive_chat_scheduler.NaiveChatCompletionScheduler"
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
                scheduler_kwargs={
                    # Pass the trainer checkpoint root so async servers can hot-reload HF exports
                    "ckpt_dir": self.config.trainer.default_local_dir,
                },
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"[CHECKPOINT] Saving checkpoint at step {self.global_steps}")
        print(f"[CHECKPOINT] local_global_step_folder: {local_global_step_folder}")
        
        # Ensure directory exists
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        print(f"[CHECKPOINT] Saving actor checkpoint to {actor_local_path}")
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        print(f"[CHECKPOINT] Writing latest checkpoint tracker to {local_latest_checkpointed_iteration}")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))
        
        print(f"[CHECKPOINT] Successfully saved checkpoint at step {self.global_steps}")
        # List the checkpoint directory contents to verify
        if os.path.exists(local_global_step_folder):
            print(f"[CHECKPOINT] Directory contents of {local_global_step_folder}:")
            for item in os.listdir(local_global_step_folder):
                print(f"  - {item}")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        if global_step_folder is None:
            return 0
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")

        # Guard against world_size mismatch by checking shard file presence.
        from verl.utils.checkpoint.checkpoint_manager import is_fsdp_ckpt_compatible
        expected_ws = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        if not is_fsdp_ckpt_compatible(actor_path, expected_ws):
            print(f"[CHECKPOINT] Incompatible FSDP checkpoint shards for world_size={expected_ws} under {actor_path}.\n"
                  f"- Expected file: model_world_size_{expected_ws}_rank_0.pt not found.\n"
                  f"- Likely saved with a different world_size. Starting from scratch.")
            self.global_steps = 0
            return 0

        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            if not is_fsdp_ckpt_compatible(critic_path, expected_ws):
                print(f"[CHECKPOINT] Incompatible FSDP checkpoint shards for world_size={expected_ws} under {critic_path}.\n"
                      f"- Expected file: model_world_size_{expected_ws}_rank_0.pt not found.\n"
                      f"- Skipping critic resume.")
            else:
                self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _sample_full_context_indices(self, batch_size: int) -> list[int]:
        sample_size = getattr(self, "full_context_monitor_sample_size", 0)
        if sample_size <= 0 or batch_size <= 0:
            return []
        sample_size = min(sample_size, batch_size)
        return random.sample(range(batch_size), sample_size)

    def _build_full_context_batch(self, batch: DataProto, event_ledgers_np, indices: list[int]) -> Optional[DataProto]:
        if self.full_context_builder is None or not indices:
            return None
        if "responses" not in batch.batch or "input_ids" not in batch.batch or "attention_mask" not in batch.batch:
            return None
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        prompts_tensor = None
        if "prompts" in batch.batch.keys():
            prompts_tensor = batch.batch["prompts"].cpu()
        return build_full_context_batch_from_ledgers(
            batch=batch,
            event_ledgers_array=event_ledgers_np,
            indices=indices,
            builder=self.full_context_builder,
            pad_token_id=pad_id,
            tokenizer=self.tokenizer,
            prompts_tensor=prompts_tensor,
        )

    def _maybe_log_full_context_consistency(self, batch: DataProto):
        # DEBUG: Add comprehensive logging to identify why consistency loss is zero
        # Always log on first call to help debug
        if not hasattr(self, '_consistency_debug_logged'):
            logger.info(f"[ConsistencyDebug] Initial check: use_full_context_supervision={self.use_full_context_supervision}, "
                       f"interval={getattr(self, 'full_context_monitor_interval', 'NOT_SET')}, "
                       f"apply_grad={getattr(self, 'full_context_monitor_apply_grad', 'NOT_SET')}, "
                       f"global_steps={self.global_steps}")
            self._consistency_debug_logged = True
        
        if not self.use_full_context_supervision:
            logger.warning(f"[ConsistencyDebug] Step {self.global_steps}: use_full_context_supervision=False, skipping consistency loss")
            return
        interval = getattr(self, "full_context_monitor_interval", 0)
        if interval <= 0 or (self.global_steps % interval) != 0:
            logger.warning(f"[ConsistencyDebug] Step {self.global_steps}: Interval check failed (interval={interval}, global_steps={self.global_steps}, remainder={self.global_steps % interval if interval > 0 else 'N/A'})")
            return
        if "old_log_probs" not in batch.batch:
            logger.warning(f"[ConsistencyDebug] Step {self.global_steps}: old_log_probs not in batch, skipping consistency loss")
            return
        # CRITICAL: Compute response_mask if not available (needed for consistency check)
        # This allows calling _maybe_log_full_context_consistency before compute_advantage
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        event_ledgers_np = batch.non_tensor_batch.get("event_ledger")
        if event_ledgers_np is None:
            logger.warning(f"[ConsistencyDebug] Step {self.global_steps}: event_ledger not in non_tensor_batch, skipping consistency loss. Available keys: {list(batch.non_tensor_batch.keys())}")
            return
        indices = self._sample_full_context_indices(batch.batch["old_log_probs"].size(0))
        if not indices:
            logger.warning(f"[ConsistencyDebug] Step {self.global_steps}: No indices sampled (sample_size={self.full_context_monitor_sample_size}, batch_size={batch.batch['old_log_probs'].size(0)})")
            return
        full_dp = self._build_full_context_batch(batch, event_ledgers_np, indices)
        if full_dp is None:
            logger.warning(f"[ConsistencyDebug] Step {self.global_steps}: Failed to build full context batch for indices={indices}")
            return
        try:
            # CRITICAL FIX: Ensure temperature matches batch's temperature to avoid union conflict
            # full_dp.meta_info has temperature=1.0, but compute_log_prob may return different temperature
            # Copy temperature from batch to ensure consistency
            if "temperature" in batch.meta_info:
                full_dp.meta_info["temperature"] = batch.meta_info["temperature"]
            
            output = self.actor_rollout_wg.compute_log_prob(full_dp)
            # CRITICAL FIX: Remove temperature from output.meta_info before union to avoid conflict
            # The temperature should come from full_dp (which we just set above)
            if "temperature" in output.meta_info:
                output.meta_info.pop("temperature")
            full_dp = full_dp.union(output)
        except Exception as e:
            logger.error(f"[ConsistencyMonitor] Failed to compute full-context log_probs: {e}")
            import traceback
            logger.error(f"[ConsistencyMonitor] Traceback: {traceback.format_exc()}")
            return

        full_log_probs = full_dp.batch.get("old_log_probs")
        if full_log_probs is None:
            logger.error(
                f"[ConsistencyDebug] Step {self.global_steps}: CRITICAL: full_dp.batch.get('old_log_probs') returned None! "
                "This means compute_log_prob failed or returned empty result. "
                "Cannot inject full_context_log_probs. Consistency loss will be skipped."
            )
            return
        
        # Validate full_log_probs shape matches indices
        if full_log_probs.size(0) != len(indices):
            logger.error(
                f"[ConsistencyDebug] Step {self.global_steps}: CRITICAL: Shape mismatch! "
                f"full_log_probs.shape[0]={full_log_probs.size(0)} but len(indices)={len(indices)}. "
                "Cannot inject full_context_log_probs. Consistency loss will be skipped."
            )
            return
        
        compressed = batch.batch["old_log_probs"][indices].to(full_log_probs.device)
        response_mask = batch.batch["response_mask"][indices].to(full_log_probs.device)
        min_len = min(full_log_probs.size(1), compressed.size(1), response_mask.size(1))
        if min_len <= 0:
            logger.warning(
                f"[ConsistencyDebug] Step {self.global_steps}: min_len={min_len} <= 0, skipping consistency monitoring"
            )
            return
        full_log_probs = full_log_probs[:, :min_len]
        compressed = compressed[:, :min_len]
        mask = response_mask[:, :min_len].float()
        denom = torch.clamp(mask.sum(), min=1.0)
        diff = torch.abs(full_log_probs - compressed)
        mean_diff = (diff * mask).sum() / denom
        mean_value = float(mean_diff.detach().cpu().item())
        if self.experiment_logger:
            self.experiment_logger.log_stability_metrics(
                step=self.global_steps,
                metrics={"consistency_logprob_diff": mean_value},
            )
        # Inject full context batch for training if weight > 0 (monitoring only if weight = 0)
        if self.full_context_monitor_apply_grad:
            # ROBUST SOLUTION: Always inject pre-computed full context log_probs
            # This completely avoids activation offload state issues by never doing extra forward passes during training
            logger.info(
                f"[ConsistencyDebug] Step {self.global_steps}: Injecting full_context data for {len(indices)} samples "
                f"(indices={indices}, full_log_probs.shape={full_log_probs.shape})"
            )
            self._inject_full_context_batch(batch, indices, full_dp, full_log_probs)
            
            # CRITICAL: Verify injection succeeded
            if 'full_context_log_probs' not in batch.batch:
                logger.error(
                    f"[ConsistencyDebug] Step {self.global_steps}: CRITICAL: Failed to inject full_context_log_probs! "
                    "This will cause activation offload state corruption if dp_actor tries to compute it. "
                    f"Available keys: {list(batch.batch.keys())}"
                )
            else:
                injected_log_probs = batch.batch['full_context_log_probs']
                valid_rows = (injected_log_probs.abs().sum(dim=1) > 0).sum().item()
                logger.info(
                    f"[ConsistencyDebug] Step {self.global_steps}: ✓ Successfully injected full_context_log_probs: "
                    f"shape={injected_log_probs.shape}, valid_rows={valid_rows}/{injected_log_probs.size(0)}"
                )
            # DIAGNOSTIC: Verify injection succeeded
            if 'full_context_input_ids' in batch.batch:
                logger.info(f"[ConsistencyDebug] Step {self.global_steps}: ✓ Injected full_context_input_ids: "
                          f"shape={batch.batch['full_context_input_ids'].shape}, "
                          f"indices={indices}, num_selected={len(indices)}")
                # Check if injected data is valid (non-zero)
                valid_rows = batch.batch['full_context_input_ids'].abs().sum(dim=1) > 0
                num_valid = valid_rows.sum().item()
                logger.info(f"[ConsistencyDebug] Step {self.global_steps}: Valid rows in full_context_input_ids: {num_valid}/{batch.batch['full_context_input_ids'].size(0)}")
            else:
                logger.error(f"[ConsistencyDebug] Step {self.global_steps}: ✗ Failed to inject full_context_input_ids! "
                           f"Available keys: {list(batch.batch.keys())}")
        else:
            logger.warning(f"[ConsistencyDebug] Step {self.global_steps}: full_context_monitor_apply_grad=False (consistency_loss_weight might be 0 or not set)")
            
        return {"consistency_logprob_diff": mean_value}

    def _inject_full_context_batch(self, batch: DataProto, indices: list[int], full_dp: DataProto, full_log_probs: torch.Tensor = None):
        """
        Inject full context batch for consistency loss computation.
        
        ELEGANT SOLUTION: Pass pre-computed full context log_probs to avoid extra forward pass.
        This prevents activation offload state issues by not doing forward pass during training.
        
        Args:
            batch: Original batch with compressed context
            indices: Sample indices selected for full context computation
            full_dp: Full context DataProto with input_ids, attention_mask, etc.
            full_log_probs: Pre-computed log_probs for full context (if None, will compute in dp_actor)
        """
        if "input_ids" not in full_dp.batch:
            return
        
        # Save full context input_ids and attention_mask for selected samples
        full_input_ids = full_dp.batch["input_ids"]  # [num_selected, full_seq_len]
        full_attention_mask = full_dp.batch.get("attention_mask")
        
        # Create tensors matching batch size, filled with zeros (padding)
        batch_size = batch.batch["input_ids"].size(0)
        device = batch.batch["input_ids"].device
        max_full_len = full_input_ids.size(1)
        
        # Initialize with zeros (will be filled for selected indices)
        full_context_input_ids = torch.zeros(
            batch_size, max_full_len,
            dtype=full_input_ids.dtype,
            device=device
        )
        
        if full_attention_mask is not None:
            full_context_attention_mask = torch.zeros(
                batch_size, max_full_len,
                dtype=full_attention_mask.dtype,
                device=device
            )
        else:
            full_context_attention_mask = None
        
        # Fill in the selected samples
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        full_context_input_ids[idx_tensor] = full_input_ids.to(device)
        if full_context_attention_mask is not None:
            full_context_attention_mask[idx_tensor] = full_attention_mask.to(device)
        
        # Save indices for dp_actor to know which samples to process
        batch.batch["full_context_input_ids"] = full_context_input_ids
        if full_attention_mask is not None:
            batch.batch["full_context_attention_mask"] = full_context_attention_mask
        
        # CRITICAL FIX: Create full_context_indices with batch_size shape
        # Use -1 to mark non-selected samples, and original index for selected samples
        full_context_indices = torch.full(
            (batch_size,), -1, dtype=torch.long, device=device
        )
        full_context_indices[idx_tensor] = idx_tensor  # Mark selected indices
        batch.batch["full_context_indices"] = full_context_indices
        
        # ELEGANT SOLUTION: Inject pre-computed full context log_probs to avoid extra forward pass
        # This prevents activation offload state issues
        if full_log_probs is not None:
            # Create tensor matching batch size, filled with zeros (padding)
            full_context_log_probs = torch.zeros(
                batch_size, full_log_probs.size(1),
                dtype=full_log_probs.dtype,
                device=device
            )
            full_context_log_probs[idx_tensor] = full_log_probs.to(device)
            batch.batch["full_context_log_probs"] = full_context_log_probs
            logger.info(f"[ConsistencyDebug] Step {self.global_steps}: Injected pre-computed full_context_log_probs: shape={full_context_log_probs.shape}")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            
            # Add debug logging for Wandb logging
            print(f"DEBUG: Logging metrics to Wandb: {val_metrics}")
            print(f"DEBUG: Logger: {logger}")
            print(f"DEBUG: Logger backends: {logger.logger}")
            
            logger.log(data=val_metrics, step=self.global_steps)
            
            # Add debug logging after Wandb logging
            print(f"DEBUG: Metrics logged to Wandb")
            
            # Log initial task metrics
            if self.experiment_logger:
                try:
                    task_metrics_payload = {
                        'success_rate': float(val_metrics.get('test/success_rate', 0.0) or 0.0),
                        'avg_reward': float(val_metrics.get('test/avg_reward', 0.0) or 0.0),
                        'avg_turns_to_success': float(val_metrics.get('test/avg_turns', 0.0) or 0.0),
                    }
                    self.experiment_logger.log_task_metrics(
                        epoch=0,
                        step=self.global_steps,
                        metrics=task_metrics_payload,
                    )
                except Exception as e:
                    print(f"[Trainer] Error logging initial task metrics: {e}")
            
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        
        # Print training configuration and checkpoint schedule
        print(f"\n{'='*80}")
        print(f"[TRAINING] Starting training from step {self.global_steps}")
        print(f"[TRAINING] Total training steps: {self.total_training_steps}")
        print(f"[TRAINING] Checkpoint directory: {self.config.trainer.default_local_dir}")
        print(f"[TRAINING] Save frequency: Every {self.config.trainer.save_freq} steps")
        print(f"[TRAINING] Expected checkpoints at steps: {', '.join(str(i) for i in range(self.config.trainer.save_freq, self.total_training_steps + 1, self.config.trainer.save_freq))}")
        print(f"[TRAINING] Test frequency: Every {self.config.trainer.test_freq} steps")
        print(f"{'='*80}\n")

        gen_config=GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,  
            retriever_num_workers= self.config.retriever.num_workers ,
            retriever_rate_limit= self.config.retriever.rate_limit ,
            retriever_timeout = self.config.retriever.timeout ,
            retriever_enable_global_rate_limit = self.config.retriever.enable_global_rate_limit,
            use_summary = self.config.use_summary,
            enable_debug_logs = self.config.enable_debug_logs,
            max_model_len = self.config.get("max_model_len", 16384)
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            async_rollout_manager=self.async_rollout_manager if getattr(self, "async_rollout_mode", False) else None,
            use_async=getattr(self, "async_rollout_mode", False),
        )        
        # Start training Loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch=batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent,interleave=True)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                print(f"\n{'*'*40}")
                print(f"[TRAINING] Processing step {self.global_steps}/{self.total_training_steps} (Epoch {epoch})")
                print(f"{'*'*40}")

                with _timer("step", timing_raw):
                    generation_for_logging = None
                    final_gen_batch_output = None
                    # generate a batch
                    if not self.config.do_search:
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            # Ensure engine is active for latest weights/cache
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                        generation_for_logging = gen_batch_output

                        # Assign unique IDs per original prompt; also set traj_uid for compatibility
                        batch.non_tensor_batch['uid'] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        # For single-turn scenarios, treat each prompt as its own trajectory
                        batch.non_tensor_batch['traj_uid'] = batch.non_tensor_batch['uid'].copy()
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)                    
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer("gen", timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                            # Filter out sequences that are too long to avoid OOM/AssertionError in log_prob computation
                            # This handles the case where max_token_len is fixed and cannot be increased
                            _max_len_cfg = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                            if _max_len_cfg is not None:
                                try:
                                    _max_len = int(_max_len_cfg)
                                    _attention_mask = final_gen_batch_output.batch['attention_mask']
                                    _seq_lens = _attention_mask.sum(dim=1)
                                    _valid_mask = _seq_lens <= _max_len
                                    
                                    if not _valid_mask.all():
                                        _num_dropped = (~_valid_mask).sum().item()
                                        _total = len(_seq_lens)
                                        print(f"[RayTrainer] Length Filtering: Dropping {_num_dropped}/{_total} sequences exceeding max_token_len {_max_len}. Max len found: {_seq_lens.max().item()}")
                                        
                                        if _num_dropped == _total:
                                            print("[RayTrainer] All sequences dropped due to length limit! Skipping batch.")
                                            continue
                                            
                                        # Filter DataProto using boolean mask
                                        final_gen_batch_output = final_gen_batch_output[_valid_mask]
                                except Exception as e:
                                    print(f"[RayTrainer] Warning: Failed to apply length filtering: {e}")

                            # ==================== CRITICAL DIAGNOSTIC FOR EMPTY RESPONSES ====================
                            if 'responses' in final_gen_batch_output.batch:
                                resp = final_gen_batch_output.batch['responses']
                                # Handle empty response tensor (shape [batch_size, 0])
                                if resp.dim() == 2 and resp.shape[1] == 0:
                                    logger.error(f"[FATAL] Empty responses generated in run_llm_loop! Shape: {resp.shape}")
                                    logger.error("Skipping this batch due to empty responses.")
                                    continue
                                
                                # Handle dummy EOS-only responses (length 1) - Filter them out
                                if resp.dim() == 2 and resp.shape[1] == 1:
                                    # Check if all tokens are EOS/PAD
                                    eos_id = self.tokenizer.eos_token_id
                                    pad_id = self.tokenizer.pad_token_id
                                    is_dummy = torch.logical_or(resp == eos_id, resp == pad_id).all(dim=1)
                                    
                                    if is_dummy.any():
                                        num_dummy = is_dummy.sum().item()
                                        total = resp.shape[0]
                                        logger.warning(f"[RayTrainer] Detected {num_dummy}/{total} dummy responses (length 1, EOS/PAD). Filtering...")
                                        
                                        if num_dummy == total:
                                            logger.error("All responses are dummy/empty. Skipping batch.")
                                            continue
                                            
                                        # Filter out dummy responses
                                        valid_mask = ~is_dummy
                                        final_gen_batch_output = final_gen_batch_output[valid_mask]
                                        
                                        # Also filter the original batch (prompts etc) to match
                                        # But wait, 'gen_batch' was already consumed by run_llm_loop.
                                        # 'batch' variable holds the full data. We need to filter 'batch' too?
                                        # Actually, final_gen_batch_output contains everything we need for the next steps
                                        # because run_llm_loop returns a union of input and output.
                                        # BUT, ray_trainer logic below does: batch = batch.union(final_gen_batch_output)
                                        # So we MUST filter 'batch' (the original prompts) to match valid_mask
                                        
                                        # HACK: Re-align batch with valid_mask
                                        # We need to filter the 'gen_batch' or 'batch' that corresponds to these responses.
                                        # Since we are inside the 'else' block of async_rollout_mode check, 
                                        # 'gen_batch' was passed to run_llm_loop.
                                        # But 'batch' is what we union with.
                                        
                                        # We need to filter 'batch' using valid_mask
                                        if isinstance(batch, DataProto):
                                            batch = batch[valid_mask]
                                        
                            # =================================================================================

                            # HYPOTHESIS VERIFICATION: Check the fingerprint after serialization
                            # Use standard Python logging to avoid conflict with Tracking logger
                            import logging
                            verification_logger = logging.getLogger(__name__)
                            
                            if 'mask_sums_fingerprint' in final_gen_batch_output.non_tensor_batch:
                                verification_logger.critical("[VERIFICATION] 'mask_sums_fingerprint' received. Checking integrity...")
                                original_sums = final_gen_batch_output.non_tensor_batch['mask_sums_fingerprint']
                                received_contexts = final_gen_batch_output.non_tensor_batch.get('per_turn_contexts', [])
                                
                                corruption_found = False
                                for traj_idx, (orig_traj_sums, recv_traj_ctx) in enumerate(zip(original_sums, received_contexts)):
                                    for turn_idx, (orig_sum, recv_turn) in enumerate(zip(orig_traj_sums, recv_traj_ctx)):
                                        if 'attention_mask' in recv_turn:
                                            recv_sum = recv_turn['attention_mask'].sum().item()
                                            if orig_sum != recv_sum:
                                                verification_logger.error(
                                                    f"[VERIFICATION FAILED] CORRUPTION DETECTED in Traj {traj_idx}, Turn {turn_idx}: "
                                                    f"Original mask sum was {orig_sum}, but received sum is {recv_sum}."
                                                )
                                                corruption_found = True
                                        else:
                                            verification_logger.warning(f"[VERIFICATION] Missing 'attention_mask' in received turn {turn_idx} of trajectory {traj_idx}.")
                                
                                if not corruption_found:
                                    verification_logger.critical("[VERIFICATION PASSED] All attention_mask sums match. No corruption detected during serialization.")
                                else:
                                    # Give a clear overall signal that the problem is confirmed
                                    verification_logger.error("[VERIFICATION CONCLUSION] Serialization corruption is CONFIRMED as the root cause.")

                            for key in final_gen_batch_output.batch.keys():
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                            with torch.no_grad():
                                # Per-turn training is automatically enabled when use_summary=True
                                # because per-turn contexts are saved during generation
                                use_summary = self.config.get("use_summary", False)
                                print(f"[RayTrainer] use_summary: {use_summary}")
                                if use_summary:
                                    try:
                                        from verl.utils.per_turn_training import PerTurnContextManager
                                        def _orig_compute(dp: DataProto):
                                            # Allow DP auto-padding so world_size chunking succeeds
                                            dp.meta_info[DataProtoConfig.auto_padding_key] = True
                                            return self.actor_rollout_wg.compute_log_prob(dp)
                                        
                                        # SELECTIVE PER-TURN TRAINING: Randomly sample turns with dropout probability
                                        # This reduces computational overhead while maintaining accuracy
                                        # When dropout is enabled, turns with mask=False are skipped in per-turn processing
                                        # and will use regular context computation (fallback behavior)
                                        per_turn_dropout_prob = self.config.get("per_turn_dropout_prob", 0.0)
                                        
                                        per_turn_mask = None
                                        if per_turn_dropout_prob > 0.0:
                                            # Extract per-turn contexts to determine structure
                                            per_turn_contexts_batch = PerTurnContextManager.extract_per_turn_contexts(final_gen_batch_output)
                                            if per_turn_contexts_batch is not None:
                                                # Generate random mask: True = use per-turn context, False = skip (use regular context)
                                                # dropout_prob is the probability of NOT using per-turn context (i.e., probability of dropping)
                                                import random
                                                per_turn_mask = []
                                                for traj in per_turn_contexts_batch:
                                                    # For each turn, randomly decide: True if random > dropout_prob (use per-turn), False otherwise (drop)
                                                    traj_mask = [random.random() > per_turn_dropout_prob for _ in traj]
                                                    per_turn_mask.append(traj_mask)
                                        
                                            # Use per-turn with optional random dropout mask
                                            log_probs = PerTurnContextManager.compute_per_turn_log_probs_batched(
                                                final_gen_batch_output, _orig_compute, 
                                                use_per_turn_context=True,
                                                per_turn_mask=per_turn_mask
                                            )
                                        else:
                                            # Use per-turn for all turns (original behavior)
                                            log_probs = PerTurnContextManager.compute_per_turn_log_probs_batched(
                                                final_gen_batch_output, _orig_compute, 
                                                use_per_turn_context=True 
                                            )
                                        
                                        # Ensure DataProto shape compatibility
                                        # CRITICAL FIX: Remove old_log_probs if it exists to avoid union conflict
                                        if 'old_log_probs' in final_gen_batch_output.batch:
                                            print(f"[Per-Turn Training] Removing existing old_log_probs before setting new one")
                                            del final_gen_batch_output.batch['old_log_probs']
                                        
                                        # CRITICAL FIX: Align old_log_prob length with responses length
                                        # Per-turn training computes old_log_prob for all turns (may be longer than max_response_length)
                                        # But responses are truncated to max_response_length, so we need to truncate old_log_prob too
                                        responses_shape = final_gen_batch_output.batch.get('responses', torch.empty(0)).shape
                                        if len(responses_shape) == 2:
                                            max_response_length = responses_shape[1]
                                            if log_probs.shape[1] > max_response_length:
                                                print(f"[Per-Turn Training] Truncating old_log_prob from {log_probs.shape[1]} to {max_response_length} "
                                                      f"to match responses length")
                                                log_probs = log_probs[:, :max_response_length]
                                                # Also update valid_mask if it exists
                                                if hasattr(log_probs, 'valid_mask'):
                                                    log_probs.valid_mask = log_probs.valid_mask[:, :max_response_length]
                                            elif log_probs.shape[1] < max_response_length:
                                                print(f"[Per-Turn Training] Padding old_log_prob from {log_probs.shape[1]} to {max_response_length} "
                                                      f"to match responses length")
                                                pad_size = max_response_length - log_probs.shape[1]
                                                padding = torch.zeros(log_probs.shape[0], pad_size, 
                                                                      dtype=log_probs.dtype, device=log_probs.device)
                                                log_probs = torch.cat([log_probs, padding], dim=1)
                                                # Update valid_mask if it exists
                                                if hasattr(log_probs, 'valid_mask'):
                                                    pad_mask = torch.zeros(log_probs.shape[0], pad_size, 
                                                                          dtype=log_probs.valid_mask.dtype, device=log_probs.device)
                                                    log_probs.valid_mask = torch.cat([log_probs.valid_mask, pad_mask], dim=1)
                                        
                                        final_gen_batch_output.batch['old_log_probs'] = log_probs
                                        
                                        # CRITICAL FIX: Store valid_mask if available for use in loss calculation
                                        if hasattr(log_probs, 'valid_mask'):
                                            valid_mask = log_probs.valid_mask
                                            # Store valid_mask both in meta_info and batch for downstream access
                                            final_gen_batch_output.meta_info['per_turn_valid_mask'] = valid_mask
                                            final_gen_batch_output.batch['per_turn_valid_mask'] = valid_mask
                                            valid_ratio = valid_mask.sum().item() / max(valid_mask.numel(), 1)
                                            print(
                                                "[Per-Turn Training] Stored valid_mask in batch/meta_info: "
                                                "shape=%s valid_ratio=%.4f",
                                                tuple(valid_mask.shape),
                                                valid_ratio,
                                            )
                                    except Exception as e:
                                        print(f"[Per-Turn Training] Fallback to original compute_log_prob due to: {e}")
                                        print(f"[Per-Turn Training] Exception type: {type(e).__name__}")
                                        import traceback
                                        print(f"[Per-Turn Training] Full traceback:")
                                        traceback.print_exc()
                                        # CRITICAL FIX: Remove old_log_probs if it exists to avoid union conflict
                                        if 'old_log_probs' in final_gen_batch_output.batch:
                                            print(f"[Per-Turn Training] Removing existing old_log_probs before fallback compute_log_prob")
                                            del final_gen_batch_output.batch['old_log_probs']
                                        
                                        output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                                        final_gen_batch_output = final_gen_batch_output.union(output)
                                else:
                                    output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                                    final_gen_batch_output = final_gen_batch_output.union(output)
                            # Use provided per-sample index as group id; mirror to traj_uid
                            batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                            batch.non_tensor_batch['traj_uid'] = batch.non_tensor_batch['uid'].copy()
                                                
                            # repeat to align with repeated responses in rollout
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(final_gen_batch_output)
                            
                            # DIAGNOSTIC: Check if responses_types is preserved after union
                            if 'responses_types' in final_gen_batch_output.batch:
                                print(f"[Trainer] ✓ responses_types found in final_gen_batch_output: "
                                          f"shape={final_gen_batch_output.batch['responses_types'].shape}")
                            else:
                                logger.warning(f"[Trainer] ✗ responses_types NOT found in final_gen_batch_output! "
                                             f"Available keys: {list(final_gen_batch_output.batch.keys())}")
                            
                            if 'responses_types' in batch.batch:
                                print(f"[Trainer] ✓ responses_types preserved after union: "
                                          f"shape={batch.batch['responses_types'].shape}")
                            else:
                                logger.warning(f"[Trainer] ✗ responses_types lost after union! "
                                             f"Available keys: {list(batch.batch.keys())}")

                        # NOTE: _maybe_log_full_context_consistency is now called AFTER compute_advantage
                        # to ensure response_mask exists before injecting full_context_input_ids

                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                    #         gen_batch_output = self.traj_collector.multi_turn_loop(
                    #                                             gen_batch=gen_batch,
                    #                                             actor_rollout_wg=self.actor_rollout_wg,
                    #                                             envs=self.envs,
                    #                                             is_train=True,
                    #                                             )
                    # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    #     with _timer("gen_max", timing_raw):
                    #         gen_baseline_batch = deepcopy(gen_batch)
                    #         gen_baseline_batch.meta_info["do_sample"] = False
                    #         gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                    #         batch = batch.union(gen_baseline_output)
                    #         reward_baseline_tensor = self.reward_fn(batch)
                    #         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    #         batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    #         batch.batch["reward_baselines"] = reward_baseline_tensor

                    #         del gen_baseline_batch, gen_baseline_output

                    # # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # # repeat to align with repeated responses in rollout
                    # # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # # batch = batch.union(gen_batch_output)
                    # del batch
                    # batch = gen_batch_output

                    # batch.batch["response_mask"] = compute_response_mask(batch)
                    # # balance the number of valid tokens on each dp rank.
                    # # Note that this breaks the order of data inside the batch.
                    # # Please take care when you implement group based adv computation such as GRPO and rloo
                    # if self.config.trainer.balance_batch:
                    #     self._balance_batch(batch, metrics=metrics)

                    # # compute global_valid tokens
                    # batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # with _timer("reward", timing_raw):
                    #     # compute reward model score
                    #     if self.use_rm:
                    #         reward_tensor = self.rm_wg.compute_rm_score(batch)
                    #         batch = batch.union(reward_tensor)

                    #     if self.config.reward_model.launch_reward_fn_async:
                    #         future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                    #     else:
                    #         reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # # recompute old_log_probs
                    # with _timer("old_log_prob", timing_raw):
                    #     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    #     entropys = old_log_prob.batch["entropys"]
                    #     response_masks = batch.batch["response_mask"]
                    #     loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    #     entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    #     old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                    #     metrics.update(old_log_prob_metrics)
                    #     old_log_prob.batch.pop("entropys")
                    #     batch = batch.union(old_log_prob)

                    #     if "rollout_log_probs" in batch.batch.keys():
                    #         # TODO: we may want to add diff of probs too.
                    #         rollout_old_log_probs = batch.batch["rollout_log_probs"]
                    #         actor_old_log_probs = batch.batch["old_log_probs"]
                    #         attention_mask = batch.batch["attention_mask"]
                    #         responses = batch.batch["responses"]
                    #         response_length = responses.size(1)
                    #         response_mask = attention_mask[:, -response_length:]

                    #         rollout_probs = torch.exp(rollout_old_log_probs)
                    #         actor_probs = torch.exp(actor_old_log_probs)
                    #         rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                    #         rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                    #         rollout_probs_diff_max = torch.max(rollout_probs_diff)
                    #         rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                    #         rollout_probs_diff_std = torch.std(rollout_probs_diff)
                    #         metrics.update(
                    #             {
                    #                 "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                    #                 "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                    #                 "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                    #             }
                    #         )
                    self._balance_batch(batch, metrics=metrics)
                    
                    # DIAGNOSTIC: Check full_context_input_ids after balance_batch
                    if 'full_context_input_ids' in batch.batch:
                        print(f"[Trainer] ✓ full_context_input_ids after balance_batch: "
                                  f"shape={batch.batch['full_context_input_ids'].shape}")
                    else:
                        print(f"[Trainer] ⚠️ full_context_input_ids NOT found after balance_batch. "
                              f"This is normal - it will be injected after compute_advantage. "
                              f"Available keys: {list(batch.batch.keys())[:10]}...")

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        # reward_extra_infos_dict: dict[str, list]
                        # if self.config.reward_model.launch_reward_fn_async:
                        #     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # batch.batch["token_level_scores"] = reward_tensor

                        # print(f"{list(reward_extra_infos_dict.keys())=}")
                        # if reward_extra_infos_dict:
                        #     batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # # compute rewards. apply_invalid_action_penalty if available
                        # if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                        #     batch, invalid_metrics = apply_invalid_action_penalty(batch,
                        #                                                           invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                        #                                                           )
                        #     metrics.update(invalid_metrics)

                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor                        

                        # compute rewards. apply_kl_penalty if enabled
                        # When using in-reward KL, the controller is initialized in __init__.
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            # No KL in reward: rewards equal scores
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        # norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        # ==================== CRITICAL INVARIANT CHECK ====================
                        # If response length is 0, downstream GAE will crash (stack expects non-empty list),
                        # and reward managers / mask builders can also produce inconsistent shapes.
                        # This should be extremely rare; when it happens, skip this batch with enough context to debug.
                        try:
                            _resp = batch.batch.get("responses", None)
                            _rews = batch.batch.get("token_level_rewards", None)
                            if isinstance(_resp, torch.Tensor) and _resp.dim() == 2 and _resp.size(1) == 0:
                                uid_preview = None
                                try:
                                    uid_preview = batch.non_tensor_batch.get("uid", None)
                                except Exception:
                                    uid_preview = None
                                logger.error(
                                    "[EmptyResponse] step=%s: responses.shape=%s token_level_rewards.shape=%s "
                                    "old_log_probs.shape=%s responses_types.shape=%s attention_mask.shape=%s uid=%s. "
                                    "Skipping this batch.",
                                    getattr(self, "global_steps", "NA"),
                                    tuple(_resp.shape),
                                    tuple(_rews.shape) if isinstance(_rews, torch.Tensor) else None,
                                    tuple(batch.batch["old_log_probs"].shape) if "old_log_probs" in batch.batch else None,
                                    tuple(batch.batch["responses_types"].shape) if "responses_types" in batch.batch else None,
                                    tuple(batch.batch["attention_mask"].shape) if "attention_mask" in batch.batch else None,
                                    uid_preview,
                                )
                                continue
                            if isinstance(_rews, torch.Tensor) and _rews.dim() == 2 and _rews.size(1) == 0:
                                logger.error(
                                    "[EmptyRewards] step=%s: token_level_rewards.shape=%s responses.shape=%s. Skipping this batch.",
                                    getattr(self, "global_steps", "NA"),
                                    tuple(_rews.shape),
                                    tuple(_resp.shape) if isinstance(_resp, torch.Tensor) else None,
                                )
                                continue
                        except Exception as _e:
                            logger.warning(f"[InvariantCheck] Exception during empty-length check: {_e}")

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            # norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            # multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            # use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            # pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            # pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )
                        
                        # CRITICAL: Call _maybe_log_full_context_consistency AFTER compute_advantage
                        # This ensures old_log_probs and response_mask exist before injecting full_context_input_ids
                        # full_context_input_ids will be preserved during subsequent operations (reorder, etc.)
                        consistency_metrics = self._maybe_log_full_context_consistency(batch)
                        if consistency_metrics:
                            metrics.update(consistency_metrics)
                        
                        # DIAGNOSTIC: Verify full_context_input_ids was injected
                        if 'full_context_input_ids' in batch.batch:
                            print(f"[Trainer] ✓ full_context_input_ids injected after compute_advantage: "
                                  f"shape={batch.batch['full_context_input_ids'].shape}")

                        # ==================== Responses & Mask Check (After compute_advantage) ====================
                        # At this point, response_mask has been computed inside compute_advantage
                        try:
                            _resp_check_freq = int(os.environ.get("RESPONSES_CHECK_FREQ", "1"))
                            _resp_check_verbose = os.environ.get("RESPONSES_CHECK_VERBOSE", "0") != "0"
                        except Exception:
                            _resp_check_freq = 10
                            _resp_check_verbose = False
                        
                        if _resp_check_freq > 0 and (self.global_steps % _resp_check_freq == 0) and self.config.enable_debug_logs:
                            try:
                                from verl.utils.check_responses import check_responses_in_training
                                passed = check_responses_in_training(
                                    batch=batch,
                                    step=self.global_steps,
                                    check_freq=_resp_check_freq,
                                    verbose=_resp_check_verbose,
                                    tokenizer=self.tokenizer if hasattr(self, 'tokenizer') else None
                                )
                                if not passed:
                                    print(f"⚠️  [Step {self.global_steps}] Responses/mask checks FAILED (after compute_advantage)")
                            except Exception as e:
                                print(f"⚠️  [RESPONSES CHECK] Exception at step {self.global_steps}: {e}")
                                import traceback
                                traceback.print_exc()
                        # ========================================================================

                        generation_for_logging = final_gen_batch_output if 'final_gen_batch_output' in locals() else None

                    # Log summary distribution (configurable interval, default every step now)
                    if self.experiment_logger and generation_for_logging is not None:
                        try:
                            summaries = collect_response_texts(
                                data_proto=generation_for_logging,
                                tokenizer=self.tokenizer,
                                max_samples=200,
                                include_all_turns=False,
                            )

                            if summaries:
                                self.experiment_logger.log_summary_distribution(
                                    step=self.global_steps,
                                    summaries=summaries,
                                    tokenizer=self.tokenizer,
                                    save_full_data=False,
                                )

                                self.experiment_logger.log_shift_metrics(
                                    step=self.global_steps,
                                    current_summaries=summaries,
                                    tokenizer=self.tokenizer,
                                    compute_info_preservation=False,
                                    full_contexts=None,
                                )
                            else:
                                print(f"[Trainer] No decoded responses available for summary logging at step {self.global_steps}")
                        except Exception as e:
                            print(f"[Trainer] Error logging summary distribution: {e}")
                            import traceback
                            traceback.print_exc()

                     

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Set required meta_info for actor update
                        batch.meta_info.update({
                            "temperature": 1.0,
                            "micro_batch_size": self.config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", 1),
                            "use_dynamic_bsz": self.config.actor_rollout_ref.actor.get("use_dynamic_bsz", False),
                            "max_token_len": self.config.actor_rollout_ref.actor.get("max_token_len", 2048),
                            "multi_turn": batch.meta_info.get("multi_turn", False),
                        })
                        
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        
                        # Log stability metrics
                        if self.experiment_logger:
                            try:
                                stability_metrics = {}
                                metric_mapping = [
                                    ("actor/pg_loss", "pg_loss"),
                                    ("actor/pg_clipfrac", "clipfrac"),
                                    ("actor/ppo_kl", "approx_kl"),
                                    ("actor/grad_norm", "grad_norm"),
                                ]
                                for source_key, target_key in metric_mapping:
                                    if source_key in actor_output_metrics:
                                        stability_metrics[target_key] = float(actor_output_metrics[source_key])

                                if stability_metrics:
                                    self.experiment_logger.log_stability_metrics(
                                        step=self.global_steps,
                                        metrics=stability_metrics,
                                    )
                            except Exception as e:
                                print(f"[Trainer] Error logging stability metrics: {e}")
                        
                        # ==================== Track Loss History for Gradient Flow Debug ====================
                        # Track actor loss and average response length for stability analysis
                        if 'actor/pg_loss' in metrics and 'response_mask' in batch.batch:
                            if not hasattr(self, '_loss_history'):
                                self._loss_history = []
                            
                            actor_loss_value = metrics['actor/pg_loss']
                            response_mask = batch.batch['response_mask']
                            
                            # Calculate average response length per sample
                            num_samples = response_mask.shape[0]
                            total_valid_tokens = response_mask.sum().item()
                            avg_resp_len = total_valid_tokens / num_samples if num_samples > 0 else 0
                            
                            # Store (loss, avg_response_length)
                            self._loss_history.append((float(actor_loss_value), int(avg_resp_len)))
                            
                            # Keep only last 100 steps to avoid memory issues
                            if len(self._loss_history) > 100:
                                self._loss_history = self._loss_history[-100:]
                        # ==================================================================================

                    # Log rollout generations if enabled
                    # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # if rollout_data_dir:
                    #     with _timer("dump_rollout_generations", timing_raw):
                    #         print(batch.batch.keys())
                    #         inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    #         outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    #         scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    #         self._dump_generations(
                    #             inputs=inputs,
                    #             outputs=outputs,
                    #             scores=scores,
                    #             reward_extra_infos_dict=reward_extra_infos_dict,
                    #             dump_path=rollout_data_dir,
                    #         )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)


                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        print(f"\n[CHECKPOINT] Triggering checkpoint save at step {self.global_steps}")
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                        print(f"[CHECKPOINT] Checkpoint save completed for step {self.global_steps}\n")

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                
                # Print current training progress
                print(f"[TRAINING] Step {self.global_steps}/{self.total_training_steps} (Epoch {epoch})")
                if self.global_steps % 10 == 0:
                    print(f"[TRAINING] Next checkpoint will be saved at step {((self.global_steps // self.config.trainer.save_freq) + 1) * self.config.trainer.save_freq}")
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                
                # Print end of step summary
                print(f"\n{'*'*40}")
                print(f"[TRAINING] Completed step {self.global_steps}/{self.total_training_steps}")
                if 'training/loss' in metrics:
                    print(f"[TRAINING] Loss: {metrics['training/loss']:.6f}")
                if 'actor/pg_loss' in metrics:
                    print(f"[TRAINING] Actor Loss: {metrics['actor/pg_loss']:.6f}")
                if 'training/throughput_tokens_per_gpu_per_sec' in metrics:
                    print(f"[TRAINING] Throughput: {metrics['training/throughput_tokens_per_gpu_per_sec']:.2f} tokens/gpu/sec")
                print(f"{'*'*40}\n")
                
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    # Close experiment logger
                    if self.experiment_logger:
                        try:
                            self.experiment_logger.close()
                            print("[Trainer] Experiment logger closed")
                        except Exception as e:
                            print(f"[Trainer] Error closing experiment logger: {e}")
                    
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
