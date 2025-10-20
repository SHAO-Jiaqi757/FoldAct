import torch
import json
import uuid
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional
import logging

# Configure the logging level and format
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s]: %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from enum import Enum
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.tools.search_tool import SearchTool
from verl.tools.schemas import OpenAIFunctionToolSchema
import shutil
import requests


TOOL_SCHEMA = OpenAIFunctionToolSchema(
    type="function",
    function={
        "name": "search",
        "description": "Searches for relevant information based on queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries"
                },
                "topk": {
                    "type": "integer",
                    "description": "Number of top results to return"
                }
            },
            "required": ["query_list"]
        }
    }
)


class ResponseType(Enum):
    think = 0
    search = 1
    answer = 2
    information = 3
    information_summary = 4
    think_summary = 5


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    retriever_num_workers:int = 5 
    retriever_rate_limit:int = 120 
    retriever_timeout:int = 30 
    retriever_enable_global_rate_limit:bool = True 
    # Whether to allow do_search during the final rollout and append <information>
    final_turn_do_search: bool = False
    # Enable sliding window context: keep only the most recent N turns (0 = keep all)
    use_sliding: bool = False
    
    # KL-Aware Training: Compare compressed vs full context policies
    # Ratio of rollouts that use full context (e.g., 0.1 = 10% full, 90% compressed)
    full_context_ratio: float = 0.1
    # Whether to compute full-context log_prob baseline for compressed rollouts
    # This enables KL(π_comp || π_full) regularization during training
    enable_kl_baseline: bool = True
    # Performance optimization: reduce logging overhead
    enable_debug_logs: bool = False

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        async_rollout_manager=None,
        use_async: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.async_rollout_manager = async_rollout_manager
        self.use_async = use_async


        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))


        self.search_tool=SearchTool(config=config,tool_schema=TOOL_SCHEMA)

    def _apply_sliding_window(self, turn_history: List[Dict[str, torch.Tensor]], 
                            initial_question: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive sliding window based on whether the last turn has information.
        
        Logic:
        - If last turn has information: apply aggressive truncation (keep only recent turns)
        - If last turn has no information: keep more context (less aggressive truncation)
        
        This ensures that when the model sees information, it can focus on recent context,
        but when it doesn't have information, it can access more historical context.
        """
        
        # ========== CONTEXT SUMMARIZATION LOGGING ==========
        if getattr(self.config, 'enable_debug_logs', False):
            print(f"\n{'='*80}")
            print(f"[CONTEXT SUMMARY] Applying adaptive sliding window context compression")
            print(f"{'='*80}")
            print(f"[CONTEXT SUMMARY] Total turns in history: {len(turn_history)}")
            print(f"[CONTEXT SUMMARY] Initial question length: {initial_question.shape[1]} tokens")
        
        if len(turn_history) == 0:
            if getattr(self.config, 'enable_debug_logs', False):
                print(f"[CONTEXT SUMMARY] No turns in history, returning initial question")
            return initial_question

        # Analyze each turn for information content
        turn_analysis = []
        for i, turn in enumerate(turn_history):
            response_len = turn['responses'].shape[1]
            obs_len = turn.get('observations', torch.tensor([])).shape[1] if turn.get('observations') is not None else 0
            total_len = response_len + obs_len
            
            # Check if turn has meaningful information (observations)
            has_information = obs_len > 0
            turn_analysis.append({
                'index': i,
                'response_len': response_len,
                'obs_len': obs_len,
                'total_len': total_len,
                'has_information': has_information
            })

        # Single-scheme truncation:
        # Keep the contiguous block consisting of the most recent info-turn
        # and any immediately preceding non-info turns. If no info exists, keep all.
        has_info_flags = []
        for i, t in enumerate(turn_history):
            flag = False
            obs = t.get('observations')
            if obs is not None and obs.shape[1] > 0:
                # Check if observation has substantial content (more than just turn headers)
                obs_tokens = obs[0]
                flag = obs_tokens.shape[0] > 20  # More than just turn headers and basic info
            has_info_flags.append(flag)
            if getattr(self.config, 'enable_debug_logs', False):
                print(f"[CONTEXT SUMMARY] Turn {i} has_info_tag={flag}")

        # Find last info index
        last_info_idx = -1
        for i in range(len(turn_history) - 1, -1, -1):
            if has_info_flags[i]:
                last_info_idx = i
                break

        if last_info_idx == -1:
            # No info yet → keep all context
            selected_turns = turn_history
            if getattr(self.config, 'enable_debug_logs', False):
                print(f"[CONTEXT SUMMARY] No <information> found in history → keeping ALL turns")
        else:
            # Include contiguous non-info turns immediately before the last info turn
            start_idx = last_info_idx
            while start_idx - 1 >= 0 and not has_info_flags[start_idx - 1]:
                start_idx -= 1
            selected_turns = turn_history[start_idx:last_info_idx + 1]
            if getattr(self.config, 'enable_debug_logs', False):
                print(f"[CONTEXT SUMMARY] Using block turns [{start_idx}..{last_info_idx}] (non-info before last info + last info)")
        all_tokens: List[torch.Tensor] = [initial_question.clone()]
        total_compressed_length = initial_question.shape[1]
        if getattr(self.config, 'enable_debug_logs', False):
            print(f"[CONTEXT SUMMARY] Added initial question: {initial_question.shape[1]} tokens")
        
        for i, turn in enumerate(selected_turns):
            all_tokens.append(turn['responses'])
            if turn.get('observations') is not None:
                all_tokens.append(turn['observations'])
            
            turn_len = turn['responses'].shape[1] + (turn.get('observations', torch.tensor([])).shape[1] if turn.get('observations') is not None else 0)
            total_compressed_length += turn_len
            if getattr(self.config, 'enable_debug_logs', False):
                print(f"[CONTEXT SUMMARY] Selected turn {i}: {turn_len} tokens")

        compressed_context = self.tensor_fn.concatenate_with_padding(all_tokens)
        final_length = compressed_context.shape[1]
        
        # Calculate compression statistics (include initial question length)
        original_length = initial_question.shape[1] + sum(ta['total_len'] for ta in turn_analysis)
        tokens_saved = original_length - final_length
        compression_ratio = tokens_saved / original_length if original_length > 0 else 0.0
        
        if getattr(self.config, 'enable_debug_logs', False):
            print(f"[CONTEXT SUMMARY] COMPRESSION RESULTS:")
            print(f"[CONTEXT SUMMARY]   Original context: {original_length} tokens")
            print(f"[CONTEXT SUMMARY]   Compressed context: {final_length} tokens")
            print(f"[CONTEXT SUMMARY]   Tokens saved: {tokens_saved} tokens")
            print(f"[CONTEXT SUMMARY]   Compression ratio: {compression_ratio:.1%}")
            print(f"{'='*80}\n")
        
        return compressed_context

    def _batch_tokenize(self, responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of responses."""
        result = self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest",
            return_offsets_mapping=True
        )
        input_ids = result['input_ids']
        # Be robust to tokenizer implementations that return 2-D or 3-D offset mappings.
        # Expected: (batch, seq_len, 2) -> take start offsets [:, :, 0].
        # Some tokenizers may already return (batch, seq_len) of start offsets.
        offsets = result.get('offset_mapping', None)
        if offsets is None:
            # Fallback: no offsets available; return zeros with same shape as input_ids.
            start_offsets = torch.zeros_like(input_ids)
        else:
            if isinstance(offsets, torch.Tensor):
                if offsets.dim() == 3 and offsets.size(-1) >= 1:
                    start_offsets = offsets[:, :, 0]
                elif offsets.dim() == 2:
                    start_offsets = offsets
                else:
                    start_offsets = torch.zeros_like(input_ids)
            else:
                # Convert to tensor if tokenizer returned a list
                try:
                    offsets_tensor = torch.tensor(offsets)
                    if offsets_tensor.dim() == 3 and offsets_tensor.size(-1) >= 1:
                        start_offsets = offsets_tensor[:, :, 0]
                    elif offsets_tensor.dim() == 2:
                        start_offsets = offsets_tensor
                    else:
                        start_offsets = torch.zeros_like(input_ids)
                except Exception:
                    start_offsets = torch.zeros_like(input_ids)
        return input_ids, start_offsets

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses, responses_offsets = self._batch_tokenize(responses_str)
        return responses, responses_offsets, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Process next observations from environment.
        
        Returns:
            next_obs_ids: Tokenized observation IDs
            information_types: Type markers for each token
            obs_too_long: Flag indicating if any observation exceeded max_obs_length
        """
        
        result = self.tokenizer(
            next_obs, 
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )

        next_obs_ids = result['input_ids']
        information_types = result['attention_mask'] * ResponseType.information.value

        obs_too_long = False
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            obs_too_long = True
            print(f"[WARNING] OBSERVATION TOO LONG ({next_obs_ids.shape[1]} > {self.config.max_obs_length}), FORCING SUMMARIZED CONTEXT")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids, information_types, obs_too_long

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor,
                prompt_step_ids: torch.Tensor,
                prompt_types: torch.Tensor,
                response: torch.Tensor, 
                step: int,
                response_types: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        id_tensors = [prompt_step_ids, torch.full(response.size(), step, dtype=prompt_step_ids.dtype, device=prompt_step_ids.device)]
        type_tensors = [prompt_types, response_types]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
            id_tensors.append(torch.full(info.size(), step, dtype=prompt_step_ids.dtype, device=prompt_step_ids.device))
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        id_concatenated = torch.cat(id_tensors, dim=1)
        type_concatenated = torch.cat(type_tensors, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        padded_id = id_concatenated.gather(1, sorted_indices)
        padded_type = type_concatenated.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, padded_id, padded_type

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          step: int,
                          cur_types: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask, step_ids, responses_types = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    right_side['step_ids'],
                    right_side['responses_types'],
                    cur_responses,
                    step,
                    cur_types,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask, step_ids, responses_types = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    right_side['step_ids'],
                    right_side['responses_types'],
                    cur_responses,
                    step,
                    cur_types,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len],
            'step_ids': step_ids[:, :max_len],
            'responses_types': responses_types[:, :max_len]
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            if self.use_async and self.async_rollout_manager is not None:
                return self.async_rollout_manager.generate_sequences(active_batch)
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            if self.use_async and self.async_rollout_manager is not None:
                return self.async_rollout_manager.generate_sequences(active_batch)
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        if self.use_async and self.async_rollout_manager is not None:
            padded_output = self.async_rollout_manager.generate_sequences(padded_active_batch)
        else:
            padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop with optional sliding window context management."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []],
            # Response token generated at step i will have i at its position
            "step_ids": initial_input_ids[:, []],
            # Response token belonging to type i (according to ResponseType enum) witll have i at its position
            "responses_types": initial_input_ids[:, []],
        }
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Track history for sliding window if enabled
        use_sliding = getattr(self.config, "use_sliding", False)
        print(f"\n[DEBUG] use_sliding: {use_sliding}")
        
        # Initialize turn_history and initial_question_ids regardless of use_sliding
        initial_question_ids = gen_batch.batch['input_ids'].clone()
        turn_history: List[Dict[str, torch.Tensor]] = []
        
        if use_sliding:
            print(f"[DEBUG] Sliding window enabled, turn history initialized")
        
        # KL-Aware Training: Decide context strategy
        import random
        full_context_ratio = getattr(self.config, "full_context_ratio", 0.1)
        enable_kl_baseline = getattr(self.config, "enable_kl_baseline", True)
        
        # Random selection: full context or compressed context
        use_full_context_this_rollout = random.random() < full_context_ratio
        
        # Track if observation is too long (forces summarized context)
        force_summarized = False
        
        # For KL-aware training: track full context separately if using compressed
        full_context_for_kl_baseline = None  # Will store full context if needed
        
        context_type = "full" if use_full_context_this_rollout else "compressed"
        logger.info(f"[KL-Aware] Rollout context: {context_type} (full_ratio={full_context_ratio:.1%}, kl_baseline={enable_kl_baseline})")
        
        # Context length monitoring
        context_lengths_per_turn = []  # Track context length at each turn
        full_context_lengths_per_turn = []  # Track what full context would be (for compressed rollouts)

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            # Capture the exact input used for generation for logging
            try:
                generation_input_ids_for_log = rollings_active.batch['input_ids']
            except Exception:
                generation_input_ids_for_log = rollings.batch['input_ids']
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            # Responses after </search> or </answer> are discarded
            responses_ids, active_offsets, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            responses_offsets = torch.zeros(
                (active_mask.shape[0], active_offsets.shape[1]),
                dtype=active_offsets.dtype, device=active_offsets.device
            )
            responses_offsets[active_mask] = active_offsets

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, action_ranges = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=True, 
                current_turn=step, max_turns=self.config.max_turns
            )
            action_types = self._build_action_types(responses_ids, responses_offsets, action_ranges, is_search, responses_str)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # If all trajectories are finished (e.g., answered), stop immediately
            if not active_mask.any():
                print(f"\n[TURN GENERATION] All trajectories completed at turn {step}. Stopping loop.")
                break

            next_obs_ids, information_types, obs_too_long = self._process_next_obs(next_obs)
            
            # ========== OBSERVATION LENGTH LOGGING ==========
            if obs_too_long:
                print(f"\n{'='*60}")
                print(f"[OBSERVATION WARNING] Turn {step} - Observation too long!")
                print(f"{'='*60}")
                print(f"[OBSERVATION WARNING] Observation length exceeds max_obs_length")
                print(f"[OBSERVATION WARNING] FORCING summarized context for remaining turns")
                print(f"[OBSERVATION WARNING] This overrides the original context strategy")
                print(f"{'='*60}\n")
                force_summarized = True  # Force summarized context for remaining turns
            
            responses_types = torch.cat([action_types, information_types], dim=1)

            # ========== ADD TURN INFORMATION TO CONTEXT ==========
            # Add turn number and constraint information to agent context
            turn_context_info = f"\n\n[Turn {step + 1}/{self.config.max_turns}] You are currently on turn {step + 1} of {self.config.max_turns} maximum turns."
            if step == self.config.max_turns - 1:
                turn_context_info += f"\n⚠️ **FINAL TURN** ⚠️ This is your LAST turn. You MUST provide your final answer now using <answer>...</answer> tags."
            else:
                remaining_turns = self.config.max_turns - step - 1
                turn_context_info += f" You have {remaining_turns} turn(s) remaining. Use <search>...</search> to search for information or <answer>...</answer> to provide your final answer."
            
            # Tokenize and add turn context to the input only if any trajectories remain active
            if active_mask.any():
                turn_context_tokens = self.tokenizer.encode(turn_context_info, add_special_tokens=False, return_tensors='pt')
                if turn_context_tokens.shape[1] > 0:
                    # Broadcast turn context tokens to match batch size
                    batch_size = rollings.batch['input_ids'].shape[0]
                    turn_context_tokens = turn_context_tokens.expand(batch_size, -1).to(rollings.batch['input_ids'].device)
                    
                    # Add turn context to the rolling state
                    rollings.batch['input_ids'] = torch.cat([rollings.batch['input_ids'], turn_context_tokens], dim=1)
                    rollings.batch['attention_mask'] = torch.cat([rollings.batch['attention_mask'], torch.ones_like(turn_context_tokens).to(rollings.batch['attention_mask'].device)], dim=1)
                    rollings.batch['position_ids'] = self.tensor_fn.create_position_ids(rollings.batch['attention_mask'])

            # ========== TURN TRACE (single structured line) ==========
            if getattr(self.config, 'enable_debug_logs', False):
                trace_id = getattr(self, '_trace_id', None)
                if trace_id is None:
                    trace_id = uuid.uuid4().hex[:8]
                    setattr(self, '_trace_id', trace_id)
                # Decode the exact input that was fed to the generator (pre-context-updates)
                current_context = generation_input_ids_for_log
                context_text_full = self.tokenizer.decode(current_context[0], skip_special_tokens=True)
                # Use postprocessed response text directly for fidelity
                response_text_full = (responses_str[0] if isinstance(responses_str, list) and len(responses_str) > 0 else "")
                # Get the observation that will be added to context after this response

                response_text = response_text_full
                
                turn_record = {
                    "trace_id": trace_id,
                    "step": int(step),
                    "context_type": context_type,
                    "context": context_text_full,  # Human-readable (no special tokens)
                    "response": response_text,
                }
                
                # Write turn record to file instead of printing
                import os
                log_dir = os.environ.get("TRAINING_LOG_DIR", ".")
                log_file = os.path.join(log_dir, "training_turn_logs.jsonl")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"turn": turn_record}, ensure_ascii=False) + "\n")

            # Update states
            if use_sliding:
                print(f"\n[DEBUG] Turn {step}: Entering sliding window block")
                print(f"[DEBUG] Turn history length before append: {len(turn_history)}")
                # Store current turn and rebuild rolling context using sliding window
                turn_history.append({
                    'responses': responses_ids.clone(),
                    'observations': next_obs_ids.clone(),
                })
                print(f"[DEBUG] Turn history length after append: {len(turn_history)}")
                
                # Decide whether to use full context for this turn
                # Priority: force_summarized > use_full_context_this_rollout
                
                # ========== CONTEXT DECISION LOGGING ==========
                print(f"\n{'='*60}")
                print(f"[CONTEXT DECISION] Turn {step} - Context Strategy Selection")
                print(f"{'='*60}")
                print(f"[CONTEXT DECISION] Force summarized: {force_summarized}")
                print(f"[CONTEXT DECISION] Use full context this rollout: {use_full_context_this_rollout}")
                print(f"[CONTEXT DECISION] Turn history length: {len(turn_history)}")
                print(f"[CONTEXT DECISION] Context type: {context_type}")
                
                # When full context is selected, bypass any truncation regardless of force_summarized
                if use_full_context_this_rollout:
                    # This rollout uses full context
                    all_tokens: List[torch.Tensor] = [initial_question_ids.clone()]
                    for t in turn_history:
                        all_tokens.append(t['responses'])
                        if t.get('observations') is not None:
                            all_tokens.append(t['observations'])
                    windowed_input_ids = self.tensor_fn.concatenate_with_padding(all_tokens)
                    
                    # Log full context statistics
                    full_context_length = windowed_input_ids.shape[1]
                    print(f"[CONTEXT DECISION] Full context length: {full_context_length} tokens")
                    print(f"[CONTEXT DECISION] Full context turns: {len(turn_history)}")
                    
                    if step == 0:
                        logger.info(f"Turn {step}: Using FULL context ({len(turn_history)} turns)")
                elif force_summarized:
                    # Observation too long: apply summarized context
                    print(f"[CONTEXT DECISION] DECISION: FORCED summarized context (observation too long)")
                    windowed_input_ids = self._apply_sliding_window(turn_history, initial_question_ids)
                    if step == 0 or step == self.config.max_turns - 1:
                        logger.info(f"Turn {step}: FORCED summarized context (observation too long)")
                else:
                    # This rollout uses compressed context (sliding window)
                    windowed_input_ids = self._apply_sliding_window(turn_history, initial_question_ids)
                    
                    # KL-Aware: Always update full context for baseline computation (keep all turns)
                    if enable_kl_baseline:
                        all_tokens: List[torch.Tensor] = [initial_question_ids.clone()]
                        for t in turn_history:
                            all_tokens.append(t['responses'])
                            if t.get('observations') is not None:
                                all_tokens.append(t['observations'])
                        full_context_for_kl_baseline = self.tensor_fn.concatenate_with_padding(all_tokens)
                        
                        # Log KL baseline statistics
                        compressed_length = windowed_input_ids.shape[1]
                        full_length = full_context_for_kl_baseline.shape[1]
                        tokens_saved = full_length - compressed_length
                        reduction_ratio = tokens_saved / full_length if full_length > 0 else 0.0
                        
                        print(f"[CONTEXT DECISION] KL Baseline context lengths:")
                        print(f"[CONTEXT DECISION]   Compressed: {compressed_length} tokens")
                        print(f"[CONTEXT DECISION]   Full (baseline): {full_length} tokens")
                        print(f"[CONTEXT DECISION]   Tokens saved: {tokens_saved} tokens")
                        print(f"[CONTEXT DECISION]   Reduction ratio: {reduction_ratio:.1%}")
                
                print(f"{'='*60}\n")
                
                rollings.batch['input_ids'] = windowed_input_ids
                rollings.batch['attention_mask'] = self.tensor_fn.create_attention_mask(windowed_input_ids)
                rollings.batch['position_ids'] = self.tensor_fn.create_position_ids(rollings.batch['attention_mask'])
                
                # ========== FINAL CONTEXT LOGGING ==========
                print(f"\n{'='*60}")
                print(f"[FINAL CONTEXT] Turn {step} - Context for Next Turn")
                print(f"{'='*60}")
                print(f"[FINAL CONTEXT] Final context length: {windowed_input_ids.shape[1]} tokens")
                if hasattr(self, 'tokenizer') and self.tokenizer is not None and getattr(self.config, 'enable_debug_logs', False):
                    final_context_text = self.tokenizer.decode(windowed_input_ids[0], skip_special_tokens=True)
                    print(f"[FINAL CONTEXT] Final context preview: {final_context_text}")  # Last 300 chars
                print(f"{'='*60}\n")
                
                # Monitor context length (after context decision is made)
                current_context_length = windowed_input_ids.shape[1]
                context_lengths_per_turn.append(current_context_length)
                
                # If compressed rollout with KL baseline, also track full context length
                if not use_full_context_this_rollout and enable_kl_baseline and full_context_for_kl_baseline is not None:
                    full_length = full_context_for_kl_baseline.shape[1]
                    full_context_lengths_per_turn.append(full_length)
            else:
                # ========== NON-SLIDING WINDOW LOGGING ==========
                print(f"\n{'='*60}")
                print(f"[NON-SLIDING] Turn {step} - Using Standard Context Update")
                print(f"{'='*60}")
                print(f"[NON-SLIDING] Using standard rolling state update")
                print(f"{'='*60}\n")
                
                rollings = self._update_rolling_state(
                    rollings,
                    responses_ids,
                    next_obs_ids
                )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                step,
                responses_types,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, active_offsets, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            responses_offsets = torch.zeros(
                (active_mask.shape[0], active_offsets.shape[1]),
                dtype=active_offsets.dtype, device=active_offsets.device
            )
            responses_offsets[active_mask] = active_offsets

            # Execute in environment and process observations
            # Optionally allow search on the final turn to include <information>
            do_search_final = getattr(self.config, "final_turn_do_search", False)
            next_obs, dones, valid_action, is_search, action_ranges = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=do_search_final,
                current_turn=self.config.max_turns, max_turns=self.config.max_turns
            )
            responses_types = self._build_action_types(responses_ids, responses_offsets, action_ranges, is_search, responses_str)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            if do_search_final:
                # Process observations: get token ids and corresponding information types
                next_obs_ids, information_types, obs_too_long = self._process_next_obs(next_obs)
                # Note: We don't update force_summarized here since this is the final turn
                # Extend response types to include information types so lengths match
                responses_types = torch.cat([responses_types, information_types], dim=1)
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    self.config.max_turns,
                    responses_types,
                    next_obs_ids,
                )
            else:
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    self.config.max_turns,
                    responses_types
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        # KL-Aware Training: Add context type and full context baseline if needed
        meta_info['context_type'] = context_type
        if not use_full_context_this_rollout and enable_kl_baseline and full_context_for_kl_baseline is not None:
            meta_info['has_kl_baseline'] = True
            # Store full context for later log_prob computation
            meta_info['full_context_input_ids'] = full_context_for_kl_baseline
        else:
            meta_info['has_kl_baseline'] = False
        
        # Context Length Monitoring: Add statistics for WandB tracking
        if context_lengths_per_turn:
            final_context_length = context_lengths_per_turn[-1]
            meta_info['final_context_length'] = final_context_length
            meta_info['avg_context_length'] = sum(context_lengths_per_turn) / len(context_lengths_per_turn)
            
            # ========== FINAL CONTEXT SUMMARY LOGGING ==========
            print(f"\n{'='*80}")
            print(f"[FINAL CONTEXT SUMMARY] Rollout Complete - Context Statistics")
            print(f"{'='*80}")
            print(f"[FINAL CONTEXT SUMMARY] Context type: {context_type}")
            print(f"[FINAL CONTEXT SUMMARY] Total turns processed: {len(context_lengths_per_turn)}")
            print(f"[FINAL CONTEXT SUMMARY] Final context length: {final_context_length} tokens")
            print(f"[FINAL CONTEXT SUMMARY] Average context length: {sum(context_lengths_per_turn) / len(context_lengths_per_turn):.1f} tokens")
            
            # If compressed rollout with full context tracking
            if full_context_lengths_per_turn:
                final_full_length = full_context_lengths_per_turn[-1]
                tokens_saved = final_full_length - final_context_length
                reduction_ratio = tokens_saved / final_full_length if final_full_length > 0 else 0.0
                
                meta_info['compression_stats'] = {
                    'compressed_length': final_context_length,
                    'full_length': final_full_length,
                    'tokens_saved': tokens_saved,
                    'reduction_ratio': reduction_ratio,
                }
                
                print(f"[FINAL CONTEXT SUMMARY] COMPRESSION RESULTS:")
                print(f"[FINAL CONTEXT SUMMARY]   Compressed context: {final_context_length} tokens")
                print(f"[FINAL CONTEXT SUMMARY]   Full context would be: {final_full_length} tokens")
                print(f"[FINAL CONTEXT SUMMARY]   Tokens saved: {tokens_saved} tokens")
                print(f"[FINAL CONTEXT SUMMARY]   Compression ratio: {reduction_ratio:.1%}")
                print(f"[FINAL CONTEXT SUMMARY]   Memory efficiency: {1 - reduction_ratio:.1%} of full context")
                
                logger.info(f"[Context Length] Compressed: {final_context_length} tokens, "
                           f"Full would be: {final_full_length} tokens, "
                           f"Saved: {tokens_saved} tokens ({reduction_ratio:.1%} reduction)")
            else:
                # Full context rollout
                print(f"[FINAL CONTEXT SUMMARY] FULL CONTEXT ROLLOUT:")
                print(f"[FINAL CONTEXT SUMMARY]   No compression applied")
                print(f"[FINAL CONTEXT SUMMARY]   Context length: {final_context_length} tokens")
                print(f"[FINAL CONTEXT SUMMARY]   All turns preserved")
                
                logger.info(f"[Context Length] Full context: {final_context_length} tokens")
            
            print(f"{'='*80}\n")
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _build_action_types(self,
                            responses_ids: torch.Tensor,
                            responses_offsets: torch.Tensor,
                            action_ranges: List[Optional[Tuple[int, int]]],
                            is_search: List[int], responses_str) -> torch.Tensor:

        # The type is think by default
        types = torch.zeros_like(responses_ids)
        for i in range(len(responses_ids)):
            if action_ranges[i]:
                # Keep only until the last non-zero index for searchsorted() to work
                nonzero_idx = (responses_offsets[i] != 0).nonzero(as_tuple=True)[0]
                if nonzero_idx.numel() == 0:
                    # No valid offsets; skip labeling for this sample
                    continue
                last_nonzero_idx = nonzero_idx[-1]
                valid_offsets = responses_offsets[i][:last_nonzero_idx+1]

                # Select the smallest range of tokens to FULLY contain the action substring
                # If s is in the middle of token i, then include token i
                # If e is in the middle of token i, then use token i + 1 as the right bound instead
                s, e = action_ranges[i]
                s_idx = torch.searchsorted(valid_offsets, s, right=True) - 1
                e_idx = torch.searchsorted(valid_offsets, e)

                # Determine the correct response type based on action
                if is_search[i]:
                    types[i, s_idx:e_idx] = ResponseType.search.value
                else:
                    # Check the action type from the response string (case-insensitive)
                    action_str = responses_str[i][s:e].lower() if s < len(responses_str[i]) and e <= len(responses_str[i]) else ""
                    if "<answer>" in action_str:
                        types[i, s_idx:e_idx] = ResponseType.answer.value
                    elif "<think_summary>" in action_str:
                        types[i, s_idx:e_idx] = ResponseType.think_summary.value
                    elif "<information_summary>" in action_str:
                        types[i, s_idx:e_idx] = ResponseType.information_summary.value
                    elif "<think>" in action_str and "<think_summary>" not in action_str:
                        types[i, s_idx:e_idx] = ResponseType.think.value
                    else:
                        # Default to think for any other content
                        types[i, s_idx:e_idx] = ResponseType.think.value

        return types

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self,predictions:List[str],pad_token:str,active_mask=None,do_search=True,current_turn:int=0,max_turns:int=None) \
        -> Tuple[List[str], List[int], List[int], List[int], List[Optional[Tuple[int, int]]]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            current_turn: Current turn number (0-indexed)
            max_turns: Maximum number of turns allowed
            
        Returns:
            List of observation strings
        """

        cur_actions,contents,action_ranges=self.postprocess_predictions(predictions)
        next_obs,dones,valid_action,is_search=[],[],[],[]

        # Build search queries only for active slots requesting search
        active_search_indices = [i for i, (action, active) in enumerate(zip(cur_actions, active_mask)) if active and action == 'search']
        search_queries = [contents[i] for i in active_search_indices]
        expected_results = len(search_queries)

        if do_search and search_queries:
            parameters={"query_list":search_queries,"topk":self.config.topk}
            import asyncio
            import json

            # Execute search; capture metrics to detect skip conditions
            instance = asyncio.run(self.search_tool.create())
            exec_result = asyncio.run(self.search_tool.execute(instance_id=instance, parameters=parameters))
            # exec_result: (result_text, tool_reward, metrics)
            if isinstance(exec_result, tuple) and len(exec_result) >= 1:
                search_results_json = exec_result[0]
                metrics = exec_result[2] if len(exec_result) > 2 else {}
            else:
                search_results_json = exec_result
                metrics = {}
            # Robustly parse results and coerce to a list
            try:
                parsed = json.loads(search_results_json)
            except Exception:
                parsed = search_results_json

            if isinstance(parsed, dict):
                result_obj = parsed.get('result', parsed.get('results', parsed))
            else:
                result_obj = parsed

            if isinstance(result_obj, list):
                search_results = result_obj
            elif isinstance(result_obj, str):
                search_results = [result_obj]
            elif isinstance(result_obj, dict):
                # Fallback: stringify dict as a single result item
                search_results = [json.dumps(result_obj, ensure_ascii=False)]
            else:
                # Unknown type -> single empty string placeholder
                search_results = ["[Unknow Type]"]
            # If SearchTool signaled skip, treat as no evidence
            if isinstance(metrics, dict) and metrics.get('skipped', False):
                search_results = [''] * expected_results
            else:
                # Normalize result length to expected active search count
                if len(search_results) < expected_results:
                    logger.warning(f"Got fewer results ({len(search_results)}) than expected ({expected_results}), padding with placeholders")
                    # Pad with explicit placeholders to avoid empty information blocks
                    missing = expected_results - len(search_results)
                    pad_items = [f"[No results] Query: '{q[:128]}' | status=no_results" for q in search_queries[-missing:]] if search_queries else ["[No results] | status=no_results"] * missing
                    search_results += pad_items
                elif len(search_results) > expected_results:
                    search_results = search_results[:expected_results]
        else:
            search_results = [''] * expected_results

        for i, (action,active) in enumerate(zip(cur_actions,active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)

            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    # Only consume search result for active entries
                    # Add turn information to observation
                    turn_info = f"\n[Turn {current_turn + 1}/{max_turns if max_turns else '?'}] "
                    search_result = search_results.pop(0).strip()
                    
                    # Check if this is the last turn and provide sharp message
                    if max_turns and current_turn >= max_turns - 1:
                        sharp_message = f"\n\n⚠️ **FINAL TURN WARNING** ⚠️\nThis is your LAST turn (Turn {current_turn + 1}/{max_turns}). You MUST provide your final answer now using <answer>...</answer> tags. No more searches are allowed.\n\n"
                        # Fix: Add proper role labeling for information blocks
                        next_obs.append(f'{turn_info}user\n<information>{search_result}</information>\n\nassistant\n{sharp_message}')
                    else:
                        # Fix: Add proper role labeling for information blocks
                        next_obs.append(f'{turn_info}user\n<information>{search_result}</information>\n\nassistant\n')
                    
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                elif action in ["think", "think_summary", "information_summary"]:
                    # Add turn information to observation
                    turn_info = f"\n[Turn {current_turn + 1}/{max_turns if max_turns else '?'}] "
                    
                    # Check if this is the last turn and provide sharp message
                    if max_turns and current_turn >= max_turns - 1:
                        sharp_message = f"\n\n⚠️ **FINAL TURN WARNING** ⚠️\nThis is your LAST turn (Turn {current_turn + 1}/{max_turns}). You MUST provide your final answer now using <answer>...</answer> tags.\n\n"
                        next_obs.append(f'{turn_info}{sharp_message}')
                    else:
                        next_obs.append(f'{turn_info}')
                    
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(0)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to present thinking process, I should put the thinking process between <think> and </think>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. \
If I want to summarize, use <think_summary> and <information_summary>, if no <information> yet, <information_summary> is invalid. Let me try again.\n') 
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search, action_ranges

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool], List[Optional[Tuple[int, int]]]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        action_ranges = []
                
        # Precompile a flexible pattern that tolerates whitespace/casing variations
        tag_pattern = re.compile(
            r'<\s*(search|answer|think|think_summary|information_summary)\b[^>]*>(.*?)</\s*\1\s*>',
            re.IGNORECASE | re.DOTALL
        )

        for prediction in predictions:
            if isinstance(prediction, str):  # for llm output
                # Priority: answer action should be recognized first (ending signal)
                answer_match = re.search(r'<\s*answer\b[^>]*>(.*?)</\s*answer\s*>', prediction, re.IGNORECASE | re.DOTALL)
                if answer_match:
                    content = answer_match.group(1).strip()
                    action = 'answer'
                    action_range = answer_match.span()
                else:
                    # Fallback to other actions
                    match = tag_pattern.search(prediction)
                    if match:
                        content = match.group(2).strip()  # content between the tags
                        action = match.group(1).lower()
                        action_range = match.span()
                        # Guard: treat <search> with blank/punctuation-only content as invalid
                        if action == 'search':
                            content_stripped = content.strip()
                            if not content_stripped or not len(content_stripped) > 5:
                                action = None
                                content = ''
                                action_range = None
                    else:
                        content = ''
                        action = None
                        action_range = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            action_ranges.append(action_range)
            
        return actions, contents, action_ranges
