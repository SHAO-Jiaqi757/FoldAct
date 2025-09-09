import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional
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

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation


        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))


        self.search_tool=SearchTool(config=config,tool_schema=TOOL_SCHEMA)

    def _batch_tokenize(self, responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of responses."""
        result = self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest",
            return_offsets_mapping=True
        )
        return result['input_ids'], result['offset_mapping'][:, :, 0]

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

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        result = self.tokenizer(
            next_obs, 
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )

        next_obs_ids = result['input_ids']
        information_types = result['attention_mask'] * ResponseType.information.value

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids, information_types

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
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
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
        """Run main LLM generation loop."""
        
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
                responses_str, self.tokenizer.pad_token, active_mask
            )
            action_types = self._build_action_types(responses_ids, responses_offsets, action_ranges, is_search, responses_str)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids, information_types = self._process_next_obs(next_obs)
            responses_types = torch.cat([action_types, information_types], dim=1)

            # Update states
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
                responses_str, self.tokenizer.pad_token, active_mask, do_search=do_search_final
            )
            responses_types = self._build_action_types(responses_ids, responses_offsets, action_ranges, is_search, responses_str)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            if do_search_final:
                next_obs_ids = self._process_next_obs(next_obs)
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
                last_nonzero_idx = (responses_offsets[i] != 0).nonzero(as_tuple=True)[0][-1]
                valid_offsets = responses_offsets[i][:last_nonzero_idx+1]

                # Select the smallest range of tokens to FULLY contain the action substring
                # If s is in the middle of token i, then include token i
                # If e is in the middle of token i, then use token i + 1 as the right bound instead
                s, e = action_ranges[i]
                s_idx = torch.searchsorted(valid_offsets, s, right=True) - 1
                e_idx = torch.searchsorted(valid_offsets, e)

                types[i, s_idx:e_idx] = ResponseType.search.value if is_search[i] \
                    else ResponseType.answer.value

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

    def execute_predictions(self,predictions:List[str],pad_token:str,active_mask=None,do_search=True) \
        -> Tuple[List[str], List[int], List[int], List[int], List[Optional[Tuple[int, int]]]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
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
                    search_results += [''] * (expected_results - len(search_results))
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
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
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
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                    action_range = (match.start(), match.end())
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