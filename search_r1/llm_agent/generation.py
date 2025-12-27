import torch
import json
import json5  # For parsing tool_call JSON (supports comments and trailing commas)
import uuid
import re
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import quote
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
from verl.utils.event_ledger import EventLedger, EventType
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
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries."
                },
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries (legacy format, use 'query' instead)"
                },
                "topk": {
                    "type": "integer",
                    "description": "Number of top results to return"
                }
            },
            "required": []
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
    sandbox_fusion_url: str = "http://10.200.14.82:10080/run_code"
    topk: int = 3
    retriever_num_workers:int = 5 
    retriever_rate_limit:int = 120 
    retriever_timeout:int = 30 
    retriever_enable_global_rate_limit:bool = True 
    python_num_workers: int = 5
    python_rate_limit: int = 10
    python_timeout: int = 30
    # Maximum model context length (for sglang/vllm engines)
    # Increased to 16384 to accommodate longer tool responses (Jina search returns 5-10x longer)
    max_model_len: int = 16384
    # Enable sliding window context: keep only the most recent N turns (0 = keep all)
    use_summary: bool = False
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

        # Initialize search tool
        # Check if JINA_API_KEYS is available - if so, use Jina directly (matching react_agent.py)
        # Note: In training scripts (like train_grpo_asearcher_slurm-7B.sh), we might explicitly unset JINA_API_KEYS
        # to force using the local retriever. This check respects that environment variable state.
        self.use_jina_search = bool(os.environ.get('JINA_API_KEYS', ''))
        
        if self.use_jina_search:
            logger.info("JINA_API_KEYS detected - will use Jina API for search (matching react_agent.py)")
            # Create a Jina-compatible search wrapper
            self.search_tool = None  # Will use direct Jina calls in execute_predictions
        else:
            logger.info("JINA_API_KEYS not found - will use internal retriever service")
            # Convert GenerationConfig to dict format expected by SearchTool
            search_config = type('Config', (), {
                'search_url': getattr(config, 'search_url', ''),
                'topk': getattr(config, 'topk', 3),
                'retriever_num_workers': getattr(config, 'retriever_num_workers', 5),
                'retriever_rate_limit': getattr(config, 'retriever_rate_limit', 120),
                'retriever_timeout': getattr(config, 'retriever_timeout', 30),
                'retriever_enable_global_rate_limit': getattr(config, 'retriever_enable_global_rate_limit', True),
                'retriever_max_query_words': getattr(config, 'retriever_max_query_words', 48),
            })()
            self.search_tool = SearchTool(config=search_config, tool_schema=TOOL_SCHEMA)
        
        # Initialize visit tool (if visit_url is available)
        if hasattr(config, 'search_url') and config.search_url:
            from verl.tools.visit_tool import VisitTool
            from verl.tools.schemas import OpenAIFunctionToolSchema
            
            visit_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "visit",
                    "description": "Visit webpage(s) and return the summary of the content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The URL(s) of the webpage(s) to visit."
                            },
                            "goal": {
                                "type": "string",
                                "description": "The specific information goal for visiting webpage(s)."
                            }
                        },
                        "required": ["url"]
                    }
                }
            )
            visit_config = {
                "visit_url": config.search_url.replace("/retrieve", "/access") if config.search_url else None,
                "retriever_timeout": getattr(config, 'retriever_timeout', 30)
            }
            self.visit_tool = VisitTool(config=visit_config, tool_schema=visit_schema)
        else:
            self.visit_tool = None
            
        # Initialize Python interpreter tool (if sandbox_fusion_url is available)
        if hasattr(config, 'sandbox_fusion_url') and config.sandbox_fusion_url:
            from verl.tools.sandbox_fusion_tools import SandboxFusionTool
            from verl.tools.schemas import OpenAIFunctionToolSchema
            
            python_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "PythonInterpreter",
                    "description": "Executes Python code in a sandboxed environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute."
                            }
                        },
                        "required": ["code"]
                    }
                }
            )
            python_config = {
                "sandbox_fusion_url": config.sandbox_fusion_url,
                "num_workers": getattr(config, 'python_num_workers', 5),
                "rate_limit": getattr(config, 'python_rate_limit', 10),
                "default_timeout": getattr(config, 'python_timeout', 30),
                "default_language": "python",
                "enable_global_rate_limit": True
            }
            try:
                self.python_tool = SandboxFusionTool(config=python_config, tool_schema=python_schema)
            except Exception as e:
                logger.warning(f"Failed to initialize PythonInterpreter tool: {e}")
                self.python_tool = None
        else:
            self.python_tool = None

        # Initialize Google Scholar tool
        if hasattr(config, 'search_url') and config.search_url:
            scholar_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "google_scholar",
                    "description": "Leverage Google Scholar to retrieve relevant information from academic publications.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The search queries for Google Scholar."
                            }
                        },
                        "required": ["query"]
                    }
                }
            )
            scholar_config = type('Config', (), {
                'search_url': config.search_url.replace("/retrieve", "/scholar"),
                'topk': getattr(config, 'topk', 3),
                'retriever_num_workers': getattr(config, 'retriever_num_workers', 5),
                'retriever_rate_limit': getattr(config, 'retriever_rate_limit', 120),
                'retriever_timeout': getattr(config, 'retriever_timeout', 30),
                'retriever_enable_global_rate_limit': getattr(config, 'retriever_enable_global_rate_limit', True),
            })()
            self.scholar_tool = SearchTool(config=scholar_config, tool_schema=scholar_schema)
        else:
            self.scholar_tool = None

        # Initialize File Parser tool
        # For now, we'll use a placeholder or check if a file tool exists
        self.file_tool = None # Placeholder for future implementation

    def _check_turn_has_summary(self, turn: Dict[str, torch.Tensor]) -> bool:
        """
        Check if a turn's response contains summary tags (think_summary or information_summary).
        
        Uses responses_types to accurately detect summary tokens instead of decoding text,
        which avoids false positives from prompt text that mentions summary tags.
        
        Args:
            turn: Turn dictionary containing 'responses' and 'responses_types' tensors
            
        Returns:
            True if the turn contains summary tags, False otherwise
        """
        if 'responses' not in turn:
            return False
        
        responses = turn['responses']
        if responses.shape[1] == 0:
            return False
        
        # Use responses_types if available (more accurate than text decoding)
        if 'responses_types' in turn:
            responses_types = turn['responses_types']
            response_len = responses.shape[1]
            if responses_types.shape[1] > 0 and response_len > 0:
                # responses_types has shape [batch, seq_len] for this turn.
                # Only the first `response_len` positions correspond to the model's response
                # (action tokens); the tail part corresponds to information tokens.
                # We also want the decision to be batch-agnostic, so we aggregate over
                # ALL trajectories instead of looking only at index 0.
                types = responses_types[:, :response_len]  # Shape: [batch, response_len]
                has_info_summary = (types == ResponseType.information_summary.value).any().item()
                has_think_summary = (types == ResponseType.think_summary.value).any().item()
                return has_info_summary or has_think_summary
        

    def _apply_sliding_window(self, turn_history: List[Dict[str, torch.Tensor]], 
                            initial_question: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive sliding window based on whether the last turn has information.
        
        Logic:
        - If last turn has information: apply aggressive truncation (keep only recent turns)
        - If last turn has no information: keep more context (less aggressive truncation)
        - If all turns are info turns: preserve context
        
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
        has_info_flags = []
        for i, turn in enumerate(turn_history):
            response_len = turn['responses'].shape[1]
            obs_len = turn.get('observations', torch.tensor([])).shape[1] if turn.get('observations') is not None else 0
            total_len = response_len + obs_len

            has_information = bool(turn.get('has_real_info', obs_len > 20))

            turn_analysis.append({
                'index': i,
                'response_len': response_len,
                'obs_len': obs_len,
                'total_len': total_len,
                'has_information': has_information,
            })

            has_info_flags.append(has_information)
            if getattr(self.config, 'enable_debug_logs', False):
                print(
                    f"[CONTEXT SUMMARY] Turn {i} has_info_tag={has_information} "
                    f"(response_len={response_len}, obs_len={obs_len})"
                )

        # ============================================================================
        # STEP 1: Find the last turn that has information (search backwards)
        # ============================================================================
        last_info_idx = -1
        for i in range(len(turn_history) - 1, -1, -1):
            if has_info_flags[i]:
                last_info_idx = i
                break

        # ============================================================================
        # STEP 2: Select which turns to keep (unified strategy)
        # ============================================================================
        if last_info_idx == -1:
            # No information turns found → keep all turns
            selected_turns = turn_history
            if getattr(self.config, 'enable_debug_logs', False):
                print(f"[CONTEXT SUMMARY] No <information> found in history → keeping ALL turns")
        else:
            # Unified strategy: Try to find summary first, fallback to contiguous non-info turns
            # Step 1: Search backwards from last_info_idx for the most recent summary turn
            most_recent_summary_idx = None
            for i in range(last_info_idx, -1, -1):
                if self._check_turn_has_summary(turn_history[i]):
                    most_recent_summary_idx = i
                    break
            
            if most_recent_summary_idx is not None:
                # Found summary → keep from summary turn to last info turn
                selected_turns = turn_history[most_recent_summary_idx:last_info_idx + 1]
                if getattr(self.config, 'enable_debug_logs', False):
                    if most_recent_summary_idx == last_info_idx:
                        print(f"[CONTEXT SUMMARY] Keeping only last turn [{last_info_idx}] (has summary)")
                    else:
                        print(f"[CONTEXT SUMMARY] Keeping from summary turn [{most_recent_summary_idx}] to last turn [{last_info_idx}]")
            else:
                # No summary found → keep all turns (no compression)
                selected_turns = turn_history
                if getattr(self.config, 'enable_debug_logs', False):
                    print(f"[CONTEXT SUMMARY] No summary found → keeping ALL turns")
        
        # CRITICAL FIX: Add length check to prevent exceeding max_model_len
        max_model_len = getattr(self.config, 'max_model_len', 16384)
        max_allowed_length = max_model_len - 1000  # Leave room for response generation
        
        # Calculate current length
        current_length = initial_question.shape[1]
        for turn in selected_turns:
            current_length += turn['responses'].shape[1]
            if turn.get('observations') is not None:
                current_length += turn['observations'].shape[1]
        
        # If still too long, apply aggressive truncation
        if current_length > max_allowed_length:
            if getattr(self.config, 'enable_debug_logs', False):
                print(f"[CONTEXT SUMMARY] ⚠️ Context too long ({current_length} > {max_allowed_length}), applying aggressive truncation")
            
            # Keep only the most recent turns to fit within limit
            truncated_turns = []
            accumulated_length = initial_question.shape[1]
            
            # Start from the most recent turns and work backwards
            for i in range(len(selected_turns) - 1, -1, -1):
                turn = selected_turns[i]
                turn_length = turn['responses'].shape[1]
                if turn.get('observations') is not None:
                    turn_length += turn['observations'].shape[1]
                
                if accumulated_length + turn_length <= max_allowed_length:
                    truncated_turns.insert(0, turn)  # Insert at beginning to maintain order
                    accumulated_length += turn_length
                else:
                    if getattr(self.config, 'enable_debug_logs', False):
                        print(f"[CONTEXT SUMMARY] Stopping at turn {i} to stay within length limit")
                    break
            
            selected_turns = truncated_turns
            if getattr(self.config, 'enable_debug_logs', False):
                print(f"[CONTEXT SUMMARY] Aggressive truncation: kept {len(selected_turns)} turns, estimated length: {accumulated_length}")
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
        """Process responses to stop at tool call, search operation, or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Support both old format (<search>) and new format (<tool_call>)
        processed_responses = []
        for resp in responses_str:
            # Check for new tool_call format first
            if '</tool_call>' in resp:
                # Stop at tool_call end
                resp = resp.split('</tool_call>')[0] + '</tool_call>'
            elif '</search>' in resp:
                # Legacy format: stop at search end
                resp = resp.split('</search>')[0] + '</search>'
            elif '</answer>' in resp:
                # Stop at answer end
                resp = resp.split('</answer>')[0] + '</answer>'
            processed_responses.append(resp)
        
        responses_str = processed_responses

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            # The downstream no_think_rl path is not supported in this workflow.
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
            #  Mark info tokens as ResponseType.information (3) for correct mask computation
            # This ensures that information blocks are correctly identified and excluded from policy gradient updates
            info_types = torch.full(info.size(), ResponseType.information.value, dtype=response_types.dtype, device=response_types.device)
            type_tensors.append(info_types)
        
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
        # CRITICAL FIX: Check input length before generation - use simple truncation
        max_model_len = getattr(self.config, 'max_model_len', 16384)
        max_allowed_prompt_len = max_model_len - 1000  # Leave room for response generation

        # Detect whether any sample exceeds allowed length
        need_truncation = any(
            seq.shape[0] > max_allowed_prompt_len for seq in active_batch.batch['input_ids']
        )

        if need_truncation:
            truncated_batch = {}
            for key, tensor in active_batch.batch.items():
                if tensor.ndim < 2:
                    truncated_batch[key] = tensor
                    continue
                seq_len = tensor.shape[1]
                if seq_len <= max_allowed_prompt_len:
                    truncated_batch[key] = tensor.clone()
                else:
                    truncated_batch[key] = tensor[:, -max_allowed_prompt_len:].clone()

            active_batch = DataProto.from_dict(truncated_batch)
        
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
    
    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> DataProto:
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
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Track history for sliding window if enabled
        use_summary = getattr(self.config, "use_summary", False)
        
        # Initialize turn_history and initial_question_ids regardless of use_summary
        initial_question_ids = gen_batch.batch['input_ids'].clone()
        turn_history: List[Dict[str, torch.Tensor]] = []
        
        # Initialize event ledgers for each example in batch (FULL CONTEXT TRACKING)
        batch_size = gen_batch.batch['input_ids'].shape[0]
        event_ledgers = [EventLedger(trajectory_id=f"traj_{i}") for i in range(batch_size)]
        logger.info(f"[Event Ledger] Initialized {batch_size} event ledgers for full context tracking")
        
        # Track if observation is too long (forces summarized context)
        force_summarized = False
        
        # Context length monitoring
        context_lengths_per_turn = []  # Track context length at each turn

        # ========== PER-TURN CONTEXT SAVING ==========
        # Save context for each turn to enable correct log_prob computation during training
        # Key insight: Each turn's response should be trained with the context it actually saw
        per_turn_contexts = [[] for _ in range(batch_size)]  # List of contexts for each trajectory
        logger.info("[Per-Turn Context] Initialized per-turn context tracking")

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
            
            # ========== SAVE PER-TURN CONTEXT (BEFORE GENERATION) ==========
            # This is the context that the model will actually see for this turn
            # We save it BEFORE generation to ensure we capture the exact state
            
            # Save context for each active trajectory (ROBUST SELF-CONTAINED DATA)
            for i in range(batch_size):
                if active_mask[i]:
                    # ROOT CAUSE ANALYSIS: An active trajectory's mask might still be all padding.
                    # We must validate the content of the mask AND input_ids BEFORE saving the context.
                    current_mask = rollings.batch['attention_mask'][i]
                    current_input_ids = rollings.batch['input_ids'][i]
                    mask_sum = current_mask.sum().item()
                    
                    # Check if input_ids contains real tokens (not just padding)
                    # Exclude both 0 and padding token (151643)
                    real_tokens = ((current_input_ids != 0) & (current_input_ids != 151643)).sum().item()
                    padding_tokens = (current_input_ids == 151643).sum().item()

                    if mask_sum == 0:
                        logger.warning(
                            f"[Per-Turn Context] Turn {step}, Traj {i}: SKIPPING context save. "
                            f"Trajectory is marked 'active', but its attention mask sum is 0. "
                            f"This is the source of downstream errors."
                        )
                        continue  # Do not save this invalid context
                    
                    # CRITICAL: Also check if input_ids contains only padding tokens
                    if real_tokens == 0:
                        logger.warning(
                            f"[Per-Turn Context] Turn {step}, Traj {i}: SKIPPING context save. "
                            f"input_ids contains NO real tokens (only padding). "
                            f"real_tokens=0, padding_tokens={padding_tokens}, mask_sum={mask_sum}. "
                            f"This will cause zero log_probs downstream."
                        )
                        continue  # Do not save this invalid context

                    # Get the starting position_id from the current position_ids
                    current_position_ids = rollings.batch.get('position_ids', None)
                    if current_position_ids is not None:
                        # For Qwen2VL, position_ids might be (3, batch_size, seq_len) or (batch_size, seq_len)
                        if current_position_ids.dim() == 3:
                            # Qwen2VL mrope case: (3, batch_size, seq_len), use first channel
                            context_start_position = current_position_ids[0, i, 0].item() if current_position_ids.size(2) > 0 else 0
                        else:
                            # Standard case: (batch_size, seq_len)
                            context_start_position = current_position_ids[i, 0].item() if current_position_ids.size(1) > 0 else 0
                    else:
                        context_start_position = 0
                        logger.warning(f"[Per-Turn Context] Turn {step}: position_ids not found, using context_start_position=0")
                    
                    # CRITICAL REFACTOR: Clone individual tensors instead of referencing the whole batch
                    input_ids_clone = rollings.batch['input_ids'][i].clone()
                    attention_mask_clone = current_mask.clone()
                    
                    # INTEGRITY CHECK: "Timestamp" the mask sum at the moment of creation
                    mask_sum_at_creation = mask_sum # Use the sum we already computed
                    
                    per_turn_contexts[i].append({
                        'turn_id': step,
                        'input_ids': input_ids_clone,  # Self-contained tensor
                        'attention_mask': attention_mask_clone,  # Self-contained tensor
                        'context_length': mask_sum, # Use pre-computed sum
                        'context_start_position': context_start_position,
                        'mask_sum_at_creation': mask_sum_at_creation # Store for downstream verification
                    })
                    
                    # DEBUGGING: Log the shape of the saved context for the first few turns/trajectories
                    if step < 2 and i < 3:
                        logger.warning(
                            f"[Per-Turn Context DEBUG] Turn {step}, Traj {i}: Saved self-contained context. "
                            f"input_ids shape: {input_ids_clone.shape}, "
                            f"attention_mask sum: {attention_mask_clone.sum().item()}"
                        )
            
            if step == 0 or step % 3 == 0:  # Log every 3 turns
                avg_context_len = sum(rollings.batch['attention_mask'][i].sum().item() 
                                     for i in range(batch_size) if active_mask[i]) / max(active_mask.sum().item(), 1)
                logger.info(f"[Per-Turn Context] Turn {step}: Saved context for {active_mask.sum().item()} trajectories (avg len: {avg_context_len:.0f})")
                
             
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

            # ========== SAVE RESPONSE INFO TO PER-TURN CONTEXT ==========
            # After generating response, add it to the saved context for this turn
            for i in range(batch_size):
                if active_mask[i] and len(per_turn_contexts[i]) > 0:
                    # Get the last saved context (for this turn)
                    turn_ctx = per_turn_contexts[i][-1]
                    turn_ctx['response'] = responses_ids[i].clone()
                    turn_ctx['response_length'] = self.tensor_fn.create_attention_mask(responses_ids[i]).sum().item()
                    # Assertions to ensure integrity of per-turn saved data (UPDATED FOR SELF-CONTAINED DATA)
                    assert 'input_ids' in turn_ctx and 'attention_mask' in turn_ctx, "Per-turn context (self-contained) missing"
                    assert isinstance(turn_ctx['response_length'], (int, float)), "response_length must be numeric"
               
            # Execute in environment and process observations
            next_obs, dones, valid_action = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, 
                current_turn=step, max_turns=self.config.max_turns
            )
            action_types = self._build_action_types(responses_ids, responses_offsets, responses_str)
            # Assertions: shapes alignment
            
            # ========== RECORD EVENTS IN LEDGER (FULL CONTEXT) ==========
            # Record actions and observations for each trajectory using action_types (no extra parsing)
            def _summarize_action_from_types(row_types: torch.Tensor) -> Tuple[str, Optional[Tuple[int,int]]]:
                """Infer a single action and a token-span from token-wise types.
                Priority: search (flag) > answer > information_summary > think_summary > think.
                """
                # For non-search actions, check for search tokens first (they override other types)
                search_sel = (row_types == ResponseType.search.value).nonzero(as_tuple=True)[0]
                if search_sel.numel() > 0:
                    return 'search', (int(search_sel.min().item()), int(search_sel.max().item() + 1))
                
                # Non-search priorities by token presence
                priorities = [
                    ('answer', ResponseType.answer.value),
                    ('information_summary', ResponseType.information_summary.value),
                    ('think_summary', ResponseType.think_summary.value),
                ]
                for name, val in priorities:
                    sel = (row_types == val).nonzero(as_tuple=True)[0]
                    if sel.numel() > 0:
                        return name, (int(sel.min().item()), int(sel.max().item() + 1))
                # Default think
                sel = (row_types == ResponseType.think.value).nonzero(as_tuple=True)[0]
                if sel.numel() > 0:
                    return 'think', (int(sel.min().item()), int(sel.max().item() + 1))
                return 'think', None

            for i, (obs, row_types) in enumerate(zip(next_obs, action_types)):
                if active_mask[i]:  # Only record for active trajectories
                    ledger = event_ledgers[i]
                    action, span = _summarize_action_from_types(row_types)
                    # Extract content from pre-decoded strings (EFFICIENT)
                    if span is not None:
                        s_tok, e_tok = span
                        try:
                            # Use character positions from the original response string
                            response_text = responses_str[i] if i < len(responses_str) else ""
                            # Map token positions back to character positions using offsets
                            s_char, e_char = self._token_to_char_range(response_text, s_tok, e_tok, responses_offsets[i])
                            if s_char is not None and e_char is not None:
                                content = response_text[s_char:e_char]
                            else:
                                content = ""
                        except Exception:
                            content = ''
                    else:
                        content = ''
                    # Record the action taken
                    if action == 'search':
                        ledger.record_search(step, content, metadata={'valid': valid_action[i]})
                    elif action == 'answer':
                        ledger.record_answer(step, content)
                    elif action == 'think_summary':
                        ledger.record_summary(step, 'think_summary', content)
                    elif action == 'information_summary':
                        ledger.record_summary(step, 'information_summary', content)
                    elif action == 'think':
                        ledger.record_event(step, EventType.THINK, content)
                    
                    # Record environment-provided information
                    if obs:
                        # Extract actual information content (remove turn markers)
                        info_content = obs
                        ledger.record_information(step, info_content, source='environment',
                                                 metadata={'search_query': content})
            
            if getattr(self.config, 'enable_debug_logs', False):
                logger.debug(f"[Event Ledger] Turn {step}: Recorded {sum(len(l.get_events_at_turn(step)) for l in event_ledgers)} events")
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)

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

            try:
                valid_action_tensor = torch.tensor(valid_action, dtype=torch.bool)
                has_valid_action = valid_action_tensor.any().item()
            except Exception:
                # Fallback: if something goes wrong, be conservative and reuse old logic
                has_valid_action = True

            obs_token_len = int(next_obs_ids.shape[1]) if next_obs_ids.ndim == 2 else 0
            info_len_threshold = 20
            turn_has_real_info = bool(has_valid_action and (obs_token_len > info_len_threshold))


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
                    "context_type": "compressed",
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
            if use_summary:
                # Store current turn and rebuild rolling context using sliding window
                turn_history.append({
                    'responses': responses_ids.clone(),
                    'observations': next_obs_ids.clone(),
                    'responses_types': responses_types.clone(),
                    # Compact per-turn flag for sliding window:
                    # True  -> this turn carries real information (used in has_info_flags)
                    # False -> treated as non-info turn in context compression
                    'has_real_info': turn_has_real_info,
                })
                
                if force_summarized:
                    # Observation too long: keep only last 2-3 turns
                    num_turns_to_keep = min(3, len(turn_history))
                    truncated_history = turn_history[-num_turns_to_keep:] if turn_history else []
                    print(f"[CONTEXT DECISION] DECISION: FORCED summarized context (keeping last {num_turns_to_keep} turns)")
                    windowed_input_ids = self._apply_sliding_window(truncated_history, initial_question_ids)
                    if step == 0 or step == self.config.max_turns - 1:
                        logger.info(f"Turn {step}: FORCED summarized context (observation too long)")
                else:
                    # Use compressed context (sliding window)
                    windowed_input_ids = self._apply_sliding_window(turn_history, initial_question_ids)
                   
                print(f"{'='*60}\n")
                
                rollings.batch['input_ids'] = windowed_input_ids
                rollings.batch['attention_mask'] = self.tensor_fn.create_attention_mask(windowed_input_ids)
                rollings.batch['position_ids'] = self.tensor_fn.create_position_ids(rollings.batch['attention_mask'])
                
                
                # Monitor context length (after context decision is made)
                current_context_length = windowed_input_ids.shape[1]
                context_lengths_per_turn.append(current_context_length)
            else:
                rollings = self._update_rolling_state(
                    rollings,
                    responses_ids,
                    next_obs_ids
                )
                
                # Monitor context length for non-sliding window mode
                current_context_length = rollings.batch['input_ids'].shape[1]
                context_lengths_per_turn.append(current_context_length)
                print(f"[NON-SLIDING] Context length: {current_context_length} tokens")
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
            next_obs, dones, valid_action = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask,
                current_turn=self.config.max_turns, max_turns=self.config.max_turns
            )
            responses_types = self._build_action_types(responses_ids, responses_offsets, responses_str)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                self.config.max_turns,
                responses_types
        )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        
        meta_info['context_type'] = "compressed"
        
        # ========== SAVE EVENT LEDGERS (FULL CONTEXT) ==========
        # CRITICAL: Save event ledgers in BOTH meta_info AND as individual items for batch processing
        # meta_info: for logging and debugging
        # Will be added to non_tensor_batch in compose_final_output for per-item access
        meta_info['event_ledgers_serialized'] = [ledger.to_dict() for ledger in event_ledgers]
        logger.info(f"[Event Ledger] Saved {len(event_ledgers)} ledgers with total {sum(len(l) for l in event_ledgers)} events")
        
        # ========== SAVE PER-TURN CONTEXTS ==========
        # This is the CORE FIX for context mismatch problem
        # Each turn's response should be trained with the context it actually saw
        meta_info['per_turn_contexts'] = per_turn_contexts
        total_turns = sum(len(contexts) for contexts in per_turn_contexts)
        logger.info(f"[Per-Turn Context] Saved {total_turns} turn contexts across {batch_size} trajectories")
        
        # HYPOTHESIS VERIFICATION: Add a "fingerprint" of the attention_mask sums BEFORE serialization
        try:
            mask_sums_fingerprint = []
            for traj_contexts in per_turn_contexts:
                traj_sums = [turn['attention_mask'].sum().item() for turn in traj_contexts]
                mask_sums_fingerprint.append(traj_sums)
            meta_info['mask_sums_fingerprint'] = mask_sums_fingerprint
            logger.critical("[VERIFICATION] Added 'mask_sums_fingerprint' to meta_info for serialization check.")
        except Exception as e:
            logger.error(f"[VERIFICATION] Failed to create mask_sums_fingerprint: {e}")

        
        # Context Length Monitoring: Add statistics for WandB tracking
        if context_lengths_per_turn:
            final_context_length = context_lengths_per_turn[-1]
            avg_context_length = sum(context_lengths_per_turn) / len(context_lengths_per_turn)
            meta_info['final_context_length'] = final_context_length
            meta_info['avg_context_length'] = avg_context_length
            
            # Calculate compression ratio using turn_history and event_ledgers
            if use_summary and turn_history:
                # Calculate full context length (all turns without compression)
                full_context_tokens: List[torch.Tensor] = [initial_question_ids.clone()]
                for turn in turn_history:
                    full_context_tokens.append(turn['responses'])
                    if turn.get('observations') is not None:
                        full_context_tokens.append(turn['observations'])
                full_context_ids = self.tensor_fn.concatenate_with_padding(full_context_tokens)
                full_context_length = full_context_ids.shape[1]
                
                # Calculate compression statistics
                tokens_saved = full_context_length - final_context_length
                compression_ratio = tokens_saved / full_context_length if full_context_length > 0 else 0.0
                
                meta_info['compression_stats'] = {
                    'full_context_length': int(full_context_length),
                    'compressed_context_length': int(final_context_length),
                    'tokens_saved': int(tokens_saved),
                    'compression_ratio': float(compression_ratio)
                }
             
            else:
                # No compression (non-sliding window or no history)
                meta_info['compression_stats'] = {
                    'full_context_length': int(final_context_length),
                    'compressed_context_length': int(final_context_length),
                    'tokens_saved': 0,
                    'compression_ratio': 0.0
                }
                
        else:
            # Fallback when context_lengths_per_turn is empty
            final_context_length = original_right_side['responses'].size(1) if 'responses' in original_right_side else 0
            meta_info['final_context_length'] = final_context_length
            meta_info['avg_context_length'] = final_context_length
            
            # Try to calculate compression ratio from turn_history if available
            if use_summary and turn_history:
                full_context_tokens: List[torch.Tensor] = [initial_question_ids.clone()]
                for turn in turn_history:
                    full_context_tokens.append(turn['responses'])
                    if turn.get('observations') is not None:
                        full_context_tokens.append(turn['observations'])
                full_context_ids = self.tensor_fn.concatenate_with_padding(full_context_tokens)
                full_context_length = full_context_ids.shape[1]
                
                tokens_saved = full_context_length - final_context_length
                compression_ratio = tokens_saved / full_context_length if full_context_length > 0 else 0.0
                
                meta_info['compression_stats'] = {
                    'full_context_length': int(full_context_length),
                    'compressed_context_length': int(final_context_length),
                    'tokens_saved': int(tokens_saved),
                    'compression_ratio': float(compression_ratio)
                }
            else:
                meta_info['compression_stats'] = {
                    'full_context_length': int(final_context_length),
                    'compressed_context_length': int(final_context_length),
                    'tokens_saved': 0,
                    'compression_ratio': 0.0
                }
            
        
        
        # Compose final output using _compose_final_output
        # The fingerprint has been added to meta_info above
        final_output = self._compose_final_output(
            left_side=original_left_side,
            right_side=original_right_side,
            meta_info=meta_info
        )
        return final_output

    def _build_action_types(self,
                            responses_ids: torch.Tensor,
                            responses_offsets: torch.Tensor,
                            responses_str: List[str]) -> torch.Tensor:
        """
        Build action types by directly parsing response strings.
        This is the single source of truth for action type determination.
        Now supports multiple action types within a single response.
        
        Args:
            responses_ids: Token IDs for responses [batch_size, seq_len]
            responses_offsets: Character offsets for each token [batch_size, seq_len]
            responses_str: Raw response strings [batch_size]
            
        Returns:
            types: Token-level action types [batch_size, seq_len]
        """
        # The type is think by default
        types = torch.zeros_like(responses_ids)
        
        # Regex patterns for different action types
        # Support both old format (<search>) and new format (<tool_call>)
        # Note: All tool calls (search, visit, python, etc.) are mapped to ResponseType.search
        # for now, as they represent tool usage actions
        action_patterns = [
            # New tool_call format - match any tool call (search, visit, python, etc.)
            (r'<tool_call>\s*\{[^}]*"name"\s*:\s*"(?:search|visit|PythonInterpreter|google_scholar|parse_file)"[^}]*\}\s*</tool_call>', ResponseType.search.value),
            # Old format tags
            (r'<\s*search\b[^>]*>(.*?)</\s*search\s*>', ResponseType.search.value),
            (r'<\s*answer\b[^>]*>(.*?)</\s*answer\s*>', ResponseType.answer.value),
            (r'<\s*think_summary\b[^>]*>(.*?)</\s*think_summary\s*>', ResponseType.think_summary.value),
            (r'<\s*information_summary\b[^>]*>(.*?)</\s*information_summary\s*>', ResponseType.information_summary.value),
            (r'<\s*think\b[^>]*>(.*?)</\s*think\s*>', ResponseType.think.value),
        ]
        
        for pattern, response_type in action_patterns:
            type_name = ResponseType(response_type).name
        
        for i in range(len(responses_ids)):
            response_text = responses_str[i] if i < len(responses_str) else ""
           
            found_any_match = False

            for pattern_idx, (pattern, response_type) in enumerate(action_patterns):
                type_name = ResponseType(response_type).name

                # Find all matches of this pattern
                matches = list(re.finditer(pattern, response_text, re.IGNORECASE | re.DOTALL))

                for match_idx, match in enumerate(matches):
                    s, e = match.span()
                    # Map character positions to token positions
                    s_idx, e_idx = self._char_to_token_range(response_text, s, e, responses_offsets[i])
                    if s_idx is not None and e_idx is not None:
                        types[i, s_idx:e_idx] = response_type
                        found_any_match = True
                    else:
                        logger.warning(f"[_build_action_types]       Failed to map char range to token range for {type_name}")

            # If no action tags found, default to think
            if not found_any_match:
                types[i, :] = ResponseType.think.value

        return types
    def _char_to_token_range(self, text: str, char_start: int, char_end: int, offsets: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        """
        Convert character positions to token positions using offsets.
        
        Args:
            text: The full text
            char_start: Start character position
            char_end: End character position  
            offsets: Token offsets [seq_len]
            
        Returns:
            (token_start, token_end) or (None, None) if conversion fails
        """
        # Keep only until the last non-zero index for searchsorted() to work
        nonzero_idx = (offsets != 0).nonzero(as_tuple=True)[0]
        if nonzero_idx.numel() == 0:
            return None, None
            
        last_nonzero_idx = nonzero_idx[-1]
        valid_offsets = offsets[:last_nonzero_idx+1]
        
        # Find token indices that contain the character range
        # If char_start is in the middle of token i, include token i
        # If char_end is in the middle of token i, use token i + 1 as the right bound
        s_idx = torch.searchsorted(valid_offsets, char_start, right=True) - 1
        e_idx = torch.searchsorted(valid_offsets, char_end)
        
        # Ensure indices are within bounds
        s_idx = max(0, min(s_idx.item(), len(valid_offsets) - 1))
        e_idx = max(0, min(e_idx.item(), len(valid_offsets)))
        
        return s_idx, e_idx
    
    def _token_to_char_range(self, text: str, token_start: int, token_end: int, offsets: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        """
        Convert token positions to character positions using offsets.
        
        Args:
            text: The full text
            token_start: Start token position
            token_end: End token position  
            offsets: Token offsets [seq_len]
            
        Returns:
            (char_start, char_end) or (None, None) if conversion fails
        """
        # Keep only until the last non-zero index for searchsorted() to work
        nonzero_idx = (offsets != 0).nonzero(as_tuple=True)[0]
        if nonzero_idx.numel() == 0:
            return None, None
            
        last_nonzero_idx = nonzero_idx[-1]
        valid_offsets = offsets[:last_nonzero_idx+1]
        
        # Ensure token indices are within bounds
        token_start = max(0, min(token_start, len(valid_offsets) - 1))
        token_end = max(token_start, min(token_end, len(valid_offsets)))
        
        # Get character positions from token offsets
        char_start = int(valid_offsets[token_start].item())
        char_end = int(valid_offsets[token_end].item()) if token_end < len(valid_offsets) else len(text)
        
        # Ensure character positions are within text bounds
        char_start = max(0, min(char_start, len(text)))
        char_end = max(char_start, min(char_end, len(text)))
        
        # Additional validation: ensure we don't go beyond text length
        if char_start >= len(text):
            return None, None
        if char_end > len(text):
            char_end = len(text)
        
        return char_start, char_end

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
        
        # Also preserve step_ids if available (used for debugging and validation)
        if 'step_ids' in right_side:
            final_output['step_ids'] = right_side['step_ids']
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        # CRITICAL FIX: Add event ledgers to non_tensor_batch for per-item access
        # Extract event ledgers from meta_info and add as numpy array
        if 'event_ledgers_serialized' in meta_info:
            import numpy as np
            # Store as object array to preserve dict structure
            final_output.non_tensor_batch['event_ledger'] = np.array(
                meta_info['event_ledgers_serialized'], 
                dtype=object
            )
            logger.info(f"[Event Ledger] Added {len(meta_info['event_ledgers_serialized'])} ledgers to non_tensor_batch")
        
        # ========== ADD PER-TURN CONTEXTS TO NON_TENSOR_BATCH ==========
        # This is the CORE FIX for context mismatch problem
        # Store per-turn contexts for correct log_prob computation during training
        if 'per_turn_contexts' in meta_info:
            import numpy as np
            final_output.non_tensor_batch['per_turn_contexts'] = np.array(
                meta_info['per_turn_contexts'],
                dtype=object
            )
            total_contexts = sum(len(contexts) for contexts in meta_info['per_turn_contexts'])
            logger.info(f"[Per-Turn Context] Added {total_contexts} turn contexts to non_tensor_batch for {len(meta_info['per_turn_contexts'])} trajectories")
        
        # HYPOTHESIS VERIFICATION: Add mask_sums_fingerprint to non_tensor_batch for verification
        if 'mask_sums_fingerprint' in meta_info:
            import numpy as np
            final_output.non_tensor_batch['mask_sums_fingerprint'] = np.array(
                meta_info['mask_sums_fingerprint'],
                dtype=object
            )
            logger.critical("[VERIFICATION] Added 'mask_sums_fingerprint' to non_tensor_batch for serialization check.")
        
        return final_output

    def _jina_search(self, query: str) -> str:
        """
        Execute Jina search directly (matching react_agent.py behavior).
        
        Args:
            query: Search query string
            
        Returns:
            Formatted search results string
        """
        jina_api_key = os.environ.get('JINA_API_KEYS', '')
        if not jina_api_key:
            return "[Search] JINA_API_KEYS not configured."
        
        max_retries = 10
        timeout = 30
        
        # Normalize query (remove extra whitespace)
        query = ' '.join(query.split()).strip()
        if not query:
            return "[Search] Empty query."
        
        for attempt in range(max_retries):
            try:
                encoded_query = quote(query)
                url = f"https://s.jina.ai/?q={encoded_query}"
                
                headers = {
                    "Authorization": f"Bearer {jina_api_key}",
                    "X-Respond-With": "no-content",
                    "X-With-Favicons": "true"
                }
                
                response = requests.get(url, headers=headers, timeout=timeout)
                
                if response.status_code == 200:
                    content = response.text
                    if not content or len(content.strip()) < 10:
                        raise ValueError("Empty response from Jina")
                    
                    # Parse Jina text format response
                    results_dict = {}
                    lines = content.split('\n')
                    current_result_num = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        if line.startswith('[') and ']' in line:
                            try:
                                end_bracket = line.index(']')
                                current_result_num = int(line[1:end_bracket])
                                if current_result_num not in results_dict:
                                    results_dict[current_result_num] = {'title': '', 'url': '', 'description': ''}
                                
                                field_part = line[end_bracket + 1:].strip()
                                if ':' in field_part:
                                    field_name = field_part.split(':', 1)[0].strip().lower()
                                    field_value = field_part.split(':', 1)[1].strip()
                                    
                                    if 'title' in field_name:
                                        results_dict[current_result_num]['title'] = field_value
                                    elif 'url' in field_name or 'source' in field_name:
                                        results_dict[current_result_num]['url'] = field_value
                                    elif 'description' in field_name:
                                        results_dict[current_result_num]['description'] = field_value
                            except (ValueError, IndexError):
                                continue
                        elif current_result_num is not None:
                            if line and current_result_num in results_dict:
                                if results_dict[current_result_num]['description']:
                                    results_dict[current_result_num]['description'] += ' ' + line
                                else:
                                    results_dict[current_result_num]['description'] = line
                    
                    # Format results
                    web_snippets = []
                    for result_num in sorted(results_dict.keys()):
                        result = results_dict[result_num]
                        title = result.get('title', 'No title').strip()
                        url = result.get('url', '').strip()
                        description = result.get('description', '').strip()
                        
                        if not title and not description:
                            continue
                        
                        if url:
                            formatted = f"{result_num}. [{title}]({url})"
                        else:
                            formatted = f"{result_num}. {title}"
                        
                        if description:
                            formatted += f"\n{description}"
                        
                        web_snippets.append(formatted)
                    
                    if web_snippets:
                        return f"A Jina search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
                    else:
                        return f"A Jina search for '{query}' returned:\n\n{content[:2000]}"
                
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = 2.0 * (2 ** attempt)
                        time.sleep(delay)
                        continue
                    return f"[Search] Jina API rate limit exceeded."
                else:
                    raise ValueError(f"Jina search returned status {response.status_code}")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2.0 * (2 ** attempt) + (time.time() % 1)
                    delay = min(delay, 60)
                    logger.warning(f"Jina search attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} Jina search attempts failed: {e}")
                    return f"Jina search failed after {max_retries} attempts: {str(e)}"
        
        return f"Jina search failed after {max_retries} attempts."

    def execute_predictions(self,predictions:List[str],pad_token:str,active_mask=None,current_turn:int=0,max_turns:int=None) \
        -> Tuple[List[str], List[int], List[int], List[Optional[Tuple[int, int]]]]:
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
            next_obs: List of observation strings
            dones: List[int]
            valid_action: List[int]
        """

        # Inline parsing of predictions into actions/contents/ranges (single source of truth)
        # Supports both old format (<search>) and new format (<tool_call> JSON)
        # Supports all tools: search, visit, PythonInterpreter, google_scholar, parse_file
        cur_actions = []
        contents = []
        
        # Pattern for old format tags
        tag_pattern = re.compile(
            r'<\s*(search|answer|think|think_summary|information_summary)\b[^>]*>(.*?)</\s*\1\s*>',
            re.IGNORECASE | re.DOTALL
        )
        # Pattern for new tool_call format
        tool_call_pattern = re.compile(
            r'<tool_call>\s*({.*?})\s*</tool_call>',
            re.IGNORECASE | re.DOTALL
        )
        
        for prediction in predictions:
            action = None
            content = ''
            if isinstance(prediction, str):
                found = False
                
                # First, try to parse new <tool_call> format
                # Use react_agent.py approach: extract everything between <tool_call> and </tool_call>
                if '<tool_call>' in prediction and '</tool_call>' in prediction:
                    tool_call_content = prediction.split('<tool_call>')[1].split('</tool_call>')[0]
                    
                    # Check if it's PythonInterpreter (has "python" in content)
                    if "python" in tool_call_content.lower():
                        try:
                            # Extract code from <code> tags
                            code_raw = tool_call_content.split('<code>')[1].split('</code>')[0].strip()
                            content = code_raw
                            action = 'python'
                            found = True
                        except Exception as e:
                            logger.warning(f"Failed to extract Python code from tool_call: {e}")
                    else:
                        # Parse as JSON for other tools
                        try:
                            tool_call_json = json5.loads(tool_call_content)
                            tool_name = tool_call_json.get('name', '').lower()
                            tool_args = tool_call_json.get('arguments', {})
                            
                            if tool_name == 'search':
                                # Extract query/queries from tool arguments
                                query_list = tool_args.get('query') or tool_args.get('query_list', [])
                                if isinstance(query_list, str):
                                    query_list = [query_list]
                                if query_list and len(query_list) > 0:
                                    # Use first query as content (for backward compatibility)
                                    content = query_list[0] if len(query_list) == 1 else ' '.join(query_list)
                                    action = 'search'
                                    found = True
                            elif tool_name == 'visit':
                                # Visit tool - extract URL and goal
                                url = tool_args.get('url') or tool_args.get('urls', [])
                                goal = tool_args.get('goal', '')
                                if isinstance(url, list) and len(url) > 0:
                                    content = f"{url[0]}|{goal}" if goal else url[0]
                                elif isinstance(url, str):
                                    content = f"{url}|{goal}" if goal else url
                                action = 'visit'
                                found = True
                            elif tool_name == 'google_scholar':
                                # Google Scholar - extract query/queries
                                query_list = tool_args.get('query', [])
                                if isinstance(query_list, str):
                                    query_list = [query_list]
                                if query_list and len(query_list) > 0:
                                    content = query_list[0] if len(query_list) == 1 else ' '.join(query_list)
                                    action = 'scholar'
                                    found = True
                            elif tool_name == 'parse_file':
                                # Parse file - extract file names
                                files = tool_args.get('files', [])
                                if isinstance(files, str):
                                    files = [files]
                                if files and len(files) > 0:
                                    content = files[0] if len(files) == 1 else ', '.join(files)
                                    action = 'parse_file'
                                    found = True
                        except (json5.JSONDecodeError, json.JSONDecodeError, Exception) as e:
                            logger.warning(f"Failed to parse tool_call JSON: {e}")
                            # Fall through to old format parsing
                
                # Fallback to old format if tool_call not found or failed
                if not found:
                    for match in tag_pattern.finditer(prediction):
                        tag = match.group(1).lower()
                        content_candidate = match.group(2).strip()
                        if tag == 'search':
                            # search 不能单纯靠字符串出现，内容需有效（>5字符）
                            if content_candidate and len(content_candidate) > 5:
                                action = 'search'
                                content = content_candidate
                                found = True
                                break  # 用第一个有效的 search
                            else:
                                continue
                        elif tag == 'answer':
                            action = 'answer'
                            content = content_candidate
                            found = True
                            break  # 用第一个 answer
                        elif tag in ['think', 'think_summary', 'information_summary']:
                            # 其他类型顺序考虑，但如果没有search/answer才接受
                            if not found:
                                action = tag
                                content = content_candidate
                # 若未找到任何指令，保持默认
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            cur_actions.append(action)
            contents.append(content)
        next_obs, dones, valid_action = [], [], []

        # Build tool execution lists for each tool type
        active_search_indices = [i for i, (action, active) in enumerate(zip(cur_actions, active_mask)) if active and action == 'search']
        active_visit_indices = [i for i, (action, active) in enumerate(zip(cur_actions, active_mask)) if active and action == 'visit']
        active_python_indices = [i for i, (action, active) in enumerate(zip(cur_actions, active_mask)) if active and action == 'python']
        active_scholar_indices = [i for i, (action, active) in enumerate(zip(cur_actions, active_mask)) if active and action == 'scholar']
        
        search_queries = [contents[i] for i in active_search_indices]
        visit_contents = [contents[i] for i in active_visit_indices]  # Format: "url|goal" or just "url"
        python_contents = [contents[i] for i in active_python_indices]  # Python code
        scholar_queries = [contents[i] for i in active_scholar_indices]
        
        expected_results = len(search_queries)
        expected_visit_results = len(active_visit_indices)
        expected_python_results = len(active_python_indices)
        expected_scholar_results = len(active_scholar_indices)

        # ========== TOOL PROCESSING DEBUG INFO ==========
        if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO):
            logger.info(f"[TOOL PROCESSING] Turn {current_turn}: Parsed {len(predictions)} predictions")
            logger.info(f"[TOOL PROCESSING] Actions: search={len(active_search_indices)}, visit={len(active_visit_indices)}, python={len(active_python_indices)}, scholar={len(active_scholar_indices)}")
        if search_queries:
            logger.info(f"[TOOL PROCESSING] Search queries: {[q[:50] + '...' if len(q) > 50 else q for q in search_queries]}")
        if python_contents:
            logger.info(f"[TOOL PROCESSING] Python code snippets: {[c[:50] + '...' if len(c) > 50 else c for c in python_contents]}")

        if search_queries:
            # Use Jina API if JINA_API_KEYS is available (matching react_agent.py behavior)
            if self.use_jina_search:
                # Execute Jina search directly for each query
                search_results = []
                jina_response_lengths = []
                for i, query in enumerate(search_queries):
                    start_time = time.time()
                    jina_result = self._jina_search(query)
                    elapsed_time = time.time() - start_time
                    result_length_chars = len(jina_result)
                    result_length_tokens = len(self.tokenizer.encode(jina_result, add_special_tokens=False))
                    jina_response_lengths.append(result_length_chars)
                    search_results.append(jina_result)
                    
                    # Debug logging for Jina search
                    if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO):
                        logger.info(
                            f"[TOOL PROCESSING] Jina Search #{i+1}/{len(search_queries)}: "
                            f"Query='{query[:60]}{'...' if len(query) > 60 else ''}' | "
                            f"Response: {result_length_chars:,} chars, {result_length_tokens:,} tokens | "
                            f"Time: {elapsed_time:.2f}s"
                        )
                        if result_length_tokens > 2000:
                            logger.warning(
                                f"[TOOL PROCESSING] ⚠️ Jina response is VERY LONG: {result_length_tokens:,} tokens "
                                f"(>2000). This is expected for Jina but may cause context length issues."
                            )
                
                # Summary statistics for Jina search
                if jina_response_lengths:
                    avg_length = sum(jina_response_lengths) / len(jina_response_lengths)
                    max_length = max(jina_response_lengths)
                    min_length = min(jina_response_lengths)
                    total_length = sum(jina_response_lengths)
                    if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO):
                        logger.info(
                            f"[TOOL PROCESSING] Jina Search Summary: "
                            f"avg={avg_length:,.0f} chars, max={max_length:,.0f} chars, min={min_length:,.0f} chars, "
                            f"total={total_length:,.0f} chars across {len(search_queries)} queries"
                        )
                
                # Normalize result length
                if len(search_results) < expected_results:
                    missing = expected_results - len(search_results)
                    pad_items = [f"[No results] Query: '{q[:128]}' | status=no_results" for q in search_queries[-missing:]] if search_queries else ["[No results] | status=no_results"] * missing
                    search_results += pad_items
                elif len(search_results) > expected_results:
                    search_results = search_results[:expected_results]
            else:
                # Use internal retriever service (original behavior)
                # Support both 'query' (new format) and 'query_list' (legacy format)
                parameters={"query": search_queries, "query_list": search_queries, "topk": self.config.topk}
                
                import asyncio
                import json

                # Execute search; capture metrics to detect skip conditions
                start_time = time.time()
                # instance = asyncio.run(self.search_tool.create())
                # exec_result = asyncio.run(self.search_tool.execute(instance_id=instance, parameters=parameters))
                exec_result = asyncio.run(self.search_tool.execute(parameters=parameters))
                elapsed_time = time.time() - start_time
                
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
                    
                    # Debug logging for local retriever
                    retriever_response_lengths = []
                    for i, result in enumerate(search_results):
                        result_length_chars = len(str(result))
                        result_length_tokens = len(self.tokenizer.encode(str(result), add_special_tokens=False))
                        retriever_response_lengths.append(result_length_chars)
                        
                        if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO):
                            query_preview = search_queries[i][:60] + '...' if i < len(search_queries) and len(search_queries[i]) > 60 else (search_queries[i] if i < len(search_queries) else 'N/A')
                            logger.info(
                                f"[TOOL PROCESSING] Local Retriever #{i+1}/{len(search_results)}: "
                                f"Query='{query_preview}' | "
                                f"Response: {result_length_chars:,} chars, {result_length_tokens:,} tokens"
                            )
                    
                    # Summary statistics for local retriever
                    if retriever_response_lengths:
                        avg_length = sum(retriever_response_lengths) / len(retriever_response_lengths)
                        max_length = max(retriever_response_lengths)
                        min_length = min(retriever_response_lengths)
                        total_length = sum(retriever_response_lengths)
                        if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO):
                            logger.info(
                                f"[TOOL PROCESSING] Local Retriever Summary: "
                                f"avg={avg_length:,.0f} chars, max={max_length:,.0f} chars, min={min_length:,.0f} chars, "
                                f"total={total_length:,.0f} chars across {len(search_queries)} queries | "
                                f"Time: {elapsed_time:.2f}s"
                            )
                            # Note: Local retriever responses are typically much shorter than Jina
                            logger.info(
                                f"[TOOL PROCESSING] Note: Local retriever responses are typically 5-10x shorter than Jina search responses"
                            )
                
                # If SearchTool signaled skip, treat as no evidence
                if isinstance(metrics, dict) and metrics.get('skipped', False):
                    search_results = [''] * expected_results
                    if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.WARNING):
                        logger.warning(f"[TOOL PROCESSING] Local Retriever signaled SKIP for queries")
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

        # Execute scholar tool if available
        scholar_results = []
        if scholar_queries and self.scholar_tool is not None:
            import asyncio
            parameters = {"query": scholar_queries, "topk": self.config.topk}
            start_time = time.time()
            # instance = asyncio.run(self.scholar_tool.create())
            # exec_result = asyncio.run(self.scholar_tool.execute(instance_id=instance, parameters=parameters))
            exec_result = asyncio.run(self.scholar_tool.execute(parameters=parameters))
            elapsed_time = time.time() - start_time
            
            # exec_result: (result_text, tool_reward, metrics)
            if isinstance(exec_result, tuple) and len(exec_result) >= 1:
                scholar_results_json = exec_result[0]
            else:
                scholar_results_json = exec_result
            
            try:
                parsed = json.loads(scholar_results_json)
                result_obj = parsed.get('result', parsed)
                if isinstance(result_obj, list):
                    scholar_results = result_obj
                else:
                    scholar_results = [str(result_obj)]
            except Exception:
                scholar_results = [scholar_results_json]
            
            # Debug logging for scholar tool
            if scholar_results:
                scholar_lengths = [len(str(r)) for r in scholar_results]
                scholar_token_lengths = [len(self.tokenizer.encode(str(r), add_special_tokens=False)) for r in scholar_results]
                if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO):
                    for i, (result, char_len, tok_len) in enumerate(zip(scholar_results, scholar_lengths, scholar_token_lengths)):
                        query_preview = scholar_queries[i][:60] + '...' if i < len(scholar_queries) and len(scholar_queries[i]) > 60 else (scholar_queries[i] if i < len(scholar_queries) else 'N/A')
                        logger.info(
                            f"[TOOL PROCESSING] Google Scholar #{i+1}/{len(scholar_results)}: "
                            f"Query='{query_preview}' | "
                            f"Response: {char_len:,} chars, {tok_len:,} tokens"
                        )
                    avg_len = sum(scholar_lengths) / len(scholar_lengths) if scholar_lengths else 0
                    logger.info(
                        f"[TOOL PROCESSING] Google Scholar Summary: "
                        f"avg={avg_len:,.0f} chars across {len(scholar_results)} queries | "
                        f"Time: {elapsed_time:.2f}s"
                    )
        
        # Merge scholar results if any (append to search results or handle separately as per model instruction)
        # For now, we'll just log them if they exist
        if scholar_results:
            # Here we might want to append to contents if the model asked for scholar specifically
            # But the current architecture merges them into search results based on the tag
            # If the tag was <google_scholar>, we should use these results.
            # But earlier parsing logic put scholar queries into 'scholar_queries' list.
            # We need to map them back to the original prediction slots.
            pass
            
            # Normalize length
            if len(scholar_results) < expected_scholar_results:
                scholar_results += ["No scholar results found."] * (expected_scholar_results - len(scholar_results))
            elif len(scholar_results) > expected_scholar_results:
                scholar_results = scholar_results[:expected_scholar_results]
        else:
            scholar_results = [''] * expected_scholar_results
        
        # Execute visit tool if available
        # ... (rest of the visit and python blocks)
        visit_results = []
        if expected_visit_results > 0 and self.visit_tool is not None:
            import asyncio
            for i, visit_content in enumerate(visit_contents):
                try:
                    # Parse URL and goal from content (format: "url|goal" or just "url")
                    if '|' in visit_content:
                        url_str, goal = visit_content.split('|', 1)
                        urls = [url_str.strip()]
                    else:
                        urls = [visit_content.strip()]
                        goal = ""
                    
                    if urls and urls[0]:
                        parameters = {"url": urls, "goal": goal}
                        start_time = time.time()
                        instance = asyncio.run(self.visit_tool.create())
                        exec_result = asyncio.run(self.visit_tool.execute(instance_id=instance, parameters=parameters))
                        elapsed_time = time.time() - start_time
                        
                        if isinstance(exec_result, tuple) and len(exec_result) >= 1:
                            visit_result_json = exec_result[0]
                        else:
                            visit_result_json = exec_result
                        
                        # Parse visit results
                        try:
                            parsed = json.loads(visit_result_json)
                            if isinstance(parsed, dict):
                                result_list = parsed.get('result', [])
                                if isinstance(result_list, list) and len(result_list) > 0:
                                    visit_results.append(result_list[0])  # Use first result
                                else:
                                    visit_results.append(str(result_list))
                            else:
                                visit_results.append(str(parsed))
                        except Exception:
                            visit_results.append(str(visit_result_json))
                        
                        # Debug logging for visit tool
                        result_str = str(visit_results[-1])
                        result_length_chars = len(result_str)
                        result_length_tokens = len(self.tokenizer.encode(result_str, add_special_tokens=False))
                        if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO):
                            logger.info(
                                f"[TOOL PROCESSING] Visit Tool #{i+1}/{len(visit_contents)}: "
                                f"URL='{urls[0][:60]}{'...' if len(urls[0]) > 60 else ''}' | "
                                f"Response: {result_length_chars:,} chars, {result_length_tokens:,} tokens | "
                                f"Time: {elapsed_time:.2f}s"
                            )
                    else:
                        visit_results.append("Error: No URL provided")
                except Exception as e:
                    logger.warning(f"Visit tool execution failed: {e}")
                    visit_results.append(f"Error visiting page: {str(e)}")
        elif expected_visit_results > 0:
            visit_results = ["Error: Visit tool not configured (visit_url not set)"] * expected_visit_results
        
        # Execute Python interpreter tool if available
        python_results = []
        if expected_python_results > 0 and self.python_tool is not None:
            import asyncio
            for i, python_code in enumerate(python_contents):
                try:
                    if python_code and python_code.strip():
                        parameters = {"code": python_code.strip()}
                        start_time = time.time()
                        instance = asyncio.run(self.python_tool.create())
                        exec_result = asyncio.run(self.python_tool.execute(instance_id=instance, parameters=parameters))
                        elapsed_time = time.time() - start_time
                        
                        if isinstance(exec_result, tuple):
                            # SandboxFusionTool returns (result, result, result.strip())
                            python_result = exec_result[0] if len(exec_result) > 0 else str(exec_result)
                        else:
                            python_result = str(exec_result)
                        
                        python_results.append(python_result)
                        
                        # Debug logging for Python interpreter
                        result_length_chars = len(python_result)
                        result_length_tokens = len(self.tokenizer.encode(python_result, add_special_tokens=False))
                        code_preview = python_code[:60] + '...' if len(python_code) > 60 else python_code
                        if getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO):
                            logger.info(
                                f"[TOOL PROCESSING] Python Interpreter #{i+1}/{len(python_contents)}: "
                                f"Code='{code_preview}' | "
                                f"Response: {result_length_chars:,} chars, {result_length_tokens:,} tokens | "
                                f"Time: {elapsed_time:.2f}s"
                            )
                    else:
                        python_results.append("Error: No code provided")
                except Exception as e:
                    logger.warning(f"Python interpreter execution failed: {e}")
                    python_results.append(f"Error executing Python code: {str(e)}")
        elif expected_python_results > 0:
            python_results = ["Error: Python interpreter not configured (sandbox_fusion_url not set)"] * expected_python_results

        # ========== TOOL PROCESSING SUMMARY ==========
        # Calculate total response lengths and provide summary
        all_tool_results = []
        all_tool_result_lengths = []
        all_tool_result_token_lengths = []
        
        # Collect all tool results for summary
        for result_list, tool_name in [
            (search_results, 'search'),
            (visit_results, 'visit'),
            (python_results, 'python'),
            (scholar_results, 'scholar')
        ]:
            for result in result_list:
                if result and str(result).strip():
                    result_str = str(result)
                    all_tool_results.append((tool_name, result_str))
                    all_tool_result_lengths.append(len(result_str))
                    all_tool_result_token_lengths.append(len(self.tokenizer.encode(result_str, add_special_tokens=False)))
        
        # Summary statistics
        if all_tool_result_lengths and (getattr(self.config, 'enable_debug_logs', False) or logger.isEnabledFor(logging.INFO)):
            total_chars = sum(all_tool_result_lengths)
            total_tokens = sum(all_tool_result_token_lengths)
            avg_chars = total_chars / len(all_tool_result_lengths) if all_tool_result_lengths else 0
            avg_tokens = total_tokens / len(all_tool_result_token_lengths) if all_tool_result_token_lengths else 0
            max_chars = max(all_tool_result_lengths) if all_tool_result_lengths else 0
            max_tokens = max(all_tool_result_token_lengths) if all_tool_result_token_lengths else 0
            
            logger.info(
                f"[TOOL PROCESSING] Turn {current_turn} Summary: "
                f"{len(all_tool_results)} tool responses | "
                f"Total: {total_chars:,} chars ({total_tokens:,} tokens) | "
                f"Avg: {avg_chars:,.0f} chars ({avg_tokens:,.0f} tokens) | "
                f"Max: {max_chars:,} chars ({max_tokens:,} tokens)"
            )
            
            # Warn if responses are very long (could cause context length issues)
            if max_tokens > 3000:
                logger.warning(
                    f"[TOOL PROCESSING] ⚠️ WARNING: Some tool responses are VERY LONG (max: {max_tokens:,} tokens). "
                    f"This may cause context length issues. Consider enabling context compression or reducing response size."
                )
            elif max_tokens > 2000:
                logger.info(
                    f"[TOOL PROCESSING] Note: Some tool responses are long (max: {max_tokens:,} tokens). "
                    f"Monitor context length if issues occur."
                )
            
            # Tool-specific counts
            tool_counts = {}
            for tool_name, _ in all_tool_results:
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            if tool_counts:
                tool_summary = ', '.join([f"{k}={v}" for k, v in tool_counts.items()])
                logger.info(f"[TOOL PROCESSING] Tool usage: {tool_summary}")

        for i, (action,active) in enumerate(zip(cur_actions,active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)

            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                elif action == 'search':
                    # Only consume search result for active entries
                    # Add turn information to observation
                    turn_info = f"\n[Turn {current_turn + 1}/{max_turns if max_turns else '?'}] "
                    search_result = search_results.pop(0).strip()
                   
                    # Check if this is the last turn and provide sharp message
                    if max_turns and current_turn >= max_turns - 2:
                        sharp_message = f"\n\nThis is my LAST turn (Turn {current_turn + 1}/{max_turns}). I MUST provide final answer now with <answer> and </answer>."
                        # Fix: Add proper role labeling for information blocks
                        next_obs.append(f'{turn_info}user\n<information>{search_result}</information>\n\nassistant\n{sharp_message}\n')
                    else:
                        # Fix: Add proper role labeling for information blocks
                        summary_prompt = (
                        "I will provide concise, high-level summaries of both my previous reasoning and the gathered information.\n"
                        "Use the <think_summary>...</think_summary> tag for my thought process summary,\n"
                        "and the <information_summary>...</information_summary> tag for key retrieved facts or evidence.\n"
                        "Focus on clarity and brevity to help guide my next response. Or I will use <answer> and </answer> to provide the final answer if information is enough."
                        )
                        next_obs.append(f'{turn_info}user\n<information>{search_result}</information>\n\nassistant\n{summary_prompt}\n')
                    
                    dones.append(0)
                    valid_action.append(1)
                elif action == 'scholar':
                    # Add turn information to observation
                    turn_info = f"\n[Turn {current_turn + 1}/{max_turns if max_turns else '?'}] "
                    scholar_result = scholar_results.pop(0).strip()
                   
                    # Check if this is the last turn
                    if max_turns and current_turn >= max_turns - 2:
                        sharp_message = f"\n\nThis is my LAST turn (Turn {current_turn + 1}/{max_turns}). I MUST provide final answer now with <answer> and </answer>."
                        next_obs.append(f'{turn_info}user\n<information>{scholar_result}</information>\n\nassistant\n{sharp_message}\n')
                    else:
                        summary_prompt = (
                        "I will provide concise, high-level summaries of both my previous reasoning and the gathered information.\n"
                        "Use the <think_summary>...</think_summary> tag for my thought process summary,\n"
                        "and the <information_summary>...</information_summary> tag for key retrieved facts or evidence.\n"
                        "Focus on clarity and brevity to help guide my next response. Or I will use <answer> and </answer> to provide the final answer if information is enough."
                        )
                        next_obs.append(f'{turn_info}user\n<information>{scholar_result}</information>\n\nassistant\n{summary_prompt}\n')
                    
                    dones.append(0)
                    valid_action.append(1)
                elif action in ["think", "think_summary", "information_summary"]:
                    # Add turn information to observation
                    turn_info = f"\n[Turn {current_turn + 1}/{max_turns if max_turns else '?'}] "
                    
                    # Check if this is the last turn and provide sharp message
                    if max_turns and current_turn >= max_turns - 2:
                        sharp_message = f"\n\nThis is my LAST turn (Turn {current_turn + 1}/{max_turns}). I MUST provide final answer now with <answer> and </answer>.\n\n"
                        next_obs.append(f'{turn_info}{sharp_message}')
                    else:
                        next_obs.append(f'{turn_info}')
                    
                    dones.append(0)
                    valid_action.append(1)
                elif action == 'visit':
                    # Execute visit tool and return results
                    turn_info = f"\n[Turn {current_turn + 1}/{max_turns if max_turns else '?'}] "
                    if visit_results:
                        visit_result = visit_results.pop(0).strip()
                        
                        # Check if this is the last turn
                        if max_turns and current_turn >= max_turns - 2:
                            sharp_message = f"\n\nThis is my LAST turn (Turn {current_turn + 1}/{max_turns}). I MUST provide final answer now with <answer> and </answer>."
                            next_obs.append(f'{turn_info}user\n<information>{visit_result}</information>\n\nassistant\n{sharp_message}\n')
                        else:
                            summary_prompt = (
                                "I will provide concise, high-level summaries of both my previous reasoning and the gathered information.\n"
                                "Use the <think_summary>...</think_summary> tag for my thought process summary,\n"
                                "and the <information_summary>...</information_summary> tag for key retrieved facts or evidence.\n"
                                "Focus on clarity and brevity to help guide my next response. Or I will use <answer> and </answer> to provide the final answer if information is enough."
                            )
                            next_obs.append(f'{turn_info}user\n<information>{visit_result}</information>\n\nassistant\n{summary_prompt}\n')
                    else:
                        next_obs.append(f'{turn_info}Error: Visit tool execution failed or no results returned.\n')
                    
                    dones.append(0)
                    valid_action.append(1)
                elif action == 'python':
                    # Execute Python interpreter and return results
                    turn_info = f"\n[Turn {current_turn + 1}/{max_turns if max_turns else '?'}] "
                    if python_results:
                        python_result = python_results.pop(0).strip()
                        
                        # Check if this is the last turn
                        if max_turns and current_turn >= max_turns - 2:
                            sharp_message = f"\n\nThis is my LAST turn (Turn {current_turn + 1}/{max_turns}). I MUST provide final answer now with <answer> and </answer>."
                            next_obs.append(f'{turn_info}user\n<information>{python_result}</information>\n\nassistant\n{sharp_message}\n')
                        else:
                            summary_prompt = (
                                "I will provide concise, high-level summaries of both my previous reasoning and the gathered information.\n"
                                "Use the <think_summary>...</think_summary> tag for my thought process summary,\n"
                                "and the <information_summary>...</information_summary> tag for key retrieved facts or evidence.\n"
                                "Focus on clarity and brevity to help guide my next response. Or I will use <answer> and </answer> to provide the final answer if information is enough."
                            )
                            next_obs.append(f'{turn_info}user\n<information>{python_result}</information>\n\nassistant\n{summary_prompt}\n')
                    else:
                        next_obs.append(f'{turn_info}Error: Python interpreter execution failed or no results returned.\n')
                    
                    dones.append(0)
                    valid_action.append(1)
                elif action == 'parse_file':
                    # Parse file tool - not yet implemented in verl framework
                    # Provide feedback that this tool needs file system access
                    turn_info = f"\n[Turn {current_turn + 1}/{max_turns if max_turns else '?'}] "
                    next_obs.append(f'{turn_info}Tool "parse_file" requires file system access which is not yet supported in this environment. Please use <search> for web search or <answer> to provide the final answer.\n')
                    dones.append(0)
                    valid_action.append(0)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
I can use the following tools:\n\
- <tool_call>{{"name": "search", "arguments": {{"query": ["query1", "query2"]}}}}</tool_call> for web search\n\
- <search>query</search> (legacy format) for web search\n\
- <answer>answer</answer> to provide the final answer\n\
- <think_summary>...</think_summary> and <information_summary>...</information_summary> for summaries\n\
Let me try again.\n') 
                    dones.append(0)
                    valid_action.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action 

    # postprocess_predictions is deprecated; parsing now occurs inside execute_predictions for single-source-of-truth.
