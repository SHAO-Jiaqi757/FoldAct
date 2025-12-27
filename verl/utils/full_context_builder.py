"""
Full Context Builder for Policy Gradient Training

This module provides utilities to reconstruct full context during training
to ensure correct log_prob computation for policy gradient.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from verl.utils.event_ledger import EventLedger, EventType, Event


class FullContextBuilder:
    """
    Reconstructs full context from compressed context and event ledger.
    
    This is crucial for correct policy gradient training where:
    1. Generation uses compressed context (efficiency)
    2. Reward calculation uses event ledger (correctness) 
    3. Log_prob computation should use full context (correctness)
    """
    
    def __init__(self, tokenizer, max_context_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        
    def reconstruct_full_context(
        self, 
        compressed_input_ids: torch.Tensor,
        compressed_attention_mask: torch.Tensor,
        event_ledger: EventLedger,
        step_ids: torch.Tensor,
        original_prompt: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct full context from compressed context and event ledger.
        
        Args:
            compressed_input_ids: Compressed input_ids from generation
            compressed_attention_mask: Compressed attention_mask from generation  
            event_ledger: Full event history
            step_ids: Mapping from token indices to turn IDs
            original_prompt: Original prompt (if available)
            
        Returns:
            Tuple of (full_input_ids, full_attention_mask, full_position_ids)
        """
        batch_size = compressed_input_ids.size(0)
        
        # For each item in batch, reconstruct full context
        full_input_ids_list = []
        full_attention_mask_list = []
        full_position_ids_list = []
        
        for batch_idx in range(batch_size):
            full_input_ids, full_attention_mask, full_position_ids = self._reconstruct_single_item(
                compressed_input_ids[batch_idx],
                compressed_attention_mask[batch_idx], 
                event_ledger,
                step_ids[batch_idx] if step_ids is not None else None,
                original_prompt
            )
            
            full_input_ids_list.append(full_input_ids)
            full_attention_mask_list.append(full_attention_mask)
            full_position_ids_list.append(full_position_ids)
        
        # Pad to same length
        max_len = max(ids.size(0) for ids in full_input_ids_list)
        
        padded_input_ids = torch.zeros(batch_size, max_len, dtype=compressed_input_ids.dtype)
        padded_attention_mask = torch.zeros(batch_size, max_len, dtype=compressed_attention_mask.dtype)
        padded_position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, (ids, mask, pos_ids) in enumerate(zip(full_input_ids_list, full_attention_mask_list, full_position_ids_list)):
            seq_len = ids.size(0)
            padded_input_ids[i, :seq_len] = ids
            padded_attention_mask[i, :seq_len] = mask
            padded_position_ids[i, :seq_len] = pos_ids
            
        return padded_input_ids, padded_attention_mask, padded_position_ids
    
    def _reconstruct_single_item(
        self,
        compressed_input_ids: torch.Tensor,
        compressed_attention_mask: torch.Tensor,
        event_ledger: EventLedger,
        step_ids: torch.Tensor,
        original_prompt: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct full context for a single item.
        """
        # Convert compressed context to text
        compressed_text = self.tokenizer.decode(compressed_input_ids[compressed_attention_mask.bool()])
        
        # Reconstruct full context from event ledger
        full_context_parts = []
        
        # Add original prompt if available
        if original_prompt:
            full_context_parts.append(original_prompt)
        
        # Reconstruct from event ledger in chronological order
        events_by_turn = {}
        for event in event_ledger.events:
            if event.turn_id not in events_by_turn:
                events_by_turn[event.turn_id] = []
            events_by_turn[event.turn_id].append(event)
        
        # Sort by turn_id and reconstruct
        for turn_id in sorted(events_by_turn.keys()):
            turn_events = events_by_turn[turn_id]
            
            for event in turn_events:
                if event.event_type == EventType.SEARCH:
                    full_context_parts.append(f"<search>\n{event.content}\n</search>")
                elif event.event_type == EventType.INFORMATION:
                    full_context_parts.append(f"<information>\n{event.content}\n</information>")
                elif event.event_type == EventType.THINK_SUMMARY:
                    full_context_parts.append(f"<think_summary>\n{event.content}\n</think_summary>")
                elif event.event_type == EventType.INFORMATION_SUMMARY:
                    full_context_parts.append(f"<information_summary>\n{event.content}\n</information_summary>")
                elif event.event_type == EventType.ANSWER:
                    full_context_parts.append(f"<answer>\n{event.content}\n</answer>")
                elif event.event_type == EventType.THINK:
                    full_context_parts.append(f"<think>\n{event.content}\n</think>")
        
        # Join all parts
        full_context_text = "\n\n".join(full_context_parts)
        
        # Tokenize full context
        full_tokens = self.tokenizer.encode(full_context_text, add_special_tokens=True)
        
        # Truncate if too long
        if len(full_tokens) > self.max_context_length:
            full_tokens = full_tokens[-self.max_context_length:]
        
        # Convert to tensors
        full_input_ids = torch.tensor(full_tokens, dtype=compressed_input_ids.dtype)
        full_attention_mask = torch.ones(len(full_tokens), dtype=compressed_attention_mask.dtype)
        full_position_ids = torch.arange(len(full_tokens), dtype=torch.long)
        
        return full_input_ids, full_attention_mask, full_position_ids
    
    def get_full_context_for_turn(
        self,
        event_ledger: EventLedger,
        turn_id: int,
        max_length: int = None
    ) -> str:
        """
        Get full context up to a specific turn.
        
        Args:
            event_ledger: Event ledger containing full history
            turn_id: Turn ID to reconstruct context up to
            max_length: Maximum context length
            
        Returns:
            Full context string up to the specified turn
        """
        context_parts = []
        
        # Get all events up to turn_id
        events_up_to_turn = [e for e in event_ledger.events if e.turn_id <= turn_id]
        
        # Sort by turn_id and timestamp
        events_up_to_turn.sort(key=lambda e: (e.turn_id, e.timestamp))
        
        for event in events_up_to_turn:
            if event.event_type == EventType.SEARCH:
                context_parts.append(f"<search>\n{event.content}\n</search>")
            elif event.event_type == EventType.INFORMATION:
                context_parts.append(f"<information>\n{event.content}\n</information>")
            elif event.event_type == EventType.THINK_SUMMARY:
                context_parts.append(f"<think_summary>\n{event.content}\n</think_summary>")
            elif event.event_type == EventType.INFORMATION_SUMMARY:
                context_parts.append(f"<information_summary>\n{event.content}\n</information_summary>")
            elif event.event_type == EventType.ANSWER:
                context_parts.append(f"<answer>\n{event.content}\n</answer>")
            elif event.event_type == EventType.THINK:
                context_parts.append(f"<think>\n{event.content}\n</think>")
        
        full_context = "\n\n".join(context_parts)
        
        # Truncate if needed
        if max_length and len(full_context) > max_length:
            full_context = full_context[-max_length:]
            
        return full_context


class FullContextLogProbComputer:
    """
    Computes log probabilities using full context for policy gradient training.
    """
    
    def __init__(self, full_context_builder: FullContextBuilder, model, tokenizer):
        self.full_context_builder = full_context_builder
        self.model = model
        self.tokenizer = tokenizer
        
    def compute_full_context_log_prob(
        self,
        compressed_data: Dict[str, torch.Tensor],
        event_ledgers: List[EventLedger],
        step_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute log probabilities using full context.
        
        Args:
            compressed_data: Data with compressed context
            event_ledgers: List of event ledgers (one per batch item)
            step_ids: Step IDs tensor
            
        Returns:
            Log probabilities computed with full context
        """
        batch_size = compressed_data["input_ids"].size(0)
        log_probs_list = []
        
        for batch_idx in range(batch_size):
            # Get full context for this item
            full_input_ids, full_attention_mask, full_position_ids = self.full_context_builder.reconstruct_full_context(
                compressed_data["input_ids"][batch_idx:batch_idx+1],
                compressed_data["attention_mask"][batch_idx:batch_idx+1],
                event_ledgers[batch_idx],
                step_ids[batch_idx:batch_idx+1] if step_ids is not None else None
            )
            
            # Compute log probabilities with full context
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    position_ids=full_position_ids
                )
                
                # Extract log probabilities for response tokens
                response_length = compressed_data["responses"][batch_idx].size(0)
                response_logits = outputs.logits[:, -response_length-1:-1]
                response_tokens = compressed_data["responses"][batch_idx]
                
                # Compute log probabilities
                log_probs = torch.log_softmax(response_logits, dim=-1)
                token_log_probs = log_probs.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)
                
                log_probs_list.append(token_log_probs)
        
        # Stack and return
        return torch.stack(log_probs_list, dim=0)