#!/usr/bin/env python3
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
"""
Process web_compress dataset into multi-turn SFT format.
Each example becomes a multi-turn conversation:
- User turn: question or tool response
- Assistant turn: <think>reasoning</think> <tool_call>...</tool_call>

The conversation is split at each tool_response step to create multiple training samples.
"""

import argparse
import json
import os
import random
import re
from typing import List, Dict, Any, Optional


def extract_tool_call(content: str) -> Optional[Dict[str, Any]]:
    """Extract tool call JSON from <tool_call> tags."""
    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def extract_tool_response(content: str) -> Optional[str]:
    """Extract tool response from <tool_response> tags."""
    match = re.search(r'<tool_response>\s*(.*?)\s*</tool_response>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_reasoning(content: str) -> Optional[str]:
    """Extract reasoning from <think> tags."""
    match = re.search(r'<think>\s*(.*?)\s*</think>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_answer(content: str) -> Optional[str]:
    """Extract answer from <answer> tags."""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def format_tool_call(tool_call: Dict[str, Any]) -> str:
    """Format tool call as <tool_call> JSON string."""
    return f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False, indent=2)}\n</tool_call>"


def truncate_tool_response_content(response: str, max_chars: int = 4000) -> str:
    """
    Truncate tool response content to reduce context length.
    Keeps the structure but limits the content size.
    """
    if len(response) <= max_chars:
        return response
    
    # Truncate to max_chars, trying to preserve structure
    truncated = response[:max_chars]
    
    # Try to cut at a reasonable boundary (end of a search result)
    # Look for the last occurrence of "=======" (separator between searches)
    last_separator = truncated.rfind("=======")
    if last_separator > max_chars * 0.7:  # If separator is in last 30%, use it
        truncated = truncated[:last_separator + 7]  # Include the separator
    else:
        # Otherwise, try to cut at end of a result number
        last_result = truncated.rfind("\n\n## Web Results")
        if last_result > max_chars * 0.7:
            truncated = truncated[:last_result]
        else:
            # Just truncate and add ellipsis
            truncated = truncated.rstrip() + "\n\n[... content truncated ...]"
    
    return truncated


def format_tool_response(response: str) -> str:
    """Format tool response with <information> tags, truncating if too long."""
    # Truncate the response content before formatting
    truncated_response = truncate_tool_response_content(response, max_chars=4000)
    return f"<information>\n{truncated_response}\n</information>"


def format_reasoning(content: str, use_think_tags: bool = True) -> str:
    """Format reasoning content with or without <think> tags."""
    if use_think_tags:
        return f"<think>\n{content}\n</think>"
    else:
        return content


def format_answer(content: str) -> str:
    """Format final answer."""
    return f"<answer>\n{content}\n</answer>"


def process_single_example_multiturn(example: Dict[str, Any], think_drop_prob: float = 0.3) -> List[Dict[str, Any]]:
    """
    Process a single example into multiple multi-turn training samples.
    
    Format:
    - Turn 0: User asks question
    - Turn 1: Assistant responds with <think>...</think> <tool_call>...</tool_call>
    - Turn 2: User provides <information>tool response</information>
    - Turn 3: Assistant responds with <think>...</think> <tool_call>...</tool_call>
    - ...
    - Turn N: Assistant provides final answer
    
    We create training samples by cutting at each tool_response step:
    - Sample 1: [User Q] → [Asst turn 1] → [User response 1] → **[Asst turn 2]**
    - Sample 2: [User Q] → ... → [User response 2] → **[Asst turn 3]**
    """
    messages = example.get("messages", [])
    question = example.get("question", "")
    
    # Extract question from first user message if not in root
    if not question:
        for msg in messages:
            if msg.get("role") == "user" and not msg.get("content", "").startswith("<tool_response>"):
                question = msg.get("content", "")
                break
    
    # Parse messages into turns
    turns = []
    current_turn = {"reasoning": None, "tool_call": None, "tool_response": None, "answer": None}
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "assistant":
            # Extract reasoning, tool_call, or answer
            reasoning = extract_reasoning(content)
            tool_call = extract_tool_call(content)
            answer = extract_answer(content)
            
            if reasoning:
                current_turn["reasoning"] = reasoning
            if tool_call:
                current_turn["tool_call"] = tool_call
            if answer:
                # Final answer - append turn and start new one
                current_turn["answer"] = answer
                turns.append(current_turn)
                current_turn = {"reasoning": None, "tool_call": None, "tool_response": None, "answer": None}
            # Don't append turn yet if it's just a tool_call - wait for tool_response
        
        elif role == "user":
            # Extract tool response
            tool_response = extract_tool_response(content)
            if tool_response:
                current_turn["tool_response"] = tool_response
                # Now we have both tool_call and tool_response, so append the turn
                if current_turn["tool_call"]:
                    turns.append(current_turn)
                    current_turn = {"reasoning": None, "tool_call": None, "tool_response": None, "answer": None}
    
    # Add remaining turn if any
    if current_turn["reasoning"] or current_turn["tool_call"] or current_turn["answer"]:
        turns.append(current_turn)
    
    # Find turns with tool_response (these are split points)
    tool_response_turns = [i for i, turn in enumerate(turns) if turn["tool_response"] is not None]
    
    if not tool_response_turns:
        # If no tool responses, check if there's a final answer
        if turns and turns[-1].get("answer"):
            # Create a single sample with just question -> answer
            messages = [{"role": "user", "content": question}]
            assistant_content = ""
            if turns[-1]["reasoning"]:
                reasoning_text = turns[-1]["reasoning"]
                use_think_tags = random.random() > think_drop_prob
                formatted_reasoning = format_reasoning(reasoning_text, use_think_tags)
                assistant_content += formatted_reasoning + "\n\n"
            if turns[-1]["answer"]:
                assistant_content += format_answer(turns[-1]["answer"])
            
            assistant_content = assistant_content.strip()
            if assistant_content:
                return [{
                    "data_source": "web_compress_multiturn",
                    "prompt": messages,
                    "ability": "search_reasoning",
                    "answer": assistant_content,
                    "extra_info": {
                        "question": question,
                        "split_index": -1,
                        "total_splits": 1,
                        "turn_index": 0,
                        "num_assistant_turns": 0,
                    },
                }]
        return []
    
    sft_samples = []
    
    # First, create a training sample for the initial assistant response (turn_index 0)
    # This teaches the model to respond to questions directly
    if turns and (turns[0]["reasoning"] or turns[0]["tool_call"]):
        messages = []
        
        # Turn 0: User asks question
        messages.append({
            "role": "user", 
            "content": question
        })
        
        # Turn 1: Assistant responds with initial reasoning/tool_call
        assistant_content = ""
        if turns[0]["reasoning"]:
            reasoning_text = turns[0]["reasoning"]
            use_think_tags = random.random() > think_drop_prob
            formatted_reasoning = format_reasoning(reasoning_text, use_think_tags)
            assistant_content += formatted_reasoning + "\n\n"
        
        if turns[0]["tool_call"]:
            assistant_content += format_tool_call(turns[0]["tool_call"])
        
        assistant_content = assistant_content.strip()
        if assistant_content:
            # Create SFT sample for initial response
            prompt_messages = [{
                "role": "user", 
                "content": question
            }]
            
            sft_sample = {
                "data_source": "web_compress_multiturn",
                "prompt": prompt_messages,
                "ability": "search_reasoning", 
                "answer": assistant_content,
                "extra_info": {
                    "question": question,
                    "split_index": -1,
                    "total_splits": len(tool_response_turns) + 1,
                    "turn_index": 0,
                    "num_assistant_turns": 0,
                },
            }
            sft_samples.append(sft_sample)
    
    # Create training samples by cutting at each tool_response
    for idx, split_turn_idx in enumerate(tool_response_turns):
        messages = []
        
        # Turn 0: User asks question
        messages.append({
            "role": "user",
            "content": question
        })
        
        # Build complete multi-turn conversation up to the target response
        # Pattern: user Q -> asst 1 -> user response 1 -> asst 2 -> ... -> asst N (target)
        for turn_idx in range(split_turn_idx + 1):  # Include all turns up to split_turn_idx
            if turn_idx >= len(turns):
                break
                
            turn = turns[turn_idx]
            
            # Assistant turn: <think>...</think> <tool_call>...</tool_call>
            assistant_content = ""
            
            # Add reasoning if present (randomly drop think tags)
            if turn["reasoning"]:
                reasoning_text = turn["reasoning"]
                use_think_tags = random.random() > think_drop_prob
                formatted_reasoning = format_reasoning(reasoning_text, use_think_tags)
                if use_think_tags:
                    assistant_content += formatted_reasoning + "\n\n"
                else:
                    assistant_content += formatted_reasoning + "\n\n"
            
            # Add tool_call
            if turn["tool_call"]:
                assistant_content += format_tool_call(turn["tool_call"])
            
            assistant_content = assistant_content.strip()
            if assistant_content:
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            # User turn with tool response (this turn should have a response)
            if turn["tool_response"] is not None:
                results_content = format_tool_response(turn["tool_response"])
                messages.append({
                    "role": "user",
                    "content": results_content
                })
        
        # Now build the target assistant turn (the one we're predicting)
        target_turn_idx = split_turn_idx + 1
        final_assistant_content = ""
        
        if target_turn_idx < len(turns):
            target_turn = turns[target_turn_idx]
            
            # Build target assistant content
            assistant_content = ""
            
            # Add reasoning if present
            if target_turn["reasoning"]:
                reasoning_text = target_turn["reasoning"]
                use_think_tags = random.random() > think_drop_prob
                formatted_reasoning = format_reasoning(reasoning_text, use_think_tags)
                if use_think_tags:
                    assistant_content += formatted_reasoning + "\n\n"
                else:
                    assistant_content += formatted_reasoning + "\n\n"
            
            # Add tool_call or answer
            if target_turn["tool_call"]:
                assistant_content += format_tool_call(target_turn["tool_call"])
            elif target_turn["answer"]:
                assistant_content += format_answer(target_turn["answer"])
            
            final_assistant_content = assistant_content.strip()
        
        # Skip if no target response
        if not final_assistant_content:
            continue
        
        # The prompt_messages should not include the target assistant response
        prompt_messages = messages
        
        # Create SFT sample with multi-turn conversation
        sft_sample = {
            "data_source": "web_compress_multiturn",
            "prompt": prompt_messages,
            "ability": "search_reasoning",
            "answer": final_assistant_content,
            "extra_info": {
                "question": question,
                "split_index": idx,
                "total_splits": len(tool_response_turns),
                "turn_index": split_turn_idx + 1,
                "num_assistant_turns": sum(1 for m in prompt_messages if m["role"] == "assistant"),
            },
        }
        sft_samples.append(sft_sample)
    
    return sft_samples


def main():
    parser = argparse.ArgumentParser(description="Process web_compress dataset for multi-turn SFT training")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output file")
    parser.add_argument("--output_format", type=str, default="parquet",
                        choices=["parquet", "jsonl"],
                        help="Output format (parquet or jsonl)")
    parser.add_argument("--think_drop_prob", type=float, default=0.3,
                        help="Probability of dropping <think> tags (0.0 = never drop, 1.0 = always drop)")
    
    args = parser.parse_args()
    
    # Read input JSONL file
    print(f"Reading from {args.input_file}...")
    print("Filtering: Only processing examples with is_correct=True")
    all_samples = []
    total_examples = 0
    correct_examples = 0
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                total_examples += 1
                # Only process examples where is_correct is True
                if not example.get("is_correct", False):
                    continue
                correct_examples += 1
                samples = process_single_example_multiturn(example, args.think_drop_prob)
                all_samples.extend(samples)
                if correct_examples % 50 == 0:
                    print(f"Processed {correct_examples} correct examples (out of {total_examples} total), generated {len(all_samples)} training samples so far")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nTotal examples in file: {total_examples}")
    print(f"Correct examples (is_correct=True): {correct_examples}")
    print(f"Total multi-turn SFT training samples generated: {len(all_samples)}")
    
    # Write output
    if args.output_format == "jsonl":
        print(f"\nWriting to {args.output_file}...")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        # Write as parquet
        print(f"\nWriting to {args.output_file}...")
        import pandas as pd
        df = pd.DataFrame(all_samples)
        df.to_parquet(args.output_file, index=False)
    
    print(f"Done! Output saved to {args.output_file}")
    
    # Print statistics
    if correct_examples > 0:
        print(f"\nStatistics:")
        print(f"  Average splits per correct example: {len(all_samples) / correct_examples:.2f}")
    
    # Print sample output
    if all_samples:
        print(f"\n{'='*80}")
        print("Sample output (first training sample):")
        print('='*80)
        sample = all_samples[0]
        print(f"Data source: {sample['data_source']}")
        print(f"Ability: {sample['ability']}")
        print(f"Extra info: {sample['extra_info']}")
        print(f"\nPrompt ({len(sample['prompt'])} turns - context for prediction):")
        for i, msg in enumerate(sample['prompt']):
            print(f"\n--- Turn {i+1} ({msg['role'].upper()}) ---")
            content = msg['content']
            # Truncate if too long
            if len(content) > 500:
                print(content[:500] + "...[truncated]")
            else:
                print(content)
        print(f"\n--- ANSWER (to predict) ---")
        answer = sample['answer']
        if len(answer) > 500:
            print(answer[:500] + "...[truncated]")
        else:
            print(answer)
        print('='*80)


if __name__ == "__main__":
    main()

