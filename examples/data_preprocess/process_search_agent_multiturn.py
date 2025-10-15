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
Process search-agent dataset into multi-turn SFT format.
Each example becomes a multi-turn conversation:
- User turn: question or search results
- Assistant turn: <think>reasoning</think> <search>query</search>

The conversation is split at each search_result step to create multiple training samples.
"""

import argparse
import json
import os
from typing import List, Dict, Any


def format_reasoning(content: str) -> str:
    """Format reasoning content with <think> tags."""
    return f"<think>\n{content}\n</think>"


def format_search(query: str) -> str:
    """Format search query with <search> tags."""
    return f"<search>\n{query}\n</search>"


def format_search_results(documents: List[Dict[str, str]]) -> str:
    """Format search results with <information> tags."""
    doc_texts = []
    for i, doc in enumerate(documents):
        title = doc.get("title", "")
        content = doc.get("content", "")
        doc_texts.append(f"Document {i+1}:\nTitle: {title}\nContent: {content}")
    return f"<information>\n" + "\n\n".join(doc_texts) + "\n</information>"


def format_answer(content: str) -> str:
    """Format final answer."""
    return f"<answer>\n{content}\n</answer>"


def process_single_example_multiturn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a single example into multiple multi-turn training samples.
    
    Format:
    - Turn 0: User asks question
    - Turn 1: Assistant responds with <think>...</think> <search>...</search>
    - Turn 2: User provides <information>search results</information>
    - Turn 3: Assistant responds with <think>...</think> <search>...</search>
    - ...
    - Turn N: Assistant provides final answer
    
    We create training samples by cutting at each search_result step:
    - Sample 1: [User Q] → [Asst turn 1] → [User results 1] → **[Asst turn 2]**
    - Sample 2: [User Q] → ... → [User results 2] → **[Asst turn 3]**
    """
    question = example.get("question", "")
    trace = example.get("trace", [])
    
    # Parse trace into turns
    # Each turn consists of: reasoning (optional) + search (optional) OR answer
    turns = []
    current_turn = {"reasoning": [], "search": None, "search_result": None, "answer": None}
    
    for step in trace:
        step_type = step.get("type", "")
        
        if step_type == "reasoning":
            current_turn["reasoning"].append(step.get("content", ""))
        elif step_type == "search":
            current_turn["search"] = step.get("query", "")
        elif step_type == "search_result":
            # Save search results and start new turn
            current_turn["search_result"] = step.get("documents", [])
            turns.append(current_turn)
            current_turn = {"reasoning": [], "search": None, "search_result": None, "answer": None}
        elif step_type == "CorrectAnswer":
            current_turn["answer"] = step.get("content", "")
            turns.append(current_turn)
            current_turn = {"reasoning": [], "search": None, "search_result": None, "answer": None}
    
    # Add remaining turn if any
    if current_turn["reasoning"] or current_turn["search"] or current_turn["answer"]:
        turns.append(current_turn)
    
    # Find turns with search_results (these are split points)
    search_result_turns = [i for i, turn in enumerate(turns) if turn["search_result"] is not None]
    
    if not search_result_turns:
        return []
    
    sft_samples = []
    
    # Create training samples by cutting at each search_result
    # Strategy: Build up the conversation progressively
    # Each sample trains on ALL assistant responses up to that point
    for idx, split_turn_idx in enumerate(search_result_turns):
        messages = []
        
        # Turn 0: User asks question
        messages.append({
            "role": "user",
            "content": question
        })
        
        # Build complete multi-turn conversation up to and including the target response
        # Pattern: user Q -> asst 1 -> user results 1 -> asst 2 -> ... -> asst N
        for turn_idx in range(split_turn_idx + 2):  # +2 to include the target turn
            if turn_idx >= len(turns):
                break
                
            turn = turns[turn_idx]
            
            # Assistant turn: <think>...</think> <search>...</search> or <answer>...</answer>
            assistant_content = ""
            
            # Add reasoning if present
            if turn["reasoning"]:
                reasoning_text = "\n".join(turn["reasoning"])
                assistant_content += format_reasoning(reasoning_text) + "\n\n"
            
            # Add search or answer
            if turn["search"]:
                assistant_content += format_search(turn["search"])
            elif turn["answer"]:
                assistant_content += format_answer(turn["answer"])
            
            assistant_content = assistant_content.strip()
            if assistant_content:  # Only add if there's content
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            # User turn with search results (if this turn has results and not the last turn)
            if turn["search_result"] is not None and turn_idx < split_turn_idx + 1:
                results_content = format_search_results(turn["search_result"])
                messages.append({
                    "role": "user",
                    "content": results_content
                })
        
        # Make sure the last message is an assistant message
        if messages and messages[-1]["role"] == "assistant":
            # Extract the final assistant response as the answer field
            final_assistant_content = messages[-1]["content"]
            
            # Create SFT sample with full multi-turn conversation
            # MultiTurnSFTDataset will compute loss on ALL assistant messages in this conversation
            sft_sample = {
                "data_source": "search_agent_multiturn",
                "prompt": messages,  # Complete multi-turn conversation
                "ability": "search_reasoning",
                "answer": final_assistant_content,  # The final assistant response
                "extra_info": {
                    "question": question,
                    "split_index": idx,
                    "total_splits": len(search_result_turns),
                    "turn_index": split_turn_idx + 1,
                    "num_assistant_turns": sum(1 for m in messages if m["role"] == "assistant"),
                },
            }
            sft_samples.append(sft_sample)
    
    return sft_samples


def main():
    parser = argparse.ArgumentParser(description="Process search-agent dataset for multi-turn SFT training")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output file")
    parser.add_argument("--output_format", type=str, default="parquet",
                        choices=["parquet", "jsonl"],
                        help="Output format (parquet or jsonl)")
    
    args = parser.parse_args()
    
    # Read input JSONL file
    print(f"Reading from {args.input_file}...")
    all_samples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                samples = process_single_example_multiturn(example)
                all_samples.extend(samples)
                if line_num % 50 == 0:
                    print(f"Processed {line_num} examples, generated {len(all_samples)} training samples so far")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nTotal examples processed: {line_num}")
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
    print(f"\nStatistics:")
    print(f"  Average splits per example: {len(all_samples) / line_num:.2f}")
    
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

