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
Process Search-R1 dataset into SFT format.
For each example, split at search_result steps to create multiple training samples.
"""

import argparse
import json
import os
from typing import List, Dict, Any


def format_trace_step(step: Dict[str, Any]) -> str:
    """Format a single trace step into text."""
    step_type = step.get("type", "")
    content = step.get("content", "")
    
    if step_type == "reasoning":
        return f"<think>\n{content}\n</think>"
    elif step_type == "search":
        query = step.get("query", "")
        return f"<search>\n{query}\n</search>"
    elif step_type == "search_result":
        documents = step.get("documents", [])
        doc_texts = []
        for i, doc in enumerate(documents):
            title = doc.get("title", "")
            doc_content = doc.get("content", "")
            doc_texts.append(f"Document {i+1}:\nTitle: {title}\nContent: {doc_content}")
        return f"<information>\n" + "\n\n".join(doc_texts) + "\n</information>"
    elif step_type == "CorrectAnswer":
        return f"<answer>\n{content}\n</answer>"
    else:
        return f"<{step_type}>\n{content}\n</{step_type}>"


def process_single_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a single example into multiple SFT training samples.
    Split at search_result steps.
    """
    question = example.get("question", "")
    trace = example.get("trace", [])
    
    # Find all search_result step indices
    search_result_indices = []
    for i, step in enumerate(trace):
        if step.get("type") == "search_result":
            search_result_indices.append(i)
    
    # If no search results, return empty
    if not search_result_indices:
        return []
    
    sft_samples = []
    
    # Create training samples by splitting at each search_result
    for idx, split_point in enumerate(search_result_indices):
        # Context: from beginning to current search_result (inclusive)
        context_steps = trace[:split_point + 1]
        
        # Response: from next step to either next search_result or end
        if idx + 1 < len(search_result_indices):
            # There's another search_result, so response goes until (not including) next search_result
            response_start = split_point + 1
            response_end = search_result_indices[idx + 1]
            response_steps = trace[response_start:response_end]
        else:
            # This is the last search_result, response goes to the end
            response_steps = trace[split_point + 1:]
        
        # Skip if response is empty
        if not response_steps:
            continue
        
        # Format context (prompt)
        context_text = f"Question: {question}\n\n"
        for step in context_steps:
            context_text += format_trace_step(step) + "\n\n"
        context_text = context_text.strip()
        
        # Format response
        response_text = ""
        for step in response_steps:
            response_text += format_trace_step(step) + "\n\n"
        response_text = response_text.strip()
        
        # Create SFT sample
        sft_sample = {
            "data_source": "search_r1",
            "prompt": [
                {
                    "role": "user",
                    "content": context_text,
                }
            ],
            "ability": "search_reasoning",
            "extra_info": {
                "question": context_text,
                "answer": response_text,
                "split_index": idx,
                "total_splits": len(search_result_indices),
            },
        }
        sft_samples.append(sft_sample)
    
    return sft_samples


def main():
    parser = argparse.ArgumentParser(description="Process Search-R1 dataset for SFT training")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output parquet file")
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
                samples = process_single_example(example)
                all_samples.extend(samples)
                if line_num % 50 == 0:
                    print(f"Processed {line_num} examples, generated {len(all_samples)} training samples so far")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"\nTotal examples processed: {line_num}")
    print(f"Total SFT training samples generated: {len(all_samples)}")
    
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


if __name__ == "__main__":
    main()



