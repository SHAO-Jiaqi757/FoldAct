#!/usr/bin/env python3
"""
Build a dataset that exposes only the previous turn's agent response (summary + action)
and the most recent observation while preserving the initial question in the prompt.

For turn_index == 0 the original prompt is retained. For later turns, the prompt keeps:
    1. The initial user question (full context for the task)
    2. The previous turn's expected agent output (summary + action)
    3. The most recent user observation (e.g., <information> block)

The training target (answer field) is rewritten as summary + action so it matches the
context shown to the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def combine_summary_and_action(summary: Optional[str], action: Optional[str]) -> str:
    pieces: List[str] = []
    if summary:
        pieces.append(summary.strip())
    if action:
        pieces.append(action.strip())
    return "\n\n".join(pieces).strip()


def split_conversations(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    conversations: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for row in rows:
        extra = row.get("extra_info", {}) or {}
        turn_index = extra.get("turn_index", 0)
        split_index = extra.get("split_index", -1)

        if current and turn_index == 0 and split_index == -1:
            conversations.append(current)
            current = []

        current.append(row)

    if current:
        conversations.append(current)

    return conversations


def find_initial_user_message(prompt: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for msg in prompt:
        if msg.get("role") == "user":
            return msg
    return None


def find_latest_user_message(prompt: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for msg in reversed(prompt):
        if msg.get("role") == "user":
            return msg
    return None


def truncate_prompt(
    original_prompt: List[Dict[str, str]],
    initial_user_message: Optional[Dict[str, str]],
    previous_agent_response: Optional[str],
) -> List[Dict[str, str]]:
    if not previous_agent_response or not initial_user_message:
        return original_prompt

    latest_user_message = find_latest_user_message(original_prompt)
    prompt_components: List[Dict[str, str]] = [initial_user_message]

    prompt_components.append({
        "role": "assistant",
        "content": previous_agent_response,
    })

    if latest_user_message and latest_user_message is not initial_user_message:
        prompt_components.append(latest_user_message)

    return prompt_components


def process_conversation(conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []

    conv_sorted = sorted(
        conv,
        key=lambda r: (
            r.get("extra_info", {}).get("turn_index", 0),
            r.get("extra_info", {}).get("split_index", 0),
        ),
    )

    initial_prompt = conv_sorted[0].get("prompt", [])
    initial_user_message = find_initial_user_message(initial_prompt)

    previous_combined_response: Optional[str] = None
    previous_turn_index: Optional[int] = None

    for row in conv_sorted:
        prompt = row.get("prompt", [])
        extra = row.get("extra_info", {}) or {}
        turn_index = extra.get("turn_index", 0)

        combined_answer = combine_summary_and_action(
            row.get("summary"),
            row.get("answer"),
        )

        new_row = dict(row)
        new_row["answer"] = combined_answer

        if turn_index == 0 or not previous_combined_response or not initial_user_message:
            truncated_prompt = prompt
            context_flag = False
        else:
            truncated_prompt = truncate_prompt(
                prompt,
                initial_user_message,
                previous_combined_response,
            )
            context_flag = True

        # For MultiTurnSFTDataset: Add the answer as the final assistant message in the conversation
        # This creates a complete conversation that MultiTurnSFTDataset can process
        complete_conversation = truncated_prompt.copy()
        complete_conversation.append({
            "role": "assistant",
            "content": combined_answer
        })
        
        new_row["prompt"] = complete_conversation

        new_extra = dict(extra)
        new_extra["context_prev_turn_only"] = context_flag
        new_extra["expected_answer_is_summary_plus_action"] = True
        new_extra["multiturn_format"] = True  # Flag for MultiTurnSFTDataset
        if context_flag and previous_turn_index is not None:
            new_extra["previous_turn_index"] = previous_turn_index
        new_row["extra_info"] = new_extra

        processed.append(new_row)

        previous_combined_response = combined_answer
        previous_turn_index = turn_index

    return processed


def process_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conversations = split_conversations(rows)
    processed: List[Dict[str, Any]] = []
    for conv in conversations:
        processed.extend(process_conversation(conv))
    return processed


def create_summary_only_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create Dataset 2: prompt = summary only, keep original answer"""
    dataset2 = []
    
    for row in rows:
        new_sample = dict(row)
        
        # Get summary and original answer
        summary = row.get("summary", "")
        original_answer = row["answer"]
        
        # For MultiTurnSFTDataset: Create a simple conversation with summary as user message
        # and original answer as assistant response
        complete_conversation = [
            {
                "role": "user",
                "content": summary
            },
            {
                "role": "assistant", 
                "content": original_answer
            }
        ]
        
        # Update prompt to include the complete conversation
        new_sample["prompt"] = complete_conversation
        
        # Update extra_info question to use summary
        new_sample["extra_info"] = row["extra_info"].copy()
        new_sample["extra_info"]["question"] = summary
        new_sample["extra_info"]["original_question_discarded"] = True
        new_sample["extra_info"]["multiturn_format"] = True  # Flag for MultiTurnSFTDataset
        
        dataset2.append(new_sample)
    
    return dataset2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create both summary prefix and summary only datasets from SFT multi-turn data."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to sft_train_multiturn_with_summary.jsonl.",
    )
    parser.add_argument(
        "--output_prefix_file",
        type=Path,
        required=True,
        help="Output path for the summary prefix dataset (Dataset 1).",
    )
    parser.add_argument(
        "--output_only_file",
        type=Path,
        required=True,
        help="Output path for the summary only dataset (Dataset 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_jsonl(args.input_file)
    
    # Create Dataset 1: Summary prefix dataset (original functionality)
    print("Creating Dataset 1: Summary prefix dataset...")
    processed_prefix = process_rows(rows)
    args.output_prefix_file.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_prefix_file, processed_prefix)
    print(f"✓ Dataset 1 created: {len(processed_prefix)} samples → {args.output_prefix_file}")
    
    # Create Dataset 2: Summary only dataset
    print("Creating Dataset 2: Summary only dataset...")
    processed_only = create_summary_only_dataset(rows)
    args.output_only_file.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_only_file, processed_only)
    print(f"✓ Dataset 2 created: {len(processed_only)} samples → {args.output_only_file}")
    
    print("\n" + "="*80)
    print("DATASET 1 EXAMPLE (Summary Prefix)")
    print("="*80)
    if processed_prefix:
        print(f"Multiturn prompt length: {len(processed_prefix[0]['prompt'])} messages")
        print(f"Answer (first 300 chars):\n{processed_prefix[0]['answer'][:300]}...")
    
    print("\n" + "="*80)
    print("DATASET 2 EXAMPLE (Summary Only)")
    print("="*80)
    if processed_only:
        print(f"Simple prompt: {len(processed_only[0]['prompt'])} message(s)")
        print(f"Prompt content (first 300 chars):\n{processed_only[0]['prompt'][0]['content'][:300]}...")
        print(f"Answer (first 200 chars):\n{processed_only[0]['answer'][:200]}...")
    print("="*80)


if __name__ == "__main__":
    main()
