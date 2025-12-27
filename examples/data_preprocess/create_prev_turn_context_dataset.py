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
import re
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


def truncate_information_content(content: str, max_chars: int = 4000) -> str:
    """
    Truncate <information> blocks in user messages to reduce context length.
    Keeps the structure but limits the content size.
    """
    if "<information>" not in content:
        return content
    
    # Pattern to match <information>...</information> blocks
    pattern = r'(<information>)(.*?)(</information>)'
    
    def truncate_match(match):
        tag_open = match.group(1)
        tag_content = match.group(2)
        tag_close = match.group(3)
        
        # If content is already short enough, return as is
        if len(tag_content) <= max_chars:
            return match.group(0)
        
        # Truncate to max_chars, trying to preserve structure
        truncated = tag_content[:max_chars]
        
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
        
        return tag_open + truncated + tag_close
    
    return re.sub(pattern, truncate_match, content, flags=re.DOTALL)


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
        # Truncate information content in the latest user message
        truncated_content = truncate_information_content(latest_user_message.get("content", ""))
        truncated_user_message = {
            "role": "user",
            "content": truncated_content
        }
        prompt_components.append(truncated_user_message)

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

        new_row["prompt"] = truncated_prompt

        new_extra = dict(extra)
        new_extra["context_prev_turn_only"] = context_flag
        new_extra["expected_answer_is_summary_plus_action"] = True
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create previous-turn-context dataset from SFT multi-turn data."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to sft_train_multiturn_with_summary.jsonl.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output path for the truncated-context dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_jsonl(args.input_file)
    processed = process_rows(rows)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_file, processed)


if __name__ == "__main__":
    main()
