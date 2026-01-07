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

import os
import sys
import json
import argparse
import numpy as np

# Add utils directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_em import compute_score_f1

# Dataset names
dataset_names = ["hotpotqa", "nq_search", "2wikimultihopqa", "musique", "triviaqa", "popqa", "bamboogle"]

# Model names (extracted from file naming patterns)
MODEL_NAMES = ["3b", "3b_s60", "7b_s100"]

# Rerun models (if needed)
rerun_models = []


def handle_dont_know(record):
    """
    Check if the record contains a "don't know" response.
    
    Args:
        record (dict): A single record from the JSONL file.
        
    Returns:
        tuple: (question, messages) if don't know is detected, else (None, None)
    """
    sequences_str = record.get('sequences_str', '')
    answer = record.get('answer', '')
    
    # Check for common "don't know" patterns
    dont_know_patterns = [
        "i don't know",
        "i do not know",
        "i cannot answer",
        "i can't answer",
        "unable to answer",
        "cannot determine",
        "no information",
        "not enough information",
        "insufficient information"
    ]
    
    text_to_check = (sequences_str + " " + answer).lower()
    
    for pattern in dont_know_patterns:
        if pattern in text_to_check:
            question = record.get('question', '')
            messages = record.get('events', [])
            return question, messages
    
    return None, None


def summarize_dataset(dataset_name, base_dir, rerun=False):
    """
    Summarizes the evaluation results for a given dataset.

    Args:
        dataset_name (str): The name of the dataset to summarize.
        base_dir (str): The base directory containing the results.
        rerun (bool): Whether to process rerun model files.
    """
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    if not os.path.isdir(dataset_dir):
        print(f"Error: Directory for dataset '{dataset_name}' not found at '{dataset_dir}'")
        return
    
    summary_data = {
        "dataset_name": dataset_name
    }
    
    model_files = []
    if rerun:
        for rerun_model in rerun_models:
            model_files.extend(["rerun_processed_" + rerun_model + "-" + model_name + ".jsonl" for model_name in MODEL_NAMES])
    else:
        # Automatically detect all jsonl files in the dataset directory
        if os.path.exists(dataset_dir):
            all_files = os.listdir(dataset_dir)
            model_files = [f for f in all_files if f.endswith('.jsonl')]
            if not model_files:
                print(f"Warning: No JSONL files found in {dataset_dir}")
                return
    
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        file_path = os.path.join(dataset_dir, model_file)
        
        scores = []
        f1_scores = []
        dont_know_questions = []
        
        if not os.path.exists(file_path):
            print(f"Warning: File for model '{file_path}' not found")
            continue
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    if "answer" not in record:
                        f1 = compute_score_f1(solution_str=record.get('sequences_str', ''), ground_truth=record.get('ground_truth', {}))
                        f1_scores.append(f1)
                    else:
                        f1 = compute_score_f1(answer=record.get('answer', ''), ground_truth=record.get('ground_truth', {}))
                        f1_scores.append(f1)
                    
                    question, messages = handle_dont_know(record)
                    if question is not None:
                        dont_know_questions.append({
                            "question": question,
                            "messages": messages
                        })
                    
                    scores.append(record.get('reward', 0.0))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from a line in {file_path}")
        
        if scores:
            avg_em_score = np.mean(scores)
            print(f"avg_em_score: {avg_em_score} for model {model_name}, dataset {dataset_name}, total {len(scores)}")
        else:
            avg_em_score = 0.0
        
        if f1_scores:
            avg_f1_score = np.mean(f1_scores)
        else:
            avg_f1_score = 0.0
        
        summary_data[model_name] = {
            "f1_score": avg_f1_score,
            "em_score": avg_em_score,
            "dont_know_questions": dont_know_questions
        }
    
    output_filename = os.path.join(base_dir, f"{dataset_name}_summary.json")
    if os.path.exists(output_filename): 
        # extend the file
        print(f"Extending file {output_filename}")
        with open(output_filename, 'r') as f:
            existing_data = json.load(f)
        existing_data.update(summary_data)
        summary_data = existing_data
        
    with open(output_filename, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Summary for dataset '{dataset_name}' saved to '{output_filename}'")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--base_dir", type=str, default="results")
    args.add_argument("--rerun", action="store_true")
    args = args.parse_args()
    
    for dataset_name in dataset_names:
        summarize_dataset(dataset_name, rerun=args.rerun, base_dir=args.base_dir)

