#!/usr/bin/env python3
"""
Add summary field to Multi-turn SFT training data using OpenAI API with batch concurrent processing.
Only processes samples with turn_index >= 1.
Summary should:
1. State the question
2. Summarize reasoning logic
3. Preserve complete and useful information from search results (DO NOT modify)
4. Ensure model can produce the correct answer based on the summarized context
"""

import argparse
import asyncio
import json
import os
from typing import List, Dict, Any
from tqdm.asyncio import tqdm as atqdm
from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()


SUMMARY_PROMPT_TEMPLATE = """You are creating a training data summary for a reasoning and search model.

Your task: Create a summary that enables the model to produce the correct next action/answer.
---

**Input Context:**
{input_data}

**Expected Next Action/Answer:**
{expected_answer}

---

**Structure:**

Question: [State the question clearly]
<think_summary>[Brief summary of logical steps taken]</think_summary>
<information_summary>
[Key Information from Search Results: Copy ALL relevant facts from search results EXACTLY as provided]
- Minimize the context length by only including the most relevant information, including important names, dates, facts
- Use direct quotes from search results
- Preserve the exact wording for critical information
</information_summary>


**Critical Requirements:**
1. Minimize the context length under following rules.
2. **State the Question**: Clearly present what needs to be answered
3. **Summarize Reasoning**: Briefly describe the logical thinking process
4. **Preserve Search Results**: Copy ALL key information from search results VERBATIM
   - DO NOT paraphrase, rewrite, or modify any facts from search results
   - Include complete relevant information (names, dates, facts, relationships)
   - Use exact quotes when possible for critical facts
5. **Enable Correct Answer**: The summary must contain sufficient information for the model to generate the expected answer
6. DO NOT include Expected Answer in the summary


Generate the summary following the structure above:"""


def extract_context_from_multiturn(sample: Dict[str, Any]) -> str:
    """Extract context from multiturn format for summary generation."""
    messages = sample["prompt"]
    
    # Build context from all messages
    context_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            if content.startswith("Question:"):
                context_parts.append(content)
            elif content.startswith("<information>"):
                context_parts.append(content)
        elif role == "assistant":
            if content.startswith("<think>") or content.startswith("<search>"):
                context_parts.append(content)
    
    return "\n\n".join(context_parts)


async def generate_summary_async(
    client: AsyncOpenAI,
    question_context: str,
    expected_answer: str,
    model: str = "gpt-4.1-mini",
    semaphore: asyncio.Semaphore = None
) -> str:
    """Generate summary using OpenAI API asynchronously."""
    
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        input_data=question_context,
        expected_answer=expected_answer
    )
    
    # Use semaphore to limit concurrent requests
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates training data summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\nError generating summary: {e}")
            return f"Error: {str(e)}"


async def process_batch_async(
    input_file: str,
    output_file: str,
    api_key: str,
    model: str,
    max_concurrent: int,
    limit: int = None
):
    """Process all samples with concurrent API calls."""
    
    # Initialize OpenAI async client
    client = AsyncOpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_URL"))
    
    # Read all samples
    all_samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))
    
    print(f"Loaded {len(all_samples)} total samples from {input_file}")
    
    # Filter samples with turn_index >= 1
    samples_to_process = []
    for sample in all_samples:
        turn_index = sample.get("extra_info", {}).get("turn_index", -1)
        if turn_index >= 1:
            samples_to_process.append(sample)
    
    print(f"Found {len(samples_to_process)} samples with turn_index >= 1 to process")
    print(f"Using model: {model}")
    print(f"Max concurrent requests: {max_concurrent}")
    
    if not samples_to_process:
        print("No samples to process. Copying input to output...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        return
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for samples that need summaries
    tasks = []
    for sample in samples_to_process:
        question_context = extract_context_from_multiturn(sample)
        expected_answer = sample["answer"]
        
        task = generate_summary_async(
            client=client,
            question_context=question_context,
            expected_answer=expected_answer,
            model=model,
            semaphore=semaphore
        )
        tasks.append(task)
    
    # Execute all tasks concurrently with progress bar
    print("\nGenerating summaries...")
    summaries = await atqdm.gather(*tasks, desc="Processing")
    
    # Add summaries to samples
    summary_index = 0
    for sample in all_samples:
        turn_index = sample.get("extra_info", {}).get("turn_index", -1)
        if turn_index >= 1:
            sample["summary"] = summaries[summary_index]
            summary_index += 1
    
    # Write output
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Done! Processed {len(samples_to_process)} samples with summaries.")
    print(f"Total samples in output: {len(all_samples)}")
    
    # Show sample summary
    if samples_to_process:
        print("\n" + "="*80)
        print("EXAMPLE SUMMARY:")
        print("="*80)
        first_sample = samples_to_process[0]
        print(f"Turn Index: {first_sample.get('extra_info', {}).get('turn_index', 'N/A')}")
        print(f"Question (first 200 chars):\n{first_sample.get('extra_info', {}).get('question', '')[:200]}...")
        print(f"\nGenerated Summary (first 500 chars):\n{first_sample['summary'][:500]}...")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Add summary field to Multi-turn SFT data using OpenAI API"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file (multiturn format)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file with summaries"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to use (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests (default: 10)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N samples (for testing)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. Set --api_key or OPENAI_API_KEY environment variable"
        )
    
    # Run async processing
    asyncio.run(process_batch_async(
        input_file=args.input_file,
        output_file=args.output_file,
        api_key=api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
        limit=args.limit
    ))


if __name__ == "__main__":
    main()

