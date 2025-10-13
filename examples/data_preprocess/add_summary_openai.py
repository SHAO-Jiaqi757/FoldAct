#!/usr/bin/env python3
"""
Add summary field to SFT training data using OpenAI API with batch concurrent processing.
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
<reasoning_summary>[Brief summary of logical steps taken]<reasoning_summary>
<search_results_summary>
[Key Information from Search Results: Copy ALL relevant facts from search results EXACTLY as provided]
- Minimize the context length by only including the most relevant information, including important names, dates, facts
- Use direct quotes from search results
- Preserve the exact wording for critical information
<search_results_summary>


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


async def generate_summary_async(
    client: AsyncOpenAI,
    question_context: str,
    expected_answer: str,
    model: str = "gpt-4o-mini",
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
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples from {input_file}")
    print(f"Using model: {model}")
    print(f"Max concurrent requests: {max_concurrent}")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all samples
    tasks = []
    for sample in samples:
        question_context = sample["extra_info"]["question"]
        expected_answer = sample["extra_info"]["answer"]
        
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
    for sample, summary in zip(samples, summaries):
        sample["summary"] = summary
    
    # Write output
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Done! Processed {len(samples)} samples.")
    
    # Show sample summary
    if samples:
        print("\n" + "="*80)
        print("EXAMPLE SUMMARY:")
        print("="*80)
        print(f"Question (first 200 chars):\n{samples[0]['extra_info']['question'][:200]}...")
        print(f"\nGenerated Summary (first 500 chars):\n{samples[0]['summary'][:500]}...")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Add summary field to SFT data using OpenAI API"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file"
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
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
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

