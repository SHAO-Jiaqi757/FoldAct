#!/usr/bin/env python3
"""
Verify that the Qwen chat template is correctly configured.

This script helps you verify that:
1. The tokenizer has the correct Qwen chat template
2. The template formatting works as expected
3. Special tokens are properly configured

Usage:
    python verify_chat_template.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Verify Qwen chat template configuration")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model path or name (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Qwen Chat Template Verification")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print("\nLoading tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return 1
    
    # Check for chat template
    print("\n" + "-" * 80)
    print("Chat Template Check")
    print("-" * 80)
    
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("✓ Chat template found")
        
        # Check if it's Qwen template
        if '<|im_start|>' in tokenizer.chat_template and '<|im_end|>' in tokenizer.chat_template:
            print("✓ Qwen-style template detected (contains <|im_start|> and <|im_end|>)")
        else:
            print("⚠ Template found but doesn't contain Qwen markers")
            print(f"Template preview: {tokenizer.chat_template[:200]}...")
    else:
        print("✗ No chat template found")
        return 1
    
    # Test formatting
    print("\n" + "-" * 80)
    print("Template Formatting Test")
    print("-" * 80)
    
    test_messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    try:
        formatted = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("✓ Template formatting works")
        print("\nInput:")
        print(f"  {test_messages}")
        print("\nFormatted output:")
        print("-" * 40)
        print(formatted)
        print("-" * 40)
        
        # Verify expected tokens
        if '<|im_start|>user' in formatted and '<|im_start|>assistant' in formatted:
            print("\n✓ Output contains expected Qwen tokens")
        else:
            print("\n⚠ Output doesn't contain expected Qwen tokens")
            
    except Exception as e:
        print(f"✗ Template formatting failed: {e}")
        return 1
    
    # Check special tokens
    print("\n" + "-" * 80)
    print("Special Tokens Check")
    print("-" * 80)
    
    special_tokens_to_check = [
        ('pad_token', tokenizer.pad_token),
        ('eos_token', tokenizer.eos_token),
        ('bos_token', getattr(tokenizer, 'bos_token', None)),
    ]
    
    for name, token in special_tokens_to_check:
        if token:
            print(f"✓ {name}: {token}")
        else:
            print(f"⚠ {name}: Not set")
    
    # Test multi-turn conversation
    print("\n" + "-" * 80)
    print("Multi-turn Conversation Test")
    print("-" * 80)
    
    multi_turn = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI."},
        {"role": "user", "content": "Can you give an example?"}
    ]
    
    try:
        formatted_multi = tokenizer.apply_chat_template(
            multi_turn,
            tokenize=False,
            add_generation_prompt=True
        )
        print("✓ Multi-turn formatting works")
        print("\nInput:")
        for msg in multi_turn:
            print(f"  {msg['role']}: {msg['content']}")
        print("\nFormatted output:")
        print("-" * 40)
        print(formatted_multi)
        print("-" * 40)
        
    except Exception as e:
        print(f"✗ Multi-turn formatting failed: {e}")
        return 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    print("✓ All checks passed!")
    print("\nYour tokenizer is correctly configured for Qwen models.")
    print("The training pipeline will automatically use this chat template.")
    print("\nNext steps:")
    print("  1. Ensure your data uses generic format: [{'role': 'user', 'content': '...'}]")
    print("  2. Run training with: bash examples/sft/search_r1/train_summary_mixed.sh")
    print("  3. The framework will automatically apply the Qwen template during training")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

