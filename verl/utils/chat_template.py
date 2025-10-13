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
"""Utilities for managing and verifying chat templates."""

import warnings

# Qwen chat template (Qwen2, Qwen2.5, etc.)
QWEN_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


def ensure_qwen_chat_template(tokenizer, force=False):
    """
    Ensure the tokenizer uses the Qwen chat template.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        force (bool): If True, override existing template with Qwen template
        
    Returns:
        tokenizer: Modified tokenizer with Qwen template
    """
    model_name = getattr(tokenizer, "name_or_path", "")
    
    # Check if it's a Qwen model
    is_qwen = "qwen" in model_name.lower()
    
    if is_qwen:
        # Verify the tokenizer has the correct special tokens
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            warnings.warn(
                f"Qwen model detected but no chat_template found. Setting Qwen template.",
                stacklevel=2
            )
            tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        elif force:
            warnings.warn(
                f"Force-setting Qwen chat template for {model_name}",
                stacklevel=2
            )
            tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        else:
            # Verify it contains Qwen markers
            if '<|im_start|>' not in tokenizer.chat_template:
                warnings.warn(
                    f"Qwen model detected but chat template doesn't contain '<|im_start|>'. "
                    f"This may cause issues. Use force=True to override.",
                    stacklevel=2
                )
        
        # Ensure special tokens are set
        if not hasattr(tokenizer, 'im_start_id') or tokenizer.im_start_id is None:
            # Try to add special tokens
            special_tokens = {
                'additional_special_tokens': ['<|im_start|>', '<|im_end|>']
            }
            tokenizer.add_special_tokens(special_tokens)
            
    return tokenizer


def verify_chat_template(tokenizer, model_type="qwen"):
    """
    Verify that the tokenizer has a valid chat template.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        model_type (str): Expected model type (qwen, llama, etc.)
        
    Returns:
        bool: True if template is valid
    """
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        warnings.warn(
            f"Tokenizer has no chat_template attribute. This may cause issues.",
            stacklevel=2
        )
        return False
    
    template = tokenizer.chat_template
    
    # Model-specific verification
    if model_type.lower() == "qwen":
        if '<|im_start|>' not in template or '<|im_end|>' not in template:
            warnings.warn(
                f"Expected Qwen template with '<|im_start|>' and '<|im_end|>' markers, "
                f"but template doesn't contain them.",
                stacklevel=2
            )
            return False
        print(f"✓ Verified Qwen chat template is active")
        return True
    
    # Generic check - template should have message loop
    if 'messages' not in template or 'role' not in template:
        warnings.warn(
            f"Chat template may be invalid - doesn't contain expected 'messages' or 'role' markers",
            stacklevel=2
        )
        return False
    
    return True


def print_chat_template_example(tokenizer, example_messages=None):
    """
    Print an example of how the chat template formats messages.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        example_messages: Optional list of message dicts. If None, uses a default example.
    """
    if example_messages is None:
        example_messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
    
    try:
        formatted = tokenizer.apply_chat_template(
            example_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("\n" + "="*60)
        print("Chat Template Example:")
        print("="*60)
        print(f"Input: {example_messages}")
        print(f"\nFormatted output:\n{formatted}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error formatting chat template: {e}")

