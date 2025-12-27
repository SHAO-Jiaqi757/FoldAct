"""
完整的 Inference 框架
整合了 generation.py 的 rollout 流程和 simple_eval.py 的 inference 模式
支持：
1. 多轮对话生成
2. 滑动窗口上下文管理
3. Summary 检测和上下文压缩
4. 事件记录和追踪
"""

import argparse
import json
import re
import torch
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import asyncio
import logging
import numpy as np

from .generation import (
    LLMGenerationManager, 
    GenerationConfig, 
    ResponseType
)
from utils.qa_em import compute_score_em
from utils.model_loading import select_checkpoint_path, _normalize_local_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== JSON SERIALIZATION HELPERS ==========
def _convert_numpy_to_native(obj):
    """
    递归地将对象转换为 JSON 可序列化的格式
    参考 simple_eval.py 的实现
    处理 numpy 数组、numpy 标量、tensor 等类型
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist() if obj.numel() > 0 else []
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_native(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_native(elem) for elem in obj)
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


# ========== PROMPT TEMPLATES ==========
FEW_SHOT_TEMPLATE = """<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and 
it will return the top searched results between <information> and </information>. You can search as many times as your want. 
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, 
without detailed illustrations. For example, <answer> Beijing </answer>.
This is a few-shot learning exercise. Examples are provided below.
Question: What is the birth date of the lead singer of Coldplay?<think>I need to find out the lead singer of Coldplay and their birth date.</think><search>lead singer of Coldplay birth date</search><information>Doc 1(Title: Chris Martin) Christopher Anthony John Martin (born 2 March 1977) is an English singer, songwriter, and musician. He is the lead singer, pianist, rhythm guitarist, and co-founder of the rock band Coldplay.
</information><think>The lead singer of Coldplay is Chris Martin, and he was born on March 2, 1977.</think><answer>March 2, 1977</answer>

Question: What is the most populous city in the United States?
<think>I need to determine which city in the United States has the largest population.</think><search>most populous city in the United States</search>
<information>Doc 1(Title: New York City) New York City is the most populous city in the United States, with an estimated population of over 8.3 million people.
</information><think>New York City is the most populous city in the United States.</think><answer>New York City</answer>

Question: {question}
<|im_end|>
"""

ZERO_SHOT_TEMPLATE = """<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}<|im_end|>
"""

PROMPT_TEMPLATES = {
    "fewshot": FEW_SHOT_TEMPLATE,
    "zeroshot": ZERO_SHOT_TEMPLATE,
}


# ========== SEARCH UTILITIES ==========
def batch_search(queries: list, retriever_url: str, top_k: int) -> List[str]:
    """执行批量搜索"""
    if not queries:
        return []
    try:
        payload = {"queries": queries, "topk": top_k, "return_scores": True}
        response = requests.post(retriever_url, json=payload, timeout=30)
        response.raise_for_status()
        results = response.json().get('result', [])

        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result or []):
                content = doc_item.get('document', {}).get('contents', '')
                title = content.split("\n")[0] if content else ""
                text = "\n".join(content.split("\n")[1:]) if len(content.split("\n")) > 1 else content
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference
        
        if len(results) != len(queries):
            logger.warning(f"Number of search results ({len(results)}) does not match number of queries ({len(queries)}).")
            return [""] * len(queries)

        return [_passages2string(res) for res in results]
    except Exception as e:
        logger.error(f"Error in batch search: {e}")
        return [""] * len(queries)


# ========== INFERENCE MANAGER ==========
class InferenceManager:
    """
    完整的 Inference 管理器
    整合了 rollout 流程和 summary 管理
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: GenerationConfig,
        retriever_url: str,
        top_k: int = 3,
        do_search: bool = True,
        use_sliding_window: bool = True,
        enable_debug_logs: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.retriever_url = retriever_url
        self.top_k = top_k
        self.do_search = do_search
        self.use_sliding_window = use_sliding_window
        self.enable_debug_logs = enable_debug_logs
        
        # 初始化 GenerationManager（用于复杂的 rollout 流程）
        # 注意：这里我们主要使用其逻辑，但简化实现
        self.gen_manager = None  # 可选：如果需要完整的 GenerationManager
        
    def parse_action(self, text: str) -> Tuple[Optional[str], str]:
        """
        解析响应中的 action 类型和内容
        优先级: search > answer > think_summary > information_summary > think
        """
        tag_pattern = re.compile(
            r'<\s*(search|answer|think|think_summary|information_summary)\b[^>]*>(.*?)</\s*\1\s*>',
            re.IGNORECASE | re.DOTALL
        )
        
        action = None
        content = ''
        found = False
        
        for match in tag_pattern.finditer(text):
            tag = match.group(1).lower()
            content_candidate = match.group(2).strip()
            
            if tag == 'search':
                if content_candidate and len(content_candidate) > 5:
                    action = 'search'
                    content = content_candidate
                    found = True
                    break
                else:
                    continue
            elif tag == 'answer':
                action = 'answer'
                content = content_candidate
                found = True
                break
            elif tag in ['think_summary', 'information_summary']:
                if not found:
                    action = tag
                    content = content_candidate
                    found = True
            elif tag == 'think':
                if not found:
                    action = 'think'
                    content = content_candidate
        
        return action, content
    
    def check_has_summary(self, text: str) -> bool:
        """检查文本中是否包含 summary tags"""
        summary_pattern = re.compile(
            r'<\s*(think_summary|information_summary)\b[^>]*>.*?</\s*\1\s*>',
            re.IGNORECASE | re.DOTALL
        )
        return bool(summary_pattern.search(text))
    
    def apply_context_compression(
        self, 
        turn_history: List[Dict],
        initial_question: str,
        max_length: int = 7000
    ) -> str:
        """
        应用上下文压缩（滑动窗口）
        策略：
        1. 如果最后有 information，找到最近的 summary turn
        2. 从 summary turn 到最后一个 info turn 保留
        3. 如果没有 summary，保留所有 turns
        4. 如果仍然太长，进行激进截断
        """
        if not turn_history:
            return initial_question
        
        # 分析每个 turn
        has_info_flags = []
        for turn in turn_history:
            has_info = turn.get('has_real_info', False)
            has_info_flags.append(has_info)
        
        # 找到最后一个有 information 的 turn
        last_info_idx = -1
        for i in range(len(turn_history) - 1, -1, -1):
            if has_info_flags[i]:
                last_info_idx = i
                break
        
        # 选择要保留的 turns
        if last_info_idx == -1:
            # 没有 information → 保留所有
            selected_turns = turn_history
        else:
            # 查找最近的 summary turn
            most_recent_summary_idx = None
            for i in range(last_info_idx, -1, -1):
                if self.check_has_summary(turn_history[i].get('response', '')):
                    most_recent_summary_idx = i
                    break
            
            if most_recent_summary_idx is not None:
                # 从 summary turn 到最后一个 info turn
                selected_turns = turn_history[most_recent_summary_idx:last_info_idx + 1]
            else:
                # 没有 summary → 保留所有
                selected_turns = turn_history
        
        # 构建压缩后的上下文
        compressed_parts = [initial_question]
        for turn in selected_turns:
            compressed_parts.append(turn.get('response', ''))
            if turn.get('observation'):
                compressed_parts.append(turn.get('observation', ''))
        
        compressed_context = '\n\n'.join(compressed_parts)
        
        # 如果仍然太长，进行激进截断
        # 简单估算：假设平均每个字符约 0.25 tokens
        estimated_tokens = len(compressed_context) * 0.25
        if estimated_tokens > max_length:
            # 只保留最近的几个 turns
            truncated_turns = selected_turns[-3:] if len(selected_turns) > 3 else selected_turns
            compressed_parts = [initial_question]
            for turn in truncated_turns:
                compressed_parts.append(turn.get('response', ''))
                if turn.get('observation'):
                    compressed_parts.append(turn.get('observation', ''))
            compressed_context = '\n\n'.join(compressed_parts)
        
        return compressed_context
    
    def generate_response(
        self, 
        prompt: Union[str, List[str]], 
        max_new_tokens: int = 500,
        temperature: float = 0.7
    ) -> Union[str, List[str]]:
        """
        生成模型响应，支持单个或批量输入
        
        Args:
            prompt: 单个 prompt 字符串或 prompt 列表
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Returns:
            单个字符串或字符串列表
        """
        # 处理单个或批量输入
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=8192
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=temperature,
            )
        
        # Decode
        input_lengths = inputs.input_ids.shape[1]
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, input_lengths:], 
            skip_special_tokens=True
        )
        
        # 在 </search> 或 </answer> 处停止
        processed_texts = []
        for generated_text in generated_texts:
            if '</search>' in generated_text:
                generated_text = generated_text.split('</search>')[0] + '</search>'
            elif '</answer>' in generated_text:
                generated_text = generated_text.split('</answer>')[0] + '</answer>'
            processed_texts.append(generated_text)
        
        # 返回单个或批量结果
        return processed_texts if is_batch else processed_texts[0]
    
    def execute_inference(
        self,
        initial_prompt: str,
        max_turns: int = 5,
        max_new_tokens: int = 500
    ) -> Dict:
        """
        执行完整的 inference 流程（单个样本）
        
        Returns:
            dict: 包含完整对话历史、最终响应、统计信息等
        """
        current_prompt = initial_prompt
        turn_history: List[Dict] = []
        active = True
        force_summarized = False
        
        # 事件记录
        events = []
        
        for turn in range(max_turns):
            if not active:
                break
            
            if self.enable_debug_logs:
                logger.info(f"\n{'='*60}")
                logger.info(f"[TURN {turn + 1}/{max_turns}]")
                logger.info(f"{'='*60}")
            
            # 应用上下文压缩（如果启用滑动窗口）
            if self.use_sliding_window and turn > 0:
                compressed_context = self.apply_context_compression(
                    turn_history, 
                    initial_prompt
                )
                current_prompt = compressed_context
                if self.enable_debug_logs:
                    logger.info(f"[CONTEXT COMPRESSION] Compressed context length: {len(compressed_context)} chars")
            
            # 添加 assistant 开始标记
            if not current_prompt.endswith('\n<|im_start|>assistant\n'):
                current_prompt += '\n<|im_start|>assistant\n'
            
            # 生成响应
            response = self.generate_response(
                current_prompt,
                max_new_tokens=max_new_tokens
            )
            
            # 更新当前 prompt
            current_prompt += response
            
            # 解析 action
            action, content = self.parse_action(response)
            
            if self.enable_debug_logs:
                logger.info(f"[ACTION] {action}: {content[:100]}...")
            
            # 记录事件
            events.append({
                'turn': turn,
                'action': action,
                'content': content,
                'response': response
            })
            
            # 处理 action
            observation = ''
            turn_has_real_info = False
            
            if action == 'answer':
                # 结束对话
                active = False
                observation = ''
            elif action == 'search' and self.do_search:
                # 执行搜索
                if content.strip():
                    search_results = batch_search([content], self.retriever_url, self.top_k)
                    search_result = search_results[0].strip() if search_results else ''
                    
                    turn_info = f"\n[Turn {turn + 1}/{max_turns}] "
                    
                    # 检查是否是最后几轮
                    if max_turns and turn >= max_turns - 2:
                        sharp_message = (
                            f"\n\nThis is my LAST turn (Turn {turn + 1}/{max_turns}). "
                            f"I MUST provide final answer now with <answer> and </answer>."
                        )
                        observation = (
                            f'{turn_info}user\n<information>{search_result}</information>\n\n'
                            f'assistant\n{sharp_message}\n'
                        )
                    else:
                        # 添加 summary prompt
                        summary_prompt = (
                            "I will provide concise, high-level summaries of both my previous reasoning and the gathered information.\n"
                            "Use the <think_summary>...</think_summary> tag for my thought process summary,\n"
                            "and the <information_summary>...</information_summary> tag for key retrieved facts or evidence.\n"
                            "Focus on clarity and brevity to help guide my next response. Or I will use <answer> and </answer> to provide the final answer if information is enough."
                        )
                        observation = (
                            f'{turn_info}user\n<information>{search_result}</information>\n\n'
                            f'assistant\n{summary_prompt}\n'
                        )
                    
                    turn_has_real_info = len(search_result) > 20
                    current_prompt += observation
                else:
                    # 空搜索查询
                    turn_info = f"\n[Turn {turn + 1}/{max_turns}] "
                    observation = f"{turn_info}user\n<information></information>\n\nassistant\n"
                    current_prompt += observation
            elif action in ['think', 'think_summary', 'information_summary']:
                # 思考或总结，继续对话
                turn_info = f"\n[Turn {turn + 1}/{max_turns}] "
                
                if max_turns and turn >= max_turns - 2:
                    sharp_message = (
                        f"\n\nThis is my LAST turn (Turn {turn + 1}/{max_turns}). "
                        f"I MUST provide final answer now with <answer> and </answer>.\n\n"
                    )
                    observation = f'{turn_info}{sharp_message}'
                else:
                    observation = f'{turn_info}'
                
                current_prompt += observation
            else:
                # 无效 action，结束对话
                active = False
                observation = ''
            
            # 保存 turn 历史
            turn_history.append({
                'turn': turn,
                'response': response,
                'observation': observation,
                'has_real_info': turn_has_real_info,
                'action': action
            })
            
            # 检查观察是否太长（强制 summary）
            if observation and len(observation) > 2000:  # 简单阈值
                force_summarized = True
                if self.enable_debug_logs:
                    logger.warning(f"[OBSERVATION TOO LONG] Forcing summarized context")
        
        # 最终响应
        final_response = current_prompt
        
        return {
            'final_response': final_response,
            'turn_history': turn_history,
            'events': events,
            'num_turns': len(turn_history),
            'final_action': turn_history[-1]['action'] if turn_history else None
        }


# ========== MAIN INFERENCE FUNCTION ==========
def run_inference(
    model,
    tokenizer,
    questions: List[str],
    ground_truths: List[List[str]],
    config: Dict,
    output_path: str
):
    """
    运行批量 inference，支持 batch_size 批量处理
    
    Args:
        model: 模型
        tokenizer: tokenizer
        questions: 问题列表
        ground_truths: 真实答案列表
        config: 配置字典，包含 batch_size 等参数
        output_path: 输出文件路径
    """
    # 检查是否有已保存的结果，用于断点续传
    processed_questions = set()
    import os
    if os.path.exists(output_path):
        logger.info(f"Found existing output file: {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        processed_questions.add(result['question'])
            logger.info(f"Loaded {len(processed_questions)} already processed questions")
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
            logger.info("Starting from scratch...")
            processed_questions = set()
    
    # 过滤出未处理的问题
    if processed_questions:
        # 创建未处理的问题列表
        unprocessed_indices = [i for i, q in enumerate(questions) if q not in processed_questions]
        if not unprocessed_indices:
            logger.info("All questions have been processed. Nothing to do.")
            return []
        
        logger.info(f"Skipping {len(processed_questions)} processed questions")
        logger.info(f"Resuming with {len(unprocessed_indices)} remaining questions")
        
        # 更新问题和答案列表
        questions = [questions[i] for i in unprocessed_indices]
        ground_truths = [ground_truths[i] for i in unprocessed_indices]
    
    # 获取 batch_size，默认为 1
    batch_size = config.get('batch_size', 1)
    
    # 创建 InferenceManager
    gen_config = GenerationConfig(
        max_turns=config.get('max_turns', 5),
        max_start_length=config.get('max_start_length', 2048),
        max_prompt_length=config.get('max_prompt_length', 8192),
        max_response_length=config.get('max_response_length', 500),
        max_obs_length=config.get('max_obs_length', 2000),
        num_gpus=config.get('num_gpus', 1),
        use_summary=config.get('use_summary', False),  # Enable summary token generation (<think_summary>, <information_summary>)
        enable_debug_logs=config.get('enable_debug_logs', False),
        search_url=config.get('retriever_url'),
        topk=config.get('top_k', 3)
    )
    
    inference_manager = InferenceManager(
        model=model,
        tokenizer=tokenizer,
        config=gen_config,
        retriever_url=config.get('retriever_url', 'http://10.201.8.114:8000/retrieve'),
        top_k=config.get('top_k', 3),
        do_search=config.get('do_search', True),
        use_sliding_window=config.get('use_sliding_window', True),  # Enable sliding window context compression
        enable_debug_logs=config.get('enable_debug_logs', False)
    )
    
    # 选择 prompt template
    template_name = config.get('prompt_template', 'zeroshot')
    template = PROMPT_TEMPLATES.get(template_name, ZERO_SHOT_TEMPLATE)
    
    results = []
    total_questions = len(questions)
    
    # 按 batch_size 分批处理
    if batch_size > 1:
        logger.info(f"Using batch processing with batch_size={batch_size}")
        num_batches = (total_questions + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_questions)
            
            batch_questions = questions[start_idx:end_idx]
            batch_ground_truths = ground_truths[start_idx:end_idx] if ground_truths else [[]] * len(batch_questions)
            
            # 构建批量 prompts
            batch_prompts = [template.format(question=q) for q in batch_questions]
            
            # 批量处理每个问题（注意：多轮对话仍需要逐个处理，但每轮的生成可以批量）
            batch_results = []
            for i, (question, ground_truth, initial_prompt) in enumerate(zip(
                batch_questions, batch_ground_truths, batch_prompts
            )):
                # 执行 inference（单个样本的多轮对话）
                inference_result = inference_manager.execute_inference(
                    initial_prompt=initial_prompt,
                    max_turns=config.get('max_turns', 5),
                    max_new_tokens=config.get('max_new_tokens', 500)
                )
                
                # 计算分数
                score = compute_score_em(
                    solution_str=inference_result['final_response'],
                    ground_truth={'target': ground_truth}
                )
                
                # 保存结果（确保所有数据都是 JSON 可序列化的）
                # 参考 simple_eval.py 的处理方式
                serializable_ground_truth = _convert_numpy_to_native({'target': ground_truth})
                result_item = {
                    "question": question,
                    "sequences_str": inference_result['final_response'],
                    "ground_truth": serializable_ground_truth,
                    "reward": _convert_numpy_to_native(score),
                    "num_turns": int(inference_result['num_turns']),
                    "final_action": inference_result['final_action'],
                    "events": _convert_numpy_to_native(inference_result['events'])
                }
                
                batch_results.append(result_item)
                results.append(result_item)
                
                # 写入文件（实时保存）
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            
            if inference_manager.enable_debug_logs:
                logger.info(f"Completed batch {batch_idx + 1}/{num_batches} ({len(batch_questions)} questions)")
    else:
        # 单个处理（原始逻辑）
        logger.info(f"Using single sample processing (batch_size=1)")
        for i in tqdm(range(len(questions)), desc="Running inference"):
            question = questions[i]
            ground_truth = ground_truths[i] if i < len(ground_truths) else []
            
            # 构建初始 prompt
            initial_prompt = template.format(question=question)
            
            # 执行 inference
            inference_result = inference_manager.execute_inference(
                initial_prompt=initial_prompt,
                max_turns=config.get('max_turns', 5),
                max_new_tokens=config.get('max_new_tokens', 500)
            )
            
            # 计算分数
            score = compute_score_em(
                solution_str=inference_result['final_response'],
                ground_truth={'target': ground_truth}
            )
            
            # 保存结果（确保所有数据都是 JSON 可序列化的）
            # 参考 simple_eval.py 的处理方式
            serializable_ground_truth = _convert_numpy_to_native({'target': ground_truth})
            result_item = {
                "question": question,
                "sequences_str": inference_result['final_response'],
                "ground_truth": serializable_ground_truth,
                "reward": _convert_numpy_to_native(score),
                "num_turns": int(inference_result['num_turns']),
                "final_action": inference_result['final_action'],
                "events": _convert_numpy_to_native(inference_result['events'])
            }
            
            results.append(result_item)
            
            # 写入文件
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
    
    return results


# ========== MAIN ENTRY POINT ==========
def main():
    parser = argparse.ArgumentParser(description="完整的 Inference 框架")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF repo ID or local model dir")
    parser.add_argument("--ckpt_root", type=str, default=None,
                        help="Training run root containing global_step_* subdirs")
    parser.add_argument("--ckpt_step", type=str, default="latest",
                        help="Checkpoint step to load (integer) or 'latest'")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to parquet data file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (for debugging)")
    
    # Generation arguments
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Maximum number of turns")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="Maximum new tokens per turn")
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help="Batch size for processing multiple questions. "
                             "When > 1, questions are processed in batches for better GPU utilization. "
                             "Note: Each question still goes through multi-turn dialogue individually.")
    
    # Search arguments
    parser.add_argument("--retriever_url", type=str, 
                        default="http://10.201.8.114:8000/retrieve",
                        help="URL for search retriever service")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top search results")
    parser.add_argument("--do_search", action='store_true',
                        help="Enable search functionality")
    
    # Context management arguments
    parser.add_argument("--use_summary", action='store_true',
                        help="Enable summary token generation (<think_summary>, <information_summary>)")
    parser.add_argument("--use_sliding_window", action='store_true',
                        help="Enable sliding window context compression")
    parser.add_argument("--enable_debug_logs", action='store_true',
                        help="Enable debug logging")
    
    # Prompt arguments
    parser.add_argument("--prompt_template", type=str, default="zeroshot",
                        choices=["fewshot", "zeroshot"],
                        help="Prompt template to use")
    
    args = parser.parse_args()
    
    # 解析模型路径
    model_path = args.model_path
    if args.ckpt_root:
        model_path = select_checkpoint_path(args.ckpt_root, args.ckpt_step)
    elif model_path:
        model_path = _normalize_local_path(model_path)
    
    logger.info(f"Loading model from: {model_path}")
    
    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        padding_side='left',
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    
    # 加载数据
    df = pd.read_parquet(args.data_path)
    if args.num_samples:
        df = df.head(args.num_samples)
    
    questions = df['question'].tolist()
    # 加载 ground_truths（会在保存时进行序列化处理）
    ground_truths = df['golden_answers'].tolist()
    
    # 准备配置
    config = {
        'max_turns': args.max_turns,
        'max_new_tokens': args.max_new_tokens,
        'retriever_url': args.retriever_url,
        'top_k': args.top_k,
        'do_search': args.do_search,
        'use_summary': args.use_summary,
        'use_sliding_window': args.use_sliding_window,
        'enable_debug_logs': args.enable_debug_logs,
        'prompt_template': args.prompt_template,
        'max_start_length': 2048,
        'max_prompt_length': 8192,
        'max_response_length': 500,
        'max_obs_length': 2000,
        'num_gpus': 1,
        'batch_size': args.val_batch_size  # 添加 batch_size 配置
    }
    
    # 运行 inference
    logger.info(f"Starting inference on {len(questions)} questions")
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        ground_truths=ground_truths,
        config=config,
        output_path=args.output_path
    )
    
    # 计算统计信息
    total_reward = sum(r['reward'] for r in results)
    avg_reward = total_reward / len(results) if results else 0
    avg_turns = sum(r['num_turns'] for r in results) / len(results) if results else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Inference Complete!")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Average reward: {avg_reward:.4f}")
    logger.info(f"Average turns: {avg_turns:.2f}")
    logger.info(f"Results saved to: {args.output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

