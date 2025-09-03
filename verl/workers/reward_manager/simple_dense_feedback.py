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

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np

from verl import DataProto
from verl.tools.schemas import TrajectoryComponent, TrajectoryFeedback
from verl.utils.reward_score import default_compute_score
from .llm_evaluator import LLMEvaluator

# 设置debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置文件日志记录器
file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)

# 创建文件处理器
import os
from datetime import datetime

# 确保logs目录存在
os.makedirs('logs', exist_ok=True)

# 创建带时间戳的日志文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/reward_manager_{timestamp}.log'

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 设置文件日志格式
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(file_formatter)

# 添加文件处理器到logger
file_logger.addHandler(file_handler)
file_logger.propagate = False  # 避免重复输出到控制台

logger.info(f"Reward manager logging to file: {log_filename}")

class SimpleDenseFeedbackRewardManager:
    """Simplified reward manager focused on grounding, information sufficiency, and refinement."""
    
    def __init__(self, tokenizer, num_examine=2, compute_score=None, reward_fn_key="data_source", 
                 enable_llm_evaluation=True, llm_model="gpt-4.1-mini"):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.enable_llm_evaluation = enable_llm_evaluation
        
        # 初始化LLM评估器
        if self.enable_llm_evaluation:
            try:
                self.llm_evaluator = LLMEvaluator(model=llm_model)
                logger.info(f"LLM Evaluator initialized with model: {llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM Evaluator: {e}")
                self.enable_llm_evaluation = False
                self.llm_evaluator = None
        else:
            self.llm_evaluator = None
        
        # 简化的配置
        self.config = self._get_simplified_config()
        
        # 设置日志级别
        self._setup_logging()
        
        # 定义patterns用于解析轨迹组件
        self.patterns = {
            "search": r"<search>(.*?)</search>",
            "information": r"<information>(.*?)</information>",
            "answer": r"<answer>(.*?)</answer>",
        }
    
        logger.info(f"SimpleDenseFeedbackRewardManager initialized with simplified config")
        logger.info(f"Using patterns for: {list(self.patterns.keys())}")
        logger.info(f"LLM evaluation enabled: {self.enable_llm_evaluation}")
    
    def _get_simplified_config(self):
        """获取简化的配置"""
        return {
            "max_tool_steps": 5,
            "insufficient_info_penalty": -0.5,        # 信息不足时直接回答的惩罚
            "refinement_bonus": 0.3,                  # 从不足到充足的改进奖励
            "grounding_bonus": 0.4,                   # 正确grounding的奖励
            "ungrounded_penalty": -0.3,               # 未grounded的惩罚
            # 防止奖励黑客：限制与衰减
            "grounding_bonus_max_steps": 2,           # grounding奖励最多应用的步数（例如只奖励前2次）
            "grounding_bonus_decay": 0.85,            # grounding奖励的衰减系数，按步数递减
            "max_grounding_bonus_total": 1.0,         # grounding奖励在一个trajectory中的总上限
            "min_component_length": 5,
            "max_log_length": 200,
            "enable_debug_logs": True,
            "reward_allocation_strategy": "component_based",
            "min_reward_value": -1.0,
            "max_reward_value": 2.0,
            "smooth_reward_transition": True,
        }
    
    def _setup_logging(self):
        """设置日志级别和格式"""
        if self.config.get("enable_debug_logs", True):
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger()
        if logger.handlers:
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def parse_trajectory_components(self, response_str: str, response_tokens: List[int]) -> List[TrajectoryComponent]:
        """Parse trajectory components with accurate token positioning."""
        logger.debug(f"Parsing response with {len(response_tokens)} tokens")
        
        # 记录到文件
        file_logger.info(f"=== PARSING TRAJECTORY COMPONENTS ===")
        file_logger.info(f"Response string: {response_str}")
        file_logger.info(f"Response tokens count: {len(response_tokens)}")
        
        components = []
        step_number = 1
        
        for component_type, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, response_str, re.DOTALL))
            for i, match in enumerate(matches):
                content = match.group(1).strip()
                if len(content) < self.config.get("min_component_length", 5):
                    logger.debug(f"Skipping {component_type} component {i+1} (too short: {len(content)})")
                    continue
                
                # 计算准确的token位置
                start_char = match.start()
                end_char = match.end()
                
                # 将字符位置转换为token位置
                start_token_idx = self._char_to_token_position(response_str, response_tokens, start_char)
                end_token_idx = self._char_to_token_position(response_str, response_tokens, end_char)
                
                if start_token_idx is None or end_token_idx is None:
                    logger.warning(f"Could not determine token positions for {component_type} component")
                    continue
                
                component = TrajectoryComponent(
                    component_type=component_type,
                    content=content,
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                    step_number=step_number
                )
                components.append(component)
                step_number += 1
        
                logger.debug(f"Found {component_type} component at tokens {start_token_idx}-{end_token_idx}: {content[:50]}...")
                
                # 记录到文件
                file_logger.info(f"Component {step_number-1}: {component_type}")
                file_logger.info(f"  Content: {content}")
                file_logger.info(f"  Token range: {start_token_idx}-{end_token_idx}")
                file_logger.info(f"  Step number: {step_number-1}")
        
        # 按token位置排序
        components.sort(key=lambda x: x.start_token_idx)
        logger.info(f"Parsed {len(components)} trajectory components")
        
        # 记录解析结果到文件
        file_logger.info(f"Total components parsed: {len(components)}")
        file_logger.info(f"Component types: {[c.component_type for c in components]}")
        file_logger.info("=" * 50)
        
        return components
    
    def _char_to_token_position(self, text: str, tokens: List[int], char_pos: int) -> Optional[int]:
        """将字符位置转换为token位置"""
        try:
            # 解码到指定字符位置
            partial_text = text[:char_pos]
            partial_tokens = self.tokenizer.encode(partial_text, add_special_tokens=False)
            return len(partial_tokens)
        except Exception as e:
            logger.warning(f"Error converting char position {char_pos} to token position: {e}")
            return None
    
    async def analyze_trajectory(self, components: List[TrajectoryComponent], 
                                ground_truth, question: str = None) -> TrajectoryFeedback:
        """分析整个轨迹并计算feedback，支持异步LLM评估"""
        logger.info(f"Analyzing trajectory with {len(components)} components")
        
        # 记录到文件
        file_logger.info(f"=== TRAJECTORY ANALYSIS ===")
        file_logger.info(f"Ground truth: {ground_truth}")
        file_logger.info(f"Question: {question}")
        file_logger.info(f"Components count: {len(components)}")
        
        # Extract key information
        search_components = [c for c in components if c.component_type == "search"]
        information_components = [c for c in components if c.component_type == "information"]
        answer_components = [c for c in components if c.component_type == "answer"]
        
        logger.debug(f"Component breakdown: search={len(search_components)}, "
                    f"information={len(information_components)}, answer={len(answer_components)}")
        
        # 记录组件分布到文件
        file_logger.info(f"Component breakdown:")
        file_logger.info(f"  Search components: {len(search_components)}")
        file_logger.info(f"  Information components: {len(information_components)}")
        file_logger.info(f"  Answer components: {len(answer_components)}")
        
        # 分析时序依赖和信息流（异步LLM评估）
        temporal_analysis = await self._analyze_temporal_dependencies(components, ground_truth, question)
        
        # Compute component scores with improved logic
        search_quality_score = 0.0  # 将使用LLM评估结果
        information_relevance_score = 0.0  # 将使用LLM评估结果
        answer_quality_score = self._score_answer_quality_simplified(answer_components, ground_truth)
        
        logger.info(f"Component scores - Search: {search_quality_score:.3f}, "
                   f"Information: {information_relevance_score:.3f}, Answer: {answer_quality_score:.3f}")
        
        # 记录评分结果到文件
        file_logger.info(f"Component scores:")
        file_logger.info(f"  Search quality: {search_quality_score:.3f}")
        file_logger.info(f"  Information relevance: {information_relevance_score:.3f}")
        file_logger.info(f"  Answer quality: {answer_quality_score:.3f}")
        
        # Check for penalties with temporal awareness
        has_insufficient_info = temporal_analysis["has_insufficient_info"]
        has_repeated_searches = self._has_repeated_searches(search_components)
        exceeds_max_steps = len(search_components) > self.config["max_tool_steps"]
        
        logger.info(f"Temporal analysis - Insufficient info: {has_insufficient_info}, "
                   f"Repeated searches: {has_repeated_searches}, Exceeds max steps: {exceeds_max_steps}")
        
        # 记录惩罚信息到文件
        file_logger.info(f"Penalty analysis:")
        file_logger.info(f"  Has insufficient info: {has_insufficient_info}")
        file_logger.info(f"  Has repeated searches: {has_repeated_searches}")
        file_logger.info(f"  Exceeds max steps: {exceeds_max_steps}")
        
        # Create trajectory feedback
        feedback = TrajectoryFeedback(
            trajectory_id="trajectory_1",
            components=components,
            final_answer=answer_components[-1].content if answer_components else "",
            ground_truth=ground_truth,
            think_score=0.0,
            tool_effectiveness_score=search_quality_score,
            information_sufficiency_score=information_relevance_score,
            answer_quality_score=answer_quality_score,
            has_insufficient_info=has_insufficient_info,
            has_repeated_tools=has_repeated_searches,
            exceeds_max_steps=exceeds_max_steps
        )
        
        # 将时序分析结果存储为类属性
        feedback._temporal_analysis = temporal_analysis
        
        # 记录时序分析结果到文件
        file_logger.info(f"Temporal analysis results:")
        for key, value in temporal_analysis.items():
            if key != "llm_evaluation_results":  # 避免记录过长的LLM结果
                file_logger.info(f"  {key}: {value}")
        
        # 记录LLM评估结果摘要
        if "llm_evaluation_results" in temporal_analysis:
            file_logger.info(f"  LLM evaluation results count: {len(temporal_analysis['llm_evaluation_results'])}")
            for i, result in enumerate(temporal_analysis["llm_evaluation_results"]):
                file_logger.info(f"    Component {i+1}: {result.get('information_quality', 'Unknown')} - {result.get('clarity_justification', 'No justification')}")
        
        file_logger.info("=" * 50)
        
        return feedback
    
    def analyze_trajectory_sync(self, components: List[TrajectoryComponent], 
                               ground_truth, question: str = None) -> TrajectoryFeedback:
        """分析整个轨迹并计算feedback（同步版本，使用fallback评估）"""
        logger.info(f"Analyzing trajectory with {len(components)} components (sync version)")
        
        # 记录到文件
        file_logger.info(f"=== TRAJECTORY ANALYSIS (SYNC) ===")
        file_logger.info(f"Ground truth: {ground_truth}")
        file_logger.info(f"Question: {question}")
        file_logger.info(f"Components count: {len(components)}")
        
        # Extract key information
        search_components = [c for c in components if c.component_type == "search"]
        information_components = [c for c in components if c.component_type == "information"]
        answer_components = [c for c in components if c.component_type == "answer"]
        
        logger.debug(f"Component breakdown: search={len(search_components)}, "
                    f"information={len(information_components)}, answer={len(answer_components)}")
        
        # 记录组件分布到文件
        file_logger.info(f"Component breakdown:")
        file_logger.info(f"  Search components: {len(search_components)}")
        file_logger.info(f"  Information components: {len(information_components)}")
        file_logger.info(f"  Answer components: {len(answer_components)}")
        
        # 分析时序依赖和信息流（同步fallback评估）
        temporal_analysis = self._analyze_temporal_dependencies_sync(components, ground_truth, question)
        
        # Compute component scores with improved logic
        search_quality_score = 0.0  # 将使用fallback评估结果
        information_relevance_score = 0.0  # 将使用fallback评估结果
        answer_quality_score = self._score_answer_quality_simplified(answer_components, ground_truth)
        
        logger.info(f"Component scores - Search: {search_quality_score:.3f}, "
                   f"Information: {information_relevance_score:.3f}, Answer: {answer_quality_score:.3f}")
        
        # 记录评分结果到文件
        file_logger.info(f"Component scores:")
        file_logger.info(f"  Search quality: {search_quality_score:.3f}")
        file_logger.info(f"  Information relevance: {information_relevance_score:.3f}")
        file_logger.info(f"  Answer quality: {answer_quality_score:.3f}")
        
        # Check for penalties with temporal awareness
        has_insufficient_info = temporal_analysis["has_insufficient_info"]
        has_repeated_searches = self._has_repeated_searches(search_components)
        exceeds_max_steps = len(search_components) > self.config["max_tool_steps"]
        
        logger.info(f"Temporal analysis - Insufficient info: {has_insufficient_info}, "
                   f"Repeated searches: {has_repeated_searches}, Exceeds max steps: {exceeds_max_steps}")
        
        # 记录惩罚信息到文件
        file_logger.info(f"Penalty analysis:")
        file_logger.info(f"  Has insufficient info: {has_insufficient_info}")
        file_logger.info(f"  Has repeated searches: {has_repeated_searches}")
        file_logger.info(f"  Exceeds max steps: {exceeds_max_steps}")
        
        # Create trajectory feedback
        feedback = TrajectoryFeedback(
            trajectory_id="trajectory_1",
            components=components,
            final_answer=answer_components[-1].content if answer_components else "",
            ground_truth=ground_truth,
            think_score=0.0,
            tool_effectiveness_score=search_quality_score,
            information_sufficiency_score=information_relevance_score,
            answer_quality_score=answer_quality_score,
            has_insufficient_info=has_insufficient_info,
            has_repeated_tools=has_repeated_searches,
            exceeds_max_steps=exceeds_max_steps
        )
        
        # 将时序分析结果存储为类属性
        feedback._temporal_analysis = temporal_analysis
        
        # 记录时序分析结果到文件
        file_logger.info(f"Temporal analysis results:")
        for key, value in temporal_analysis.items():
            if key != "llm_evaluation_results":  # 避免记录过长的LLM结果
                file_logger.info(f"  {key}: {value}")
        
        # 记录fallback评估结果摘要
        if "llm_evaluation_results" in temporal_analysis:
            file_logger.info(f"  Fallback evaluation results count: {len(temporal_analysis['llm_evaluation_results'])}")
            for i, result in enumerate(temporal_analysis["llm_evaluation_results"]):
                file_logger.info(f"    Component {i+1}: {result.get('information_quality', 'Unknown')} - {result.get('clarity_justification', 'No justification')}")
        
        file_logger.info("=" * 50)
        
        return feedback
    
    def create_dense_reward_tensor(self, feedback: TrajectoryFeedback, 
                                  response_length: int) -> torch.Tensor:
        """Create dense reward tensor with intelligent allocation strategy."""
        try:
            logger.info(f"Creating dense reward tensor for response length: {response_length}")
            
            # 记录到文件
            file_logger.info(f"=== CREATING DENSE REWARD TENSOR ===")
            file_logger.info(f"Response length: {response_length}")
            file_logger.info(f"Components count: {len(feedback.components)}")
            file_logger.info(f"Reward allocation strategy: {self.config['reward_allocation_strategy']}")
            
            # 参数验证
            if response_length <= 0:
                logger.warning(f"Invalid response_length: {response_length}, using default 500")
                response_length = 500
            
            # 创建reward tensor
            reward_tensor = torch.zeros(response_length, dtype=torch.float32)
            
            if not feedback.components:
                file_logger.info("No components found, applying uniform score: 0.0")
                file_logger.info("=" * 50)
                return reward_tensor
            
            # 使用改进的reward分配策略
            if self.config["reward_allocation_strategy"] == "component_based":
                reward_tensor = self._allocate_rewards_component_based(feedback, response_length)
                file_logger.info("Used component-based reward allocation")
            else:
                reward_tensor = self._allocate_rewards_uniform(feedback, response_length)
                file_logger.info("Used uniform reward allocation")
            
            # 应用平滑过渡
            if self.config["smooth_reward_transition"]:
                reward_tensor = self._smooth_reward_transitions(reward_tensor)
                file_logger.info("Applied smooth reward transitions")
            
            # 限制reward范围
            reward_tensor = torch.clamp(reward_tensor, 
                                        self.config["min_reward_value"], 
                                        self.config["max_reward_value"])
            
            # Log reward tensor statistics
            self._log_reward_statistics(reward_tensor)
            
            # 记录reward tensor详细信息到文件
            self._log_reward_tensor_details(reward_tensor, feedback)
            
            return reward_tensor
            
        except Exception as e:
            logger.error(f"Error creating dense reward tensor: {e}")
            # 返回默认的reward tensor
            default_reward = torch.zeros(response_length, dtype=torch.float32)
            default_reward[:] = 0.5
            return default_reward
    
    def _apply_core_reward_adjustments(self, base_score: float, feedback: TrajectoryFeedback, 
                                     component: TrajectoryComponent, sorted_components: List[TrajectoryComponent], 
                                     component_idx: int, temporal_meta: Dict) -> float:
        """应用四个核心方面的reward调整（grounding仅对search/think，带衰减与上限）"""
        score = base_score
        
        # 1. 信息不足直接答penalty
        if temporal_meta.get("has_insufficient_info", False):
            if component.component_type == "answer":
                # 检查前面是否有不足的information
                for i in range(component_idx - 1, -1, -1):
                    prev_component = sorted_components[i]
                    if prev_component.component_type == "information":
                        # 应用信息不足惩罚
                        score += self.config["insufficient_info_penalty"]
                        logger.debug(f"Penalizing answer after insufficient info: {score:.3f}")
                        break
                    elif prev_component.component_type == "search":
                        # 如果前面是search，说明这个answer没有直接受到信息不足影响
                        break
        
        # 2. 推理grounding评估（仅对search/think，且加入衰减与总上限，避免reward hacking）
        if component.component_type in ["search", "think"]:
            reasoning_grounded = temporal_meta.get("reasoning_grounded", True)
            if reasoning_grounded:
                applied_steps = temporal_meta.get("grounding_applied", 0)
                total_bonus = temporal_meta.get("grounding_bonus_total", 0.0)
                max_steps = self.config.get("grounding_bonus_max_steps", 2)
                decay = self.config.get("grounding_bonus_decay", 0.85)
                max_total = self.config.get("max_grounding_bonus_total", 1.0)
                
                if applied_steps < max_steps and total_bonus < max_total:
                    step_decay = decay ** applied_steps
                    raw_bonus = self.config.get("grounding_bonus", 0.4) * step_decay
                    # 保证总bonus不超过上限
                    allowed_bonus = min(raw_bonus, max_total - total_bonus)
                    score += allowed_bonus
                    temporal_meta["grounding_applied"] = applied_steps + 1
                    temporal_meta["grounding_bonus_total"] = total_bonus + allowed_bonus
                    logger.debug(f"Rewarding grounded reasoning (step {applied_steps+1}, bonus {allowed_bonus:.3f}): {score:.3f}")
            else:
                score += self.config.get("ungrounded_penalty", -0.3)
                logger.debug(f"Penalizing ungrounded reasoning: {score:.3f}")
        
        # 3. 信息改进奖励（从不足到充足）
        if component.component_type == "information":
            refinement_success = temporal_meta.get("refinement_success", False)
            if refinement_success:
                refinement_steps = temporal_meta.get("refinement_steps", 0)
                # 改进步数越少，奖励越多
                refinement_bonus = self.config["refinement_bonus"] * (1.0 - refinement_steps * 0.1)
                score += refinement_bonus
                logger.debug(f"Rewarding information refinement: {score:.3f}")
        
        # 4. 最终答案质量（已经在base_score中考虑）
        
        # 确保分数在合理范围内
        return max(self.config["min_reward_value"], score)
    
    def _allocate_rewards_component_based(self, feedback: TrajectoryFeedback, response_length: int) -> torch.Tensor:
        """基于四个核心方面分配reward：1. grounding, 2. insufficient info penalty, 3. refinement, 4. final answer"""
        reward_tensor = torch.zeros(response_length, dtype=torch.float32)
        
        # 获取时序分析结果
        temporal_meta = getattr(feedback, '_temporal_analysis', {})
        
        # 初始化grounding奖励追踪
        if "grounding_applied" not in temporal_meta:
            temporal_meta["grounding_applied"] = 0
        if "grounding_bonus_total" not in temporal_meta:
            temporal_meta["grounding_bonus_total"] = 0.0
        
        # 按时间顺序排序组件
        sorted_components = sorted(feedback.components, key=lambda x: x.start_token_idx)
        
        for i, component in enumerate(sorted_components):
            if component.start_token_idx >= response_length or component.end_token_idx > response_length:
                continue
            
            # 基础分数：仅基于最终答案质量
            base_score = feedback.answer_quality_score if component.component_type == "answer" else 0.0
            
            # 应用四个核心方面的reward调整
            final_score = self._apply_core_reward_adjustments(
                base_score, feedback, component, sorted_components, i, temporal_meta
            )
            
            # 分配reward
            start_idx = min(component.start_token_idx, response_length - 1)
            end_idx = min(component.end_token_idx, response_length)
            reward_tensor[start_idx:end_idx] = final_score
            
            # 记录到文件
            file_logger.info(f"Component {i+1} ({component.component_type}): base_score={base_score:.3f}, final_score={final_score:.3f}, tokens={start_idx}-{end_idx}")
        
        # 记录grounding奖励统计
        grounding_applied = temporal_meta.get("grounding_applied", 0)
        grounding_total = temporal_meta.get("grounding_bonus_total", 0.0)
        file_logger.info(f"Grounding rewards applied: {grounding_applied} steps, total bonus: {grounding_total:.3f}")
        
        return reward_tensor
    
    

    def _smooth_reward_transitions(self, reward_tensor: torch.Tensor) -> torch.Tensor:
        """平滑reward过渡，避免突变"""
        if len(reward_tensor) < 3:
            return reward_tensor
        
        # 使用简单的移动平均平滑
        smoothed = reward_tensor.clone()
        kernel_size = 3
        
        for i in range(1, len(reward_tensor) - 1):
            start_idx = max(0, i - kernel_size // 2)
            end_idx = min(len(reward_tensor), i + kernel_size // 2 + 1)
            smoothed[i] = reward_tensor[start_idx:end_idx].mean()
        
        return smoothed
    
    def _log_reward_statistics(self, reward_tensor: torch.Tensor):
        """记录reward tensor的统计信息"""
        non_zero_rewards = reward_tensor[reward_tensor != 0]
        if len(non_zero_rewards) > 0:
            logger.info(f"Reward tensor created - Shape: {reward_tensor.shape}, "
                       f"Non-zero rewards: {len(non_zero_rewards)}, "
                       f"Min: {non_zero_rewards.min().item():.3f}, "
                       f"Max: {non_zero_rewards.max().item():.3f}, "
                       f"Mean: {non_zero_rewards.mean().item():.3f}, "
                       f"Std: {non_zero_rewards.std().item():.3f}")
        else:
            logger.warning("No non-zero rewards in tensor")
    
    def _log_reward_tensor_details(self, reward_tensor: torch.Tensor, feedback: TrajectoryFeedback):
        """记录reward tensor的详细信息到文件"""
        file_logger.info(f"Reward tensor details:")
        file_logger.info(f"  Shape: {reward_tensor.shape}")
        file_logger.info(f"  Min value: {reward_tensor.min().item():.3f}")
        file_logger.info(f"  Max value: {reward_tensor.max().item():.3f}")
        file_logger.info(f"  Mean value: {reward_tensor.mean().item():.3f}")
        
        # 统计非零reward的数量和分布
        non_zero_rewards = reward_tensor[reward_tensor != 0]
        file_logger.info(f"  Non-zero rewards: {len(non_zero_rewards)}")
        if len(non_zero_rewards) > 0:
            file_logger.info(f"  Non-zero min: {non_zero_rewards.min().item():.3f}")
            file_logger.info(f"  Non-zero max: {non_zero_rewards.max().item():.3f}")
            file_logger.info(f"  Non-zero mean: {non_zero_rewards.mean().item():.3f}")
            file_logger.info(f"  Non-zero std: {non_zero_rewards.std().item():.3f}")
        
        # 记录每个组件的reward分配
        if feedback.components:
            file_logger.info(f"  Component reward allocation:")
            for i, component in enumerate(feedback.components):
                start_idx = component.start_token_idx
                end_idx = min(component.end_token_idx, len(reward_tensor))
                if start_idx < len(reward_tensor):
                    component_rewards = reward_tensor[start_idx:end_idx]
                    if len(component_rewards) > 0:
                        file_logger.info(f"    {component.component_type} (step {component.step_number}): "
                                       f"tokens {start_idx}-{end_idx}, "
                                       f"reward range [{component_rewards.min().item():.3f}, {component_rewards.max().item():.3f}], "
                                       f"mean {component_rewards.mean().item():.3f}")
        
        file_logger.info("=" * 50)
    
    
    def _score_answer_quality_simplified(self, answer_components: List[TrajectoryComponent], 
                                       ground_truth) -> float:
        """简化的答案质量评分"""
        if not answer_components:
            return 0.0
        
        final_answer = answer_components[-1].content
        logger.debug(f"Scoring final answer: {final_answer[:100]}...")
        
        try:
            # 尝试使用compute_score
            score = self.compute_score(
                data_source="answer_evaluation",
                solution_str=final_answer,
                ground_truth=ground_truth,
                extra_info=None,
            )
            
            if isinstance(score, dict):
                final_score = score.get("score", 0.0)
            else:
                final_score = score
            
            logger.info(f"Computed answer quality score: {final_score:.3f}")
            return final_score
            
        except Exception as e:
            logger.warning(f"Error computing answer quality score: {e}")
            # 使用简单的fallback评分
            return 0.0
    
    def _check_repeated_searches_improved(self, search_components: List[TrajectoryComponent]) -> List[Tuple[int, int, float]]:
        """改进的重复搜索检测，返回重复查询的详细信息"""
        if len(search_components) < 2:
            return []
        
        search_contents = [comp.content.lower().strip() for comp in search_components]
        repeated_pairs = []
        threshold = 0.7  # 固定阈值
        
        for i in range(len(search_contents)):
            for j in range(i + 1, len(search_contents)):
                similarity = self._compute_query_similarity(search_contents[i], search_contents[j])
                if similarity > threshold:
                    repeated_pairs.append((i, j, similarity))
                    logger.debug(f"Found similar queries: '{search_contents[i]}' and '{search_contents[j]}' (similarity: {similarity:.3f})")
        
        return repeated_pairs
    
    def _has_repeated_searches(self, search_components: List[TrajectoryComponent]) -> bool:
        """检查是否有重复搜索"""
        return len(self._check_repeated_searches_improved(search_components)) > 0
    
    def _compute_query_similarity(self, query1: str, query2: str) -> float:
        """计算两个查询的相似度"""
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _check_insufficient_information(self, information_components: List[TrajectoryComponent]) -> bool:
        """检查信息是否不足"""
        for component in information_components:
            content = component.content.lower()
            if any(indicator in content for indicator in ["error", "failed", "not found", "insufficient", "no results", "empty"]):
                logger.debug(f"Found insufficient info indicator: {component.content[:50]}...")
                return True
        return False
    
    async def _evaluate_reasoning_grounding(self, reasoning_text: str, search_evidence: List[Dict], 
                                          question: str) -> Dict[str, Any]:
        """使用LLM评估推理是否grounded"""
        if not self.enable_llm_evaluation or not self.llm_evaluator:
            # Fallback: 简单的启发式评估
            return {
                "premise_grounding": "Unspecified",
                "evaluation_success": False,
                "fallback_reason": "LLM evaluation disabled"
            }
        
        try:
            result = await self.llm_evaluator.evaluate_reasoning_grounding(
                reasoning_text, search_evidence, question
            )
            return result
        except Exception as e:
            logger.error(f"Error in LLM grounding evaluation: {e}")
            return {
                "premise_grounding": "Unspecified",
                "evaluation_success": False,
                "fallback_reason": f"LLM evaluation error: {str(e)}"
            }
    
    async def _evaluate_search_result_quality(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """使用LLM评估搜索结果质量"""
        if not self.enable_llm_evaluation or not self.llm_evaluator:
            # Fallback: 简单的启发式评估
            return {
                "information_quality": "Unspecified",
                "evaluation_success": False,
                "fallback_reason": "LLM evaluation disabled"
            }
        
        try:
            result = await self.llm_evaluator.evaluate_information_quality(query, documents)
            return result
        except Exception as e:
            logger.error(f"Error in LLM search result evaluation: {e}")
            return {
                "information_quality": "Unspecified",
                "evaluation_success": False,
                "fallback_reason": f"LLM evaluation error: {str(e)}"
            }
    
    async def _evaluate_information_quality_batch(self, information_components: List[TrajectoryComponent], 
                                                search_components: List[TrajectoryComponent], 
                                                ground_truth: str) -> List[Dict[str, Any]]:
        """批量评估信息质量，使用llm_judge.py的prompt"""
        if not self.enable_llm_evaluation or not self.llm_evaluator:
            # Fallback: 使用简化的启发式评估
            return self._fallback_information_quality_evaluation(information_components, ground_truth)
        
        try:
            # 准备批量评估请求
            evaluation_requests = []
            for i, info_comp in enumerate(information_components):
                # 找到对应的search query
                search_query = self._find_corresponding_search_query(info_comp, search_components)
                
                # 将information内容作为"documents"处理
                documents = [{"content": info_comp.content}]
                
                evaluation_requests.append({
                    "type": "information_quality",
                    "query": search_query,
                    "documents": documents,
                    "component_index": i
                })
            
            # 批量调用LLM评估器
            results = await self.llm_evaluator.batch_evaluate(evaluation_requests)
            
            # 处理结果
            evaluation_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"LLM evaluation failed for component {i}: {result}")
                    # 使用fallback评估
                    fallback_result = self._fallback_single_information_quality(
                        information_components[i], ground_truth
                    )
                    evaluation_results.append(fallback_result)
                else:
                    evaluation_results.append(result)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in batch information quality evaluation: {e}")
            # 使用fallback评估
            return self._fallback_information_quality_evaluation(information_components, ground_truth)
    
    def _find_corresponding_search_query(self, info_component: TrajectoryComponent, 
                                       search_components: List[TrajectoryComponent]) -> str:
        """找到对应的search query"""
        # 简单的启发式：找到最近的search组件
        for search_comp in reversed(search_components):
            if search_comp.start_token_idx < info_component.start_token_idx:
                return search_comp.content
        return "unknown query"
    
    def _fallback_information_quality_evaluation(self, information_components: List[TrajectoryComponent], 
                                               ground_truth: str) -> List[Dict[str, Any]]:
        """Fallback信息质量评估（启发式）"""
        results = []
        for component in information_components:
            result = self._fallback_single_information_quality(component, ground_truth)
            results.append(result)
        return results
    
    def _fallback_single_information_quality(self, component: TrajectoryComponent, 
                                           ground_truth: str) -> Dict[str, Any]:
        """单个组件的fallback信息质量评估"""
        content = component.content
        
        # 检查错误指示符
        if any(indicator in content.lower() for indicator in ["error", "failed", "not found", "insufficient", "no results"]):
            return {
                "information_quality": "Insufficient",
                "information_clarity": "Clear",
                "clarity_justification": "Contains error indicators",
                "evaluation_success": True,
                "fallback": True
            }
        
        # 基于长度的简单评估
        word_count = len(content.split())
        if word_count > 20:
            quality = "Sufficient"
        elif word_count > 10:
            quality = "Sufficient"
        else:
            quality = "Insufficient"
        
        return {
            "information_quality": quality,
            "information_clarity": "Clear",
            "clarity_justification": f"Length-based fallback evaluation ({word_count} words)",
            "evaluation_success": True,
            "fallback": True
        }
    
    def _evaluate_information_quality(self, information_content: str, ground_truth: str) -> float:
        """评估信息质量，返回0-1的分数（保持向后兼容）"""
        if not information_content or not ground_truth:
            return 0.0
        
        # 简化的质量评估：只关注长度和基本内容
        word_count = len(information_content.split())
        if word_count > 20:
            return 0.8
        elif word_count > 10:
            return 0.5
        else:
            return 0.2
    
    def _is_information_sufficient(self, information_content: str, ground_truth: str) -> bool:
        """判断信息是否足够（保持向后兼容）"""
        quality_score = self._evaluate_information_quality(information_content, ground_truth)
        threshold = 0.5  # 固定阈值
        return quality_score >= threshold
    
    async def _is_information_sufficient_llm(self, evaluation_result: Dict[str, Any]) -> bool:
        """基于LLM评估结果判断信息是否足够"""
        if not evaluation_result.get("evaluation_success", False):
            return False
        
        quality = evaluation_result.get("information_quality", "Unspecified")
        return quality == "Sufficient"
    
    async def _analyze_temporal_dependencies(self, components: List[TrajectoryComponent], 
                                           ground_truth, question: str = None) -> Dict:
        """分析时序依赖，专注于四个核心方面，使用批量LLM评估"""
        logger.info("Analyzing temporal dependencies with simplified focus and LLM evaluation")
        
        # 按时间顺序排序组件
        sorted_components = sorted(components, key=lambda x: x.start_token_idx)
        
        # 提取不同类型的组件
        search_components = [c for c in sorted_components if c.component_type == "search"]
        information_components = [c for c in sorted_components if c.component_type == "information"]
        
        # 1. 信息质量分析（使用LLM批量评估）
        info_quality_analysis = await self._analyze_information_quality_flow_llm(
            sorted_components, ground_truth, search_components, information_components
        )
        
        # 2. 推理grounding分析
        grounding_analysis = await self._analyze_reasoning_grounding(sorted_components, question, search_components)
        
        # 3. 改进分析（从不足到充足）
        refinement_analysis = self._analyze_information_refinement_llm(
            sorted_components, ground_truth, info_quality_analysis["llm_evaluation_results"]
        )
        
        temporal_analysis = {
            "has_insufficient_info": info_quality_analysis["has_insufficient_info"],
            "info_flow_quality": info_quality_analysis["flow_quality"],
            "reasoning_grounded": grounding_analysis["reasoning_grounded"],
            "refinement_success": refinement_analysis["refinement_success"],
            "refinement_steps": refinement_analysis["refinement_steps"],
            "temporal_sequence": [comp.component_type for comp in sorted_components],
            "llm_evaluation_results": info_quality_analysis["llm_evaluation_results"]
        }
        
        logger.info(f"Enhanced temporal analysis results: {temporal_analysis}")
        return temporal_analysis
    
    def _analyze_temporal_dependencies_sync(self, components: List[TrajectoryComponent], 
                                           ground_truth, question: str = None) -> Dict:
        """分析时序依赖，专注于四个核心方面（同步版本，使用fallback评估）"""
        logger.info("Analyzing temporal dependencies with simplified focus and fallback evaluation (sync)")
        
        # 按时间顺序排序组件
        sorted_components = sorted(components, key=lambda x: x.start_token_idx)
        
        # 提取不同类型的组件
        search_components = [c for c in sorted_components if c.component_type == "search"]
        information_components = [c for c in sorted_components if c.component_type == "information"]
        
        # 1. 信息质量分析（使用fallback评估）
        info_quality_analysis = self._analyze_information_quality_flow_fallback(
            sorted_components, ground_truth, search_components, information_components
        )
        
        # 2. 推理grounding分析（使用fallback评估）
        grounding_analysis = self._analyze_reasoning_grounding_fallback(sorted_components, question, search_components)
        
        # 3. 改进分析（从不足到充足）
        refinement_analysis = self._analyze_information_refinement(
            sorted_components, ground_truth
        )
        
        temporal_analysis = {
            "has_insufficient_info": info_quality_analysis["has_insufficient_info"],
            "info_flow_quality": info_quality_analysis["flow_quality"],
            "reasoning_grounded": grounding_analysis["reasoning_grounded"],
            "refinement_success": refinement_analysis["refinement_success"],
            "refinement_steps": refinement_analysis["refinement_steps"],
            "temporal_sequence": [comp.component_type for comp in sorted_components],
            "llm_evaluation_results": info_quality_analysis["fallback_evaluation_results"]
        }
        
        logger.info(f"Fallback temporal analysis results: {temporal_analysis}")
        return temporal_analysis
    
    async def _analyze_information_quality_flow_llm(self, sorted_components: List[TrajectoryComponent], 
                                                  ground_truth, search_components: List[TrajectoryComponent], 
                                                  information_components: List[TrajectoryComponent]) -> Dict:
        """使用LLM批量评估信息质量流"""
        flow_quality = 1.0
        has_insufficient_info = False
        
        if information_components:
            # 批量LLM评估
            llm_evaluation_results = await self._evaluate_information_quality_batch(
                information_components, search_components, ground_truth
            )
            
            # 分析结果
            for i, (component, eval_result) in enumerate(zip(information_components, llm_evaluation_results)):
                is_sufficient = await self._is_information_sufficient_llm(eval_result)
                
                if not is_sufficient:
                    has_insufficient_info = True
                    flow_quality *= 0.7
                    logger.debug(f"LLM evaluation: Low quality information at step {i+1}: {eval_result.get('information_quality', 'Unknown')}")
                    
                    # 记录到文件
                    file_logger.info(f"LLM evaluation result for {component.component_type} component {i+1}:")
                    file_logger.info(f"  Content: {component.content[:100]}...")
                    file_logger.info(f"  Quality: {eval_result.get('information_quality', 'Unknown')}")
                    file_logger.info(f"  Clarity: {eval_result.get('information_clarity', 'Unknown')}")
                    file_logger.info(f"  Justification: {eval_result.get('clarity_justification', 'Unknown')}")
        else:
            llm_evaluation_results = []
        
        return {
            "has_insufficient_info": has_insufficient_info,
            "flow_quality": flow_quality,
            "llm_evaluation_results": llm_evaluation_results
        }
    
    def _analyze_information_quality_flow_fallback(self, sorted_components: List[TrajectoryComponent], 
                                                  ground_truth, search_components: List[TrajectoryComponent], 
                                                  information_components: List[TrajectoryComponent]) -> Dict:
        """使用fallback评估信息质量流（同步版本）"""
        flow_quality = 1.0
        has_insufficient_info = False
        
        if information_components:
            # 使用fallback评估
            fallback_evaluation_results = []
            for i, component in enumerate(information_components):
                eval_result = self._fallback_single_information_quality(component, ground_truth)
                fallback_evaluation_results.append(eval_result)
                
                is_sufficient = eval_result.get("information_quality") == "Sufficient"
                if not is_sufficient:
                    has_insufficient_info = True
                    flow_quality *= 0.7
                    logger.debug(f"Fallback evaluation: Low quality information at step {i+1}: {eval_result.get('information_quality', 'Unknown')}")
                    
                    # 记录到文件
                    file_logger.info(f"Fallback evaluation result for {component.component_type} component {i+1}:")
                    file_logger.info(f"  Content: {component.content[:100]}...")
                    file_logger.info(f"  Quality: {eval_result.get('information_quality', 'Unknown')}")
                    file_logger.info(f"  Clarity: {eval_result.get('information_clarity', 'Unknown')}")
                    file_logger.info(f"  Justification: {eval_result.get('clarity_justification', 'Unknown')}")
        else:
            fallback_evaluation_results = []
        
        return {
            "has_insufficient_info": has_insufficient_info,
            "flow_quality": flow_quality,
            "fallback_evaluation_results": fallback_evaluation_results
        }
    
    async def _analyze_reasoning_grounding(self, sorted_components: List[TrajectoryComponent], 
                                         question: str, search_components: List[TrajectoryComponent]) -> Dict:
        """分析推理grounding（使用LLM评估器）"""
        if not self.enable_llm_evaluation or not self.llm_evaluator:
            # Fallback: 简单的启发式评估
            return {
                "reasoning_grounded": True,
                "grounding_details": "LLM evaluation disabled, using fallback"
            }
        
        try:
            # 收集所有search证据
            search_evidence = []
            for search_comp in search_components:
                # 将search内容作为证据
                search_evidence.append({
                    "content": search_comp.content,
                    "type": "search_query",
                    "step": search_comp.step_number
                })
            
            # 评估每个search组件的grounding
            grounding_results = []
            for search_comp in search_components:
                if question:  # 如果有question，使用LLM评估
                    grounding_result = await self.llm_evaluator.evaluate_reasoning_grounding(
                        search_comp.content, search_evidence, question
                    )
                    grounding_results.append(grounding_result)
                else:
                    # 没有question时，假设是grounded
                    grounding_results.append({
                        "premise_grounding": "Directly Grounded",
                        "evaluation_success": True,
                        "fallback": True
                    })
            
            # 综合评估结果
            grounded_count = sum(1 for result in grounding_results 
                               if result.get("premise_grounding") == "Directly Grounded")
            total_count = len(grounding_results)
            
            reasoning_grounded = grounded_count > 0 and grounded_count / total_count >= 0.5
            
            return {
                "reasoning_grounded": reasoning_grounded,
                "grounding_details": f"LLM evaluation: {grounded_count}/{total_count} grounded",
                "grounding_results": grounding_results
            }
            
        except Exception as e:
            logger.error(f"Error in LLM grounding evaluation: {e}")
            return {
                "reasoning_grounded": True,
                "grounding_details": f"LLM evaluation error: {str(e)}"
            }
    
    def _analyze_reasoning_grounding_fallback(self, sorted_components: List[TrajectoryComponent], 
                                             question: str, search_components: List[TrajectoryComponent]) -> Dict:
        """分析推理grounding（同步fallback版本）"""
        # Fallback: 简单的启发式评估
        if not search_components:
            return {
                "reasoning_grounded": True,
                "grounding_details": "No search components, using fallback"
            }
        
        # 简单的启发式：检查search内容是否包含基本的关键词
        grounded_count = 0
        total_count = len(search_components)
        
        for search_comp in search_components:
            content = search_comp.content.lower()
            # 检查是否包含基本的搜索关键词
            if any(keyword in content for keyword in ["what", "how", "when", "where", "why", "who", "which"]):
                grounded_count += 1
        
        reasoning_grounded = grounded_count > 0 and grounded_count / total_count >= 0.5
        
        return {
            "reasoning_grounded": reasoning_grounded,
            "grounding_details": f"Fallback evaluation: {grounded_count}/{total_count} grounded",
            "grounding_results": []
        }
    
    def _analyze_information_refinement_llm(self, sorted_components: List[TrajectoryComponent], 
                                          ground_truth, llm_evaluation_results: List[Dict[str, Any]]) -> Dict:
        """基于LLM评估结果分析信息改进"""
        refinement_success = False
        refinement_steps = 0
        
        if not llm_evaluation_results:
            # 如果没有LLM评估结果，使用fallback
            return self._analyze_information_refinement(sorted_components, ground_truth)
        
        # 找到第一个信息不足的information
        first_insufficient_idx = None
        for i, eval_result in enumerate(llm_evaluation_results):
            if eval_result.get("information_quality") == "Insufficient":
                first_insufficient_idx = i
                break
        
        if first_insufficient_idx is not None:
            # 检查后续是否有改进
            for i in range(first_insufficient_idx + 1, len(llm_evaluation_results)):
                eval_result = llm_evaluation_results[i]
                if eval_result.get("information_quality") == "Sufficient":
                    refinement_success = True
                    refinement_steps = i - first_insufficient_idx
                    logger.debug(f"LLM evaluation: Information refined after {refinement_steps} steps")
                    break
        
        return {
            "refinement_success": refinement_success,
            "refinement_steps": refinement_steps
        }
    
    def _analyze_information_refinement(self, sorted_components: List[TrajectoryComponent], 
                                      ground_truth) -> Dict:
        """分析信息改进（从不足到充足）- fallback版本"""
        refinement_success = False
        refinement_steps = 0
        
        # 找到第一个信息不足的information
        first_insufficient_idx = None
        for i, component in enumerate(sorted_components):
            if component.component_type == "information":
                if not self._is_information_sufficient(component.content, ground_truth):
                    first_insufficient_idx = i
                    break
        
        if first_insufficient_idx is not None:
            # 检查后续是否有改进
            for i in range(first_insufficient_idx + 1, len(sorted_components)):
                component = sorted_components[i]
                if component.component_type == "information":
                    if self._is_information_sufficient(component.content, ground_truth):
                        refinement_success = True
                        refinement_steps = i - first_insufficient_idx
                        logger.debug(f"Information refined after {refinement_steps} steps")
                        break
        
        return {
            "refinement_success": refinement_success,
            "refinement_steps": refinement_steps
        }
    
    def __call__(self, data: DataProto, return_dict=False):
        """处理batch数据并返回dense reward tensors（同步版本）"""
        logger.info(f"Processing batch with {len(data)} items")
        
        # 记录到文件
        file_logger.info(f"=== BATCH PROCESSING START ===")
        file_logger.info(f"Batch size: {len(data)}")
        file_logger.info(f"Return dict: {return_dict}")
        
        # 安全检查
        if not data or len(data) == 0:
            logger.warning("Empty data batch received")
            file_logger.warning("Empty data batch received")
            if return_dict:
                return {"reward_tensor": torch.tensor([])}
            else:
                return torch.tensor([])
        
        dense_reward_tensors = []
        try:
            for i in range(len(data)):
                try:
                    data_item = data[i]
                    file_logger.info(f"--- Processing item {i+1}/{len(data)} ---")
                    
                    # 安全检查数据项
                    if not hasattr(data_item, 'batch') or not hasattr(data_item, 'non_tensor_batch'):
                        logger.warning(f"Data item {i} missing required attributes, skipping")
                        file_logger.warning(f"Data item {i} missing required attributes, skipping")
                        continue
                    
                    # 检查必要的字段
                    required_fields = ["prompts", "responses", "attention_mask"]
                    for field in required_fields:
                        if field not in data_item.batch:
                            logger.warning(f"Data item {i} missing required field: {field}, skipping")
                            file_logger.warning(f"Data item {i} missing required field: {field}, skipping")
                            continue
                    
                    if "reward_model" not in data_item.non_tensor_batch:
                        logger.warning(f"Data item {i} missing reward_model field, skipping")
                        file_logger.warning(f"Data item {i} missing reward_model field, skipping")
                        continue
                    
                    # 处理单个数据项（同步调用）
                    result = self._process_single_item_sync(data_item, i)
                    if result:
                        dense_reward_tensors.append(result["reward_tensor"])
                        file_logger.info(f"Item {i+1} processed successfully")
                    else:
                        file_logger.warning(f"Item {i+1} processing returned None")
                    
                except Exception as e:
                    logger.error(f"Error processing data item {i}: {e}")
                    file_logger.error(f"Error processing data item {i}: {e}")
                    # 创建默认的reward tensor作为fallback
                    default_reward = torch.zeros(500, dtype=torch.float32)
                    dense_reward_tensors.append(default_reward)
                    continue
            
            # 检查是否有有效的reward tensors
            if not dense_reward_tensors:
                logger.warning("No valid reward tensors generated")
                if return_dict:
                    return {"reward_tensor": torch.tensor([])}
                else:
                    return torch.tensor([])
            
            # 确保所有reward tensor长度一致
            target_length = 500
            normalized_rewards = []
            for reward_tensor in dense_reward_tensors:
                if len(reward_tensor) != target_length:
                    # 调整长度
                    if len(reward_tensor) > target_length:
                        normalized_reward = reward_tensor[:target_length]
                    else:
                        normalized_reward = torch.zeros(target_length, dtype=torch.float32)
                        normalized_reward[:len(reward_tensor)] = reward_tensor
                    normalized_rewards.append(normalized_reward)
                else:
                    normalized_rewards.append(reward_tensor)
            
            # 直接stack，避免padding问题
            stacked_rewards = torch.stack(normalized_rewards, dim=0)
            logger.info(f"Final stacked rewards shape: {stacked_rewards.shape}")
            
            # 返回结果
            if return_dict:
                result = {"reward_tensor": stacked_rewards}
                logger.info(f"Returning dict with keys: {list(result.keys())}")
                return result
            else:
                logger.info(f"Returning tensor directly: {stacked_rewards.shape}")
                return stacked_rewards
                
        except Exception as e:
            logger.error(f"Critical error in reward manager: {e}")
            # 返回默认值
            default_reward = torch.zeros((len(data), 500), dtype=torch.float32)
            if return_dict:
                return {"reward_tensor": default_reward}
            else:
                return default_reward
    
    def _process_single_item_sync(self, data_item, item_index):
        """处理单个数据项，返回feedback和reward tensor（同步版本）"""
        try:
            # 记录到文件
            file_logger.info(f"=== PROCESSING SINGLE ITEM {item_index+1} ===")
            
            # 获取response数据
            response_ids = data_item.batch["responses"]
            response_ids_shape = response_ids.shape
            logger.debug(f"Response IDs shape: {response_ids_shape}")
            file_logger.info(f"Response IDs shape: {response_ids_shape}")
            
            # 计算prompt长度
            prompts_shape = data_item.batch["prompts"].shape
            logger.debug(f"Prompts shape: {prompts_shape}")
            file_logger.info(f"Prompts shape: {prompts_shape}")
            
            # 安全地获取prompt长度
            if len(prompts_shape) >= 2:
                prompt_length = prompts_shape[1]
            elif len(prompts_shape) == 1:
                prompt_length = prompts_shape[0]
            else:
                prompt_length = 0
                logger.warning(f"Unexpected prompts shape: {prompts_shape}, using default prompt_length=0")
                file_logger.warning(f"Unexpected prompts shape: {prompts_shape}, using default prompt_length=0")
            
            logger.debug(f"Prompt length: {prompt_length}")
            file_logger.info(f"Prompt length: {prompt_length}")
            
            # 获取attention mask
            attention_mask_shape = data_item.batch["attention_mask"].shape
            logger.debug(f"Attention mask shape: {attention_mask_shape}")
            file_logger.info(f"Attention mask shape: {attention_mask_shape}")
            
            # 安全地移除batch维度
            if len(attention_mask_shape) >= 2:
                attention_mask = data_item.batch["attention_mask"][0]
            elif len(attention_mask_shape) == 1:
                attention_mask = data_item.batch["attention_mask"]
            else:
                logger.warning(f"Unexpected attention_mask shape: {attention_mask_shape}, using default")
                file_logger.warning(f"Unexpected attention_mask shape: {attention_mask_shape}, using default")
                attention_mask = torch.ones(1000, dtype=torch.long)
            
            # 确保attention_mask是1D
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            
            if attention_mask.dim() != 1:
                raise ValueError(f"Invalid attention_mask dimensions: {attention_mask.dim()}")
            
            logger.debug(f"Final attention_mask shape: {attention_mask.shape}")
            file_logger.info(f"Final attention_mask shape: {attention_mask.shape}")
            
            # 计算有效的response长度
            if prompt_length > 0 and prompt_length < len(attention_mask):
                valid_response_length = attention_mask[prompt_length:].sum()
            else:
                valid_response_length = 100
            
            # 确保长度是有效的整数
            if hasattr(valid_response_length, 'item'):
                valid_response_length = valid_response_length.item()
            valid_response_length = int(valid_response_length)
                
            logger.info(f"Final valid response length: {valid_response_length}")
            file_logger.info(f"Final valid response length: {valid_response_length}")
            
            # 获取有效的response IDs
            if len(response_ids_shape) >= 2:
                if valid_response_length > 0 and valid_response_length < response_ids.shape[1]:
                    valid_response_ids = response_ids[0, :valid_response_length]
                else:
                    valid_response_ids = response_ids[0, :100]
            else:
                if valid_response_length > 0 and valid_response_length < len(response_ids):
                    valid_response_ids = response_ids[:valid_response_length]
                else:
                    valid_response_ids = response_ids[:100]
            
            # 确保valid_response_ids是1D tensor
            if valid_response_ids.dim() > 1:
                valid_response_ids = valid_response_ids.flatten()
            
            # 解码response
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            
            # 尝试获取question（如果存在）
            question = None
            if "question" in data_item.non_tensor_batch:
                question = data_item.non_tensor_batch["question"]
            elif "prompt" in data_item.non_tensor_batch:
                # 从prompt字段提取问题内容
                prompt_data = data_item.non_tensor_batch["prompt"]
                if isinstance(prompt_data, (list, np.ndarray)) and len(prompt_data) > 0:
                    # prompt是消息列表，提取第一个用户消息的内容
                    first_message = prompt_data[0] if isinstance(prompt_data, list) else prompt_data.tolist()[0]
                    if isinstance(first_message, dict) and "content" in first_message:
                        question = first_message["content"]
                    elif isinstance(first_message, (list, np.ndarray)) and len(first_message) > 0:
                        # 处理嵌套结构
                        first_content = first_message[0] if isinstance(first_message, list) else first_message.tolist()[0]
                        if isinstance(first_content, dict) and "content" in first_content:
                            question = first_content["content"]
            elif "raw_prompt" in data_item.non_tensor_batch:
                # 从raw_prompt字段提取问题内容
                raw_prompt = data_item.non_tensor_batch["raw_prompt"]
                if isinstance(raw_prompt, (list, np.ndarray)) and len(raw_prompt) > 0:
                    # raw_prompt是消息列表，提取第一个用户消息的内容
                    first_message = raw_prompt[0] if isinstance(raw_prompt, list) else raw_prompt.tolist()[0]
                    if isinstance(first_message, dict) and "content" in first_message:
                        question = first_message["content"]
                    elif isinstance(first_message, (list, np.ndarray)) and len(first_message) > 0:
                        # 处理嵌套结构
                        first_content = first_message[0] if isinstance(first_message, list) else first_message.tolist()[0]
                        if isinstance(first_content, dict) and "content" in first_content:
                            question = first_content["content"]

            
            logger.info(f"Data source: {data_source}")
            logger.info(f"Ground truth: {ground_truth}")
            logger.info(f"Question: {question}")
            logger.info(f"Response: {response_str[:self.config.get('max_log_length', 200)]}...")
            
            # 记录到文件
            file_logger.info(f"Data source: {data_source}")
            file_logger.info(f"Ground truth: {ground_truth}")
            file_logger.info(f"Question: {question}")
            file_logger.info(f"Response: {response_str}")
            
            # 解析轨迹组件（使用改进的token位置计算）
            components = self.parse_trajectory_components(response_str, valid_response_ids.tolist())
            
            # 分析整个轨迹（同步调用）
            feedback = self.analyze_trajectory_sync(components, ground_truth, question)
            
            # 创建dense reward tensor
            target_length = 500
            dense_reward = self.create_dense_reward_tensor(feedback, target_length)
            
            # 打印分析摘要
            if item_index < self.num_examine:
                self._print_analysis_summary(data_source, item_index, response_str, ground_truth, components, feedback, dense_reward)
            
            file_logger.info(f"Item {item_index+1} processing completed successfully")
            
            return {
                "feedback": feedback,
                "reward_tensor": dense_reward
            }
            
        except Exception as e:
            logger.error(f"Error processing single item {item_index}: {e}")
            file_logger.error(f"Error processing single item {item_index}: {e}")
            return None
    
    def _print_analysis_summary(self, data_source, item_index, response_str, ground_truth, components, feedback, dense_reward):
        """打印分析摘要"""
        print(f"\n{'='*80}")
        print(f"[{data_source}] Trajectory Analysis for Item {item_index+1}:")
        print(f"{'='*80}")
        print(f"Response: {response_str[:300]}...")
        print(f"Ground Truth: {ground_truth}")
        print(f"Components Found: {len(components)}")
        for j, comp in enumerate(components):
            print(f"  {j+1}. {comp.component_type}: {comp.content[:100]}...")
        print(f"\nScores:")
        print(f"  Search Quality: {feedback.tool_effectiveness_score:.3f}")
        print(f"  Information Relevance: {feedback.information_sufficiency_score:.3f}")
        print(f"  Answer Quality: {feedback.answer_quality_score:.3f}")
        print(f"\nTemporal Analysis:")
        temporal_meta = getattr(feedback, '_temporal_analysis', {})
        print(f"  Info Flow Quality: {temporal_meta.get('info_flow_quality', 0.0):.3f}")
        print(f"  Recovery Steps: {temporal_meta.get('recovery_steps', 0)}")
        print(f"  Recovery Success: {temporal_meta.get('recovery_success', False)}")
        print(f"  Repetition Penalty: {temporal_meta.get('repetition_penalty', 0.0):.3f}")
        print(f"  Temporal Sequence: {' -> '.join(temporal_meta.get('temporal_sequence', []))}")
        print(f"\nPenalties:")
        print(f"  Insufficient Info: {feedback.has_insufficient_info}")
        print(f"  Repeated Searches: {feedback.has_repeated_tools}")
        print(f"  Exceeds Max Steps: {feedback.exceeds_max_steps}")
        print(f"\nReward Tensor Shape: {dense_reward.shape}")
        print(f"Non-zero rewards: {(dense_reward != 0).sum().item()}")
        print(f"Reward range: [{dense_reward.min().item():.3f}, {dense_reward.max().item():.3f}]")
        print(f"{'='*80}") 

    def _allocate_rewards_uniform(self, feedback: TrajectoryFeedback, response_length: int) -> torch.Tensor:
        """均匀分配reward到整个response（备用策略）"""
        reward_tensor = torch.zeros(response_length, dtype=torch.float32)
        
        # 仅使用最终答案质量作为稳定均匀信号，避免累积导致的reward hacking
        avg_score = float(feedback.answer_quality_score) if feedback.answer_quality_score is not None else 0.0
        reward_tensor[:] = avg_score
        logger.info(f"Applied uniform score {avg_score:.3f} to entire response")
        return reward_tensor 