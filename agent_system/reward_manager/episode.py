# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from verl import DataProto
import torch
import numpy as np
from verl.utils.reward_score import search_r1_like_qa_em

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return search_r1_like_qa_em.compute_score
    else:
       return search_r1_like_qa_em.compute_score



class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.,normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        self.format_score=format_score

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            # else:
            #     return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)


            ground_truth = data_item.non_tensor_batch.get('reward_model', {}).get('ground_truth', '')
            # select rm_score

            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            compute_score_fn=_select_rm_score_fn(data_source)

            # Convert ground_truth to the expected format if it's not already a dictionary
            if isinstance(ground_truth, str):
                ground_truth = {"target": ground_truth}
            elif isinstance(ground_truth, list) or isinstance(ground_truth, np.ndarray):
                ground_truth = {"target": ground_truth}

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            # extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            # if multi_modal_inputs is not None:
            #     pixel_values = multi_modal_inputs['pixel_values']
            #     image_grid_thw = multi_modal_inputs['image_grid_thw']


            # episode_rewards = data_item.non_tensor_batch['episode_rewards']
            # episode_lengths = data_item.non_tensor_batch['episode_lengths']

            # if self.normalize_by_length:
            #     score = episode_rewards / episode_lengths
            # else:
            #     score = episode_rewards
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
