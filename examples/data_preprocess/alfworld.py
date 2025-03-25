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
"""
Preprocess the Geometry3k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='tw', choices=['thor', 'tw'])
    parser.add_argument('--local_dir', default='~/data/alfworld')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--add_few_shot', default=False, type=bool)

    args = parser.parse_args()
    print(f"processing data for task: {args.task}; shot({args.add_few_shot})")
    args.local_dir = os.path.join(args.local_dir, args.task)
    args.local_dir = os.path.join(args.local_dir, 'few_shot' if args.add_few_shot else 'zero_shot')

    data_source = 'hiyouga/geometry3k'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train'].select(range(256))
    test_dataset = dataset['test'].select(range(256))

    instruction_following = {
        "thor": "<image>",
        "tw": "",
        }
    few_shot_instruction = {
        "thor": "",
        "tw": "",
    }

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('problem')
            prompt = instruction_following[args.task]
            if args.add_few_shot:
                few_shot_prompt = few_shot_instruction[args.task]
                prompt = prompt + '\n' + few_shot_prompt
            # answer = example.pop('answer')
            images = example.pop('images')

            if args.task == 'thor':
                data = {
                    "data_source": args.task,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "images": images,
                    "ability": "game",
                    # "reward_model": {
                    #     "style": "rule",
                    #     "ground_truth": answer
                    # },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        # 'answer': answer,
                        # "question": problem,
                    }
                }
            else:
                data = {
                    "data_source": args.task,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "ability": "game",
                    # "reward_model": {
                    #     "style": "rule",
                    #     "ground_truth": answer
                    # },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        # 'answer': answer,
                        # "question": problem,
                    }
                }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
