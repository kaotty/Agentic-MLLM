# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Preprocess Geometry3k for diverse-tool RL.

This is the RL counterpart of ``sft/sft_dataset/diverse``: same Qwen2.5-VL-7B
problem set as ``geo3k_zoom.py``, but the tool menu surfaced to the model is
the **full DeepEyesV2 set** of four tools

    image_zoom_in_tool, code_interpreter, search, image_search

so the system prompt and tool schemas line up with what the SFT-diverse
checkpoint was trained on.

For Geo3K the ``search`` and ``image_search`` calls run in soft-fail mode at
runtime (search has no retrieval backend; image_search has no cache nor a
SerpAPI key). The point of exposing them here is consistency with the SFT
phase, not actual retrieval signal -- see ``sft/README.md`` for the full
ablation design.

Usage::

    python examples/data_preprocess/geo3k_diverse.py \
        --local_save_dir ~/data/geo3k_multiturn_diverse_tool
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


DIVERSE_SYSTEM_PROMPT = (
    "You are a helpful multimodal expert. You are given a geometry question and an "
    "accompanying image. Reason step by step. You may invoke any of the following "
    "tools to gather more evidence before answering:\n"
    "  * `image_zoom_in_tool`  -- crop and inspect a region of the image\n"
    "  * `code_interpreter`     -- run Python (computation, plotting, image processing)\n"
    "  * `search`               -- issue web search queries\n"
    "  * `image_search`         -- reverse-image search the most recent image\n\n"
    "When you call a tool you MUST output exactly:\n"
    "<tool_call>\n"
    '{"name": "<tool_name>", "arguments": {...}}\n'
    "</tool_call>\n\n"
    "Use tools sparingly and only when they help answer the question. After "
    "gathering enough evidence, wrap your reasoning in <think> </think> tags and "
    "put the final answer in \\boxed{}."
)

INSTRUCTION_FOLLOWING = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{}."
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated, use --local_save_dir.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Optional path to a local snapshot of hiyouga/geometry3k.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/geo3k_multiturn_diverse_tool",
        help="Output directory for {train,test}.parquet.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    data_source = "hiyouga/geometry3k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + INSTRUCTION_FOLLOWING
            answer = example.pop("answer")
            images = example.pop("images")

            tool_image = images[0] if len(images) > 0 else None

            tools_kwargs = {}
            if tool_image is not None:
                tools_kwargs["image_zoom_in_tool"] = {
                    "create_kwargs": {"image": tool_image},
                }
                tools_kwargs["code_interpreter"] = {
                    "create_kwargs": {"image": tool_image},
                }
                # ``search`` is parameter-driven (query_list), no per-sample state
                # needed -- but we register an empty kwargs dict so the agent loop
                # treats the tool as available for this trajectory.
                tools_kwargs["search"] = {"create_kwargs": {}}
                # image_search is no-arg at call time but needs the latest image
                # at create time so that, in the future, hooking up a real backend
                # (SerpAPI / Yandex) is a one-line config change.
                tools_kwargs["image_search"] = {
                    "create_kwargs": {"image": tool_image, "data_idx": f"geo3k_{split}_{idx}"},
                }

            return {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": DIVERSE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "data_idx": f"geo3k_{split}_{idx}",
                    "need_tools_kwargs": True,
                    "tools_kwargs": tools_kwargs,
                    "agent_name": "tool_agent",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: --local_dir is deprecated. Use --local_save_dir instead.")
    else:
        local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

    print(f"Wrote train ({len(train_dataset)}) and test ({len(test_dataset)}) parquets to {local_save_dir}")
