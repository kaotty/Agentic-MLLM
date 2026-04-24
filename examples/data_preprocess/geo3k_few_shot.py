# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 Reallm Labs Ltd. or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/geo3k_few_shot",
        help="The save directory for the preprocessed dataset.",
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

    # instruction_following = (
    #     r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    #     r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    #     r"The final answer MUST BE put in \boxed{}."
    # )

    instruction_following = (
        r"Please output your reasoning process within <think> </think> tags. "
        r"During your reasoning, you MUST use the `execute_python_code` tool to verify your calculations or process the image before answering. "
        r"The tool calls MUST BE enclosed within <tool_call> </tool_call> tags. "
        r"After receiving the observation, wrap your final answer in \\boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")

            system_prompt = (
                "You are an expert mathematical, geometric, and visual reasoner. \n"
                "Analyze the user's request and the provided image step by step to solve the problem.\n\n"
                "**Available Tools**\n"
                "You have access to a powerful, persistent Python sandbox via the `execute_python_code` tool.\n\n"
                "Tool Schema:\n"
                "{\n"
                "  \"name\": \"execute_python_code\",\n"
                "  \"description\": \"Executes Python code in a stateful sandbox. Useful for heavy math, data analysis, and image processing (cropping, color sampling, etc.).\",\n"
                "  \"parameters\": {\"code\": \"A string containing the Python code to execute. Properly escape newlines (\\\\n).\"}\n"
                "}\n\n"
                "**Sandbox Environment & Rules (CRITICAL)**:\n"
                "1. **Accessing the Image:** The user's input image is saved in the current working directory as 'input_image.jpg'. You can load it using: `from PIL import Image\\nimg = Image.open('input_image.jpg')`.\n"
                "2. **Auto-Display Mechanism:** The sandbox automatically evaluates the LAST expression in your code.\n"
                "   - If the last line evaluates to a PIL Image or Numpy array (e.g., just writing `sub_img`), the system will automatically display it in the observation.\n"
                "   - If the last line is a calculation, its value will be returned.\n"
                "   - You can also use standard `print()` or `plt.show()` (matplotlib is supported).\n"
                "3. **Persistent State:** Variables, functions, and imported modules from one tool call are preserved and can be used in subsequent tool calls within the same session.\n"
                "4. **Restrictions:** GUI blocking functions (like `cv2.imshow`, `cv2.waitKey`) and system termination commands (`exit()`, `quit()`, `sys.exit()`) are strictly **DISABLED**. Doing so will cause a framework error.\n\n"
                "**Workflow**\n"
                "1. **<think>:** Always plan your reasoning or code logic inside <think>...</think> tags first.\n"
                "2. **Execute:** Output your tool call EXACTLY in this format:\n"
                "   <tool_call> {\"name\": \"execute_python_code\", \"parameters\": {\"code\": \"from PIL import Image\\nimg = Image.open('input_image.jpg')\\nsub_img = img.crop((100, 100, 300, 300))\\nsub_img\"}} </tool_call>\n"
                "3. **Observe:** The system will return text outputs, error tracebacks, or displayed images. Use this observation to continue reasoning. If an error occurs, analyze it and write corrected code.\n"
                "4. **Final Answer:** Once the problem is fully solved, wrap your final, concise answer in \\\\boxed{}.\n\n"
                "**Examples of Tool Usage**\n\n"
                "Example 1: Mathematical Calculation\n"
                "User: What is the exact volume of a sphere with radius 7.5?\n"
                "Assistant: \n"
                "<think>\n"
                "I need to calculate V = (4/3) * pi * r^3. Python will prevent floating-point errors.\n"
                "</think>\n"
                "<tool_call> {\"name\": \"execute_python_code\", \"parameters\": {\"code\": \"import math\\nvolume = (4/3) * math.pi * (7.5 ** 3)\\nround(volume, 2)\"}} </tool_call>\n"
                "Observation: 1767.15\n"
                "Assistant: \n"
                "<think>\n"
                "The exact volume is 1767.15.\n"
                "</think>\n"
                "The volume of the sphere is 1767.15.\n"
                "\\\\boxed{1767.15}\n\n"
                "Example 2: Image Processing (Reading fine details)\n"
                "User: [Image attached] What is the tiny number written in the bottom right corner (approx x:1600-1700, y:1100-1200)?\n"
                "Assistant: \n"
                "<think>\n"
                "The text is too small. I will use PIL to crop the specific bounding box and leave the cropped image as the last expression so the system displays it for me.\n"
                "</think>\n"
                "<tool_call> {\"name\": \"execute_python_code\", \"parameters\": {\"code\": \"from PIL import Image\\nimg = Image.open('input_image.jpg')\\nbbox = [1600, 1100, 1700, 1200]\\nroi = img.crop(bbox)\\nroi\"}} </tool_call>\n"
                "Observation: [System provides the cropped image showing the number '42']\n"
                "Assistant: \n"
                "<think>\n"
                "The zoomed-in region clearly displays the number 42.\n"
                "</think>\n"
                "The number in the bottom right corner is 42.\n"
                "\\\\boxed{42}"
            )

            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,

                    "need_tools_kwargs": True,

                    "tools_kwargs": {
                        "execute_python_code": {
                            "class_path": "verl.tools.code_tool.CodeExecuteTool",
                            "create_kwargs": {
                                "config": {
                                    "type": "native"
                                }
                            },
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
