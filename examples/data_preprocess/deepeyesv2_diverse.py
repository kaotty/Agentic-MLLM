"""Preprocess DeepEyesV2 RL data for the full diverse tool set.

Reads the DeepEyesV2 RL parquet files (perception + reason + search) and
produces a single training parquet with ``tools_kwargs`` for all four tools:
``image_zoom_in_tool``, ``code_interpreter``, ``search``, ``image_search``.

Usage::

    python examples/data_preprocess/deepeyesv2_diverse.py \
        --rl_data_dir ~/data/deepeyesv2/rl \
        --local_save_dir ~/data/deepeyesv2_diverse
"""

import argparse
import glob
import os

import datasets
import pandas as pd


DIVERSE_SYSTEM_PROMPT = (
    "You are a helpful multimodal expert. You are given a question and an accompanying image. "
    "You must reason step by step and may invoke any of the following tools to gather more "
    "evidence before answering:\n"
    "  * `image_zoom_in_tool`  — crop and inspect a region of the image\n"
    "  * `code_interpreter`     — run Python code (image processing, computation, plotting)\n"
    "  * `search`               — issue web search queries\n"
    "  * `image_search`         — reverse-image search the most recent image\n\n"
    "When you call a tool you MUST output exactly:\n"
    "<tool_call>\n"
    '{"name": "<tool_name>", "arguments": {...}}\n'
    "</tool_call>\n\n"
    "Use tools sparingly and only when they will help you answer the question. After gathering "
    "enough evidence, wrap your reasoning in <think> </think> tags and put the final answer in "
    "\\boxed{}."
)

INSTRUCTION_FOLLOWING = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{}."
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--rl_data_dir",
        type=str,
        default="~/data/deepeyesv2/rl",
        help="Directory containing DeepEyesV2 RL parquet files.",
    )
    p.add_argument(
        "--local_save_dir",
        type=str,
        default="~/data/deepeyesv2_diverse",
        help="Output directory for the preprocessed dataset.",
    )
    p.add_argument(
        "--image_search_cache_json",
        type=str,
        default=None,
        help="Path to image search cache JSON (for image_search create_kwargs).",
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.02,
        help="Fraction held out for validation.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rl_parquets(rl_data_dir: str) -> pd.DataFrame:
    """Load and concatenate all parquet files in the RL data directory."""
    rl_data_dir = os.path.expanduser(rl_data_dir)
    parquet_files = sorted(glob.glob(os.path.join(rl_data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {rl_data_dir}")
    print(f"Found {len(parquet_files)} parquet file(s) in {rl_data_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


def build_sample(row, idx: int, split: str, image_search_cache_json: str = None):
    """Convert one RL parquet row into a training-ready sample with tools_kwargs."""
    question = row.get("question") or row.get("problem") or ""
    answer = row.get("answer") or row.get("ground_truth") or ""
    images = row.get("images") or row.get("image") or []
    if not isinstance(images, list):
        images = [images]

    data_source = row.get("data_source", "deepeyesv2_rl")
    data_idx = row.get("data_idx") or row.get("idx") or f"{data_source}_{idx}"

    tool_image = images[0] if images else None
    prompt_text = question
    if INSTRUCTION_FOLLOWING not in question:
        prompt_text = question + " " + INSTRUCTION_FOLLOWING

    tools_kwargs = {}

    # image_zoom_in_tool needs the image passed at create time
    if tool_image is not None:
        tools_kwargs["image_zoom_in_tool"] = {
            "create_kwargs": {"image": tool_image},
        }

    # code_interpreter needs the image for input_image access
    if tool_image is not None:
        tools_kwargs["code_interpreter"] = {
            "create_kwargs": {"image": tool_image},
        }

    # image_search needs data_idx for cache lookup
    if tool_image is not None:
        create_kwargs = {}
        if image_search_cache_json:
            create_kwargs["data_idx"] = str(data_idx)
        create_kwargs["image"] = tool_image
        tools_kwargs["image_search"] = {
            "create_kwargs": create_kwargs,
        }

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": DIVERSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "images": images,
        "ability": row.get("ability", "general"),
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer,
            "question": question,
            "data_idx": str(data_idx),
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
            "agent_name": "tool_agent",
        },
    }


def main():
    args = parse_args()

    df = load_rl_parquets(args.rl_data_dir)
    print(f"Total samples: {len(df)}")

    # Split
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_test = max(1, int(len(df) * args.test_frac))
    test_df = df.iloc[:n_test]
    train_df = df.iloc[n_test:]

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        samples = []
        for idx, (_, row) in enumerate(split_df.iterrows()):
            sample = build_sample(
                row, idx, split_name,
                image_search_cache_json=args.image_search_cache_json,
            )
            samples.append(sample)

        ds = datasets.Dataset.from_list(samples)
        out_path = os.path.join(local_save_dir, f"{split_name}.parquet")
        ds.to_parquet(out_path)
        print(f"Wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
