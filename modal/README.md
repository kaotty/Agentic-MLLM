# Modal: SFT-then-RL × Diverse-Tool

Run order on **1 node × 4 H100 (80GB)**.

This cell of the 2×3 ablation matrix:

| Setup | Tool set |
| --- | --- |
| **(3) Pretrain via SFT, then GRPO** | **Diverse**: `image_zoom_in_tool` + `code_interpreter` + `search` + `image_search` |

Search and image_search run in **soft-fail mode** for Geo3K (returns a fixed "service unavailable" message); Bing Visual Search API was retired 2025-08-11 and Geo3K reverse-image-search would be noise anyway. The point of the diverse tool config is matching the SFT-time tool schema, not real retrieval.


## 0. One-time setup

```bash
pip install modal
modal token new                         # browser auth

# wandb key — Modal stores it as an env var on tagged Functions.
modal secret create wandb WANDB_API_KEY=<your key>

# (Project root)
cd /path/to/Agentic-MLLM
```

Make sure you've already built the SFT parquets locally (only needed once, check sft/README.md; they're then cached in a Modal Volume forever):

```bash
python -m sft.build_dataset.build      # produces sft/sft_dataset/{zoom_only,diverse}/
```

---

## 1. Push prebuilt SFT data → Modal Volume

```bash
modal run modal/app.py::upload_sft_data
```

This rsyncs `sft/sft_dataset/*` (~3.7 GB) into the `agentic-mllm-data` Volume.

---

## 2. SFT (maybe 6 hours, detached)

```bash
modal run --detach modal/app.py::sft_diverse
```

Default args (in `app.py::sft_diverse`):

| arg | value | why |
| --- | --- | --- |
| `epochs` | 2 | 13k samples × 2 epoch ≈ 800 steps; longer over-fits the small zoom subset |
| `train_bsz` | 32 | 4×H100 + SP=2 + offload, the original 64 doesn't fit |
| `max_length` | 10240 | tool-rich trajectories are long; shorter than default 12288 to leave headroom |
| `max_token_len_per_gpu` | 16384 | dynamic-bsz packing budget per GPU |

If need to override:

```bash
modal run --detach modal/app.py::sft_diverse --epochs 3 --train-bsz 24
```

Monitor:

```bash
modal app logs agentic-mllm-sft-then-rl-diverse
# or wandb (project: agentic-mllm-sft, exp: qwen2_5_vl_7b_sft_diverse_4xh100)
```

When it finishes, note the **last** `global_step_N` printed in the logs.

---

## 3. Merge FSDP shards → HuggingFace folder

```bash
modal run modal/app.py::merge_ckpt --global-step <N>
```

Replace <N> with the actual last step, e.g. 800.
Output: `agentic-mllm-ckpt://qwen2_5_vl_7b_sft_diverse_4xh100/hf_merged/`

---

## 4. RL data preprocess

```bash
modal run modal/app.py::prep_rl_data
```

Runs `examples/data_preprocess/geo3k_diverse.py` inside the container. 
Writes parquets with the **DIVERSE_SYSTEM_PROMPT** (4-tool menu) and `tools_kwargs` for all four tools to `agentic-mllm-ckpt://rl_data/geo3k_diverse/`.

---

## 5. RL (maybe 6 hours, detached)

```bash
modal run --detach modal/app.py::rl_diverse
```

Notes on what this run does (vs. the baseline `run_qwen2_5_vl_7b_geo3k_multiturn.sh`):

* `actor_rollout_ref.model.path` → SFT-merged HF folder
* `actor_rollout_ref.rollout.multi_turn.tool_config_path` → `deepeyesv2_diverse_config.yaml` (4 tools, search/image_search soft-fail)
* `data.train_files` → diverse-tools parquet from step 4
* `trainer.save_freq=50` (was `-1`) so we don't lose progress


Override common knobs:

```bash
modal run --detach modal/app.py::rl_diverse --epochs 3 --rollout-n 4
```

Monitor: same as SFT (wandb project: `agentic-mllm-rl`, exp: `qwen2_5_vl_7b_rl_diverse_after_sft`).

---

## 6. (Optional) Eval

```bash
modal run modal/app.py::eval_diverse                # uses latest checkpoint
modal run modal/app.py::eval_diverse --global-step 350
```

---

# Inside Modal containers:
/workspace/Agentic-MLLM         ← repo (rebuilds on local code change)
/checkpoints                    ← VOL_CKPT (SFT + RL ckpts, RL data parquet)
/hf_cache                       ← VOL_HF  (Qwen2.5-VL weights, geo3k dataset)
/workspace/Agentic-MLLM/sft/sft_dataset
                                ← VOL_DATA (the diverse parquet bundle)
```
