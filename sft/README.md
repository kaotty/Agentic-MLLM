# Agentic SFT pipeline

This directory bundles everything for the SFT phase of our project:

```
sft/
├── build_dataset/    # python package that converts DeepEyesV2 into our parquet schema
├── train/            # bash launchers that hand the parquets to verl.trainer.sft_trainer
└── sft_dataset/      # generated parquets (gitignored, ~3.7 GB)
```

It builds two SFT datasets that plug straight into `verl.trainer.sft_trainer`:

| Dataset | Tools used in trajectories | System prompt | Purpose |
| --- | --- | --- | --- |
| `zoom_only` | `image_zoom_in_tool` only | zoom-in only | "Cold-start the zoom-in tool RL" |
| `diverse`   | `image_zoom_in_tool` + `code_interpreter` + `search` + `image_search` | multi-tool | "Cold-start with broader tool exposure, same zoom-in subset for fair comparison" |

Both datasets share the exact same pure-zoom-in trajectories (sampled once and materialized into both files), so any RL-time delta between SFT-A and SFT-B on the zoom-tool axis is attributable to the diverse-tool exposure rather than to a different zoom-in sample.

Mixed zoom-using trajectories (zoom + at least one other tool) live in `diverse` only.

## Sources

| Source | HF repo | Used for |
| --- | --- | --- |
| DeepEyesV2 SFT (12.7 GB) | [`honglyhly/DeepEyesV2_SFT`](https://huggingface.co/datasets/honglyhly/DeepEyesV2_SFT) | All tool trajectories. Crop-style Python is rewritten as `image_zoom_in_tool`; non-crop Python becomes `code_interpreter`; web `search` calls are reshaped to our `search.query_list` schema; `image_search` (reverse-image search, no args) is surfaced as a separate tool. |

The dataset has 65,807 raw samples; ~52k are text-only math problems with no images (used as `code_interpreter`-only material elsewhere).
The ~13k image-bearing samples are what feed the five pools below.


## Workflow

The build is fully deterministic given a fixed `--seed` (default 42) and the immutable DeepEyesV2 source. 

### One-shot reproducible build (run on everyone's server)

From the project root:

```bash
python -m sft.build_dataset.build
```

By default this:

* downloads `honglyhly/DeepEyesV2_SFT` into `$HF_HOME` (default `~/.cache/huggingface`); ~12 GB zip + ~13 GB extracted on first run, then short-circuits on the HF cache;
* uses all image-bearing trajectories (no sampling cap);
* writes parquets under `<project_root>/sft/sft_dataset/`.

End-to-end wall-clock on a warm cache: ~5 min. Disk: ~25 GB for the HF cache, ~3.7 GB for the produced parquets (the `diverse/train.parquet` is ~3.6 GB on its own because each row carries the source images as JPEG bytes).

You can override the output location explicitly if you want the parquets to live somewhere else (e.g. on a different volume):

```bash
python -m sft.build_dataset.build --output_dir /path/to/agentic_sft
```

### Output

```
<project_root>/sft/sft_dataset/
  zoom_only/train.parquet          #   84 rows  — pure-zoom only (DeepEyesV2 ceiling)
  zoom_only/test.parquet           #    1 row
  diverse/train.parquet            # 12,972 rows — pure-zoom + code + image_search
                                   #               + mixed_zoom + mixed_non_zoom
  diverse/test.parquet             #  264 rows
```

Tool-call distribution on `diverse/train`:
~12.6 k `code_interpreter`, ~530 `image_zoom_in_tool`, ~480 `image_search`, ~310 `search` — all four tools are exercised at SFT time.
(`search` calls show up only as part of `mixed_non_zoom` trajectories; the source has no pure-search examples.)


### Empirical pool sizes (all 5 shards)

13,244 image-bearing samples (the other ~52k samples in DeepEyesV2 are text-only `code_interpreter` math problems we ignore):

| Pool | Total | Goes into |
| --- | --- | --- |
| `zoom` (pure-zoom + single user image) | 89 | `zoom_only` AND `diverse` |
| `code` (only `code_interpreter`) | 12,356 | `diverse` only |
| `search` (only web `search`) | 0 | `diverse` only |
| `image_search` (only reverse-image-search) | 176 | `diverse` only |
| `mixed_zoom` (zoom + at least one other tool) | 295 | `diverse` only |
| `mixed_non_zoom` (multi-tool, no zoom — mostly `search`+`image_search`) | 320 | `diverse` only |


### Note: `code_interpreter` dominates `diverse`

We use the full DeepEyesV2 pool, and thus `code_interpreter` makes up ~12.6 k of the ~13 k turns in `diverse/train` — roughly 95% of the tool-call mass.
That's fine for breadth-of-tool exposure (the model still sees every tool's call format and response format), but if SFT eval shows the smaller tools (`image_zoom_in_tool` / `image_search` / `search`) getting drowned out, the cheap fix is to oversample the smaller pools when materialising rows.


## Output schema

Each parquet row matches the contract of `verl/utils/dataset/multiturn_sft_dataset.py`:

| column | type | notes |
| --- | --- | --- |
| `messages` | `list[dict]` | `system` / `user` / `assistant` / `tool` roles. `<image>` placeholders match `images` order. Assistant turns with tool invocations carry an OpenAI-style `tool_calls` field. |
| `images` | `list[{bytes, path}]` | JPEG bytes (re-encoded via PIL). For zoom turns the cropped image is freshly baked with the runtime tool's `MIN_DIMENSION`-clamp logic so SFT input matches RL output. |
| `tools` | `list[dict]` | OpenAI tool schemas advertised to the model. Same list for every row in a given dataset. |
| `data_source` | `str` | `deepeyesv2_sft.zoom_only` / `deepeyesv2_sft.diverse` |
| `ability` | `str` | Heuristic tag: `perception` / `math_code` / `search` / `image_search` |
| `extra_info` | `dict` | Source file + raw index + per-call counts (zoom / code / search / image_search) for debugging. |

The trainer needs no code changes — `data.messages_key=messages`, `data.tools_key=tools`, `data.image_key=images` (the default).


## Filtering rules when I construct the SFT datasets (so converted samples never mislead the model)

* **Single user image only** — `image_zoom_in_tool` operates on one image; we keep multi-image questions out of zoom-only and out of any zoom turn.
* **At most one tool call per assistant turn** — the multi-turn agent loop iterates per turn anyway. Avoids parallel-call ambiguity.
* **Static crop bbox required** — if the source `image.crop()` bbox is built from variables we can't recover the integer `[x1, y1, x2, y2]` without executing the Python; such trajectories fall through to `code_interpreter` (and are dropped from `zoom_only`).
* **No ML-detection helpers in "pure crop"** — code blocks that contain `cv2.canny`, OCR, or model-forward calls are classified as `code_interpreter` even when they happen to call `.crop(...)` afterwards.
* **Pure-crop classification uses response-image-count as ground truth**: exactly one `image_*.crop((x1,y1,x2,y2))` on the input-image variable AND a tool response that contains exactly one image. Multi-imshow blocks (`imshow(whole)` → `imshow(crop_a)` → `imshow(crop_b)`) automatically demote to `code_interpreter` because their response carries multiple images, which a single `image_zoom_in_tool` call cannot explain.
* **Tool-response image counts must match the call** — `zoom` → 1 image, `search` → 0 images, `image_search` → ≥1 thumbnails. Trajectories that violate this invariant are discarded outright (not silently re-aligned).
* **`<answer>X</answer>` → `\boxed{X}`** — keeps SFT outputs aligned with the RL reward function in `verl/utils/reward_score/geo3k.py`.
* **`image_search` is now first-class** — it's a no-arg tool that reverse-image-searches the latest image and returns visually similar web thumbnails plus titles. Trajectories that use it are kept (formerly dropped as "unsupported").


## Where do these files plug in?

After building, kick off SFT from the project root:

```bash
bash sft/train/run_qwen2_5_vl_7b_sft_zoom.sh
bash sft/train/run_qwen2_5_vl_7b_sft_diverse.sh
```

Both scripts default to `<project_root>/sft/sft_dataset/{zoom_only,diverse}/{train,test}.parquet` (matching the build's default output) and feed `verl.trainer.sft_trainer`.
After SFT finishes, run the existing RL pipeline (`examples/sglang_multiturn/geo3k/run_qwen2_5_vl_7b_geo3k_multiturn.sh`) twice — once initialized from each SFT checkpoint — to do the head-to-head comparison.
