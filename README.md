This is the code repository for the course projects of CS285 and CS288 in spring 2026.

Our code is built upon [veRL](https://github.com/volcengine/verl).

## Installation

Run the following commands for installation:

```
conda create -n verl python==3.12
conda activate verl
# cd verl
bash scripts/install_vllm_sglang_mcore.sh
```

To match the sglang version, torch should be >= 2.9.
If there is a problem with flash attention, try the following command:

```
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation
```

## Baseline

For the baseline w/o tool use, run the following commands:

```
python examples/data_preprocess/geo3k.py
./examples/grpo_trainer/run_qwen2_5_vl_7b_npu.sh
```

For the baseline w/ tool use, run the following commands:
```
python examples/data_preprocess/geo3k_multiturn_w_tool.py
./examples/sglang_multiturn/geo3k/run_qwen2_5_vl_7b_geo3k_multiturn.sh
```

# SFT
SFT data is from DeepEyesV2.
To adapt to our pipeline, we transfered the data schema.
Check sft/README.md for details.

# Tool Set

The runtime tools live in `verl/tools/` and originate from veRL's [multi-turn agent loop](https://github.com/volcengine/verl). We extended them to match the full tool set used by [DeepEyesV2](https://arxiv.org/abs/2511.05271), which organizes tools into two categories:

**Operation tools** (code execution) — crop/zoom, numerical analysis, and image annotation all run through a single code interpreter:

| Tool | Class | Description |
|------|-------|-------------|
| `image_zoom_in_tool` | `ImageZoomInTool` | Crops a bounding-box region from the input image for fine-grained perception. |
| `code_interpreter` | `CodeExecuteTool` / `SandboxFusionTool` | Executes arbitrary Python (computation, plotting, image processing) in a sandboxed environment. |

**Information retrieval tools** (web search) — access external knowledge:

| Tool | Class | Description |
|------|-------|-------------|
| `search` | `SearchTool` | Text-based web search returning titles, snippets, and links. |
| `image_search` | `ImageSearchTool` | Reverse-image search returning visually matched web results with thumbnails. |

Tool configs are YAML files in `examples/sglang_multiturn/config/tool_config/`. Use `deepeyesv2_diverse_config.yaml` to load all four tools together for RL training with the full DeepEyesV2 tool set.
