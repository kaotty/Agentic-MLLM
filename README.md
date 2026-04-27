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
