#!/usr/bin/env bash
# Run diverse-tool SFT on Qwen2.5-VL-7B using the parquets produced by
#   python -m sft.build_dataset.build --target diverse
# By default those parquets live under <project_root>/sft/sft_dataset/diverse/.
#
# This is the "control B" of the SFT comparison: same pure-zoom trajectories
# as the zoom-only run, plus code_interpreter / search / image_search /
# mixed-tool trajectories.
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/sft/sft_dataset/diverse}
TRAIN_FILES=${TRAIN_FILES:-${DATA_ROOT}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_ROOT}/test.parquet}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}

PROJECT_NAME=${PROJECT_NAME:-agentic-mllm-sft}
EXP_NAME=${EXP_NAME:-qwen2_5_vl_7b_sft_diverse}
CKPT_HOME=${CKPT_HOME:-${HOME}/checkpoints/${PROJECT_NAME}/${EXP_NAME}}
mkdir -p "${CKPT_HOME}"

NUM_TRAINERS=${NUM_TRAINERS:-8}
SP_SIZE=${SP_SIZE:-2}
FSDP_SIZE=${FSDP_SIZE:--1}
FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}

MAX_LENGTH=${MAX_LENGTH:-12288}
TRAIN_BSZ=${TRAIN_BSZ:-64}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-5}

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BSZ} \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=${MAX_LENGTH} \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu=20480 \
    data.messages_key=messages \
    data.tools_key=tools \
    data.num_workers=8 \
    model.path=${MODEL_ID} \
    model.use_remove_padding=true \
    engine=fsdp \
    optim=fsdp \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.05 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.total_epochs=${EPOCHS} \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.logger=['console','wandb'] \
    trainer.save_freq=1000 \
    trainer.test_freq=500 \
    trainer.max_ckpt_to_keep=3 \
    trainer.resume_mode=auto \
    checkpoint.save_contents=[model,optimizer,extra] \
    "$@"
