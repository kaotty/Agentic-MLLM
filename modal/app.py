"""Modal app for the SFT-then-RL × diverse-tool experiment cell.

Runs on 1 node × 4 H100 (80GB). Five entrypoints, intended to be invoked in order:

    modal run modal/app.py::upload_sft_data       # one-shot, ~3.7 GB local → Volume
    modal run --detach modal/app.py::sft_diverse  # ~6h SFT
    modal run modal/app.py::merge_ckpt --global-step <N>
    modal run modal/app.py::prep_rl_data          # geo3k → tools_kwargs parquet
    modal run --detach modal/app.py::rl_diverse   # ~6h RL
    modal run modal/app.py::eval_diverse --global-step <N>

See ``modal/README.md`` for the end-to-end walk-through.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "agentic-mllm-sft-then-rl-diverse"
GPU = "H100:4"
TIMEOUT = 24 * 60 * 60  # 24h per call; long-running jobs use --detach

# Where things live inside the container.
WORKSPACE = "/workspace/Agentic-MLLM"
DATA_DIR = f"{WORKSPACE}/sft/sft_dataset"     # mounted from VOL_DATA
CKPT_DIR = "/checkpoints"                       # mounted from VOL_CKPT
HF_CACHE = "/hf_cache"                          # mounted from VOL_HF

# Project root on the user's local machine, used by add_local_dir.
LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Volumes (persisted across runs)
# ---------------------------------------------------------------------------

VOL_DATA = modal.Volume.from_name("agentic-mllm-data", create_if_missing=True)
VOL_CKPT = modal.Volume.from_name("agentic-mllm-ckpt", create_if_missing=True)
VOL_HF = modal.Volume.from_name("agentic-mllm-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

# Heavy deps (sglang, vllm, flash-attn) are installed once during image build.
# Repo code is added via ``add_local_dir`` so that editing local files only
# triggers a fast layer rebuild (not a full reinstall).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "wget", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    .pip_install("packaging", "wheel", "setuptools>=61.0", "ninja")
    # Inference frameworks. Skip Megatron (USE_MEGATRON=0 in the install script's spirit).
    .pip_install(
        "sglang[all]==0.5.2",
        "torch-memory-saver",
    )
    .pip_install("vllm==0.11.0")
    # Core deps from scripts/install_vllm_sglang_mcore.sh.
    .pip_install(
        "transformers[hf_xet]>=4.51.0",
        "accelerate",
        "datasets",
        "peft",
        "hf-transfer",
        "numpy<2.0.0",
        "pyarrow>=15.0.0",
        "pandas",
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
        "torchdata",
        "ray[default]",
        "codetiming",
        "hydra-core",
        "pylatexenc",
        "qwen-vl-utils",
        "wandb",
        "dill",
        "pybind11",
        "liger-kernel",
        "mathruler",
        "nvidia-ml-py>=12.560.30",
        "fastapi[standard]>=0.115.0",
        "optree>=0.13.0",
        "pydantic>=2.9",
        "grpcio>=1.62.1",
    )
    # Flash-Attention 2.8.1 prebuilt wheel (cu12 + torch2.8 + python3.12).
    # NB: pip validates the wheel filename against PEP 427, so we MUST NOT
    # rename it via ``wget -O``. Pass the URL directly to pip instead.
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/"
        "flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .pip_install("flashinfer-python==0.3.1")
    # Bring the repo into the image.
    .add_local_dir(str(LOCAL_REPO_ROOT), WORKSPACE, copy=True, ignore=["sft/sft_dataset/*"])
    # Install verl in-place so ``python -m verl.trainer.sft_trainer`` resolves.
    .run_commands(f"cd {WORKSPACE} && pip install -e . --no-deps")
    .env({
        "HF_HOME": HF_CACHE,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Soft-fail search backend (see verl/tools/search_tool.py patch).
        "SEARCH_SERVICE_URL": "",
    })
)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str] | str, env: dict | None = None, cwd: str = WORKSPACE) -> None:
    """Run a subprocess and stream its output. Raise on non-zero exit."""
    print(f"\n$ {cmd}\n", flush=True)
    full_env = {**os.environ, **(env or {})}
    if isinstance(cmd, str):
        subprocess.check_call(cmd, shell=True, env=full_env, cwd=cwd)
    else:
        subprocess.check_call(cmd, env=full_env, cwd=cwd)


# ---------------------------------------------------------------------------
# 1. Upload prebuilt SFT data from local → Volume
# ---------------------------------------------------------------------------

@app.function(
    timeout=60 * 60,
    volumes={DATA_DIR: VOL_DATA},
    # The Volume mount target must be empty/owned by Modal; we copy local data
    # in via ``add_local_dir`` of the *parquet folder* and rsync into the Volume.
    image=(
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("rsync")
        .add_local_dir(
            str(LOCAL_REPO_ROOT / "sft" / "sft_dataset"),
            "/local_sft_dataset",
            copy=True,
        )
    ),
)
def upload_sft_data() -> None:
    """One-shot: copy ``sft/sft_dataset/{zoom_only,diverse}`` into VOL_DATA.

    Run this exactly once after building parquets locally (or after rebuilding).
    """
    _run("mkdir -p /workspace/Agentic-MLLM/sft/sft_dataset")
    _run("rsync -av --progress /local_sft_dataset/ /workspace/Agentic-MLLM/sft/sft_dataset/")
    VOL_DATA.commit()
    _run("ls -la /workspace/Agentic-MLLM/sft/sft_dataset/diverse")


# ---------------------------------------------------------------------------
# 2. SFT (diverse, 4×H100)
# ---------------------------------------------------------------------------

@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={DATA_DIR: VOL_DATA, CKPT_DIR: VOL_CKPT, HF_CACHE: VOL_HF},
    secrets=[modal.Secret.from_name("wandb")],
)
def sft_diverse(
    epochs: int = 2,
    train_bsz: int = 32,
    max_length: int = 24576,
    max_token_len_per_gpu: int = 16384,
    lr: float = 1e-5,
    exp_name: str = "qwen2_5_vl_7b_sft_diverse_4xh100",
) -> None:
    """Cold-start SFT on the diverse parquets. ~6-10 hours on 4×H100."""
    env = {
        "NUM_TRAINERS": "4",
        "SP_SIZE": "2",
        "FSDP_SIZE": "-1",
        "FSDP_STRATEGY": "fsdp2",
        "MAX_LENGTH": str(max_length),
        "TRAIN_BSZ": str(train_bsz),
        "EPOCHS": str(epochs),
        "LR": str(lr),
        "EXP_NAME": exp_name,
        "PROJECT_NAME": "agentic-mllm-sft",
        "CKPT_HOME": f"{CKPT_DIR}/{exp_name}",
        "MODEL_ID": "Qwen/Qwen2.5-VL-7B-Instruct",
    }
    _run(
        [
            "bash", "sft/train/run_qwen2_5_vl_7b_sft_diverse.sh",
            f"data.max_token_len_per_gpu={max_token_len_per_gpu}",
        ],
        env=env,
    )
    VOL_CKPT.commit()
    print(f"\nSFT done. Checkpoints under {env['CKPT_HOME']}")
    print("Next step: modal run modal/app.py::merge_ckpt --global-step <last_step>")


# ---------------------------------------------------------------------------
# 3. Merge FSDP shards → HuggingFace folder (so RL rollout can load it)
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100:1",
    timeout=2 * 60 * 60,
    volumes={CKPT_DIR: VOL_CKPT, HF_CACHE: VOL_HF},
)
def merge_ckpt(
    global_step: int,
    exp_name: str = "qwen2_5_vl_7b_sft_diverse_4xh100",
) -> None:
    """Convert the FSDP-sharded SFT ckpt into a Hugging Face folder.

    ``RL`` step uses ``actor_rollout_ref.model.path`` pointing at the merged dir.
    """
    src_root = f"{CKPT_DIR}/{exp_name}/global_step_{global_step}"
    # veRL's sft_trainer writes either ``<step>/`` or ``<step>/actor/``.
    src_actor = f"{src_root}/actor"
    src = src_actor if Path(src_actor).exists() else src_root
    dst = f"{CKPT_DIR}/{exp_name}/hf_merged"
    _run(
        [
            "python", "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--local_dir", src,
            "--target_dir", dst,
        ],
    )
    VOL_CKPT.commit()
    _run(f"ls -la {dst}")
    print(f"\nMerged HF model written to {dst}")
    print("Next step: modal run modal/app.py::prep_rl_data")


# ---------------------------------------------------------------------------
# 4. RL data preprocessing (geo3k + diverse tools_kwargs)
# ---------------------------------------------------------------------------

@app.function(
    timeout=60 * 60,
    volumes={CKPT_DIR: VOL_CKPT, HF_CACHE: VOL_HF},
)
def prep_rl_data(
    save_dir: str = f"{CKPT_DIR}/rl_data/geo3k_diverse",
) -> None:
    """Build ``geo3k`` parquets with full DeepEyesV2 tools_kwargs."""
    _run(
        [
            "python", "examples/data_preprocess/geo3k_diverse.py",
            f"--local_save_dir={save_dir}",
        ],
    )
    VOL_CKPT.commit()
    _run(f"ls -la {save_dir}")
    print("Next step: modal run --detach modal/app.py::rl_diverse")


# ---------------------------------------------------------------------------
# 5. RL with sglang multi-turn rollout, full DeepEyesV2 tool config
# ---------------------------------------------------------------------------

@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={CKPT_DIR: VOL_CKPT, HF_CACHE: VOL_HF},
    secrets=[modal.Secret.from_name("wandb")],
)
def rl_diverse(
    sft_exp_name: str = "qwen2_5_vl_7b_sft_diverse_4xh100",
    rl_data_dir: str = f"{CKPT_DIR}/rl_data/geo3k_diverse",
    rl_exp_name: str = "qwen2_5_vl_7b_rl_diverse_after_sft",
    epochs: int = 5,
    rollout_n: int = 5,
    train_batch_size: int = 512,
    save_freq: int = 50,
    test_freq: int = 25,
) -> None:
    """GRPO with the diverse 4-tool config, initialized from the SFT-merged ckpt."""
    sft_hf = f"{CKPT_DIR}/{sft_exp_name}/hf_merged"
    if not Path(sft_hf).exists():
        raise RuntimeError(
            f"SFT-merged checkpoint not found at {sft_hf}. "
            f"Run merge_ckpt first."
        )

    cfg_path = f"{WORKSPACE}/examples/sglang_multiturn/config"
    tool_cfg = f"{cfg_path}/tool_config/deepeyesv2_diverse_config.yaml"
    reward_path = f"{WORKSPACE}/verl/utils/reward_score/geo3k_w_tools.py"
    ckpt_home = f"{CKPT_DIR}/rl/{rl_exp_name}"

    cmd = " \\\n    ".join([
        "python3 -m verl.trainer.main_ppo",
        f"--config-path={cfg_path}",
        "--config-name=geo3k_multiturn_grpo",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        "algorithm.adv_estimator=grpo",
        f"data.train_batch_size={train_batch_size}",
        "data.max_prompt_length=2048",
        "data.max_response_length=2048",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"data.train_files={rl_data_dir}/train.parquet",
        f"data.val_files={rl_data_dir}/test.parquet",
        f"actor_rollout_ref.model.path={sft_hf}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5",
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.watchdog_timeout=600",
        f'actor_rollout_ref.rollout.multi_turn.tool_config_path="{tool_cfg}"',
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        'trainer.logger=["console","wandb"]',
        "trainer.project_name=agentic-mllm-rl",
        f"trainer.experiment_name={rl_exp_name}",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f'trainer.default_local_dir="{ckpt_home}"',
        "trainer.ray_wait_register_center_timeout=300",
        "trainer.resume_mode=auto",
        f"trainer.total_epochs={epochs}",
    ])
    env = {
        "SGLANG_WATCHDOG_TIMEOUT": "1800",
        # search tool reads this; empty string → soft-fail mode.
        "SEARCH_SERVICE_URL": "",
        # Reward weights (REWARD_W_*) come from defaults baked into
        # verl/utils/reward_score/geo3k_w_tools.py — namely {1.0, 0.1, 1.0, 0.5}
        # for {acc, base_fmt, tool_fmt, penalty}. We don't override them here
        # so that this run uses exactly the same reward shaping as every other
        # cell in the 2×3 ablation.
    }
    _run(f"ulimit -n 65535 && {cmd}", env=env)
    VOL_CKPT.commit()


# ---------------------------------------------------------------------------
# 6. (Optional) Eval a saved RL checkpoint
# ---------------------------------------------------------------------------

@app.function(
    gpu=GPU,
    timeout=4 * 60 * 60,
    volumes={CKPT_DIR: VOL_CKPT, HF_CACHE: VOL_HF},
    secrets=[modal.Secret.from_name("wandb")],
)
def eval_diverse(
    rl_exp_name: str = "qwen2_5_vl_7b_rl_diverse_after_sft",
    global_step: int | None = None,
) -> None:
    """Run the existing eval driver against an RL checkpoint."""
    ckpt_home = f"{CKPT_DIR}/rl/{rl_exp_name}"
    if global_step is None:
        # Pick the latest global_step_<N> by mtime.
        candidates = sorted(Path(ckpt_home).glob("global_step_*"))
        if not candidates:
            raise RuntimeError(f"No global_step_* under {ckpt_home}")
        ckpt = str(candidates[-1])
    else:
        ckpt = f"{ckpt_home}/global_step_{global_step}"
    _run(["python", "eval_checkpoints.py", "--checkpoint_dir", ckpt])


# ---------------------------------------------------------------------------
# Local entrypoint helper (so ``modal run modal/app.py`` prints help).
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print(__doc__)
