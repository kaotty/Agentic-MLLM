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

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

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
        # torch==2.9.1 resolves to the CUDA 12.8 wheel (`+cu128`). SGLang
        # scheduler subprocesses import torch after CUDA library paths are set,
        # so a CUDA 12.4 base image can make them load the older system
        # libcudart.so.12 first and crash with:
        # `undefined symbol: cudaGetDriverEntryPointByVersion`.
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
        # libnuma1 is REQUIRED at import time by sgl_kernel (sglang's C++ kernel
        # backend). Without it, `import sglang.srt.entrypoints.engine` raises
        # `ImportError: libnuma.so.1: cannot open shared object file`, which
        # crashes the rollout init in `rl_diverse` before training even starts.
        "libnuma1",
        # veRL's NUMA affinity helper loads the unversioned soname via
        # ctypes.CDLL("libnuma.so"), which is provided by the dev package.
        "libnuma-dev",
    )
    .pip_install("packaging", "wheel", "setuptools>=61.0", "ninja")
    # Pin torch == 2.9.1 BEFORE sglang so that pip doesn't downgrade us. sglang
    # 0.5.8's prebuilt sgl_kernel wheel is compiled against torch 2.9's C++ ABI
    # (specifically the `c10::SymInt::maybe_as_int_slow_path` symbol introduced
    # in 2.9). Any older torch -> sgl_kernel fails to load with
    # `undefined symbol: _ZNK3c106SymInt22maybe_as_int_slow_pathEv`.
    # 2.9.1 is what sglang 0.5.8 itself strict-pins (`torch==2.9.1`); using the
    # exact same version avoids pip wasting a pass upgrading us during the
    # sglang install.
    .pip_install("torch==2.9.1", "torchvision", "torchaudio")
    # Inference framework: sglang only.
    # NB: scripts/install_vllm_sglang_mcore.sh pins sglang==0.5.2, but the verl
    # source in this fork already calls into the newer pause/continue-generation
    # API (`ContinueGenerationReqInput`, `PauseGenerationReqInput` in
    # async_sglang_server.py), which only exists in sglang>=0.5.5. The install
    # script lags the source. Pin to 0.5.8 to match setup.py and
    # requirements_sglang.txt.
    #
    # We deliberately DO NOT install vllm:
    #  - We always run `actor_rollout_ref.rollout.name=sglang`, so vllm is never
    #    loaded at runtime.
    #  - Every `import vllm` in this fork lives under `verl/workers/rollout/
    #    vllm_rollout/*` (only reached by `_load_vllm`), `verl/utils/vllm/*`
    #    (only imported transitively from vllm_rollout), or in two checkpoint
    #    engines (`hccl_checkpoint_engine.py`, `mooncake_checkpoint_engine.py`)
    #    whose imports are wrapped in try/except in `verl/checkpoint_engine/
    #    __init__.py`.
    #  - `_load_sglang` already injects a mock `vllm` module if `import vllm`
    #    fails, so the sglang rollout path works without vllm installed.
    #  - vllm 0.11.0 hard-pins `torch==2.8.0`, vllm 0.11.1/0.11.2/0.12.0 pin
    #    `torch==2.9.0`, but sglang 0.5.8 pins `torch==2.9.1`. There is no vllm
    #    version whose torch pin is compatible with our pinned sglang, so any
    #    attempt to coinstall the two silently downgrades torch and breaks
    #    sgl_kernel + flash_attn ABI.
    .pip_install(
        "sglang==0.5.8",
        "torch-memory-saver",
    )
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
        "cachetools",
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
        # Code-interpreter sandbox dependencies. `verl/tools/code_tool.py`'s
        # `worker_main_loop` unconditionally imports matplotlib (for plotting,
        # forced to the 'Agg' backend) and cv2 (for image processing); without
        # these every CodeExecWorker dies on startup with ModuleNotFoundError
        # and the `code_interpreter` tool becomes a no-op. scipy/sympy are
        # commonly emitted by the model for geometry problems on Geo3K, so we
        # ship them too rather than fail the first time a tool call uses them.
        "matplotlib",
        "opencv-python-headless",
        "scipy",
        "sympy",
    )
    # Flash-Attention prebuilt wheel matched to torch 2.9.
    # IMPORTANT: torch 2.9 on Linux x86_64 ships C++11 ABI = TRUE by default
    # (manylinux_2_28), so we MUST pick the `cxx11abiTRUE` wheel. Picking the
    # abiFALSE wheel will load but crash later with std::string ABI mismatches.
    # NB: pip validates the wheel filename against PEP 427, so we MUST NOT
    # rename it via ``wget -O``. Pass the URL directly to pip instead.
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    )
    # SGLang 0.5.8 refuses to start with torch 2.9.1 + CuDNN < 9.15 because of
    # a known PyTorch Conv3d/CuDNN regression. Install this after torch/sglang
    # so the newer Python CuDNN wheel wins over torch's transitive default.
    .pip_install("nvidia-cudnn-cu12==9.16.0.29")
    # NB: don't pin flashinfer-python here. sglang==0.5.8 already pulls in
    # flashinfer_python==0.6.1 (matching sgl_kernel==0.3.21's compiled ABI).
    # The previous explicit pin to 0.3.1 came from an older sglang==0.5.2 setup
    # in scripts/install_vllm_sglang_mcore.sh, and would silently DOWNGRADE
    # flashinfer back to 0.3.1, which breaks sgl_kernel at runtime.
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

def _run(
    cmd: list[str] | str,
    env: dict | None = None,
    cwd: str = WORKSPACE,
    log_path: str | Path | None = None,
) -> None:
    """Run a subprocess and stream its output. Raise on non-zero exit."""
    print(f"\n$ {cmd}\n", flush=True)
    full_env = {**os.environ, **(env or {})}
    if log_path is None:
        if isinstance(cmd, str):
            subprocess.check_call(cmd, shell=True, env=full_env, cwd=cwd)
        else:
            subprocess.check_call(cmd, env=full_env, cwd=cwd)
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"{datetime.now(timezone.utc).isoformat()} cwd={cwd}\n")
        log.write(f"$ {cmd}\n\n")

        process = subprocess.Popen(
            cmd,
            shell=isinstance(cmd, str),
            env=full_env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)

        retcode = process.wait()
        log.write(f"\n[exit status: {retcode}]\n")
        if retcode:
            raise subprocess.CalledProcessError(retcode, cmd)


def _repair_qwen25vl_config(model_dir: str | Path) -> None:
    """Repair Qwen2.5-VL merged checkpoints whose root config is saved as text-only."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"HF config not found, skipping Qwen2.5-VL config repair: {config_path}", flush=True)
        return

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    old_model_type = cfg.get("model_type")
    changed = False

    # The nested text_config is expected to have model_type=qwen2_5_vl_text. The
    # root config must remain qwen2_5_vl so SGLang's multimodal rope code takes
    # the Qwen2.5-VL path instead of raising "Unimplemented model type".
    if old_model_type == "qwen2_5_vl_text":
        cfg["model_type"] = "qwen2_5_vl"
        changed = True

    if cfg.get("model_type") == "qwen2_5_vl":
        expected_arch = ["Qwen2_5_VLForConditionalGeneration"]
        if cfg.get("architectures") != expected_arch:
            cfg["architectures"] = expected_arch
            changed = True

    if not changed:
        print(
            f"HF config OK: model_type={cfg.get('model_type')} architectures={cfg.get('architectures')}",
            flush=True,
        )
        return

    backup_path = config_path.with_suffix(".json.before_qwen25vl_repair")
    if not backup_path.exists():
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(
        "Repaired HF config: "
        f"{config_path} model_type {old_model_type!r} -> {cfg.get('model_type')!r}, "
        f"architectures={cfg.get('architectures')}",
        flush=True,
    )


def _patch_sglang_qwen25vl_text_rope(log_path: str | Path | None = None) -> None:
    """Patch SGLang 0.5.x to treat Qwen2.5-VL's text_config type as Qwen2.5-VL."""
    script = "\n".join([
        "from pathlib import Path",
        "import importlib.util",
        "import sglang.srt.layers.rotary_embedding as rotary_embedding",
        "path = Path(rotary_embedding.__file__)",
        "text = path.read_text(encoding='utf-8')",
        "marker = '# Agentic-MLLM compatibility: Qwen2.5-VL text_config model type'",
        "if marker in text:",
        "    print(f'SGLang Qwen2.5-VL RoPE text_config patch already present in {path}', flush=True)",
        "    raise SystemExit(0)",
        "if '\"qwen2_5_vl\"' in text or \"'qwen2_5_vl'\" in text:",
        "    target_model_type = 'qwen2_5_vl'",
        "elif '\"qwen2_vl\"' in text or \"'qwen2_vl'\" in text:",
        "    target_model_type = 'qwen2_vl'",
        "else:",
        "    raise RuntimeError(f'Could not find a Qwen-VL branch to patch in {path}')",
        "lines = text.splitlines(keepends=True)",
        "start = None",
        "for i, line in enumerate(lines):",
        "    if line.lstrip().startswith('def get_rope_index('):",
        "        start = i",
        "        break",
        "if start is None:",
        "    raise RuntimeError(f'Could not find get_rope_index in {path}')",
        "balance = 0",
        "end = None",
        "for i in range(start, len(lines)):",
        "    balance += lines[i].count('(') - lines[i].count(')')",
        "    if balance <= 0 and lines[i].rstrip().endswith(':'):",
        "        end = i",
        "        break",
        "if end is None:",
        "    raise RuntimeError(f'Could not find get_rope_index signature end in {path}')",
        "def_indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]",
        "body_indent = def_indent + '    '",
        "insert = [",
        "    f'{body_indent}{marker}\\n',",
        "    f'{body_indent}if model_type in (\\'qwen2_5_vl\\', \\'qwen2_5_vl_text\\'):\\n',",
        "    f'{body_indent}    model_type = {target_model_type!r}\\n',",
        "]",
        "lines[end + 1:end + 1] = insert",
        "path.write_text(''.join(lines), encoding='utf-8')",
        "cache_path = Path(importlib.util.cache_from_source(str(path)))",
        "cache_path.unlink(missing_ok=True)",
        "print(",
        "    f'Patched SGLang Qwen2.5-VL RoPE text_config support in {path}: '",
        "    f'qwen2_5_vl_text -> {target_model_type}',",
        "    flush=True,",
        ")",
    ])
    _run(["python3", "-c", script], log_path=log_path)


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
    _repair_qwen25vl_config(dst)
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
    resume_mode: str = "auto",
) -> None:
    """GRPO with the diverse 4-tool config, initialized from the SFT-merged ckpt."""
    sft_hf = f"{CKPT_DIR}/{sft_exp_name}/hf_merged"
    if not Path(sft_hf).exists():
        raise RuntimeError(
            f"SFT-merged checkpoint not found at {sft_hf}. "
            f"Run merge_ckpt first."
        )
    _repair_qwen25vl_config(sft_hf)
    VOL_CKPT.commit()

    cfg_path = f"{WORKSPACE}/examples/sglang_multiturn/config"
    tool_cfg = f"{cfg_path}/tool_config/deepeyesv2_diverse_config.yaml"
    reward_path = f"{WORKSPACE}/verl/utils/reward_score/geo3k_w_tools.py"
    ckpt_home = f"{CKPT_DIR}/rl/{rl_exp_name}"
    log_path = f"{ckpt_home}/logs/rl_diverse.log"

    cmd = " \\\n    ".join([
        "python3 -m verl.trainer.main_ppo",
        f"--config-path={cfg_path}",
        "--config-name=geo3k_multiturn_grpo",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        "algorithm.adv_estimator=grpo",
        f"data.train_batch_size={train_batch_size}",
        "data.max_prompt_length=4096",
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
        f"trainer.resume_mode={resume_mode}",
        f"trainer.total_epochs={epochs}",
    ])
    env = {
        "HYDRA_FULL_ERROR": "1",
        "RAY_DEDUP_LOGS": "0",
        "SGLANG_WATCHDOG_TIMEOUT": "1800",
        # search tool reads this; empty string → soft-fail mode.
        "SEARCH_SERVICE_URL": "",
        # Reward weights (REWARD_W_*) come from defaults baked into
        # verl/utils/reward_score/geo3k_w_tools.py — namely {1.0, 0.1, 1.0, 0.5}
        # for {acc, base_fmt, tool_fmt, penalty}. We don't override them here
        # so that this run uses exactly the same reward shaping as every other
        # cell in the 2×3 ablation.
    }
    print(f"\nRL subprocess log will be written to {log_path}", flush=True)
    try:
        _patch_sglang_qwen25vl_text_rope(log_path=log_path)
        _run(
            [
                "python3",
                "-c",
                "\n".join([
                    "import importlib",
                    "import importlib.metadata as md",
                    "import json",
                    f"cfg=json.load(open({str(Path(sft_hf) / 'config.json')!r}))",
                    "print(f\"hf_config model_type={cfg.get('model_type')} architectures={cfg.get('architectures')}\", flush=True)",
                    "import torch",
                    "print(f'torch=={torch.__version__} cuda={torch.version.cuda}', flush=True)",
                    "print(f'cudnn={torch.backends.cudnn.version()}', flush=True)",
                    "for mod in ['cachetools', 'sglang', 'sglang.srt.entrypoints.engine', 'sgl_kernel', 'flashinfer', 'flash_attn']:",
                    "    importlib.import_module(mod)",
                    "    print(f'import ok: {mod}', flush=True)",
                    "for pkg in ['sglang', 'sgl-kernel', 'flashinfer-python', 'flash-attn', 'transformers', 'torch', 'nvidia-cudnn-cu12']:",
                    "    print(f'{pkg}=={md.version(pkg)}', flush=True)",
                ]),
            ],
            env=env,
            log_path=log_path,
        )
        _run(f"ulimit -n 65535 && {cmd}", env=env, log_path=log_path)
    finally:
        VOL_CKPT.commit()
        print(f"\nCommitted checkpoint volume. RL log path: {log_path}", flush=True)


# ---------------------------------------------------------------------------
# 5b. Debug helper: inspect the persisted RL log and resume tracker
# ---------------------------------------------------------------------------

@app.function(
    timeout=10 * 60,
    volumes={CKPT_DIR: VOL_CKPT},
)
def tail_rl_log(
    rl_exp_name: str = "qwen2_5_vl_7b_rl_diverse_after_sft",
    lines: int = 300,
) -> None:
    """Print the tail of the persisted rl_diverse log and checkpoint tracker."""
    ckpt_home = Path(CKPT_DIR) / "rl" / rl_exp_name
    log_path = ckpt_home / "logs" / "rl_diverse.log"
    tracker_path = ckpt_home / "latest_checkpointed_iteration.txt"

    print(f"checkpoint dir: {ckpt_home}")
    if tracker_path.exists():
        step = tracker_path.read_text(encoding="utf-8").strip()
        print(f"resume tracker: {tracker_path} -> global_step_{step}")
    else:
        print(f"resume tracker missing: {tracker_path}")

    if ckpt_home.exists():
        global_steps = sorted(p.name for p in ckpt_home.glob("global_step_*"))
        print(f"global_step dirs: {global_steps[-10:] if global_steps else []}")
    else:
        print("checkpoint dir does not exist yet")

    print(f"\nlog path: {log_path}")
    if not log_path.exists():
        print("log file does not exist yet")
        return

    tail = deque(maxlen=max(1, lines))
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        tail.extend(f)

    print(f"\n--- last {len(tail)} lines ---")
    for line in tail:
        print(line, end="")


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
