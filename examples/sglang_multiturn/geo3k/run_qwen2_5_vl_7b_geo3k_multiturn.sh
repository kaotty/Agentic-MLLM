# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535
# export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=1,3,4,5
export SGLANG_WATCHDOG_TIMEOUT=1800
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
# export RAY_TMPDIR="$PROJECT_DIR/my_ray_logs"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geo3k_multiturn_grpo' \
    custom_reward_function.path=$PROJECT_DIR/verl/utils/reward_score/geo3k_w_tools.py \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/home/hyunin_sakana_ai/models/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.watchdog_timeout=600 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo' \
    trainer.experiment_name='qwen2_5_vl_7b_zoom' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.ray_wait_register_center_timeout=300 \
    data.train_files=$HOME/data/geo3k_multiturn_zoom_tool/train.parquet \
    data.val_files=$HOME/data/geo3k_multiturn_zoom_tool/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/geo3k_zoom_config.yaml" \
    trainer.total_epochs=5 $@

