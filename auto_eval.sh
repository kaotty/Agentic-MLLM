#!/bin/bash
set -ex  # 开启严格模式：任何报错立刻停止，绝不带病运行
ulimit -n 65535

# ==========================================
# 0. 🌟 全局配置 (每次测新模型，只需修改这里！)
# ==========================================
EXPERIMENT_NAME="qwen2_5_vl_7b_zoom_w0.5"

export CUDA_VISIBLE_DEVICES=3,4,6,7
export SGLANG_WATCHDOG_TIMEOUT=1800
PROJECT_DIR="$(pwd)"
BASE_MODEL="/home/hyunin_sakana_ai/models/Qwen2.5-VL-7B-Instruct"

CKPT_DIR="$PROJECT_DIR/checkpoints/$EXPERIMENT_NAME"
LOG_FILE="$PROJECT_DIR/eval_results_${EXPERIMENT_NAME}_fresh.txt"

echo "=========================================="
echo "🎯 启动全新自动化评测流水线: $EXPERIMENT_NAME"
echo "=========================================="

# ==========================================
# 🔍 步骤 1: 自动定位最新的 Checkpoint
# ==========================================
LATEST_STEP=$(ls -vd "$CKPT_DIR"/global_step_* 2>/dev/null | tail -n 1)

if [ -z "$LATEST_STEP" ]; then
    echo "❌ 错误：在 $CKPT_DIR 下没找到 global_step 文件夹！请检查 EXPERIMENT_NAME。"
    exit 1
fi

ACTOR_DIR="$LATEST_STEP/actor"

# 💡 使用全新的目录名称，彻底阻断对历史文件的依赖！
ACTOR_HF_DIR="$CKPT_DIR/fresh_hf_weights"
MERGED_DIR="$CKPT_DIR/fresh_merged_model"

echo "✅ 锁定最新训练成果: $ACTOR_DIR"
echo "📂 将解包至全新目录: $ACTOR_HF_DIR"
echo "📂 将缝合至全新目录: $MERGED_DIR"

# ==========================================
# 📦 步骤 2: 将 FSDP 转换为 Hugging Face 格式
# ==========================================
echo "📦 步骤 2: 正在提取语言模型 safetensors..."
# 如果之前的旧解包目录偶然重名，先清理掉，确保 100% 重新生成
rm -rf "$ACTOR_HF_DIR"

python3 scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$ACTOR_HF_DIR"

# ==========================================
# 🧠 步骤 3: 绝对安全的物理缝合 (注入视觉权重)
# ==========================================
echo "🛠️ 步骤 3: 完美缝合 (原厂视觉底座 + 强化学习大脑)..."

cat << 'EOF' > merge_final_absolute.py
import os, sys, json, shutil

base_path = sys.argv[1]
actor_path = sys.argv[2]
out_path = sys.argv[3]

print(f"   -> 准备重置并创建全新输出目录: {out_path}")
if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)

print("   -> [1/4] 100% 复制原厂底座作为地基 (确保多模态架构和分词器完整无损)...")
for item in os.listdir(base_path):
    src = os.path.join(base_path, item)
    dst = os.path.join(out_path, item)
    if os.path.isfile(src):
        shutil.copy(src, dst)

print("   -> [2/4] 读取权重索引...")
with open(os.path.join(actor_path, "model.safetensors.index.json"), "r") as f:
    actor_index = json.load(f)
with open(os.path.join(out_path, "model.safetensors.index.json"), "r") as f:
    base_index = json.load(f)

print("   -> [3/4] 物理注入 Actor 权重碎片...")
for param, st_file in actor_index["weight_map"].items():
    new_st_file = "actor_" + st_file
    src_st = os.path.join(actor_path, st_file)
    dst_st = os.path.join(out_path, new_st_file)
    
    if not os.path.exists(dst_st):
        shutil.copy(src_st, dst_st)
        
    base_index["weight_map"][param] = new_st_file

print("   -> [4/4] 固化新版索引表...")
with open(os.path.join(out_path, "model.safetensors.index.json"), "w") as f:
    json.dump(base_index, f, indent=2)

print("✨ 缝合彻底完成！现在这是一个具备完整视觉能力的高阶模型。")
EOF

python3 merge_final_absolute.py "$BASE_MODEL" "$ACTOR_HF_DIR" "$MERGED_DIR"

# ==========================================
# 🚀 步骤 4: 执行 Benchmark 离线评测
# ==========================================
echo "🚀 步骤 4: 启动 SGLang 推理引擎，开始评测..."

# 将全新的合并目录通过环境变量传给 eval_checkpoints.py
export EVAL_CKPT_PATH="$MERGED_DIR"

python3 eval_checkpoints.py | tee "$LOG_FILE"

echo "=========================================="
echo "🎉 全新流水线测试大功告成！"
echo "📊 核心准确率摘要如下："
grep "Result:" "$LOG_FILE" || echo "未找到 Result 关键词，请查看完整日志。"
echo "=========================================="