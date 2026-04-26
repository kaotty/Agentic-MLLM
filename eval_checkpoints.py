import os
import gc
import torch
import datasets
import base64
from io import BytesIO
from datasets import load_dataset
from transformers import AutoTokenizer
import sglang as sgl
from mathruler.grader import extract_boxed_content, grade_answer

# ==========================================
# 1. Global Configurations
# ==========================================
CHECKPOINT_PATH = "/home/hyunin_sakana_ai/verl/checkpoints/qwen2_5_vl_7b_zoom_w0.5/merged_eval_ready"
TENSOR_PARALLEL_SIZE = 4

DATASET_CONFIGS = {
    "Geometry3K": {
        "path": "hiyouga/geometry3k", "split": "test", 
        "problem_col": "problem", "answer_col": "answer", "image_col": "images",
        "local_file": "~/data/eval_geo3k_zoom/test.parquet"
    },
    "MathVista": {
        "path": "AI4Math/MathVista", "split": "testmini", 
        "problem_col": "query", "answer_col": "answer", "image_col": "decoded_image",
        "local_file": "~/data/eval_mathvista_zoom/test.parquet"
    },
    "ChartQA": {
        "path": "HuggingFaceM4/ChartQA", "split": "test", 
        "problem_col": "query", "answer_col": "label", "image_col": "image",
        "local_file": "~/data/eval_chartqa_zoom/test.parquet"
    },
    "MathVerse": {
        "path": "AI4Math/MathVerse", "split": "testmini", 
        "problem_col": "question", "answer_col": "answer", "image_col": "image",
        "local_file": "~/data/eval_mathverse_zoom/test_final.parquet"
    }
}

INSTRUCTION_FOLLOWING = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{}."
)

SYSTEM_PROMPT = (
    "You are a math expert. You are given a geometry question and an accompanying image. "
    "You must reason step by step. "
    "You MUST use the `image_zoom_in_tool` by providing a bounding box [x1, y1, x2, y2] in your response. "
    "To use the tool, you MUST output exactly in this format:\n"
    "<tool_call>\n"
    '{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]} }\n'
    "</tool_call>\n"
    "After you have gathered enough information from the zoomed regions, provide your final reasoning and answer."
)

# ==========================================
# 2. Helper Functions
# ==========================================
def pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def prepare_dataset_if_needed(name, cfg):
    local_path = os.path.expanduser(cfg["local_file"])
    if os.path.exists(local_path):
        print(f"✅ [{name}] Local cache found.")
        return local_path
        
    print(f"⬇️ [{name}] Downloading and processing...")
    raw_ds = load_dataset(cfg["path"], split=cfg["split"])
    
    def process_fn(example, idx):
        problem = example[cfg["problem_col"]]
        prompt = problem + "\n" + INSTRUCTION_FOLLOWING
        raw_answer = example[cfg["answer_col"]]
        answer = str(raw_answer[0]) if isinstance(raw_answer, list) else str(raw_answer)
        raw_image = example[cfg["image_col"]]
        images = raw_image if isinstance(raw_image, list) else ([raw_image] if raw_image else [])
        tool_image = images[0] if len(images) > 0 else None
        return {
            "data_source": cfg["path"],
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "images": images, "extra_info": {"answer": answer}
        }

    mapped_ds = raw_ds.map(process_fn, with_indices=True, num_proc=8)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    mapped_ds.to_parquet(local_path)
    return local_path

# ==========================================
# 3. Main Execution Block (The Fix!)
# ==========================================
if __name__ == '__main__':
    print(f"\n🚀 Loading SGLang engine: {CHECKPOINT_PATH} ...")
    engine = sgl.Engine(
        model_path=CHECKPOINT_PATH,
        tp_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True
    )
    
    print(f"📖 Loading tokenizer from {CHECKPOINT_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n" + "="*50)
        print(f"🔥 Evaluating: {ds_name}")
        print(f"="*50)
        
        parquet_path = prepare_dataset_if_needed(ds_name, cfg)
        ds = load_dataset("parquet", data_files=parquet_path, split="train")

        # 💡 这里是彻底治好 SGLang ValueError 的关键：拆分两个独立列表
        prompts_list = []
        image_data_list = []
        ground_truths = []

        for row in ds:
            pil_image = row["images"][0] if row["images"] else None
            
            # 使用完全契合你缓存数据的读取方式
            sys_prompt = row["prompt"][0]["content"]
            user_text = row["prompt"][1]["content"]

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": user_text}
                ]}
            ]
            
            prompt_str = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # 分别存入纯文本列表和图片列表
            prompts_list.append(prompt_str)
            image_data_list.append(pil_image)
            ground_truths.append(row["extra_info"]["answer"])

        print(f"⏳ Running SGLang inference on {len(prompts_list)} samples...")
        
        # 💡 SGLang 正确且唯一的批量输入姿势
        outputs = engine.generate(
            prompts_list, 
            sampling_params={"temperature": 0.0, "max_new_tokens": 2048},
            image_data=image_data_list
        )

        correct_count = 0
        for i, out in enumerate(outputs):
            response = out["text"]
            gt = ground_truths[i]
            if grade_answer(extract_boxed_content(response), gt):
                correct_count += 1

        acc = (correct_count / len(prompts_list)) * 100
        print(f"\n✅ {ds_name} Result: {acc:.2f}% ({correct_count}/{len(prompts_list)})")

    print("\n🎉 All datasets evaluated successfully!")