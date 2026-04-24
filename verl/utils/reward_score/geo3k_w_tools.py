import os
import re
import json

from mathruler.grader import extract_boxed_content, grade_answer

# def base_format_reward(solution_str: str) -> float:
#     pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
#     match_result = re.fullmatch(pattern, solution_str)
#     return 1.0 if match_result else 0.0
def base_format_reward(solution_str: str) -> float:
    # 改为 search，只要文本中包含了思维链和答案框，就给分
    pattern = re.compile(r"<think>.*?</think>.*?\\boxed\{.*?\}", re.DOTALL)
    match_result = re.search(pattern, solution_str)
    return 1.0 if match_result else 0.0

def acc_reward(solution_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(solution_str)
    else:
        answer = solution_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def tool_reward_and_penalty(solution_str: str):
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    tool_calls = pattern.findall(solution_str)

    num_calls = len(tool_calls)
    valid_format_count = 0

    for call_str in tool_calls:
        try:
            parsed = json.loads(call_str.strip())

            if isinstance(parsed, dict):
                valid_format_count += 1

        except json.JSONDecodeError:
            pass

    if num_calls > 0:
        tool_format_score = float(valid_format_count) / float(num_calls)
    else:
        tool_format_score = 0.0

    if num_calls <= 2:
        penalty = 0.0
    else:
        penalty = float(num_calls - 2) / float(num_calls)

    return tool_format_score, penalty, num_calls, valid_format_count


def compute_score(solution_str: str, ground_truth: str, **kwargs):
    with open("my_absolute_debug_log.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "!"*50 + "\n")
        f.write(f"MODEL WROTE THIS CODE:\n{solution_str}\n")
        f.write("!"*50 + "\n")
    w_acc = float(os.environ.get("REWARD_W_ACC", "1.0"))
    w_base_fmt = float(os.environ.get("REWARD_W_BASE_FMT", "0.1"))
    w_tool_fmt = float(os.environ.get("REWARD_W_TOOL_FMT", "100.0"))
    w_penalty = float(os.environ.get("REWARD_W_PENALTY", "0.5"))

    r_acc = acc_reward(solution_str, ground_truth)
    r_base_fmt = base_format_reward(solution_str)
    r_tool_fmt, r_penalty, num_calls, valid_format_count = tool_reward_and_penalty(solution_str)

    final_score = (w_acc * r_acc) + \
                (w_base_fmt * r_base_fmt) + \
                (w_tool_fmt * r_tool_fmt) - \
                (w_penalty * r_penalty)

    # extra_metrics = {
    #     "w_acc": w_acc,
    #     "w_base_fmt": w_base_fmt,
    #     "w_tool_fmt": w_tool_fmt,
    #     "w_penalty": w_penalty,

    #     "r_acc": r_acc,
    #     "r_base_fmt": r_base_fmt,
    #     "r_tool_fmt": r_tool_fmt,
    #     "r_penalty": r_penalty,

    #     "num_tool_calls": num_calls,
    #     "valid_tool_calls": valid_format_count
    # }

    return {
        "score": float(final_score), 
        "w_acc": float(w_acc),
        "w_base_fmt": float(w_base_fmt),
        "w_tool_fmt": float(w_tool_fmt),
        "w_penalty": float(w_penalty),
        "r_acc": float(r_acc),
        "r_base_fmt": float(r_base_fmt),
        "r_tool_fmt": float(r_tool_fmt),
        "r_penalty": float(r_penalty),
        "num_tool_calls": float(num_calls),
        "valid_tool_calls": float(valid_format_count),
        "model_response": solution_str
    }