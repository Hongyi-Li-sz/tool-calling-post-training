"""
自动评测脚本
评测指标: JSON合法率、工具准确率、参数准确率、拒调用准确率
错误分类: JSON格式错、工具选错、参数漏抽、参数值错、多余解释、该拒未拒、该追问未追问
输出: bad_cases.json + 评测汇总报告
"""
import argparse
import json
import os
import re
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.infer import SYSTEM_PROMPT


def load_base(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def load_trained(base_path: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def load_dpo_full(base_path: str, sft_adapter: str, dpo_adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, sft_adapter)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, dpo_adapter)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def infer(model, tokenizer, query: str, max_new_tokens=256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 temperature=0.1, do_sample=True, top_p=0.9,
                                 pad_token_id=tokenizer.eos_token_id)
    input_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def parse_response(response: str) -> tuple[bool, Optional[dict], str]:
    """解析模型输出，返回 (是否JSON, 解析结果, 错误信息)"""
    # 尝试直接解析
    try:
        parsed = json.loads(response)
        return True, parsed, ""
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块（支持嵌套）
    try:
        # 找第一个 { 到最后一个 }
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end > start:
            candidate = response[start:end + 1]
            parsed = json.loads(candidate)
            return True, parsed, "has_extra_text"
    except json.JSONDecodeError:
        pass

    return False, None, "json_parse_failed"


def classify_error(parsed: Optional[dict], expected: dict, response: str
                   ) -> tuple[str, str]:
    """
    错误分类
    返回 (error_type, error_detail)
    """
    # 0. JSON 解析失败
    if parsed is None:
        return "json_parse_error", "输出无法解析为 JSON"

    # 1. 检查是否有多余文本
    stripped = response.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        # 可能有额外文字，但 JSON 被提取了
        pass

    # 2. 工具检查
    actual_tool = parsed.get("tool", "__missing__")
    expected_tool = expected.get("expected_tool", "__missing__")

    # 该拒绝（expected=none）但调了工具
    if expected_tool == "none" and actual_tool != "none":
        clarify = expected.get("expected_clarify", False) or expected.get("need_clarification", False)
        if clarify:
            return "should_clarify", f"应追问但调用了 {actual_tool}"
        else:
            return "should_reject", f"应输出 none 但调用了 {actual_tool}"

    # 该调工具但拒绝了
    if expected_tool != "none" and actual_tool == "none":
        if parsed.get("need_clarification"):
            return "unnecessary_clarify", f"应调用 {expected_tool} 但输出了追问"
        return "should_call_tool", f"应调用 {expected_tool} 但输出了 none"

    # 工具选错
    if expected_tool != "none" and actual_tool != "none" and actual_tool != expected_tool:
        return "wrong_tool", f"期望 {expected_tool}，实际 {actual_tool}"

    # 3. 参数检查（仅当工具正确时）
    if expected_tool != "none" and actual_tool == expected_tool:
        expected_args = expected.get("expected_arguments", {})
        actual_args = parsed.get("arguments", {})

        # arguments 可能不是 dict（模型偶尔输出错误类型）
        if not isinstance(actual_args, dict):
            return "json_parse_error", f"arguments 不是对象: {type(actual_args).__name__}"

        # 检查参数名
        expected_keys = set(expected_args.keys())
        actual_keys = set(actual_args.keys())

        # __any__ 表示不检查值
        strict_keys = {k for k in expected_keys if expected_args.get(k) != "__any__"}

        missing = strict_keys - actual_keys
        extra = actual_keys - expected_keys - {"__any__"}

        if missing:
            return "missing_param", f"缺少参数: {missing}"

        if extra and strict_keys:
            return "extra_param", f"多余参数: {extra}"

        # 检查参数值
        for k in strict_keys:
            if k in actual_args and expected_args[k] != "__any__":
                if str(actual_args[k]) != str(expected_args[k]):
                    return "wrong_param_value", f"参数 {k}: 期望 '{expected_args[k]}'，实际 '{actual_args[k]}'"

    # 4. 对 none 的额外交互检查
    if expected_tool == "none" and actual_tool == "none":
        clarify = expected.get("expected_clarify", False) or expected.get("need_clarification", False)
        actual_clarify = parsed.get("need_clarification", False)

        if clarify and not actual_clarify:
            return "missing_clarification", "需要追问但未设置 need_clarification"
        if not clarify and actual_clarify:
            return "unnecessary_clarify", "不需要追问但输出了 need_clarification"

    return "correct", ""


def evaluate_model(model, tokenizer, test_set: list[dict], model_name: str) -> list[dict]:
    """评测单个模型"""
    results = []
    for i, case in enumerate(test_set):
        instruction = case["instruction"]
        expected = {
            "expected_tool": case["expected_tool"],
            "expected_arguments": case.get("expected_arguments", {}),
            "expected_clarify": case.get("expected_clarify", False) or case.get("need_clarification", False),
        }

        response = infer(model, tokenizer, instruction)
        is_json, parsed, parse_msg = parse_response(response)
        error_type, error_detail = classify_error(parsed, expected, response)

        result = {
            "id": i,
            "instruction": instruction,
            "category": case.get("category", "unknown"),
            "response": response,
            "json_valid": is_json,
            "parsed": parsed,
            "expected_tool": expected["expected_tool"],
            "expected_arguments": expected["expected_arguments"],
            "expected_clarify": expected.get("expected_clarify", False) or expected.get("need_clarification", False),
            "actual_tool": parsed.get("tool") if parsed else None,
            "actual_arguments": parsed.get("arguments") if parsed else None,
            "error_type": error_type,
            "error_detail": error_detail,
            "is_correct": (error_type == "correct"),
        }
        results.append(result)

    # 统计
    total = len(results)
    json_valid = sum(1 for r in results if r["json_valid"])
    tool_correct = sum(1 for r in results
                       if r["json_valid"] and r["actual_tool"] == r["expected_tool"])
    param_correct = sum(1 for r in results if r["is_correct"]
                        and r["expected_tool"] != "none")
    reject_correct = sum(1 for r in results if r["is_correct"]
                         and r["expected_tool"] == "none"
                         and not r.get("expected_clarify"))
    clarify_correct = sum(1 for r in results if r["is_correct"]
                          and r.get("expected_clarify"))

    stats = {
        "model": model_name,
        "total": total,
        "json_valid": json_valid,
        "json_valid_rate": json_valid / total * 100,
        "tool_correct": tool_correct,
        "tool_accuracy": tool_correct / max(1, sum(1 for r in results if r["expected_tool"] != "none")) * 100,
        "param_correct": param_correct,
        "reject_correct": reject_correct,
        "clarify_correct": clarify_correct,
        "fully_correct": sum(1 for r in results if r["is_correct"]),
        "fully_correct_rate": sum(1 for r in results if r["is_correct"]) / total * 100,
    }

    return results, stats


def print_summary(all_stats: list[dict]):
    """打印评测汇总表"""
    print(f"\n{'=' * 80}")
    print("评测汇总")
    print("=" * 80)
    headers = ["指标", "Base", "SFT", "DPO"]
    print(f"{headers[0]:<22} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12}")
    print("-" * 58)

    metrics = [
        ("JSON 合法率", "json_valid_rate", "%"),
        ("工具选择准确率", "tool_accuracy", "%"),
        ("完全正确率", "fully_correct_rate", "%"),
    ]
    for label, key, unit in metrics:
        vals = [f"{s[key]:.1f}{unit}" if unit == "%" else str(s[key]) for s in all_stats]
        print(f"{label:<22} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

    print(f"\n{'指标':<22} {'Base':<12} {'SFT':<12} {'DPO':<12}")
    print("-" * 58)
    for label, key in [("JSON 合法", "json_valid"), ("工具正确", "tool_correct"),
                       ("完全正确", "fully_correct")]:
        vals = [f"{s[key]}/{s['total']}" for s in all_stats]
        print(f"{label:<22} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")


def print_error_analysis(results: list[dict], model_name: str):
    """打印错误分析"""
    errors = [r for r in results if not r["is_correct"]]
    print(f"\n{'=' * 60}")
    print(f"{model_name} Bad Cases ({len(errors)} 条)")
    print("=" * 60)

    error_counts = {}
    for r in errors:
        et = r["error_type"]
        error_counts[et] = error_counts.get(et, 0) + 1

    ERROR_CN = {
        "json_parse_error": "JSON 格式错误",
        "should_reject": "该拒未拒",
        "should_call_tool": "该调用却输出none",
        "should_clarify": "该追问却调工具",
        "wrong_tool": "工具选错",
        "missing_param": "参数缺失",
        "extra_param": "多余参数",
        "wrong_param_value": "参数值错误",
        "missing_clarification": "未设置追问标记",
        "unnecessary_clarify": "不必要的追问",
    }

    for et, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  [{ERROR_CN.get(et, et)}] x{count}")

    # 展示前5条错误详情
    print(f"\n  前 5 条错误详情:")
    for i, r in enumerate(errors[:5]):
        print(f"\n  [{i+1}] {r['category']}")
        print(f"      输入: {r['instruction']}")
        print(f"      输出: {r['response'][:100]}")
        print(f"      类型: {ERROR_CN.get(r['error_type'], r['error_type'])}")
        print(f"      详情: {r['error_detail']}")


def main():
    parser = argparse.ArgumentParser(description="自动评测")
    parser.add_argument("--base_model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_adapter", default="outputs/sft/adapter")
    parser.add_argument("--dpo_adapter", default="outputs/dpo/adapter")
    parser.add_argument("--test_set", default="data/eval/test_set.json")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--models", default="base,sft,dpo",
                        help="要评测的模型，逗号分隔")
    args = parser.parse_args()

    # 加载测试集
    with open(args.test_set, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"测试集: {len(test_set)} 条")

    models_to_eval = args.models.split(",")
    all_results = {}
    all_stats = []

    # 逐个模型评测
    if "base" in models_to_eval:
        print("\n加载 Base Model...")
        m, t = load_base(args.base_model)
        results, stats = evaluate_model(m, t, test_set, "base")
        all_results["base"] = results
        all_stats.append(stats)
        print_error_analysis(results, "Base")

    if "sft" in models_to_eval:
        print("\n加载 SFT Model...")
        m, t = load_trained(args.base_model, args.sft_adapter)
        results, stats = evaluate_model(m, t, test_set, "sft")
        all_results["sft"] = results
        all_stats.append(stats)
        print_error_analysis(results, "SFT")

    if "dpo" in models_to_eval:
        print("\n加载 DPO Model...")
        m, t = load_dpo_full(args.base_model, args.sft_adapter, args.dpo_adapter)
        results, stats = evaluate_model(m, t, test_set, "dpo")
        all_results["dpo"] = results
        all_stats.append(stats)
        print_error_analysis(results, "DPO")

    # 打印汇总
    if all_stats:
        print_summary(all_stats)

    # 保存 bad cases
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name, results in all_results.items():
        bad_cases = [r for r in results if not r["is_correct"]]
        path = os.path.join(args.output_dir, f"bad_cases_{model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=2)
        print(f"\n{model_name} Bad Cases 已保存: {path} ({len(bad_cases)} 条)")

    # 保存完整评测结果
    full_path = os.path.join(args.output_dir, "evaluation_full.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "stats": {s["model"]: s for s in all_stats}},
                  f, ensure_ascii=False, indent=2)
    print(f"完整评测结果已保存: {full_path}")


if __name__ == "__main__":
    main()
