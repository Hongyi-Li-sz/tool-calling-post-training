"""
V2 升级版评测脚本
7 项指标 + 分场景统计 + CSV 导出

用法:
  python src/evaluate_v2.py --test_set data/eval/test_set_v2.json
"""
import argparse
import csv
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.infer import SYSTEM_PROMPT
# 复用 V1 的模型加载和推理逻辑
from src.evaluate import (
    load_base, load_trained, load_dpo_full, infer,
    parse_response, classify_error
)


CATEGORY_LABELS = {
    "single_tool": "单工具调用",
    "no_tool": "无需工具",
    "clarification": "参数缺失追问",
    "semantic_interference": "语义干扰",
    "compound": "复合请求",
    "colloquial": "口语化表达",
}


def compute_metrics(results: list[dict], category: str = None) -> dict:
    """
    计算 7 项评测指标
    如果指定 category，仅统计该类别
    """
    filtered = results
    if category:
        filtered = [r for r in results if r.get("category") == category]

    total = len(filtered)
    if total == 0:
        return {"total": 0}

    # 1. JSON 合法率
    json_valid = sum(1 for r in filtered if r["json_valid"])

    # 2. 工具选择准确率 (tool 字段匹配 expected_tool)
    tool_correct = sum(1 for r in filtered
                       if r["json_valid"] and r["actual_tool"] == r["expected_tool"])

    # 3. 参数精确匹配 (arguments 完全匹配)
    arg_match = sum(1 for r in filtered
                    if r["is_correct"]
                    and r["expected_tool"] != "none"
                    and r["actual_tool"] == r["expected_tool"])

    # 4. 澄清追问准确率 (需要追问时是否正确追问)
    need_clarify_cases = [r for r in filtered
                          if r["expected_clarify"]]
    clarify_total = len(need_clarify_cases)
    clarify_correct = sum(1 for r in need_clarify_cases if r["is_correct"])

    # 5. 过度追问率 (不需要追问时错误追问)
    no_clarify_cases = [r for r in filtered
                        if not r["expected_clarify"]
                        and r["json_valid"]
                        and r["actual_tool"] == "none"]
    over_clarify = sum(1 for r in no_clarify_cases
                       if r.get("parsed") and r["parsed"].get("need_clarification"))

    # 6. 不必要工具调用率 (不需要工具时错误调用了工具)
    no_tool_cases = [r for r in filtered
                     if r["expected_tool"] == "none"
                     and r["json_valid"]]
    unnecessary_tool = sum(1 for r in no_tool_cases
                           if r["actual_tool"] != "none")

    # 7. 完全正确率
    fully_correct = sum(1 for r in filtered if r["is_correct"])

    # 错误分类统计
    error_counts = {}
    for r in filtered:
        if not r["is_correct"]:
            et = r.get("error_type", "unknown")
            error_counts[et] = error_counts.get(et, 0) + 1

    return {
        "total": total,
        "json_valid": json_valid,
        "json_valid_rate": round(json_valid / total * 100, 1),
        "tool_correct": tool_correct,
        "tool_accuracy": round(tool_correct / total * 100, 1),
        "argument_exact_match": arg_match,
        "argument_exact_match_rate": round(arg_match / max(1, total) * 100, 1),
        "clarification_accuracy": round(clarify_correct / max(1, clarify_total) * 100, 1),
        "clarify_total": clarify_total,
        "clarify_correct": clarify_correct,
        "over_clarification_rate": round(over_clarify / max(1, len(no_clarify_cases)) * 100, 1),
        "over_clarify_count": over_clarify,
        "unnecessary_tool_call_rate": round(unnecessary_tool / max(1, len(no_tool_cases)) * 100, 1),
        "unnecessary_tool_call_count": unnecessary_tool,
        "fully_correct": fully_correct,
        "fully_correct_rate": round(fully_correct / total * 100, 1),
        "bad_cases": total - fully_correct,
        "error_counts": error_counts,
    }


def print_full_report(all_stats: list[dict], all_results: dict):
    """打印完整报告：总体 + 分场景"""

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
        "correct": "正确",
    }

    model_names = [s["model"] for s in all_stats]

    # ═══ 总体指标 ═══
    print(f"\n{'=' * 90}")
    print("V2 评测报告 — 总体指标（7 项）")
    print("=" * 90)
    header = f"{'指标':<24}" + "".join(f"{m:<14}" for m in model_names)
    print(header)
    print("-" * 90)

    metric_rows = [
        ("1. JSON 合法率", "json_valid_rate", "%"),
        ("2. 工具选择准确率", "tool_accuracy", "%"),
        ("3. 参数完全匹配率", "argument_exact_match_rate", "%"),
        ("4. 澄清追问准确率", "clarification_accuracy", "%"),
        ("5. 过度追问率 ⚠", "over_clarification_rate", "%"),
        ("6. 不必要工具调用率 ⚠", "unnecessary_tool_call_rate", "%"),
        ("7. 完全正确率", "fully_correct_rate", "%"),
    ]
    for label, key, unit in metric_rows:
        vals = [f"{s[key]}{unit}" for s in all_stats]
        print(f"{label:<24}" + "".join(f"{v:<14}" for v in vals))

    # 计数行
    print(f"\n{'─' * 90}")
    print(f"{'明细计数':<24}" + "".join(f"{m:<14}" for m in model_names))
    print("-" * 90)
    for label, key in [("JSON 合法", "json_valid"), ("工具正确", "tool_correct"),
                       ("参数匹配", "argument_exact_match"),
                       ("完全正确", "fully_correct"), ("总样本", "total")]:
        vals = [f"{s[key]}/{s['total']}" if key in s and key != 'total' else str(s['total'])
                for s in all_stats]
        print(f"{label:<24}" + "".join(f"{v:<14}" for v in vals))

    # ═══ 错误类型对比 ═══
    print(f"\n{'=' * 90}")
    print("错误类型对比")
    print("=" * 90)
    all_error_types = set()
    for s in all_stats:
        all_error_types.update(s.get("error_counts", {}).keys())
    all_error_types = sorted(all_error_types,
                             key=lambda et: -sum(s.get("error_counts", {}).get(et, 0) for s in all_stats))

    header = f"{'错误类型':<24}" + "".join(f"{m:<14}" for m in model_names)
    print(header)
    print("-" * 90)
    for et in all_error_types:
        if et == "correct":
            continue
        vals = [str(s.get("error_counts", {}).get(et, 0)) for s in all_stats]
        print(f"{ERROR_CN.get(et, et):<24}" + "".join(f"{v:<14}" for v in vals))

    # ═══ 分场景指标 ═══
    print(f"\n{'=' * 90}")
    print("分场景完全正确率")
    print("=" * 90)
    header = f"{'场景':<20}" + "".join(f"{m:<14}" for m in model_names)
    print(header)
    print("-" * 90)

    for cat_key, cat_label in CATEGORY_LABELS.items():
        vals = []
        for name in model_names:
            cat_stats = compute_metrics(all_results[name], cat_key)
            if cat_stats["total"] > 0:
                vals.append(f"{cat_stats['fully_correct_rate']}% ({cat_stats['fully_correct']}/{cat_stats['total']})")
            else:
                vals.append("N/A")
        print(f"{cat_label:<20}" + "".join(f"{v:<14}" for v in vals))


def export_csv(all_stats: list[dict], all_results: dict, output_path: str):
    """导出分场景结果为 CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_names = [s["model"] for s in all_stats]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "model", "total", "json_valid_rate",
                         "tool_accuracy", "argument_exact_match_rate",
                         "clarification_accuracy", "over_clarification_rate",
                         "unnecessary_tool_call_rate", "fully_correct_rate",
                         "fully_correct", "bad_cases"])

        for cat_key, cat_label in CATEGORY_LABELS.items():
            for name in model_names:
                s = compute_metrics(all_results[name], cat_key)
                if s["total"] > 0:
                    writer.writerow([
                        cat_label, name, s["total"],
                        s["json_valid_rate"], s["tool_accuracy"],
                        s["argument_exact_match_rate"],
                        s["clarification_accuracy"], s["over_clarification_rate"],
                        s["unnecessary_tool_call_rate"], s["fully_correct_rate"],
                        s["fully_correct"], s["bad_cases"],
                    ])
    print(f"\n📊 分场景 CSV 已导出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="V2 升级版评测")
    parser.add_argument("--base_model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_adapter", default="outputs/sft/adapter")
    parser.add_argument("--dpo_adapter", default="outputs/dpo/adapter")
    parser.add_argument("--test_set", default="data/eval/test_set_v2.json")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--models", default="base,sft,dpo")
    args = parser.parse_args()

    with open(args.test_set, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"测试集: {len(test_set)} 条")

    models_to_eval = args.models.split(",")
    all_results = {}
    all_stats = []

    if "base" in models_to_eval:
        print("加载 Base Model...")
        m, t = load_base(args.base_model)
        results, _ = run_eval(m, t, test_set, "base")
        stats = compute_metrics(results)
        stats["model"] = "base"
        all_results["base"] = results
        all_stats.append(stats)

    if "sft" in models_to_eval:
        print("加载 SFT Model...")
        m, t = load_trained(args.base_model, args.sft_adapter)
        results, _ = run_eval(m, t, test_set, "sft")
        stats = compute_metrics(results)
        stats["model"] = "sft"
        all_results["sft"] = results
        all_stats.append(stats)

    if "dpo" in models_to_eval:
        print("加载 DPO Model...")
        m, t = load_dpo_full(args.base_model, args.sft_adapter, args.dpo_adapter)
        results, _ = run_eval(m, t, test_set, "dpo")
        stats = compute_metrics(results)
        stats["model"] = "dpo"
        all_results["dpo"] = results
        all_stats.append(stats)

    if all_stats:
        print_full_report(all_stats, all_results)

    # 导出
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.output_dir, "v2_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"stats": [s for s in all_stats]}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 评测汇总: {summary_path}")

    csv_path = os.path.join(args.output_dir, "v2_eval_by_category.csv")
    export_csv(all_stats, all_results, csv_path)

    # Bad cases
    for name, results in all_results.items():
        bad = [r for r in results if not r["is_correct"]]
        bp = os.path.join(args.output_dir, f"v2_bad_cases_{name}.json")
        with open(bp, "w", encoding="utf-8") as f:
            json.dump(bad, f, ensure_ascii=False, indent=2)


def run_eval(model, tokenizer, test_set, model_name):
    """运行评测（与 V1 兼容）"""
    from src.evaluate import evaluate_model
    return evaluate_model(model, tokenizer, test_set, model_name)


if __name__ == "__main__":
    main()
