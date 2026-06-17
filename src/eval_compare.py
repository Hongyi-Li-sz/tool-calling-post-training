"""
对比评测脚本：base model vs SFT model（通过 adapter 加载）

用法:
  python src/eval_compare.py --base_model models/Qwen2.5-0.5B-Instruct --adapter outputs/sft/adapter
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.infer import SYSTEM_PROMPT, infer, try_parse_json

TEST_QUERIES = [
    "帮我查一下订单 A10293 的状态",
    "计算 123 * 456 等于多少",
    "帮我搜索一下大模型后训练的资料",
    "你是谁？",
    "帮我预约明天下午3点的项目评审会议",
    "给 admin@company.com 发一封关于系统升级的通知邮件",
    "帮我预约个会议",
    "搜一下 Python 教程，然后帮我发邮件给经理",
    "今天是几号",
]


def load_base_model(model_path: str):
    """加载 base model"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_sft_model(base_path: str, adapter_path: str):
    """加载 base model + adapter"""
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # 合并用于更快的推理
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Base vs SFT 对比评测")
    parser.add_argument("--base_model", type=str, default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/sft/adapter")
    parser.add_argument("--output", type=str, default="reports/sft_vs_base.json")
    args = parser.parse_args()

    models = {}

    # 加载 Base Model
    print("=" * 60)
    print("加载 Base Model...")
    print("=" * 60)
    models["base"] = load_base_model(args.base_model)

    # 加载 SFT Model
    print("\n" + "=" * 60)
    print("加载 SFT Model (base + adapter)...")
    print("=" * 60)
    models["sft"] = load_sft_model(args.base_model, args.adapter)

    # 对比测试
    results = {"base": [], "sft": []}

    for model_name, (model, tokenizer) in models.items():
        print(f"\n{'=' * 60}")
        print(f"评测: {model_name.upper()}")
        print("=" * 60)

        for i, query in enumerate(TEST_QUERIES):
            response = infer(model, tokenizer, query, max_new_tokens=256)
            parsed = try_parse_json(response)
            valid = parsed is not None

            print(f"\n[{i + 1}/{len(TEST_QUERIES)}] {query}")
            print(f"  输出: {response[:150]}")

            if valid:
                tool = parsed.get("tool", "?")
                need_clarify = parsed.get("need_clarification", False)
                print(f"  ✅ JSON 合法 | tool={tool}" +
                      (f" | 追问: {parsed.get('question', '')}" if need_clarify else ""))
            else:
                print(f"  ❌ JSON 非法")

            results[model_name].append({
                "query": query,
                "response": response,
                "json_valid": valid,
                "parsed": parsed,
            })

    # ── 汇总对比 ──
    print(f"\n{'=' * 60}")
    print("对比汇总")
    print("=" * 60)
    print(f"{'指标':<20} {'Base':<15} {'SFT':<15}")
    print("-" * 50)

    for model_name in ["base", "sft"]:
        r = results[model_name]
        valid = sum(1 for x in r if x["json_valid"])
        print(f"{'JSON 合法率':<20} {valid}/{len(r)} ({valid / len(r) * 100:.1f}%)")

    print()

    # 逐条对比
    for i, query in enumerate(TEST_QUERIES):
        base_valid = results["base"][i]["json_valid"]
        sft_valid = results["sft"][i]["json_valid"]

        base_tool = results["base"][i]["parsed"].get("tool", "?") if results["base"][i]["parsed"] else "?"
        sft_tool = results["sft"][i]["parsed"].get("tool", "?") if results["sft"][i]["parsed"] else "?"

        base_status = "✅" if base_valid else "❌"
        sft_status = "✅" if sft_valid else "❌"

        print(f"[{i + 1}] {query[:50]}...")
        print(f"    Base: {base_status} tool={base_tool}")
        print(f"    SFT:  {sft_status} tool={sft_tool}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
