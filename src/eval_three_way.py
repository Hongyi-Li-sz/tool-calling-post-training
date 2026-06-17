"""
三方对比评测：Base vs SFT vs DPO

用法:
  python src/eval_three_way.py
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.infer import SYSTEM_PROMPT

TEST_QUERIES = [
    ("查订单", "帮我查一下订单 A10293 的状态", "query_order"),
    ("计算", "计算 123 * 456 等于多少", "calculator"),
    ("搜索", "帮我搜索一下大模型后训练的资料", "search_docs"),
    ("拒调用", "你是谁？", "none"),
    ("预约", "帮我预约明天下午3点的项目评审会议", "book_meeting"),
    ("发邮件", "给 admin@company.com 发一封关于系统升级的通知邮件", "send_email"),
    ("缺失参数", "帮我预约个会议", "none+clarify"),
    ("复合意图", "搜一下 Python 教程，然后帮我发邮件给经理", "search_docs"),
    ("语义干扰", "今天是几号", "none"),
]


def load_base(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_sft(base_path: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def load_dpo(base_path: str, sft_adapter: str, dpo_adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True,
    )
    # 先挂 SFT adapter
    model = PeftModel.from_pretrained(model, sft_adapter)
    model = model.merge_and_unload()
    # 再挂 DPO adapter
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_adapter", default="outputs/sft/adapter")
    parser.add_argument("--dpo_adapter", default="outputs/dpo/adapter")
    parser.add_argument("--output", default="reports/three_way_comparison.json")
    args = parser.parse_args()

    models = {}
    print("加载 Base Model...")
    models["base"] = load_base(args.base_model)

    print("加载 SFT Model...")
    models["sft"] = load_sft(args.base_model, args.sft_adapter)

    print("加载 DPO Model...")
    models["dpo"] = load_dpo(args.base_model, args.sft_adapter, args.dpo_adapter)

    results = {"base": [], "sft": [], "dpo": []}

    for name, (model, tokenizer) in models.items():
        print(f"\n{'=' * 60}")
        print(f"评测: {name.upper()}")
        print("=" * 60)
        for label, query, expected in TEST_QUERIES:
            response = infer(model, tokenizer, query)
            try:
                parsed = json.loads(response)
                valid = True
                tool = parsed.get("tool", "?")
                clarify = parsed.get("need_clarification", False)
            except Exception:
                valid = False
                tool = "?"
                clarify = False
                parsed = None

            # 简单判分
            if valid:
                if expected == "none" and tool == "none":
                    score = "✅"
                elif expected == "none+clarify" and tool == "none" and clarify:
                    score = "✅"
                elif expected not in ("none", "none+clarify") and tool == expected:
                    score = "✅"
                elif tool == expected:
                    score = "✅"
                else:
                    score = "⚠️"
            else:
                # 对"拒调用"用例，如果输出不以 JSON 开头，也算一种"拒调用"
                if expected == "none" and not response.startswith("{"):
                    score = "~"  # 不算严格正确，但方向对
                else:
                    score = "❌"

            print(f"  [{label}] {score} tool={tool}" +
                  (f" clarify={clarify}" if clarify else "") +
                  (f" | {response[:60]}..." if len(response) > 60 else f" | {response}"))

            results[name].append({
                "label": label, "query": query, "expected": expected,
                "response": response, "json_valid": valid,
                "tool": tool, "need_clarification": clarify, "score": score,
            })

    # ── 汇总表 ──
    print(f"\n{'=' * 70}")
    print("三方对比汇总")
    print("=" * 70)
    header = f"{'用例':<12} {'预期':<15} {'Base':<10} {'SFT':<10} {'DPO':<10}"
    print(header)
    print("-" * 70)
    for i, (label, query, expected) in enumerate(TEST_QUERIES):
        b = results["base"][i]
        s = results["sft"][i]
        d = results["dpo"][i]
        print(f"{label:<12} {expected:<15} {b['score']:<10} {s['score']:<10} {d['score']:<10}")

    # 统计
    def count_score(model_results, target):
        return sum(1 for r in model_results if r["score"] == target)

    print(f"\n{'指标':<20} {'Base':<12} {'SFT':<12} {'DPO':<12}")
    print("-" * 56)
    for metric_name, score_char in [("✅ 完全正确", "✅"), ("⚠️ 格式对但工具错", "⚠️"),
                                      ("~ 自然语言拒调用", "~"), ("❌ 错误", "❌")]:
        b_c = count_score(results["base"], score_char)
        s_c = count_score(results["sft"], score_char)
        d_c = count_score(results["dpo"], score_char)
        print(f"{metric_name:<20} {b_c:<12} {s_c:<12} {d_c:<12}")

    # JSON 合法率
    for name in ["base", "sft", "dpo"]:
        valid = sum(1 for r in results[name] if r["json_valid"])
        total = len(results[name])
        print(f"{'JSON 合法率 ' + name:<20} {valid}/{total} ({valid/total*100:.0f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
