"""
构造 DPO 偏好数据

从 SFT 数据出发，将标准答案作为 chosen，程序化生成各种错误作为 rejected。
覆盖 7 种错误类型，目标 400-600 对 preference pairs。

DPO 数据格式：
{
  "prompt": "用户指令",
  "chosen": "标准/正确的 JSON 输出",
  "rejected": "错误的 JSON 输出"
}
"""
import argparse
import json
import os
import random
import re
import sys
from copy import deepcopy

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sft_data(path: str = None) -> list[dict]:
    """加载 SFT 训练数据"""
    path = path or os.path.join(PROJECT_ROOT, "data", "sft", "train.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════
#  Rejected 生成策略 — 每种模拟一个典型错误
# ══════════════════════════════════════════════════════════════════

ALL_TOOLS = ["search_docs", "calculator", "query_order", "book_meeting", "send_email"]

SAMPLE_ARGUMENTS = {
    "search_docs": {"query": "随机搜索词"},
    "calculator": {"expression": "1 + 1"},
    "query_order": {"order_id": "X99999"},
    "book_meeting": {"date": "2024-01-01", "time": "09:00", "topic": "随机会议"},
    "send_email": {
        "recipient": "random@example.com",
        "subject": "随机主题",
        "content": "随机内容",
    },
}


def get_tool(output: dict) -> str | None:
    return output.get("tool") if isinstance(output, dict) else None


def make_wrong_tool(chosen: dict) -> dict:
    """策略1：工具选错 — 替换为随机其他工具，保留原 arguments 结构"""
    current_tool = chosen.get("tool", "none")
    other_tools = [t for t in ALL_TOOLS if t != current_tool]
    if not other_tools:
        return None
    wrong_tool = random.choice(other_tools)
    # 尽力沿用原参数结构，但用目标工具的参数名
    target_sample = deepcopy(SAMPLE_ARGUMENTS[wrong_tool])
    return {"tool": wrong_tool, "arguments": target_sample}


def make_missing_param(chosen: dict) -> dict:
    """策略2：参数缺失 — 删除一个必填参数"""
    args = chosen.get("arguments", {})
    if not args or len(args) <= 1:
        return None
    result = deepcopy(chosen)
    key_to_remove = random.choice(list(result["arguments"].keys()))
    del result["arguments"][key_to_remove]
    return result


def make_wrong_param_name(chosen: dict) -> dict:
    """策略3：参数名错误 — 把参数名换成错的"""
    args = chosen.get("arguments", {})
    if not args:
        return None
    WRONG_NAMES = ["id", "name", "type", "value", "data", "input", "keyword", "text", "param", "arg"]
    result = deepcopy(chosen)
    key_to_replace = random.choice(list(result["arguments"].keys()))
    wrong_name = random.choice([n for n in WRONG_NAMES if n != key_to_replace])
    result["arguments"][wrong_name] = result["arguments"].pop(key_to_replace)
    return result


def make_bad_json(chosen: dict) -> str:
    """策略4：JSON 格式错误 — 去掉引号、多余逗号等（返回字符串）"""
    # 直接返回一个格式有问题的 JSON 字符串
    errors = [
        # 缺少引号的 key
        lambda d: json.dumps(d, ensure_ascii=False).replace('"tool"', 'tool').replace('"arguments"', 'arguments'),
        # 尾部多余逗号
        lambda d: json.dumps(d, ensure_ascii=False)[:-1] + ",}",
        # 单引号替代双引号
        lambda d: json.dumps(d, ensure_ascii=False).replace('"', "'"),
    ]
    error_func = random.choice(errors)
    return error_func(chosen)


def make_natural_lang(chosen: dict) -> str:
    """策略5：输出自然语言而非 JSON"""
    responses = [
        "好的，我来帮你处理。",
        "根据您的需求，我建议调用相应的工具。",
        "正在为您查询中，请稍候...",
        "好的，这就为您处理。",
        "收到，马上帮您操作。",
        "已了解您的需求，正在执行。",
        "让我来帮您完成这个任务。",
    ]
    return random.choice(responses)


def make_wrong_none(chosen: dict) -> dict:
    """策略6a：应该调工具却输出 none"""
    return {"tool": "none", "arguments": {}}


def make_wrong_tool_call(chosen: dict) -> dict:
    """策略6b：不该调工具却调了工具"""
    tool = random.choice(ALL_TOOLS)
    return {"tool": tool, "arguments": deepcopy(SAMPLE_ARGUMENTS[tool])}


def make_extra_text(chosen: dict) -> str:
    """策略7：JSON 前后有多余文字（返回字符串）"""
    prefixes = ["好的，这是查询结果：", "根据您的需求：", "以下是操作结果：", "好的，我来帮您："]
    suffixes = ["。以上就是全部信息。", "，操作完成。", "。如果还需要其他帮助请告诉我。", "。"]
    text = json.dumps(chosen, ensure_ascii=False)
    return random.choice(prefixes) + text + random.choice(suffixes)


def make_hallucinated_param(chosen: dict) -> dict:
    """额外策略：编造不存在的参数"""
    result = deepcopy(chosen)
    result["arguments"]["extra_field"] = "编造的值"
    return result


# ══════════════════════════════════════════════════════════════════
#  主生成逻辑
# ══════════════════════════════════════════════════════════════════

def generate_dpo_pairs(sft_data: list[dict]) -> list[dict]:
    """从 SFT 数据生成 DPO 偏好对"""

    # 注册错误生成策略（每个策略返回 (rejected_output, error_type_str)）
    strategies = []

    pairs = []

    for sample in sft_data:
        prompt = sample["instruction"]
        chosen_str = sample["output"]

        try:
            chosen_obj = json.loads(chosen_str)
        except json.JSONDecodeError:
            continue

        tool = chosen_obj.get("tool", "none")
        is_none = (tool == "none")
        has_clarify = chosen_obj.get("need_clarification", False)

        # 为每个样本生成 1-2 个 rejected 变体
        rejected_list = []

        # ── 选择适用的策略 ──
        available_strategies = []

        if not is_none:
            # 工具调用 → 工具相关错误
            available_strategies.extend([
                ("wrong_tool", lambda c=chosen_obj: make_wrong_tool(c)),
                ("missing_param", lambda c=chosen_obj: make_missing_param(c)),
                ("wrong_param_name", lambda c=chosen_obj: make_wrong_param_name(c)),
                ("hallucinated_param", lambda c=chosen_obj: make_hallucinated_param(c)),
                ("tool_to_none", lambda c=chosen_obj: make_wrong_none(c)),
            ])

        if is_none and not has_clarify:
            # 不需要工具 → 错调工具
            available_strategies.append(
                ("none_to_tool", lambda c=chosen_obj: make_wrong_tool_call(c))
            )

        if is_none and has_clarify:
            # 该追问但直接编造了工具调用
            for wrong_tool in random.sample(ALL_TOOLS, min(3, len(ALL_TOOLS))):
                available_strategies.append(
                    ("clarify_to_tool", lambda wt=wrong_tool: {
                        "tool": wt, "arguments": deepcopy(SAMPLE_ARGUMENTS[wt])
                    })
                )

        # 通用错误（适用于所有类型）
        available_strategies.extend([
            ("bad_json", lambda c=chosen_obj: make_bad_json(c)),
            ("natural_lang", lambda c=chosen_obj: make_natural_lang(c)),
            ("extra_text", lambda c=chosen_obj: make_extra_text(c)),
        ])

        # 随机选 1-2 个策略
        num_rejected = random.choices([1, 2], weights=[0.6, 0.4])[0]
        selected = random.sample(available_strategies, min(num_rejected, len(available_strategies)))

        for error_type, strategy_fn in selected:
            try:
                rejected_output = strategy_fn()
                if rejected_output is None:
                    continue
                # 统一转为字符串
                if isinstance(rejected_output, dict):
                    rejected_str = json.dumps(rejected_output, ensure_ascii=False)
                else:
                    rejected_str = str(rejected_output)

                # 跳过 chosen == rejected 的情况
                if rejected_str.strip() == chosen_str.strip():
                    continue

                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_str,
                    "rejected": rejected_str,
                    "error_type": error_type,
                })
            except Exception:
                continue

    # 打乱
    random.shuffle(pairs)
    return pairs


def validate_pairs(pairs: list[dict]) -> tuple[bool, list[str]]:
    """验证 DPO 数据"""
    errors = []
    for i, p in enumerate(pairs):
        for field in ["prompt", "chosen", "rejected"]:
            if field not in p:
                errors.append(f"#{i}: 缺少字段 {field}")
        if p.get("chosen") == p.get("rejected"):
            errors.append(f"#{i}: chosen == rejected")
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="构造 DPO 偏好数据")
    parser.add_argument("--sft_data", type=str, default=None,
                        help="SFT 数据路径（默认 data/sft/train.json）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认 data/dpo/train.json）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载 SFT 数据
    sft_path = args.sft_data or os.path.join(PROJECT_ROOT, "data", "sft", "train.json")
    print(f"加载 SFT 数据: {sft_path}")
    sft_data = load_sft_data(sft_path)
    print(f"  {len(sft_data)} 条 SFT 样本")

    # 生成 DPO 数据
    print("\n生成 DPO 偏好对...")
    pairs = generate_dpo_pairs(sft_data)
    print(f"  生成 {len(pairs)} 对 DPO 数据")

    # 验证
    ok, errs = validate_pairs(pairs)
    if not ok:
        print(f"\n❌ 验证失败: {len(errs)} 个错误")
        for e in errs[:10]:
            print(f"   {e}")
    else:
        print("✅ 验证通过")

    # 统计错误类型分布
    error_counts = {}
    for p in pairs:
        et = p.get("error_type", "unknown")
        error_counts[et] = error_counts.get(et, 0) + 1

    print(f"\n📊 错误类型分布:")
    ERROR_LABELS = {
        "wrong_tool": "工具选错",
        "missing_param": "参数缺失",
        "wrong_param_name": "参数名错误",
        "bad_json": "JSON 格式错误",
        "natural_lang": "输出自然语言",
        "extra_text": "多余解释文本",
        "tool_to_none": "该调却输出none",
        "none_to_tool": "不该调却调工具",
        "clarify_to_tool": "该追问却调工具",
        "hallucinated_param": "编造参数",
    }
    for et, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        label = ERROR_LABELS.get(et, et)
        print(f"   {label} ({et}): {count} 条 ({count/len(pairs)*100:.1f}%)")

    # 保存
    output_path = args.output or os.path.join(PROJECT_ROOT, "data", "dpo", "train.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"\n✅ DPO 数据已保存到: {output_path}")

    # 展示几个样例
    print(f"\n{'=' * 60}")
    print("样例展示（前 3 对）")
    print("=" * 60)
    for i, p in enumerate(pairs[:3]):
        print(f"\n--- Pair {i + 1} [{ERROR_LABELS.get(p['error_type'], p['error_type'])}] ---")
        print(f"Prompt:  {p['prompt'][:60]}")
        print(f"Chosen:  {p['chosen'][:80]}")
        print(f"Rejected:{p['rejected'][:80]}")


if __name__ == "__main__":
    main()
