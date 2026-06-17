"""
推理脚本：加载 base model 进行 Tool Calling 推理测试
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """你是一个智能助手，可以根据用户的需求调用以下工具：

可用工具：
1. search_docs - 检索知识库
   参数：query (string, 必填) - 搜索关键词或问题

2. calculator - 计算数学表达式
   参数：expression (string, 必填) - 数学表达式

3. query_order - 查询订单状态
   参数：order_id (string, 必填) - 订单号

4. book_meeting - 预约会议
   参数：date (string, 必填) - 会议日期 (YYYY-MM-DD)
         time (string, 必填) - 会议时间 (HH:MM)
         topic (string, 必填) - 会议主题

5. send_email - 发送邮件
   参数：recipient (string, 必填) - 收件人邮箱
         subject (string, 必填) - 邮件主题
         content (string, 必填) - 邮件正文

输出格式要求：
- 如果需要调用工具，输出严格如下的 JSON，不要包含任何额外文字：
  {"tool": "工具名", "arguments": {参数}}

- 如果不需要调用工具，输出：
  {"tool": "none", "arguments": {}}

- 如果用户意图不明确（如缺少必要参数），输出：
  {"tool": "none", "arguments": {}, "need_clarification": true, "question": "追问内容"}
"""

TEST_QUERIES = [
    "帮我查一下订单 A10293 的状态",
    "计算 123 * 456 等于多少",
    "帮我搜索一下大模型后训练的资料",
    "你是谁？",
    "帮我预约明天下午3点的项目评审会议",
    "给 admin@company.com 发一封关于系统升级的通知邮件",
    "帮我预约个会议",
    "搜一下 Python 教程，然后帮我发邮件给经理",
    "今天是几号"
]


def load_model(model_path: str, device: str = "auto"):
    """加载模型和 tokenizer"""
    print(f"正在加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"模型加载完成，设备: {device}")
    return model, tokenizer


def format_prompt(user_query: str) -> str:
    """构建对话 prompt"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    # 使用 Qwen 的 chat template
    return messages


def infer(model, tokenizer, user_query: str, max_new_tokens: int = 256) -> str:
    """执行推理"""
    messages = format_prompt(user_query)

    # 应用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # 只取生成的部分
    input_len = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def try_parse_json(response: str) -> dict | None:
    """尝试从响应中解析 JSON"""
    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块
    import re
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Tool Calling 推理测试")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（本地路径或 HuggingFace 模型名）")
    parser.add_argument("--query", type=str, default=None,
                        help="单次查询（不指定则运行测试集）")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备 (auto/cuda/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="最大生成 token 数")
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(args.model_path, args.device)

    queries = [args.query] if args.query else TEST_QUERIES

    print("\n" + "=" * 60)
    print("推理测试")
    print("=" * 60)

    results = []
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] 用户: {query}")
        response = infer(model, tokenizer, query, args.max_new_tokens)
        print(f"输出: {response}")

        parsed = try_parse_json(response)
        if parsed:
            print(f"✅ JSON 合法: {json.dumps(parsed, ensure_ascii=False)}")
        else:
            print(f"❌ JSON 无法解析")

        results.append({
            "query": query,
            "response": response,
            "json_valid": parsed is not None,
            "parsed": parsed,
        })

    # 统计
    valid_count = sum(1 for r in results if r["json_valid"])
    print(f"\n{'=' * 60}")
    print(f"JSON 合法率: {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
