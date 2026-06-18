"""
Gradio Demo — 中文 Tool Calling 后训练项目交互展示

启动: python app.py
访问: http://localhost:7860

支持 5 个模型版本对比：Base / SFT-v1 / DPO-v1 / SFT-v2 / DPO-v2
"""
import json
import os
import sys

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.infer import SYSTEM_PROMPT

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-0.5B-Instruct")

MODEL_CONFIGS = {
    "Base (Qwen2.5-0.5B)": {"type": "base"},
    "SFT-v1": {"type": "adapter", "adapter": "outputs/sft/adapter"},
    "DPO-v1": {"type": "double", "adapter1": "outputs/sft/adapter", "adapter2": "outputs/dpo/adapter"},
    "SFT-v2 (推荐)": {"type": "adapter", "adapter": "outputs/sft_v2/adapter"},
    "DPO-v2": {"type": "double", "adapter1": "outputs/sft_v2/adapter", "adapter2": "outputs/dpo_v2/adapter"},
}

EXAMPLES = [
    "帮我查一下订单 A10293 的状态",
    "计算一下 123 * 456",
    "帮我预约明天下午三点的项目复盘会议",
    "帮我发邮件给 Tom，主题是会议纪要，内容是请查看附件",
    "你好，介绍一下你自己",
    "我今天不想开会",
    "帮我预约个会议",
    "帮我发封邮件",
    "今天是几号",
    "这个订单号看起来像 A10293，但我不是要查询它",
]

# 全局模型缓存
_models = {}


def load_models():
    """预加载所有模型"""
    print("加载 Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    base.eval()
    _models["Base (Qwen2.5-0.5B)"] = (tokenizer, base)

    # SFT-v1
    print("加载 SFT-v1...")
    sft1 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    sft1 = PeftModel.from_pretrained(sft1, "outputs/sft/adapter")
    sft1 = sft1.merge_and_unload()
    sft1.eval()
    _models["SFT-v1"] = (tokenizer, sft1)

    # SFT-v2
    print("加载 SFT-v2...")
    sft2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    sft2 = PeftModel.from_pretrained(sft2, "outputs/sft_v2/adapter")
    sft2 = sft2.merge_and_unload()
    sft2.eval()
    _models["SFT-v2 (推荐)"] = (tokenizer, sft2)

    # DPO-v1
    print("加载 DPO-v1...")
    dpo1 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    dpo1 = PeftModel.from_pretrained(dpo1, "outputs/sft/adapter")
    dpo1 = dpo1.merge_and_unload()
    dpo1 = PeftModel.from_pretrained(dpo1, "outputs/dpo/adapter")
    dpo1 = dpo1.merge_and_unload()
    dpo1.eval()
    _models["DPO-v1"] = (tokenizer, dpo1)

    # DPO-v2
    print("加载 DPO-v2...")
    dpo2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    dpo2 = PeftModel.from_pretrained(dpo2, "outputs/sft_v2/adapter")
    dpo2 = dpo2.merge_and_unload()
    dpo2 = PeftModel.from_pretrained(dpo2, "outputs/dpo_v2/adapter")
    dpo2 = dpo2.merge_and_unload()
    dpo2.eval()
    _models["DPO-v2"] = (tokenizer, dpo2)

    print("✅ 全部模型加载完成！")


def predict(user_input: str, model_name: str):
    """执行推理"""
    if model_name not in _models:
        return "模型未加载", "—", "—", "—"

    tokenizer, model = _models[model_name]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.1,
            do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id)
    input_len = inputs.input_ids.shape[1]
    raw_output = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # 解析 JSON
    try:
        parsed = json.loads(raw_output)
        json_status = "✅ 合法 JSON"
        tool = parsed.get("tool", "—")
        args = json.dumps(parsed.get("arguments", {}), ensure_ascii=False, indent=2)

        clarify = parsed.get("need_clarification", False)
        if clarify:
            tool += " 🔍 追问"
            question = parsed.get("question", "")
            args = f"[追问] {question}\n\n{args}"
    except json.JSONDecodeError:
        json_status = "❌ 非法 JSON"
        tool = "—"
        args = "—"

    return raw_output, json_status, tool, args


# ── UI ──
with gr.Blocks(title="Tool Calling 后训练 Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔧 中文 Tool Calling 后训练 Demo
    ### 基于 Qwen2.5-0.5B × SFT + DPO × QLoRA

    选择一个模型版本，输入中文请求，观察模型如何输出 JSON 工具调用。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value="SFT-v2 (推荐)",
                label="模型版本",
            )
            user_input = gr.Textbox(
                label="输入你的请求",
                placeholder="例如：帮我查一下订单 A10293 的状态",
                lines=3,
            )
            submit_btn = gr.Button("🚀 发送", variant="primary", size="lg")

            gr.Markdown("### 示例")
            gr.Examples(examples=EXAMPLES, inputs=[user_input])

        with gr.Column(scale=2):
            gr.Markdown("### 模型原始输出")
            raw_output = gr.Textbox(label="Raw Output", lines=3, interactive=False)

            with gr.Row():
                json_status = gr.Textbox(label="JSON 状态", interactive=False)
                tool_name = gr.Textbox(label="Tool", interactive=False)

            gr.Markdown("### Arguments")
            arguments = gr.Textbox(label="Arguments", lines=5, interactive=False)

    submit_btn.click(
        fn=predict,
        inputs=[user_input, model_selector],
        outputs=[raw_output, json_status, tool_name, arguments],
    )

    gr.Markdown("""
    ---
    ### 实验数据

    | 模型 | JSON 合法率 | 完全正确率 | 过度追问率 |
    |------|-----------|-----------|-----------|
    | Base | 80.0% | 18.0% | 0.0% |
    | SFT-v1 | 99.5% | 32.0% | 84.0% |
    | DPO-v1 | 95.5% | 25.5% | 96.3% |
    | SFT-v2 | 99.0% | 40.0% | 74.6% |
    | **DPO-v2** | **91.0%** | **52.0%** 🏆 | **2.1%** |

    > 基于 200 条标注评测集。DPO-v2 是最终推荐模型。
    > 项目地址：[GitHub](https://github.com/Hongyi-Li-sz/tool-calling-post-training)
    """)


if __name__ == "__main__":
    load_models()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
