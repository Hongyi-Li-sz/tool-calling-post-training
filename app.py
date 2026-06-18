"""
Gradio Demo — 中文 Tool Calling 后训练项目
启动: python app.py
"""
import json
import os
import sys
sys.stdout.reconfigure(line_buffering=True)

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from src.infer import SYSTEM_PROMPT

BASE = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-0.5B-Instruct")
MODELS = {}

EXAMPLES = [
    "帮我查一下订单 A10293 的状态",
    "计算一下 123 * 456",
    "帮我预约明天下午三点的项目复盘会议",
    "你好，介绍一下你自己",
    "我今天不想开会",
    "帮我预约个会议",
    "帮我发封邮件",
    "今天是几号",
    "这个订单号看起来像 A10293，但我不是要查询它",
    "搜一下 Docker 怎么部署",
]


def load_one(name, adapters):
    print(f"加载 {name}...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    for adp in adapters:
        m = PeftModel.from_pretrained(m, adp)
        m = m.merge_and_unload()
    m.eval()
    return m


def predict(user_input, model_name):
    if not user_input.strip():
        return "", "", "", ""
    tokenizer, model = MODELS[model_name]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_input}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        p = json.loads(raw)
        js = "✅ 合法 JSON"
        tool = p.get("tool", "—")
        args = json.dumps(p.get("arguments", {}), ensure_ascii=False, indent=2)
        if p.get("need_clarification"):
            tool += " 🔍"
            args = f"[追问] {p.get('question', '')}\n{args}"
    except Exception:
        js, tool, args = "❌ 非法 JSON", "—", "—"
    return raw, js, tool, args


# 启动时加载
print("加载模型...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
MODELS["Base"] = (tok, load_one("Base", []))
MODELS["SFT-v2"] = (tok, load_one("SFT-v2", ["outputs/sft_v2/adapter"]))
MODELS["DPO-v2"] = (tok, load_one("DPO-v2", ["outputs/sft_v2/adapter", "outputs/dpo_v2/adapter"]))
print("✅ 就绪！", flush=True)

with gr.Blocks(title="Tool Calling Demo") as demo:
    gr.Markdown("""# 🔧 中文 Tool Calling 后训练 Demo
### Qwen2.5-0.5B × QLoRA | Base / SFT-v2 / DPO-v2""")

    with gr.Row():
        with gr.Column(scale=1):
            model_sel = gr.Dropdown(choices=list(MODELS.keys()), value="DPO-v2", label="模型")
            inp = gr.Textbox(label="输入请求", lines=3, placeholder="帮我查一下订单 A10293 的状态")
            btn = gr.Button("🚀 发送", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=[inp])

        with gr.Column(scale=2):
            raw = gr.Textbox(label="原始输出", lines=3, interactive=False)
            with gr.Row():
                js_stat = gr.Textbox(label="JSON", interactive=False)
                tool_name = gr.Textbox(label="Tool", interactive=False)
            args = gr.Textbox(label="Arguments", lines=5, interactive=False)

    btn.click(fn=predict, inputs=[inp, model_sel], outputs=[raw, js_stat, tool_name, args])

    gr.Markdown("""---
| 模型 | JSON合法率 | 完全正确率 | 过度追问率 |
|------|-----------|-----------|-----------|
| Base | 80.0% | 18.0% | 0.0% |
| SFT-v2 | 99.0% | 40.0% | 74.6% |
| **DPO-v2** | **91.0%** | **52.0%** 🏆 | **2.1%** |
> 200条标注评测集 | [GitHub](https://github.com/Hongyi-Li-sz/tool-calling-post-training)""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
