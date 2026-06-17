"""
SFT 训练脚本：使用 QLoRA + TRL SFTTrainer 对 Qwen base model 进行监督微调

流程：
  1. 加载 base model（4-bit 量化）
  2. 注入 LoRA adapter
  3. 加载 Day 2 构造的 SFT 数据
  4. SFT 训练（仅更新 adapter 参数）
  5. 保存 adapter
  6. 在同测试集上对比 base vs SFT
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ── 项目路径 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 与 infer.py 一致的系统提示 ──
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


def load_model_and_tokenizer(model_path: str):
    """加载 4-bit 量化的 base model 和 tokenizer"""
    print(f"正在加载模型: {model_path}")

    # 4-bit 量化配置 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    # 设 pad_token（Qwen 没有默认 pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # 禁用缓存（训练时需要，避免警告）
    model.config.use_cache = False

    print(f"模型加载完成（4-bit 量化）")
    return model, tokenizer


def load_sft_data(data_path: str) -> Dataset:
    """加载 SFT 训练数据，转为 HuggingFace Dataset"""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 转为 Dataset 对象
    dataset = Dataset.from_list(raw_data)
    print(f"加载了 {len(dataset)} 条训练数据")
    return dataset


def formatting_func(example):
    """将每条数据格式化为 Qwen chat template 文本"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    # 使用 tokenizer 的 chat template（需要在训练时通过 SFTTrainer 获得）
    # 这里返回 messages，由 SFTTrainer 在内部处理
    return example


def run_training(args):
    """主训练流程"""

    # ── Step 1: 加载模型 ──
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # ── Step 2: 配置 LoRA ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,                # LoRA 秩
        lora_alpha=args.lora_alpha,    # 缩放系数
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # ── Step 3: 加载数据 ──
    dataset = load_sft_data(args.data_path)

    # ── Step 4: 定义训练参数 ──
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=int(0.05 * (len(dataset) / (args.batch_size * args.gradient_accumulation)) * args.epochs),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        max_length=512,               # 最大序列长度（样本约 70 tokens，512 足够）
        bf16=False,
        fp16=False,
        optim="adamw_8bit",
        report_to="none",
        ddp_find_unused_parameters=False,
        packing=False,                # 不打包
        gradient_checkpointing=True,  # 用计算换显存
    )

    # ── Step 5: 创建 SFTTrainer ──
    def format_chatml(example):
        """格式化函数：将 instruction/output 拼接成 chatml 文本"""
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=format_chatml,
    )

    # ── Step 6: 开始训练 ──
    print("\n" + "=" * 60)
    print("开始 SFT 训练")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  数据: {len(dataset)} 条")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} × {args.gradient_accumulation} (grad accum)")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60 + "\n")

    trainer.train()

    # ── Step 7: 保存 adapter ──
    adapter_path = os.path.join(args.output_dir, "adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✅ Adapter 已保存到: {adapter_path}")

    # 也保存合并后的完整模型（可选，方便直接推理）
    print("正在合并并保存完整模型...")
    merged_path = os.path.join(args.output_dir, "merged")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"✅ 合并模型已保存到: {merged_path}")

    # ── Step 8: 保存训练配置记录 ──
    record = {
        "model": args.model_path,
        "data": args.data_path,
        "data_size": len(dataset),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
    }
    with open(os.path.join(args.output_dir, "training_record.json"), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return trainer


def main():
    parser = argparse.ArgumentParser(description="SFT LoRA 训练")
    # 模型和数据
    parser.add_argument("--model_path", type=str,
                        default="models/Qwen2.5-0.5B-Instruct",
                        help="Base model 路径")
    parser.add_argument("--data_path", type=str,
                        default="data/sft/train.json",
                        help="SFT 训练数据路径")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/sft",
                        help="输出目录")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
