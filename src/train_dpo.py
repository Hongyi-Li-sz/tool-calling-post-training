"""
DPO 训练脚本：在 SFT 模型基础上，使用 TRL DPOTrainer + QLoRA 做偏好优化

流程：
  1. 加载 SFT merged 模型（已具备基本 Tool Calling 能力）
  2. 应用 4-bit 量化 + 注入新 LoRA adapter
  3. 加载 Day 4 构造的 DPO 偏好数据（760 对）
  4. DPO 训练：让模型更偏好正确/简洁/合法的 JSON 输出
  5. 保存 DPO adapter
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 复用 SFT 训练中的系统提示
from src.train_sft import SYSTEM_PROMPT


def load_sft_model(model_path: str):
    """加载 SFT 模型（float16）+ 应用 4-bit 量化"""
    print(f"正在加载 SFT 模型: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},        # 强制单卡，避免 DPO ref model 多卡设备不一致
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print("SFT 模型加载完成（4-bit 量化）")
    return model, tokenizer


def load_dpo_data(data_path: str) -> Dataset:
    """加载 DPO 偏好数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list(raw_data)
    print(f"加载了 {len(dataset)} 对 DPO 数据")
    return dataset


def run_training(args):
    """主训练流程"""

    # ── Step 1: 加载 SFT 模型 ──
    model, tokenizer = load_sft_model(args.model_path)

    # ── Step 2: LoRA 配置（在 SFT 基础之上新建 adapter）──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # ── Step 3: 加载 DPO 数据 ──
    dataset = load_dpo_data(args.data_path)

    # ── Step 4: 训练参数 ──
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        max_length=512,
        bf16=False,
        fp16=False,
        optim="adamw_8bit",
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        # DPO 特有参数
        beta=args.beta,
        loss_type="sigmoid",
        precompute_ref_log_probs=False,  # 训练时实时计算（避免缓存文件丢失问题）
    )

    # ── Step 5: 定义格式化函数 ──
    def format_dpo(example):
        """将 prompt/chosen/rejected 格式化为 chatml"""
        # prompt 部分（system + user）
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

        # chosen 部分
        chosen_text = example["chosen"]

        # rejected 部分
        rejected_text = example["rejected"]

        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    # 预处理
    formatted_dataset = dataset.map(format_dpo)

    # ── Step 6: 创建 DPOTrainer ──
    trainer = DPOTrainer(
        model=model,
        ref_model=None,           # 自动从 model 创建参考模型（即 SFT 模型）
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ── Step 7: 开始训练 ──
    print("\n" + "=" * 60)
    print("开始 DPO 训练")
    print("=" * 60)
    print(f"  基准模型: {args.model_path} (SFT)")
    print(f"  数据: {len(dataset)} 对")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  DPO beta: {args.beta}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60 + "\n")

    trainer.train()

    # ── Step 8: 保存 DPO adapter ──
    adapter_path = os.path.join(args.output_dir, "adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✅ DPO Adapter 已保存到: {adapter_path}")

    # 保存训练记录
    record = {
        "base_model": args.model_path,
        "data": args.data_path,
        "data_size": len(dataset),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "dpo_beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    with open(os.path.join(args.output_dir, "training_record.json"), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return trainer


def main():
    parser = argparse.ArgumentParser(description="DPO LoRA 训练")
    parser.add_argument("--model_path", type=str,
                        default="outputs/sft/merged",
                        help="SFT 模型路径")
    parser.add_argument("--data_path", type=str,
                        default="data/dpo/train.json",
                        help="DPO 训练数据路径")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/dpo",
                        help="输出目录")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta 参数（越大越接近参考模型）")

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
