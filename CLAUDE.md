# Tool Calling 后训练项目

## 项目范围
- 本项目所有操作限定在 `/home/lhy/post-training/` 目录下
- 不在项目目录之外创建、修改或删除任何文件
- 所有数据、模型、脚本、输出均存放在此目录内
- 模型权重下载到 `models/` 目录
- 数据集存放在 `data/` 目录
- 训练输出（adapter、日志）存放在 `outputs/` 目录

## 项目目标
基于 Qwen 小模型（0.5B/1.5B），通过 SFT + DPO 训练，实现中文 Tool Calling 的 JSON 稳定输出。

## 技术栈
- 模型：Qwen2.5 / Qwen3 0.5B 或 1.5B
- 训练方法：LoRA / QLoRA
- 框架：LLaMA-Factory（优先）或 TRL
- 任务：中文 Tool Calling JSON 输出

## 硬件环境
- GPU：3× NVIDIA RTX 4090 (24GB VRAM each)
- CUDA：12.4
- PyTorch：2.6.0+cu124
- Python：3.11.7

## 约束
- 使用 Conda 环境管理 Python 依赖（环境名：tool-calling）
- 数据格式统一为 Alpaca/ShareGPT 格式
- 所有脚本使用 Python 编写
- 评测脚本必须支持自动化批量运行

## 代码规范
- Python 脚本使用 argparse 传参
- 配置使用 YAML 文件管理
- 所有路径使用相对路径或基于项目根目录的路径
- 关键步骤输出日志和进度条
