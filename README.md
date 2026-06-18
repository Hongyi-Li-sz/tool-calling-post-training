# 中文 Tool Calling 后训练：基于 SFT + DPO 的 JSON 工具调用优化

## 项目背景

Tool Calling（工具调用）是 LLM Agent 的核心能力——模型需要根据用户意图，准确选择工具并以规范的 JSON 格式输出参数。小模型（<1B）在指令遵循和格式化输出方面不够稳定，常出现 JSON 格式错误、工具选择错误、参数缺失等问题。

本项目通过 **SFT（监督微调）+ DPO（偏好优化）** 的后训练方案，经过 V1/V2 两轮数据驱动迭代，将 Qwen2.5-0.5B 的 **完全正确率从 18% 提升至 52%（200 条评测集，提升约 3x）**，过度追问率从 84% 降至 2.1%。

## 项目目标

基于 Qwen2.5-0.5B-Instruct，通过两阶段后训练实现：
1. 根据用户问题正确选择工具（或正确拒绝）
2. 稳定输出合法 JSON 格式的工具调用参数
3. 参数缺失时主动追问而非编造 

## 方法

```
V1: Base → SFT-v1 → DPO-v1          (50条评测 → 发现过度追问问题)
V2: Base → SFT-v2 → DPO-v2          (200条评测 → 精准修复，52% 完全正确率)

V1 教训: DPO 通用数据引入过度追问     V2 突破: Bad Case 驱动精准数据迭代
```

## 工具设计

定义了 5 个常见业务工具，遵循标准 JSON Schema：

| 工具 | 用途 | 必填参数 |
|------|------|---------|
| `search_docs` | 检索知识库 | `query` |
| `calculator` | 计算数学表达式 | `expression` |
| `query_order` | 查询订单状态 | `order_id` |
| `book_meeting` | 预约会议 | `date`, `time`, `topic` |
| `send_email` | 发送邮件 | `recipient`, `subject`, `content` |

输出格式：`{"tool": "工具名", "arguments": {...}}`，不调用工具时输出 `{"tool": "none", "arguments": {}}`。

## 数据构造

### 两轮迭代

| 版本 | SFT 数据 | DPO 数据 | 策略 |
|------|---------|---------|------|
| V1 | 573 条 | 760 对 | 模板批量生成，通用覆盖 |
| V2 | 600 条 | 313 对 | Bad Case 驱动，精准修复 |

> V2 不是 V1 的叠加。每个版本都是从 Base Model 独立训练。

### V1 SFT 数据（573 条）

采用模板 + 随机参数填充生成，覆盖 5 种场景：

| 场景 | 数量 | 说明 |
|------|------|------|
| 单工具调用 | 400 | 5 个工具 × 80 条，16+ 种句式变体 |
| 不需要工具 | 87 | 闲聊、常识、问候 → `{"tool": "none"}` |
| 参数缺失追问 | 40 | 缺参数时输出 `need_clarification: true` |
| 格式强化 | 20 | 确定性样本巩固 JSON 输出模式 |
| 干扰/复合请求 | 26 | 多意图混杂，只处理首要意图 |

### DPO 数据（760 对）

从 SFT 数据出发，程序化生成 10 种错误类型作为 rejected：

| 错误类型 | 数量 | 示例 |
|---------|------|------|
| 输出自然语言 | 130 | 用"好的，我来处理"代替 JSON |
| 多余解释文本 | 130 | JSON 前后加自然语言 |
| JSON 格式错误 | 112 | 缺引号、多余逗号 |
| 编造参数 | 80 | 添加不存在的字段 |
| 该调却输出 none | 79 | 需要工具却说不需要 |
| 参数名错误 | 74 | `order_id` → `data` |
| 工具选错 | 60 | 发邮件 → 预约会议 |
| 该追问却调工具 | 32 | 缺参数时编造 |
| 不该调却调工具 | 32 | 闲聊 → 调用工具 |
| 参数缺失 | 31 | 少必填参数 |

## 训练配置

V1 和 V2 使用相同的训练超参数，区别仅在于训练数据和模型版本。

### SFT（V1 和 V2 共用配置）

| 参数 | 值 |
|------|-----|
| 框架 | TRL SFTTrainer |
| 量化 | 4-bit QLoRA (NF4) |
| LoRA | rank=8, alpha=16 |
| Epochs | 8 |
| Batch size | 2 |
| Learning rate | 5e-5 (cosine) |
| 可训练参数 | ~440 万 (~1%) |

| 版本 | 数据 | 训练时间 | Final loss |
|------|------|---------|-----------|
| SFT-v1 | 573 条 | 140s | 0.43 |
| SFT-v2 | 600 条 | 778s | 0.09 |

### DPO（V1 和 V2 共用配置）

| 参数 | 值 |
|------|-----|
| 框架 | TRL DPOTrainer |
| LoRA | rank=8, alpha=16 |
| Epochs | 3 |
| DPO beta | 0.1 |
| Learning rate | 5e-6 (cosine) |

| 版本 | 数据 | 基准模型 | 训练时间 | Rewards margin |
|------|------|---------|---------|---------------|
| DPO-v1 | 760 对 | SFT-v1 | 455s | 4.55 |
| DPO-v2 | 313 对 | SFT-v2 | 188s | 5.03 |

## 实验结果

### V2 最终评测（200 条标注评测集，7 项指标）

| 指标 | Base | SFT-v1 | DPO-v1 | SFT-v2 | **DPO-v2** |
|------|------|--------|--------|--------|------------|
| JSON 合法率 | 80.0% | 99.5% | 95.5% | 99.0% | 91.0% |
| 工具选择准确率 | 33.0% | 57.5% | 55.5% | 75.5% | 73.0% |
| **完全正确率** | 18.0% | 32.0% | 25.5% | 40.0% | **52.0% 🏆** |
| 过度追问率 | 0.0% | 84.0% | 96.3% | 74.6% | **2.1%** |
| 不必要工具调用率 | 100.0% | 72.8% | 70.6% | 32.5% | **5.2%** |

### 全程提升轨迹

```
Base → SFT-v1 → SFT-v2 → DPO-v2
18%     32%      40%      52%
   +14%     +8%     +12%

总提升: 18% → 52%（+34 个百分点，约 3x）
```

### V1 vs V2 核心发现

| 发现 | V1 | V2 |
|------|-----|-----|
| 最佳模型 | SFT-v1 (54%/50条) | **DPO-v2 (52%/200条)** |
| DPO 作用 | 负面（过度追问 +10） | **正面（过度追问 -72.5%）** |
| 数据策略 | 模板批量生成 | Bad Case 精准修复 |
| 参数追问 | 0/5 正确 | 42.2% 追问准确率 |
| 语义干扰 | 未评测 | 突破 0%→23.3% |

> 详见 [reports/v2_comparison.md](reports/v2_comparison.md)

## 项目结构

```
.
├── data/
│   ├── sft/
│   │   ├── train.json            # V1: 573 条 SFT 数据
│   │   └── train_v2.json         # V2: 600 条修复型 SFT 数据
│   ├── dpo/
│   │   ├── train.json            # V1: 760 对 DPO 数据
│   │   └── train_v2.json         # V2: 313 对精准 DPO 数据
│   └── eval/
│       ├── test_set.json         # V1: 50 条评测集
│       └── test_set_v2.json      # V2: 200 条评测集
├── configs/
│   └── sft.yaml
├── src/
│   ├── build_sft_data.py / build_sft_v2.py
│   ├── build_dpo_data.py / build_dpo_v2.py
│   ├── train_sft.py / train_dpo.py
│   ├── infer.py / evaluate.py / evaluate_v2.py
│   ├── eval_compare.py / eval_three_way.py
│   └── build_eval_v2.py
├── models/                       # (gitignored)
├── outputs/                      # (gitignored)
├── reports/
│   ├── v1_summary.md / v2_comparison.md
│   ├── error_analysis.md / sft_vs_dpo.md
│   └── v2_eval_summary.json / v2_eval_by_category.csv
├── app.py                        # Gradio Demo
├── tools_schema.json
├── plan.txt / plan_v2.txt
├── README.md / CLAUDE.md
├── PROGRESS.md / PROGRESS_v2.md
└── requirements.txt
```

## 快速开始

```bash
# 1. 环境配置
conda create -n tool-calling python=3.11 -y
conda activate tool-calling
pip install -r requirements.txt

# 2. 下载模型（需 HuggingFace 网络）
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

# 3. Base Model 推理测试
python src/infer.py --model_path models/Qwen2.5-0.5B-Instruct

# 4. 构造 SFT 数据
python src/build_sft_data.py

# 5. SFT 训练
python src/train_sft.py

# 6. 构造 DPO 数据
python src/build_dpo_data.py

# 7. DPO 训练
python src/train_dpo.py

# 8. 自动评测
python src/evaluate.py
```

## 技术栈

- **模型**: Qwen2.5-0.5B-Instruct
- **训练框架**: TRL (SFTTrainer + DPOTrainer)
- **高效微调**: PEFT LoRA + QLoRA (4-bit NF4 量化)
- **可训练参数**: ~440 万（仅占模型 1%）
- **硬件**: 单卡 RTX 4090 (24GB)
- **显存占用**: ~4GB（QLoRA 训练时）

## Gradio Demo

```bash
python app.py
# 访问 http://localhost:7861
```

功能：
- 3 个模型一键切换（Base / SFT-v2 / DPO-v2）
- 实时显示原始输出 + JSON 合法性 + 工具名 + 参数
- 10 个预设样例覆盖典型场景

## 下一步优化

- **DPO JSON 回退**: 过度追问抑制过于激进，需平衡 JSON 格式与追问抑制
- **参数追问准确率**: 26.7% 仍偏低，需更多多轮对话型训练数据
- **语义干扰**: 虽有突破（0→23.3%）但绝对值仍低，需更真实的数据
- **多工具并行**: 支持一次输出多个工具调用
- **更大模型**: 尝试 Qwen2.5-1.5B 提升语义理解上限
- **HuggingFace**: 上传 adapter 供社区使用

## 面试亮点

> 基于 Qwen2.5-0.5B 构建中文 Tool Calling 后训练项目，设计 5 类业务工具及 JSON Schema。通过 V1/V2 两轮数据驱动迭代完成 SFT + DPO 训练：V1 模板生成 573 条 SFT + 760 对 DPO 跑通流程并暴露过度追问问题；V2 基于 Bad Case 分析精准构造 600 条修复型 SFT + 313 对 DPO 偏好数据。最终 DPO-v2 完全正确率 52%（200 条评测集），过度追问率从 84% 降至 2.1%。实现 7 项指标分场景自动评测脚本 + Gradio 交互 Demo，完整验证了"DPO 效果取决于偏好数据是否对准模型真实弱点"。