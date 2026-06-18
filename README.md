# 中文 Tool Calling 后训练：基于 SFT + DPO 的 JSON 工具调用优化

## 项目背景

Tool Calling（工具调用）是 LLM Agent 的核心能力——模型需要根据用户意图，准确选择工具并以规范的 JSON 格式输出参数。小模型（<1B）在指令遵循和格式化输出方面不够稳定，常出现 JSON 格式错误、工具选择错误、参数缺失等问题。

本项目通过 **SFT（监督微调）+ DPO（偏好优化）** 的后训练方案，将 Qwen2.5-0.5B 的 **JSON 合法率从 82% 提升至 100%，完全正确率从 28% 提升至 54%**。

## 项目目标

基于 Qwen2.5-0.5B-Instruct，通过两阶段后训练实现：
1. 根据用户问题正确选择工具（或正确拒绝）
2. 稳定输出合法 JSON 格式的工具调用参数
3. 参数缺失时主动追问而非编造 

## 方法

```
Base Model (Qwen2.5-0.5B) → SFT (监督微调) → DPO (偏好优化) → Evaluation (自动评测)
         ↓                        ↓                    ↓                ↓
    JSON 合法 82%           JSON 合法 100%       rewards/margin    错误分类 ×7
    完全正确 28%            完全正确 54%          4.55              分析报告
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

### SFT 数据（573 条）

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

### SFT 训练

| 参数 | 值 |
|------|-----|
| 框架 | TRL SFTTrainer |
| 量化 | 4-bit QLoRA (NF4) |
| LoRA | rank=8, alpha=16 |
| Epochs | 8 |
| 有效 batch size | 2 |
| Learning rate | 5e-5 (cosine) |
| 可训练参数 | 4,399,104 (~1%) |
| 训练时间 | 140s (RTX 4090) |
| Final loss | 0.43 |

### DPO 训练

| 参数 | 值 |
|------|-----|
| 框架 | TRL DPOTrainer |
| 基准模型 | SFT merged |
| LoRA | rank=8, alpha=16 |
| Epochs | 3 |
| DPO beta | 0.1 |
| Learning rate | 5e-6 (cosine) |
| 训练时间 | 455s (RTX 4090) |
| Final loss | 0.14 |
| Rewards margin | 4.55 |

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
│   ├── sft/train.json        # 573 条 SFT 训练数据
│   ├── dpo/train.json        # 760 对 DPO 偏好数据
│   └── eval/test_set.json    # 50 条标注评测集
├── configs/
│   └── sft.yaml              # SFT 训练超参数
├── src/
│   ├── build_sft_data.py     # SFT 数据构造脚本
│   ├── build_dpo_data.py     # DPO 数据构造脚本
│   ├── train_sft.py          # SFT 训练脚本
│   ├── train_dpo.py          # DPO 训练脚本
│   ├── infer.py              # 单模型推理脚本
│   ├── eval_compare.py       # Base vs SFT 对比
│   ├── eval_three_way.py     # Base vs SFT vs DPO 三方对比
│   └── evaluate.py           # 自动评测（4指标 + 7错误分类）
├── models/                   # 模型权重（.gitignore）
├── outputs/
│   ├── sft/adapter/          # SFT LoRA adapter (~17MB)
│   └── dpo/adapter/          # DPO LoRA adapter (~17MB)
├── reports/
│   ├── error_analysis.md     # Bad Case 深度分析报告
│   ├── evaluation_full.json  # 完整评测数据
│   └── bad_cases_*.json      # 三模型 bad cases 详情
├── tools_schema.json         # 5 个工具的 JSON Schema 定义
├── requirements.txt
├── CLAUDE.md
├── PROGRESS.md               # 7 天进度记录
└── README.md
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
# 启动交互式 Demo
python app.py
# 访问 http://localhost:7860

# 支持 5 个模型版本切换，10 个典型样例
```

功能：
- 5 个模型版本一键切换（Base / SFT-v1 / DPO-v1 / SFT-v2 / DPO-v2）
- 实时显示原始输出 + JSON 解析 + Tool + Arguments
- 10 个预设样例覆盖典型场景

## 下一步优化

- **数据层面**: 增加参数追问型数据（当前仅 7%），减少过度追问
- **训练层面**: 迭代 DPO 数据，针对性修复"过度追问"回退
- **功能层面**: 支持多工具并行调用、多轮参数补全
- **评估层面**: 引入 GPT-4 作为 judge 进行语义级评测
- **部署层面**: Gradio Demo、上传 adapter 到 HuggingFace

## 面试亮点

> 基于 Qwen2.5-0.5B 构建中文 Tool Calling 后训练项目，设计 5 类业务工具及 JSON Schema，通过模板生成构造 573 条 SFT 指令数据和 760 对 DPO 偏好数据，使用 QLoRA 完成监督微调与偏好优化。SFT 模型 JSON 合法率从 82% 提升至 100%，完全正确率从 28% 提升至 54%。实现了支持 4 项指标 + 7 类错误的自动评测脚本，并对三阶段模型进行了系统性 Bad Case 分析，识别出"过度追问"是 DPO 的主要副作用、参数追问是三模型共同盲区。