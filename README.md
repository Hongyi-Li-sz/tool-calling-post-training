# 中文 Tool Calling 后训练：基于 SFT + DPO 的 JSON 工具调用优化

## 项目背景

Tool Calling（工具调用）是 LLM Agent 的核心能力之一——模型需要根据用户意图，准确选择工具并以规范的 JSON 格式输出参数。然而，小模型在指令遵循和格式化输出方面往往不够稳定，容易出现 JSON 格式错误、工具选择错误、参数缺失等问题。

本项目通过 **监督微调（SFT）+ 偏好优化（DPO）** 的后训练方案，显著提升小模型在中文 Tool Calling 场景下的表现。

## 项目目标

让小模型（Qwen 0.5B/1.5B）能够：
1. 根据用户问题正确选择工具（或不调用工具）
2. 稳定输出合法 JSON 格式的工具调用参数
3. 在参数缺失时主动追问用户

## 方法

```
Base Model (Qwen2.5-0.5B) → SFT (监督微调) → DPO (偏好优化) → Evaluation (自动评测)
```

## 工具设计

定义了 5 个常见业务工具：

| 工具名 | 用途 | 参数 |
|--------|------|------|
| `search_docs` | 检索知识库 | `query` |
| `calculator` | 计算数学表达式 | `expression` |
| `query_order` | 查询订单状态 | `order_id` |
| `book_meeting` | 预约会议 | `date`, `time`, `topic` |
| `send_email` | 发送邮件 | `recipient`, `subject`, `content` |

详见 [tools_schema.json](tools_schema.json)

## 数据构造

- **SFT 数据**: 500+ 条指令数据，覆盖单工具调用、拒调用、参数追问、干扰问题等场景
- **DPO 数据**: 300+ 对偏好数据（chosen/rejected），覆盖 JSON 格式错误、工具选错、参数缺失等错误类型

## 训练配置

- 模型: Qwen2.5-0.5B-Instruct
- 训练方法: LoRA / QLoRA
- 框架: LLaMA-Factory
- 硬件: 3× RTX 4090 (24GB)

详细配置见 [configs/](configs/)

## 评测指标

| 指标 | 说明 |
|------|------|
| JSON 合法率 | `json.loads` 是否成功 |
| 工具选择准确率 | `tool` 字段是否正确 |
| 参数准确率 | `arguments` 是否完整且正确 |
| 拒调用准确率 | 不该调用工具时是否正确拒绝 |

## 实验结果

| 模型 | JSON 合法率 | 工具准确率 | 参数准确率 | 拒调用准确率 |
|------|-------------|------------|------------|--------------|
| Base | - | - | - | - |
| SFT | - | - | - | - |
| DPO | - | - | - | - |

## 项目结构

```
.
├── data/               # 数据集
│   ├── raw/            # 原始数据
│   ├── sft/            # SFT 训练数据
│   ├── dpo/            # DPO 训练数据
│   └── eval/           # 评测数据
├── configs/            # 训练配置文件
├── src/                # 源代码
│   ├── build_sft_data.py
│   ├── build_dpo_data.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── evaluate.py
│   └── infer.py
├── models/             # 模型权重
├── outputs/            # 训练输出
├── reports/            # 实验报告
├── tools_schema.json   # 工具定义
├── README.md
└── PROGRESS.md
```

## 快速开始

```bash
# 1. 环境配置
conda create -n tool-calling python=3.11 -y
conda activate tool-calling
pip install -r requirements.txt

# 2. 下载模型
# modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir models/Qwen2.5-0.5B-Instruct

# 3. 推理测试
python src/infer.py --model_path models/Qwen2.5-0.5B-Instruct

# 4. 训练
# ...（详见各阶段文档）
```

## 下一步优化

- 扩展更多工具
- 支持多工具并行调用
- 支持多轮参数补全
- 引入 RAG 检索工具
- 增加人工偏好数据
- 上传模型 adapter 到 HuggingFace
