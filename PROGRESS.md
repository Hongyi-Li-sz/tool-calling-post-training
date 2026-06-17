# 项目进度

## 总览

| 日期 | 天数 | 主题 | 状态 |
|------|------|------|------|
| 2026-06-17 | Day 1 | 项目范围 + 搭建仓库 | ✅ 完成 |
| - | Day 2 | 构造 SFT 数据 | ✅ 完成 |
| 2026-06-17 | Day 3 | 跑 SFT 训练 | ✅ 完成 |
| - | Day 4 | 构造 DPO 偏好数据 | ✅ 完成 |
| - | Day 5 | 跑 DPO 训练 | ⬜ 待开始 |
| - | Day 6 | 自动评测 + Bad Case 分析 | ⬜ 待开始 |
| - | Day 7 | 整理 README + 实验报告 | ⬜ 待开始 |

---

## Day 1 (2026-06-17)：项目范围 + 搭建仓库 ✅

### 任务清单
- [x] 创建 CLAUDE.md
- [x] 创建项目进度文件
- [x] 创建目录结构
- [x] 初始化 Git 仓库
- [x] 定义 5 个工具的 JSON Schema
- [x] 创建 Conda 环境并安装依赖
- [x] 下载 Qwen base model
- [x] 跑通 base model 推理
- [x] 写 README 初稿

### 产出物
- [x] CLAUDE.md
- [x] PROGRESS.md
- [x] data/ 目录结构
- [x] src/ 目录结构
- [x] tools_schema.json
- [x] conda 环境: `tool-calling` (Python 3.11, PyTorch 2.6.0+cu124)
- [x] base model 推理脚本: `src/infer.py`
- [x] README.md

### 环境确认
- GPU: 3× RTX 4090 (24GB) ✅
- CUDA: 12.4 ✅
- PyTorch: 2.6.0+cu124 ✅
- Python: 3.11.7 ✅
- Conda: 可用 ✅
- 磁盘: 324GB 可用 ✅
- 模型: Qwen2.5-0.5B-Instruct (已下载到 models/) ✅

### Base Model 推理结果

| 指标 | 结果 |
|------|------|
| JSON 合法率 | 7/8 (87.5%) |
| 工具选择正确 | 6/8 |

**发现的主要问题：**
1. **拒调用失败**："你是谁？" 输出自然语言而非 `{"tool": "none"}`
2. **参数缺失时编造**："帮我预约个会议" 应该追问参数但编造了值
3. **日期幻觉**："明天下午3点" 编造了 2023-04-07
4. **多意图处理不足**：只处理了第一个意图

→ 这些正是 SFT 和 DPO 训练需要解决的问题 🎯

### Git 提交
- `141788a` Day 1: 项目初始化

---

## Day 2 (2026-06-17)：构造 SFT 数据 ✅

### 任务清单
- [x] 编写数据构造脚本 `src/build_sft_data.py`
- [x] 覆盖 5 种场景的数据类型
- [x] 生成 573 条训练数据
- [x] 随机抽查 50 条验证通过

### 数据构造方法
- **模板生成**：为每个工具编写 15+ 种句式模板，随机填充参数值
- **参数值池**：维护搜索词、数学表达式、订单号、会议主题、邮箱、邮件内容等候选池
- **5 种数据类型全覆盖**：

| 类型 | 说明 | 数量 |
|------|------|------|
| 单工具调用 | search_docs / calculator / query_order / book_meeting / send_email | 400 条 |
| 不需要工具 | 闲聊、常识问答、问候语 | 87 条 |
| 参数缺失追问 | 缺参数时输出 `need_clarification: true` | 40 条 |
| 格式强化 | 确定性样本强化 JSON 输出 | 20 条 |
| 干扰/复合请求 | 多意图混杂、模糊意图 | 26 条 |

### 数据分布
| 工具 | 数量 | 占比 |
|------|------|------|
| none（不调用/追问） | 142 | 24.8% |
| search_docs | 89 | 15.5% |
| calculator | 87 | 15.2% |
| query_order | 85 | 14.8% |
| send_email | 85 | 14.8% |
| book_meeting | 85 | 14.8% |

### 数据格式 (Alpaca)
```json
{
  "instruction": "帮我查一下订单 A10293 的状态",
  "input": "",
  "output": "{\"tool\":\"query_order\",\"arguments\":{\"order_id\":\"A10293\"}}"
}
```

### 产出物
- [x] `src/build_sft_data.py` — 数据构造脚本
- [x] `data/sft/train.json` — 573 条训练数据

### Git 提交
- `8a1c9bd` Day 2: 构造 SFT 训练数据

## Day 3 (2026-06-17)：跑 SFT 训练 ✅

### 任务清单
- [x] 编写训练脚本 `src/train_sft.py` (TRL SFTTrainer + PEFT LoRA)
- [x] 创建训练配置 `configs/sft.yaml`
- [x] 运行 SFT 训练 (QLoRA 4bit, LoRA rank=8, 8 epochs)
- [x] 对比评测 Base vs SFT
- [x] 发现并解决 4-bit merge 推理质量问题

### 训练配置
| 参数 | 值 |
|------|-----|
| 框架 | TRL SFTTrainer |
| 量化 | 4-bit QLoRA (NF4) |
| LoRA rank | 8, alpha=16 |
| Epochs | 8 |
| Batch size | 2 (无 grad accum) |
| Learning rate | 5e-5 (cosine schedule) |
| 可训练参数 | 4,399,104 (~1% 原模型) |
| 训练时间 | 140s (288 steps, RTX 4090) |
| Final loss | 0.43 |
| Final token accuracy | 98.8% |

### 评测结果：Base vs SFT

| # | 测试用例 | Base | SFT |
|---|---------|------|-----|
| 1 | 查订单 A10293 | ✅ query_order | ✅ query_order |
| 2 | 计算 123*456 | ✅ calculator | ✅ calculator |
| 3 | 搜索大模型后训练 | ✅ search_docs | ✅ search_docs |
| 4 | 你是谁？ | ❌ 自然语言 | ✅ **{"tool": "none"}** 🎉 |
| 5 | 预约明天会议 | ✅ book_meeting | ✅ book_meeting (日期正确!) |
| 6 | 发系统升级通知邮件 | ✅ send_email | ✅ send_email |
| 7 | 帮我预约个会议 | ⚠️ 编造参数 | ⚠️ 仍编造参数 |
| 8 | 搜Python+发邮件 | ✅ search_docs | ✅ search_docs |
| 9 | 今天是几号 | ❌ 调query_order | ❌ 仍调query_order |

| 指标 | Base | SFT |
|------|------|-----|
| JSON 合法率 | 88.9% | **100%** |
| 工具选择正确 | ~78% | ~89% |

### 关键发现
1. **"你是谁？" → {"tool": "none"}** 生效了！SFT 教会了模型拒绝不相关请求
2. **日期幻觉已修复**：从 2023-04-07 → 2026-07-18（正确识别"明天"）
3. **4-bit merged 模型有舍入误差** → 推理时应使用 float16 加载 base + adapter
4. **剩余问题**（留给 DPO）：参数缺失仍编造、语义理解偏差

### 产出物
- [x] `src/train_sft.py` — SFT 训练脚本
- [x] `configs/sft.yaml` — 训练超参数记录
- [x] `outputs/sft/adapter/` — LoRA adapter 权重
- [x] `outputs/sft/merged/` — 合并模型（float16 推理用）
- [x] `src/eval_compare.py` — 对比评测脚本
- [x] `reports/sft_vs_base.json` — 详细对比数据

### Git 提交
- 待提交

## Day 4 (2026-06-17)：构造 DPO 偏好数据 ✅

### 任务清单
- [x] 编写 DPO 数据构造脚本 `src/build_dpo_data.py`
- [x] 从 SFT 数据生成 chosen/rejected 偏好对
- [x] 覆盖 10 种错误类型
- [x] 保存 760 对到 `data/dpo/train.json`

### 构造方法
- **chosen**：SFT 数据的标准答案（合法 JSON、正确工具、完整参数）
- **rejected**：程序化为每个样本生成 1-2 个错误变体

### 错误类型分布（760 对）

| 错误类型 | 数量 | 占比 | 说明 |
|---------|------|------|------|
| 输出自然语言 | 130 | 17.1% | 用自然语言代替 JSON |
| 多余解释文本 | 130 | 17.1% | JSON 前后加"好的，我来帮你..." |
| JSON 格式错误 | 112 | 14.7% | 缺引号、多余逗号、单引号 |
| 编造参数 | 80 | 10.5% | 添加不存在的参数 |
| 该调却输出 none | 79 | 10.4% | 需要工具却说不需要 |
| 参数名错误 | 74 | 9.7% | 参数名写错 |
| 工具选错 | 60 | 7.9% | 选了错误的工具 |
| 该追问却调工具 | 32 | 4.2% | 缺参数时编造调用 |
| 不该调却调工具 | 32 | 4.2% | 闲聊问题调了工具 |
| 参数缺失 | 31 | 4.1% | 缺少必填参数 |

### DPO 数据格式
```json
{
  "prompt": "查订单：TBA1029384756",
  "chosen": "{\"tool\":\"query_order\",\"arguments\":{\"order_id\":\"TBA1029384756\"}}",
  "rejected": "{\"tool\":\"query_order\",\"arguments\":{\"data\":\"TBA1029384756\"}}",
  "error_type": "wrong_param_name"
}
```

### 产出物
- [x] `src/build_dpo_data.py` — DPO 数据构造脚本
- [x] `data/dpo/train.json` — 760 对偏好数据

## Day 5：跑 DPO 训练
⬜ 待开始

## Day 6：自动评测 + Bad Case 分析
⬜ 待开始

## Day 7：整理 README + 实验报告
⬜ 待开始
