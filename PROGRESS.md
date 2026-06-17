# 项目进度

## 总览

| 日期 | 天数 | 主题 | 状态 |
|------|------|------|------|
| 2026-06-17 | Day 1 | 项目范围 + 搭建仓库 | ✅ 完成 |
| - | Day 2 | 构造 SFT 数据 | ✅ 完成 |
| - | Day 3 | 跑 SFT 训练 | ⬜ 待开始 |
| - | Day 4 | 构造 DPO 偏好数据 | ⬜ 待开始 |
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

## Day 3：跑 SFT 训练
⬜ 待开始

## Day 4：构造 DPO 偏好数据
⬜ 待开始

## Day 5：跑 DPO 训练
⬜ 待开始

## Day 6：自动评测 + Bad Case 分析
⬜ 待开始

## Day 7：整理 README + 实验报告
⬜ 待开始
