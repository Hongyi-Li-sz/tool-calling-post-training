# V2 项目进度

> V2 从 Day 8 开始，与 V1（Day 1-7）明确区分。
> V1 最佳结果：SFT-v1 完全正确率 54%，JSON 合法率 100%。

## 总览

| 日期 | 天数 | 主题 | 状态 |
|------|------|------|------|
| 2026-06-18 | Day 8 | 冻结 V1 Baseline + 扩展评测集（200条） | ✅ 完成 |
| - | Day 9 | 升级评测脚本（分场景+7项指标） | ⬜ 待开始 |
| - | Day 10 | 构造 SFT-v2 修复型数据（600条） | ⬜ 待开始 |
| - | Day 11 | 训练 SFT-v2 + 四模型对比 | ⬜ 待开始 |
| - | Day 12 | 构造 DPO-v2 数据（300-500对） | ⬜ 待开始 |
| - | Day 13 | 训练 DPO-v2 + 五模型对比 | ⬜ 待开始 |
| - | Day 14 | Gradio Demo + 最终整理 | ⬜ 待开始 |

---

## V1 Baseline（Day 1-7 已完成）

| 指标 | Base | SFT-v1 | DPO-v1 |
|------|------|--------|--------|
| JSON 合法率 | 82% | **100%** | 98% |
| 完全正确率 | 28% | **54%** | 42% |
| Bad Cases | 36 | **23** | 29 |

**V1 核心发现**：
- 最佳模型：SFT-v1
- DPO-v1 未超过 SFT-v1，过度追问从 0→10
- 参数追问是三模型共同盲区（0/5）

---

## V2 关键改进

与 V1 的三处关键调整：
1. **评测脚本前置**：Day 9 升级 evaluate.py → 指导 SFT-v2 训练
2. **冻结 Baseline + 扩展评测集合并在 Day 8**
3. **DPO-v2 有条件才做**：不超 SFT-v2 就诚实记录

---

## Day 8 (2026-06-18)：冻结 V1 Baseline + 扩展评测集 ✅

### 任务清单
- [x] 创建 reports/v1_summary.md（固化 V1 结论）
- [x] 创建 data/eval/test_set_v2.json（200 条，6 类场景）
- [x] 用 V1 三模型在 test_set_v2.json 上重评

### V2 Baseline 评测结果（200 条）

| 指标 | Base | SFT-v1 | DPO-v1 |
|------|------|--------|--------|
| JSON 合法率 | 82.0% | **99.5%** | 95.0% |
| 工具选择正确 | 33.0% | **57.0%** | 56.0% |
| 完全正确率 | 18.0% | **33.0%** | 25.0% |
| Bad Cases | 164 | **134** | 150 |

### 错误类型分布

| 错误类型 | Base | SFT-v1 | DPO-v1 |
|---------|------|--------|--------|
| 该拒未拒 | 44 | 45 | 37 |
| 该追问却调工具 | 39 | **39** | 38 |
| 参数值错误 | 28 | 30 | 34 |
| 不必要的追问 | 0 | 18 | **26** |
| JSON 格式错误 | 36 | 1 | **12** |
| 工具选错 | 15 | 1 | 3 |

### 关键确认
- SFT-v1 仍是最佳模型（完全正确率 33%，比 DPO-v1 高 8%）
- 参数追问是三模型共同盲区（38-39 例，几乎完全相同）
- DPO-v1 JSON 格式回退确认（12 例 vs SFT-v1 的 1 例）

### 产出物
- [x] reports/v1_summary.md
- [x] data/eval/test_set_v2.json（200 条）
- [x] src/build_eval_v2.py
- [x] reports/bad_cases_*.json（V2 评测集上的 V1 结果）

---

## Day 9：升级自动评测脚本
⬜ 待开始

### 任务清单
- [ ] 升级 src/evaluate.py（分场景 + 7 项指标）
- [ ] 新增指标：过度追问率、不必要工具调用率
- [ ] 支持按 category 输出分场景结果
- [ ] 自动导出 bad cases

### 7 项指标
1. json_valid_rate
2. tool_accuracy
3. argument_exact_match
4. clarification_accuracy
5. over_clarification_rate（新增）
6. unnecessary_tool_call_rate（新增）
7. complete_success_rate

### 产出物
- [ ] src/evaluate_v2.py（或更新 src/evaluate.py）

---

## Day 10：构造 SFT-v2 修复型数据
⬜ 待开始

### 任务清单
- [ ] 构造参数缺失追问样本（200 条）
- [ ] 构造语义干扰拒调用样本（120 条，GPT-4 辅助）
- [ ] 构造无需工具+无追问样本（150 条）
- [ ] 构造标准单工具调用样本（100 条）
- [ ] 构造复合请求样本（30 条）
- [ ] 随机抽查 100 条验证

### 数据分布（600 条）
| 类别 | 数量 | 目标 |
|------|------|------|
| 参数缺失追问 | 200 | 修复 V1 盲区（7%→20%+） |
| 语义干扰拒调用 | 120 | 区分关键词 ≠ 真实意图 |
| 无需工具无追问 | 150 | 抑制过度追问 |
| 标准单工具调用 | 100 | 保持已有能力 |
| 复合/模糊请求 | 30 | 边界处理 |

### 产出物
- [ ] data/sft/train_v2.json

---

## Day 11：训练 SFT-v2 + 四模型对比
⬜ 待开始

### 任务清单
- [ ] 训练 SFT-v2（超参数参考 V1，必要时调整）
- [ ] 保存 adapter → outputs/sft_v2/
- [ ] 评测 4 模型：Base / SFT-v1 / DPO-v1 / SFT-v2
- [ ] 输出分场景 7 项指标

### 预期目标
- [ ] JSON 合法率 ≥ 98%
- [ ] 完全正确率 > SFT-v1 (54%)
- [ ] 参数缺失追问准确率显著提升
- [ ] 过度追问率 < DPO-v1
- [ ] 标准单工具调用准确率不下降

### 产出物
- [ ] outputs/sft_v2/adapter/
- [ ] reports/v2_eval_sft_v2.json

---

## Day 12：构造 DPO-v2 数据
⬜ 待开始

> 前提：SFT-v2 在 Day 11 中超过 SFT-v1。
> 若未超过，本阶段仅做小规模实验。

### 任务清单
- [ ] 构造"抑制过度追问"偏好对
- [ ] 构造"参数缺失必须追问"偏好对
- [ ] 构造"保持 JSON 格式"偏好对
- [ ] 构造"语义干扰拒调用"偏好对
- [ ] 控制总数 300-500 对

### 产出物
- [ ] data/dpo/train_v2.json

---

## Day 13：训练 DPO-v2 + 五模型对比
⬜ 待开始

### 任务清单
- [ ] 以 SFT-v2 为基础训练 DPO-v2
- [ ] 保存 adapter → outputs/dpo_v2/
- [ ] 评测 5 模型：Base / SFT-v1 / DPO-v1 / SFT-v2 / DPO-v2
- [ ] 判断 DPO-v2 是否超过 SFT-v2

### DPO-v2 作为最佳模型的条件（全部满足）
- [ ] 完全正确率 ≥ SFT-v2
- [ ] 过度追问率 ≤ SFT-v2
- [ ] 参数追问准确率 ≥ SFT-v2
- [ ] 若未满足 → 推荐 SFT-v2 为最终模型

### 产出物
- [ ] outputs/dpo_v2/adapter/
- [ ] reports/v2_eval_final.json

---

## Day 14：Gradio Demo + 最终整理
⬜ 待开始

### 任务清单
- [ ] 编写 app.py（Gradio Demo）
- [ ] 撰写 reports/v2_comparison.md
- [ ] 撰写 reports/v2_error_analysis.md
- [ ] 更新 README.md（加入 V2 结果 + Demo 信息）
- [ ] 提交全部 V2 代码

### 产出物
- [ ] app.py
- [ ] reports/v2_comparison.md
- [ ] reports/v2_error_analysis.md
- [ ] README.md（更新）
- [ ] PROGRESS_v2.md（最终更新）

---

## V2 最终交付物清单

- [ ] data/eval/test_set_v2.json
- [ ] data/sft/train_v2.json
- [ ] data/dpo/train_v2.json
- [ ] outputs/sft_v2/adapter/
- [ ] outputs/dpo_v2/adapter/
- [ ] reports/v1_summary.md
- [ ] reports/v2_comparison.md
- [ ] reports/v2_error_analysis.md
- [ ] reports/v2_eval_summary.json
- [ ] reports/v2_eval_by_category.csv
- [ ] reports/v2_bad_cases.json
- [ ] app.py
- [ ] README.md（更新）
- [ ] PROGRESS_v2.md（完成）
