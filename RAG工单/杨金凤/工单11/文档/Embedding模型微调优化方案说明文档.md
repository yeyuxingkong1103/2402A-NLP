# Embedding 模型微调 —— 优化方案说明文档

- **工单编号**：人工智能NLP-RAG-Embedding模型微调任务
- **项目名称**：Embedding 模型微调
- **日期**：2025 年

---

## 1. 背景与目标

工单背景指出：通用嵌入模型在专业领域（如金融年报）存在"语义鸿沟"，可能导致检索不准。工单要求通过微调嵌入模型弥合此鸿沟。

| 目标 | 说明 |
| --- | --- |
| 核心任务 | 走通嵌入模型微调流程（数据生成 → 训练 → 评估） |
| 基础模型 | BAAI/bge-small-zh-v1.5 |
| 微调方法 | 对比学习（MultipleNegativesRankingLoss） |
| 评估指标 | context_precision、context_recall |

---

## 2. 微调流程

```text
第1步：构造训练数据
  CCF向量库 + 10个问题 → 检索top3文档 → 30条 (query, positive) 对

第2步：微调嵌入模型
  bge-small-zh-v1.5 + MNRL损失 → 5 epochs → finetuned_bge/

第3步：重建向量库
  用微调后的模型重新向量化CCF年报 → ccf_vector_db_finetuned/

第4步：评估对比
  RAGAS 测量微调前后的 context_precision 和 context_recall
```

---

## 3. 关键实现

### 3.1 训练数据构造

| 项目 | 内容 |
| --- | --- |
| 数据来源 | 工单 07 的 10 个金融年报问题 |
| 正例获取 | 用原始 bge 模型从 CCF 向量库检索 top3 文档 |
| 训练对数量 | 30 条 (query, positive) |
| 格式 | JSONL（`{"query": ..., "positive": ...}`） |

### 3.2 损失函数：MultipleNegativesRankingLoss（MNRL）

MNRL 是对比学习的核心损失函数，其原理：

- 给定 query `q` 和正例 `d⁺`，同批次中其他文档自动作为负例
- 训练目标：让 `q` 与 `d⁺` 的相似度 > `q` 与任何负例的相似度
- 公式：`L = -log(exp(sim(q, d⁺)/τ) / Σ exp(sim(q, dᵢ)/τ))`

### 3.3 训练配置

| 参数 | 值 |
| --- | --- |
| epochs | 5 |
| batch_size | 8 |
| warmup_steps | 5 |
| 损失函数 | MultipleNegativesRankingLoss |
| 设备 | CUDA |

---

## 4. 评估结果（如实报告）

| 指标 | 微调前 | 微调后 | 变化 |
| --- | --- | --- | --- |
| context_precision | 0.7625 | nan | 评估失败 |
| context_recall | 0.6111 | 0.5433 | ↓ 11% |

**结论**：本次微调未提升检索性能，recall 反而下降。

---

## 5. 失败原因分析

| 原因 | 说明 |
| --- | --- |
| 训练数据严重不足 | 仅 30 条数据，远低于工业界微调所需的几千到几万条 |
| 过拟合 | 5 epochs × 30 条数据，模型只记住了这 10 个问题，泛化能力下降 |
| 原始模型已很强 | bge-small-zh-v1.5 在通用中文场景表现优异，小数据集微调反而破坏其泛化能力 |
| precision=nan | RAGAS 评估时 LLM 未按格式输出，该指标计算失败（评估工具本身的问题） |

---

## 6. 经验总结

- **微调嵌入模型需要大量数据**：工业级微调通常需要 1万+ 条 (query, positive) 对
- **小数据集微调有风险**：可能损害模型的通用语义能力（灾难性遗忘）
- **数据构造是关键**：本次数据来自"自己检索自己"，缺乏多样性

**更务实的替代方案：**

1. 增加训练数据（用 LLM 从年报生成更多问题）
2. 使用 LoRA 参数高效微调（防止灾难性遗忘）
3. 直接换用更强的嵌入模型（如 bge-m3）
