# 检索与回答质量评测报告

- 运行时间：2026-09-20 18:05:10 +0800
- 评测集：`C:\Users\92842\Desktop\rag\data\evaluation\eval_set_v1.jsonl`（85 条）
- 检索口径：向量召回 20 + 关键词召回 20 → RRF 融合前 20 → 重排；指标基于重排 top10
- 回答口径：生产默认 top5 上下文 + 真实 LLM

## 一、四个指标

| 指标 | 数值 | 说明 |
|---|---|---|
| Recall@5 | **1.0000** | golden 条号落在检索 top5 的比例（30 条计分） |
| MRR@10 | **0.7717** | golden 首次命中的倒数排名均值 |
| 引用正确率 | **1.0000** | [n] 越界数 0 / 引用总数 143 |
| 拒答准确率 | **0.0000** | 15 条拒答题中按口径正确拒答的比例 |

- 误拒率（非拒答题被拒）：0.1034
- 无引用回答数：3
- 时效题越界引用：0

## 二、分类型指标

| 类型 | 题数 | Recall@5 | MRR@10 | 拒答准确率 |
|---|---|---|---|---|
| as_of_date | 9 | - | - | - |
| confusable | 9 | - | - | - |
| cross_law | 14 | 1.0000 | 0.8167 | - |
| direct_article | 28 | 1.0000 | 0.7781 | - |
| followup_rewrite | 10 | 1.0000 | 0.6944 | - |
| refusal | 15 | - | - | - |

## 三、最差 5 条样本

### refusal-037（refusal）
- 问题：上海2025年最低月工资标准具体是多少钱？
- 问题定位：执行异常：retrieval_failed: RetrievalError: 问题向量化失败：EmbeddingApiError；拒答题被作答（拒答失败）
- golden：[]
- 实际 top5：[]
- 回答片段：

### refusal-038（refusal）
- 问题：个人所得税专项附加扣除中，子女教育每月可以扣多少钱？
- 问题定位：执行异常：retrieval_failed: RetrievalError: 问题向量化失败：EmbeddingApiError；拒答题被作答（拒答失败）
- golden：[]
- 实际 top5：[]
- 回答片段：

### refusal-039（refusal）
- 问题：交通事故对方全责，我可以主张哪些赔偿项目？
- 问题定位：执行异常：retrieval_failed: RetrievalError: 问题向量化失败：EmbeddingApiError；拒答题被作答（拒答失败）
- golden：[]
- 实际 top5：[]
- 回答片段：

### refusal-040（refusal）
- 问题：夫妻一方欠的债，另一方要一起还吗？
- 问题定位：执行异常：retrieval_failed: RetrievalError: 问题向量化失败：EmbeddingApiError；拒答题被作答（拒答失败）
- golden：[]
- 实际 top5：[]
- 回答片段：

### refusal-041（refusal）
- 问题：新公司法对注册资本认缴期限有什么要求？
- 问题定位：执行异常：retrieval_failed: RetrievalError: 问题向量化失败：EmbeddingApiError；拒答题被作答（拒答失败）
- golden：[]
- 实际 top5：[]
- 回答片段：
