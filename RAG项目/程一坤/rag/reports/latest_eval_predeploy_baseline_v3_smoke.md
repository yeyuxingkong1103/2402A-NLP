# 检索与回答质量评测报告

- 运行时间：2026-09-27 12:00:29 +0800
- 评测集：`C:\Users\92842\Desktop\rag\data\evaluation\eval_set_v1.jsonl`（1 条）
- 检索口径：向量召回 20 + 关键词召回 20 → RRF 融合前 20 → 重排；指标基于重排 top10
- 回答口径：生产默认 top5 上下文 + 真实 LLM
- 当轮变量（`docs/目录与命名约定.md` 第 10 条；JSON 里同源字段 `run_config`）：
  - LLM：`deepseek-flash`（temperature 0.2 / max_tokens 4096 / timeout 120.0s）
  - 提示词：`backend/app/chat/prompt_builder.py` md5 `f43ee63672ff2b7e136a968441bac640`
  - 召回窗口：向量 20 + 关键词 20 → 融合前 20 → 重排
  - 拒答阈值：REFUSAL_MIN_VECTOR_SCORE = 0.6304
  - 评测集 md5：`03575280ead4bbd54df799c1b04bfc91`
- 本轮 Reranker/API 失败次数：0
- API 失败重试题目：无
- 基线有效性：仅接受无最终 Reranker fallback 的完整 100 题轮次；若仍失败则本轮作废。

## 一、四个指标

| 指标 | 数值 | 说明 |
|---|---|---|
| Recall@5 | **1.0000** | golden 条号落在检索 top5 的比例（1 条计分） |
| MRR@10 | **0.5000** | golden 首次命中的倒数排名均值 |
| 引用正确率 | **1.0000** | [n] 越界数 0 / 引用总数 5 |
| 拒答准确率 | **0.0000** | 0 条拒答题中按口径正确拒答的比例 |

- 误拒率（非拒答题被拒）：0.0000
- 对比基准：predeploy_baseline_v2（Recall@5 0.9529 / MRR@10 0.7554；同为 1557 块语料口径）。
- 相对 v2：Recall@5 +0.0471，MRR@10 -0.2554。
- 本次最大收益：非拒答误拒率由 10.59% 降至 0.00%，降幅 10.59 个百分点。
- 无引用回答数：0
- 时效题越界引用：0

## 二、分类型指标

| 类型 | 题数 | Recall@5 | MRR@10 | 拒答准确率 |
|---|---|---|---|---|
| direct_article | 1 | 1.0000 | 0.5000 | - |

## 三、最差 5 条样本
