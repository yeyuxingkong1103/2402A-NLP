# results 目录说明

**工单编号**：人工智能NLP-RAG-基于PDF文档的问答系统优化

本目录存放四个实验脚本**真实运行后**产出的结果文件。
所有数字均由脚本实际建索引、实际检索、实际调用模型计算得到，
不含任何硬编码；删掉本目录后重新执行脚本即可完整复现。

---

## 一、文件清单与生成方式

| 文件 | 生成脚本 | 内容 |
|---|---|---|
| `chunking_comparison.json` / `.md` | `src/optimize_chunking.py` | 四种分块策略的命中率 / MRR / Recall@k / 块长 / 耗时对比 |
| `retrieval_comparison.json` / `.md` | `src/optimize_retrieval.py` | 四种重排配置的准确率 / 分阶段耗时对比 |
| `prompt_comparison.json` / `.md` | `src/optimize_prompt.py` | 朴素 Prompt vs 优化 Prompt 的准确率 / RAGAS 指标对比 |
| `before_after.json` / `.md` | `src/run_full_optimization.py` | 优化前 vs 优化后：总体指标 + 逐题答案并排 |

`.json` 是机器可读的原始记录（含逐题明细与检索片段），
`.md` 是据此生成的人读报告（表格 + 结论），两者数据同源。

## 二、JSON 结构

### 2.1 通用外层

```json
{
  "wo_id": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
  "params": { "top_k": 5, "recall_k": 20, "...": "本次运行的超参" },
  "metrics": [ /* 各配置/策略的指标块 */ ],
  "generated_at": "YYYY-MM-DD HH:MM:SS"
}
```

### 2.2 `chunking_comparison.json` 的指标块字段

| 字段 | 含义 |
|---|---|
| `strategy` | 分块策略：fixed / recursive / semantic / structure |
| `n_chunks` / `avg_chunk_chars` / `median_chunk_chars` / `max_chunk_chars` | 块数、平均/中位/最长块长（字符） |
| `build_seconds` | 建索引耗时（含向量编码与 BM25 构建） |
| `evidence_hit_rate` / `evidence_mrr` / `evidence_recall_at_k` / `page_hit_rate` | 证据级检索指标 |
| `retrieval_seconds_avg` / `_p95` | 单题检索耗时 |
| `details[]` | 逐题：是否命中、首个命中排名、证据覆盖 `m/n`、命中页码、证据页码 |

### 2.3 `retrieval_comparison.json` 的指标块字段

| 字段 | 含义 |
|---|---|
| `config` | a_纯向量检索 / b_向量+TFIDF重排 / c_向量+LLM重排 / d_向量+级联重排 |
| `accuracy` / `correct` / `total` | 关键词口径准确率与计数 |
| `evidence_*` | 同上，证据级检索指标 |
| `retrieval_seconds` / `rerank_seconds` / `generate_seconds` / `total_seconds` | `{avg, p50, p95, max, n}` 分阶段耗时（秒） |
| `within_3s_retrieval` / `within_3s_end2end` | 检索+重排 ≤3s 比例 / 端到端 ≤3s 比例 |
| `accuracy_details[]` / `evidence_details[]` | 逐题判定与证据明细 |
| `records[]` | 逐题答案、Top-3 片段（chunk_id / 页码 / 分数 / 重排理由 / 摘录）、分阶段耗时 |

### 2.4 `prompt_comparison.json`

顶层含 `naive` 与 `optimized` 两个对称块，每块字段：
`accuracy`、`accuracy_details`、`ragas`（忠实度 / 答案相关性 / 答案正确性等）、
`refused_count`、`refused_rate`、`avg_answer_chars`、`generate_seconds`、`records[]`。

### 2.5 `before_after.json`

顶层为 `before` / `after` / `delta` 三块，`before` 与 `after` 的结构相同：

| 字段 | 含义 |
|---|---|
| `config` | 该侧的完整配置（collection / strategy / fusion / reranker / prompt） |
| `accuracy` 等 | 与检索对比报告同名的指标 |
| `ragas` | 忠实度 / 答案正确性 / 上下文召回等汇总 |
| `details[]` | 逐题：问题、两版答案、参考答案、检索页码、耗时 |

## 三、阅读顺序建议

1. 先看 `before_after.md` 第二节（总体指标）与第三节（逐题答案并排）；
2. 再看三份专项报告解释「提升来自哪一层」：
   `chunking_comparison.md` → `retrieval_comparison.md` → `prompt_comparison.md`；
3. 需要核对某个数字时，回到同名 `.json` 查逐题明细。

## 四、复现与清理

```bash
# 完整复现（先跑 1~3，再跑 4 汇总；也可直接只跑 4）
python src/optimize_chunking.py
python src/optimize_retrieval.py
python src/optimize_prompt.py
python src/run_full_optimization.py

# 清理结果与索引缓存（下次运行将全量重建）
#   results/*.json  results/*.md   —— 结果
#   data/index/wo02_*              —— 本工单的向量库与 BM25 索引
#   data/cache/parsed_招股说明书1_* —— PDF 解析缓存（可选，删后需重新解析）
```

> 提示：`docs/优化方案.md` 中的三处实测表格带 `AUTO:*` 标记，
> 每次运行对应脚本都会把最新实测数据回填进去，
> 因此**文档数字与结果文件始终一致**。
