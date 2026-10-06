# results 目录说明

> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
> 本目录存放工单01 各脚本运行后自动生成的结果文件。**脚本未运行前本目录只有本说明。**

---

## 一、文件总览

| 文件 | 生成脚本 | 内容 |
| --- | --- | --- |
| `index_stats.json` | `src/build_index.py` | 索引构建统计 |
| `rag_vs_llm.json` | `src/compare_rag_vs_llm.py` | RAG vs 纯 LLM 的 10 问全量对比数据 |
| `rag_vs_llm.md` | 同上 | 对比报告（人读版） |
| `evaluation.json` | `src/run_evaluation.py` | RAGAS 评估结果（逐题 + 汇总） |
| `evaluation.md` | 同上 | 评估报告（人读版） |

---

## 二、`index_stats.json` —— 索引统计

生成命令：

```bash
python "工单01-基于PDF文档的问答系统/src/build_index.py"
# 含表格：python .../build_index.py --preset wo03_table --with-tables
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `collection` | 向量库集合名（默认 `prospectus`） |
| `pdf` / `pdf_size_mb` / `doc_name` | 被解析的 PDF 路径、大小、文档名（答案溯源用） |
| `preset` / `preset_config` | 建索引使用的流水线预设及其完整参数（分块策略/是否含表格/检索与重排配置） |
| `docs[].pages` / `docs[].blocks` | 每个文档的页数与解析出的块数 |
| `n_chunks` | 分块总数（送入索引的知识块数量） |
| `n_vectors` / `vector_backend` | 向量库条数与后端（`chroma` 或降级为 `numpy`） |
| `n_bm25_docs` | BM25 倒排索引收录的文档数 |
| `chunk_types` | 块类型分布（text / table / image），验证表格是否解析成功 |
| `build_seconds` | 解析 + 分块 + 向量化 + 建 BM25 的总耗时（秒） |
| `chunk_chars_total/avg/min/max/p50` | 块长度分布，用于评估分块质量与资源消耗 |
| `embed_model` / `llm_model` | 嵌入模型与生成模型名 |
| `index_dir` | 索引落盘目录（`data/index/`） |
| `built_at` / `python` | 生成时间与环境版本 |

---

## 三、`rag_vs_llm.json` —— RAG 与纯 LLM 对比

生成命令：

```bash
python "工单01-基于PDF文档的问答系统/src/compare_rag_vs_llm.py"
# 指定题号：python .../compare_rag_vs_llm.py --ids 260,95
```

结构：`{ meta, summary, records }`

### `meta`（本次运行环境）

`generated_at` 生成时间；`preset` RAG 侧预设；`collection` 集合名；
`top_k` 检索片段数；`llm_model` 生成模型；`embed_model` 嵌入模型。

### `summary`（汇总指标）

| 字段 | 含义 |
| --- | --- |
| `n` | 对比的问题数（默认 10） |
| `rag_avg_latency` / `rag_p95_latency` / `rag_max_latency` | RAG 侧耗时（秒）：均值 / P95 / 最大 |
| `llm_avg_latency` | 纯 LLM 侧平均耗时（秒） |
| `rag_citation_rate` | RAG 答案带引用来源的比例（纯 LLM 恒为 0） |
| `rag_refused` | RAG 明确拒答的题数（纯 LLM 无拒答机制） |
| `rag_keyword_accuracy` / `llm_keyword_accuracy` | 两侧的关键词准确率（全部关键信息点命中才算对） |
| `rag_win` / `llm_win` / `tie` | 逐题胜负统计（按关键词命中数判定） |

### `records[]`（逐题明细）

| 字段 | 含义 |
| --- | --- |
| `id` / `question` / `keywords` | 题号 / 问题原文 / 判分关键词 |
| `rag.answer` / `llm_only.answer` | 两侧的完整答案文本 |
| `rag.latency` / `llm_only.latency` | 两侧耗时（秒） |
| `rag.citations` | 结构化引用（`doc` 文档名、`page` 页码、`chunk_id`、`snippet` 摘录） |
| `rag.docs` | 实际检索到的片段（含页码、类型、分数、前 200 字） |
| `rag.refused` | 是否拒答（上下文不足） |
| `rag.keyword_hits / keyword_miss` | RAG 答案命中/漏答的关键词 |
| `llm_only.keyword_hits / keyword_miss` | 纯 LLM 答案命中/漏答的关键词 |
| `rag.error` / `llm_only.error` | 该侧调用失败时的错误信息（正常为空串） |
| `winner` | 本题优胜方：`RAG` / `纯LLM` / `持平` |

> 提示：`llm.py` 带磁盘缓存，重复运行的耗时仅供参考；演示前清理
> `data/cache/llm/` 可得到更真实的延迟对比。

---

## 四、`evaluation.json` —— RAG 评估结果

生成命令：

```bash
python "工单01-基于PDF文档的问答系统/src/run_evaluation.py"
# 覆盖参考答案：python .../run_evaluation.py --answers my_answers.json
# 官方 ragas：python .../run_evaluation.py --ragas
```

结构：`{ summary, records }`（由 `evaluate.save_report()` 落盘）

### `summary`（汇总）

| 字段 | 含义 |
| --- | --- |
| `n` | 评估题数 |
| `evaluator` | 实际使用的评估器：`builtin`（内置自实现）/ `ragas(official)` |
| `faithfulness` | 忠实度：答案论断被检索上下文支撑的比例（0~1，越高越不幻觉） |
| `answer_relevancy` | 答案相关性：答案反推问题与原问题的平均相似度 |
| `context_precision` | 上下文精度：有用片段是否排在检索结果前列 |
| `context_recall` | 上下文召回：参考答案的信息有多少能从检索上下文找到 |
| `answer_correctness` | 答案正确性（0.7×事实重叠 + 0.3×语义相似，需用 `--metrics` 显式加入） |
| `keyword_accuracy` / `keyword_correct` / `keyword_total` | 关键词准确率及答对题数（工单「准确率」口径） |
| `latency_avg` / `latency_p95` / `latency_max` | 端到端耗时（秒），用于核对「不超过 3 秒」 |
| `hit_rate` / `mrr` / `recall_at_k` | 检索层指标：命中率 / 平均倒数排名 / 召回率 |
| `degraded_from` | 指标降级时记录原始指标列表（如缺少向量模型时） |
| `preset` / `collection` / `generated_at` | 本次评估的配置与时间 |

### `records[]`（逐题明细）

沿用 `evaluate.EvalRecord.to_dict()`：`id` / `question` / `answer` /
`ground_truth` / `reference_doc` / `retrieved_docs` / `retrieved_pages` /
`latency` / `contexts`（截断到 300 字的检索上下文）/ 各指标分值。

> **重要**：`run_evaluation.py` 内置的参考答案与关键词为**人工整理初稿**，
> 请用 `evaluation.md` 第三节的「关键词命中明细」核对；如与文档口径不符，
> 用 `--answers answers.json` 传入修正版后重跑，格式见脚本 `load_answers()`。

---

## 五、`*.md` 两份报告怎么用

| 报告 | 用途 |
| --- | --- |
| `rag_vs_llm.md` | 验收「对比基于 PDF 的结果与只使用 LLM 的结果」：总览表可直接截图进答辩材料 |
| `evaluation.md` | 验收「选择 RAG 评估体系进行评估，返回评估结果」：四大指标 + 准确率 + 逐题明细 |

---

## 六、复现全部结果的顺序

```bash
python "工单01-基于PDF文档的问答系统/src/build_index.py"        # -> index_stats.json
python "工单01-基于PDF文档的问答系统/src/compare_rag_vs_llm.py" # -> rag_vs_llm.json/.md
python "工单01-基于PDF文档的问答系统/src/run_evaluation.py"     # -> evaluation.json/.md
```

> 说明：三个脚本都依赖已建立的索引与 `DEEPSEEK_API_KEY`；
> 评估还会用到本地嵌入模型（sentence-transformers）。
> 若中途更换 `--preset`，建议重建索引后再跑对比与评估，保证结果口径一致。
