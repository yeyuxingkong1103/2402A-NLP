# results/ 结果文件说明

> 工单编号：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**

本目录存放脚本运行后的全部产出。**目录内容由脚本自动生成**，
可以随时删除后重跑；本 README 说明每个文件的格式与字段含义。

---

## 目录结构

```
results/
├── README.md                      本文件
├── tables/                        表格产物（由 table_extractor.py 生成）
│   ├── <文档>_p<页码>_t<序号>.md              每张表一个 Markdown 文件
│   ├── <文档>_p<页码>_t<序号>_merged.md       跨页合并后的表（文件名带 _merged）
│   └── <文档>_records.json                   表格记录缓存（建索引复用，避免重复解析）
├── table_inventory.json / table_inventory_<文档>.json   表清单 + 跨页合并事件
├── index_stats.json               索引统计（build_index_with_tables.py）
├── table_qa.json / table_qa.md    4 个表格问题问答 + 检索精确度（table_qa.py）
├── table_ablation.json / .md      消融实验 A/B 对比（compare_with_without_tables.py）
└── evaluation.json / evaluation.md 14 问 RAG 评估（run_evaluation.py）
```

---

## 1. `tables/*.md`

每张表一个文件，文件头是 HTML 注释形式的元信息，正文是标准 Markdown 表格：

```markdown
<!-- 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化 -->
<!-- 文档：招股说明书2 | 物理页：22 | 印刷页：21 | 表序号：2 | 抽取策略：lines -->
<!-- 规格：6 行 × 3 列（数据行 5） | 字符数：xxx | 质量标记：无 -->
<!-- 标题：本次募集资金拟投资以下项目： -->

| 序号 | 项目名称 | 计划总投资(万元) |
| --- | --- | --- |
| 1 | 仓储及物流中心 | 3,393.40 |
...
```

- **物理页** = PDF 阅读器页码；**印刷页** = 页脚印刷页码（物理页 − 1）；
- **质量标记**：`merged`（由跨页合并而来）、`header_lost`（续页丢失表头）；
- 跨页合并的表文件名带 `_merged` 后缀，头部另有一行 `<!-- 跨页合并：... -->`。

---

## 2. `table_inventory*.json` —— 表清单

```jsonc
{
  "doc": "招股说明书2",
  "summary": {
    "n_tables": 0,              // 表格总数
    "n_pages_with_tables": 0,   // 含表格的页数
    "n_merged_tables": 0,       // 跨页合并后的表数
    "n_header_lost": 0,         // 无表头续页数
    "total_chars": 0,           // 全部表格 Markdown 字符数
    "avg_rows": 0, "avg_cols": 0,
    "page_range": [0, 0],
    "by_strategy": {}           // 各抽取策略命中数
  },
  "merge_events": [
    // 每次跨页合并的判定记录，成功与拒绝都会记录
    {"action": "merged",   "pages": [204, 205], "reason": "续排标志+表头兼容+位置连续", "caption": "合并资产负债表(续)"},
    {"action": "rejected", "pages": [157, 158], "reason": "表头相同但属于新小节「3、…」，合并会造成语义串类", "caption": "3、…"}
  ],
  "tables": [
    {
      "doc": "招股说明书2",
      "page": 22,               // 物理页码
      "printed_page": 21,       // 印刷页码
      "table_index": 2,         // 该页内第几张表（0 起）
      "caption": "本次募集资金拟投资以下项目：",   // 表格上方的标题
      "header": ["序号", "项目名称", "计划总投资(万元)"],
      "rows": 6,                // 规范化后总行数（含表头）
      "cols": 3,
      "data_rows": 5,           // 数据行数
      "char_len": 0,            // Markdown 字符数
      "strategy": "lines",      // lines | lines+text | text+text
      "flags": [],              // merged / header_lost
      "merged_pages": [],       // 合并涉及的页码
      "merged_from": [],        // 被并入的表编号，如 "p205#t0"
      "file": "tables/招股说明书2_p022_t2.md"
    }
  ]
}
```

**怎么用**：核对 id 1~4 对应表格的 `header` 是否为正确的 3 列；
查看 `merge_events` 确认跨页合并的判断是否合理。

---

## 3. `index_stats.json` —— 索引统计

```jsonc
{
  "collection": "wo03_with_table",
  "with_tables": true,
  "use_enhanced_tables": true,      // 是否使用质量校验+跨页合并后的表格
  "vector_backend": "chroma",       // chroma | numpy
  "embed_model": "BAAI/bge-large-zh-v1.5",
  "chunk_strategy": "structure",
  "chunk_size": 400, "chunk_overlap": 80,
  "docs": [{"name": "招股说明书2", "pages": 350,
            "text_blocks": 0, "table_blocks": 0, "chunks": 0, "table_chunks": 0}],
  "n_text_blocks": 0,
  "n_table_records": 0,
  "n_merged_tables": 0,
  "n_chunks": 0,
  "chunk_types": {"text": 0, "table": 0},   // 文本块 / 表格块数量
  "n_vectors": 0,
  "n_bm25_docs": 0,
  "bm25_path": "data/index/wo03_with_table.bm25.pkl",
  "top_table_chunks": [
    {"chunk_id": "招股说明书2-tab-00001", "doc": "招股说明书2", "page": 22,
     "rows": 6, "cols": 3, "caption": "本次募集资金拟投资以下项目：", "char_len": 0}
  ],
  "build_seconds": 0.0
}
```

---

## 4. `table_qa.{json,md}` —— 表格类问题问答（id 1~4）

```jsonc
{
  "summary": {
    "collection": "wo03_with_table",
    "n_questions": 4,
    "top_k": 5,
    "avg_retrieval_precision": 0.0,   // 平均检索精确度
    "avg_keypoint_recall": 0.0,       // 平均要点召回
    "table_hit_rate": 0.0,            // 表格命中率
    "answer_accuracy": 0.0,           // 答案准确率（要点全中）
    "avg_answer_accuracy": 0.0,       // 答案要点覆盖率均值
    "avg_retrieve_ms": 0.0,
    "max_total_seconds": 0.0
  },
  "records": [
    {
      "id": 1,
      "question": "...",
      "reference_answer": "...",
      "must_points": ["1,670万股", "25.04%"],
      "bonus_points": ["6,670万股", "人民币普通股"],
      "answer_grounding_pages": [2, 22, 24],
      "answer_grounding_tables": ["本次发行概况（摘要「四、本次发行情况」表）", "..."],
      "answer": "...",
      "answer_mode": "rag",           // rag | extractive(离线兜底：...) | extractive(--no-llm)
      "retrieved_tables": [           // 检索到的表格原文（最多 2 张）
        {"chunk_id": "...", "doc": "招股说明书2", "page": 24,
         "caption": "四、本次发行情况", "rows": 5, "cols": 2,
         "score": 0.0, "table_boost": 1.35, "text": "【表格】...Markdown..."}
      ],
      "retrieved_pages": [24, 22, 23, 157, 26],
      "retrieval": {
        "precision_at_k": 0.0,        // 命中要点的片段数 / 返回片段数
        "keypoint_recall": 0.0,       // 检索上下文里的要点覆盖率
        "retrieval_precision": 0.0,   // = 0.5×precision_at_k + 0.5×keypoint_recall
        "table_hit": true,            // 前 k 条里是否命中含要点的表格片段
        "hit_pages": [24],
        "relevant_chunks": ["..."],
        "missing": []                 // 未召回的要点
      },
      "answer_score": {
        "answer_accuracy": 1.0,
        "correct": true,              // 要点是否全中
        "hit_points": ["..."],
        "missing_points": [],
        "bonus_hit": ["6,670万股"]
      },
      "timings": {"vector_recall": 0.0, "rerank": 0.0, "retrieve": 0.0, "generate": 0.0, "total": 0.0}
    }
  ]
}
```

`table_qa.md` 是同一内容的人读版本，含**检索到的表格原文**的 Markdown 代码块，
演示视频可直接投屏。

---

## 5. `table_ablation.{json,md}` —— 消融实验（核心证据）

```jsonc
{
  "meta": {
    "实验设计": "A/B 消融：唯一变量为「索引是否包含表格块」",
    "受控变量": {"chunk_strategy": "structure", "chunk_size": 400,
                 "embed_model": "...", "retrieval": "vector + tfidf 重排",
                 "top_k": 5, "use_table_boost": "仅 B 组开启", "use_llm": true},
    "问题构成": {"表格类": 4, "文本类": 10, "合计": 14},
    "判分口径": "归一化关键词匹配（数字去千分位、全角转半角）"
  },
  "index_a_without_tables": {"collection": "wo03_no_table", "n_chunks": 0, "chunk_types": {}},
  "index_b_with_tables":    {"collection": "wo03_with_table", "n_chunks": 0,
                             "chunk_types": {}, "n_table_records": 0, "n_merged_tables": 0},
  "summary": {
    "A_without_tables": {"table_questions": {...}, "text_questions": {...}, "all": {...}},
    "B_with_tables":    {"table_questions": {...}, "text_questions": {...}, "all": {...}},
    "delta_B_minus_A":  {"table_questions_accuracy": 0.0,
                         "text_questions_accuracy": 0.0,
                         "all_accuracy": 0.0,
                         "table_questions_retrieval_recall": 0.0}
  },
  "records_a_without_tables": [ /* 每题：retrieval_recall / answer_accuracy / correct /
                                   missing_in_context / missing_in_answer /
                                   n_table_in_topk / retrieved_pages / answer / ... */ ],
  "records_b_with_tables": [ /* 同上 */ ],
  "elapsed_seconds": 0.0
}
```

每档（`table_questions` / `text_questions` / `all`）内都含：

| 字段 | 含义 |
| --- | --- |
| `accuracy` | 要点全中的问题占比（工单「准确率」口径） |
| `retrieval_recall` | 检索上下文里答案要点的覆盖率 |
| `answer_accuracy` | 生成答案里答案要点的覆盖率 |
| `retrieval_accuracy` | 检索上下文要点全中的问题占比 |
| `avg_retrieve_ms` | 平均检索耗时 |
| `max_total_s` | 单问最大端到端耗时 |
| `avg_table_chunks_in_topk` | 返回片段中的表格块平均数量 |

---

## 6. `evaluation.{json,md}` —— 14 问 RAG 评估

```jsonc
{
  "meta": {"问题构成": {"表格类(id 1-4)": 4, "文本类(兴图新科)": 10, "合计": 14},
           "判分口径": "归一化关键词匹配；must 要点全中即判对"},
  "summary": {
    "keyword_accuracy": 0.0,        // 准确率（要点全中）
    "keyword_correct": 0,
    "keypoint_coverage": 0.0,       // 检索要点覆盖率
    "page_hit_rate": 0.0,           // 命中含答案页面的比例
    "hit_rate": 0.0, "mrr": 0.0,    // 检索层
    "latency_avg_s": 0.0,
    "latency_max_s": 0.0,
    "latency_under_3s_ratio": 0.0,  // 3 秒内完成占比（工单响应时间要求）
    "ragas": {                      // 有 LLM 时才有；否则为 null
      "faithfulness": 0.0, "answer_relevancy": 0.0,
      "context_precision": 0.0, "context_recall": 0.0,
      "answer_correctness": 0.0, "evaluator": "builtin"
    }
  },
  "retrieval_layer": {
    "keypoint_coverage": 0.0, "page_hit_rate": 0.0,
    "latency_avg_s": 0.0, "latency_max_s": 0.0, "latency_under_3s": 0.0,
    "per_question": [{"id": 1, "group": "table", "keypoint_coverage": 1.0,
                      "missing": [], "page_hit": true}],
    "hit_rate": 0.0, "mrr": 0.0, "recall_at_k": 0.0, "n": 14
  },
  "keyword_details": [{"id": 1, "正确": true, "命中": ["..."], "漏答": []}],
  "records": [/* rag_core.evaluate.EvalRecord.to_dict() 的完整明细 */]
}
```

---

## 7. 生成模型的两种情况

| 情况 | `answer_mode` | 影响 |
| --- | --- | --- |
| 已配置 `DEEPSEEK_API_KEY` | `rag` | 答案由大模型基于检索上下文生成，RAGAS 生成层指标可用 |
| 未配置 / 网络异常 | `extractive(...)` | 直接把命中的表格/正文原文回填为答案；检索层与关键词准确率指标不受影响，RAGAS 指标跳过并在报告中标注 |

无论哪种情况，**检索精确度、要点覆盖率、准确率（关键词口径）都是确定性计算**，
可复现、可对比。
