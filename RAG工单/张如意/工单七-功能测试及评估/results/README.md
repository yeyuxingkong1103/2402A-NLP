# results 目录说明

> 工单编号：人工智能NLP-RAG-功能测试及评估

本目录存放 `src/` 下脚本的**运行产出**。
交付时本目录仅含本说明文件，实际结果文件需运行脚本生成——
这样可以避免出现「未经运行就写死的指标」。

---

## 一、文件清单与生成方式

| 文件 | 生成脚本 | 内容 |
|---|---|---|
| `corpus_stats.json` | `prepare_corpus.py` | 语料统计：每份年报的文档名（已还原）、页数、文本块数、表格数、解析耗时 |
| `test_cases.json` | `build_questions.py` | 10 题测试集（结构化） |
| `test_cases.md` | `build_questions.py` | 10 题测试集（人读表格） |
| `rag_test_results.json` | `run_rag_test.py` | 逐题检索结果 + 答案 + 耗时（结构化） |
| `rag_test_results.md` | `run_rag_test.py` | 同上（人读） |
| `evaluation.json` | `run_evaluation.py` | 评估指标：汇总 + 逐题明细 |
| `evaluation.md` | `run_evaluation.py` | 评估报告（人读，含判读结论） |
| `problem_analysis.md` | `analyze_problems.py` | 六类问题的自动统计与归类 |
| `serve.log` | `serve.py` | 服务运行日志（若启动过服务） |

---

## 二、字段说明

### `corpus_stats.json`

```json
{
  "docs": [
    {
      "file": "2020-02-14__...__000001__...pdf",   // 原始文件名（可能是乱码）
      "doc_id": "000001_2019",                      // 还原后的唯一标识
      "company": "平安银行",                        // 还原后的公司名
      "year": "2019",
      "report_type": "年度报告",
      "pages": 380,
      "text_blocks": 376,
      "table_blocks": 412,
      "parse_seconds": 42.5
    }
  ],
  "total_pages": 3300,
  "total_chunks": 12000,
  "decode_failures": []          // 还原失败的文件名列表（应为空）
}
```

> `doc_id` 是引用来源的唯一标识。`company` 还原失败时退化为股票代码。

### `test_cases.json`

```json
{
  "cases": [
    {
      "id": 1,
      "question": "xx银行在其2019年年报中提到的创新商业模式有哪些？",
      "type": "列表型",                    // 数值型 / 事实型 / 分析型 / 归纳型 / 列表型
      "reference_doc": "平安银行2019年年度报告",  // 应命中的年报（算 Hit Rate 用）
      "reference_pages": [12, 13],
      "ground_truth": "……",                // 参考答案（算 context_recall 用）
      "keywords": ["逾越者联盟", "咖啡零售", "出行预订"],  // 算准确率用
      "source": "sample_questions.pdf#S2"  // 来源：示例问题编号，或 "extended"
    }
  ]
}
```

### `rag_test_results.json`

```json
{
  "summary": {
    "n": 10, "hit": 8, "hit_rate": 0.8,
    "latency_avg": 2.1, "latency_p95": 3.2
  },
  "results": [
    {
      "id": 1,
      "question": "……",
      "retrieved": [
        {"doc": "平安银行2019年年度报告", "page": 12, "score": 0.83,
         "type": "text", "section": "第三节 管理层讨论与分析", "snippet": "……"}
      ],
      "answer": "……",
      "citations": [{"doc": "……", "page": 12}],
      "latency": 2.3,
      "stage_ms": {"vector_recall": 45, "fulltext_recall": 12, "fusion": 2,
                   "rerank": 820, "generate": 1420},
      "hit": true,
      "refused": false
    }
  ]
}
```

> `stage_ms` 是工单13 做性能瓶颈分析的数据来源。

### `evaluation.json`

```json
{
  "summary": {
    "n": 10, "evaluator": "builtin",
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_precision": 0.0,
    "context_recall": 0.0,
    "answer_correctness": 0.0,
    "hit_rate": 0.0, "mrr": 0.0, "recall_at_k": 0.0,
    "latency_avg": 0.0, "latency_p95": 0.0, "latency_max": 0.0
  },
  "records": [
    {
      "id": 1, "question": "……", "answer": "……", "ground_truth": "……",
      "contexts": ["片段1……", "片段2……"],
      "retrieved_docs": ["平安银行2019年年度报告"],
      "retrieved_pages": [12],
      "latency": 2.3,
      "faithfulness": 0.0, "answer_relevancy": 0.0,
      "context_precision": 0.0, "context_recall": 0.0
    }
  ]
}
```

> 指标为 `null` 表示该题无参考答案（`context_recall` / `answer_correctness` 无法计算），
> 已从均值中剔除，不会拉低总体指标。

---

## 三、如何解读

| 想了解 | 看哪个文件 | 看什么 |
|---|---|---|
| 系统答得对不对 | `rag_test_results.md` | 逐题答案 + 引用来源 |
| 检索准不准 | `evaluation.md` | `hit_rate` / `mrr` / `context_precision` |
| 答案有没有编造 | `evaluation.md` | `faithfulness`，低于 0.7 需警惕 |
| 慢在哪里 | `rag_test_results.json` | 每题的 `stage_ms` 分解 |
| 系统哪里不行 | `problem_analysis.md` | 六类问题的统计与案例 |

---

## 四、重新生成

```bash
cd 工单07-功能测试及评估/src
python prepare_corpus.py --force      # --force 强制重新解析，忽略缓存
python build_questions.py
python run_rag_test.py
python run_evaluation.py
python analyze_problems.py
```

**提示**：`rag_core.llm` 带磁盘缓存（`data/cache/llm/`）。
重复运行同一批问题不会重复调用 API，也不会重复计费。
若要强制重新调用（例如换了 Prompt 想重新评估），清空该目录即可。
