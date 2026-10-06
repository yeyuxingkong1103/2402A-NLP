# results 目录说明

> 工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

本目录存放 `src/` 脚本的运行产出。交付时仅含本说明文件，实际结果需运行脚本生成。

---

## 一、文件清单

| 文件 | 生成脚本 | 内容 |
|---|---|---|
| `corpus_stats.json` | `prepare_corpus.py` | 9 份年报的解析统计（含 GBK 文件名还原结果） |
| `graph_stats.json` | `build_graph.py` | 图谱统计：实体数/关系数/类型分布/社区数 |
| `graph/knowledge_graph.png` | `visualize_graph.py` | 知识图谱静态图 |
| `graph/knowledge_graph.html` | `visualize_graph.py` | **交互式图谱**（验收标准 1 的核心证据） |
| `extraction_examples.md` | `visualize_graph.py` | 抽取示例：原文 → 抽取结果对照（回答演示视频要求） |
| **`graph_qa_results.json`** | `graph_qa.py` | 10 题的 Graph RAG 问答结果（子图 + 答案 + 耗时） |
| **`graph_qa_results.md`** | `graph_qa.py` | 同上（人读） |
| **`compare_wo07.json`** | `compare_with_wo07.py` | 与工单07 的逐题对比数据 |
| **`compare_wo07.md`** | `compare_with_wo07.py` | 同上（人读）+ RAGAS 指标对比 |
| `financial_report.md` | `research_report.py` | 基于图谱生成的金融研报 |

---

## 二、核心文件字段说明

### `graph_stats.json`

```json
{
  "n_entities": 0,
  "n_relations": 0,
  "n_communities": 0,
  "entity_types": {"公司": 0, "财务指标": 0, "风险类型": 0, "其他": 0},
  "relation_types": {"指标数值": 0, "同比变化": 0, "控股股东": 0},
  "top_entities": [{"name": "平安银行股份有限公司", "degree": 0, "type": "公司"}],
  "source_chunks": 0,
  "extraction_profile": "optimized"
}
```

> `entity_types` 中「其他」占比应尽可能低——占比高说明类型体系覆盖不足。
> `top_entities` 按度数排序，通常是核心主体公司。

### `graph_qa_results.json`

```json
{
  "n": 10,
  "results": [
    {
      "id": 1,
      "question": "……",
      "mode": "hybrid",
      "seeds": ["平安银行股份有限公司"],
      "subgraph": {
        "n_nodes": 0, "n_edges": 0,
        "entities": [{"name": "……", "type": "……", "description": "……"}],
        "relations": [{"source": "……", "target": "……", "type": "……"}]
      },
      "answer": "……",
      "latency": 2.3,
      "trace": {"local": {"n_nodes": 0}, "global": {"n_communities": 0}}
    }
  ]
}
```

> `subgraph` 字段是「解析出来的知识图谱结构」——
> 工单明确要求「输出检索结果、以及解析出来的知识图谱结构」。

### `compare_wo07.json`

```json
{
  "per_question": [
    {
      "id": 1, "question": "……",
      "wo07_answer": "……", "wo08_answer": "……",
      "wo07_hits": ["平安银行2019年年度报告"],
      "wo08_seeds": ["平安银行股份有限公司"],
      "winner": "wo08", "diff_note": "……"
    }
  ],
  "ragas": {
    "wo07": {"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0},
    "wo08": {"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0}
  },
  "wo07_source": "工单07-功能测试及评估/results/rag_test_results.json"
}
```

> `wo07_source` 记录数据来源。若工单07 的结果文件不存在，
> 脚本会给出友好提示并只输出本工单结果。

---

## 三、如何解读

| 想了解 | 看哪个文件 | 看什么 |
|---|---|---|
| 图谱规模与质量 | `graph_stats.json` | 实体/关系数、「其他」类型占比 |
| 实体关系是怎么抽的 | `extraction_examples.md` | 原文与抽取结果的对照 |
| 图谱长什么样 | `graph/knowledge_graph.html` | 交互式浏览 |
| 图谱问答效果 | `graph_qa_results.md` | 每题的检索子图与答案 |
| 相比工单07 有没有提升 | `compare_wo07.md` | 逐题对比 + RAGAS 指标 |
| 图谱能生成什么研报 | `financial_report.md` | 基于社区摘要的归纳性输出 |

---

## 四、重新生成

```bash
cd 工单08-GraphRAG金融问答/src

python prepare_corpus.py                    # 语料准备
python build_graph.py                       # 建图谱（约 40~70 min，有增量缓存）
python visualize_graph.py                   # 可视化 + 抽取示例
python graph_qa.py                          # 图谱问答
python compare_with_wo07.py                 # 与工单07 对比
python research_report.py                   # 金融研报生成
```

**提示**：
- `build_graph.py` 的抽取结果按 chunk 增量缓存在 `data/graph/extractions_*.jsonl`，
  中断后重跑同一命令即可续跑。除非改了抽取 prompt，否则不要删除该缓存。
- 若想先快速验证流程，可用 `--limit 100` 只处理前 100 个 chunk。

---

## 五、说明

`eval_question.md` 的位置：附件中**未提供**该文件，
本工单依据工单07 的测试集在 `questions/eval_question.md` 构建了等价的问题集。
`compare_with_wo07.py` 会从 `工单07-功能测试及评估/results/` 读取结果做对比。
