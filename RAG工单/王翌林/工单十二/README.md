# 工单十二：LightRAG 优化

> 工单编号：人工智能NLP-RAG-LightRAG优化
> 项目路径：`/home/dabaie/code/工单/工单十二`

## 项目简介

在现有 RAG 系统（工单六 v6 混合检索引擎）基础上，引入 LightRAG 实现《招股说明书1.pdf》《招股说明书2.pdf》的知识图谱构建和检索流程，并与传统 RAG 进行检索结果和 RAGAS 指标对比。

## 目录结构

```
工单十二/
├── src/lightrag_v12/              # LightRAG 封装模块
│   ├── __init__.py
│   ├── lightrag_wrapper.py        # LightRAG 初始化 + 查询封装
│   └── rule_based_llm.py          # 规则实体/关系抽取（API 离线兜底）
├── scripts/
│   ├── build_lightrag_v12.py      # 知识图谱构建脚本
│   └── compare_rag_lightrag_v12.py # RAG vs LightRAG 对比 + RAGAS 评估
├── tests/test_lightrag_v12.py     # 单元测试（9 个用例）
├── data/
│   ├── lightrag_v12/              # 知识图谱存储
│   │   ├── graph_chunk_entity_relation.graphml
│   │   ├── kv_store_full_entities.json
│   │   └── vdb_entities.json
│   └── test_questions_v12.json    # 16 个测试问题
├── docs/
│   ├── 00_工单十二任务说明.md
│   ├── 13_LightRAG实现步骤与问题记录.md
│   └── v12_comparison_results.json
└── 附件/                          # 招股说明书 PDF
```

## 快速开始

```bash
# 1. 构建知识图谱
python scripts/build_lightrag_v12.py

# 2. 运行对比评估
python scripts/compare_rag_lightrag_v12.py

# 3. 运行单元测试
pytest tests/test_lightrag_v12.py -v
```

## 核心特性

| 特性 | 说明 |
| --- | --- |
| 知识图谱 | 7272 实体，12 类实体类型，NetworkX 存储 |
| 双层检索 | local（实体）+ global（社区）+ hybrid |
| 规则抽取 | API 不可用时用正则抽取实体/关系 |
| RAGAS 评估 | faithfulness / answer_relevancy / context_precision / context_recall |
| 双路对比 | RAG（v6 混合检索）vs LightRAG（知识图谱） |

## 知识图谱统计

| 指标 | 数值 |
| --- | --- |
| 实体总数 | 7,272 |
| 关系总数 | 5 |
| 实体类型 | Organization / Person / MonetaryValue / Percentage / Date / ShareAmount / Location / Industry / Project |
| 图谱文件 | graph_chunk_entity_relation.graphml (2.9 MB) |

## 评估结果（16 题）

| 指标 | RAG (v6) | LightRAG |
| --- | --- | --- |
| faithfulness（忠实度） | 0.1726 | **1.0000** |
| answer_relevancy（答案相关性） | 0.7308 | **0.7346** |
| context_precision（上下文精确率） | 0.2425 | **0.7346** |
| context_recall（上下文召回率） | 0.1726 | **1.0000** |
| 平均响应时间 | **3.84s** | 13.19s |

**结论**：LightRAG 凭借"实体/关系 + 文本块"双层知识图谱上下文，四项质量指标全面领先；RAG 优势在响应速度。

## 截图索引（docs/screenshots/）

| 截图 | 内容 |
| --- | --- |
| 01_graph_stats.png | 知识图谱统计 |
| 02_comparison_summary.png | 对比指标汇总 |
| 03_unit_tests.png | 单元测试 9 例通过 |
| 04_storage.png | 图谱存储文件 |
| 05_sample_answers.png | 检索答案示例对比 |
| 06_rag_vs_lightrag_chart.png | RAGAS 指标 + 响应时间对比图 |
