# 功能测试及评估任务 —— 工单七

> 工单编号：人工智能NLP-RAG-功能测试及评估

本项目基于工单六（混合检索）复制而来，位于 `/home/dabaie/code/工单/工单七`。工单七不新增 RAG 能力，而是**使用 01-06 工单实现的 RAG 系统**（RAGEngineV6）对附件 `ccf_competition.zip` 中的 9 份 A 股年报做功能测试与检索评估：10 个问题（基于 `sample_questions.pdf` 整理）→ RAG 检索结果 → 评估框架打分 → 检索问题分析。

## 工单七新增内容

```
data/ccf_reports/
├── ccf_corpus.json            # 9 份年报语料注册表（doc_id/公司/年份/路径）
├── *_年报.pdf                  # 解压并规范命名的年报（GBK 文件名已还原）
├── chunks/*_chunks.json       # 各年报分块落盘
└── test_questions_v7.json     # 10 个测试题及金标（关键词/参考答案要点）
scripts/
├── prepare_ccf_reports.py     # 解压→PyMuPDF抽文本→既有chunker→bge-m3入rag_chunks（幂等）
└── evaluate_v7.py             # warmup→10题RAG(hybrid)→评估→docs/eval_v7_results.json
src/evaluation_v7.py           # 评估框架（纯函数）：doc_recall@5/MRR/上下文召回/答案准确率/问题归类
tests/test_evaluation_v7.py    # 评估框架单元测试（9 用例，不依赖 Milvus）
docs/00_工单七任务说明.md       # 工单七需求、验收标准、实施方案
docs/08_功能测试评估报告.md     # 10 题检索结果、评估结果、问题分析
```

## 快速开始

```bash
# 1) 语料准备（解压 + 分块 + 嵌入入库，CPU 约 30~40 分钟；已入库可重复运行，按 doc_id 幂等）
python scripts/prepare_ccf_reports.py            # 全流程
python scripts/prepare_ccf_reports.py --skip-extract   # 跳过解压
python scripts/prepare_ccf_reports.py --skip-ingest    # 仅解压+注册表

# 2) 工单七评估框架单元测试
python -m pytest tests/test_evaluation_v7.py -v

# 3) 10 题 RAG 测试与检索评估（结果写入 docs/eval_v7_results.json）
python scripts/evaluate_v7.py
python scripts/evaluate_v7.py --strategy vector     # 也可换单策略对比
```

## 评估指标

| 指标 | 含义 |
|------|------|
| doc_recall@5 | top5 中金标文档覆盖率（跨文档题按命中文档比例） |
| MRR | 首个金标文档排名倒数 |
| context_recall | 金标关键词在检索上下文中的覆盖率 |
| answer_acc | 金标关键词在 RAG 答案中的全命中率（答案正确性代理指标） |
| issues | 自动归类：金标文档漏检/排序靠后/跨公司串档/上下文缺口/答案漏点/超时(>3s) |

## 文档索引（工单七）

| 文档 | 说明 |
|------|------|
| docs/00_工单七任务说明.md | 工单七需求、验收标准、实施方案 |
| docs/08_功能测试评估报告.md | 10 题检索结果 + 评估结果 + 检索问题分析 |
| docs/eval_v7_results.json | 10 题完整检索结果/答案/指标原始数据 |

---

# 附：工单六（混合检索，本工单被测试的 RAG 系统）

> 工单编号：人工智能NLP-RAG-混合检索任务

## 项目概述

- **基础版本**：工单五（多轮对话）+ 工单四（图文表三路融合 RAG，单轮准确率 100%）
- **能力**：向量检索（召回+重排）/ 全文检索（倒排索引+布尔/短语/模糊+多字段）/ 混合检索（RRF 投票、加权平均、权重可调）三类策略可配置
- **三种重排器**：基于 LLM 的重排器（bge-reranker-v2-m3）、基于 TF-IDF 的重排器、基于用户反馈的自适应重排器
- **验收基线**：准确率 93.8%、召回 96.9%、平均响应 2206ms；全量 265 passed / 1 skipped

## 工单六核心模块

```
src/retrieval/
├── retrieval_config.py      # 检索策略配置中心（mode/fusion/reranker/weights/match/fields + 预设）
├── fulltext_retriever.py    # 全文检索：三字段倒排索引 + 布尔 AND/OR/NOT + 短语 + 模糊(bigram)
├── rerankers.py             # 三种重排器：LLM / TF-IDF / Adaptive(用户反馈)
├── fusion.py                # 融合：RRF 投票机制 / 加权平均(min-max 归一化)
└── hybrid_retriever_v6.py   # 统一混合检索器（三模式编排，两路并行召回）
src/rag_engine_v6.py         # v6 引擎：文本走可配混合检索，表格/图像沿用 v4
src/api_v6.py / schemas_v6.py  # FastAPI v6（8006）
app/streamlit_app_v6.py      # 策略配置 + 多轮对话界面（8506）
scripts/evaluate_v6.py       # vector/fulltext/hybrid 三策略 × 16 题对比评估
```

## 工单六服务（工单七评估直接实例化引擎，服务可选）

```bash
bash scripts/start_v6.sh   # 端口 8006 API / 8506 UI
bash scripts/stop_v6.sh
```

检索策略配置（API `retrieval` 字段，默认 hybrid + rrf + llm）：

| 字段 | 取值 |
|------|------|
| mode | vector / fulltext / hybrid |
| fusion | rrf（投票机制）/ weighted（加权平均） |
| reranker | llm / tfidf / adaptive |
| vector_weight / fulltext_weight | 通道权重（自动归一化） |
| match | and / or / phrase / fuzzy |
| fields | title / content / summary |
| top_k | 返回条数 |

工单六文档：[docs/00_工单六任务说明.md](docs/00_工单六任务说明.md) ｜ [docs/07_混合检索测试报告.md](docs/07_混合检索测试报告.md) ｜ [docs/09_混合检索部署文档.md](docs/09_混合检索部署文档.md)
