# 混合检索策略（向量 + 全文 + 混合）—— 工单六

> 工单编号：人工智能NLP-RAG-混合检索任务

本项目基于工单五（多轮对话）复制而来，在 `/home/dabaie/code/工单/工单六` 目录下进行混合检索策略开发，与 v3/v4/v5 服务共存、互不影响。

## 项目概述

- **基础版本**：工单五（多轮对话）+ 工单四（图文表三路融合 RAG，单轮准确率 100%）
- **新增能力**：向量检索（召回+重排）/ 全文检索（倒排索引+布尔/短语/模糊+多字段）/ 混合检索（RRF 投票、加权平均、权重可调）三类策略的配置及应用
- **三种重排器**：基于 LLM 的重排器（bge-reranker-v2-m3）、基于 TF-IDF 的重排器、基于用户反馈的自适应重排器
- **验收目标**：准确率 ≥ 90%、召回率 ≥ 95%、响应时间 ≤ 3 秒、中英文问答、多轮对话 + 用户反馈、高并发稳定

## 核心模块（工单六新增）

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

## 快速开始

```bash
# 启动（端口 8006 API / 8506 UI）
bash scripts/start_v6.sh

# 停止
bash scripts/stop_v6.sh

# 工单六单元测试（40 项）
python -m pytest tests/test_fulltext_retriever.py tests/test_fusion_rerankers.py \
                 tests/test_hybrid_retriever_v6.py tests/test_api_v6.py -v

# 全量回归（含工单一~五全部用例）
python -m pytest tests/ -q

# 三策略对比评估（结果写入 docs/eval_v6_results.json）
python scripts/evaluate_v6.py
```

## 检索策略配置

API 请求体 `retrieval` 字段（均可空，默认 hybrid + rrf + llm）：

| 字段 | 取值 |
|------|------|
| mode | vector / fulltext / hybrid |
| fusion | rrf（投票机制）/ weighted（加权平均） |
| reranker | llm / tfidf / adaptive |
| vector_weight / fulltext_weight | 通道权重（自动归一化） |
| match | and / or / phrase / fuzzy |
| fields | title / content / summary |
| top_k | 返回条数 |

预设：`vector_only`、`fulltext_only`、`hybrid_rrf_llm`（默认）、`hybrid_weighted_adaptive`，
可通过 `GET /api/v6/config` 查询。

## 文档索引

| 文档 | 说明 |
|------|------|
| docs/00_工单六任务说明.md | 工单六需求、验收标准、技术方案 |
| docs/07_混合检索测试报告.md | 单元测试 + 三策略对比评估结果 |
| docs/09_混合检索部署文档.md | 8006/8506 部署、接口清单、FAQ |
