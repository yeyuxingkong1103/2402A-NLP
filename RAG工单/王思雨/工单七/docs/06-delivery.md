# 工单七 · 交付与验收对照

> 工单编号：**人工智能NLP-RAG-功能测试及评估**
> 一页纸回答："工单要什么 → 交付在哪 → 实测状态如何"。详细论证见 01–05 各文档。

---

## 1. 产出物对照（工单要求 → 交付物 → 实测状态）

| 工单要求 | 交付物 | 实测状态 |
|---|---|---|
| 10 个问题及其对应的 RAG 检索结果 | `tests/questions.json` + `results/raw_*.json` + `results/eval_*.json` | ✅ 10 题（sample 原题 4 + 仿写 6），每题 top-8 片段带公司/页码/通道/重排分 |
| RAG 评估结果 | RAGAS 四指标，`results/eval_20261009_194604.json`（基线）/ `_204719.json`（优化） | ✅ 四指标四轮均 10/10 有效、无 nan |
| 分析检索结果存在的问题 | `docs/03-test.md` §4 + `docs/04-optimize.md` §3 | ✅ 5 类问题，逐条带证据与**归因**（系统的账 / 题面的账 / 造题的账 / 指标口径的账） |
| 演示（FastAPI + Vue） | `app/main.py` + `frontend/index.html` | ✅ 已端到端跑通（见 §3） |
| 10 题体现 `ccf_competition` 内容相关性 | ground_truth 逐字取自对应 PDF 原文并核对页码 | ✅ 无任何语料外知识 |
| Python ≤ 300 行且逐行中文注释 | `app/` 核心 4 模块 | ✅ **36+99+97+68 = 300 行**（逐文件核过），行尾注释 |

## 2. 硬性技术选型落地

| 要求 | 实现 | 备注 |
|---|---|---|
| Python 3.11 | `D:/myconda1/python.exe` = **Python 3.12.7** | 唯一同时有 CUDA torch + MinerU + ragas 的解释器；差异已标注 |
| FastAPI + Vue | `app/main.py`(:8010) + `frontend/index.html`(Vue3 本地构建) | 同源托管，无 npm/无 CDN |
| MinerU + 降级 | 四级链：MinerU → pymupdf → 附件 txt | MinerU 主路径已冒烟验证（`results/mineru_smoke.json`，5.3 s/页） |
| 语义分块 | `app/ingest.py::sem_chunk`（句向量 + 语义漂移切块） | — |
| bge-m3 | `D:/模型/bge-m3`，1024 维，归一化 COSINE | — |
| Milvus | `localhost:19530`，集合 `gd7_chunks`（独立，不污染 `gd6_chunks`） | 5660 块 |
| BM25 | `rank_bm25.BM25Okapi` + jieba，`data/bm25.pkl` | — |
| bge-reranker-v2-m3 | `D:/模型/bge-reranker-v2-m3`，FlagReranker + fp16 | 输入截断 512 字符 |
| DeepSeek | `DEEPSEEK_API_KEY` 环境变量/`.env`，`deepseek-chat`，温度 0 | Key 不进代码 |
| Redis | `localhost:6379/0`，**`protocol=2`** | Redis 5 不支持 RESP3 |
| RAGAS | ragas 0.4.3，四指标**逐指标串行**、`max_workers=2` | 并发过高会把样本静默打成 nan |

## 3. 端到端实测证据（不是"应该能跑"）

| 验证 | 命令 | 结果 |
|---|---|---|
| 自检 | `tests/test_pipeline.py` | **25/25 PASS，退出码 0** |
| 健康检查 | `GET /health` | `{milvus: up, redis: up, llm: configured, chunks: 5660}` |
| 检索 | `POST /retrieve`（Q06） | top-1 = 邮储银行 p.61，重排分 **1.000**，通道 `vec+bm25` |
| 生成 | `POST /ask`（Q06） | 「绿色贷款余额 **2,433.01 亿元**，较上年末增长 **27.78%**」——与 ground_truth 逐字一致 |

## 4. 关键实测数字（全部来自本机实跑）

| 指标 | 值 | 结论 |
|---|---|---|
| 语料规模 | 9 份年报 / 2812 页 / 5660 块 | — |
| 基线 RAGAS（194604） | F 0.9167｜AR 0.7812｜CR 0.7467｜CP 0.7848 | 人工复核 8/10 完全正确 |
| **噪声底** | **0.027**（`workers=2` 逐指标串行，同输入重打分） | Δ≤0.027 一律判"测不出变化" |
| 页眉页脚过滤 | 模板噪声 **28.7% → 6.2%** | 确定性有效；**RAGAS 测不出改善**（原因见 §1.4） |
| `RERANK_TOP_N=6` | 占比 0.653→0.685，token −25% | **建议上线** |
| 融合权重 0.3/0.7~0.85/0.15 | 首条重排分恒 **0.868** | **不是有效杠杆**（重排把次序差异抹平） |
| 实体约束检索 | 触发 6/10 题，占比 **0.708→1.000** | 题面含糊的 4 题**不触发** |

## 5. 边界与未做项（如实标注）

1. **题面 `xx银行` 指代不明**（Q01/Q02）：工单要求保留原题问法，故不改题面；正确处理属"查询澄清"，超出本工单边界。
2. **Q03/Q04 的 Context Recall = 0.0 是造题的账**（reference 写成综述体），不是检索失败——已标注，未改。
3. **方案④（改写 reference）仍为预期、未实测**。
4. **未扫描分块参数**（需重建索引，超出本轮预算）；**未做全量回归**（所有新结论都是检索端确定性结论，未改线上配置）。
5. 详细归因见 `docs/04-optimize.md` §3；每项都写了**前提**与**前提失效时的处置**。

## 6. 环境复活（Docker 重启后必做）

```bash
docker start rag-etcd rag-minio rag-milvus   # 三个容器会随 Docker 重启集体 Exited，数据不丢
docker ps --filter name=rag-                 # 确认 healthy 再继续
```
⚠ 不要 `docker compose up`（端口已被既有容器占用）。

---

## 7. 五步文档索引

| 步骤 | 文档 | 内容 |
|---|---|---|
| 一 设计 | [`01-design.md`](01-design.md) | 架构、目录、评估流程、行数口径 |
| 二 研发 | [`02-dev.md`](02-dev.md) | 各模块实现要点、300 行口径 |
| 三 测试 | [`03-test.md`](03-test.md) | 基线轮完整数据、检索问题分析、自检 |
| 四 优化 | [`04-optimize.md`](04-optimize.md) | 噪声底、已实施优化、参数扫描、优化方案与边界 |
| 五 部署 | [`05-deploy.md`](05-deploy.md) | 启动命令、接口清单、演示视频分镜、故障排查 |
