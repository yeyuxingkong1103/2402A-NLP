# 工单七 · RAG 功能测试及评估

> **工单编号：人工智能NLP-RAG-功能测试及评估**
> 语料：`ccf_competition`（9 份 A 股上市公司 2019–2021 年报 PDF）
> 题集：《sample_questions.pdf》原题 4 道 + 基于语料原文仿写 6 道 = **10 题**

对已实现的 RAG（01–06）做**功能测试与检索评估**：建库 → 10 题跑通链路 → RAGAS 四指标 → 检索问题分析 → 优化方案 → FastAPI + Vue 演示。

---

## 快速开始

```bash
# 环境：本机 Milvus(19530) 与 Redis(6379) 已在运行，不要再执行 docker compose
cd 工单七

# 1) 建索引（解析 9 份 PDF → 语义分块 → Milvus + BM25；约 20 分钟）
D:/myconda1/python.exe scripts/build_index.py

# 2) 跑评估（10 题 + RAGAS 四指标；约 15–25 分钟）
D:/myconda1/python.exe scripts/run_eval.py
# 只想先看检索效果：D:/myconda1/python.exe scripts/run_eval.py 10 --no-ragas

# 3) 自检（解析降级链 / 融合 / 缓存指纹 / RAGAS 样本构造；秒级）
D:/myconda1/python.exe tests/test_pipeline.py

# 4) 启动演示（FastAPI + Vue，浏览器打开 http://127.0.0.1:8010）
scripts/start.bat
```

---

## 目录

| 路径 | 说明 |
|---|---|
| `app/config.py` | 全部配置（路径/阈值/模型/集合名），唯一配置源 |
| `app/ingest.py` | 解析（MinerU → pymupdf → 附件 txt 四级降级）+ 语义分块 + 写 Milvus/BM25 |
| `app/rag_core.py` | 向量 + BM25 → 通道内归一化加权融合 → bge-reranker-v2-m3 → DeepSeek → Redis 缓存 |
| `app/evaluate.py` | 10 题作答（先落盘）+ RAGAS 四指标（逐指标串行、nan 记 None） |
| `app/main.py` | FastAPI 路由层 |
| `frontend/index.html` | Vue 3 演示页（本地 `vue.global.prod.js`，不走 CDN） |
| `tests/questions.json` | 10 题 + ground_truth（逐字取自 PDF 原文） |
| `tests/test_pipeline.py` | 自检脚本（正例全过 + 反例全拦） |
| `docs/01-design.md` ~ `05-deploy.md` | 五步交付文档 |
| `results/` | `raw_*.json`（作答原始）、`eval_*.json`（含指标） |

---

## 五步文档

| 步骤 | 文档 |
|---|---|
| 第一步 设计 | [`docs/01-design.md`](docs/01-design.md) |
| 第二步 研发 | [`docs/02-dev.md`](docs/02-dev.md) |
| 第三步 测试 | [`docs/03-test.md`](docs/03-test.md) |
| 第四步 优化 | [`docs/04-optimize.md`](docs/04-optimize.md) |
| 第五步 部署 | [`docs/05-deploy.md`](docs/05-deploy.md) |
| 交付与验收对照 | [`docs/06-delivery.md`](docs/06-delivery.md) |

---

## 环境要点（踩过的坑，别再踩）

1. **解释器用 `D:/myconda1/python.exe`**（Python 3.12.7）：唯一同时具备 CUDA torch、MinerU、ragas 的解释器。
2. **`.env` 路径一律正斜杠**：`D:\模型\bge-m3` 里的 `\b` 会被解析成退格符。
3. **Redis 必须 `protocol=2`**：本机 Redis 5.0.14 不支持 RESP3，redis-py 8 用默认协议会连接失败。
4. **Milvus 查询用 `consistency_level="Strong"`**：默认 Bounded 会导致刚写入的数据读不到。
5. **`MINERU_ENABLED=0` 可关闭主解析路径**：MinerU 对 9 份年报（共 2812 页）耗时过长，批量建索引时用 pymupdf 快速通道；MinerU 主路径的可用性另行冒烟验证。
6. **RAGAS 并发必须压到 2**：默认 16 会打满 DeepSeek 限流，把样本静默打成 `nan`；四指标也不能混跑。
7. **文件名中文已损坏**：`ccf_competition/pdf/` 下的中文文件名在磁盘上是 `U+FFFD`，不可逆；唯一稳定标识是 6 位股票代码，用 `config.CODE2NAME` 回认公司名。
8. **8000 端口被其它服务占用**，本工单用 **8010**。
