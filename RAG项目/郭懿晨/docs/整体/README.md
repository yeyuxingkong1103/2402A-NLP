# RAG 知识问答系统 · 完整文档链（v2）

> **项目**：基于 GB/T 44653-2024《六氟化硫气体现场循环再利用导则》的**本地 RAG PDF 问答系统**（v1 MVP 基线 → v2 检索重排优化，贯穿 专高六 W1-W2 教学主线）。
> **文档定位**：从**需求分析 → 设计 → 实现 → 验证 → 部署**的完整流程文档链。各环节给「决策 + 关键点」，详细方案/报告按环节索引（不重复堆内容）。

---

## 一、文档地图（从需求到实现）

| # | 文档 | 环节 | 说明 |
|---|---|---|---|
| 01 | `01-需求分析.md` | 需求 | 功能/非功能需求（FR1-18）、目标、范围边界、验收标准（AC，Given/When/Then） |
| 02 | `02-总体架构.md` | 架构 | 分层/模块/技术栈/数据流（含 v2 重排链路） |
| 03 | `03-数据管线设计.md` | 数据 | 解析→清洗→分块→向量化→入库 |
| 04 | `04-检索与问答设计.md` | 检索 | **v2：RRF 召回 → 粗排 → 精排 → 阈值过滤** + Ollama 生成 |
| 05 | `05-后端服务设计.md` | 后端 | 上传/任务/文档/向量库/问答接口 + 轮询任务 |
| 06 | `06-接口文档.md` | 接口 | API 全清单 |
| 07 | `07-测试与验收.md` | 测试 | 评测集（43 题）+ v1/v2 对比结果 + pytest |
| 08 | `08-部署与运行.md` | 部署 | 环境/依赖（含 v2 模型与补丁）/启动/端口 |
| 09 | `09-构建服务运行手册.md` | 运行 | 启动/上传/任务/前端/常见错误/验证 |
| 10 | `10-离线到在线工程化复盘.md` | 工程 | 服务化要点/v2 重排踩坑/checklist |

## 二、版本迭代

| 版本 | 日期 | 类型 | 改动摘要 |
|---|---|---|---|
| v1 | 2026-08-30 | MVP 基线 | 本地 RAG 问答闭环：上传、MinerU 解析、分块、bge-m3 向量化、Qdrant 入库、RRF 复合检索、Ollama 问答、页码引用 |
| v2 | 2026-09-01 | 检索优化 | RRF 召回后新增「粗排 + 精排」两阶段重排；候选规模/各层 top_k/阈值/打分方式可配置；`rerank_enabled` 开关保留 v1 行为；评测支持 v1/v2 对比 recall/MRR/拒答率/延迟 |
| v2 迭代 | 2026-09-02 | 评测口径收严 | 三项主指标 recall@k/MRR/refusal_accuracy + 生成集 coverage 补充；三套集全跑 43 题（检索 17 + 生成 6 + 安全 20）；`final_top_k` 调为 4（recall@4）；评测**直接对当前真实检索库打分，不再 reset/rebuild**；v2 达标（recall +0.083 / MRR +0.111 / 拒答 0→0.92，延迟 5.60s→4.29s） |

## 三、已有详细文档（按环节索引）

| 环节 | 详细文档位置 |
|---|---|
| 需求 | `docs/需求说明.md`（唯一需求文档，v2 版本头） |
| 规格/计划 | `docs/v2/spec.md`、`docs/v2/plan.md`、`docs/v2/tasks.md`、`docs/v2/checklist.md` |
| v2 设计 | `docs/superpowers/specs/2026-09-01-v2-retrieval-rerank-design.md` |
| 架构 | `docs/架构/架构图-v2.md`、`docs/架构/02-总体架构.md` |
| 接口 | `docs/接口文档.md` |
| 版本 | `docs/版本迭代.md` |
| 评测 | `eval/sets/`（检索 17 / 生成 6 / 安全 20）、`eval/results/`（指标 JSON）、`eval/baseline/`（v1 基线） |
| 指标对比 | `docs/指标对比/优化指标对比.md`（v1/v2 记录） |

## 四、技术栈

- **解析**：MinerU 3.4.5（pipeline，本地模型，需 transformers 5.x + `patch_mineru_tf5.py` 补丁）；失败自动回退 `pypdf`
- **VLM 后处理**：`qwen-vl-plus`（阿里云 DashScope OpenAI 兼容接口；`.env` 已启用 `VLM_ENABLED=true`，代码默认关闭）
- **嵌入**：BGE-M3 本地模型（dense 1024 + sparse）
- **重排（v2）**：粗排复用 BGE-M3 dense 余弦；精排本地 `bge-reranker-v2-m3`（FlagEmbedding 1.4.2）
- **存储**：**Qdrant embedded**（`data/qdrant/`，named vectors: dense + sparse）+ `data/state.json`
- **后端**：FastAPI + BackgroundTasks（异步构建）+ 轮询
- **前端**：原生 HTML/CSS/JS 单页（`frontend/`，无框架）
- **LLM**：Ollama 本地 `deepseek-r1:7b`（问答生成）
- **评测**：page 级 recall@k / MRR / refusal_accuracy / 检索延迟 + 生成覆盖（默认 43 题）