# 基于 PDF 文档的问答系统

> 工单编号：**人工智能NLP-RAG-基于PDF文档的问答系统**
> 依据：《人工智能NLP-RAG项目-01-基于PDF文档的问答系统任务工单V1.1-20250206.pdf》
> 目标文档：《招股说明书1.pdf》（武汉兴图新科电子股份有限公司招股意向书，548 页）

一个针对 PDF 招股说明书的 RAG 问答系统：解析文字与表格 → 语义分块 → 向量 + BM25 混合检索 → 重排 → DeepSeek 生成，并提供 Vue 界面、知识库管理、RAG/纯 LLM 对比与 RAGAS 评估。

---

## 快速开始

```bat
:: 1. 安装依赖（本机已装齐，可跳过）
pip install -r requirements.txt

:: 2. 配置 DeepSeek Key
::    .env 含密钥、不入库，克隆下来的环境里没有这个文件，先从模板复制一份
copy .env.example .env
notepad .env

:: 3. 一键启动（自动拉起 Redis、Milvus、FastAPI 并打开浏览器）
start.bat
```

浏览器打开 <http://127.0.0.1:8000> 即可使用。

首次使用需先建立索引（二选一）：

```bat
:: 命令行建索引（推荐，可看到详细进度）
python scripts/build_index.py --pdf "data/uploads/招股说明书1.pdf"

:: 或直接在网页「知识库」标签页上传 PDF
```

---

## 目录结构

```
工单一/
├── app/                    后端
│   ├── main.py             FastAPI 入口：路由、SSE、上传、评估触发
│   ├── config.py           配置加载（.env + 环境变量）
│   ├── rag.py              PDF 解析（MinerU/降级）、语义分块、向量化
│   ├── store.py            Milvus 与 BM25 存储层、知识库管理
│   ├── llm.py              DeepSeek 调用、Redis 缓存、Query 理解
│   ├── retrieval.py        混合召回、RRF 融合、重排、生成
│   ├── feedback.py         点赞/点踩、响应时间指标
│   └── eval.py             RAGAS 评估与 RAG vs 纯 LLM 对比
├── scripts/build_index.py  离线建索引 CLI
├── scripts/export_reranker_onnx.py  导出并量化重排模型为 ONNX（可选旁路，一次性构建）
├── scripts/bench_rerank_onnx.py     对比 torch / ONNX 重排后端的耗时与排序一致性
├── tests/                  回归测试（python -m pytest tests -v）
├── frontend/index.html     Vue 3 单文件前端（问答/知识库/对比/评估）
├── deploy/docker-compose.yml  Milvus standalone 编排
├── eval/questions.json     工单 10 问 + 原文标准答案
├── docs/                   设计/技术/用户/测试/优化/部署 文档
├── data/                   uploads 上传、parsed 解析产物、index 索引、milvus 数据
├── requirements.txt
├── start.bat               Windows 一键启动
├── .env.example            配置模板（复制为 .env 后填 Key）
└── .env                    配置（含 DeepSeek Key，不入库）
```

---

## 技术栈

| 环节 | 选型 |
|---|---|
| 语言 / 框架 | Python 3.12 + FastAPI + Vue 3（CDN 单文件） |
| PDF 解析 | MinerU 3.4.5（主）；pymupdf + pdfplumber（降级） |
| 分块 | 语义分块（bge-m3 句向量 + 相邻句相似度断点） |
| 向量模型 | bge-m3（`D:/模型/bge-m3`，1024 维） |
| 向量库 | Milvus 2.6.2（Docker standalone）；可降级 milvus-lite |
| 关键词检索 | rank_bm25 |
| 重排 | bge-reranker-v2-m3（`D:/模型/bge-reranker-v2-m3`） |
| 大模型 | DeepSeek（`deepseek-v4-flash`） |
| 缓存 | Redis |
| 评估 | RAGAS 0.4.3 |

---

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（Milvus/Redis/模型/索引） |
| POST | `/api/kb/upload` | 上传 PDF 并异步建索引 |
| GET | `/api/kb/task/{id}` | 查询建索引/评估进度 |
| GET | `/api/kb/documents` | 已入库文档列表 |
| GET | `/api/kb/documents/{id}/chunks` | 查看解析后的分块（知识库管理） |
| DELETE | `/api/kb/documents/{id}` | 删除文档及全部索引 |
| POST | `/api/chat` | 问答（非流式） |
| POST | `/api/chat/stream` | 问答（SSE 流式） |
| POST | `/api/chat/llm_only` | 纯 LLM 对照组（不传上下文） |
| POST | `/api/feedback` | 点赞/点踩 |
| GET | `/api/feedback/stats` | 反馈统计 |
| POST | `/api/eval/run` | 运行工单 10 问 + RAGAS 评估 |
| GET | `/api/eval/result` | 取最近一次评估结果 |
| GET | `/api/metrics` | 响应时间 P50/P95/3 秒达标率 |

交互式文档：<http://127.0.0.1:8000/docs>

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/设计文档.md](docs/设计文档.md) | 架构、模块职责、数据流、接口、Schema、风险与降级 |
| [docs/技术文档.md](docs/技术文档.md) | 系统架构、技术选型、开发流程、使用方法 |
| [docs/用户手册.md](docs/用户手册.md) | 上传 PDF、提问、查看结果、反馈 |
| [docs/测试报告.md](docs/测试报告.md) | 解析验证、10 问逐题记录、RAG vs LLM、RAGAS 指标、响应时间 |
| [docs/优化记录.md](docs/优化记录.md) | 分块/top_k/阈值/缓存/并行检索的调优过程与前后对比 |
| [docs/部署文档.md](docs/部署文档.md) | 依赖安装、服务启动、模型加载、演示视频录制步骤 |

---

## 已知限制

- 响应时间口径已定稿：**指用户提问到 UI 渲染首字的时间（≤3s）**；完整答案时长单独统计（`/api/metrics` 的 `total_*` 字段）并在测试报告如实给出，不纳入验收。见 [docs/测试报告.md](docs/测试报告.md) §5.3。
- **响应时间 SLA 按可控性拆分**：检索 + 重排（本系统自控）可承诺 ≤1.5s，实测 P50 1.13~1.20s 且不随服务端波动；LLM 生成段（外部依赖）**不承诺**；端到端首字为**条件承诺**，须标注"测量于服务端正常时段"。见 §5.2.1。
- 对话不保留历史（工单未要求），因此跨轮指代（"它""该公司"）无法消歧。
- MinerU 首次解析 548 页约需 15–25 分钟，属一次性离线成本；重复执行会复用已有产物。
