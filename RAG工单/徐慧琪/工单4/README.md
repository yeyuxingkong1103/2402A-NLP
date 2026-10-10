# 工单04：PDF 文档的图像内容解析及检索优化（三模态 RAG）

> **工单编号**：人工智能NLP-RAG-图像内容解析及检索优化
> **版本**：V1.1 ｜ **工时预估**：2 人日
> **语料**：《招股说明书1.pdf》（武汉兴图新科电子股份有限公司）、《招股说明书2.pdf》（武汉力源信息技术股份有限公司）
> **硬性要求验收**：逐条对照见 `docs/验收对照表.md`

## 一、项目简介

在工单 01/02/03（PDF 问答 → 系统优化 → 表格解析及检索优化）基础上，本工单把系统从单文档升级为**多文档**，
并新增**图像语义解析与图像检索**：16 道题中的 id 5（组织结构图）、id 6（IC 市场增长图）答案**既不在正文、也不在表格里**，
只存在于 PDF 插图中。

系统为三模态（文本 / 表格 / 图像）中文招股书 RAG，全链路：

```
PDF ─► 解析层（文本块 │ 表格块 │ 图区检测三层算法 → 高 DPI 渲染）
     ─► 分块层（文本语义分块 / 整表成块 / 图像 = VLM 结构化描述 + CLIP 向量）
     ─► 索引层（Qdrant 三 collection：text_chunks / table_chunks / image_chunks）
     ─► 检索层（dense bge-m3 ∥ BM25 ∥ CLIP 文→图 → RRF 融合 → bge-reranker-base 重排 top-5）
     ─► 生成层（DeepSeek 主 + Ollama qwen2.5:3b 兜底；按提问语言作答 + 引用溯源 + 拒答）
     ─► 答案 + 引用来源（文档/页码/类型/来源ID/相似度）；精确度由 16 题评估器给出
```

- **四个入口**：FastAPI 服务（`src/rag04/api/`，压测与集成入口）、Streamlit 中英双语界面（`src/rag04/ui/`）、
  16 题评估器（`src/rag04/eval/`）、并发压测脚本（`benchmarks/loadtest.py`）。
- **双模式**：`baseline_03`（代表工单 01/02/03 能力）与 `full_04`（本工单），**同一套代码、同一批 16 题、同一份语料**，
  由 `config.py` 的开关覆写，命令行首参或环境变量 `RAG04_MODE` 选择；机制见 `docs/技术文档.md` §六。
- **图像双通道**：CLIP 负责「找到图」（跨模态以文搜图），多模态大模型负责「读懂图」（结构化描述进文本通道）。

## 二、质量现状（如实记录，不粉饰）

| 硬指标 | 目标 | 实测（full_04，16 题） | 结论 |
| --- | --- | --- | --- |
| 答案准确率 | ≥ 90% | 0.875（14/16；id 1 拒答、id 3 覆盖率 0.75） | **未达成** |
| 端到端响应 | ≤ 3 s | 热态 p50 = **1730.3 ms**、15/16 题 ≤ 3 s；首题冷启动 41672.9 ms、mean = 4245.2 ms | **热态达标；冷启动与并发档未达标** |
| 检索 Hit Rate@5 | — | 0.9375（15/16 题至少召回 1 个金标页） | 已测 |
| 检索 recall@5 / mrr / ndcg@5 | — | 0.6979 / 0.7396 / 0.6422 | 已测 |

数据源：`docs/reports/检索精确度报告_full_04.md`（16 题中文，2026-10-07 15:10 运行；
那之前的 p50≈3.9 s 是 `ollama_url` 的 `localhost`→IPv4 修复前的口径，见 `src/rag04/config.py` 注释）。
服务与界面按单次实测如实上报 `within_budget`，超预算不截断、不缓存、不重试。
两道残留卡点（id 1 由对转错的回归归因、id 3 同一事实不同表述召回不足）的证据链见
`docs/图像解析与检索专项说明.md` §六 与 `.superpowers/sdd/task-rc2-report.md` §6。

**Task 18 / Task 21 产物（已回填）：**

- 双模式对比 `docs/reports/优化前后对比分析.md`：答案准确率 0.125 → 0.875、hit_rate 0.5 → 0.9375
  （两侧时延非同一时点测量，口径见该报告「注 2」）。
- RAGAS `docs/reports/RAGAS评估报告.md`：faithfulness 0.4702、answer_relevancy 0.8874、
  context_precision 0.7381、context_recall 0.7598（faithfulness 缺 3/16 条，如实标注）。
- 高并发压测 `docs/reports/高并发压测报告.md`（2026-10-10 实跑）：10 档 p50 5242.6 ms、
  50 档 p50 4552.5 ms / p95 67309.6 ms，均超 3 秒预算；**并发 100 档服务进程 GPU 显存
  耗尽而崩溃**——高并发稳定性未达标，崩溃现场与原始日志见报告。

## 三、环境要求

| 项 | 要求 | 本机实测 |
| --- | --- | --- |
| 操作系统 | Windows 10/11（开发与实测环境）；Linux / macOS 未实测 | Windows 11 |
| Python | 3.10+（`X \| Y` 联合类型标注 + `from __future__ import annotations`） | 3.12.10 |
| 依赖 | `python -m pip install -r requirements.txt`（torch / transformers / qdrant-client / fastapi / uvicorn / httpx / openai / langchain-ollama / psutil / streamlit / pymupdf / pdfplumber / jieba / rank-bm25 / ragas / rapidocr-onnxruntime） | 已装齐（按 `importlib.metadata` 逐个核对版本） |
| Ollama（本地服务） | `ollama pull bge-m3`（稠密向量，1024 维）、`ollama pull qwen2.5:3b`（生成兜底） | 已装且模型已拉取（设计文档 §2.1 环境勘察） |
| 模型权重 | `python scripts/download_models.py` 一键下载（走 hf-mirror） | CLIP `pytorch_model.bin` 605,247,071 B、bge-reranker-base 1,112,251,061 B，已落盘 `data/models/` |
| GPU | 可选（CPU 可跑；reranker 有 CUDA 则自动用，否则 CPU） | RTX 4060 Laptop 8.6 GB |
| LLM / VLM API key | `export API=sk-xxx`（依次尝试 `API` → `QWEN_API` → `OPENAI_API_KEY`；base URL 默认 `https://api.deepseek.com`，可用 `OPENAI_BASE_URL` 覆盖） | 已配置（设计文档 §2.1） |
| 磁盘 | 语料 + 模型 + 图缓存 + 索引约 2.1 GB | `data/models` 1743.3 MB、`data/qdrant` 283.1 MB、`data/figures` 35.4 MB、两份语料 18.7 MB（另有重建备份使 `data/` 合计 2.3 GB） |

> 本机列中，模型与索引体积为收尾阶段按文件字节实测，其余来自设计文档 §2.1 的环境勘察记录。

`httpx`、`openai`、`langchain-ollama`、`psutil` 已从传递依赖升为 `requirements.txt` 里的**显式声明**：
`openai` 是生成与 VLM 的 SDK、`httpx` 用于 Ollama 探测、`langchain-ollama` 供 RAGAS 的本地嵌入与评判；
`psutil` 让内存读数成真——未安装时 `src/rag04/obs/logging.py` 的 `rss_mb` 取 0.0，压测报告据此渲染
「未测得」，**不编造正数**。

权重缺失不阻断：reranker 缺失自动降级启发式重排，CLIP 缺失跳过该路检索（`/health` 会显示 `clip: missing`）；
无 LLM key 时生成降级本地 Ollama 并在响应里如实回显 `llm_backend`。

## 四、三步快速开始（下载模型 → 建库 → 起服务）

### 第 1 步：下载模型

```bash
python -m pip install -r requirements.txt
ollama pull bge-m3                 # 稠密向量模型（1024 维）
ollama pull qwen2.5:3b             # 生成兜底模型
python scripts/download_models.py  # CLIP 605.2MB + bge-reranker-base 1112.3MB（hf-mirror，断点续传）
export API=sk-xxx                 # 生成与 VLM 的 API key（API → QWEN_API → OPENAI_API_KEY）
```

### 第 2 步：建库

```bash
python scripts/build_index.py full_04          # 全量建库，本机实测约 15 分钟（5279 块）
python scripts/build_index.py full_04 --reset  # 先清空三库再建：分块/口径变更后必须用
```

- `--reset` 的必要性：分块改动会改变 `chunk_id`，不清空直接 upsert 会让**新旧块共存并污染检索**；
  重建前请先停掉 API / 界面进程（本地模式对 `data/qdrant` 持独占文件锁），建议先备份
  `cp -r data/qdrant data/qdrant_bak`。
- 建库产物：`text_chunks` 4436 / `table_chunks` 681 / `image_chunks` 162，合计 **5279** 块；
  日志 `logs/rag04.build.full_04.log` 给出逐份语料的进度、告警与用时。
- 对照模式：`python scripts/build_index.py baseline_03`（与 full_04 **共用同一批 collection**，
  重建会互相覆盖，两模式需串行跑评估）。

### 第 3 步：起服务

```bash
# 终端 A：FastAPI（工厂模式，避免导入即加载模型；启动时同步预热存储/reranker/CLIP/embedding）
uvicorn rag04.api.server:_get_app --factory --port 8000 --app-dir src

# 终端 B：Streamlit 界面（headless，端口 8501，中英可在侧栏切换）
python scripts/run_ui.py

# 自检（地址写 127.0.0.1：服务绑 IPv4，写 localhost 会先等 ::1 失败，每次多付约 2 秒）
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/ask -H "Content-Type: application/json" \
     -d "{\"question\":\"组织结构图中销售部有几个部门构成？\"}"
```

索引未建时 `/health` 的 `qdrant` 会显示 `down: ...` 并附原因——先完成第 2 步；
重建期间问答返回 503，跨进程撞锁返回 409（错误体统一顶层 `error` 字段，附可执行文案）。

## 五、目录结构

```
工单4/
├── src/rag04/                 系统源码（pytest pythonpath=src）
│   ├── config.py              配置中心：双模式开关、模型名、路径、并发与 3 秒预算
│   ├── schema.py              数据契约：Text/Table/FigureBlock、Chunk、Hit、Answer
│   ├── pipeline.py            端到端编排：build_all / ask / RAGPipeline（惰性加载 + 预热）
│   ├── ingest/                loader / figures / vlparser / tables / chunker / store / clip_dl
│   ├── retrieve/              embed / bm25 / image_index / hybrid / rerank / boilerplate
│   ├── generate/              prompt（中英双语）/ llm（主 + 兜底 + 拒答判定）
│   ├── eval/                  questions（16 题）/ metrics / runner / compare / ragas_eval
│   ├── api/server.py          FastAPI：/health、/api/ask、/api/stats、/api/ingest
│   ├── ui/app.py              Streamlit 中英双语界面（答案 / 引用 / 图像溯源）
│   └── obs/logging.py         轮转日志（10MB×10）、计时、内存快照
├── scripts/                   download_models / build_index / run_eval / run_ui / gen_location_report
├── benchmarks/loadtest.py     10/50/100 并发压测与报告（客户端墙钟为主口径）
├── tests/                     pytest（默认排除 integration）；fixtures/ 为图区真值等夹具
├── docs/                      交付文档 + reports/（报告）+ superpowers/（设计文档与实施计划）
├── data/                      运行期产物：qdrant 索引、bm25.pkl、figures/、models/、vlm_cache/
└── logs/                      轮转日志（10MB × 10）
```

## 六、常用命令

| 目的 | 命令 | 说明 |
| --- | --- | --- |
| 全量单元测试 | `python -m pytest -q` | `integration` 用例由 `pytest.ini` 自动 deselect |
| 集成测试 | `python -m pytest -m integration -v` | 需权重已下载、Ollama 运行中、API key 已配置 |
| 建库 / 重建 | `python scripts/build_index.py full_04 [--reset]` | 见第 2 步；`baseline_03` 为对照模式 |
| 中文评估 + 报告 | `python scripts/run_eval.py full_04` | → `docs/reports/检索精确度报告_full_04.md` + `eval_full_04.json` |
| 英文评估 | `python scripts/run_eval.py full_04 en` | 英文提问、中文标准要点：`answer_accuracy` 记 null，检索指标照常 |
| 答案定位报告 | `python scripts/gen_location_report.py` | 16 题逐题「要点 → 出处（页码/类型/来源ID）→ 召回 → 判定」 |
| 起 API 服务 | `uvicorn rag04.api.server:_get_app --factory --port 8000 --app-dir src` | 工厂模式，启动即预热 |
| 起界面 | `python scripts/run_ui.py` | Streamlit headless，8501 |
| 并发压测 | `python benchmarks/loadtest.py http://127.0.0.1:8000 10,50,100` | 每档请求数缺省 = 最大并发 × 3（100 并发时每档 300）；小于并发数的显式值会被上调并告警。**URL 必须写 IPv4 字面量**：服务绑 `0.0.0.0`，写 `localhost` 会让客户端每次新建连接先等 `::1` 失败（实测 2053ms vs 15ms），把 2 秒算进客户端墙钟 |
| 工单编号注释校验 | 脚本原文与实跑输出见 `docs/验收对照表.md` §四 | `src/`、`scripts/` 下全部 `.py` 文件头必须含工单编号注释；Windows 控制台为 GBK 时需 `PYTHONIOENCODING=utf-8` 才能打印 ✅ |

以上命令均在**仓库根目录**执行（脚本与配置按相对路径解析 `data/`、`logs/`、`docs/reports/`）。
评估类命令的产物文件名固定：`scripts/run_eval.py full_04` 覆写 `docs/reports/检索精确度报告_full_04.md` 与
`eval_full_04.json`——**重跑才会刷新**。磁盘上的报告分三代，引用前先看清是哪一份：
`..._full_04_rc2.md`（RC-2 索引，历史口径 p50 3888.6 ms，仅作对照）、
**当前有效**的 `..._full_04.md`（2026-10-07 15:10 运行，`ollama_url` 的 IPv4 修复后重测），
以及同为最终索引重跑的 `..._full_04_en.md`（英文提问、hit_rate 0.5，`answer_accuracy` 记「不适用」）；
更早的 Task 17 产物（answer_accuracy 0.1875）不得再引用。

## 七、文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/验收对照表.md` | 8 条硬性要求 → 实现位置 → 证据；未达标项与回填记录 |
| `docs/技术文档.md` | 架构、技术选型理由、开发流程、目录、运行方法、双模式、接口契约、质量现状、遗留清单 |
| `docs/图像解析与检索专项说明.md` | 图像为什么必须专项、三层图区检测、双通道多模态解析、入库与检索策略、实测效果与差距 |
| `docs/用户手册.md` | 环境准备、启动、上传、提问、看引用与图像、中英切换、故障排查 |
| `docs/答案定位报告.md` | 16 题逐题：问题 → 标准要点 → 出处 → 召回页码 → 答案与判定 |
| `docs/演示视频录制脚本.md` | 5~8 分钟演示分镜与解说词要点 |
| `docs/reports/检索精确度报告_full_04.md` | 16 题总体指标与逐题引用（**当前有效的评估口径**，2026-10-07 15:10 运行） |
| `docs/reports/优化前后对比分析.md`、`docs/reports/高并发压测报告.md` | 双模式对比（含两侧时延不同时点的口径注 2）/ 10、50、100 并发与预算逐级判定 |
| `docs/superpowers/specs/`、`docs/superpowers/plans/` | 设计文档（含 §10 工时拆解）与实施计划 |
