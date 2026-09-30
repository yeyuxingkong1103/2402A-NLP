# 基于 RAG 的角色扮演系统

一个可运行、可部署的 RAG（检索增强生成）多角色对话系统。第一版 MVP 跑通
「文档解析 → 入库 → 混合检索 → 重排 → LLM 角色对话」主链路，内置 2 个角色
（**心理咨询师、律师**），知识库语料来自公开数据集（见
[数据来源](docs/数据来源.md)，由 `scripts/fetch_datasets.py` 拉取）。

## 技术栈

| 层 | 组件 | MVP（默认） | 生产可替换 |
|---|---|---|---|
| Web | FastAPI + uvicorn | 同步 + SSE 流式 | 不变 |
| 大模型 | OpenAI 兼容 | 本地 Ollama（`qwen3:0.6b`） | 硅基流动/DeepSeek/豆包/千问 |
| 向量化 | BGE-m3 | Ollama `bge-m3` | 硅基流动 BGE-m3 API |
| 重排序 | BGE-rerank 交叉编码 | ScoreFusion（RRF 融合分数） | 本地 bge-reranker 服务 / 硅基流动 API |
| 向量库 | Milvus | Milvus Lite（本地文件） | Milvus 服务器 |
| 文档解析 | PDF/文本/图片 | PyMuPDF + pdfplumber + RapidOCR | 多模态大模型 / MinerU |
| 混合检索 | 稠密 + BM25 | Python `rank_bm25` + jieba | Milvus 稀疏向量 |
| 分块 | 5 种策略 | 段落贪心合并（默认） | 标题 / 语义分块 |
| 召回增强 | Query 改写 / 摘要 / 去重 | LLM 改写扩写（默认关） | — |
| 关系库 | 用户/角色 | SQLite | MySQL |
| 短期记忆 | 多轮上下文 | 进程内内存 | Redis |

## 项目流程（一图看懂）

### 整体架构

```
浏览器 / Postman
   │  HTTP（JSON 或 SSE 流式）
   ▼
FastAPI 应用（app/main.py，启动时组装 RAGPipeline）
   │
   ├── 在线链路（对话）──────────► 需要：Ollama（LLM+embedding）、可选 rerank 服务
   └── 离线链路（入库）──────────► 需要：Ollama（embedding）、Milvus Lite
```

三个外部组件（都在本地）：

| 组件 | 作用 | 地址 |
|---|---|---|
| Ollama（Windows） | 大模型 qwen3 + 向量化 bge-m3 | `http://<主机>:11434/v1` |
| Milvus Lite | 向量库（本地文件，免服务） | `./data/milvus.db` |
| rerank 服务 | bge-reranker 交叉编码精排（可选） | `http://127.0.0.1:8001/v1` |

### 离线链路：把文档变成可检索的知识

```
文档（PDF / 图片 / md / txt）
  → parser.py       解析（PyMuPDF / pdfplumber / RapidOCR）
  → cleaner.py      清洗（去水印、去噪）
  → chunker.py      分块（段落，约 500 字 + 50 字重叠）
  → embedding.py    向量化（Ollama bge-m3 → 1024 维）
  → milvus_store.py 入库（role_knowledge 集合，按 role_id 隔离）
```

对应命令：`python scripts/ingest.py --dir data/corpus/lawyer --role lawyer`

### 在线链路：一次提问的完整旅程

```
用户问题："试用期最长可以约定多久？"
  ① embedding.py              问题向量化
  ② hybrid_retriever.py       两路召回：
       - 稠密：Milvus 向量检索（余弦相似度 + 阈值过滤）
       - 关键词：BM25（jieba 分词）
     → RRF 融合去重、排序
  ③ reranker.py               精排（score_fusion 融合分，或 bge-reranker 交叉编码）
  ④ prompt/templates.py       拼提示词（角色人设 + 知识 + 多轮记忆 + 问题）
  ⑤ llm.py                    LLM 生成（Ollama qwen3）
  ⑥ postprocess.py            后处理（去 <think>、正则清洗、去幻觉引用）
  ⑦ 返回用户 + 写回短期记忆（支撑多轮对话）
```

对应接口：`POST /chat`（非流式）、`POST /chat/stream`（SSE 流式）

### 关键文件地图（从哪里看起）

| 想了解… | 看这个文件 |
|---|---|
| 程序入口 / 启动装配 | `app/main.py` |
| 一次对话的主流程编排 | `app/core/pipeline.py` |
| 三个模型怎么调用 | `app/core/llm.py`、`embedding.py`、`reranker.py` |
| 检索与融合 | `app/core/retrieve/hybrid_retriever.py` |
| 提示词 & 角色人设 | `app/core/prompt/templates.py`、`role_presets.py` |
| 存储（向量 / 关系 / 记忆） | `app/core/store/`（milvus_store / sql_store / memory） |
| 文档解析入库 | `app/core/document/` + `scripts/ingest.py` |
| 重排服务（独立进程） | `rerank_service/server.py` |
| 所有配置项 | `.env` + `app/core/config.py` |

## 目录结构

```
rag-roleplay/
├── app/                # 应用代码（api / core / schemas）
│   └── core/           # config、llm、embedding、document、store、retrieve、prompt、pipeline
├── scripts/            # start_all.sh 一键启动、run.sh、shutdown.sh、ingest.py、fetch_datasets.py、install.sh
├── data/corpus/        # 语料（由 fetch_datasets.py 生成，不入库 git；manifest.json 记录来源与校验和）
├── tests/              # pytest 单元 + 接口测试
│   └── web/            # 前端行为测试（jsdom 驱动真实页面）
├── eval/               # RAGAS 风格评测（简化版）
├── docs/               # 需求规格说明书 / 设计文档 / 接口文档
├── requirements.txt
└── .env.example
```

## 快速开始（本地，无需 Docker）

### 0. 前置

- Python 3.10 ~ 3.14（本项目开发环境实测 **3.14.4**：OCR 依赖 onnxruntime 1.30.0 /
  opencv-python 5.0.0.93 在 3.14 上都有预编译轮子，未遇到"过新"问题。若你的 Python 版本
  恰好缺轮子，把 `OCR_ENABLED=false` 即可跳过扫描件解析）
- 大模型二选一：
  - **本地 Ollama**（推荐演示）：安装 Ollama 并拉取模型
    ```bash
    ollama pull qwen3:0.6b
    ollama pull bge-m3
    ```
  - **在线 API**：编辑 `.env` 指向硅基流动/DeepSeek 等 OpenAI 兼容服务。

### 1. 安装

```bash
cd rag-roleplay
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # 按需修改 LLM/Embedding 配置
```

### 2. 入库（离线链路）

```bash
# 先拉取语料（公开数据集，约 29MB，支持断点续传，只需跑一次）
python scripts/fetch_datasets.py

# 用拉下来的语料入库
python scripts/ingest.py --dir data/corpus/lawyer --role lawyer

# 或用你自己的 PDF/txt/md/图片（扫描件 PDF 与纯图片自动走 OCR）
python scripts/ingest.py --dir data --role lawyer --re-ingest
```

> 想一次灌入全部演示数据（2 个角色 + 各角色语料 + 示例用户），改用一键脚本：
> `python scripts/seed.py`（幂等，可重复执行；`--dry-run` 只预览不写入）。

### 3. 启动服务

一键启动（自检 Ollama / MySQL / Redis / Milvus，能自动拉起的自动拉起，最后校验 `/health`）：

```bash
bash scripts/start_all.sh              # 常规启动
bash scripts/start_all.sh --no-start   # 只自检不起服务
bash scripts/start_all.sh --strict     # 有组件不可用即退出非 0（交付演示用）
```

**Windows 上双击即用**：项目根目录的 `start.bat`（本质就是替你进 WSL 跑上面那条命令）：

| 双击 / 命令 | 作用 |
|---|---|
| 双击 `start.bat` | 启动；成功后自动打开浏览器到 `http://localhost:8000/`（聊天界面） |
| `start.bat --no-start` | 只做依赖自检，不起服务 |
| `start.bat --strict` | 有组件不可用即退出非 0 |
| `start.bat stop` | 停止服务（等价 `bash scripts/shutdown.sh`） |

> `start.bat` 里刻意**只有 ASCII 字符**：cmd.exe 是按控制台代码页**逐字节**解析 .bat 的，
> UTF-8 文件里的中文会被从半个字处切开——实测把 `echo` 行拆成两半当命令执行、`rem` 行还能
> 跳成死循环。中文提示一律由 WSL 侧的脚本输出，`chcp 65001` 保证它在控制台显示正常。

启动时会**自动拉起**这些外部依赖（本来就有的能力，2026-09-21 补齐了前两项）：

| 依赖 | 自动拉起 | 说明 |
|---|---|---|
| Ollama | ✅ | Windows 侧托盘程序（路径按 `%LOCALAPPDATA%` 找，不写死用户名），起来到应答约 3–8 秒 |
| Docker Desktop | ✅ | Redis 容器靠它。冷启动要 30–90 秒，比其他都慢 |
| Redis 容器 `redis-memory` | ✅ | 引擎就绪后 `docker start` 兜底 |
| bge 重排服务 | ✅ | 冷启动 1–3 分钟（加载 2GB 权重），脚本会等它就绪再继续 |
| **MySQL** | ❌ | Windows 服务需要管理员权限——实测非管理员执行 `net start MySQL80` 返回「拒绝访问」。脚本会探测、失败后给出管理员命令（`Start-Service MySQL80`，或弹 UAC 的一行命令） |

也可以分步手动起（`run.sh` 只负责 uvicorn 这一个进程）：

```bash
# 若 .env 里 RERANKER=bge，先下载模型 + 启动本地重排服务
bash scripts/download_rerank_model.sh   # 下 bge-reranker-v2-m3 到 /mnt/d/models（可重复执行，断点续传）
bash scripts/run_rerank.sh              # 端口默认 8001
bash scripts/verify_rerank.sh           # 验证真的在重排（不是只起了个空壳）

bash scripts/run.sh            # 后台启动（也可：uvicorn app.main:app --reload）
curl http://localhost:8000/health
```

> 本地 bge-reranker 重排服务依赖 torch + sentence-transformers，装在 `rerank_service/.venv`；
> 首次需 `python3 -m venv rerank_service/.venv && rerank_service/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && rerank_service/.venv/bin/pip install -r rerank_service/requirements.txt`。
>
> 开精排时召回池按 `RERANK_POOL`（默认 20）放宽再截断回 `TOP_K`：精排只能从粗排池里挑人，
> 池宽等于 `TOP_K` 时它就只能换顺序、捞不出粗排漏掉的候选，等于白开。

### 4. 对话（在线链路）

```bash
# 非流式
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"试用期最长可以约定多久？","role_id":"lawyer"}'

# 流式 SSE
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"精神障碍的诊断应当由谁作出？","role_id":"psychologist"}'
```

停止服务：`bash scripts/shutdown.sh`

## 测试

后端（pytest）：

```bash
source .venv/bin/activate
pytest -q
```

测试使用 `LLM_PROVIDER/EMBED_PROVIDER=dummy` 离线模式，不依赖 Ollama / Milvus Lite。

前端（jsdom，覆盖流式生成期间切角色/切会话等交互，无需启动服务）：

```bash
cd tests/web && npm install && npm test
```

前端测试用可控的 SSE 流驱动真实的 `app/static/index.html`，详见 `tests/web/README.md`。

## 评测（RAGAS 简化版）

```bash
python eval/ragas_eval.py --role lawyer   # 需 LLM 可用；--role 决定用哪组题（lawyer / psychologist）
```

## 无 Ollama 环境的离线冒烟

把 `.env` 中 `LLM_PROVIDER=dummy`、`EMBED_PROVIDER=dummy` 可跑通全部链路（返回离线回显答案），
用于验证解析→入库→检索→融合→后处理流程，不产生网络请求。

## 配置说明（.env 关键项）

| 变量 | 说明 |
|---|---|
| `LLM_PROVIDER` | `ollama` / `openai_compat` / `dummy` |
| `OLLAMA_BASE_URL` | 默认 `http://localhost:11434/v1` |
| `LLM_MODEL` | 默认 `qwen3:0.6b` |
| `EMBED_MODEL` | 默认 `bge-m3` |
| `OCR_ENABLED` | 扫描件/图片 OCR 兜底（RapidOCR，本地免服务），默认 `true`；设 `false` 关闭 |
| `MIN_CHUNK_CHARS` | 入库低质量过滤：低于此字符数的 chunk 丢弃（默认 10） |
| 摘要生成 | 入库时用 LLM 给每个 chunk 生成一句摘要（知识库数据增强，费 LLM、入库变慢）。全局开关是 `SUMMARY_ENABLED`（默认 `false`）；上传接口的 `summary` 参数与 `scripts/ingest.py --summary` 可对单次入库覆盖它，不传则跟随全局值 |
| `QUERY_REWRITE_ENABLED` | 对话时 LLM 改写/扩写 query 多查询召回（默认 `false`） |
| `APP_ENV` | 环境切换：`dev` / `test` / `prod`，加载对应的 `.env.{env}` 覆盖 `.env` |
| `RERANKER` | `score_fusion` / `bge`（bge 需先启动本地重排服务，见下） |
| `RERANK_POOL` | 精排粗排池宽度，默认 `20`（仅 `RERANKER=bge` 生效，见下） |
| `RERANK_BASE_URL` | bge 重排服务地址，默认 `http://127.0.0.1:8001/v1` |
| `RERANK_MODEL` | 默认 `BAAI/bge-reranker-v2-m3` |
| `MILVUS_DB_URI` | `.db` 路径 = Milvus Lite；`http://host:19530` = 服务器 |
| `SQL_URL` | `sqlite:///./data/app.db` 或 MySQL 连接串 |
| `MEMORY_BACKEND` | `memory` / `redis` |
| `HISTORY_MAX_CHARS` | 送进提示词的历史字符预算（默认 12000）。按字符而非轮数，因为决定装不装得下的是**回答长度** |
| `PROMPT_MAX_CHARS` | 整个提示词（系统消息 + 历史 + 提问）的字符预算（默认 14000）。系统消息里的检索资料不受 `HISTORY_MAX_CHARS` 约束，靠这项兜底 |

## 服务器部署（Ubuntu / 算力云 / 腾讯云 / 阿里云）

```bash
# 1. 上传代码（git 拉取或打包 zip/tar 解压到项目目录）
# 2. 执行安装脚本（检测环境→建 venv→装依赖）
bash scripts/install.sh
# 3. 拉取语料（或放入自己的知识库文档到 data/），配置 .env
# 4. 入库 + 启动
.venv/bin/python scripts/fetch_datasets.py
.venv/bin/python scripts/ingest.py --dir data/corpus/lawyer --role lawyer
bash scripts/run.sh
```

生产建议：`MILVUS_DB_URI` 换成 Milvus 服务器，`SQL_URL` 换 MySQL，`MEMORY_BACKEND=redis`，
并用 Jmeter 做压测（关注 QPS），用 Nginx/负载均衡做横向扩展。
完整的部署指南（BGE-rerank 启用、多路召回多数据源、压测、三环境）见 [deploy/README.md](deploy/README.md)；
**部署完别只看「脚本没报错」**，照 [deploy/README.md 第六节](deploy/README.md) 的自证清单逐条验
（依赖自检 exit 0、端到端 `/chat` 出有据答案、同会话记得住且换会话记不住、负载均衡落点分布），
那一节里附了本机彩排的实测基线可以直接对照。

## 文档

- [项目说明（先看这个，一页看懂整个项目）](docs/项目说明.md)
- [代码地图（每个文件负责什么 + 我要改 X 动哪个文件）](docs/代码地图.md)
- [流程图（架构 / 入库 / 对话 / 检索，另附可拖进 Word 的 SVG）](docs/流程图.md)
- [RAG + 微调学习路径](docs/学习路径.md)
- [需求规格说明书](docs/需求规格说明书.md)
- [设计文档](docs/设计文档.md)
- [接口文档](docs/接口文档.md)
- [部署指南（含部署后自证清单）](deploy/README.md)
