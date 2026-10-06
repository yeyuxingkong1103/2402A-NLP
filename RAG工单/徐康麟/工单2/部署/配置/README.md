# 部署说明（工单2）

> 工单：**人工智能NLP-RAG-基于PDF文档的问答系统优化**
> 阶段：部署 / 配置
> 适用：本机 Windows 演示 与 算力云 Linux+GPU 生产部署

本文件描述**如何把本仓库跑起来**：环境准备 → 模型下载 → 建索引 → 启动 → 验证 → 排障。
所有命令都以**仓库根目录**（`E:\gao6gongdan\工单2`，云端为 `/workspace`）为工作目录。

---

## 0. 一句话上手

| 场景 | 命令 |
| --- | --- |
| **本机 Windows 演示**（Ollama + 标准库界面） | `pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action serve -Port 8600` |
| 本机一次性提问 | `pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action ask -Question "注册资本是多少？"` |
| 本机健康自检 | `pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action health` |
| 本机跑评估 | `pwsh -NoProfile -File 部署/脚本/evaluate.ps1` |
| **云端**起 vLLM | `./部署/脚本/run_vllm.sh`（备选：`./部署/脚本/run_sglang.sh`） |
| 云端起应用 | `./部署/脚本/run_app.sh` |
| 云端跑评估 | `./部署/脚本/evaluate.sh` |

---

## 1. 目录结构（本阶段交付物）

```
部署/
├── 脚本/
│   ├── run_vllm.sh          # 云端：vLLM OpenAI 服务（首选后端）
│   ├── run_sglang.sh        # 云端：SGLang OpenAI 服务（备选后端）
│   ├── run_app.sh           # 云端：Streamlit 应用（可回退标准库界面）
│   ├── evaluate.sh          # 云端：优化前后对比评估
│   ├── run_app.ps1          # 本机：Ollama + serve_fallback（serve/ask/health/stats）
│   ├── evaluate.ps1         # 本机：对比评估（缺失对比脚本时需 -AllowFallback）
│   └── verify_config_env.py # 校验 config.example.env 与 config.py 字段一致性
├── 配置/
│   ├── README.md            # 本文件
│   ├── config.example.env   # 全部可覆盖的 RAG_* 参数（与 config.py 字段一一对应）
│   └── requirements-cloud.txt # 云端额外依赖（本机不可用，已标注等价实现）
├── docker/
│   ├── Dockerfile           # 应用侧镜像（Python 3.11 + 依赖 + 源码）
│   └── docker-compose.yml   # 应用 + vLLM 两服务（GPU 直通、模型只读挂载）
└── 日志/
    ├── app.log              # 全量结构化日志（JSON Lines）
    ├── error.log            # ERROR 及以上 + 完整 traceback
    ├── rag_trace.jsonl      # 事件流水（enter/exit/error/retrieval/generation/llm_io）
    └── 日志字段说明.md       # 字段 schema、真实行数、生成命令
```

---

## 2. 环境准备

### 2.1 本机（Windows，**离线**）

| 项 | 实测事实 |
| --- | --- |
| Python 解释器 | `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15）。可用 `-Python` 或环境变量 `RAG_SCHEDULER_PYTHON` 覆盖 |
| 统一入口 | `pwsh -NoProfile -File run_py.ps1 <脚本或 -m 模块>` |
| GPU | NVIDIA RTX 2060 6 GB，但本机 `torch` 为 **CPU 版**，不写 `torch.cuda` 路径 |
| 已就绪依赖 | pymupdf / jieba / numpy / pydantic / requests / httpx / torch(CPU) / transformers / sentence-transformers / faiss / pytest |
| **不可用且无法安装** | streamlit、loguru、pdfplumber、rank_bm25、chromadb、langchain、**ragas**、pandas、matplotlib、datasets、tiktoken、FlagEmbedding、accelerate、bitsandbytes |
| 网络 | **断网**：`pip install` 会挂死；`pypi.org` TLS 握手失败 |

> 因此本机：日志走自实现 JSON logger、BM25 自实现、界面走 `serve_fallback.py`、表格走 PyMuPDF
> `find_tables()`、RAGAS **未运行**。详见 `环境事实.md` §2.2 与 `部署/配置/requirements-cloud.txt`。

### 2.2 算力云（Linux + GPU，联网）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r 研发/requirements.txt
pip install -r 部署/配置/requirements-cloud.txt      # streamlit / vllm / ragas / chromadb …
# GPU 版 torch（按 CUDA 版本选择）：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 服务依赖

| 服务 | 本机 | 云端 |
| --- | --- | --- |
| 生成 LLM | Ollama `qwen2.5:3b`（实测首字 0.20 s） | vLLM（首选）/ SGLang（备选），OpenAI 兼容接口 |
| 嵌入 | Ollama `bge-m3:latest`（1024 维，多语言） | 同上（**必须与建索引时一致**） |
| 重排 | 无权重 → 自动回落 `mode=rule` | 可选 `BAAI/bge-reranker-base` |

---

## 3. 模型下载

**本机（Ollama，已就绪，仅列出以备重装）**

```powershell
ollama pull qwen2.5:3b        # 生成
ollama pull bge-m3:latest     # 嵌入（1024 维，中英文）
ollama list                   # 验证
```

**云端（vLLM / SGLang 用 HuggingFace 权重）**

```bash
# 示例：7B AWQ 量化权重（显存友好）
export MODEL_DIR=/data/models/Qwen2.5-7B-Instruct-AWQ
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir "$MODEL_DIR"
```

> 嵌入模型若走 Ollama：`docker compose exec ollama ollama pull bge-m3`（见 compose 中注释的可选服务）。
> 若改用 `sentence_transformers`，**索引维度会从 1024 变成 512，必须重建索引**（禁止跨维度复用）。

---

## 4. 索引构建（**必须先做，否则检索为空**）

```powershell
# 本机：全量重建（解析 548 页 + 分块 + 嵌入 + BM25，约 2.6 min）
pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --rebuild
# 复用已解析产物（更快）
pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --reuse-processed

# 云端：
python3 研发/scripts/build_index.py --rebuild
```

产物落在 `研发/data/index/bge-m3-1024/`：
`vectors.npy`(1669×1024)、`segments.npy`(4949×1024，子块级召回)、`bm25_index.pkl`、`meta.json`；
SQLite 在 `研发/data/index/rag.sqlite3`。

---

## 5. 启动

### 5.1 本机 Windows

```powershell
# 界面（默认走 serve_fallback；Ollama 后端）
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action serve -Port 8600
# 打开 http://127.0.0.1:8600

# 命令行提问
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action ask -Question "法定代表人是谁？"

# 无 LLM 时的兜底演示（抽取式路径）
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action serve -LlmBackend extractive

# 只看命令不执行（排障/交付评审）
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action health -DryRun
```

`run_app.ps1` 会先做四项前置检查：**解释器 → 索引就绪 → Ollama 可达 → 端口可用**，
任一失败给出明确原因与退出码（2/3/4/5/6），不会静默失败。

### 5.2 云端 Linux

```bash
# 1) 起推理服务（vLLM 优先）
MODEL_PATH=/data/models/Qwen2.5-7B-Instruct-AWQ ./部署/脚本/run_vllm.sh
#    备选：MODEL_PATH=/data/models/Qwen2.5-7B-Instruct ./部署/脚本/run_sglang.sh

# 2) 起应用（默认 openai 后端 + streamlit；无 streamlit 自动回退标准库界面）
./部署/脚本/run_app.sh --port 8600
```

### 5.3 Docker

```bash
export MODEL_DIR=/data/models/Qwen2.5-7B-Instruct-AWQ
docker compose -f 部署/docker/docker-compose.yml up -d
# 应用 http://<host>:8600 ；vLLM http://<host>:8000/v1
```

GPU 直通与模型挂载说明见 `部署/docker/docker-compose.yml` 头部注释（NVIDIA Container Toolkit +
`deploy.resources.reservations.devices`；旧版 `runtime: nvidia` 的替代写法也已给出）。

---

## 6. 验证

按“从里到外”的顺序，全部可复现：

| 步骤 | 命令 | 期望 |
| --- | --- | --- |
| 配置自检 | `pwsh -NoProfile -File run_py.ps1 -B 部署/脚本/verify_config_env.py` | `102 变量 / 被拒 0 / 抽查通过`，退出码 0 |
| 健康检查 | `pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action health` | `"ready": true`，LLM `available: true` |
| 库表统计 | `... -Action stats` | chunks=1669、documents=1、页面数 548 |
| 检索自测 | `pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_retrieval.py` | 10 题证据进最终 top-5 = 10/10 |
| 端到端评估 | `pwsh -NoProfile -File 部署/脚本/evaluate.ps1` | 输出到 `优化/评估结果`；RAGAS 一栏为「未运行」 |
| 日志落盘 | `Get-Content 部署/日志/rag_trace.jsonl -Tail 3` | 每行一个 JSON 事件；含 `retrieval`/`generation` |

> `evaluate.ps1` 默认调用 `优化/脚本/compare_optimization.py`（T7 产出）。该脚本尚未产出时，
> 加 `-AllowFallback` 可退回 `研发/scripts/evaluate.py` 先跑优化后系统评估。

---

## 7. 常见问题（FAQ，均来自本项目实测）

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `索引未就绪` / 检索为空 | 未建索引或索引目录被换 | 执行第 4 节建索引；确认 `研发/data/index/bge-m3-1024/vectors.npy` 存在 |
| `未安装 streamlit` | 本机无 streamlit 且断网 | 本机去掉 `-UseStreamlit` 走 `serve_fallback.py`；云端 `pip install streamlit` 后再用 |
| 首字响应变慢（>3 s） | LLM 后端探测/冷启动慢 | 确认 Ollama 已启动并预热；`RAG_LLM__PROBE_TIMEOUT=0.5` 保证探测失败快返回；不要用 `deepseek-r1:7b` 走在线路径（实测首字 18.68 s） |
| 检索结果里释义页霸榜 | 招股书前 30 页大量「释义/基本用语」样板 | 已内置释义降权（`RAG_RETRIEVAL__BOILERPLATE_PENALTY=0.35` 等），一般无需手工干预 |
| 重排未生效 | 本机无 `bge-reranker-base` 权重 | 系统自动回落 `mode=rule` 并记日志；**不得**声称跑过模型重排 |
| `RAGAS 未运行` | ragas 依赖不可用且断网 | 报告如实标注；确定性指标（准确率/命中率/引用/首字）作为替代证据 |
| Docker 相关命令失败 | 本机 Docker 守护进程不可访问（named pipe 权限拒绝） | 属本机限制；请在云端验证，本仓库不声称本机构建/运行过镜像 |
| 中文乱码 | 控制台编码 | 脚本已设 UTF-8；Python 侧统一 `sys.stdout.reconfigure(encoding="utf-8")` |
| 端口占用 | 上一次进程未退出 | `run_app.ps1`/`run_app.sh` 前置检查会直接报错；换 `-Port` |

---

## 8. 算力云与本机差异（**必须如实区分**）

| 维度 | 本机（Windows，离线） | 算力云（Linux + GPU） |
| --- | --- | --- |
| 生成后端 | Ollama `qwen2.5:3b` | vLLM（首选）/ SGLang（备选），OpenAI 兼容 |
| `RAG_LLM__BACKEND` | `ollama` | `openai` |
| `RAG_LLM__BASE_URL` | `http://127.0.0.1:11434`（Ollama 自有 API） | `http://127.0.0.1:8000/v1`（vLLM）/ `:30000/v1`（SGLang） |
| 界面 | `serve_fallback.py`（标准库 http.server） | `streamlit_app.py`（真实 Streamlit 应用） |
| 表格抽取 | PyMuPDF `find_tables()` | 可换 pdfplumber（代码已做能力探测） |
| 日志 | 自实现 JSON logger（结构同 loguru serialize） | 可装 loguru（文件结构不变） |
| 稀疏检索 | 自实现 PureBM25 | 可换 rank_bm25（接口不变） |
| 向量库 | numpy 精确检索 | 可换 Chroma（配置切换） |
| 重排 | `mode=rule` | 可上 `bge-reranker-base`（`mode=model`） |
| RAGAS | **未运行** | 可选运行 |
| 验证状态 | 脚本已实测（见 `部署/验证状态.md`） | **脚本已交付，待云端验证** |

---

## 9. 日志与错误码

- 日志目录：`部署/日志/`；字段 schema 见 `部署/日志/日志字段说明.md`（与 `设计/接口设计.md` §7 一致）。
- 轮转：`app.log` 50 MB / 保留 10；`error.log` 20 MB / 保留 10；`rag_trace.jsonl` 32 MB / 保留 3。
- 错误码与用户文案：`PDF_PARSE_ERROR` / `INDEX_NOT_READY` / `RETRIEVAL_ERROR` / `NO_EVIDENCE` /
  `PAGE_FILTER_EMPTY` / `LLM_UNAVAILABLE` / `STORAGE_ERROR` / `CONFIG_ERROR`（详见接口设计 §9）。

---

## 10. 纪律声明（交付边界）

1. **未在本机运行** vLLM、SGLang、streamlit、ragas、chromadb、pdfplumber、rank_bm25；
   `.sh` 脚本与 Docker 文件为「**已交付待云端验证**」（本机 WSL 被拒绝访问、Git Bash 因沙箱
   禁止创建命名管道、Docker 守护进程不可访问，均无法执行）。
2. **未伪造任何日志**：`部署/日志/*` 全部由系统真实运行产生（T2/T3/T4/T5 与部署自测），
   行数与生成命令见 `部署/日志/日志字段说明.md`。
3. 本机实测状态逐项记录在 `部署/验证状态.md`，包含**失败/受限项**。
4. `E:\gao6gongdan\工单1\` 为只读参考，全程未写入。
