# 法律 RAG 助手

> 📖 **只想用起来？直接看 [`docs/USER-GUIDE.md`](docs/USER-GUIDE.md)（中文使用手册）** ——
> 怎么打开界面、怎么问、怎么上传资料、怎么起停服务、坏了看哪里、有哪些已知限制。
> 本 README 是**开发者向**的：环境、目录结构、接口清单与工程约定位。

面向「基于 RAG 的多用户多角色多会话动态角色扮演系统」的**法律垂直领域第一阶段实现**：
把「文档 → 向量 → 检索 → 生成」整条链路跑通 —— 既能在**离线兜底模式**下零外部依赖、零密钥运行，
也能切到**真实后端**（bge-m3 嵌入 + Milvus 向量库 + 本地 Ollama 生成）端到端运行；
微调（Qwen3.8-27B + LLaMA-Factory）与算力云部署属于后续阶段。

当前状态：

* 法律 SFT 微调数据 **已就绪**（4,500 训练 / 500 验证，见 `data_pipeline/`）
  —— ⚠️ **这是早期数据准备阶段的资产，没有用于后续任何一次 LoRA 训练**
  （实际训练数据由 `scripts/build_sft_data.py` 现场构建：**912 训练 / 80 验证**，
  见 [`docs/FINETUNE-REPORT-FINAL.md`](docs/FINETUNE-REPORT-FINAL.md) §2.1–§2.2）
* **微调已做完并给出结论**：LoRA（Qwen3-4B，bf16 基座，**非 QLoRA**）**测不出答案质量增益**
  ⇒ 生产**不加载** LoRA，用基座 27B。完整报告见
  [`docs/FINETUNE-REPORT-FINAL.md`](docs/FINETUNE-REPORT-FINAL.md)（定稿）
* 法律 RAG 应用**已在本地端到端跑通**（离线模式零外部依赖、零密钥）
* **双链路已在真实后端上跑通并完成 Windows + Ubuntu 双环境独立验证**
  —— 离线链路 PDF → bge-m3（1024 维）→ Milvus，在线链路 提问 → 检索 → Ollama 真实 LLM 回答；
  详见 [`handoff/DELIVERY.md`](handoff/DELIVERY.md)、[`VERIFICATION.md`](VERIFICATION.md)、
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)（均为**历史验收记录**，不代表本机此刻的状态）

## 目录结构

```
.
├── data/                    数据目录（只读原始语料，产物写入）
│   ├── DISC-Law-SFT-Pair-QA-released.jsonl   原始语料（79,692 行）
│   ├── legal_train.json                      训练集产物（4,500 条，Alpaca）
│   ├── legal_val.json                        验证集产物（500 条，Alpaca）
│   └── dataset_info.json                     LLaMA-Factory 数据集注册表
├── data_pipeline/           法律 SFT 数据准备链路（4 步，已完成）
│   ├── 01_validate_raw.py         字段体检
│   ├── 02_explore_distribution.py 分布探查
│   ├── 03_build_dataset.py        主构建：去重→过滤→分层抽样 5000→Alpaca→9:1
│   ├── 04_check_quality.py        质检（中英混杂 / 法条引用 / 长度）
│   ├── run_all.py                 一键顺序执行
│   └── README.md
├── legal_rag/               法律 RAG 应用包
│   ├── config.py                  全部后端可插拔的配置中心
│   ├── schemas.py                 Chunk / SearchHit / Role / Message / Answer
│   ├── roles.py                   12 个角色与人设模板
│   ├── engine.py                  检索→重排→提示词→生成→后处理 编排
│   ├── ingest/                    离线链路：解析 / 分块 / 入库编排
│   ├── embedding/                 向量化：OfflineEmbedder + OllamaEmbedder + BgeM3Embedder
│   ├── store/                     向量库：MemoryVectorStore + MilvusVectorStore + ChromaVectorStore
│   ├── retrieve/                  在线链路：BM25 混合检索 + 重排（含时间衰减）
│   ├── generate/                  提示词 / LLM 客户端（Mock/Ollama/DeepSeek/OpenAI 兼容）/ 模型路由 / 后处理
│   ├── memory/                    短期记忆（Redis/内存，滑窗）+ 长期记忆扩展点
│   ├── business/                  业务数据（MySQL / SQLite）
│   └── api/                       FastAPI 接口
├── knowledge/               示例法条文档（自写，用于跑通）
├── scripts/                 可执行入口
│   ├── build_index.py             建知识库索引
│   ├── chat_cli.py                命令行多轮对话
│   ├── run_api.py                 启动 HTTP 服务
│   ├── deploy.sh                  一键部署 + 自检（在目标 Ubuntu / WSL 主机上运行）
│   └── sync_to_remote.ps1         Windows → 远端同步 + 逐字节自检（PowerShell）
├── sql/schema.sql           MySQL 建表语句
├── tests/                   单元 / 接口测试（`pytest tests -q` 全部通过）
├── run.sh / shutdown.sh     Ubuntu 启停脚本
├── requirements.txt         轻量核心依赖（含 HTTP 服务与接口测试所需的 fastapi / uvicorn / httpx）
├── requirements-full.txt    真实后端重依赖（torch / chromadb / BGE 权重 …，按需安装）
└── .env.example             环境变量样例
```

> ⚠️ 仓库里**没有** `docker-compose.yml`（全仓无 `*.yml` / `*.yaml`）。Milvus / Redis 需要你自己用官方 compose 或镜像启动，
> **`run.sh` 不会拉起 Milvus**（详见下文「后续阶段」与「已知限制」）。

## 快速开始

```bash
# 1) 创建虚拟环境并安装轻量核心依赖
#    requirements.txt 已包含：pydantic / pypdf / pytest + fastapi / uvicorn / httpx
#    （后三个是 HTTP 服务与接口测试的硬依赖，纯 Python 轻量包，不需要 requirements-full.txt）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS

# 2) 建知识库索引（离线模式：内存向量库 + 零依赖向量化）
.venv\Scripts\python.exe scripts/build_index.py --source knowledge/ --offline --rebuild

# 3) 命令行问答（无需 API key）
.venv\Scripts\python.exe scripts/chat_cli.py --offline --once "民间借贷的利率上限是多少"
.venv\Scripts\python.exe scripts/chat_cli.py --offline --role 律师      # 交互式多轮

# 4) 启动 HTTP 服务（依赖第 1 步已装好的 fastapi / uvicorn）
.venv\Scripts\python.exe scripts/run_api.py --offline --port 8000

# 5) 调用接口（curl 写法按 shell 区分，请整段复制下方「第 5 步：调用接口」小节中的一段）

# 6) 跑测试（接口测试需要第 1 步已装好的 httpx）
.venv\Scripts\python.exe -m pytest tests -q
```

> 📌 **跑测试的两条命令都行（2026-09-27 补）**
> 上面这条 `pytest tests -q` 是规范命令。**不带路径**的 `pytest`（在仓库根目录直接跑）
> 现在也能用：根目录的 `conftest.py` 把 `handoff/artifacts/`（历次交付按内容哈希冻结的
> **文件快照**）挡在收集之外。以前不行 —— 那里面有个 `test_api.<hash>.py`，
> 文件名带点号会让 pytest 报 `ModuleNotFoundError: No module named 'test_api.<hash>'`，
> **整轮在收集阶段就中断，一条用例都不跑**（看起来像"测试全挂"，实际是"没开始跑"）。
> 这条守卫本身有测试：`tests/test_pytest_collection_guard.py`。

> ⚠️ **默认值行为变更（本地优先）**：`legal_rag/config.py` 里 Milvus / Redis 的**代码内默认值**
> 已从旧版 Ubuntu VM `192.168.188.128` 改为**本机 `127.0.0.1`**
> （`DEFAULT_MILVUS_HOST = "127.0.0.1"`、`DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"`）。
> 也就是说：**不设 `MILVUS_HOST` / `REDIS_URL` 时现在连本机，不再连 VM；要连远端必须显式设置。**
> 旧文档 / 旧脚本里写死的 `192.168.188.128` 都是**历史值**，不是当前默认值。

### 第 5 步：调用接口

第 1～4 步的命令在 Windows / Linux 下同样可用；**第 5 步的 curl 写法按 shell 区分**，
下面两段请挑当前 shell 的那一段整段复制，不要混用。

**bash / Git Bash / WSL / Linux / macOS：**

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/roles
curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id":"u1","role_id":"lawyer","session_id":"s1","message":"民间借贷利率上限"}'
```

**Windows PowerShell：**（用 `curl.exe`；PowerShell 5.1 里 `curl` 是 `Invoke-WebRequest` 的别名，参数不兼容。
下面两段都已在 PowerShell 5.1 实测返回 200，任选其一）

```powershell
curl.exe http://127.0.0.1:8000/health -o NUL -s -w "http=%{http_code}`n"
curl.exe http://127.0.0.1:8000/roles

# 写法 A（推荐）：here-string 写入 UTF-8 无 BOM 的 chat.json，再按文件提交
$body = @'
{"user_id":"u1","role_id":"lawyer","session_id":"s1","message":"民间借贷利率上限"}
'@
[IO.File]::WriteAllText("$PWD\chat.json", $body)
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" --data-binary "@chat.json"

# 写法 B：完全不依赖 curl 的原生写法（自包含）
$payload = @{user_id='u1';role_id='lawyer';session_id='s1';message='民间借贷利率上限'} | ConvertTo-Json -Compress
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" `
  -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

> **不要**在 PowerShell 里用 bash 的 `-d "{\"user_id\":\"u1\",…}"`，也不要写
> `--data-binary $body`：PowerShell 5.1 向原生程序传参时会再给内嵌双引号加一层转义，
> 服务端会返回 `422 json_invalid`（服务端行为正确，只是写法不适用 PowerShell）。

### 接入真实大模型

代码里大模型调用已抽象成统一接口，切换只改配置：

```bash
# 把 .env.example 复制为 .env，然后：
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxx          # 只放环境变量，绝不写进代码
```

> ⚠️ **`.env` 由谁读（2026-09-27 更正）**
>
> 早先这里写的是"代码里没有 `load_dotenv`，所以直接跑 `scripts/run_api.py` 时 `.env` 会被静默忽略"。
> **这一句已经过时，别再照它操作**：`scripts/run_api.py` 现在自带一个**不依赖 `python-dotenv`**
> 的读取函数（`load_dotenv()`，在 `main()` 开头调用），把仓库根目录的 `.env` 读进环境变量，
> **已存在的环境变量优先**（即 `export` 过的值不会被 `.env` 覆盖）。
>
> 启动时会**把结果打印出来**，不静默：
>
> ```
> 配置文件   : /path/to/.env（已加载 3 条；已存在的环境变量优先）
> 配置文件   : /path/to/.env（不存在或为空）
> ```
>
> 边界与细节：
> * 只有**仓库根目录**的 `.env`（`ROOT/.env`）会被读；子目录里的不读；
> * 语法是**简化版**：支持 `KEY=value`、`#` 注释、可选的 `export ` 前缀、
>   值两侧的单/双引号会被剥掉；**不支持**多行值、变量插值（`$OTHER`）、`${}` 展开；
> * `legal_rag/` 包内部**依旧零 `dotenv` 命中** —— 配置只在进程启动入口注入，
>   库代码只认 `os.environ`（这样测试里 `monkeypatch.setenv` 依然完全可控）；
> * `bash run.sh` 走的是另一条路（`set -a; source "$APP_DIR/.env"; set +a`）,
>   两条路都读同一个文件。**共同点**：别把密钥写进代码。

> 另注：`run.sh` 把 `API_HOST` 默认成 `0.0.0.0`（见 `run.sh` 里的 `API_HOST="${API_HOST:-0.0.0.0}"` 一行），而 `config.py` 里 `api_host`
> 的代码内默认值是 **`127.0.0.1`** —— 两者**不一致**：Ubuntu 上 `bash run.sh` 默认监听所有网卡，
> 而直接 `python scripts/run_api.py` 默认只监听本机回环。需要对外提供服务时请显式设置 `API_HOST`。

按量付费的在线 API 也可换成任何 OpenAI 兼容端点（豆包 / 硅基流动 / 千问 / 本地 vLLM / SGLang）：

```bash
LLM_PROVIDER=openai_compat
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=Qwen3.8-27B
```

## 接口清单

以 `legal_rag/api/app.py` 的实际路由装饰器为准，共 **34 条路由装饰器**（含 `/livez`、`/auth/*`、
`/prefs`、`/ui`、登录/注册页等；完整清单见 `docs/API.md` §2 与 `/docs`）。下表只列**主链路**：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 各组件健康状态（含降级后的实际后端）。**默认读快照**（O(1)，最多 15 s 旧，`stale` 标明）；`?deep=1` 才现采 |
| GET | `/livez` | 存活探针：**不探依赖**、O(1)，LB/K8s 的高频探活用它 |
| GET | `/metrics` | Prometheus 文本指标（业务 + 系统层），**轻量**：不触发检索/生成/后端探测 |
| GET | `/roles` | 角色库与人设模板 |
| POST | `/sessions` | 创建/登记会话 |
| GET | `/sessions?user_id=` | 某用户的会话列表 |
| POST | `/chat` | 一次问答，入参 `user_id` / `role_id` / `session_id` / `message`（可流式） |
| POST | `/ingest` | 重建或增量更新知识库 |
| GET | `/documents/upload` | 上传能力自描述（允许的扩展名、体积/数量上限、落盘目录） |
| POST | `/documents/upload` | 上传 PDF（兼容 txt/md/json）并**立即入库**，返回 `job_id` |
| GET | `/documents` | 已入库文档列表 |
| GET | `/documents/jobs` | 最近的上传入库 job（`?limit=`，1–200，默认 20） |
| GET | `/documents/jobs/{job_id}` | 单个 job 进度（pending / running / completed / failed） |
| DELETE | `/documents/{doc_id}` | 删除该文档的向量分块 + 磁盘文件 |

> 上表是**主链路**子集（不是全量路由）。`api/app.py` 的 `@app.{get,post,delete}` 装饰器
> **实测计数 = 34**（2026-09-29；历史版本写过 13，那是加鉴权/偏好/文档/页面族之前的数字，
> 已作废）。其中 `/sessions` 的 POST 与 GET、`/auth/logout` 的 POST 与 GET 各是**两条**独立
> 装饰器但共用同一路径，所以**路径数少于装饰器数**。完整清单以 `docs/API.md` §2 与 `/docs` 为准。
> 接口的详细契约（并发模型、状态码语义、流式约定）见 [`docs/API.md`](docs/API.md)。

**隔离约定**：所有问答请求必须携带 `user_id + role_id + session_id`。
短期记忆以这三元组为键；同一 `session_id` 若已归属其他用户，接口直接返回 `403`。

## 双链路设计

* **离线链路**（`legal_rag/ingest/`）：解析 → 分块（固定/句子/段落 + 父子块 + overlap）
  → 向量化 → 入库；以文件 MD5 为指纹做增量去重，重复入库不会产生重复 chunk。
* **在线链路**（`legal_rag/retrieve` + `generate`）：问题向量化 → 向量召回 + BM25 召回
  → 融合（weighted / RRF）→ 重排（BGE-rerank 或余弦兜底，含时间衰减与低分过滤）
  → 四段式提示词（角色设定 + 检索知识 + 历史对话 + 用户问题）→ 模型路由调用大模型
  → 后处理（敏感词 / 格式校验 / Markdown 清洗 / 流式）→ 返回答案与引用来源。

## 可插拔后端

下表「真实实现」一列**以 `legal_rag/embedding/`、`legal_rag/store/`、`legal_rag/generate/`
下的实际文件为准**（类名逐个核对）：

| 能力 | 兜底实现（默认，零重依赖） | 真实实现 |
|---|---|---|
| 向量化 | `OfflineEmbedder`（纯标准库哈希 TF） | **`OllamaEmbedder`**（Ollama bge-m3，免 torch —— 本阶段实际链路）/ `BgeM3Embedder`（BGE-m3，torch + sentence-transformers 重依赖） |
| 向量库 | `MemoryVectorStore` | **`MilvusVectorStore`**（gRPC 19530，pymilvus）/ `ChromaVectorStore`（本地轻量替代） |
| 重排 | `CosineReranker` | `BgeReranker` |
| 大模型 | `MockLLMClient` | **`OllamaClient`**（Ollama `/api/chat`，免密钥 —— 本阶段实际链路）/ `DeepSeekClient` / `OpenAICompatClient`（vLLM / SGLang / 豆包 / 千问） |
| 短期记忆 | `InMemorySessionStore` | `RedisSessionStore` |
| 业务数据 | `SQLiteBusinessStore` | `MySQLBusinessStore` |

切换方式统一走环境变量（见 `.env.example`）：`EMBEDDING_PROVIDER=offline|bge_m3|ollama`、
`VECTOR_STORE=memory|chroma|milvus`、`LLM_PROVIDER=mock|ollama|deepseek|openai_compat`（及别名）。
缺依赖时会给出带修复命令的清晰报错，而不是把 `ImportError` 堆栈甩出来。

> 注：`ChromaVectorStore` 与 `BgeM3Embedder` 需要 `requirements-full.txt` 的重依赖；
> 本阶段跑通的真实链路用的是 **`OllamaEmbedder` + `MilvusVectorStore` + `OllamaClient`**。
> 生成侧的 provider 可用取值与别名以 `legal_rag/generate/llm_base.py::build_llm_client()` 为准，
> 未知值会直接 `ValueError`（严格模式，不静默兜底）。

## 文档与一键脚本

### `deploy/`：四个 Linux shell 脚本（**推荐入口**）

安装 → 部署 → 运行 → 结束，一条命令一步；**都是 Linux shell**（`#!/usr/bin/env bash` + LF），
**都写日志**（`logs/<脚本>-YYYYMMDD.log`，同时打屏），**都带注释**（文件头写清用途/用法/退出码）。
`--dry-run` 可以先看要做什么再决定跑不跑。

⚠️ **开发机没有可用的 bash**（`bash.exe` 是 WSL 启动器、服务拒绝访问）⇒ 本机只能做**静态**
检查（`tests/test_deploy_scripts.py`）。所以在真机上**先跑 `deploy/selfcheck.sh`**：
它会把 `bash -n`、`--help`、`--dry-run`、`lib.sh` 助手齐全这四步真跑一遍。

```bash
# ⓪ 先自检（**真机第一步**）：bash -n 四个脚本 + --help + --dry-run + lib.sh 助手齐全
bash deploy/selfcheck.sh                     # 全量；约 10 秒，不改动磁盘/进程
bash deploy/selfcheck.sh --quick             # 只做语法检查

# ① 一键安装环境：建 .venv、装依赖、可选装 Redis/Ollama、五项导入自检
bash deploy/install.sh                       # 最小可跑集合
bash deploy/install.sh --full --with-redis   # 加 torch 等大件 + 装 Redis
bash deploy/install.sh --dry-run             # 只打印

# ② 一键部署：目录准备 + 生成 .env（**绝不覆盖已有**）+ 可选建索引
bash deploy/deploy.sh
bash deploy/deploy.sh --index --rebuild      # 顺便重建索引
bash deploy/deploy.sh --archive pkg.tgz --dir /opt/legal-rag   # 从压缩包部署

# ③ 运行：Redis →（可选）vLLM → API，并等健康检查（用 /livez，不占线程池）
bash deploy/start.sh                         # 用 .env 的配置
bash deploy/start.sh --offline               # 无密钥/无卡也能起
bash deploy/start.sh --with-vllm --vllm-model /models/qwen27b

# ④ 结束：按 **API → vLLM → Redis** 的顺序停干净，并检查残留
bash deploy/stop.sh
bash deploy/stop.sh --keep-redis             # 只停应用与模型
```

| 脚本 | 退出码（详见各自的 `--help`） |
|---|---|
| `deploy/install.sh` | 0 成功；10 平台不对；11 缺 python3；12 版本过低；13 空间不足；14 建 venv 失败；15 装依赖失败；16 自检未过 |
| `deploy/deploy.sh` | 0 成功；20 结构不对；21 缺 python3；22 生成配置失败；23 依赖未就绪；24 建索引失败 |
| `deploy/start.sh` | 0 成功；30 依赖未就绪；32 vLLM 未就绪；33 API 启动失败；34 健康检查超时 |
| `deploy/stop.sh` | 0 停干净；40 停完仍有残留进程 |

> ⚠️ **停止顺序不能反**（先 API → 再 vLLM → 最后 Redis）：API 持有 Milvus Lite 的
> **单进程文件锁**，顺序反了会出现"锁被抢 ⇒ 静默回落内存库"，后面的读数全是假的。
> 同理，**探活用 `/livez`**（O(1)），别用 `/health`（深度体检：默认读快照、已便宜，
> 但 `?deep=1` 现采时依赖挂掉单次 2.2s 且吃线程池 —— 见 `docs/API.md` §6.1）。

> 本机（Windows）**没有可用的 bash**，所以这四个脚本是通过**静态验收**保证质量的：
> `tests/test_deploy_scripts.py` 逐条检查 **LF 行尾**（CRLF 会让 Linux 上一行都跑不了）、
> shebang、`set -euo pipefail`、`source lib.sh`、`--help`、
> 以及"停止顺序"和"探活用 `/livez`"；有 `bash` 的环境会自动补跑 `bash -n`。

### 历史脚本（保留）

| 文档 | 内容 |
|---|---|
| [`docs/PROJECT-OVERVIEW.md`](docs/PROJECT-OVERVIEW.md) | **项目全景（一页看全）+ 答辩索引**：技术栈 / 模型清单 / 端到端流程 / 三层记忆各存什么（含真跑出的例子）/ 微调数据与参数 / 评测读数 / 证据文件地图 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **架构总览（单页）**：分层与依赖方向、双链路数据流、维度治理、降级模型、并发模型、指标口径 |
| [`docs/API.md`](docs/API.md) | 接口契约、并发模型、状态码语义 |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 双环境事实核对（Windows + Ubuntu 实测数字与依赖就绪状态） |
| [`docs/WSL.md`](docs/WSL.md) | 迁移到 WSL2 的指引（含 §0 的默认值**行为变更**说明、GPU 直通、网络拓扑坑） |
| [`docs/CLOUD.md`](docs/CLOUD.md) | 算力云部署与 Qwen3.8-27B（vLLM / SGLang）接入指引 |
| [`docs/MONITORING.md`](docs/MONITORING.md) / [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) / [`docs/LOGGING.md`](docs/LOGGING.md) | 监控、可观测性、日志 |
| [`VM-DEPLOY.md`](VM-DEPLOY.md) | Ubuntu VM 部署与同步记录（历史验收） |
| [`handoff/DELIVERY.md`](handoff/DELIVERY.md) | 交付说明（含实测数字与来源口径） |
| [`VERIFICATION.md`](VERIFICATION.md) / [`CHECKPOINT.md`](CHECKPOINT.md) | 多轮独立验收报告与集成核查记录 |

**两个历史脚本**（`deploy/` 那套是它们的整理版，二者可共存；参数以脚本自带的 `--help` 为准）：

```bash
# 1) 在目标 Ubuntu / WSL 主机上「部署 + 自检」（5 步，默认不起服务）
bash scripts/deploy.sh
bash scripts/deploy.sh --start              # 自检通过后顺手起服务（内部 bash run.sh </dev/null）
bash scripts/deploy.sh --start --offline    # 起服务时走离线模式（无密钥 / 无 GPU）
bash scripts/deploy.sh --skip-deps          # 跳过刷依赖，快速自检
bash scripts/deploy.sh --help               # 看用法
# 退出码：0=全通过 1=项目结构不对（缺 legal_rag/） 2=缺 python3 3=版本过低
#         4=建虚拟环境失败 5=刷依赖失败 6=清理字节码失败 7=导入自检未全过 8=起服务失败
```

```powershell
# 2) 在 Windows 侧「同步到远端 + 逐字节自检」（4 步）
powershell -ExecutionPolicy Bypass -File scripts\sync_to_remote.ps1
powershell -ExecutionPolicy Bypass -File scripts\sync_to_remote.ps1 -WhatIf   # 只打印，不动远端
# 可显式传参换机器：
powershell -ExecutionPolicy Bypass -File scripts\sync_to_remote.ps1 `
    -RemoteUser ubuntu -RemoteHost 10.10.0.21 -Port 2222 `
    -KeyPath C:\keys\id_ed25519 -RemoteDir projects\legal-rag
```

`sync_to_remote.ps1` 的参数（默认值取自脚本 `param()` 块）：
`-RemoteUser`（默认 `dshagent`）、`-RemoteHost`（默认 `192.168.188.128`，只是历史 VM 值，换机器请显式传）、
`-Port`（默认 `22`）、`-KeyPath`（默认 `<工作区根>\.ssh\dsh_ubuntu_ed25519`）、
`-KnownHosts`（默认 `<工作区根>\.ssh\known_hosts`）、`-RemoteDir`（默认 `legal-rag`）、
`-WorkspaceRoot`、`-ReportPath`、`-WhatIf`。
它同步 **7 个目录 + 7 个顶层文件**（`legal_rag` / `scripts` / `tests` / `knowledge` / `docs` / `sql` /
`data_pipeline`，以及 `requirements.txt` / `requirements-full.txt` / `README.md` / `.env.example` /
`run.sh` / `shutdown.sh` / `.gitignore`），**刻意不传 `data/`**，也不动 Redis / Milvus；
比对结果打出 `identical` / 缺失 / 不同 / 多余 四个数字。
退出码（脚本 `exit` 点）：`0`=两侧逐字节一致、`5`=不一致、`2`=前置校验失败（缺 ssh/scp、私钥不存在、
工作区根不对、参数非法）、`3`=scp 失败、`4`=远端 `cd` 失败、`6`=清单比对失败（缺 sha256 工具或输出异常）。

## data/ 目录里的四个文件

| 文件 | 角色 |
|---|---|
| `DISC-Law-SFT-Pair-QA-released.jsonl` | **原始语料**，只读，字段 `id` / `input` / `output`（79,692 行） |
| `legal_train.json` | **训练集产物**，4,500 条，Alpaca 格式 ⚠️ **未被任何训练使用**（见下） |
| `legal_val.json` | **验证集产物**，500 条，Alpaca 格式 ⚠️ **未被任何训练使用**（见下） |
| `dataset_info.json` | **LLaMA-Factory 注册表**，注册 `legal_train` / `legal_val`（该路线未采用） |

> ⚠️ **澄清（2026-09-29，常被问）**：上面 4,500 / 500 这套是**早期数据准备阶段的资产**，
> 准备给"LLaMA-Factory + 通用法律问答"那条路线用的。**后续实际做的 LoRA 微调一条都没用它们**：
> 全仓 `scripts/`、`legal_rag/` 对 `legal_train`/`legal_val` **零引用**；
> 训练数据是由 `scripts/build_sft_data.py` **现场构建**的（27B 教师 + 语料证据 + 三道质量闸门），
> 最终配置是 **912 训练 / 80 验证**。原因与逐臂数据账目见
> [`docs/FINETUNE-REPORT-FINAL.md`](docs/FINETUNE-REPORT-FINAL.md) §2.1–§2.2。

数据准备链路的细节见 [`data_pipeline/README.md`](data_pipeline/README.md)。

## 后续阶段（占位）

1. **部署算力云**：48G+ 显存（L20/A100）；基础设施需自己准备
   —— ⚠️ **`run.sh` 只处理 Redis 与 MySQL，完全不碰 Milvus**（见 `run.sh` 里 `redis-server` / `systemctl start mysql` 两段，本文不写行号），
   仓库里也**没有** `docker-compose.yml`，Milvus 必须按官方 compose / 镜像自行拉起后再设 `MILVUS_HOST`
2. **模型微调**：ModelScope 下载 Qwen3-4B 基座 → **本仓自写的 `scripts/train_lora.py`
   （peft + transformers）** 做 LoRA SFT。
   ⚠️ **不是 LLaMA-Factory**：本仓**没有装也没有用** LLaMA-Factory
   （`scripts/train_lora.py` 的文件头第一行就写着"不需要 LLaMA-Factory"）；
   工作目录名 `LLaMA-Factory` 是**遗留名**，别被它误导。
   训练数据的两道闸（**引用有据** + **与评测集不重叠**）在 `scripts/build_sft_data.py` 里；
   训练用的是它现场产出的 `data/sft/train.neg25.jsonl`（**912 条 = 684 正 + 228 负**）与
   `val.neg25.jsonl`（80 条）—— **不是** `data/legal_train.json`（那套未被使用，见上文澄清）。
   ⚠️ **它还不是生产模型**：六臂复验（2026-09-28，同窗口、每臂全新会话）
   通过率 0.7290 / 0.7477（v1 旧数据）/ 0.7290（v5a）/ **0.7477**（v5b）/ 0.7196（v5c）/
   0.7383（**v5d = 截断修好、8192 窗口重训**），两两差 **−1 ~ +2 题**，
   而本项目**同条件抖动地板是 4 题** ⇒ 答案质量**测不出增益**；默认推理仍走云端 27B。
   唯一一个**超过标尺**的正向信号在**分流行为**上：29 题分流子集里基座 24/29、v5d **26/29**
   （+6.9 点，且子集内无退化；A04 五个 LoRA 全修好、A12 只有 v5d 修好）——
   **该信号已被 87 条扰动复测否定**（净翻转 −2、三方向不一致）⇒ 判为措辞敏感的噪声。
   （这里先后写过「根因是缺负样本」「瓶颈是 4B 的规模」—— **都被后续证据改写**：
   ① 真 bug 是训练提示里**没有把法条放进去**（0/796 条）；② 修好后又有 `encode_chat` **右截断**
   把答案切掉（912 条里 514 条只剩 ≤1 个监督 token）；③ v5d 用 8192 窗口重训、监督中位
   1 → **69**、饿死 0/912，**仍然测不出答案侧增益** ⇒ 结论只能是「这套评测分辨不出来」，
   不是「4B 学不会」。**最终版报告**：`docs/FINETUNE-REPORT-FINAL.md`。）
   报告与逐题证据：`eval/results/FINETUNE-SUITE-REPORT.md`、`eval/results/arm3-*.jsonl`。
3. **微调模型接入 RAG**：`LLM_PROVIDER=openai_compat` 指向本地 vLLM/SGLang
4. **RAGAS 评测**：忠实度 / 答案相关性 / 上下文精确率与召回率
5. **知识库动态更新**：按 `role_id` 分区 + 先删后插 + 定时增量同步
6. **可观测性**：Prometheus + Grafana、ELK、SkyWalking

## 已知限制

* **huggingface.co 在本机不可达**，BGE 系列权重需走 ModelScope（`modelscope.cn` 可达）；
  也可设置 `HF_ENDPOINT=https://hf-mirror.com` 作为备选。
* **`BgeM3Embedder` / `BgeReranker` / `ChromaVectorStore` 这条重依赖路线（torch /
  sentence-transformers / chromadb / FlagEmbedding / modelscope）未在本环境安装、未验证。**
  但**真实后端链路本身已经验证过了** —— 走的是**免 torch 的 `OllamaEmbedder`（bge-m3，1024 维）
  + `MilvusVectorStore` + `OllamaClient`**，已在 Windows 与 Ubuntu VM 双环境独立验证
  （见 [`handoff/DELIVERY.md`](handoff/DELIVERY.md)、[`VERIFICATION.md`](VERIFICATION.md)）。
  离线兜底链路同样完整可跑；测试基线（**2026-09-16 本机实跑**，收集 344 / skip 5）
  **339 passed / 5 skipped / 0 failed**（`pytest tests -q`，约 226 s）。
  修复前的复核基线是 329 passed（收集 334）；更早的冻结版数字（300 passed）见
  `handoff/DELIVERY.md` / `VERIFICATION.md`。
  **最新基线（2026-09-29 本机实跑）**：`1229 passed / 11 skipped / 6 deselected / 0 failed`
  （`pytest -q` 全仓，533 s）。同一套代码在算力云（Linux / Python 3.10）上是
  `1239 passed / 1 skipped / 6 deselected / 0 failed`（156 s，见 `eval/results/cloud-suite-20260929.txt`）。
  ⚠️ 这里的 skip **全部**是"本机没有可用的 bash，所以 `bash -n` 跳过"，
  **不是**"用例通过"；deselect 的 6 条是需要真实
  Redis/Milvus 的集成用例（默认安全，见 `tests/conftest.py`）。
  基线演进：339（09-16）→ 974（09-27 §49）→ 997（09-27 §50）→ 1038（09-27 §51）
  → 1210（09-28）→ 1221（09-29 含 D4）→ **1229**（09-29 部署脚本守卫）。
  同一套代码在**云端 Linux / Python 3.10** 上也跑过：`bash -n` 真的执行、
  `world-writable` 那条权限检查也真的生效（本机 Windows 会跳过这两类）。
* HTTP 服务与接口测试所需的 **fastapi / uvicorn / httpx 属轻量依赖**，已随快速开始第 1 步的
  `requirements.txt` 一并安装，**不需要** `requirements-full.txt`；只有真实向量后端与重排模型
  才需要该文件里的 torch / chromadb / sentence-transformers 等重依赖。
* Redis / MySQL 未部署时自动降级为内存 + SQLite，日志中会明确提示。
* `.venv/bootstrap/sitecustomize.py` 与 `tests/conftest.py` 中的临时目录处理是
  **受限沙箱专用兼容层**（某些沙箱禁止写入 `mode=0o700` 创建的目录）：
  在普通环境下可以删除 `sitecustomize.py`，`conftest.py` 的 `tmp_path` 覆盖行为等价。
---

## 快速开始与发布前必做（t103 集成交付补记）

* 网页入口：`http://127.0.0.1:<port>/ui`（未登录会 302 到 `/login`；`/`、`/ui/` 的 302/307 均可）。
* 一条最短路径：起服务 → `/ui` 注册 → 登录 → 选角色 → 新建会话 → 提问（SSE + 引用）→ 刷新看历史 → 重命名/删除 → 知识库「添加」（入库前判定）。
* **发布前必做（硬要求）**：① **`AUTH_REQUIRED=true`（发布门禁）** —— 默认关闭时接口接受任意 `user_id`；② **不要把 Milvus `19530` 暴露公网**（Milvus 无按用户鉴权，三层隔离只对「经应用访问」成立）。
* 交付清单、报告索引与**已知留白**见 `handoff/DELIVERY.md`。
