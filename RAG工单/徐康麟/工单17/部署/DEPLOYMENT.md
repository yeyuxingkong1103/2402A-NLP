# DEPLOYMENT.md —— 部署文档

**工单编号**：`人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单`
**目标目录**：`E:\gao6gongdan\工单17\`
**适用读者**：第一次接手本服务的运维 / 后端工程师

---

## 0. 一分钟速览

| 项 | 值 |
| --- | --- |
| 服务 | 金融问答系统（RAG）FastAPI 服务，端口 **8200** |
| 代码 | `研发/service/`（部署层）+ `研发/app/`（RAG 内核，来自工单7） |
| 索引 | `E:\gao6gongdan\工单7\研发\data\index\bge-m3_1024`（**8847 块 / 1024 维，只读引用，不复制**） |
| 外部依赖 | Ollama `127.0.0.1:11434`（`qwen2.5:3b` + `bge-m3:latest`）、Redis `127.0.0.1:6379`（可选，挂了自动降级文件卷） |
| 解释器 | `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15） |
| 内存 | 实测私有内存峰值 **≈2.4 GB** ⇒ 容器限额必须 **≥3g**（见 §5 的 OOM 事故说明） |
| 启动耗时 | 索引加载 + 预热实测 **约 22 秒**（`/healthz` 就绪） |

---

## 1. 部署方式一：Docker Compose（生产推荐）

### 1.1 前置条件

```powershell
docker version          # 需要能访问 Docker API
docker compose version  # Compose V2
```

> ⚠️ **本机实测：docker CLI 不可用**（走命名管道 `npipe:////./pipe/dockerDesktopLinuxEngine`，
> 沙箱拒绝，退出码 1）。完整证据见 `设计/环境事实.md` §2.1。
> **本工单的实测数据全部来自「方式二：本机等价部署」**（同源码、同端口、同卷语义）。

### 1.2 步骤

```powershell
cd E:\gao6gongdan\工单17

# ① 准备环境变量（可选；所有键都有默认值，见 部署/配置/.env.example）
Copy-Item 部署\配置\.env.example .env
# 按需修改：API_MEMORY（默认 3g）、PERF_MAX_INFLIGHT（默认 8）、PERF_OLLAMA_KEEP_ALIVE（默认 5m）

# ② 构建镜像
pwsh -NoProfile -File 部署/脚本/build.ps1 -Tag 1.7.0

# ③ 起服务（rag-api + rag-ui + redis）
pwsh -NoProfile -File 部署/脚本/up.ps1

# ④ 健康检查
pwsh -NoProfile -File 部署/脚本/health.ps1
Invoke-WebRequest http://127.0.0.1:8200/health -UseBasicParsing | Select-Object -Expand Content

# ⑤ 需要监控面板时（Prometheus 在 profile=monitoring 下）
docker compose -f 部署/docker/docker-compose.yml --profile monitoring up -d prometheus
# → http://127.0.0.1:9090  （抓取 rag-api 的 /metrics）

# ⑥ 停服务（保留数据卷）
pwsh -NoProfile -File 部署/脚本/down.ps1
```

### 1.3 索引数据怎么进容器

compose 里索引走**具名卷 `rag-index` + 宿主绑定挂载**两种方式（见 `volumes` 段）：

```powershell
# 方式 A（推荐）：绑定挂载 —— 直接把宿主机上已建好的工单7 索引只读挂进容器
$env:HOST_INDEX_DIR = 'E:\gao6gongdan\工单7\研发\data\index'
pwsh -NoProfile -File 部署/脚本/up.ps1

# 方式 B：先灌进具名卷
docker run --rm -v rag-finqa-index:/dst -v E:\gao6gongdan\工单7\研发\data\index:/src:ro `
  alpine sh -c "cp -a /src/. /dst/"
```

> **索引目录必须只读挂载**（`:ro`）。运行期 sqlite（`qa_runtime.sqlite3`）与答案缓存
> 必须落在**可写的 `rag-runtime` 卷**上 —— 写进索引卷会失败（工单10 的实测坑）。

---

## 2. 部署方式二：本机等价部署（本机唯一可用的方式）

Docker 不可用时的替代路径：**同一份源码、同一个端口、同样的卷语义**，
用宿主 Python 直接跑 uvicorn。

```powershell
cd E:\gao6gongdan\工单17

# 起服务（默认 PERF_ENABLED=0 = 工单10 基座行为）
pwsh -NoProfile -File 部署/脚本/起服务.ps1

# 起服务（工单17 优化后口径）
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1 -Tag 优化后

# 生产默认限流参数（max_inflight=8 / queue_max=16）
$env:PERF_MAX_INFLIGHT='8'; $env:PERF_QUEUE_MAX='16'
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1 -Tag 优化后_限流8

# 停服务（按 PID 精确停止 + 端口确认 + python 回归基线 + 清缓存）
pwsh -NoProfile -File 部署/脚本/本地停止.ps1
```

脚本做的事（都可在日志里核对）：

| 步骤 | 说明 |
| --- | --- |
| 准备卷目录 | `部署/卷/{logs,sessions,runtime,uploads}`（对应容器里的 4 个具名卷） |
| 设环境变量 | `RAG_DATA__INDEX_DIR` 指向工单7 真实索引；`DEPLOY_*` 指向卷目录；`DEPLOY_REDIS_URL=redis://127.0.0.1:6379/15` |
| 记录 python 基线 | `部署/日志/python基线.json`（供停止脚本做差集，**绝不误杀会话基线进程**） |
| 启动 uvicorn | `python -m uvicorn service.api:app --host 127.0.0.1 --port 8200 --no-access-log` |
| 记录真实 PID | `Start-Process` 出来的 PID 是 1 MB 的**启动器 stub**，真实服务进程 PID 从 `/healthz` 的 `pid` 字段取，两个都记进 `部署/日志/本地进程.json` |
| 等就绪 | `/healthz` = 200 且 `/readyz` = 200（索引就绪） |

---

## 3. 配置项（全部环境变量，代码里不写死）

完整清单见 `部署/配置/.env.example`。最关键的几项：

| 变量 | 默认 | 作用 | 调整建议 |
| --- | --- | --- | --- |
| `PERF_ENABLED` | `1`（本工单）/ `0`（起服务脚本默认） | **性能优化总开关**；`0` = 工单10 基座行为 | 对照实验时用 0 |
| `PERF_MAX_INFLIGHT` | `8` | 同时在途请求上限，超出走 429 | 单 GPU Ollama 串行推理，**不建议超过 CPU/GPU 并发能力** |
| `PERF_QUEUE_MAX` | `32` | 有界等待位；满了就直接 429（不堆积） | 想「宁可排队不要拒绝」就调大，但要接受 P95 上升 |
| `PERF_ANSWER_CACHE_MAX` | `256` | 答案内存缓存条数上限（LRU） | 内存紧张就调小 |
| `PERF_ANSWER_CACHE_TTL_S` | `1800` | 答案缓存 TTL | 语料更新频繁就调小 |
| `PERF_OLLAMA_KEEP_ALIVE` | `5m` | Ollama 模型驻留时长 | **显存紧张调到 `30s`**；基座是写死的 `30m` |
| `PERF_PARSE_WORKERS` | `2` | 文档解析工作线程数 | 上传量大再加，但会抢 CPU |
| `PERF_PARSE_QUEUE_MAX` | `32` | 解析队列上限，满了返回 429 | |
| `API_MEMORY`（compose） | `3g` | 容器内存上限 | **不得低于 2.6g**（实测私有内存 2.4 GB） |
| `API_CPUS`（compose） | `2.0` | 容器 CPU 上限 | |

---

## 4. 验证部署成功

```powershell
# ① 进程存活
(Invoke-WebRequest http://127.0.0.1:8200/healthz -UseBasicParsing).Content
# → {"status":"alive","pid":30020,"uptime_s":...}

# ② 索引就绪
(Invoke-WebRequest http://127.0.0.1:8200/readyz -UseBasicParsing).StatusCode   # → 200

# ③ 健康检查（每一项都要 ok=true）
$h = (Invoke-WebRequest http://127.0.0.1:8200/health -UseBasicParsing).Content | ConvertFrom-Json
$h.status; $h.checks.engine.ok; $h.checks.session_store.ok; $h.checks.volume_index.ok

# ④ 真问答（首次约 1.7 s，第二次命中缓存约 30 ms）
$body = @{ question='平安银行2019年的营业收入是多少？'; session_id='deploy-check' } | ConvertTo-Json
Invoke-WebRequest http://127.0.0.1:8200/api/v1/ask -Method Post -Body $body `
  -ContentType 'application/json; charset=utf-8' -UseBasicParsing | Select-Object -Expand Content

# ⑤ 性能自观测
(Invoke-WebRequest http://127.0.0.1:8200/api/v1/perf -UseBasicParsing).Content | ConvertFrom-Json
```

**判定标准**：`/health.status = "ok"`、`checks.*.ok` 全为 `true`、
`checks.engine.checks.index.chunks = 8847`。

---

## 5. ⚠️ 部署避坑清单（全部来自实测）

| # | 坑 | 现象 | 处置 |
| --- | --- | --- | --- |
| 1 | **容器内存上限比实测占用还小** | 服务跑一会儿就被 OOM Killer 杀掉、反复重启 | 工单10 给 `rag-api` 限的是 **2g**，而实测私有内存 **2.41~2.44 GB**。本工单已把默认提到 **3g** |
| 2 | **只写 `deploy.resources` 不写 `mem_limit`** | 明明配了限额却不生效（或反过来，限额在小规模部署里根本没应用） | 非 swarm 模式下 `deploy.resources` 不生效；compose 里已同时给 `mem_limit` / `memswaplimit` / `cpus` |
| 3 | **运行期 sqlite 写进只读索引卷** | `/health` 报 `volume_index` 只读、`storage.ready` 缺失 | 必须设 `DEPLOY_RUNTIME_DIR` 指向**可写卷**（compose 里是 `rag-runtime`） |
| 4 | **`DEPLOY_INDEX_DIR` 留默认 `/app/data/index`** | 本机等价部署时报「索引卷不可读」 | 本机跑必须指向工单7 的真实索引绝对路径 |
| 5 | **Ollama 模型双驻留** | `nvidia-smi` 显示显存被吃满、问答变慢 | `keep_alive` 要短（`5m`）；切换生成/嵌入模型时先 `POST /api/v1/admin/unload`。**绝不允许同时驻留 2 个模型** |
| 6 | **索引目录被复制** | 磁盘暴涨、索引版本分叉 | 索引**只读引用**，不复制（`_共享约定.md §3`） |

---

## 6. 回滚

```powershell
# 方式 A：只关优化（不改代码）—— 行为回到工单10 基座
$env:PERF_ENABLED='0'
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 0

# 方式 B：整包回滚代码（基座快照 = 优化前的 app/ 与 service/ 全量副本）
robocopy 研发\基座快照\app     研发\app     /E /NFL /NDL /NJH /NJS
robocopy 研发\基座快照\service 研发\service /E /NFL /NDL /NJH /NJS
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 0
```

> `研发/基座快照/` 保存的是**优化前**（工单10）的 `app/` 与 `service/` 原始文件，
> 用于复现基准数据与紧急回滚。
