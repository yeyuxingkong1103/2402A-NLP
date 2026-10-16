# 部署 README（工单10 · 金融问答系统）

> 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
> 一句话：把工单7 已跑通的金融问答 RAG 系统容器化，对外提供 **API（8200）** 与 **Web 界面（8201）**。

---

## 1. 30 秒速览

| 我要做什么 | 命令（在 `工单10/` 目录下执行） |
| --- | --- |
| 构建镜像 | `pwsh -NoProfile -File 部署/脚本/build.ps1` |
| 启动全部服务 | `pwsh -NoProfile -File 部署/脚本/up.ps1` |
| 健康检查（一键验四项验收） | `pwsh -NoProfile -File 部署/脚本/health.ps1` |
| 看日志 | `pwsh -NoProfile -File 部署/脚本/logs.ps1 -Follow` |
| 停服务（**卷保留，数据不丢**） | `pwsh -NoProfile -File 部署/脚本/down.ps1` |
| **Docker 不可用时**（本机现状） | `pwsh -NoProfile -File 部署/脚本/本地等价验证.ps1` |

服务起来后：

| 入口 | 地址 | 说明 |
| --- | --- | --- |
| 问答 API | `http://127.0.0.1:8200/api/v1/ask` | `POST`，JSON |
| 健康检查 | `http://127.0.0.1:8200/health` | 含索引/Ollama/Redis/卷 逐项状态 |
| 接口文档 | `http://127.0.0.1:8200/docs` | FastAPI 自带 Swagger UI |
| 中文问答界面 | `http://127.0.0.1:8201/` | 工单7 的人机对话界面 |
| 指标 | `http://127.0.0.1:8200/metrics` | Prometheus 文本格式 |

---

## 2. ⚠️ 本机 Docker 现状（**先读这一段，否则会困惑**）

Docker Desktop 在运行，但 `docker` CLI 访问 daemon 走命名管道
`npipe:////./pipe/dockerDesktopLinuxEngine`，在当前沙箱下**被拒绝**：

```
$ docker ps
permission denied while trying to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
[exit=1]

$ docker compose version          # 纯客户端子命令正常
Docker Compose version v5.5.1
[exit=0]
```

后果与对策：

| 能做的 | 不能做的 | 替代证据 |
| --- | --- | --- |
| ✅ `docker compose config` 校验编排文件（**客户端解析，不需 daemon**） | ❌ `docker build` / `docker run` / `compose up` | 等价本机验证（同源码/同端口/同卷语义） |
| ✅ 全部源码、脚本、配置、文档 | ❌ `docker volume ls` / `docker inspect` | 卷目录 + 健康检查写探针 + Redis 键实证 |

完整证据：`部署/日志/docker提权失败证据.log`；
排查全过程：`部署/过程问题记录.md` 问题 1；
**未达标项如实清单**：`设计/验收标准.md` §五。

---

## 3. 前置条件

| 项 | 要求 | 检查命令 |
| --- | --- | --- |
| Docker | Docker Desktop ≥ 4.x（Compose V2）；**且当前 shell 有 daemon 访问权** | `docker ps` |
| Ollama | 宿主机运行，含 `bge-m3:latest`(1024d) 与 `qwen2.5:3b` | `curl http://127.0.0.1:11434/api/tags` |
| 索引 | 工单7 的 `研发/data/index/bge-m3_1024`（8847 块），或本工单的 `工单7索引` 联接 | `ls 工单7索引/bge-m3_1024` |
| 端口 | 8200 / 8201（+可选 6379 / 9090）空闲 | `Get-NetTCPConnection -LocalPort 8200 -State Listen` |
| 磁盘 | ≥ 2 GB 可用（镜像 ~1 GB + 卷） | `Get-PSDrive E` |

---

## 4. 目录结构

```
部署/
├── docker/
│   ├── Dockerfile           多阶段 / 非 root(uid 10001) / HEALTHCHECK / 时区字体 / PYTHONUNBUFFERED
│   ├── docker-compose.yml   rag-api:8200 + rag-ui:8201 + redis + prometheus(profile)
│   ├── requirements.txt     最小依赖闭包（不含 torch/transformers）
│   ├── entrypoint.sh        目录体检 + 环境快照 + exec
│   ├── healthcheck.py       标准库探针（探 /healthz）
│   └── prometheus.yml       指标抓取配置
├── 配置/.env.example        全部可调项（端口/卷/阈值/会话/资源限制）
├── 脚本/
│   ├── build.ps1  up.ps1  down.ps1  logs.ps1  health.ps1   ← Docker 路径
│   └── 本地等价验证.ps1  本地停止.ps1                       ← Docker 不可用时的替代路径
├── 日志/                    真实构建/运行/pytest/证据日志
├── README.md                本文件
├── 部署文档.md              给新人的完整步骤（含预期输出与回滚）
└── 过程问题记录.md          10 条真实问题（现象→报错原文→排查→解决）
```

---

## 5. 快速开始（Docker 可用时）

```powershell
cd E:\gao6gongdan\工单10
Copy-Item 部署\配置\.env.example 部署\配置\.env     # 按需改端口/密钥/路径

pwsh -NoProfile -File 部署\脚本\build.ps1           # 构建镜像（日志落 部署/日志/构建-*.log）
pwsh -NoProfile -File 部署\脚本\up.ps1              # 起服务（含 compose config 预校验 + 等健康）
pwsh -NoProfile -File 部署\脚本\health.ps1          # 六项检查，逐项 PASS/FAIL

# 真实提问
$body = @{ question = "平安银行2019年年度报告中，报告期末的资产总额是多少？"; session_id = "demo" } |
        ConvertTo-Json -Compress
Invoke-WebRequest http://127.0.0.1:8200/api/v1/ask -Method POST -Body $body `
    -ContentType 'application/json; charset=utf-8' -UseBasicParsing |
    Select-Object -ExpandProperty Content
```

带监控栈：`pwsh -NoProfile -File 部署\脚本\up.ps1 -WithMonitoring`（额外起 prometheus:9090）。

---

## 6. 关键设计（详见 `设计/`）

* **业务逻辑零复制**：`研发/app` 是工单7 原样复用；`研发/service` 只做「HTTP 协议 + 会话落盘 + 健康检查」。
* **一个镜像两个服务**：`rag-ui` 复用同一镜像、只换 `command` → 版本天然一致。
* **索引只读挂载 + 运行期数据独立卷**：索引 `:ro`（防误写），`qa_runtime.sqlite3` 落 `rag-runtime` 卷
  （**这是实测踩坑后加的**，见 `部署/过程问题记录.md` 问题 7）。
* **会话双写**：文件卷（数据不丢失）+ Redis（跨容器实时共享，内置零依赖 RESP 客户端，见问题 8）。
* **大模型在宿主机**：嵌入/生成走 Ollama，镜像不装 torch → 体积小、换模型不动镜像。
* **三档探针**：`/healthz` 存活（探针用）、`/health` 全量（诚实报告）、`/readyz` 就绪（索引不可用 503）。

---

## 7. 排障入口

| 症状 | 先看哪里 |
| --- | --- |
| 容器起不来 | `部署/日志/启动-*.log`（含 compose config 输出）；entrypoint 退出码 **78 = 卷不可写** |
| 服务 503 | `GET /health` 的 `checks.engine`（索引/LLM 哪一项 `ok:false`） |
| 问答慢/超时 | `/health` → `checks.engine.checks.llm`；Ollama 是否只有一个模型驻留 |
| 会话读不到 | `/health` → `checks.session_store.backend`（`file` 表示 Redis 降级了，看 `degraded_reason`） |
| 日志里找异常 | `pwsh -File 部署/脚本/logs.ps1 -ScanErrors`（关键字扫描，产出 `异常扫描-*.log`） |
| Docker 权限 | `部署/过程问题记录.md` 问题 1 + `部署/日志/docker提权失败证据.log` |

---

## 8. 相关文档

* [`部署文档.md`](部署文档.md) —— 完整步骤、预期输出、回滚（给新人）
* [`过程问题记录.md`](过程问题记录.md) —— 10 条真实问题与解法
* [`../设计/README.md`](../设计/README.md) —— 设计与事实文档索引
* [`../测试/测试报告.md`](../测试/测试报告.md) —— 测试结果与真实数字
* [`../优化/部署优化说明.md`](../优化/部署优化说明.md) —— 体积/冷启动/资源优化对比
