# 部署目录说明（工单17）

**工单编号**：`人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单`

---

## 目录结构

```
部署/
├── docker/
│   ├── docker-compose.yml   ★继承工单10，补齐 mem_limit/cpus + PERF_* + Prometheus 采集
│   ├── Dockerfile           容器镜像定义（继承工单10）
│   ├── entrypoint.sh        容器入口（继承工单10）
│   ├── healthcheck.py       HEALTHCHECK 脚本（三档探针）
│   ├── prometheus.yml       抓取配置（rag-api /metrics）
│   └── requirements.txt     容器内依赖清单
├── 脚本/
│   ├── 起服务.ps1           ★本机等价部署启动（对应 docker compose up）
│   ├── 本地停止.ps1         ★按 PID 精确停止 + 端口确认 + python 回归基线 + 清缓存
│   ├── 本地等价验证.ps1      （继承工单10；本工单用 起服务.ps1）
│   ├── up.ps1 / down.ps1 / build.ps1 / health.ps1 / logs.ps1   （继承工单10，容器用）
├── 配置/
│   └── .env.example         ★全部环境变量（DEPLOY_* / RAG_* / PERF_*）
├── 日志/                    运行日志与留痕（本地进程.json / python基线.json / parse_tasks.jsonl / 本地运行-API-*.log）
├── 卷/                      本机等价部署的「具名卷」落地目录
│   ├── logs/  sessions/  runtime/  uploads/
├── DEPLOYMENT.md            ★部署文档（工单原文点名要求）
├── OPERATION.md             ★运维文档（工单原文点名要求）
├── README.md                本文件
└── 交付物核验.md             交付物清单与核验结果
```

---

## 两条部署路径

| 路径 | 状态 | 入口 | 说明 |
| --- | --- | --- | --- |
| **Docker Compose** | 交付物（**本机未实跑**） | `部署/脚本/up.ps1` | 本机 `docker` CLI 走 npipe 被沙箱拒绝（退出码 1，证据见 `设计/环境事实.md` §2.1） |
| **本机等价部署** | **实测路径** | `部署/脚本/起服务.ps1` | 同源码、同端口 8200、同卷语义；本工单全部实测数据来自这条路径 |

---

## 本机等价部署的三个要点（踩过的坑）

1. **真实 PID ≠ Start-Process 的 PID**
   `Start-Process python` 起的是个 1 MB 的**启动器 stub**，真正的 uvicorn 进程是它的子进程。
   真实 PID 从 `/healthz` 的 `pid` 字段取，两个都写进 `部署/日志/本地进程.json` ——
   否则停止脚本会留下 **2.4 GB 的孤儿进程**（工单9 就是这么被批评的）。

2. **python 基线必须先记录**
   `部署/日志/python基线.json` 在**起服务之前**记录当前 python PID 集合
   （会话基线约 4 个、均 < 200 MB）。停止脚本只做差集清理；
   **基线文件缺失时绝不做差集**（空基线 = 差集等于「全部 python」，会误杀会话进程）。

3. **`Get-CimInstance` 在本沙箱不可用**
   报「无法从客户端中访问 CIM 资源」，因此**不能**按命令行特征找进程 ——
   一律按 PID + 基线差集处理。

---

## 常用命令

```powershell
cd E:\gao6gongdan\工单17

# 起服务（基座口径，用于复现基准）
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 0 -Tag 基准_优化前

# 起服务（优化后口径）
$env:PERF_MAX_INFLIGHT='20'; $env:PERF_QUEUE_MAX='64'
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1 -Tag 优化后

# 起服务（生产默认限流 8/16）
$env:PERF_MAX_INFLIGHT='8'; $env:PERF_QUEUE_MAX='16'
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1 -Tag 优化后_限流8

# 停服务
pwsh -NoProfile -File 部署/脚本/本地停止.ps1

# 容器路径（本机不可用，仅在生产执行）
pwsh -NoProfile -File 部署/脚本/build.ps1 -Tag 1.7.0
pwsh -NoProfile -File 部署/脚本/up.ps1
pwsh -NoProfile -File 部署/脚本/health.ps1
pwsh -NoProfile -File 部署/脚本/down.ps1
```

---

## 卷映射对照（本机 ↔ 容器）

| 本机路径 | 容器路径 | 容器卷名 | 读写 |
| --- | --- | --- | --- |
| `工单7\研发\data\index` | `/app/data/index` | `rag-finqa-index` | **ro** |
| `部署\卷\logs` | `/app/logs` | `rag-finqa-log` | rw |
| `部署\卷\sessions` | `/app/sessions` | `rag-finqa-session` | rw |
| `部署\卷\runtime` | `/app/data/runtime` | `rag-finqa-runtime` | rw |
| `部署\卷\uploads` | `/app/data/uploads` | `rag-finqa-upload` | rw |
| `部署\日志` | `/app/logs-host` | （绑定挂载） | rw |

---

## ★ 工单17 最重要的一条 compose 改动

工单10 给 `rag-api` 的容器内存上限是 **2g**，而工单10 自己实测的 API 私有内存峰值是
**2.41~2.44 GB** —— **上限比实测占用还小**。容器一进入稳定运行就贴着 limit 跑，
稍有堆积即触发 OOM Killer 重启，**完全吻合工单原文描述的崩溃现象**。

本工单已把默认上限提到 **3g**，并显式写出 `mem_limit` / `memswap_limit` / `cpus`
（Compose V2 非 swarm 模式下 `deploy.resources` 不生效，这是「配了限额却仍被 OOM」的隐蔽原因）。
详见 `部署/docker/docker-compose.yml` 文件头注释与 `优化/瓶颈排查与修复报告.md` §2.5。
