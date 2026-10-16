# 部署说明（安装 → 运行 → 测试 三步法）

> 工单编号：**人工智能NLP-RAG-图像内容解析及检索优化**
> 仓库：`E:\gao6gongdan\工单4`（本文件属 `部署/` 交付物）
> 交付形态：**本机原生一键启动**（Windows + `run_py.ps1`）为主；Docker 为**可选**交付（本机沙箱拒绝
> Docker 命名管道，**未构建**，见 §6）。中文 Web + CLI + FastAPI 三种入口一套引擎。

| 交付项 | 状态 | 证据 |
| --- | --- | --- |
| 一键自检（起服务 → `/health` 200 → 停服务） | **通过（退出码 0）** | `pwsh -NoProfile -File 部署/脚本/run_app.ps1 -HealthCheckOnly` |
| FastAPI 服务 `/health` `/ask` `/retrieve` `/eval` | **实测 HTTP 200** | `部署/日志/自检结果.json`（本次实跑写出） |
| 中文 Web 界面（纯标准库） | **实测 HTTP 200 / status=ok** | 同上（`-Mode web -Port 8125` 实测） |
| 环境前置检查 | **通过（FAIL=0，warn=5）** | `部署/脚本/verify_config_env.py` |
| Docker 镜像构建 | **未构建（沙箱拒绝命名管道）** | `docker compose config` 静态校验通过（exit 0）；原因见 §6 |
| `run_app.sh`（Linux/macOS/WSL） | **未实跑（本机无可用 bash）** | 静态审查；原因见 §8 |

---

## 1. 安装（**只读、零安装**：本机已具备全部依赖）

```powershell
cd E:\gao6gongdan\工单4

# 1) 环境体检：解释器 / 依赖 / 语料 / 索引 / Ollama / CLIP 权重 / 端口 / Docker
pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py
#    期望末行：结论：环境满足运行条件，退出码 0（FAIL=0）
```

**解释器约定**：一切 Python 调用都走仓库统一入口 `run_py.ps1`，它固定唯一可用解释器
`E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15），并设置
`PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`、`PYTHONDONTWRITEBYTECODE=1`（防止向只读的工单1 写 `.pyc`）、
`PYTHONPATH=<仓库根>/研发`。想要换解释器（例如算力云上的 conda）用环境变量：

```powershell
$env:RAG_SCHEDULER_PYTHON = 'C:\path\to\python.exe'
```

### 🔴 禁止裸 `pip install`（实测红线，写错会写坏别人的环境）

* 本机 `sysconfig purelib` = `E:\gao6gongdan\工单1\.venv\Lib\site-packages`
  ——那是**只读的工单1 环境**，裸装会污染它（红线：工单1/2/3 文件不得改动）。
* 确实要装包时**只允许**装到本仓库目录，并用 `PYTHONPATH` 生效：

```powershell
pwsh -NoProfile -File run_py.ps1 -m pip install --target 研发/第三方库 <包名>
```

### 依赖一览（本机实测版本；服务端只用前 10 个）

| 包 | 版本 | 包 | 版本 |
| --- | --- | --- | --- |
| fastapi | 0.109.0 | PyMuPDF | 1.28.2 |
| uvicorn | 0.27.0 | jieba | 0.42.1 |
| pydantic | 2.5.3 | scikit-learn | 1.8.0 |
| numpy | 1.26.4 | requests | 2.33.1 |
| faiss-cpu | 1.7.4 | openai | 1.10.0 |

> 离线**图像解析**阶段还需要 torch 2.11.0+cpu / transformers 4.57.6 / Pillow 12.1.1 / huggingface_hub 0.36.2
> （多模态编码在运行时才 import，故 **API 服务不需要它们**）。清单见 `部署/docker/requirements*.txt`。
> 本机**未安装**图像文字识别类包（pytesseract、深度学习识别包、opencv 均缺失）与 streamlit/gradio —— 详见 §8 方法学限制。

---

## 2. 运行

### 2.1 一条命令：FastAPI 服务（推荐，端点最全）

```powershell
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode api -Port 8600
#   中文问答页    http://127.0.0.1:8600/
#   健康检查      http://127.0.0.1:8600/health      → 200 + {"status":"ok"}
#   OpenAPI 文档  http://127.0.0.1:8600/docs
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 依赖逐项体检；`status=ok` → **200**；未就绪 → **503** 并在 body 里说明缺哪一项（不伪装 ok） |
| POST | `/ask` | 单次问答（拒答也是 200：那是正确行为） |
| POST | `/ask/stream` | SSE；**切片下发**（首个 delta 不是首字延迟，真值取 `first_token_ms`） |
| POST | `/retrieve` | 只检索不生成（诊断/检索精度留痕） |
| GET | `/questions` | 示例问题清单（含第 5、6 题） |
| GET | `/eval` | 评测端点口径说明（含 golden 路径与 sha256） |
| POST | `/eval` | 触发 16 题 golden 评测（同步；判分复用 `测试/common/golden.py`；已在跑 → 409） |

示例：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8600/ask -ContentType 'application/json' `
  -Body '{"question":"武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？","lang":"zh"}'
```

### 2.2 中文 Web 界面（纯标准库，依赖最少）

```powershell
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode web -Port 8123
#   http://127.0.0.1:8123/  → 200，页面标题「招股说明书图像内容 RAG 问答系统（工单4）」
```

### 2.3 CLI

```powershell
# 单次提问（中/英）
pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask "武汉兴图新科电子股份有限公司注册资本是多少？"
pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?" --lang en

# 批量 / 交互 / 只检索 / 健康检查
pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask-file 研发/data/eval/golden_qa.jsonl --out 部署/日志/batch.jsonl
pwsh -NoProfile -File run_py.ps1 研发/app/main.py --interactive
pwsh -NoProfile -File run_py.ps1 研发/app/main.py --retrieve "增长率最快的是哪个行业？"
pwsh -NoProfile -File run_py.ps1 研发/app/main.py --health

# 或者用统一封装（带自检留痕）
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode cli -Question "武汉兴图新科电子股份有限公司注册资本是多少？"
```

---

## 3. 测试

```powershell
# 3.1 部署自检（验收用；起服务→查 /health 200+ok→冒烟全部端点→停服务→退出码）
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -HealthCheckOnly            # 期望 exit 0
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -HealthCheckOnly -SmokeAll  # 加跑 /ask /retrieve /ask/stream /eval
#     留痕：部署/日志/自检结果.json

# 3.2 三级测试一键串跑（离线 → 在线 → 用户；服务未起时脚本会自拉起，跑完自动关）
pwsh -NoProfile -File 测试/run_tests.ps1
#     T6 实测：离线 140 passed｜在线 12 passed, 2 xfailed｜用户 7 passed, 1 xfailed；exit 0

# 3.3 16 题端到端评测（准确率 + 延迟 + 逐题明细；**需要先有一个在跑的服务**——脚本走 /health 与 /api/ask）
pwsh -NoProfile -File run_py.ps1 测试/在线/run_16q_eval.py --run-tag accept --base-url http://127.0.0.1:8600
#    默认 base-url 是 http://127.0.0.1:8123（纯标准库 Web）；FastAPI 服务在 8600，故显式给 --base-url
#     产物：测试/测试数据/{qa_results.jsonl,qa_results.csv,accuracy_report.json,negatives.jsonl,bilingual.jsonl}
```

> **延迟测量纪律**：先预热、**独占 Ollama**（不并发跑别的评测），否则数字不可用于验收。
> 服务未起时 `测试/run_tests.ps1` 会自行拉起 8123（等待上限 120 s）。

---

## 4. 本机沙箱与提权说明（**实测结论，不写会让人卡死**）

本机**有网**，但 DSH 沙箱在 `workspace-write` 模式下对外网出口与本地命名管道有限制。
需要联网或访问 Docker 的命令，**必须由调用方显式提权**：
工具参数 `sandbox_permissions='danger-full-access'` + 一句 justification。

| 场景 | 是否需提权 | 实测证据 |
| --- | --- | --- |
| 纯本机运行：起服务、问答、跑测试、建/读索引 | **不需要** | 本文件 §2/§3 的命令均已在本机实跑通过 |
| `pip install` / `pip download`（联网） | 需要；**且落点有红线**（见 §1） | 未提权时 `pip download` 实测 180~300 s 无输出被强杀（`设计/环境事实.md` §4.3） |
| 下载 CLIP 权重（hf-mirror，718 MB 级） | 需要（或按 §5 的 UA 姿势） | 见 §5 |
| `docker`（构建/运行/查看容器） | 需要 | `docker version` → `permission denied … npipe:////./pipe/dockerDesktopLinuxEngine`（退出码 1） |

> 本会话（T8）**未做任何提权**（该会话不允许审批弹窗），因此 Docker 未构建——这不是"没试"，是"试了被拒"，
> 原始报错见 §6。

---

## 5. 拿 CLIP 权重：⚠️ 两条实测坑（照抄通用教程必踩）

```powershell
# ✅ 正确姿势（每个请求自带浏览器 UA，幂等，自带权重体检）
pwsh -NoProfile -File run_py.ps1 研发/scripts/download_clip.py
#   基准：sha256(pytorch_model.bin)=3b8ca5a9ba36830e008bef8f980184cf29eeff986f1f3c1be9e5153f8712e7cc
#   T8 复核实测：文件 771,304,190 B，sha256 与基准**一致**
#   （早期转述的 753,177,983 B 是 hf-mirror 上报的 Content-Length，非落盘体积 → 体积以本地实测为准、正确性以 sha256 判等）
```

1. 📌 **hf-mirror 有 User-Agent 门禁**（`设计/环境事实.md` §4.4，captain 与 T2 两测结论相反，根因已定位）：
   带**文件路径**的请求用默认 `python-urllib` UA → **HTTP 403（cloudflare）**；带 `User-Agent: Mozilla/5.0` → **HTTP 200**。
   裸域名 (`https://hf-mirror.com`) 不带 UA 也 200 —— 只打裸域名看不出差别。`研发/scripts/download_clip.py` 已自带浏览器 UA。
2. 📌 **不要照 `设计/环境事实.md` §4.2 的 `huggingface_hub.snapshot_download` 命令下权重**：
   该路径在本机会因软链落点 `snapshots/…/tmpXXXX/dummy_file_src` 触发 **`PermissionError`** 失败（**与网络无关**），
   该节已就地标注为「待修正」。探测矩阵（可达性/体积/速率）仍然有效。

---

## 6. Docker（文件齐备；**本机未构建**，如实标注）

```powershell
$ docker version
Client: Version 29.7.2 …（客户端信息正常）
permission denied while trying to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine        → 退出码 1
```

* 命名管道 `\\.\pipe\dockerDesktopLinuxEngine` **存在**，但沙箱下访问被拒 → **未执行 `docker build` / `up`**。
  **严禁**把静态校验说成「已构建成功」。
* 已做的**静态校验**（可复现，实测 exit 0）：

```powershell
docker compose -f 部署/docker/docker-compose.yml config              # → 解析通过（73 行），exit 0
docker compose -f 部署/docker/docker-compose.yml config --services   # → rag-api
docker compose -f 部署/docker/docker-compose.yml --profile web config --services   # → rag-api / rag-web
```

* 文件：`部署/docker/Dockerfile`、`docker-compose.yml`、`requirements.txt`、`requirements-multimodal.txt`；
  运行说明与设计取舍见 **`部署/docker/README.md`**（含真机上的构建/启动命令）。
* 镜像策略：**不烘焙**语料/索引/权重（只读卷挂载），嵌入与生成走**宿主机 Ollama**；
  服务镜像不装 torch/transformers（仅离线图像解析需要，`--build-arg WITH_MULTIMODAL=1` 才装）。

---

## 7. 端点与启动命令速查

| 目标 | 命令 | 期望 |
| --- | --- | --- |
| 环境体检 | `pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py` | 退出码 0，FAIL=0 |
| 部署自检（**验收命令**） | `pwsh -NoProfile -File 部署/脚本/run_app.ps1 -HealthCheckOnly` | 退出码 0，`部署/日志/自检结果.json` 全 PASS |
| FastAPI 服务 | `pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode api -Port 8600` | `/health` 200 + `status=ok` |
| 中文 Web | `pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode web -Port 8123` | `/` 200、`/health` 200 |
| CLI | `pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask "…"` | 答案 + `[文件名: 第N页]` 引用 |
| 三级测试 | `pwsh -NoProfile -File 测试/run_tests.ps1` | 退出码 0（三级全通过） |
| **日志归档/轮转** | `pwsh -NoProfile -File 部署/脚本/rotate_logs.ps1`（先 `-DryRun` 预览） | 退出码 0；`app.log` 归档为 gzip + 保留最近 3000 行；留痕 `部署/日志/轮转记录.json` |
| Linux 等价脚本 | `bash 部署/脚本/run_app.sh --health-check-only` | **本机未实跑**（见 §8） |

---

## 8. 已知限制与缺陷（**如实登记，不夸大、不掩盖**）

### 8.1 评测口径与结论（T6 实跑基线，`测试/测试报告.md`）

| 指标 | 目标 | 实测（T6） | 结论 |
| --- | --- | --- | --- |
| 16 题准确率（确定性判分，非 LLM 自评） | ≥90% | **16/16 = 100%** | 达标 |
| 单次问答延迟（客户端 wall，`--no-cache` 全量） | 逐题 ≤3000 ms | **P50 595.8 / P95 929.8 / max 1062.8 ms**（服务端 total P50 582.8 / P95 903.2 / max 1049.5）；超线 **0** 题 | 达标 |
| 检索命中（权威层，证据原文落在 Top-5） | 报告值 | **14/16**（BM25 14 / 向量 11） | 见下 ③ |
| 引用可回溯 | 引用页承载证据 | **16/16** | 达标 |
| 拒答红线（4 条负例） | 全部回「不清楚」 | **3/4**（N1 未通过） | 未通过 → 见下 ① |
| 中英文问答 | 同题中英各一次 | 中文 3/3；**英文 0/3**（原子口径） | 部分未通过 → 见下 ② |
| 索引规模 | — | `chunk_total=2676`（text 1981 / table 618 / image 77；diagram 12 / chart 22 / picture 43） | — |

> 优化环节（T7）在**同一协议**下复测：准确率 16/16 未退化；**拒答 4/4**、**英文 3/3 连续两次一致**、
> 权威层命中 14/16（BM25 14 / 向量 12）；延迟含英文三题的 wall P50 776.5/669.0 ms、max 2017.8/1971.1 ms（两次冷进程全量），
> 超线 0 题。明细见 `优化/评估结果/优化报告.md`。**上表 T6 数字仍是验收引用的基线**，两套口径分列，未互相覆盖。

### 8.2 三条已定位的功能缺陷（T6 实测；T7 修复并复测）

1. **N1 泛化名实体未拒答（T6 实测）**：问「某某科技有限公司的注册资本是多少？」→ 未拒答，返回「注册资本为100万元」
   并引用 PDF1 第 64 页（无关页）；根因是 `unknown_entity` 闸门不覆盖泛化/占位名。T6 口径拒答红线 **3/4**。
   **状态：T7 已修**（泛化名/占位名模式 + 强指向性字段却点不出语料主体的硬前置校验），复测 **4/4**。
2. **英文路径缺口（T6 实测）**：英文 **0/3**（中文 3/3）——含静默回落中文（id 543）、枚举不完整（id 5 只答 4 个部门）、
   英文图像意图未命中（id 6 被拒答），且跨运行不稳定。细节见 `测试/测试报告.md` §8.1。
   **状态：T7 已修**（译文结构校验 + 一次重试、显式 `language_degraded` 禁止静默回落、数值单位等价护栏），
   复测 **3/3 且两次独立冷进程一致**。
3. **向量通道弱于 BM25（T6 实测）**：权威层 **11/16 vs 14/16**。**状态：T7 部分改善**（向量查询富化后 **12/16**；
   RRF 重加权五组扫描反而更差，故保留 1.0×1.0，未用调参掩盖）。

### 8.3 两条方法学限制（与实现方式绑定，不要指望调参解决）

* **本机无图像文字识别依赖**：第 6 题页 72 的图表数值来自**像素级几何抽取 + 正文锚点定标 + 视觉复核**；
  第 5 题页 39 框图文本按 **x 列聚类复原**；位图内的文字走**字形模板匹配**。
  个别识别有残差（页 72 的 IC卡 -2.0% 印刷值被读成 -2.6%，**如实记 warning，不覆盖几何值**）。
  **不得声称使用了图像文字识别（本机无此依赖）。**
* **3D 饼图不做角度反算**：透视失真下只登记标签与数值，不做定量反推。

### 8.4 交付形态限制

* **Docker 未构建**（§6）：沙箱拒绝 Docker 命名管道；文件齐备 + `compose config` 静态校验通过；未提权、未构建。
* **`部署/脚本/run_app.sh` 未实跑**：本机 `bash.exe` 只是 WSL 转发器，沙箱下 `wsl -l -q` 返回 `E_ACCESSDENIED`
  → 该脚本为**静态审查**（与 `run_app.ps1` 行为对齐），实跑证据由 PowerShell 版提供。
* **演示视频**：按用户口径已改为**截图留痕**（`测试/测试截图/` 共 25 张 PNG）；如仍要 mp4，
  录制步骤见 `部署/脚本/录屏步骤.md`（本机 **无 ffmpeg**，用 Win+G）。
* **RAGAS 未运行**：依赖不可用，报告中不出现任何 RAGAS 数值。
* **冷启动 ≠ 验收口径**：环境固有冷启动首字曾达 3430.8 ms；验收一律用**预热**口径，且需独占 Ollama。
* **日志为离线轮转（进程内未轮转）**：结构化日志是追加式实现（为多进程写入安全），单文件会持续增长
  （实测 12.5 小时到 40 MiB）→ 由 `部署/脚本/rotate_logs.ps1` 在**服务空载窗口**做 gzip 归档 + 保留最近 3000 行；
  若服务正在写入，脚本会在截断前做 2 秒活跃度采样并**放弃改写活动文件**（归档仍保留），不丢在途日志。
  已实跑一次：`app.log` 40.04 MiB → 1.59 MiB（归档 2.91 MiB，回读校验一致），见 `部署/日志/日志字段说明.md` §5。

---

## 9. 常见故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `--health` 输出 `"status": "degraded"` | 索引/LLM 依赖未就绪（body 的 `checks.*.reason` 会点名） | 看 `checks` 里 `ok:false` 的那项：索引缺失先跑 `研发/scripts/build_index.py --incremental --include-images`；Ollama 未起先 `ollama serve` |
| 起服务报 `[索引缺失]`（退出码 3） | 索引未构建 | 同上；或 `研发/scripts/build_index.py --verify-only` 只读体检 |
| 端口被占用 | 8123/8600 上有旧进程 | 换端口：`-Port 8601`；或先确认占用者（`netstat -ano \| Select-String ':8600'`） |
| 首次问答很慢（秒级 → 3 s+） | 冷启动（加载索引/模型） | 等 `/health` 就绪并先问一次预热；不要用冷启动数字做验收 |
| `docker` 报 `permission denied … npipe` | 沙箱拒绝命名管道 | 提权 `sandbox_permissions='danger-full-access'`；或改用本机原生运行（不依赖 Docker） |
| `pip install` 卡住 / 写坏工单1 | pip 联网挂死 + 落点是只读 venv | 用 `--target 研发/第三方库`，并给 `Start-Process` 硬超时（见 `设计/环境事实.md` §4.3） |
| 下权重 403 | hf-mirror 的 UA 门禁 | 用 `研发/scripts/download_clip.py`（自带浏览器 UA，见 §5） |

---

## 10. 相关文档

| 想了解什么 | 文件 |
| --- | --- |
| 交付物逐项核验（含未完成项与原因） | `部署/交付物核验.md` |
| 项目总览 / 快速开始 / 结果结论 | `README.md`（顶层） |
| **索引/解析侧口径与已知限制** | **`研发/README.md`** |
| Docker 与设计取舍 | `部署/docker/README.md` |
| **日志字段/审计口径/轮转策略** | **`部署/日志/日志字段说明.md`** |
| 演示录屏（可选） | `部署/脚本/录屏步骤.md` |
| 测试报告（16 题逐题明细/延迟/拒答/英文） | `测试/测试报告.md` |
| 优化前后对比 | `优化/评估结果/优化报告.md` |
| 环境实测事实与复现命令 | `设计/环境事实.md` |
| 接口契约（端点/字段/错误码） | `设计/接口设计.md` |
