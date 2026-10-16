# 部署说明（安装 → 运行 → 测试）

> 工单编号：**人工智能NLP-RAG-Query 理解优化**
> 仓库：`E:\gao6gongdan\工单5`（只读基础：`E:\gao6gongdan\工单4`）
> 交付形态：**本机原生一键启动**（Windows + `run_py.ps1`）为主；Docker 为**可选**交付
> （沿用工单4 的 docker 文件，本机沙箱拒绝 Docker 命名管道，**未构建**）。
> 多轮界面 + 多轮 CLI + FastAPI 三种入口共用一套引擎。

---

## 一、入口总览（四条路径，按推荐顺序）

| # | 入口 | 端口 | 依赖 | 启动命令 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | **多轮 CLI**（最快体验） | — | 零额外依赖 | `pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --chat` | **实测可用** |
| 2 | **标准库 Web 多轮界面** | 8123 | **零第三方依赖**（纯 `http.server`） | `pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode web -Port 8123` | **实测可用**（截图即出自此界面） |
| 3 | **FastAPI 服务**（端点最全） | 8600 | fastapi + uvicorn（已装） | `pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode api -Port 8600` | **实测可用** |
| 4 | Gradio 界面 | 7860 | **需自行安装 gradio** | `pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Port 7860` | **未真机运行**（本机无 gradio，缺依赖时打印安装指引并退出码 3） |

> **为什么默认不是 Gradio**：本机主解释器（`工单1\.venv`，Python 3.11.15）**没有 `gradio`**，
> 而本机红线禁止 `pip install` 裸装（落点是只读的 `工单1`）。因此多轮界面以**纯标准库**实现交付，
> Gradio 作为「装了就能用」的可选增强，**不构成运行依赖**。详见 `部署/界面/README.md`。

---

## 二、安装（本机已具备依赖，只需体检）

```powershell
cd E:\gao6gongdan\工单5
pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py     # 期望「FAIL=0」
```

**环境事实（实测，勿凭提示词臆测）**

| 项 | 值 |
| --- | --- |
| 唯一可用解释器 | `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python **3.11.15**） |
| 统一入口 | `run_py.ps1`（自动设 `PYTHONPATH=研发`、UTF-8 I/O、`PYTHONDONTWRITEBYTECODE=1`） |
| 已装 | pymupdf / numpy / jieba / scikit-learn / faiss 1.7.4 / torch(CPU) / transformers / sentence-transformers / Pillow / fastapi / uvicorn / pytest / requests / openai / httpx |
| **未装** | `gradio`、`streamlit`、`pandas`、`cv2`、`pytesseract`、`paddleocr` |
| 生成模型 | Ollama `qwen2.5:3b` @ `http://127.0.0.1:11434`（**不要关闭**） |
| 嵌入模型 | Ollama `bge-m3:latest`（1024 维，**不要关闭**） |
| 索引 | `研发/data/index/bge-m3_1024/`（FAISS + BM25，**2676 chunk**） |

🔴 **红线**：不要 `pip install`（落点是只读的 `工单1\.venv\Lib\site-packages`）；
需要新依赖时用 `--target 研发/第三方库`，例如
`... -m pip install --target 研发/第三方库 gradio`（联网需提权）。

---

## 三、运行

### 3.1 多轮 CLI（推荐先用它跑通五轮序列）

```powershell
cd E:\gao6gongdan\工单5

# 交互式多轮（每轮都会打印「改写后的独立问句」「策略」「焦点实体」）
pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --chat --session-id demo

# 单次提问（同一 --session-id 连续调用即为多轮追问）
pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --ask "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？" --session-id demo
pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --ask "他参与的哪个工程荣获了国家科技进步一等奖？" --session-id demo
pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --ask "这个公司的法定代表人是谁？" --session-id demo
pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --ask "那武汉力源信息技术股份有限公司呢？" --session-id demo

# 查看 / 清空会话；健康检查
pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --state --session-id demo
pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --reset --session-id demo
pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --health
```

交互模式内置命令：`/help` `/reset` `/state` `/history` `/rewrite on|off` `/quit`。

> 原工单4 的 CLI 也仍然可用，并新增了 `--chat` 转交多轮：
> `pwsh -NoProfile -File run_py.ps1 研发/app/main.py --chat`

### 3.2 多轮 Web 界面（零第三方依赖，端口 8123）

```powershell
pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode web -Port 8123
# 浏览器打开 http://127.0.0.1:8123/
```

界面能力：

* 多轮对话气泡；**侧栏实时展示「本轮 Query 理解」**：改写后的独立问句、改写策略、
  指代消解明细（谁→谁）、焦点实体、上一轮槽位、引用页码、四段耗时（改写/检索/生成/合计）；
* **清空对话**（换新 `session_id` 并清掉服务端该会话状态与缓存）；
* 每条回答下方 **👍 有帮助 / 👎 没帮助**，追加写入 `部署/日志/用户反馈.jsonl`；
* 内置示例按钮（首轮·显式实体 / 指代·他它 / 省略·换实体 / 话题继承·无实体 / 英文提问）。

HTTP 路由：`GET /`、`GET /health`、`POST /api/chat`、`POST /api/feedback`、`POST /api/reset`。

### 3.3 FastAPI 服务（端口 8600，含 `/docs`）

```powershell
pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode api -Port 8600
# 交互式文档 http://127.0.0.1:8600/docs
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 依赖逐项体检（`status=ok` → 200，否则 503，**不伪装**） |
| POST | `/api/chat` | **多轮问答**（工单5）：`{question, session_id, language, use_cache, with_history}` |
| POST | `/api/reset` | 清空某会话的多轮状态与缓存 |
| POST | `/api/ask` | 单轮问答（工单4，保留） |
| POST | `/api/retrieve` | 只检索不生成（诊断用） |
| GET/POST | `/eval` | 工单4 的 16 题 golden 评测 |

`/api/chat` 响应同时给出 `question` / `rewritten_query` / `strategy` / `focus_entity` /
`latency_ms{rewrite,retrieve,generate,total}` / `history` / `state`，便于前端展示改写过程。

### 3.4 Gradio 界面（可选，端口 7860）

```powershell
# 先装（需提权联网；不要裸装）
pwsh -NoProfile -File run_py.ps1 -m pip install --target 研发/第三方库 gradio
pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Port 7860
```
缺 gradio 时会打印中文安装指引并返回**退出码 3**。

---

## 四、测试

```powershell
cd E:\gao6gongdan\工单5
$env:PYTEST_ADDOPTS='-p no:cacheprovider'

# ① 研发自测层（89 passed：工单5 新增 65 项 + 工单4 表格 24 项）
pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests -q

# ② 工单5 验收层：多轮 + 独立会话对照 + 单轮回归16 + 异常输入（四套一次跑）
pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine local --run-tag mt-local

# ③ 多轮一键自检（中文 5 轮 + 中英双语 3 轮；写 部署/日志/多轮自检.json）
pwsh -NoProfile -File run_py.ps1 测试/多轮/多轮自检.py

# ④ 证据页（把实跑 JSON 渲染成可截图的自包含 HTML）
pwsh -NoProfile -File run_py.ps1 测试/多轮/生成截图页.py --run-tag mt-local

# ⑤ 走 HTTP 的同一套评测（需先起界面）
pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine http `
     --base-url http://127.0.0.1:8123 --run-tag mt-http
```

> **延迟测量纪律**：正式验收必须**预热**并**独占 Ollama**（不并发跑其它评测）。
> 冷启动实测 ≈2.3~2.9 s，冷口径必违约；评测脚本会把预热那一次单独记
> `cold_start_ms` 并**排除**在统计之外。

---

## 五、目录与产物

```
部署/
├── 界面/
│   ├── web_app.py        ★ 纯标准库多轮界面（端口 8123）——本工单截图出自此界面
│   ├── gradio_app.py     Gradio 多轮界面（端口 7860，需自行安装 gradio）
│   └── README.md         三入口对照、HTTP 契约、反馈字段、降级路径
├── 脚本/
│   ├── run_web.ps1       统一启动：-Mode web|gradio|api，-Port，-Force
│   ├── serve_api.py      FastAPI（含 /api/chat 多轮端点）
│   ├── run_app.ps1       工单4 启动脚本（保留；-HealthCheckOnly 仍可用于单轮自检）
│   ├── run_app.sh        工单4 bash 版（未实跑，见工单4 已知限制）
│   ├── verify_config_env.py  环境体检
│   ├── rotate_logs.ps1   日志轮转
│   └── 录屏步骤.md        演示视频录制步骤（本机无 ffmpeg，可用 Win+G）
├── 日志/
│   ├── 多轮自检.json      ★ 多轮自检结果（5/5 + 3/3）
│   ├── 用户反馈.jsonl     界面 👍/👎 反馈（首次投票时创建）
│   ├── app.log / error.log / rag_trace.jsonl   运行日志（含工单4 历史）
│   └── 归档/             轮转归档（工单4 历史）
├── docker/               工单4 Docker 定义（**未构建**，可选交付）
├── 交付物核验.md          工单4 的交付物逐项核验（历史留存）
└── 工单4基线部署说明.md    工单4 的部署说明原文（历史留存，供对照）
```

---

## 六、故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `[端口占用] 8123 已被监听：python(…)` | 上一次的服务没退干净 | 换端口（`-Port 8124`）或用 `-Force` 抢占（会结束该进程，慎用） |
| `/health` 返回 503 `E_ENGINE_UNAVAILABLE` | 引擎还没构造好 / 索引缺失 | 等预热完成；确认 `研发/data/index/bge-m3_1024/` 存在 |
| 回答全是「不清楚」 | Ollama 没起来 | 确认 `http://127.0.0.1:11434` 可达且 `bge-m3:latest` / `qwen2.5:3b` 在 |
| 首答特别慢（>3 s） | 冷启动（模型/索引首次加载） | 先预热（CLI 会自动预热；评测脚本自带预热轮） |
| `ModuleNotFoundError: app.multiturn_engine` | 没用 `run_py.ps1`，`PYTHONPATH` 缺 `研发` | 一律经 `run_py.ps1` 调用 |
| pytest 报 `WinError 5`（临时目录） | 本机沙箱下 `tmp_path` / `%TEMP%` 不可写 | 已改用仓内 `研发/data/runtime/_pytest_tmp/`（见 `研发/tests/test_multiturn_core.py::_mk_tmp`） |
| 仓库根残留 `测试/` 之外的截图/产物 | 没按目录纪律落盘 | 截图一律进 `测试/测试截图/`；产物一律进五个中文文件夹内 |

---

## 七、与工单4 的关系（迁移说明）

工单5 **不修改**工单4 的能力，而是在外面套一层：

```
工单5 新增： MultiTurnEngine（多轮状态 + Query 改写）
                    ↓ 改写后的「独立可检索问句」
工单4 复用： QAEngine（混合检索 → RRF → 可答性闸门 → 生成 → 引用）
```

* **单轮不退化**：改写策略为 `none` 时，送入检索的问句与用户原句**逐字相同**，
  链路与工单4 完全一致；实测 16 题回归 **16/16**。
* **不重复加载**：多轮引擎与单轮引擎各自持有单例，底层不重复加载索引；
  多轮引擎把**已消解**的问句交给底层，故底层不再需要它自己的会话状态。
* **附带修复**：`研发/app/core/subject_binding.py` 修掉了「力源的法定代表人被同页保荐机构
  法定代表人污染」的缺陷（详见根 `README.md` §三·改动 3）。

更多细节见根 `README.md`、`设计/多轮对话架构.md`、`测试/测试报告.md`、`部署/界面/README.md`。
