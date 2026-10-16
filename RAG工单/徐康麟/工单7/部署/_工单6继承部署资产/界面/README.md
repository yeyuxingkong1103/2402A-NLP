# 部署/界面 —— 多轮对话界面（工单5 · 人工智能NLP-RAG-Query 理解优化）

本目录交付**多轮对话**的三个可视化/服务入口，以及它们的降级路径说明。
核心逻辑全部复用研发侧冻结契约 `设计/接口契约_冻结.md`（§1.3 `MultiTurnEngine` / §2 `POST /api/chat`），
界面层**只做 HTTP/交互适配**，不复制任何业务逻辑。

---

## 1. 三个入口一览

| 交付 | 文件 | 默认地址 | 依赖 | 定位 |
| --- | --- | --- | --- | --- |
| Gradio 界面（首选） | `部署/界面/gradio_app.py` | `http://127.0.0.1:7860` | **需要 gradio**（本机未装） | 功能最全：多轮气泡、改写面板、反馈按钮、重新生成 |
| 标准库界面（兜底） | `部署/界面/web_app.py` | `http://127.0.0.1:8123` | **零第三方依赖**（仅标准库） | 必可用；同源 `fetch` 直连进程内引擎 |
| FastAPI 服务 | `部署/脚本/serve_api.py`（captain 维护） | `http://127.0.0.1:8600` | fastapi + uvicorn | 供程序调用/评测；`POST /api/chat`（多轮）、`/docs` |

三者共用**同一个** `MultiTurnEngine`（进程内单例，启动时构造一次 + `warmup()` 预热一次），
互不依赖：`web_app.py` **不**调用 8600 服务，`gradio_app.py` **不**调用 8600 服务。

---

## 2. 启动命令（推荐用启动器）

```powershell
# 工作目录 = 仓库根 E:\gao6gongdan\工单5
pwsh -NoProfile -File 部署/脚本/run_web.ps1                   # 默认 -Mode web（8123）
pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Port 7860
pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode api -Port 8600
pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode web -Port 8123 -NoWarmup   # 跳过预热（仅调试）
pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Force              # 端口被占则先清占用者
```

`run_web.ps1` 做什么：切换工作目录 → 设置 UTF-8 / `PYTHONDONTWRITEBYTECODE` / `PYTHONPATH`（含
`研发/第三方库`，若存在）→ **检查端口占用**（默认只提示、不擅自杀进程，`-Force` 才抢占）→
前台调用统一入口 `run_py.ps1`（保证解释器 = 工单1 venv，即 `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`）。

等价的直接启动方式：

```powershell
pwsh -NoProfile -File run_py.ps1 部署/界面/web_app.py    --host 127.0.0.1 --port 8123
pwsh -NoProfile -File run_py.ps1 部署/界面/gradio_app.py --host 127.0.0.1 --port 7860 [--share] [--no-warmup]
pwsh -NoProfile -File run_py.ps1 部署/脚本/serve_api.py  --host 127.0.0.1 --port 8600
```

---

## 3. 依赖情况与降级路径（重要）

| 包 | 本机状态 | 影响 |
| --- | --- | --- |
| `gradio` | **未装** | `gradio_app.py` 不可用 → 走兜底 |
| `streamlit`、`pandas` | 未装 | 本目录不依赖它们 |
| fastapi / uvicorn | 已装 | `-Mode api` 可用 |
| 其它（标准库） | 天然可用 | `web_app.py` 永远可用 |

**gradio 装不上时怎么办（三级降级）**：

1. **首选**：`-Mode gradio`（7860）。缺 gradio 时该命令会打印中文安装指引并以**退出码 3** 结束；
2. **安装到仓库私有目录**（禁止裸 `pip install`，落点是只读的工单1 venv）：
   ```powershell
   pwsh -NoProfile -File run_py.ps1 -m pip install --target 研发/第三方库 gradio
   # 需要联网/提权；装好后 run_web.ps1 会自动把 研发/第三方库 加进 PYTHONPATH
   pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Port 7860
   ```
3. **兜底（必可用）**：`-Mode web`（8123），零第三方依赖，功能覆盖
   多轮气泡 / 改写可见 / 引用页码与耗时 / 👍👎 反馈 / 清空对话。

---

## 4. 界面能力对照

| 能力 | `gradio_app.py`（7860） | `web_app.py`（8123） |
| --- | --- | --- |
| 多轮对话（服务端会话状态） | ✅ `gr.Chatbot` + `gr.State(session_id)` | ✅ 气泡区（用户右 / 助手左） |
| 改写后查询 / 策略 / 焦点实体可见 | ✅ 独立 Markdown 面板 + 指代消解明细 | ✅ 每轮气泡内 + 右侧栏 |
| 引用页码 + 响应耗时（ms） | ✅（含 rewrite/retrieve/generate/total 四段） | ✅ 同 |
| 每轮 👍/👎 反馈 | ✅（作用于最后一轮，写明轮次） | ✅（每条回答下独立按钮） |
| 重新生成（`use_cache=False`） | ✅ 按钮 | ➖（接口支持 `use_cache`，未设按钮） |
| 清空对话（`engine.reset`） | ✅ | ✅（同时换新 session_id） |
| 空输入处理 | 面板中文提示，不检索 | HTTP 400 + 中文提示，不检索 |
| 依赖 | gradio | 无 |

---

## 5. `web_app.py` 的 HTTP 契约

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 单页中文界面（内嵌 HTML，无 CDN） |
| GET | `/health` | `{"status":"ok", ...}`；引擎未就绪 → **503** + `status=degraded` + 中文原因 |
| POST | `/api/chat` | 入参 `{"question","session_id","language","use_cache"}`；出参 = 契约 §2 字段 + `history` + `state` + `wall_ms` |
| POST | `/api/feedback` | 追加写 `部署/日志/用户反馈.jsonl` |
| POST | `/api/reset` | 附加路由（服务于「清空对话」）→ `engine.reset(session_id)` |

`POST /api/chat` 响应示例（字段与 `设计/接口契约_冻结.md` §2 一致）：

```jsonc
{"ok": true, "session_id": "web-xxxxxxxx", "turn_index": 2,
 "question": "它的法定代表人是谁？",
 "rewritten_query": "武汉兴图新科电子股份有限公司的法定代表人是谁？",
 "strategy": "pronoun", "focus_entity": "武汉兴图新科电子股份有限公司",
 "answer": "……\n引用：[招股说明书1.pdf: 第68页]",
 "refused": false, "cited_pages": [68],
 "latency_ms": {"rewrite": 1.2, "retrieve": 180.0, "generate": 600.0, "total": 820.0},
 "rewrite": {"resolved_refs": [{"surface": "它", "replacement": "武汉兴图新科电子股份有限公司"}]},
 "history": [{"question": "…", "answer": "…"}],
 "state": {"focus_entity": "…", "last_slot": "法定代表人", "last_query": "…"}}
```

### 5.1 用户反馈留痕 `部署/日志/用户反馈.jsonl`

每行一个 JSON，首次投票时自动创建（目录不存在会自动建）：

```json
{"session_id":"web-xxxxxxxx","turn_index":2,"question":"它的法定代表人是谁？",
 "rewritten_query":"武汉兴图新科电子股份有限公司的法定代表人是谁？","answer":"……",
 "vote":"up","ts":1791353626.417,"ts_iso":"2026-10-07T14:13:46","source":"web_app"}
```

* `vote`：`up`（👍 有帮助）/ `down`（👎 没帮助）；也接受 `👍/👎`、`有帮助/没帮助`，非法值返回 400；
* `ts`：epoch 秒（float），`ts_iso`：本地时间字符串；
* `source`：`web_app` / `gradio_app`（附加字段，便于区分入口）；两个界面写的是**同一个文件**。

---

## 6. 退出码

| 脚本 | 码 | 含义 |
| --- | --- | --- |
| `run_web.ps1` | 0 / 1 / 2 / 3 / 127 | 正常 / 端口被占用（未加 `-Force`）/ 参数错误 / 子进程依赖缺失 / 解释器缺失 |
| `gradio_app.py` | 0 / 2 / **3** | 正常退出 / 端口占用或参数错误 / **gradio 不可用**（打印中文指引） |
| `web_app.py` | 0 / 2 | 正常退出 / 端口被占用或参数错误 |

---

## 7. 本次实测记录（实跑，未运行项一律标注）

解释器 `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`，工作目录 `E:\gao6gongdan\工单5`。

| # | 命令 / 动作 | 结果 |
| --- | --- | --- |
| 1 | `pwsh -NoProfile -File run_py.ps1 -c "import ast,sys; [ast.parse(open(p,encoding='utf-8').read(),p) for p in ['部署/界面/gradio_app.py','部署/界面/web_app.py']]; print('AST OK')"` | **AST OK，退出码 0** |
| 2 | 导入自检（`importlib` 真正 exec 两个模块） | **两个文件均 IMPORT OK**（顶层零第三方依赖，`WORK_ORDER`/`REPO_ROOT` 正确解析到 `E:\gao6gongdan\工单5`） |
| 3 | `Parser::ParseFile('部署/脚本/run_web.ps1')` | **PS SYNTAX OK** |
| 4 | `pwsh -NoProfile -File run_py.ps1 部署/界面/gradio_app.py --port 7860` | **退出码 3** + 中文安装指引（缺 gradio 的降级路径按设计生效） |
| 5 | `pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode web -Port 8123`（后台实跑） | 启动器横幅/端口检查/前台转发均正常；服务就绪 |
| 6 | 实跑 `GET /` | **HTTP 200**，返回中文单页（含 `清空对话`、`/api/chat`） |
| 7 | 实跑 `GET /health` | **HTTP 503**，`status=degraded`，`error.code=E_ENGINE_UNAVAILABLE` + 中文原因（如实反映引擎未就绪） |
| 8 | 实跑 `POST /api/chat`（空 `question`） | **HTTP 400** + 中文提示（不是 500） |
| 9 | 实跑 `POST /api/chat`（正常问句） | **HTTP 503** + 引擎未就绪中文原因 |
| 10 | 实跑 `POST /api/feedback`（`vote` 非法 / 合法） | 非法 **400**；合法 **200**，写入 `部署/日志/用户反馈.jsonl`（验证后已清理该探针行） |
| 11 | 实跑 `POST /api/reset`（空 `session_id`） | **HTTP 400** + 中文提示 |
| 12 | 浏览器实跑 `http://127.0.0.1:8123/` | 页面正常渲染、JS 初始化成功；发送后出现用户气泡 + 友好中文错误气泡；「清空对话」换新 `session_id` 并如实提示服务端 reset 失败原因 |

**未运行项（如实声明）**：

* ❌ **真实多轮问答未实跑**：本次实跑时研发侧 `研发/app/multiturn_engine.py`、
  `研发/app/core/dialogue_state.py`、`研发/app/core/query_rewriter.py` **均不存在**，
  因此改写/检索/生成链路、反馈与 `reset` 的**成功路径**、耗时数字均未产生。
  引擎就绪后请重跑第 5~12 项（或直接用界面提问）取得真实数字。
* ❌ **Gradio 界面本体未运行**：本机无 gradio（且禁止裸装），只实跑了「缺 gradio」的降级路径（第 4 项）。
  `build_app()` / `Chatbot` 兼容分支（`type="messages"` → tuples）**未经真机验证**。
* ❌ **FastAPI `-Mode api` 未实跑**：该脚本由 captain 负责多轮化，本目录不改动它。

---

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| `[失败] 无法监听 127.0.0.1:8123` | 端口被占：`netstat -ano \| findstr :8123`；换端口 `-Port 8124`，或确认后 `-Force` |
| `gradio_app.py` 退出码 3 | 正常降级：改用 `-Mode web`（8123），或按 §3 装到 `研发/第三方库` |
| `/health` 返回 503 且提示 `E_ENGINE_UNAVAILABLE` | 研发侧多轮引擎模块未就绪（非界面故障）；就绪后重启本服务即可 |
| 控制台中文乱码 | 一律用 `run_py.ps1` / `run_web.ps1` 启动（已设 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`） |
| 反馈没落盘 | 检查 `部署/日志/` 是否可写；接口会返回 `path` 与累计行数 `lines` |
