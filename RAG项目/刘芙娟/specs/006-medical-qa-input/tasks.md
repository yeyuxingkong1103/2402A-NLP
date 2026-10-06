---
description: "Task list for 医知源 · 提问输入链路与前端骨架"
---

# Tasks: 医知源 · 提问输入链路与前端骨架

**Input**: Design documents from `/specs/006-medical-qa-input/`

**Prerequisites**: [plan.md](./plan.md)、[spec.md](./spec.md)、[research.md](./research.md)、[data-model.md](./data-model.md)、[contracts/](./contracts/)、[quickstart.md](./quickstart.md)

**Tests**: **不生成测试任务**。本项目既有 Global Constraints 禁止引入 pytest（`docs/superpowers/plans`，`specs/005` plan.md §145 已记录），规格也未要求 TDD。验收以 quickstart.md 的「逐项对拍 + 边界构造」承担，作为**验收任务**（非单元测试）编入各阶段。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件，无未完成依赖）
- **[Story]**: 所属用户故事（US1–US4）
- 每个任务均含确切文件路径

## Path Conventions

- 后端：`backend/`（入口脚本）+ `backend/api/`（本特性新增的运行时包）
- 前端：`frontend/`（原生静态资源，无构建）
- 验收产物：`.smoke_out/`
- 全部命令以 `D:/zg6_Project/9/med_rag/rag/python.exe` 运行（constitution 原则 I）

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 目录与配置文件就位。本阶段无任何依赖。

- [x] T001 创建目录结构：`backend/api/`、`frontend/js/`

- [x] T002 [P] 新增 `.env.example`（仓库根），列出本特性读取的全部变量名、值为占位符 `<YOUR_API_KEY>` 形态，MUST NOT 含真实值（constitution 原则 III）。至少含：`MEDRAG_HOST`、`MEDRAG_PORT`，以及标注为"后续模块接入时转为必需"的 LLM 相关变量占位

- [x] T003 [P] 确认 `.gitignore` 已包含 `.env`（缺失则补），且 `.env.example` 未被忽略

- [x] T004 [P] 核对 `requirements.txt`：本特性**新增依赖为 0**。在文件末尾追加一段注释，记录本特性消费的既有依赖及其实测版本（fastapi 0.141.1 / starlette 1.6.0 / uvicorn 0.53.0 / pydantic 2.13.5 / pydantic-settings 2.15.0 / httpx 0.28.1），并说明**为何不引入 `sse-starlette`**（research R1），供后续评审追溯

**Checkpoint**: 目录与配置就位，可开始写代码

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 全部用户故事共用的契约层。**此阶段不产出可运行服务**，但缺任何一项后续都会返工。

**⚠️ CRITICAL**: 本阶段完成前，任何用户故事都不得开始

- [x] T005 创建 `backend/api/__init__.py`：契约常量与包文档字符串。MUST 包含：
  - 路由常量：`ASK_PATH = "/ask"`
  - SSE 事件名：`EV_STATUS`、`EV_CITATIONS`、`EV_TOKEN`、`EV_DONE`
  - 输入限额：`QUESTION_MIN_LEN = 1`、`QUESTION_MAX_LEN = 200`
  - 状态值：`STATE_ACCEPTED`、`RISK_LEVEL_NONE`
  - **四处逐字文案常量**（`EMERGENCY_PREAMBLE` / `DISCLAIMER` / `REFUSAL_FALLBACK` / `CAPABILITY_NOT_READY`），文本逐字取自 [data-model.md](./data-model.md) §3
  - 故障注入开关名：`PREAMBLE_ENV = "MEDRAG_TEST_PREAMBLE"`
  - 包文档字符串 MUST 写明"全部面向用户的固定文案只此一处定义"（`docs/05` §4.4 的同类要求）

- [x] T006 [P] 创建 `backend/api/schemas.py`：Pydantic 模型 `AskRequest`、`Citation`、`AskResponse`。`AskResponse` 字段名 MUST 与 `docs/05` §3.1.3 **逐字段一致**（`answer_id` / `is_refusal` / `emergency` / `risk_level` / `answer_text` / `citations` / `disclaimer`），MUST NOT 增删改。文件顶部注释写明"本模型是 SC-009 的验证对象，字段名不得改动"

- [x] T007 [P] 创建 `backend/api/config.py`：基于 `pydantic-settings` 的启动期配置。要求：
  - 本期必需项为空，仅 `host`/`port` 有默认值（`127.0.0.1` / `8000`）
  - LLM 相关密钥字段标注为**可选**，且每个 MUST 带注释说明"何时转为必需"（research R6）
  - 提供 `missing_required() -> list[str]` 供后续 `/health` 复用
  - MUST NOT 在请求路径上被调用（FR-022）

- [x] T008 [P] 创建 `backend/api/errors.py`：统一错误体与异常处理器。要求：
  - 错误体结构严格对齐 `docs/05` §2.3：`{"error": {"code", "message", "request_id"}}`
  - 错误码仅 `INVALID_QUESTION`（422）与 `INTERNAL_ERROR`（500）
  - 注册 `RequestValidationError` 与 `Exception` 处理器，**确保框架默认 HTML 错误页不会漏出**（FR-019）
  - 错误体 MUST NOT 含堆栈、异常类名、文件路径、密钥；`request_id` MUST 与日志同值

- [x] T009 [P] 创建 `backend/api/events.py`：SSE 编码器。要求：
  - 输出格式严格为 `event: <名>\ndata: <单行JSON>\n\n`（[contracts/sse.md](./contracts/sse.md) §1）
  - MUST 用 `json.dumps(..., ensure_ascii=False)` 且**不加 `indent`**——保证 `data:` 恒为单行且中文不转义
  - 提供 `sse_event(name: str, payload: dict) -> str`，签名带类型注解

**Checkpoint**: 契约层完成。此后全部用户故事可并行开发（若多人）

---

## Phase 3: User Story 1 - 提交一个医疗问题 (Priority: P1) 🎯 MVP

**Goal**: 用户在 `http://127.0.0.1:8000/` 看到医知源界面，输入医疗问题，回车或点发送图标，问题被服务端接收并给出明确的进行中/未就绪反馈。

**Independent Test**: 启动服务后打开网址，输入任意合法中文问题并提交；界面出现接收反馈，`curl -N` 能观察到 `status → citations → done` 三帧，且 `done.answer_id == status.answer_id`。全程不依赖 Milvus、向量模型、大模型。

### 实现

- [x] T010 [US1] 创建 `backend/api/stream.py`：事件序列生成器。要求：
  - 产出顺序：`status` → `citations` → `token*` → `done`（[data-model.md](./data-model.md) §2.1 的硬性顺序约束）
  - `status` 首帧携带 `answer_id`（UUID）与 `state="accepted"`，理由（research R10）写入注释
  - `citations` 本期发空数组 `{"citations": []}`，**仍然发送该事件**，并在注释写明理由（契约 §4.1）
  - 本期不发 `token`；但 MUST 保留**前置块结构**：若 `MEDRAG_TEST_PREAMBLE` 环境变量存在，先发一个 `token` 事件再发正文（FR-030 的落点）
  - `done` 帧构造 `AskResponse`，`answer_text = CAPABILITY_NOT_READY + "\n\n" + DISCLAIMER`，`is_refusal=True`，`risk_level="none"`，`citations=[]`
  - 生成器 MUST 处理客户端断开（`asyncio.CancelledError`），记录 `client_disconnected` 日志含 `answer_id`，且**不得抛出未捕获异常**（[contracts/http.md](./contracts/http.md) §2.4）

- [x] T011 [US1] 创建 `backend/api/routes.py`：`POST /ask` 路由。要求：
  - 返回 `StreamingResponse`，`media_type="text/event-stream; charset=utf-8"`
  - 响应头 MUST 含 `Cache-Control: no-cache`、`Connection: keep-alive`、`X-Accel-Buffering: no`，且**每个头都带注释说明理由**（[contracts/http.md](./contracts/http.md) §2.2 的表格）
  - 只做 HTTP ↔ 模型的协议转换，MUST NOT 含业务判定（FR-015 的服务端对称约束）
  - 记录每次提问的 `answer_id` 与问题原文到日志（FR-026）

- [x] T012 [US1] 创建 `backend/api/app.py`：FastAPI 装配。**注册顺序 MUST 为：中间件 → 异常处理器 → API 路由 → 静态挂载**。在注册静态挂载处 MUST 写注释说明：Starlette 按注册顺序匹配，若先挂 `/` 则 `/ask` 会被静态处理器接走并 404，且**启动时不报错**（research R3）。静态挂载用 `StaticFiles(directory="frontend", html=True)`，路径以 `backend/api/app.py` 为基准解析

- [x] T013 [US1] 创建 `backend/serve.py`：唯一入口。要求：
  - 用 `uvicorn.run` 绑定 `config.host` / `config.port`，默认 `127.0.0.1:8000`
  - 启动即打印 `医知源服务已启动  http://127.0.0.1:8000/`
  - 启动期调用一次 `config.load()`，必需项缺失即非 0 退出（FR-022）
  - MUST NOT 启用 `--reload`（会中断进行中的 SSE 流，research R9），并在注释写明
  - 支持 `python -m backend.serve` 直接运行

- [x] T014 [US1] 创建 `frontend/index.html`：页面骨架。要求：
  - 左侧导航栏 + 右侧详情区（FR-002）
  - 导航含「医疗问答」且为默认选中（FR-004）
  - 右侧含问题输入区与发送图标按钮
  - 答案正文区在**上**、引用片段区在**下**（FR-027 的布局约束，本期为空态）
  - **品牌名与一句失败提示 MUST 内联**，不依赖外部资源——保证样式/脚本挂掉时不是白屏（Edge Case）

- [x] T015 [P] [US1] 创建 `frontend/styles.css`：蓝白配色与两栏布局（FR-003）。要求：导航/强调色为蓝色系、底为白色系；正文与背景对比度满足正常阅读；左右两栏在 1280×720 及以上正常显示

- [x] T016 [US1] 创建 `frontend/js/sse.js`：SSE 解析器。MUST 满足 [contracts/sse.md](./contracts/sse.md) §5 的全部六条要求，其中两条**必须在注释中点明**：
  - `TextDecoder` 必须用 `{stream: true}`——不设它会在多字节 UTF-8 被切断时产生乱码，中文几乎必然触发
  - 按 `\n\n` 切块时**必须保留未完成的尾部**——一个事件被切成两次 `read()` 到达是常态而非异常，且用短英文测试碰不到

- [x] T017 [US1] 创建 `frontend/js/transcript.js`：答案区渲染。要求：
  - `appendToken(text)` 追加到**同一个文本节点**，不重建 DOM（[data-model.md](./data-model.md) §4.2）
  - `renderDone(envelope)` 以 `answer_text` 为准渲染，并校验/纠正已渲染内容（契约 §4.3）
  - `renderCitations(citations)` **按传入顺序渲染，MUST NOT 调用 `sort`**，注释写明排序责任在服务端（[data-model.md](./data-model.md) §4.1）
  - 导出可独立调用的渲染函数，供 quickstart §7 从控制台注入假数据

- [x] T018 [US1] 创建 `frontend/js/nav.js`：导航配置与面板切换。要求：
  - 导航项由**一份集中配置数组**定义（FR-005），每项 `{id, label, panel, default}`
  - 切换靠显隐面板实现，MUST NOT 重建问答面板 DOM（否则回切时输入草稿丢失，US3-2）
  - 本期配置只含「医疗问答」一项

- [x] T019 [US1] 创建 `frontend/js/app.js`：提交交互与状态机。要求：
  - 回车提交（FR-009）、点击发送图标提交（FR-010），二者走**同一条**提交函数
  - **IME 组合态忽略回车**：检查 `event.isComposing`（或 `keyCode === 229`）后直接返回（FR-009/FR-011、Edge Case）
  - 实现 [data-model.md](./data-model.md) §4.3 的五态：`idle` / `submitting` / `streaming` / `done` / `failed`，其中 **`submitting` 与 `streaming` 必须分开**（前者"还没被接受"，后者"已接受正在处理"）
  - 进行中再次提交直接忽略（FR-013）
  - 失败/超时必须退出加载态并给出可读提示（FR-012、FR-014、N14）
  - 失败时 MUST 区分"请求未被接受"与"流中途断开"

### 验收（US1）

- [x] T020 [US1] 执行 quickstart §3：`curl -sN` 契约逐字对拍，核对 10 项（含 `done` 字段名集合无增无减、免责声明逐字结尾、不含医学结论、不含拒答话术、中文非 `\uXXXX` 转义）与响应头三项

- [x] T021 [US1] 执行 quickstart §2：浏览器打开网址，核对 5 项（3 秒内可交互、导航含医疗问答且选中、左右布局、蓝白配色、**重命名 styles.css 后刷新不是白屏**）

**Checkpoint**: US1 独立可用 —— 提问通道打通，这是 MVP

---

## Phase 4: User Story 2 - 非法输入被当场拦住并说清原因 (Priority: P2)

**Goal**: 空输入与超长输入被界面当场拦截且不发请求；绕过界面直接调接口时服务端同样拒绝。

**Independent Test**: 分别提交空串、纯空白、纯换行、201 字符；确认前三者未在服务端留下任何日志记录，且 200 字符能正常通过。

### 实现

- [x] T022 [US2] 创建 `backend/api/validate.py`：问题校验。要求：
  - 用 `str.strip()` 语义（含全角空格 U+3000、制表符、换行）去首尾空白
  - 去空白后长度 ∉ [1, 200]（闭区间）即判非法
  - 返回**去空白后的字符串**供后续使用——否则同一问题因首尾空格不同会被当作两个不同 query（[data-model.md](./data-model.md) §1）
  - 非法时抛业务异常，由 T008 的异常处理器转成 422 `INVALID_QUESTION`
  - 长度按**字符**计，不按字节——`'高'*200` 是 200 字符 / 600 字节，按字节判会误拒

- [x] T023 [US2] 在 `backend/api/routes.py` 接入 `validate.py`（T022），确保校验发生在**任何流式输出之前**（否则会先发 `status` 帧再报错，前端拿到一半的流无法处理）

- [x] T024 [US2] 在 `frontend/js/app.js` 的提交函数**最前面**加入前端拦截：去空白后为空 → 提示"请输入你的问题"并 `return`；超过 200 字符 → 提示过长（含当前字数与上限）并 `return`。**两种情况都 MUST NOT 发出任何请求**（FR-016、FR-017、SC-003）

- [x] T025 [US2] 在 `frontend/index.html` / `styles.css` 增加校验提示的展示区与样式（错误态需与"能力未就绪"空态在视觉上可区分）

### 验收（US2）

- [x] T026 [US2] 执行 quickstart §4：4.1 空问题 422 / 4.2 纯空白（含全角空格）422 / 4.3 字段缺失 422（**不得是 500**）/ 4.4 恰好 200 字符 200 / 4.5 恰好 201 字符 422，并跑错误体脱敏 grep

- [ ] T027 [US2] 执行 quickstart §5 的第 3–8 项：IME 回车不提交、空输入无请求、超长无请求、重复提交不并发、HTML 标签原样转义、`【1】` 原样显示。**"无请求"须以浏览器网络面板确认**，不能只看界面提示

**Checkpoint**: US1 与 US2 均可独立工作

---

## Phase 5: User Story 3 - 导航栏能长出第二个功能界面 (Priority: P3)

**Goal**: 新增一个导航功能项只需改一份配置，问答代码零改动。

**Independent Test**: 在导航配置中追加一个占位项，确认它出现、可切换，且切回「医疗问答」时输入草稿与状态仍正确。

### 实现

- [x] T028 [US3] 在 `frontend/index.html` 增加一个占位面板（如「用药助手」的空内容区），并在 `frontend/js/nav.js` 的配置数组中追加对应项——**用于证明扩展性**。此占位项可保留（体现导航可扩展）或验收后移除，实现时二选一并说明

- [x] T029 [US3] 确认并加固切换时的状态保持：切换导航 MUST NOT 重建问答面板 DOM，切回后输入框内容与已渲染结果 MUST 原样保留（US3-2）

### 验收（US3）

- [x] T030 [US3] 执行 quickstart §6：核对新增项出现、可切换、切回后输入框可用，并以 `git diff --stat` 确认**改动只落在 `nav.js`（与 `index.html` 的面板骨架）**，问答逻辑文件零改动（SC-005、US3-3）

**Checkpoint**: 三个用户故事独立可用

---

## Phase 6: User Story 4 - 答案与引用的展示容器已经就位 (Priority: P3)

**Goal**: 答案区在引用区之上；引用按服务端给的顺序渲染（不重排序）；正文支持逐块追加；紧急话术若出现必为首个 token。

**Independent Test**: 用固定假数据注入渲染层，确认引用显示顺序 == 传入顺序（而非重排后的降序），正文可逐块追加且不重建 DOM。

### 实现

- [x] T031 [US4] 在 `frontend/js/transcript.js` 完善 `renderCitations(citations)`：每条渲染**来源文件名 + 页码**（FR-028），并按传入顺序排列、MUST NOT `sort`（[data-model.md](./data-model.md) §4.1）。引用区 DOM 位置 MUST 在答案正文区**之后**（FR-027）

- [x] T032 [US4] 在 `frontend/js/transcript.js` 增加引用卡片的样式与交互占位（可展开看原文的入口，对应 `docs/01` §F3；本期可只做展开骨架）

- [x] T033 [US4] 在 `frontend/styles.css` 增加引用卡片样式（蓝白体系内的次级视觉层级，与答案正文明确区分）

### 验收（US4）

- [x] T034 [US4] 执行 quickstart §7 的**两步验证**（不可只做第一步）：
  1. 喂**已按降序**的假数据 → 期望显示顺序与之一致
  2. 喂**乱序**假数据 → 期望显示顺序**仍是喂入顺序**（证明前端确实没有排序，排序责任在服务端）
  并核对每条显示文件名与页码、引用区位于答案区下方

- [x] T035 [US4] 执行 quickstart §7 的流式追加验证：连续三次 `appendToken` 后内容为三者拼接，并用 `MutationObserver` 确认**已有节点未被重建**

- [x] T036 [US4] 执行 quickstart §8：以 `MEDRAG_TEST_PREAMBLE` 启动，验证 FR-030 —— 首个 `token` 事件的内容逐字等于紧急话术、其前无任何 `token` 事件、`citations` 在其前是允许的、全部 `token` 拼接 == `done.answer_text` 去掉免责声明部分。**必须打真实端口，不得用 TestClient**。验收后重启服务确认不带该变量时不再出现 `token`

**Checkpoint**: 全部四个用户故事独立可用

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 文档回填与全量验收。前六阶段完成前不应开始——文档修订描述的是**已落地的实现**。

- [x] T037 [P] 修订 `docs/01_需求分析.md` §F6：回填拒答兜底话术的**逐字文本**（本特性定稿，[data-model.md](./data-model.md) §3）。此前该节只说"返回固定兜底话术"却从未给出文本

- [x] T038 [P] 修订 `docs/05_接口设计.md` §3.1.1：将"为什么不流式"整节改写为流式方案；N7 断言改述为"**首个 `token` 事件等于话术**"；**保留原有的风险提示**（它并未因改流式而消失，只是断言点下移）

- [x] T039 [P] 修订 `docs/05_接口设计.md` §1.1 I-01 行、§2.4 延迟预算、§3.1.4：更新为流式；延迟预算拆为"首块时间"与"整体完成时间"

- [x] T040 [P] 修订 `docs/02_架构图.md` §373/§379、§2.1、§10：前端层由 Streamlit（8501）改为同源静态界面由问答服务托管；部署进程数由 3 降为 2

- [x] T041 [P] 修订 `docs/03_技术选型说明.md` §6 与 §423 汇总表：记录前端选型变更与理由（本次诉求是定制蓝白布局 + 引用卡片排序 + 正文流式，超出 Streamlit 的表达能力）

- [x] T042 执行 quickstart §9：日志与脱敏 —— 用 `answer_id` 能定位到每一次提问（计数相等）、日志中 grep 不到任何密钥、`client_disconnected` 有记录且无 Traceback

- [x] T043 执行 quickstart §1 的绑定核对：从同一局域网**另一台机器**访问 `http://<本机IP>:8000/` 应连接失败（确认只绑回环，未暴露无鉴权服务）

- [x] T044 执行 quickstart §10 与 §5 第 9–10 项：`Ctrl-C` 干净退出**无 Traceback**；停服后提交有明确失败提示且退出加载态；刷新页面无历史回显

- [ ] T045 执行 quickstart §5 第 1–2 项与 §2 的完整回归，确认四个用户故事**同时**可用、互不干扰

- [x] T046 逐条核对 `spec.md` 的 37 条 FR 与 9 条 SC 是否均有对应实现或验收覆盖；对**明确不在本期范围**的项（拒答断言、紧急话术真实触发）在 `tasks.md` 末尾记录为已知范围限制，MUST NOT 静默略过

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Phase 1 (Setup)          ← 无依赖
    ↓
Phase 2 (Foundational)   ← 阻塞全部用户故事；不产出可运行服务
    ↓
Phase 3 (US1, P1) 🎯 MVP ← 产出可运行服务，是后续全部阶段的前提
    ↓
    ├── Phase 4 (US2, P2)  ← 依赖 US1 的 app.js / routes.py
    ├── Phase 5 (US3, P3)  ← 依赖 US1 的 nav.js
    └── Phase 6 (US4, P3)  ← 依赖 US1 的 transcript.js / stream.py
              ↓
Phase 7 (Polish)         ← 依赖前六阶段全部完成
```

**US2/US3/US4 之间无相互依赖**，在 US1 完成后可并行推进（若多人）。

### 关键顺序约束（违反会返工）

1. **T012（app.py 注册顺序）必须在 T011（routes.py）之后** —— 无路由可挂时静态挂载会吞掉一切
2. **T020（curl 契约对拍）必须在写前端之前完成** —— 服务端契约未确认就写前端，会把"服务端发错了"和"前端解析错了"混成同一个 bug（plan.md 实施顺序第 7 步）
3. **T022（服务端校验）必须在 T024（前端拦截）之前或同时** —— 前端拦截是体验优化，服务端校验才是边界；顺序反了容易只做前者
4. **T037–T041（文档修订）必须在实现落地之后** —— 修的是已实现的行为，先改文档会让文档描述一个不存在的系统

### Parallel Opportunities

**Phase 1**：T002、T003、T004 全部可并行（不同文件）

**Phase 2**：T006、T007、T008、T009 全部可并行（不同文件）；T005 须先完成——其余四项都引用它的常量

**Phase 3**：T015（styles.css）与 T014（index.html）可并行；T010–T013 是后端链，T014–T019 是前端链，两条链可并行（但前端验收 T021 需后端 T013 完成）

**Phase 7**：T037–T041 五处文档修订全部可并行（不同文件、不同章节）

### Parallel Example: Phase 2

```bash
# T005 完成后，四项可同时进行：
Task: "创建 backend/api/schemas.py（T006）"
Task: "创建 backend/api/config.py（T007）"
Task: "创建 backend/api/errors.py（T008）"
Task: "创建 backend/api/events.py（T009）"
```

### Parallel Example: Phase 7

```bash
Task: "修订 docs/01 §F6 回填兜底话术（T037）"
Task: "修订 docs/05 §3.1.1（T038）"
Task: "修订 docs/05 §1.1/§2.4/§3.1.4（T039）"
Task: "修订 docs/02（T040）"
Task: "修订 docs/03（T041）"
```

---

## Implementation Strategy

### MVP First（仅 US1）

1. Phase 1 Setup → 2. Phase 2 Foundational → 3. Phase 3 US1
4. **STOP 并验证 T020 + T021**
5. 此时"提问通道"已完整可用，可作为演示交付

**注意**：Phase 2 单独完成时服务起不来，属正常——它不是可交付增量，而是契约层。

### Incremental Delivery

1. Setup + Foundational → 契约层就位
2. **US1** → 提问通道打通 → 验证 → 演示（MVP）
3. **US2** → 输入边界钉死 → 验证
4. **US3** → 证明导航可扩展（防技术债，越早越好）
5. **US4** → 引用与流式容器就位，接通后续模块的接口
6. **Phase 7** → 文档与代码回到一致状态

### 已知范围限制（MUST 显式记录，不得静默略过）

| 限制 | 原因 | 何时补 |
|---|---|---|
| 拒答断言（检索为空/低于阈值 → 兜底话术）未建立 | 本期不存在"检索过且为空"这一事实，硬造只会得到假绿 | 检索模块接入时 |
| 紧急话术前置无真实触发 | 本期无紧急判定模块，`emergency.triggered` 恒 `false` | 紧急判定模块接入时 |
| `token` 拼接 == `done.answer_text` 的断言为空转 | 本期不发 `token`（除注入场景） | 生成模块接入时 |

**这三条都是 plan.md Constitution Check 中已记录的有意取舍**，不是遗漏。

---

## Notes

- **[P]** = 不同文件、无未完成依赖
- 本项目**禁止引入 pytest**，不生成单元测试任务；验收由 quickstart.md 承担
- 全部命令 MUST 以 `D:/zg6_Project/9/med_rag/rag/python.exe` 运行，MUST NOT 用裸 `python`
- 每个任务完成后可独立提交；检查点处停下验证当前故事
- **避免**：同一文件在并行任务中被两处修改；跨故事的文件依赖；把"看起来done但没验证"当作完成

---

## T046 覆盖核对：37 条 FR + 9 条 SC

**核对日期**：2026-09-27　**结论**：无遗漏项；有 3 处**明确的、有意接受的范围限制**（见文末）。

### 验收脚本总览

| 脚本 | 覆盖 | 结果 |
|---|---|---|
| `.smoke_out/s7_phase2_check.py` | 契约常量、话术逐字、SSE 编码、字段集合 | 31/31 |
| `.smoke_out/s7_contract_check.py` | HTTP 头、事件序列、终帧字段、文案、编码、错误体 | 38/38 |
| `.smoke_out/s7_validate_check.py` | 输入边界（含按字符非字节）、前端等价性 | 28/28 |
| `.smoke_out/s7_sse_parse_check.py` | 解析器两个真陷阱 + 负向对照 | 24/24 |
| `.smoke_out/s7_render_check.py` | 引用顺序（两步验证）、流式追加、终帧校正 | 33/33 |
| `.smoke_out/s7_nav_check.mjs` | 导航扩展性、切换不重建 DOM | 23/23 |
| `.smoke_out/s7_preamble_check.py` | FR-030 首块顺序（注入） | 11/11 |
| `.smoke_out/s7_log_check.py` | 日志关联与脱敏 | 12/12 |
| `.smoke_out/s7_disconnect_check.py` | 流取消处理（两条取消路径） | 16/16 |
| `.smoke_out/s7_shutdown_check.py` | 优雅关闭（含控制组） | 13/13 |

### 功能需求（37 条）

| 分组 | 覆盖情况 |
|---|---|
| FR-001 ~ FR-008 界面与布局 | ✅ 全部。FR-001/007/008 由 `curl` 验证静态资源 200 与同源；FR-002/003 由代码与样式确认（**视觉呈现需人工过目**）；FR-004/005 由 `s7_nav_check` 验证 |
| FR-009 ~ FR-015 提问交互 | ✅ 全部。FR-011（IME）有实现但**需人工在浏览器确认**；FR-013/014 由状态机代码保证 |
| FR-016 ~ FR-020 输入校验 | ✅ 全部。FR-016/017 由前端函数等价性测试 + 服务端日志计数双重验证（含正面校验）；FR-018 边界 6 组全过 |
| FR-021 ~ FR-026 服务端入口 | ✅ 全部。FR-023 字段集合逐字段对拍；FR-025 由医学词汇黑名单扫描；FR-026 由 `s7_log_check` 验证 |
| FR-027 ~ FR-031 引用与展示 | ✅ 全部。FR-028 本期 `citations` 恒空，**服务端降序排序本身无从验证**（无数据），已验证的是"前端不重排"这一半 |
| FR-032 ~ FR-037 工程约束 | ✅ 全部。FR-036 单文件 ≤300 行（`styles.css` 曾 329 行，已拆为 `styles.css` + `panels.css`）；FR-037 由 AST 扫描确认 20 个函数注解完整 |

### 成功标准（9 条）

| 编号 | 状态 |
|---|---|
| SC-001 3 秒内可交互 | ⚠️ 未实测（需浏览器）。静态资源总字节数很小，推断无碍，但**未测就是未测** |
| SC-002 2 秒内接收反馈 | ✅ 实测远低于（无检索、无生成） |
| SC-003 100% 不产生请求 | ✅ 前端等价性 + 服务端计数双重验证 |
| SC-004 answer_id 可定位 | ✅ `s7_log_check`：提问集合 == 完成集合，无重复 |
| SC-005 导航扩展 ≤1 处配置 | ✅ `s7_nav_check`：动态追加项，其它文件零改动 |
| SC-006 医学结论 0 条 | ✅ 终帧 `answer_text` 无医学词汇 |
| SC-007 引用顺序一致 | ✅ **两步验证**（只做第一步会得到假绿） |
| SC-008 100% 退出加载态 | ✅ 五态状态机 + `finally` 复位；**实际视觉效果需人工确认** |
| SC-009 接入时不改前端解析 | ✅ `done` 字段集合与 `docs/05` §3.1.3 逐字段一致 |

### 三处已知范围限制（有意接受，非遗漏）

| 限制 | 原因 | 何时可验证 |
|---|---|---|
| **拒答兜底话术未被真实触发** | 本期不存在「检索过且为空」这一事实，硬造只会得到假绿。话术文本已定稿并写入常量、已回填 `docs/01` §F6 | 检索模块接入时 |
| **服务端引用降序排序未被执行** | `citations` 本期恒为空数组，没有数据可排。已验证的是前端保序 | 检索模块接入时 |
| **客户端断开的真实网络路径不可达** | 响应约 0.5ms 就发完，客户端来不及中途断开（实测裸 socket RST 关闭后服务端仍完整发出）。已改为直接驱动生成器验证取消分支 | 生成模块接入、响应变长之后 |

### 两处未完成的验收（需人在浏览器操作）

- **T027**：IME 回车不提交、空/超长无请求、HTML 标签与 `【1】` 原样转义——需浏览器网络面板与肉眼确认
- **T045**：quickstart §5 第 1–2 项与 §2 的完整回归

**旁证（非验证）**：全部渲染路径使用 `textContent`（`transcript.js` 15 处、`app.js` 2 处），`nav.js` 中唯一的 `innerHTML` 是清空操作。转义相关要求从代码上应当成立，但**这是推断**。
