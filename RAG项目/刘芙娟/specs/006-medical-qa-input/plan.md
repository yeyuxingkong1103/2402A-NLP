# Implementation Plan: 医知源 · 提问输入链路与前端骨架

**Branch**: `006-medical-qa-input` | **Date**: 2026-09-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/006-medical-qa-input/spec.md`

**Note**: 本文件由 `/speckit-plan` 生成；其执行流程定义见 `.claude/skills/speckit-plan`。

---

## Summary

把"用户在界面里敲下的一句话"变成一条**可追溯、可流式、契约已冻结**的通道。后端是一个 FastAPI 应用（`backend/serve.py` 唯一入口，实现拆到 `backend/api/` 包），前端是一组无构建步骤的原生静态资源（`frontend/`），由同一个服务在 `127.0.0.1:8000` 同源提供。问答以 **SSE**（`text/event-stream` + 命名事件）应答：`status` → `citations` → `token*` → `done`，其中 `done` 帧**逐字段复用** `docs/05` §3.1.3 的响应契约。

向量化、语义检索、提示词、大模型生成**本期一律不实现**。终帧的答案字段以显式的"能力未就绪"状态表达——**不是**伪装的拒答，理由见 [research.md](./research.md) R5（`docs/05` §3.1.4 规则 3 明令禁止把"能力没建成"伪装成"知识库没内容"）。

技术方案的全部取舍与依据见 [research.md](./research.md)；线格式契约见 [contracts/sse.md](./contracts/sse.md)。

---

## Technical Context

**Language/Version**: Python 3.12.14 —— 唯一受支持的解释器 `D:/zg6_Project/9/med_rag/rag/python.exe`（constitution 原则 I，NON-NEGOTIABLE）

**Primary Dependencies**: FastAPI 0.141.1、Starlette 1.6.0、Uvicorn 0.53.0、Pydantic 2.13.5、pydantic-settings 2.15.0 —— **全部实测已装，本期新增依赖为 0**。前端为原生 HTML/CSS/JS，无框架、无构建工具链。

**Storage**: 无。系统不做历史记录（`docs/01` §7.2），本特性不引入任何持久化。

**Testing**: **无测试框架**。项目既有约束禁止引入 pytest（`docs/superpowers/plans` Global Constraints，`specs/005` plan.md §145 已记录）。验证方式为「逐项对拍 + 边界构造」，用 `httpx 0.28.1`（已装）打真实端口，产物落 `.smoke_out/`。验证清单见 [quickstart.md](./quickstart.md)。

**Target Platform**: Windows 本机进程，`127.0.0.1:8000`，仅回环可达（`docs/05` §2.2：服务不暴露公网，鉴权由部署层承担）

**Project Type**: Web 应用（同源前后端，单进程提供静态界面 + API）

**Performance Goals**: 页面可交互 ≤ 3s（SC-001）；提问到首个事件响应 ≤ 2s（SC-002）。本期无模型推理、无网络出向调用，实际瓶颈在进程启动与静态资源加载，不在请求处理。

**Constraints**:
- 单文件 ≤ 300 行（FR-036），超出即拆模块，对外入口保持唯一
- 全程类型注解 + snake_case（constitution 代码规范）
- 密钥零硬编码，仅经环境变量（constitution 原则 III）
- 全部面向用户的固定文案（紧急话术 / 免责声明 / 拒答兜底 / 未就绪文案）**只有一处定义**

**Scale/Scope**: 单用户本机演示；后端约 9 个模块、前端 1 页 + 1 样式 + 4 脚本，全部文件均远低于 300 行。

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 本方案的落实 | 状态 |
|---|---|---|
| **I. 环境锁定与依赖治理** | 入口与全部文档以 `rag/python.exe` 绝对路径书写；**本期新增依赖为 0**（SSE 用 Starlette 自带 `StreamingResponse`，不引入 `sse-starlette`，research R1）；前端无 Node 工具链，不引入第二套运行时 | ✅ |
| **II. 无据不答与强制溯源引用** | 本期不产出任何知识性内容：终帧 `answer_text` 只有"能力未就绪"文案 + 免责声明，`citations` 为空数组；FR-025 要求服务端 MUST NOT 输出未经知识库支撑的医学结论，由 quickstart §3 第 8 项断言 `answer_text` 不含任何医学结论 | ✅ |
| **III. 密钥零硬编码** | 配置仅经 `pydantic-settings` 从环境变量加载，启动期一次性读取（FR-022）；本期必需项为空（research R6），可选密钥标注转正时机；日志脱敏由 quickstart §9.2 断言 | ✅ |
| **IV. 紧急症状前置响应** | 本期无紧急判定模块，`emergency.triggered` 恒 `false`，该话术**不发出**。但流式的事件结构已把"前置块"位置固定下来（FR-030），并以 `MEDRAG_TEST_PREAMBLE` 注入建立断言——**这是本期唯一为未实现模块预置的约束**，理由见 research R8 | ✅ |
| **V. 面向群众的医疗安全边界** | 免责声明逐字附加于 `answer_text` 末尾（所有路径）；不输出个体化诊断/处方/剂量；`docs/05` §3.1.8 要求不暴露内部判定细节——本方案不返回 `emergency.source`、`matched_terms`、`refusal_reason_code` | ✅ |

**门禁结论：无违规项，无需 Complexity Tracking。**

**一处需记录在案的张力**（不构成违规，但必须写下来）：constitution 的「Development Workflow」要求医疗答案的测试覆盖**一类拒答用例**（检索为空或低于阈值时返回兜底话术）。本特性**本期不满足该断言**——因为本期不存在"检索过且为空"这一事实，硬造只会得到假绿。兜底话术的逐字文本已在本特性定稿并写入常量（[data-model.md](./data-model.md) §3），待检索模块接入时首次生效。**这是有意接受的范围限制，不是遗漏。**

---

## Project Structure

### Documentation (this feature)

```text
specs/006-medical-qa-input/
├── plan.md              # 本文件
├── spec.md              # 规格（已通过质量校验）
├── research.md          # Phase 0：10 项技术裁决
├── data-model.md        # Phase 1：传输模型 + 逐字话术常量 + 前端渲染模型
├── quickstart.md        # Phase 1：11 节验收清单
├── contracts/
│   ├── sse.md           # SSE 线格式契约（核心）
│   └── http.md          # HTTP 入口与错误契约
├── checklists/
│   └── requirements.md  # 规格质量清单
└── tasks.md             # Phase 2 输出（由 /speckit-tasks 生成）
```

### Source Code (repository root)

```text
backend/
├── serve.py                 # ★ 唯一入口：uvicorn 启动 + 启动自检
├── parse_pdf.py             # 既有 S2（本特性不改）
├── clean_parsed.py          # 既有 S3（本特性不改）
├── chunk_clean.py           # 既有 S4（本特性不改）
├── embed_chunks.py          # 既有 S5（本特性不改）
├── index_milvus.py          # 既有 S6（本特性不改）
├── parse/  clean/  chunk/  embed/  index/     # 既有实现包（本特性不改）
└── api/                     # ★ 本特性新增
    ├── __init__.py          # 契约常量：路由、事件名、限额、状态值、逐字话术
    ├── config.py            # 启动期配置加载（pydantic-settings）
    ├── schemas.py           # Pydantic 模型：AskRequest / AskResponse / Citation
    ├── validate.py          # 问题校验（去空白、1–200）
    ├── errors.py            # 统一错误体 + 异常处理器
    ├── events.py            # SSE 事件构造与单行 JSON 编码
    ├── stream.py            # 事件序列生成器（本期：未就绪路径 + 前置块结构）
    ├── routes.py            # POST /ask 路由
    └── app.py               # FastAPI 装配：中间件 → 异常处理器 → API 路由 → 静态挂载

frontend/                    # ★ 本特性新增（原生静态资源，无构建）
├── index.html               # 页面骨架 + 内联的品牌名与失败提示
├── styles.css               # 蓝白配色、左右两栏、引用卡片
└── js/
    ├── nav.js               # 导航项配置 + 面板切换（唯一需要改动的扩展点）
    ├── sse.js               # fetch + ReadableStream 的 SSE 解析器
    ├── transcript.js        # 答案区追加 + 引用区渲染（不排序）
    └── app.js               # 提交交互：回车/发送/IME/去重/加载态

.smoke_out/                  # 验收产物（既有约定）
requirements.txt             # 无变更（本期不新增依赖）
.env.example                 # 需新增：变量名占位符，无真实值（constitution 原则 III）
```

**Structure Decision**：

采用**同源 Web 应用**结构——`backend/` 持有全部服务端代码（含 API 包），`frontend/` 持有静态资源，由 `backend/api/app.py` 通过 `StaticFiles(html=True)` 挂载到根路径。选择依据：

1. **`frontend/` 位置沿用 `docs/02` §379 的既有目录树**，避免目录语义漂移（research R4）；
2. **API 包沿用 `backend/` 既有的"包 + 根级入口脚本"模式**（`backend/chunk/` + `backend/chunk_clean.py`、`backend/index/` + `backend/index_milvus.py`），使本特性在结构上与 S2–S6 一致；
3. **不与既有管线代码混放**：`backend/api/` 是运行时服务，S2–S6 是离线管线，二者的调用者、生命周期、失败模式完全不同（`docs/05` §5 明确"管线不属于运行时接口，不被后端进程调用"）。

### `backend/api/` 的模块切分理由

切分依据是 **"谁在什么时机可能出错"**，不是"代码属于哪一层"：

| 模块 | 只做 | 为什么单独切出来 |
|---|---|---|
| `__init__.py` | 常量：路由、事件名、限额、三句逐字话术、未就绪文案 | 让"全仓只有一处定义用户可见文案"**在结构上可见**（`docs/05` §4.4 的同类要求） |
| `config.py` | 启动期读环境变量 | 与请求路径隔离，使 FR-022"请求路径上不读环境变量"可被**一眼检查** |
| `schemas.py` | Pydantic 模型 | 终帧字段与 `docs/05` §3.1.3 的对齐关系集中一处，SC-009 的验证对象 |
| `validate.py` | 问题校验 | 边界值（0 / 1 / 200 / 201）的判定只此一处 |
| `errors.py` | 统一错误体 | 防止框架默认 HTML 错误页漏出堆栈（FR-019） |
| `events.py` | 事件编码 | SSE 线格式（单行 JSON、`ensure_ascii=False`）只此一处 |
| `stream.py` | 事件序列 | **本期与后续的关键接缝**——检索/生成模块接入时只改这里 |
| `routes.py` | HTTP 协议转换 | 只管 HTTP ↔ 模型，不做业务判定 |
| `app.py` | 装配 | **路由注册顺序在此**（research R3：API 必须先于静态挂载，否则 `/ask` 被静态处理器吞掉且启动不报错） |

---

## 实施顺序（供 tasks 阶段展开）

依赖关系决定了顺序，不按此序会出现返工：

1. **契约常量与模型**（`__init__.py`、`schemas.py`、`config.py`）—— 先冻结字段名与话术文本，后面所有代码引用它们；
2. **SSE 编码器**（`events.py`）—— 单行 JSON 是唯一有真实陷阱的编码点，先做先验；
3. **校验与错误**（`validate.py`、`errors.py`）；
4. **事件序列**（`stream.py`）—— 含 FR-030 的前置块结构与 `MEDRAG_TEST_PREAMBLE` 注入点；
5. **路由与装配**（`routes.py`、`app.py`）—— **注册顺序在此，注释必须写明理由**；
6. **入口**（`serve.py`）—— 启动打印、回环绑定、日志落盘；
7. **后端验收**（quickstart §3/§4/§8/§9）—— 在写前端之前跑通 curl，确保前端调试时面对的是已确认的服务端；
8. **前端骨架**（`index.html`、`styles.css`）—— 先把蓝白配色与左右布局立起来；
9. **前端脚本**（`nav.js` → `sse.js` → `transcript.js` → `app.js`）—— nav 先做以便 US3 的结构验证；
10. **前端验收**（quickstart §2/§5/§6/§7）；
11. **文档回填**（`docs/01` §F6 拒答话术、`docs/02`、`docs/03`、`docs/05` 五处修订、`.env.example`）。

**第 7 步不可跳过**：在服务端契约未经 curl 确认前开始写前端，会把"服务端发错了"和"前端解析错了"混成同一个 bug。

---

## 与既有文档的差异（需同步修订）

| 文档 | 位置 | 现状 | 应改为 |
|---|---|---|---|
| `docs/01_需求分析.md` | §F6 | "返回固定兜底话术"但**未给出文本** | 回填逐字文本（本特性定稿，[data-model.md](./data-model.md) §3） |
| `docs/05_接口设计.md` | §3.1.1 | "为什么不流式"整节，主张 V1 同步阻塞 | 改为流式方案；N7 断言改述为"首个 `token` 事件等于话术"；保留原风险提示 |
| `docs/05_接口设计.md` | §1.1 I-01 行、§2.4、§3.1.4 | "同步请求-响应"；延迟预算按一次性返回计 | 更新为流式；延迟预算拆为"首块时间"与"整体完成时间" |
| `docs/02_架构图.md` | §373/§379、§2.1、§10 | 前端层写作 Streamlit（默认 8501） | 改为同源静态界面由问答服务托管；部署进程数由 3 降为 2 |
| `docs/03_技术选型说明.md` | §6、§423 汇总表 | 选定 Streamlit 并说明其局限 | 记录选型变更与理由 |

---

## Complexity Tracking

> 本次 Constitution Check 无违规项，本节留空。

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
