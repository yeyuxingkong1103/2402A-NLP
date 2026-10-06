# Implementation Plan: 混合检索（语义 + BM25 关键词）（S9）

**Branch**: `008-hybrid-retrieval` | **Date**: 2026-09-28 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/008-hybrid-retrieval/spec.md`

## Summary

实现 `docs/05` §4.2 的 **I-05 `retrieval_service.search`**：消费 S8 已算出的查询向量做语义检索，同时用 jieba 分词 + 自实现 Okapi BM25 做关键词检索，两路以 **RRF（k=60）** 融合，取 `top_k` 条原文片段，通过既有 `citations` SSE 事件送达前端。

四项裁决已定：**D1** 语料从 Milvus 全量回捞 + jieba 分词；**D2** RRF k=60、对外 `score` 保余弦；**D3** 命中后正文走「检索就绪、生成未就绪」过渡文案；**Q4/R7** 拒答阈值只管语义路，关键词命中可过。

新增 `backend/retrieve/` 包（7 个模块，单文件均 ≤ 300 行）+ CLI 入口 `backend/retrieve_search.py`；改动 `backend/api/` 四处（`capture.py` 回传向量、`stream.py` 接入检索、`__init__.py` 新增过渡文案、`config.py` 配置转正）与 `backend/serve.py`（启动期构建索引）。**前端零改动**是硬验收项（SC-008）。

## Technical Context

**Language/Version**: Python 3.12.14 —— 唯一受支持运行时 `D:/zg6_Project/9/med_rag/rag/python.exe`（constitution 原则 I）

**Primary Dependencies**:
- 既有：`pymilvus`（Milvus 客户端）、`fastapi` 0.141.1 / `starlette` 1.6.0（SSE）、`pydantic` 2.13.5 / `pydantic-settings` 2.15.0
- 本特性新增：**`jieba` 0.42.1（实测已安装，仅需登记）**
- 本特性明确**不引入**：`rank_bm25` / `bm25s`（R1）、`pytest`（R11）

**Storage**: Milvus v2.6.9 standalone（Docker，`http://localhost:19530`，无认证），collection `med_rag_v1`（FLAT / COSINE / 1024 维，实测 65 chunks / 1 文档）。BM25 索引**常驻进程内存**（R4），不落盘。

**Testing**: 无 pytest（未安装，既有 specs 均未使用）。验收沿用项目惯例：`quickstart.md` 的可复制命令 + 人工核对断言（SC-001~SC-009）。核心纯函数（`fuse.py` 的 RRF、`lexical.py` 的 BM25 打分）在 CLI 的 `selfcheck` 子命令中带自检断言。

**Target Platform**: Windows 11 单机；后端单进程 uvicorn（`backend/serve.py`，刻意不启用 reload）；前端为同源静态资源。

**Project Type**: Web 服务（后端 + 静态前端）+ 离线管线 CLI。本特性同时产出**运行时服务**与**命令行脚本**两条路径，二者 MUST 共用同一份实现（FR-023）。

**Performance Goals**: 检索耗时（不含编码与生成）在 65 chunks 规模下 **< 100 ms**；用户从提交到看见引用片段的感知等待 **≤ 2 s**（SC-003，含 S8 的编码耗时）。

**Constraints**:
- 单文件 ≤ 300 行（FR-022）
- 检索路径 MUST NOT 重新编码问题（FR-003，复用 S8 产出的向量）
- `search()` 参数 `top_k` / `threshold` 显式传入，MUST NOT 读全局默认值（docs/05 §4.2 约束 3）
- 全部配置从启动期配置读取，请求路径 MUST NOT 读环境变量（FR-024）
- 前端 `frontend/js/sse.js` 与 `frontend/js/transcript.js` **零改动**（SC-008）

**Scale/Scope**: 1 篇文档 / 65 chunks（实测）。设计在**十万条**以内不需结构性改动；以此为界记录在 R4，超过则改走 Milvus 原生稀疏向量。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 门禁内容 | 本特性的处置 | 结论 |
|---|---|---|---|
| **I. 环境锁定与依赖治理** (NON-NEGOTIABLE) | 只用 `rag/python.exe`；新增依赖登记进 `requirements.txt` | 全部命令以绝对路径书写；新增依赖 0 个包、1 条登记（`jieba`）。**发现既有矛盾**：`pymilvus` 实际 3.0.2 vs 声明 `>=2.6,<3` —— 见 Complexity Tracking | **有条件通过**（须在实现前核实） |
| **II. 无据不答与强制溯源引用** (NON-NEGOTIABLE) | 检索为空/低于阈值 MUST 拒答；每处知识 MUST 可溯源到文件名+页码 | FR-004 保证每条片段带 `file_name` / `page_start` / `page_end`；FR-011~FR-014 实现阈值判定与拒答；R9 定义基础设施失败也走拒答而非编造 | **通过** |
| **III. 密钥零硬编码** (NON-NEGOTIABLE) | 凭据只走环境变量；缺失 MUST 报错非 0 退出 | `MILVUS_URI` / `MILVUS_COLLECTION` / `MILVUS_TOKEN` 沿用既有启动期配置路径（`backend/api/config.py`）；本特性把 `REQUIRED_WHEN_RETRIEVAL_LANDS` 转正为必需项，缺失即启动失败。日志不回显 token | **通过** |
| **IV. 紧急症状前置响应** (NON-NEGOTIABLE) | 紧急话术 MUST 是首个 token | I-04 未实现，不属本特性范围。本特性 MUST NOT 破坏 `stream.py` 既有的「前置片段先于正文片段」结构 | **不适用**（不违规） |
| **V. 面向群众的医疗安全边界** (NON-NEGOTIABLE) | 输出收敛于信息参考，不越位诊疗 | 引用区呈现**原文**而非结论；D3 的过渡文案不含任何诊疗建议；`answer_text` 末尾免责声明由既有装配路径保证 | **通过** |

**Additional Constraints 复核**：
- 「检索参数 MUST 在 spec/plan 中显式记录，MUST NOT 依赖库默认值」—— `top_k`、`similarity_threshold`、`retrieval_candidates`、`rrf_k`、`bm25_k1`、`bm25_b`、`lexical_admit_rank` 全部显式配置，见 data-model.md §配置契约。
- 「函数签名 MUST 带类型注解」—— 全部新函数带完整注解。
- 「MUST NOT 以 try/except 吞掉证据不足」—— R9 按项目既有的判定标准（"事后能不能查出来"）处理，并留 ERROR 日志。

**Development Workflow 复核**：本特性的全部实现工作走 `/speckit-specify` → `/speckit-plan` → `/speckit-tasks` → 实现，未跳过规格。

**Phase 1 后复检**：见文末。

## Project Structure

### Documentation (this feature)

```text
specs/008-hybrid-retrieval/
├── plan.md              # 本文件
├── spec.md              # 已定稿（含 D1/D2/D3 + Q4 裁决）
├── research.md          # Phase 0 输出（R0–R12）
├── data-model.md        # Phase 1 输出
├── quickstart.md        # Phase 1 输出
├── checklists/
│   └── requirements.md  # 规格质量检查清单
├── contracts/
│   ├── retrieval.md     # I-05 search() 的契约与语义澄清（含 R7 的阈值扩展）
│   ├── cli.md           # backend/retrieve_search.py 的命令行契约
│   └── integration.md   # 与 S7/S8 的接缝：capture→routes→stream 的改动点
└── tasks.md             # Phase 2 输出（/speckit-tasks 生成，本命令不创建）
```

### Source Code (repository root)

> ⚠️ **与实现后的实际结构有出入，已按实际修正。** 预估的 7 个模块在实现时
> 四处超了 300 行上限（`report.py` 443、`service.py` 409、`checkup.py` 354、
> `retrieve_search.py` 335），随后 `selfcheck.py` 在补入 SC-10 后又涨到 340。
> 按 FR-022 逐层拆分为 **13 个文件**。**每个文件的实际行数见括注**，全部 ≤ 300。

```text
backend/
├── retrieve/                    # 【新增】S9 检索实现包（实际 13 个文件，2519 行）
│   ├── __init__.py              # 常量 / 错误类型 / 配置契约（185 行）
│   ├── models.py                # RetrievedPassage / RetrievalResult / Candidate（141 行）
│   ├── store.py                 # 唯一碰 Milvus：连接 + 全量语料 + 向量检索（236 行）
│   ├── lexical.py               # jieba 分词 + BM25 索引 + 检索 + 覆盖率（255 行）
│   ├── fuse.py                  # RRF 融合 + 去重 + 确定性排序（142 行）
│   ├── bundle.py                # IndexBundle / SearchTrace / 索引的构建与生命周期（230 行）
│   ├── service.py               # search() 唯一入口 + 阈值裁定 + 日志（281 行）
│   ├── report.py                # CLI 报告渲染（163 行）
│   ├── checkup.py               # 语料一致性检查（102 行）
│   ├── cli_support.py           # CLI 装配：参数 + .env → 索引 / 查询向量（143 行）
│   ├── selfcheck.py             # 自检执行器（56 行）
│   ├── selftest_cases.py        # 10 条断言（294 行）
│   └── selftest_data.py         # 自检用的构造语料（50 行）
├── retrieve_search.py           # 【新增】CLI 唯一入口（241 行）
├── api/
│   ├── __init__.py              # 【改】新增 RETRIEVAL_READY_NOTICE 过渡文案
│   ├── capture.py               # 【改】capture_question 回传查询向量（R8）
│   ├── stream.py                # 【改】status → 检索 → citations → token → done
│   └── config.py                # 【改】REQUIRED_WHEN_RETRIEVAL_LANDS → REQUIRED_NOW；新增检索配置项
├── serve.py                     # 【改】启动期构建 BM25 索引；Milvus 不可达即启动失败
└── query/                       # 【不改】S8，仅被读取（service.get_encoder / gate）

frontend/                        # 【零改动】硬验收项 SC-008
├── js/sse.js
└── js/transcript.js

requirements.txt                 # 【改】登记 jieba；修正 pymilvus 版本声明
.env.example                     # 【改】新增检索配置项；转正 milvus_* / top_k / similarity_threshold
docs/05_接口设计.md               # 【改】回写 §4.2 的阈值适用范围（R7）
```

**Structure Decision**：选**方案 2（Web 应用）**。项目既有布局即 `backend/`（Python 服务 + 离线管线）+ `frontend/`（同源静态资源），本特性沿用，不新增顶层目录。

`backend/retrieve/` 的划分依据是**「什么时机可能出错」**（与 `backend/query/`、`backend/index/` 同一取向）：
- 启动期会出错 → `store.py`（Milvus 不可达）、`lexical.py`（语料为空、分词器未预热）
- 请求期会出错 → `service.py`（编排、阈值判定）
- 纯计算、只可能算错 → `fuse.py`（可作为纯函数单独验证）
- 纯输出 → `report.py`

这样切分让 FR-022（单文件 ≤ 300 行）与「检索 MUST NOT 写库」两件事在**结构上可见**：只有 `store.py` 持有 Milvus 连接，而它是只读的。

## Complexity Tracking

| 违规 / 风险项 | 为何需要 | 更简单的替代方案为何不可行 |
|---|---|---|
| **`pymilvus` 版本声明与实际不符**（实际 3.0.2 / 声明 `>=2.6,<3`） | 本特性首次在**运行时服务**中使用 `MilvusClient`。已知 S6（离线 CLI）能工作，但声明矛盾会让"换台机器装依赖"直接装出不同大版本 | 视而不见不可行：constitution 原则 I 要求登记表与实际安装一致，且这是本特性唯一可能"在别人的机器上直接跑不起来"的点。**处置见 tasks.md 的首个任务：先核实 `MilvusClient` 的 `query` / `search` API 兼容性，再决定是放开上界还是降版本** |
| **多了 8 个模块**（`backend/retrieve/` 7 个 + CLI 1 个） | FR-021/FR-022 要求模块化且单文件 ≤ 300 行；实现总量预估约 1090 行，单文件必然超限 | 单文件不可行（直接违反 FR-022，且与用户原始要求「超过 300 行就分模块写」冲突）。归并到 3–4 个模块可行，但会让 `store.py` 同时承担 Milvus 访问与 BM25 计算，破坏"只有 store 碰 Milvus"的结构保证 |
| **对 `docs/05` §4.2 的契约扩展**（R7：阈值只管语义路，`below_threshold` 重定义） | 原契约只设想单路检索；混合检索下按字面执行会让 US1 的精确命中被判为不相关，本特性价值归零 | 不改契约不可行（见 R7 的三方案对比）。**但 MUST 回写 `docs/05` §4.2**，MUST NOT 静默改契约 —— 已列为独立任务 |
| **S8 行为变更：Milvus 由可选变必需** | 语料构建在启动期（FR-010），索引为空则关键词路失效 | "Milvus 不可用时启动成功但每次检索降级"被 spec 的 Edge Case 明令禁止（静默降级）。变更本身正确，但 MUST 在启动输出与 quickstart 中写明前置条件 |
| **对 `backend/api/capture.py` 返回值的改动** | FR-003 禁止重新编码，而向量已经算好并当前被丢弃 | 改 `stream.py` 让它自己编码不可行（第二编码入口，S8 整篇规格在防它）；让 `routes.py` 调两次 `capture_and_embed` 不可行（双倍推理耗时且写两条记录） |

## 关键实现约束（供 tasks 拆解时逐条落为任务）

1. **语料拉取**：`backend/retrieve/store.py` 的 Milvus 全量 `query` MUST 显式给 `limit`，且在返回条数等于 limit 时告警（R3）。MUST 与 `index_manifest.json` 的 `total_chunks` 比对。
2. **分词一致性**：`lexical.py` 的 `tokenize()` MUST 是索引期与查询期的**唯一分词入口**，MUST NOT 存在第二个 `jieba.lcut` 调用点（R2）。
3. **BM25 公式锁定**：IDF 用 `ln(1 + (N − df + 0.5)/(df + 0.5))`（恒正），`k1` / `b` 为显式常量（R1）。
4. **排序确定性**：`fuse.py` 的排序键 MUST 为 `(-rrf_score, -cosine, chunk_id)` —— 末位 `chunk_id` 保证同分时顺序稳定（SC-006）。
5. **去重**：融合前按 `chunk_id` 去重，同一 chunk MUST 只出现一次并合并两路名次（FR-005）。
6. **启动期预热**：jieba 首次调用加载词典，MUST 在 `serve.py` 的启动序列里预热（R2）。
7. **启动顺序**：指纹门禁 → 加载 BGE-M3 → **构建 BM25 索引** → 起 HTTP。门禁（最便宜的检查）MUST 仍最先（S8 的既有取向）。
8. **请求链路**：`routes.py` 把 `capture_question` 的返回值透传给 `stream_answer`；`stream_answer` 在 `status` 帧之后、`citations` 帧之前执行检索。
9. **只读检测**：`backend/retrieve/` 内 MUST NOT 出现任何 Milvus 写操作（`insert` / `delete` / `upsert` / `create_collection`）。

## Phase 1 后 Constitution Check 复检

| 原则 | Phase 1 设计产物的影响 | 结论 |
|---|---|---|
| I. 环境锁定与依赖治理 | data-model.md 的配置契约把 7 个检索参数全部显式化；contracts/cli.md 的调用示例全部用 `rag/python.exe` 绝对路径 | 通过（`pymilvus` 版本待核实项已进 Complexity Tracking 与 tasks 首项） |
| II. 无据不答与强制溯源引用 | contracts/retrieval.md 明确 `RetrievalResult` 四字段语义与 R9 的三条失败处置；data-model.md 的 `RetrievedPassage` 强制携带文件名与页码 | 通过 |
| III. 密钥零硬编码 | contracts/integration.md 规定 `MILVUS_TOKEN` 只从启动期配置读取、日志脱敏 | 通过 |
| IV. 紧急症状前置响应 | contracts/integration.md 明确 `stream_answer` 的片段顺序中「紧急话术前置」的位置不变（本特性只在 citations 之前插入检索） | 不适用（未破坏既有结构） |
| V. 面向群众的医疗安全边界 | data-model.md 收录 D3 过渡文案的措辞要求：不含诊断、不含剂量、明示"生成能力未就绪" | 通过 |

**门禁结论**：全部通过（`pymilvus` 版本核实为**实现期首个任务**，不阻塞 plan/tasks）。
