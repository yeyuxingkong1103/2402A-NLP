---
description: "Task list for 提问落盘与问题向量化（S8）"
---

# Tasks: 提问落盘与问题向量化（S8）

**Input**: Design documents from `/specs/007-query-embedding/`

**Prerequisites**: [plan.md](./plan.md)、[spec.md](./spec.md)、[research.md](./research.md)、[data-model.md](./data-model.md)、[contracts/](./contracts/)、[quickstart.md](./quickstart.md)

**Tests**: **不生成测试任务**。项目禁用 pytest（`docs/superpowers/plans` Global Constraints，`specs/005` plan.md §145）。验收以 quickstart.md 的「逐项对拍 + 边界构造」承担，作为**验收任务**编入各阶段。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件，无未完成依赖）
- **[Story]**: 所属用户故事（US1–US4）
- 每个任务均含确切文件路径

## Path Conventions

- 实现包：`backend/query/`；CLI 入口：`backend/query_embed.py`；服务接缝：`backend/api/capture.py`
- 产物：`data/questions/{yyyymmdd}.jsonl`；验收产物：`.smoke_out/`
- 全部命令以 `D:/zg6_Project/9/med_rag/rag/python.exe` 运行（constitution 原则 I）

---

## Phase 1: Setup (Shared Infrastructure)

- [x] T001 创建目录 `backend/query/` 与 `data/questions/`（含一个 `.gitkeep` 以保证空目录可辨识）

- [x] T002 [P] 在 `.gitignore` 的「数据产物」段增加 `data/questions/`（FR-029）

- [x] T003 [P] 在 `requirements.txt` 末尾追加 specs/007 的说明：**新增依赖 0**，复用 S5 已登记的 `torch` / `transformers` / `numpy` 与 `backend/embed/model.py`；并注明本特性**不引入**任何编码库（编码口径由既有模块独占，见 research R1）

**Checkpoint**: 目录与忽略规则就位

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: 本阶段完成前，任何用户故事都不得开始

- [x] T004 创建 `backend/query/__init__.py`：契约常量与错误类型。MUST 包含：
  - 路径模板：`QUESTIONS_DIR = "data/questions"`、`FILE_TEMPLATE = "{yyyymmdd}.jsonl"`
  - JSONL 字段名常量（`F_ANSWER_ID` / `F_QUESTION` / `F_ASKED_AT` / `F_EMBEDDING` / `F_VECTOR` / `F_STATUS` / `F_ERROR`）**与固定顺序元组**（contracts/store.md §2）
  - 状态值：`STATUS_OK` / `STATUS_FAILED`
  - 退出码：`EXIT_OK/ARGS/GATE/MODEL/DATA`（contracts/cli.md §2）
  - `QueryError(code, message)` 异常（带退出码，供 CLI 映射）
  - 包文档字符串 MUST 写明：**编码口径由 `backend.embed.model` 独占，本包 MUST NOT 自行编码**（FR-009/FR-032）
  - 向量维度常量 `DIM = 1024`（用于校验；MUST 与 `backend.embed` 的 `DIM` 一致，写断言或引用）

- [x] T005 创建 `backend/query/store.py`：JSONL 读写与清理。MUST 实现：
  - `append_record(record: dict, asked_at: datetime) -> Path` —— 追加写。**MUST 一次 `write()` 写入整行**（含末尾 `\n`），随即 `flush()`；`open(..., "a", encoding="utf-8", newline="\n")`（contracts/store.md §4）
  - `iter_records(paths) -> Iterator[tuple[Path, int, dict]]` —— 逐行读取，**返回行号**（`verify` 报错要指行号）
  - `rewrite_records(path, transform) -> int` —— 批量回写，MUST 走 `.tmp` + `os.replace` 原子替换（contracts/store.md §5）
  - `list_files(date=None) -> list[Path]`、`purge_files(before=None, date=None, dry_run=True) -> list[tuple[Path, int, int]]`（返回文件名/条数/字节数）
  - **模块注释 MUST 写明并发前提**：单进程 + 单事件循环 + 写入路径无 `await`；启用 `--workers N` 即失效，需改用文件锁（data-model §6）
  - 序列化 MUST 用 `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`

- [x] T006 创建 `backend/query_embed.py`：CLI 唯一入口骨架。MUST 实现：
  - `argparse` 顶层 + 四个子命令的分派（`embed` / `verify` / `purge` / `show`，各自的参数按 contracts/cli.md §1 定义）
  - `QueryError` → 退出码的映射；未预期异常 → 打印可读信息 + 非 0
  - 文件头 docstring MUST 说明必须用 `-m backend.query_embed` 运行（理由同 `backend/serve.py`）
  - 四个子命令的实现函数先留 `raise NotImplementedError`，由后续阶段填充

**Checkpoint**: 契约与持久化层就位

---

## Phase 3: User Story 1 - 提问被完整留存下来 (Priority: P1) 🎯 MVP

**Goal**: 通过校验的提问被写入 `data/questions/{yyyymmdd}.jsonl`，可追溯、无身份信息；被拒输入不进入留存。

**Independent Test**: 提交 3 个合法问题 + 3 个非法问题；磁盘恰好新增 3 行，逐字比对问题原文，扫描无身份字段。

### 实现

- [x] T007 [US1] 创建 `backend/api/capture.py`：提问采集的接缝。MUST 实现：
  - `capture_question(question: str, answer_id: str) -> None`
  - 调用 `store.append_record(...)` 写入记录（本阶段 `vector=None`、`status=failed`、`error="尚未向量化"` —— US2 会替换掉这条路径）
  - **失败 MUST NOT 抛给调用方**：捕获异常 → 记 `capture_failed` 日志（含 `answer_id` 与异常类型）→ 正常返回。日志**MUST NOT 含堆栈全文**（constitution 原则 III）
  - 模块 docstring MUST 说明这里与 constitution「不得吞掉失败条件」的关系：被禁止的是**静默**，不是**捕获** —— 失败必须可见（research R6）

- [x] T008 [US1] 在 `backend/api/routes.py` 的 `ask` 中，于 `validate_question` 之后、构造 `StreamingResponse` 之前调用 `capture_question(question, answer_id)`。**MUST 在 answer_id 生成之后**（记录里要有它），且 MUST NOT 影响返回的响应对象

### 验收（US1）

- [x] T009 [US1] 编写并运行 `.smoke_out/s8_capture_check.py`，覆盖 quickstart §2 的 7 项。**必须实测的边界**：
  - 恰好 3 行新增（不是"至少 3"）—— 多说明被拒输入进了留存，少说明有合法提问丢失
  - `question` 与提交内容逐字相同（中文，用 `httpx` 发，**不得用 `curl -d` 传中文**）
  - 扫描全部字段名与值，无 IP / User-Agent / 会话标识 / 账号
  - `answer_id` 与服务端日志中的集合相等
  - 纯空白（含全角空格 U+3000）与 201 字符均不产生新行

**Checkpoint**: US1 独立可用 —— 提问文本在磁盘上可追溯

---

## Phase 4: User Story 2 - 每个问题得到与知识库同一空间的向量 (Priority: P1)

**Goal**: 留存记录里的 `vector` 是真实的 1024 维 L2 归一化向量，由 `backend/embed/model.py` 的 `Encoder` 产出。

**Independent Test**: 取一条记录，校验维度、范数、逐位确定性；与批量路径编码同一文本结果逐位相同；与知识库向量算余弦相似度落在 [-1,1]。

### 实现

- [x] T010 [US2] 创建 `backend/query/service.py`。MUST 实现：
  - 模块级**常驻 Encoder 单例**（`get_encoder() -> Encoder`），**直接 import 自 `backend.embed.model`，MUST NOT 包装成第二层编码实现**（research R1）
  - `build_record(question: str, answer_id: str, asked_at: datetime) -> dict` —— 组装一行；`embedding` 字段取自 `encoder.fingerprint`（**已存的那份，不是重新算的**）
  - `capture_and_embed(question, answer_id) -> dict` —— 编码 + 组装（服务端与 CLI 共用的唯一入口，FR-012/FR-032）
  - 编码失败时 MUST 返回 `status=failed` + `error`（**原因，非堆栈**）的完整记录，MUST NOT 抛出让调用方决定
  - 字段顺序 MUST 与 `query/__init__.py` 的顺序元组一致

- [x] T011 [US2] 修改 `backend/api/capture.py`（T007）：改调 `service.capture_and_embed`，把返回的记录交给 `store.append_record`。失败处理保持 T007 的语义不变

- [x] T012 [US2] 修改 `backend/serve.py`：启动期加载模型并打印四行（research R7）。MUST：
  - 在 `load()` 之后、`uvicorn.run` 之前调用 `service.get_encoder()`
  - 打印：`正在加载 BGE-M3 权重（约 2.3 GB，首次约 10 秒）…` → `权重已加载（X.X s），编码指纹 XXXXXXXX…` →（门禁行由 T015 加）→ `医知源服务已启动  http://…`
  - **加载失败 → 打印可读错误 → 非 0 退出**（复用 `EXIT_CONFIG` 或 `EXIT_MODEL`），MUST NOT 以"能提问但算不出向量"的状态继续（research R2）
  - 注释 MUST 写明：启动从 < 1 s 变为 7–11 s 是设计内的，不是卡死

### 验收（US2）

- [x] T013 [US2] 编写并运行 `.smoke_out/s8_vector_check.py`，覆盖 quickstart §3 的 10 项。**重点核对**：
  - 维度 1024、L2 范数与 1 的偏差 < 1e-5、**float32 逐位往返无损**（SC-010 的前提，已实测成立）
  - 同一问题编码两次**逐位相同**（SC-004）
  - **批量路径 vs 单条查询路径**编码同一文本**逐位相同**（SC-005 —— FR-012/FR-032 的落点）
  - 1 字符与 200 字符的问题均能编码
  - 与 `data/embeddings/d6da41b5d356.npy` 第 0 行算余弦相似度，落在 [-1,1] 且与内积一致 —— **这一项证明查询向量与文档向量确实同空间**

**Checkpoint**: US1 + US2 可用 —— 问题与向量都在磁盘上，且向量与知识库同空间

---

## Phase 5: User Story 3 - 向量空间不一致时拒绝而非静默 (Priority: P2)

**Goal**: 编码参数与索引清单不一致时，启动/自检以非 0 状态失败并点名差异，且不产生任何写入。

**Independent Test**: 改动 `index_manifest.json` 里的任一指纹项，门禁必须失败并点出参数名与两个取值；失败时磁盘无新增。

> **📌 顺序提示**：US3 **不依赖 US1/US2**（门禁只用 `fingerprint()` 与清单文件，不需要模型加载）。plan.md 的实施顺序把它排在第 2 位，理由是它是本特性**唯一事后无法补救**的一项。若追求更早暴露风险，可在 Phase 2 完成后立即启动本阶段。

### 实现

- [x] T014 [US3] 创建 `backend/query/gate.py`。MUST 实现：
  - `load_index_fingerprint(manifest_path) -> dict` —— 读 `data/index_manifest.json` 的 `pipeline_config.embed` 块
  - `check_gate(manifest_path, model_dir, max_length) -> dict` —— 与本进程 `fingerprint()` 逐字段比对
  - 10 个比对字段按 research R3 的表格（`model_dir` / `config_sha256` / `weight_file` / `weight_bytes` / `max_length` / `pooling` / `normalization` / `dtype` / `dim` / `rule_version`）
  - 不一致 → 抛 `QueryError(EXIT_GATE, ...)`，消息 MUST 点出**参数名 + 索引侧值 + 查询侧值**（FR-016）
  - 清单不存在 / 非法 JSON / 缺 `embed` 块 → `QueryError(EXIT_GATE, ...)` 并说明原因，MUST NOT 跳过校验（FR-018）
  - **纯函数式、无副作用** —— 它 MUST NOT 写任何文件（FR-017）

- [x] T015 [US3] 修改 `backend/serve.py`（T012）：在模型加载之后、开始监听之前调用 `check_gate`，通过则打印 `指纹与索引一致（<collection> / <N> chunks）`；失败则打印差异并非 0 退出

- [x] T016 [US3] 实现 `backend/query_embed.py` 的 `verify` 子命令（T006 的占位）：
  - 门禁：逐字段列出 10 项「一致」或差异
  - 格式：每行合法 JSON、字段齐全、**字段顺序固定**、无空行（空行 MUST 报错而非跳过）
  - 自洽：`status=="ok"` ⟺ `vector` 非 null；`status=="failed"` ⟹ `error` 非空
  - 向量：维度 1024、范数偏差 < 1e-5
  - 统计：总条数 / ok / failed / 未处理
  - **MUST NOT 写任何文件**（quickstart §5.4 会核对 mtime 与字节数不变）

### 验收（US3）

- [x] T017 [US3] 执行 quickstart §4 的**十步破坏性对拍**。**执行前后 MUST 备份并逐字节还原** `data/index_manifest.json`：
  - `max_length` 改 2048 → 非 0 且点出 `max_length` / `4096` / `2048` 三个信息（SC-006）
  - 失败时 `data/questions/` **无新增行**（SC-007）
  - `weight_bytes` 改 1 → 同上
  - 清单改名不存在 / 内容改非法 JSON → 明确失败，MUST NOT 跳过（FR-018）
  - 还原后核对与备份逐字节相同

**Checkpoint**: 门禁可用 —— 向量空间漂移被挡在启动之前

---

## Phase 6: User Story 4 - 脚本可独立运行，也可被服务调用 (Priority: P3)

**Goal**: CLI 可批量处理已留存记录、按日期清理、查看单条；且与服务的编码路径逐位一致。

**Independent Test**: 重复运行 `embed` 第二次处理 0 条；`purge` 默认 dry-run；`show` 打印可读摘要。

### 实现

- [x] T018 [US4] 实现 `embed` 子命令（T006 的占位）：批量处理 `status != "ok"` 的记录（FR-013 幂等）。选项 `--date` / `--dry-run` / `--limit`。MUST 走 `store.rewrite_records` 的原子回写，MUST NOT 用追加写更新已有行

- [x] T019 [US4] 实现 `purge` 子命令：`--date` / `--before` / `--dry-run`。**默认 dry-run，需 `--yes` 才真删**（破坏性操作须显式确认，与 `docs/05` §5 的 `--confirm-rebuild` 同取向）。删除前打印将删的文件名、条数、字节数

- [x] T020 [US4] 实现 `show` 子命令：`--answer-id`。打印问题、时间、状态、**向量前 5 维与范数**、指纹的 `rule_version`。MUST NOT 打印完整的 1024 个数

### 验收（US4）

- [x] T021 [US4] 执行 quickstart §5：`embed` 第一次处理 N 条、**第二次处理 0 条**（SC-009）；`verify` 退出码 0；**`verify` 前后所有 jsonl 的 mtime 与字节数完全不变**；`purge` 默认 dry-run 不删；`--yes` 后目标日期文件消失且**其它日期不变**（SC-011）；`.gitignore` 命中 `data/questions/`（SC-012）

- [x] T022 [US4] 执行 quickstart §7 的并发与损坏检测：并发提交 20 个问题 → 恰好 20 行且**每行都能 `json.loads`**（无半行、无交错）；手工构造三类损坏（行间空行 / `ok` + `vector:null` / 非法 JSON）→ `verify` **必须报错并指行号**。**这三类损坏不会让正常读取失败**，只有主动构造才能证明检测真的存在。验收后还原文件

**Checkpoint**: 四个用户故事全部独立可用

---

## Phase 7: Polish & Cross-Cutting Concerns

- [x] T023 [P] 修订 `docs/01_需求分析.md` §7.2：为「用户账号、登录、历史记录」一行增加注记 —— 该条指**面向用户的会话历史**；管线侧的提问留存见 `specs/007-query-embedding`，用途限定为检索质量评估与语料缺口分析，清理方式为 `purge` 子命令。**这是 FR-030，是本特性的交付物，不是可选项** —— 不改它就会留下"文档说不做、代码在做"的矛盾

- [x] T024 执行 quickstart §6：把 `data/questions/` 置为不可写后提交合法问题 → **仍返回 200 且正常回答**；服务端日志出现 `capture_failed` 含 `answer_id`。**这一节的要害是"不失败"与"必须留痕"两件都要满足** —— 只满足前者就是 constitution 禁止的静默吞掉

- [x] T025 执行 quickstart §8 延迟预算：首个 SSE 事件的时间增加 **≤ 1 s**（SC-008）。MUST 用 `--data-binary @文件` 发中文（`curl -d '中文'` 在 git bash 下会按 cp936 发出，服务端会当成解析失败拒绝）

- [x] T026 执行 quickstart §1：核对启动的四行输出顺序与内容；实测启动耗时落在 5–15 s 并**记录到 `specs/007-query-embedding/plan.md` 的 Performance Goals**（若与 7.4/11.1 s 差异显著，说明环境变了）

- [x] T027 执行 quickstart §9 与全量回归（重跑 `.smoke_out/s8_*.py` 全部脚本）：
  - `Ctrl-C` 停止后无 Traceback
  - **重启服务后新提问追加在原有记录之后**，不覆盖、不截断（Edge Case）
  - 重复运行全部 `s8_*` 验收脚本，确认互不干扰

- [x] T028 逐条核对 `spec.md` 的 34 条 FR 与 12 条 SC 是否均有实现或验收覆盖；对**明确不在本期范围**或**未能验证**的项，在 `tasks.md` 末尾记录为已知限制，MUST NOT 静默略过

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Phase 1 (Setup)          ← 无依赖
    ↓
Phase 2 (Foundational)   ← 阻塞全部用户故事（常量 + 持久化层 + CLI 骨架）
    ↓
    ├── Phase 3 (US1, P1) 🎯 MVP  ── 留存
    │        ↓
    ├── Phase 4 (US2, P1)         ── 向量化（US1 提供记录载体）
    │        ↓
    ├── Phase 5 (US3, P2)         ── 门禁（**不依赖 US1/US2，可提前并行**）
    │        ↓
    └── Phase 6 (US4, P3)         ── CLI（依赖 US2 的 service、US3 的 gate）
             ↓
Phase 7 (Polish)         ← 依赖前六阶段全部完成
```

### 关键顺序约束（违反会返工）

1. **T010（service）必须在 T011（capture 改调 service）之前** —— 否则 capture 调一个不存在的函数
2. **T014（gate）必须在 T015（serve 接入门禁）之前**
3. **T015 必须在 T017（门禁验收）之前**，且 T017 **必须先备份 `index_manifest.json`** —— 它是 S6 的产物，改坏会连带影响后续
4. **T018–T020（CLI 子命令）必须在 T021/T022（CLI 验收）之前**
5. **T023（`docs/01` 注记）必须在本特性实现落地之后** —— 它描述的是已存在的行为
6. **T012（serve 加载模型）之后，任何"启动服务"的操作都要把等待时间放宽到 15 s 以上**，否则会得到假的"服务未就绪"

### Parallel Opportunities

**Phase 1**：T002、T003 可并行（不同文件）

**Phase 2**：T005、T006 可并行（不同文件）；T004 须先完成 —— 其余都引用它的常量

**Phase 3–5 之间**：**US3（门禁）与 US1/US2 完全无依赖**，可在 Phase 2 结束后立即并行启动。plan.md 建议尽早做（唯一事后无法补救的一项）

**Phase 7**：T023 与 T024–T026 可并行（文档 vs 验收）

### Parallel Example: Phase 2

```bash
# T004 完成后，两项可同时进行：
Task: "创建 backend/query/store.py（T005）"
Task: "创建 backend/query_embed.py CLI 骨架（T006）"
```

### Parallel Example: US3 与 US1/US2 并行

```bash
# Phase 2 完成后即可同时启动：
Task: "US1 —— capture.py + routes 接入（T007、T008）"
Task: "US3 —— gate.py + serve 接入 + verify（T014–T016）"
```

---

## Implementation Strategy

### MVP First（US1 + US2）

US1 与 US2 同为 P1，且 US1 单独交付的价值有限（记录里没有向量就只是日志）。**最小可用增量是 US1 + US2**：

1. Phase 1 Setup → 2. Phase 2 Foundational → 3. US1 → 4. US2
5. **STOP 并验证 T009 + T013**
6. 此时"提问 → 落盘 → 同空间向量"已完整，可交付

### Incremental Delivery

1. Setup + Foundational → 契约与持久化层就位
2. **US1** → 提问可追溯 → 验证
3. **US2** → 向量与知识库同空间 → 验证（**MVP**）
4. **US3** → 漂移被挡在启动之前 → 验证（可与 US1/US2 并行）
5. **US4** → CLI 可批量/清理/查看 → 验证
6. **Phase 7** → 文档与代码一致，全部 SC 有验收覆盖

### 已知限制（MUST 显式记录，不得静默略过）

实施过程中若发现以下项无法在本期验证，须在 T028 中如实记录：

| 限制 | 可能的原因 |
|---|---|
| 长时间空闲后编码是否退化 | 本机内存紧张（可用 1.41 GB / 私有 3.26 GB）。空闲 45 s 未退化已实测，更长时间未验 |
| 多 worker 下的并发写 | 本期单进程；启用 `--workers N` 即失效，需文件锁 |
| 运行期重新入库后的指纹陈旧 | 只在启动期校验一次；正解是 `docs/05` §3.3 的 `/health` `matches_index`，不在本期范围 |

---

## T028 覆盖核对：34 条 FR + 12 条 SC

**核对日期**：2026-09-27　**结论**：无遗漏项；有 4 处**已知限制**（见文末），均已记录原因与可验证时机。

### 验收脚本总览

| 脚本 | 覆盖 | 结果 |
|---|---|---|
| `.smoke_out/s8_phase2_check.py` | 字段契约、序列化、损坏检测、原子回写、清理 | 41/41 |
| `.smoke_out/s8_capture_check.py` | 提问留存、无身份信息、answer_id 关联 | 19/19 |
| `.smoke_out/s8_vector_check.py` | 向量属性、确定性、两条路径一致、与知识库同空间 | 23/23 |
| `.smoke_out/s8_gate_check.py` | 指纹门禁六类漂移 + 三类清单异常（破坏性对拍） | 25/25 |
| `.smoke_out/s8_cli_check.py` | 幂等、verify 只读性、purge 预演与真删、gitignore | 26/26 |
| `.smoke_out/s8_robust_check.py` | 20 路并发写、四类损坏检测 | 16/16 |
| `.smoke_out/s8_failpath_check.py` | 落盘失败不拖垮请求 + 必须留痕 | 14/14 |
| `.smoke_out/s8_latency_check.py` | 增量耗时、端到端首字节、写盘非瓶颈 | 5/5 |
| `.smoke_out/s8_restart_check.py` | Ctrl-C 优雅退出、重启后追加而非覆盖 | 16/16 |

**合计 185 项，另有 specs/006 的 162 项回归全部保持通过（共 347 项）。**

### 功能需求（34 条）

| 分组 | 覆盖情况 |
|---|---|
| FR-001 ~ FR-006 留存 | ✅ 全部。FR-005（被拒输入不进留存）由"恰好 3 条"这一断言双向卡住 |
| FR-007 ~ FR-013 向量化 | ✅ 全部。FR-012/FR-032（服务端与脚本同一实现）由"落盘向量 == 脚本重算向量"逐位验证 |
| FR-014 ~ FR-018 门禁 | ✅ 全部。六类漂移实测被挡；FR-017（校验先于写入）由"失败时留存行数不变"验证 |
| FR-019 ~ FR-021 落盘 | ✅ 全部。FR-021 每行自带指纹，使每行可独立判定口径 |
| FR-022 ~ FR-027 工程约束 | ✅ 全部。FR-023 单文件 ≤300 行（最大 `store.py` 294 行）；FR-024 全部命令用 `rag/python.exe` |
| FR-028 ~ FR-030 隐私约束 | ✅ 全部。FR-030 已写入 `docs/01` §7.2 |
| FR-031 ~ FR-034 | ✅ 全部 |

> **FR-023 处置记录（已闭环）**：`backend/query_embed.py` 在四个子命令全部填充后曾达到 **336 行**，超过 300 行上限。
> 收尾时已按原定方案拆分：`embed` → `backend/query/batch.py`（103 行）、`verify` → `report.py`（125 行）、
> `purge`/`show` → `maintain.py`（112 行）；`query_embed.py` 只保留参数解析与退出码映射（**164 行**）。
> **入口仍然唯一**（`-m backend.query_embed`），拆的是实现而非接口，`contracts/cli.md` 的表述边界不变。
> 拆分后已重跑 CLI 相关验收（`s8_cli_check` 26/26、`s8_robust_check` 16/16、`s8_restart_check` 16/16、
> `s8_phase2_check` 41/41），全部通过。**本项不再是未达标项。**

### 成功标准（12 条）

| 编号 | 状态 |
|---|---|
| SC-001 留存条数与原文逐字一致 | ✅ 100%（19 项断言） |
| SC-002 被拒输入新增 0 条 | ✅ |
| SC-003 身份信息字段 0 个 | ✅ 字段名与值双向扫描 |
| SC-004 编码确定性 | ✅ 逐位比较 |
| SC-005 批量 vs 单条一致 | ✅ 两条路径 + 服务端落盘三方比对 |
| SC-006 参数改动后非 0 失败 | ✅ 六类漂移全中 |
| SC-007 校验失败无新产物 | ✅ |
| SC-008 增量 ≤ 1 s | ✅ **实测中位 0.084 s** |
| SC-009 幂等 | ✅ 4 → 0 → 0 |
| SC-010 落盘读回逐位相同 | ✅ |
| SC-011 清理只影响目标日期 | ✅ 其它日期字节数不变 |
| SC-012 gitignore 命中 | ✅ |

### 三处已知限制（有意接受或未验证，非遗漏）

| 限制 | 性质 | 原因与处置 |
|---|---|---|
| **运行期重新入库后指纹陈旧** | 已知局限 | 只在启动期校验一次。**重新入库后 MUST 重启服务**（quickstart §9）。正解是 `docs/05` §3.3 的 `/health` `matches_index`，不在本期 |
| **长时间空闲后编码是否退化** | 未验证 | 空闲 45 s 未退化已实测；更长时间未验。本机可用内存 1.41 GB 而进程需 3.26 GB，靠页面文件兜住 |
| **多 worker 下的并发写** | 前提，非保证 | 单进程 + 写入路径无 `await`。启用 `--workers N` 即失效，需改用文件锁（已写在 `store.py` 模块注释里） |

### 一处实测修正

`spec.md` 起草时估"每行约 10 KB"，实测 **22,213 字节**（1024 个 float32 各约 19 字符）。已在规格中修正并标注来源。

---

## Notes

- **[P]** = 不同文件、无未完成依赖
- 项目**禁止引入 pytest**，不生成单元测试任务；验收由 quickstart.md 承担
- 全部命令 MUST 以 `D:/zg6_Project/9/med_rag/rag/python.exe -m backend.<模块>` 形式运行
- **中文请求体必须用 `httpx` 或 `curl --data-binary @文件`** —— `curl -d '{"question":"中文"}'` 在 Windows git bash 下按 cp936 发出，会得到假的 422（本项目已踩过两次）
- **避免**：把编码逻辑写进 `query/`（口径必须独占于 `backend/embed/model.py`）；用追加写更新已有行；在 `verify` 里写文件
