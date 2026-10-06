---
description: "Task list for 005-milvus-ingest (S6 入库)"
---

# Tasks: 入库（S6 步骤）

**Input**: Design documents from `specs/005-milvus-ingest/`

**Prerequisites**: [plan.md](./plan.md)、[spec.md](./spec.md)、[research.md](./research.md)、[data-model.md](./data-model.md)、[contracts/cli.md](./contracts/cli.md)、[quickstart.md](./quickstart.md)

**Tests**: 本项目**禁止引入 pytest**（`docs/superpowers/plans` 的 Global Constraints，既有约定）。验证 = **逐项对拍 + 边界构造**，脚本落在 `.smoke_out/`。因此本文件里的验证任务不是"可选测试"，而是**每个故事达到完成状态的定义**。

**Organization**: 按用户故事分组，每个故事可独立实现与验证。

## 进度（2026-09-27）

**T001–T034 全部完成。** 由用户实际执行入库与五个验收脚本，结果：

| 验收 | 结果 |
|---|---|
| `s6_us1`（US1 / SC-001 / SC-007） | PASS 10 / FAIL 0 —— 含自检索命中自己 `score 1.000000` |
| `s6_us2`（US2 / SC-002） | PASS 5 / FAIL 0 —— 65 行不翻倍，`chunk_id` 与 `text` 均无重复 |
| `s6_us3`（US3 / SC-003） | PASS 2 / FAIL 0 —— 库内哈希 == 现场重算值（**仅验了"放行"方向，见下**） |
| `s6_us4`（US4 / SC-009） | PASS 30 / FAIL 0 —— 清单字段与库/产物逐项一致 |
| `s6_us5`（US5 / SC-004） | PASS 6 / FAIL 0 —— 故障注入触发补偿回滚，库回到 65 行，退出码 2 |

**仍未实测覆盖的两条路径**（如实记录，不得当作已验证）：

1. **退出码 4（版本门禁拒绝写入）** —— 只验了"一致则放行"。构造不一致的步骤已打印在
   `s6_us3.py` 的输出里（方式 A：把 `backend/chunk/core.py` 的 `HIGH` 临时改成 701）。
   **这是 US3 的核心行为，建议补测。**
2. **退出码 5（回滚也失败）** —— 需要让回滚阶段的写入也失败，无法稳定构造。以代码走查
   覆盖；`s6_us5.py` 输出了手工构造步骤。

### schema 扩充：新增 `chunk_meta`（JSON 字段，2026-09-27 用户指示）

**问题**：原设计里 `text` 列存的是**展示用原文**（不含章节路标），而向量是拿
`text_for_embedding`（**路标 + 正文**）算出来的。于是"**这条 1024 维向量到底是从哪串文本
算出来的**"在库里无法核对 —— 只能拿 `text` 和 `section` 现场重拼，而重拼口径一旦与 S5
不一致，你发现不了。这正是 `docs/04 §8` 要防的静默漂移。

**修正**：新增 `chunk_meta`(JSON) 字段，装「**S4 整条记录减去已单独成列的字段**」。
用户从 A/B/C 三选项中选了 **C（单个 JSON 字段）**。理由：Milvus 没有数组类型，
`heading_path` / `block_ids` 无论如何都要自己 JSON 编解码；既然反正要编解码，整条塞进一个
JSON 字段比加 4 个 VARCHAR + 1 个 INT32 更省事，且 **S4 将来加字段时 S6 一行都不用改**。

**改动的文件**：

| 文件 | 改动 |
|---|---|
| `backend/index/__init__.py` | 新增 `META_FIELD` / `META_MAX_BYTES`(65536) / `META_KEY_PATTERN` / `META_REQUIRED_KEYS`；`SCALAR_FIELDS` += `chunk_meta`（该元组同时驱动 schema 核对、回滚备份取回字段、meta 排除集） |
| `backend/index/inputs.py` | 新增 `build_chunk_meta()`（派生规则只有这一处）与 **V12 校验**（必需键 / 键名字符集 / 序列化 ≤ 65536 字节） |
| `backend/index/store.py` | `_build_schema()` += `FieldSchema(dtype=DataType.JSON)`；`build_entities()` += 1 行；schema 不符的报错信息里**直接给出 drop 重建命令** |
| `docs/04 §9.1` | schema 表 += `chunk_meta` 行 + 专节说明 |
| `specs/005/spec.md` | FR-009 补 `chunk_meta`；新增 FR-009a（内容规则）/ FR-009b（三项校验） |
| `specs/005/data-model.md` | §1 表 += 1 行；新增 §1.1 专节；§2.3 更正（原「只消费 11 个字段」已不成立）；§4 校验表 += V12 |
| `.smoke_out/s6_us1.py` | 新增 chunk_meta 校验段（**直接 import `build_chunk_meta`**，不重写一遍派生规则，否则等于自己验自己） |

**两个约束（已实测确认）**：Milvus 单个 JSON 字段上限 **65536 字节**（实测最大约 8 KB，
8 倍余量）；JSON 键只允许字母/数字/下划线。另记录一个 pymilvus 坑：JSON 值含 **numpy 类型**
时插入失败且报错信息指向错误（[#2886](https://github.com/milvus-io/pymilvus/issues/2886)）——
本管线取值自 `json.load`，不受影响。

**运维动作**：schema 变更 → **必须 drop 重建 collection**（代码按设计不会自动改 schema）。
65 行，源产物都在，drop 后重跑入库即可。

### 产出路径的更正（2026-09-27，用户指示）

产出**最终落在项目内的 `med_rag\data\`**，与 `docs/04 §9.3` 的 `data/index_manifest.json` 一致：

- `med_rag\data\index_manifest.json`
- `med_rag\data\index_backup\{doc_id}.rollback.jsonl`

实施中途曾按用户口头指示把产出放到项目外的 `D:\zg6_Project\data`，随后用户更正为项目内。
已完成的更正：`index_milvus.py` 的默认路径、`contracts/cli.md §2/§6`、`docs/04 §9.3`、
`s6_us4.py` 的 `MANIFEST` 常量、`s6_us5.py` 的说明文本；`index_manifest.json` 与
`index_backup\` 已从 `D:\zg6_Project\data` 迁回项目内，源目录恢复原样（仅剩原有的
`MakerDown\`、`source\`）。

### 实现阶段对 plan 的两处偏离（运行后才暴露）

1. **`create_collection` 的 `index_params` 必须是 `IndexParams` 对象**，不能传 `list[dict]`。
   原实现传了 list → `ParamError`。改为 `client.prepare_index_params()` + `add_index()`。
2. **新增 `store.ensure_index()`**：`create_collection` 是「先建表、再建索引」两步，中途失败会
   留下**没有索引的空 collection**（第 1 版实现就是这么留下的），而它既不能 load 也不能
   delete/query。原 `assert_collection_compatible` 遇到这种情况直接报错，只能靠人工 drop 重建。
   现改为「索引缺失则按 V1 锁定值补建」（唯一确定、非破坏性），schema 不符仍报错。
   `ensure_collection` 的返回值随之从 `bool` 变成 `(是否新建, 是否补建索引)`。

### 用户授权范围内、由控制者执行的只读检查

- `compileall` 语法编译通过；入口 **276 行**（≤300 ✓）
- `--print-env` / `--print-config` 通过；后者打出 `chunk_max_chars = 700`，
  证明常量是从 `backend/chunk/core.py` import 的而非从文档抄写
- 反例构造（覆盖 V1/V2/V4/V5/V6/V7/V9/V10）全部正确触发并退出码 2；
  **V12 只有正向路径被覆盖**（65 条产物的 chunk_meta 全部通过校验），
  它的三条失败分支（键名非法 / 超过 65536 字节 / 缺必需键）**未构造反例**

## Global Constraints（贯穿全部任务，逐条 MUST 遵守）

1. **解释器**：一律用绝对路径 `D:/zg6_Project/9/med_rag/rag/python.exe`。**MUST NOT** 出现裸 `python` / `python3` / `py`（宪法原则 I）。
2. **禁止自行运行入库脚本**（用户明确要求）。本特性的**控制者只做**：静态检查、语法编译（`compileall`）、以及**只读**命令（`--print-env` / `--print-config` / 连库查询）。**任何会写 Milvus 或写 `data/index_manifest.json` 的验证，MUST 交由用户执行**（见 `quickstart.md`）。
3. **禁止自行安装依赖**。`pymilvus` 的安装命令交用户执行（宪法「开发工作流」）。
4. **入口行数 ≤ 300 行**（用户要求）。超出拆入 `backend/index/`，对外入口只有一个。
5. **不得为了让代码通过而放宽校验**：V1–V12（`data-model.md §4`）任一失败 MUST 以退出码 2 结束，MUST NOT 降级为警告。
6. **`except` 不得吞掉失败条件**（宪法「代码规范」）。补偿式回滚的异常 MUST 重新抛出并落盘证据。
7. **凭据只走环境变量**（宪法原则 III）。日志与错误信息 MUST NOT 回显 `MILVUS_TOKEN` 的值。
8. **类型注解**：函数签名 MUST 带类型注解，公开函数 MUST 带返回类型注解；标识符 `snake_case`。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件，且不依赖未完成任务）
- **[Story]**: 所属用户故事
- 每条任务都带确切文件路径

## Path Conventions

单项目布局。本特性触及的路径：

```text
backend/index_milvus.py        # 唯一入口
backend/index/__init__.py      # 常量、退出码、IndexError
backend/index/inputs.py        # 读产物 + V1–V12 校验
backend/index/versions.py      # pipeline_config_hash
backend/index/store.py         # Milvus 交互（唯一会写库的模块）
backend/index/manifest.py      # index_manifest.json
.smoke_out/s6_us*.py           # 验收脚本（交用户执行）
data/index_manifest.json       # 产出物
data/index_backup/             # 回滚备份
```

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 目录骨架与依赖登记

- [x] T001 [P] 建 `backend/index/` 包目录与占位 `backend/index/__init__.py`（仅模块 docstring），并确认 `data/index_backup/` 为运行时创建的目录（**不预建、不入库**，由 `store.py` 在需要时创建）
- [x] T002 [P] 在 `requirements.txt` 的"S6 入库"注释行处登记 `pymilvus`（不锁小版本；安装后回填实测版本号）

> **T002 的执行依赖人工**：`pymilvus` 当前未安装（实测 `ModuleNotFoundError`）。安装命令见 `quickstart.md` 第 1 步，**由用户执行**。

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 全部故事共用的常量与错误类型

**⚠️ CRITICAL**: 本阶段完成前，任何用户故事都不得开始

- [x] T003 实现 `backend/index/__init__.py` 的常量与错误类型：
  - `COLLECTION_NAME = "med_rag_v1"`、`VECTOR_FIELD = "vector"`、`PRIMARY_FIELD = "chunk_id"`
  - `DIM = 1024`、`INDEX_TYPE = "FLAT"`、`METRIC_TYPE = "COSINE"`
  - `MAX_LENGTHS: dict[str, int]` —— 按 `plan.md`「已确定 1」的取值（`text=16384`、`section=512`、`file_name=512`、`chunk_id=64`、`doc_id=64`、`source_hash=64`、`pipeline_config_hash=64`、`block_type=32`）
  - `EXIT_OK, EXIT_ARGS, EXIT_VALIDATION, EXIT_DEP, EXIT_GATE, EXIT_ROLLBACK = 0, 1, 2, 3, 4, 5`（`contracts/cli.md §3`）
  - `STRONG = "Strong"`、`DEFAULT_URI = "http://localhost:19530"`、`TOKEN_ENV = "MILVUS_TOKEN"`
  - `class IndexError(Exception)`，携带 `code: int` 与 `message: str`（**与内建 `IndexError` 重名，须用 `from __future__ import annotations` 并在模块内以别名导出**，实现时注意不要遮蔽内建名）

**Checkpoint**: 常量就位，可以开始用户故事

---

## Phase 3: User Story 1 - 一份文档的 65 个分块完整进入检索库 (Priority: P1) 🎯 MVP

**Goal**: 跑一条命令把 `d6da41b5d356` 的 65 个分块写入 Milvus，跑通"建表 → 建索引 → load → 写入 → 校验"全链路。

**Independent Test**: 清空 collection 后跑一次，断言 collection 行数 == 65，索引为 FLAT/COSINE，且抽 20 条 `text`/`page_start`/`section` 与 `chunks.jsonl` 逐字一致。

### Implementation for User Story 1

- [x] T004 [P] [US1] 在 `backend/index/inputs.py` 实现四个 loader：`load_matrix()`（`.npy`，校验 dtype=float32、shape[1]==1024）、`load_rows()`、`load_chunks()`、`load_fingerprint()`；**只认精确文件名** `{doc_id}.npy`，MUST 忽略 `{doc_id}.npy.tmp.npy`（E9）
- [x] T005 [US1] 在 `backend/index/inputs.py` 实现 `validate()` —— V1–V12 全量校验（`data-model.md §4`）。要点：**V2 行对齐必须逐条比**（`rows[i].chunk_id == chunks[i].chunk_id`，MUST NOT 退化为集合比较）；**V10 用 `len(s.encode("utf-8"))` 按字节判长**并报出是哪条 `chunk_id` 的哪个字段、实际多少字节、上限多少（依赖 T004）
- [x] T006 [P] [US1] 在 `backend/index/store.py` 实现 `connect()` 与 `ensure_collection()`：`create_collection(schema=…, index_params=[{field_name:"vector", index_type:"FLAT", metric_type:"COSINE"}])` 一次调用完成建表 + 建索引 + load；schema 用 `CollectionSchema` 显式声明 11 个字段（`data-model.md §1`），`enable_dynamic_field=False`；collection 已存在时 MUST 用 `describe_collection` 核对 schema 与 §9.1 一致，不符则报错退出（E11）
- [x] T007 [US1] 在 `backend/index/store.py` 实现 `build_entities()`：按 `rows[i].row_index` 取 `.npy` 的第 `row_index` 行，与 `chunks[i]` 的字段组装成实体 dict。**MUST NOT 依赖"读文件顺序恰好一致"**（依赖 T004、T006）
- [x] T008 [US1] 在 `backend/index/store.py` 实现 `insert_and_count()`：`insert()` → `flush()` → 以 `consistency_level=STRONG` 查 `count(*)`；返回 `(insert_count, count)`（依赖 T006）
- [x] T009 [US1] 在 `backend/index_milvus.py` 实现入口骨架：`argparse`（位置参数 `doc_ids`、`--collection`/`--uri`/`--chunks-dir`/`--emb-dir`/`--manifest`/`--backup-dir`、`--dry-run`/`--print-env`/`--print-config`/`--rebuild`，见 `contracts/cli.md §2`）、`main()` 返回退出码、六段报告骨架（`contracts/cli.md §4`）、`sys.path` 引导与 `stdout` UTF-8 重配置（对齐 `backend/embed_chunks.py` 的既有写法）（依赖 T005、T007、T008）
- [x] T010 [US1] 写 `.smoke_out/s6_us1.py` 验收脚本：断言行数 65、`describe_index` 返回 `FLAT`/`COSINE`、抽 20 条逐字对拍 `text`/`page_start`/`page_end`/`section`/`block_type`/`doc_id`/`file_name`（**交用户执行**）

**Checkpoint**: 首次入库链路跑通，US1 可独立验证

---

## Phase 4: User Story 2 - 同一份文档重复入库，库里内容不变 (Priority: P1)

**Goal**: 重复运行同一命令，库内行数不变、`chunk_id` 集合不变。

**Independent Test**: 连跑两次同样命令，两次退出码 0，第二次后行数仍为 65，无重复 `chunk_id`。

### Implementation for User Story 2

- [x] T011 [US2] 在 `backend/index/store.py` 实现 `count_for_doc()` 与 `delete_for_doc()`：删除 MUST 用 `filter="doc_id == '<id>'"`（D1），MUST NOT 出现无过滤条件的 delete/drop（依赖 T006）
- [x] T012 [US2] 在 `backend/index_milvus.py` 接线"先删后插"路径与 `[4/6]`/`[5/6]` 报告段（依赖 T009、T011）
- [x] T013 [US2] 写 `.smoke_out/s6_us2.py`：连跑两次后断言行数仍 65、两次 `chunk_id` 集合相同、检索同一 chunk 结果一致（**交用户执行**）

**Checkpoint**: US1 与 US2 均可独立验证

---

## Phase 5: User Story 3 - 参数变更时拒绝写入 (Priority: P1)

**Goal**: 参数版本与库内不符时拒绝写入、非 0 退出、库一字未改，并报出差异的具体参数。

**Independent Test**: 构造一处参数不一致，确认退出码 4、库内容逐条不变、错误信息点名差异字段。

### Implementation for User Story 3

- [x] T014 [P] [US3] 在 `backend/index/versions.py` 实现 `collect_pipeline_config()`：数值参数 **MUST 从上游代码 import**（`from chunk.core import LOW, HIGH, MIN_RATIO`、`from clean import CLEAN_RULE_VERSION`），MUST NOT 从文档抄写；再并入 `chunks.jsonl` 的 `chunk_rule_version` 与 `fingerprint.json` 的全部 10 个字段（`research.md` R5）
- [x] T015 [US3] 在 `backend/index/versions.py` 实现 `compute_hash()`（字典序确定性序列化 + sha256）与 `describe_diff(old, new)`（逐项列出差异参数；旧值取自 `data/index_manifest.json` 的 `pipeline_config`，缺失时如实说明"无法逐项比对"）（依赖 T014）
- [x] T016 [US3] 在 `backend/index/store.py` 实现 `read_stored_hash()`（从任意一行读 `pipeline_config_hash`）与门禁判定：集合为空 → 无门禁；不一致且未带 `--rebuild` → 退出码 4（依赖 T006、T015）
- [x] T017 [US3] 在 `backend/index_milvus.py` 接线 `--print-env` / `--print-config` / `--dry-run` / `--rebuild` 四个开关；`--rebuild` MUST 显式给出才绕过门禁（FR-020、FR-021）（依赖 T009、T015、T016）
- [x] T018 [US3] 写 `.smoke_out/s6_us3.py`：用 `--print-config` 取得本次 hash 与库内值比对；构造不一致（临时改 `HIGH`）断言退出码 4 且库未变（**交用户执行**）

**Checkpoint**: 三个 P1 故事全部可独立验证 —— MVP 达成

---

## Phase 6: User Story 4 - 索引的「身份证」被落盘 (Priority: P2)

**Goal**: 入库后写出/更新 `data/index_manifest.json`，且与库互为镜像。

**Independent Test**: 读该文件逐项核对；重复入库不追加；多文档时两份都在。

### Implementation for User Story 4

- [x] T019 [P] [US4] 在 `backend/index/manifest.py` 实现 `read_manifest()` / `merge_by_doc_id()` / `write_manifest()`：按 `doc_id` 覆盖条目而非追加；`built_at` 用本次运行时刻刷新（ISO-8601 带时区）
- [x] T020 [US4] 在 `backend/index/manifest.py` 实现 `rebuild_documents_from_store()`：`doc_id` 全集 = 已有 manifest ∪ 本次处理集，逐个 `count(*)` 核实，**计数为 0 的条目移除**（FR-022a / `research.md` R8）
- [x] T021 [US4] 在 `backend/index_milvus.py` 接线 manifest 写入：`app_config = {"top_k": 3, "similarity_threshold": 0.6}`（`plan.md`「已确定 2」），`page_count = max(page_end)`（依赖 T012、T019、T020）
- [x] T022 [US4] 写 `.smoke_out/s6_us4.py`：断言字段齐备、`total_chunks==65`、`documents` 长度 1（重复跑后仍为 1）、`pipeline_config_hash` 与库内一致、`pipeline_config.chunk_max_chars==700`（**交用户执行**）

**Checkpoint**: US1–US4 均可独立验证

---

## Phase 7: User Story 5 - 失败不留半份，错误可定位 (Priority: P2)

**Goal**: 任何写入失败都让库回到运行前状态；回滚本身失败时明确报告"库状态不确定"并给出恢复依据。

**Independent Test**: 注入一次插入计数不符，断言回滚触发、库回到 65 行、退出码 2；再注入回滚失败，断言退出码 5 且给出备份文件路径。

### Implementation for User Story 5

- [x] T023 [US5] 在 `backend/index/store.py` 实现 `capture_old_rows()`：`count_for_doc > 0` 时查回该 `doc_id` 全部字段并落盘 `data/index_backup/{doc_id}.rollback.jsonl`；目录不存在则创建（`research.md` R4 第 2 步，**回滚能力的唯一来源**）
- [x] T024 [US5] 在 `backend/index/store.py` 实现 `verify_and_rollback()`：计数不符 → 删 `doc_id` → 回填备份 → 复验 `== N_old`；复验失败 MUST 以退出码 5 结束并**在错误信息中给出备份文件绝对路径**（FR-013、`contracts/cli.md §3`）（依赖 T011、T023）
- [x] T025 [US5] 在 `backend/index/store.py` 加故障注入钩子：**仅当** 环境变量 `MEDRAG_TEST_FAULT=insert_count` 时伪造 `insert_count`，用于验证回滚路径；MUST NOT 影响正常路径（`quickstart.md` §7）（依赖 T008）
- [x] T026 [US5] 在 `backend/index_milvus.py` 接线退出码 5 路径与多文档的"逐份报结果 + 取最严重退出码"逻辑（`contracts/cli.md §3`）（依赖 T009、T024）
- [x] T027 [US5] 写 `.smoke_out/s6_us5.py`：注入 `insert_count` 故障断言回滚到 65 行 + 退出码 2；构造回滚失败断言退出码 5 + 备份路径出现在错误信息里（**交用户执行**）

**Checkpoint**: 五个用户故事全部可独立验证

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: 交付说明、文档回写、全量走查

- [x] T028 [P] 在 `backend/index_milvus.py` 头部写完整 docstring：这个脚本做什么 / 运行方法（含**依赖安装命令**）/ 退出码表 / 两条必须知道的约定（索引建在空集合上、`max_length` 按字节计）。对齐 `backend/embed_chunks.py` 的既有格式
- [x] T029 验证 `backend/index_milvus.py` **≤ 300 行**；若超出，把可搬的逻辑移入 `backend/index/` 的相应模块，**对外入口保持唯一**（用户要求）
- [x] T030 [P] 回写 `docs/04_数据管线设计.md §10.1`：更正 `source_hash` 与 `doc_id` 的定义（D1 要求；当前说法与实测矛盾 —— `source_hash` 是 MinerU `_origin.pdf` 的 sha256，`doc_id` 才是源 PDF 的 sha256 前 12 位）
- [x] T031 [P] 回写 `docs/04_数据管线设计.md §7`：把已被 S4 的 D1 作废的「300–500 字 + 禁止跨章节合并」更正为实际生效的 **300–700**（下限 500→300 的理由见 `backend/chunk/core.py:10-12`）与"仅同父章节可合并"
- [x] T032 [P] 回写 `docs/04_数据管线设计.md §9.3`：`pipeline_config` 示例中的 `chunk_max_chars: 500` → **700**（`chunk_min_chars: 300` 本就正确，保留）
- [x] T033 按 `quickstart.md` 的 10 步做**全量走查**：第 2 步（三条只读命令）由控制者执行；第 3–10 步（会写库）**交用户执行**，控制者核对用户回传的输出
- [x] T034 校验 `--print-env` / `--print-config` 的实际输出与 `contracts/cli.md §4` 的报告格式约定一致；`--print-config` 的 `chunk_max_chars` 必须是 **700**（若打出 500，说明常量不是 import 来的，违反 T014）

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Phase 1 Setup (T001–T002)
      │
      ▼
Phase 2 Foundational (T003)  ← BLOCKS 全部用户故事
      │
      ├──────────────┬──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼              ▼
   US1 (T004–T010)  US2           US3            US4            US5
   🎯 MVP          (T011–T013)   (T014–T018)    (T019–T022)    (T023–T027)
      │              │              │              │              │
      └──────────────┴──────────────┴──────────────┴──────────────┘
                                     │
                                     ▼
                        Phase 8 Polish (T028–T034)
```

### User Story Dependencies

| 故事 | 依赖 | 说明 |
|---|---|---|
| **US1** (P1) | 仅 Phase 2 | **无跨故事依赖**，可独立交付 |
| **US2** (P1) | US1 的 `store.py` / 入口骨架 | 复用 US1 的连接与建表；验证独立 |
| **US3** (P1) | US1 的连接 | 门禁读的是 US1 写入的字段 |
| **US4** (P2) | US1 的写入 + US2 的计数函数 | 用 `count(*)` 核实 documents |
| **US5** (P2) | US2 的 `delete_for_doc` | 回滚 = 删 + 回填 |

**结论**：US1 是真正的 MVP；US2/US3 紧随其后（同为 P1，三者合起来才构成"敢用的入库"）；US4/US5 是完备性。

### Within Each User Story

- loader / 校验 → store 的写库函数 → 入口接线 → 验收脚本
- 校验（T005）MUST 在写库函数之前完成并接进入口：**FR-005 要求一切校验先于任何写操作**，顺序错了就失去意义

### Parallel Opportunities

- **T001 / T002** 可并行
- **T004 / T006** 可并行（不同文件：`inputs.py` vs `store.py`）
- **T014** 可与 US1/US2 的收尾并行（只依赖 `__init__.py`）
- **T019** 可与 US3 并行（不同文件）
- **T028 / T030 / T031 / T032** 可并行（入口 docstring 与三处 docs 回写互不相干）

**冲突提示（MUST NOT 并行）**：T005 与 T004 同文件；T011/T023/T024/T025 同在 `store.py`，**必须串行**；T012/T017/T021/T026 同在入口文件，**必须串行**。

---

## Parallel Example: User Story 1 起步

```bash
# 两条互不相干的支路可同时开工：
Task: "在 backend/index/inputs.py 实现四个 loader（T004）"
Task: "在 backend/index/store.py 实现 connect + ensure_collection（T006）"

# T004 完成后：
Task: "在 backend/index/inputs.py 实现 validate V1–V12（T005）"
```

---

## Implementation Strategy

### MVP First

1. Phase 1 Setup → 2. Phase 2 Foundational → 3. **Phase 3 (US1)** → **STOP**：US1 跑通即证明"磁盘产物 → 可检索的库"这条链是通的
2. 再补 US2 + US3（同为 P1）→ 此时才具备"敢重跑、参数变了会拒绝"的可用性
3. 最后 US4 + US5 + Polish

### 交付形态（用户明确要求）

**脚本写完不运行**。交付时给出：

- `pymilvus` 的安装命令（由用户执行）
- 可直接复制的入库命令
- 退出码表
- `.smoke_out/s6_us1.py` … `s6_us5.py` 五个验收脚本（由用户执行）

控制者只执行：`compileall` 语法检查、`--print-env`、`--print-config`、以及 `--dry-run`（只读，不写库）。

---

## Notes

- [P] = 不同文件且无未完成依赖
- 每个用户故事可独立完成与验证
- **验证脚本必须先写、且预期在实现前失败**（本项目以对拍替代 pytest，但"先失败后通过"的顺序不变）
- 每条任务完成后即可标记 `- [x]`；跨文件的重构性改动须回到本文件更新受影响的任务
