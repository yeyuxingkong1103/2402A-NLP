---
description: "Task list for 混合检索（语义 + BM25）实现"
---

# Tasks: 混合检索（语义 + BM25 关键词）（S9）

**Input**: Design documents from `specs/008-hybrid-retrieval/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: 本特性**不生成测试任务**。项目未安装 pytest（R11），既有 001–007 全部 specs 均不用测试框架，验收靠 `quickstart.md` 的可复制命令 + 人工核对。替代方案是 CLI 的 `selfcheck` 子命令（T029–T031），它把纯函数的断言做成可执行的形式。

> ## ⚠️ 实现期偏离记录（2026-09-28，全部 38 项已执行）
>
> 下面几处**任务描述与最终实现不一致**。保留原描述（它是当时的计划），偏离记在这里：
>
> | 任务 | 计划 | 实际 | 原因 |
> |---|---|---|---|
> | T002 | `backend/retrieve/` 7 个模块 | **11 个模块** | 实现后 4 个文件超 300 行（`report.py` 443、`service.py` 409、`checkup.py` 354、`retrieve_search.py` 335），按 FR-022 逐层拆分：`bundle.py`（索引生命周期）、`selfcheck.py`（纯函数断言）、`checkup.py`（语料检查）、`cli_support.py`（CLI 装配） |
> | T004 | 新增 5 个检索配置项 | **6 个** | 补了 `LEXICAL_MIN_COVERAGE`（见 T020 行） |
> | T020 | 纳入条件 = 余弦 ≥ 阈值 **或** 名次 ≤ 排名线 | 再加**查询词覆盖率 ≥ 0.5** | 对真实语料验证时发现无关提问被放行（违反 SC-004 / 宪法原则 II），提交裁决后补入。详见 spec.md 的 FR-013a 与 contracts/retrieval.md §2 |
> | T014 | `below_threshold` 按 `docs/05` 原文语义 | 收紧为「有候选但全部不达标」 | 原公式 `(not is_empty) and (...)` 让 `is_empty=True` 的两种情形拿到同一个值，违反 FR-013。失效现场是日志里 `below_threshold=False` 与 `state=below_threshold` 自相矛盾 |
> | T026/T027 | 终帧复用 `docs/05` §3.1.3 | 另加一条：**终帧的 `citations` MUST 与 `citations` 事件一致** | `schemas.not_ready_response` 原先把 citations 写死为 `[]`，而前端 `renderDone` 用终帧重渲染引用区 —— 表现为引用在终帧到达时**被清空**。SSE 端到端验证时发现 |
> | T001 | 修正 `pymilvus` 版本声明 | 核实为 3.0.2 与 2.x 用法兼容，声明改为 `>=3.0,<4` | 实测 `query`/`search` 签名；（另：`jieba` 并非新依赖，`backend/clean/space_merge.py` 从 S2 起就在用，本特性只是**补登记**） |
>
> **T036 的一个坑**：任务是「grep 确认无 `insert`/`delete` 等写操作」，但
> `checkup.py` 有一行 `lines.insert(0, ...)`（**list** 方法）会被正则误报。
> 精确的检查是看 `client.*` 调用 —— 本特性只有 `client.query` 与 `client.search`。

**Organization**: 按用户故事分组。注意两处**刻意的顺序调整**，理由见下：

- **US2 的 phase 排在 US1 之前**：两者同为 P1（spec 里 US1 是术语命中、US2 是口语命中）。但 RRF 融合需要**两路都存在**，而语义路是关键词路之前就能独立交付的那一半。先建语义路，US1 的增量才是纯粹的"加了关键词路"。**优先级未变**（都是 P1），变的只是构建顺序。
- **CLI 的 `search` 子命令落在 Foundational，而不是 US4**：它是 US1/US2 独立验收的**唯一手段**（不启服务、不开浏览器即可断言命中）。US4 保留的是 `selfcheck` 与 `corpus` 两个子命令 —— 那才是 US4 独有的交付。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- **[Story]**: 所属用户故事（US1 / US2 / US3 / US4）
- 每个任务都带**确切文件路径**

## Path Conventions

项目为 Web 应用布局：`backend/`（Python 服务 + CLI）+ `frontend/`（同源静态资源，**本特性零改动**）。

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 先把会阻塞一切的既有欠账和环境准备清掉

- [X] T001 **核实 `pymilvus` 版本声明**：实测运行时装的是 `3.0.2`，而 `requirements.txt` 声明 `pymilvus>=2.6,<3` —— 两者矛盾。用 `D:/zg6_Project/9/med_rag/rag/python.exe -c "from pymilvus import MilvusClient; import inspect; print(inspect.signature(MilvusClient.query)); print(inspect.signature(MilvusClient.search))"` 确认 3.0.2 的 `query`/`search` 签名是否与 `backend/index/store.py`（按 2.x 写的）兼容，据结论**修正 `requirements.txt` 的版本声明**（放开上界或指定 `==3.0.2`）。依据：plan.md Complexity Tracking 第 1 项、research.md R11
- [X] T002 创建 `backend/retrieve/` 包骨架（7 个空模块文件 + `__init__.py` 导出约定）：`backend/retrieve/__init__.py`、`models.py`、`store.py`、`lexical.py`、`fuse.py`、`service.py`、`report.py`。依据：plan.md Project Structure
- [X] T003 [P] 在 `requirements.txt` 登记 `jieba==0.42.1`（实测已安装，仅补登记），并在注释中说明它的用途与"BM25 自实现、不引入 rank_bm25"的裁决。依据：research.md R1/R11、FR-025
- [X] T004 [P] 在 `.env.example` 新增 5 个检索配置项（`RETRIEVAL_CANDIDATES=20`、`RRF_K=60`、`LEXICAL_ADMIT_RANK=3`、`BM25_K1=1.5`、`BM25_B=0.75`），并把 `MILVUS_URI` / `MILVUS_COLLECTION` / `EMBED_MODEL_PATH` / `SIMILARITY_THRESHOLD` / `TOP_K` 的"转为必需"注释改成"必需"。依据：data-model.md §4、contracts/integration.md §7
- [X] T005 [P] 确认前置条件可满足：`docker ps` 含 `milvusdb/milvus:v2.6.9`；`data/index_manifest.json` 的 `total_chunks == 65`。若不满足，**先解决再继续**（本特性让 Milvus 成为启动硬前提）。依据：quickstart.md §0.1、§0.3

**Checkpoint**: `pymilvus` 版本矛盾已澄清，包骨架就位，配置模板已更新

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 用户故事的共同地基 —— 契约模型、语料、纯函数、配置、验证入口

**⚠️ CRITICAL**: 本阶段完成前，任何用户故事都无法开始

- [X] T006 实现常量与错误类型：`backend/retrieve/__init__.py` —— 路径常量（`PROJECT_ROOT` / `INDEX_MANIFEST`）、Milvus 契约常量（`COLLECTION_NAME` / `VECTOR_FIELD` / `SCALAR_FIELDS` / `DIM` / `QUERY_LIMIT` / `MAX_LENGTHS`）、`RetrievalError`（**不继承 `ValueError`**）、退出码（0/1/2/3）、`assert_dim_matches_index_package()`。依据：data-model.md §4、research.md R3、contracts/cli.md §3
- [X] T007 定义契约模型：`backend/retrieve/models.py` —— `RetrievedPassage`（8 字段，Pydantic）、`RetrievalResult`（4 字段）、`CorpusChunk`、`Candidate`。字段名 MUST 与 `docs/05` §4.2 逐字段一致。依据：data-model.md §1、§2
- [X] T008 [P] 实现 RRF 融合纯函数：`backend/retrieve/fuse.py` —— `merge(semantic_hits, lexical_hits, k)` 按 `chunk_id` 去重、合并两路名次、算 `rrf_score`、排序键为 `(-rrf_score, -cosine, chunk_id)`。**必须是纯函数**（无 I/O），便于 `selfcheck` 断言。依据：research.md R5、data-model.md §2.1、plan.md 关键约束 4/5
- [X] T009 [P] 实现报告渲染：`backend/retrieve/report.py` —— `search` 子命令的人类可读输出（含 `余弦`/`BM25`/`语义名次`/`关键词名次`/`RRF`/`来源` 六列）、`--json` 输出、`selfcheck` 与 `corpus` 的结果渲染。依据：contracts/cli.md §2
- [X] T010 实现 Milvus 只读访问：`backend/retrieve/store.py` —— `connect()`（惰性导入 pymilvus，缺依赖报退出码 3）、`fetch_corpus()`（全量 `query`，**显式 `limit`**，条数 == limit 时告警，与 `index_manifest.json` 的 `total_chunks` 比对）。**本模块 MUST 只读**，MUST NOT 出现 `insert`/`delete`/`upsert`/`create_collection`。依据：research.md R3、plan.md 关键约束 1/9
- [X] T011 配置转正与新增字段：`backend/api/config.py` —— 把 `REQUIRED_WHEN_RETRIEVAL_LANDS` 的 5 项整体移入 `REQUIRED_NOW`（S7 已标好转正时机），新增 `retrieval_candidates` / `rrf_k` / `lexical_admit_rank` / `bm25_k1` / `bm25_b` 五个带默认值的字段，并加校验（`TOP_K >= 1`、`RETRIEVAL_CANDIDATES >= TOP_K`、`0 <= BM25_B <= 1` 等，见 data-model.md §4）。报错 MUST NOT 回显 `MILVUS_TOKEN` 的值。依据：contracts/integration.md §7、constitution 原则 III
- [X] T012 创建 CLI 入口骨架：`backend/retrieve_search.py` —— 以 `-m backend.retrieve_search` 方式运行，子命令框架 + `search` 子命令（`--top-k` / `--threshold` / `--candidates` / `--json` / `--no-semantic` / `--no-lexical`，后两者互斥）。**CLI 只做参数解析与调 `service.search`**，MUST NOT 自己实现检索。依据：contracts/cli.md §1/§2/§4、FR-023

**Checkpoint**: 能连库拉语料、能跑 RRF 纯函数、CLI 能起（此时 `search` 尚无实现，预期报错）

---

## Phase 3: User Story 2 - 口语化提问也能检索到 (Priority: P1)

**Goal**: 用户用日常说法描述症状，仍能得到知识库里相关的原文片段。

**Independent Test**: 用不含任何原文术语的口语化问题跑 CLI，断言引用区非空且首条与主题相关。**不依赖 US1 的关键词路**（此时 `--no-lexical` 与默认行为等价）。

- [X] T013 [US2] 实现语义检索：`backend/retrieve/store.py` 的 `search_semantic(client, query_vector, limit)` —— 调 Milvus `search`，`output_fields` 取自 `SCALAR_FIELDS`（**不含 `vector`**，避免把 1024 维向量拉回来），返回 `[(chunk_index, cosine)]`，已按余弦降序。维度不符（≠ 1024）时抛 `RetrievalError`。依据：contracts/retrieval.md §5、data-model.md §5
- [X] T014 [US2] 实现检索编排（单路版本）：`backend/retrieve/service.py` 的 `search(question, query_vector, top_k, threshold)` —— 调用语义路、组装 `RetrievalResult`、实现 R7 的阈值判定与 `is_empty` / `below_threshold` / `top_score` 三字段语义、按 §6 的状态表留结构化日志。索引单例走 `build_index()` / `get_index()`（**启动期构建、请求期复用**）。依据：contracts/retrieval.md §1/§2/§6、data-model.md §6、research.md R7
- [X] T015 [US2] 打通 CLI 的 `search` 子命令到 `service.search`，含 `--no-lexical`。依据：contracts/cli.md §2
- [X] T016 [US2] 验证：跑 `quickstart.md` §3.1（口语化问题）与 §3.4（标定阈值），人工确认首条相关。记录观察到的余弦分布

**Checkpoint**: 口语化提问能命中；`is_empty` / `below_threshold` 四象限行为正确

---

## Phase 4: User Story 1 - 照抄指南术语也能检索到那一页 (Priority: P1) 🎯 MVP 差异化价值

**Goal**: 用户把纸质指南里的术语原样敲进去，看到**写着这几个字的那一段**。

**Independent Test**: 用知识库中确实存在的罕见术语跑 CLI，断言首条引用的 `text` 包含该术语，且 `file_name` / `page_start` 与原文位置一致。**本故事是"混合"相对于"纯语义"的全部增量。**

- [X] T017 [US1] 实现分词：`backend/retrieve/lexical.py` 的 `tokenize(text)` —— `jieba.lcut` + 过滤空白/纯标点。**本函数 MUST 是索引期与查询期的唯一分词入口**，全仓 MUST NOT 存在第二个 `jieba.lcut` 调用点。依据：research.md R2、plan.md 关键约束 2
- [X] T018 [US1] 实现 BM25 索引构建：`backend/retrieve/lexical.py` 的 `build_index(corpus)` —— 倒排 `postings` / `doc_len` / `avgdl` / `df` / `n_docs`，文档长度在**启动期缓存**（`tokens` / `tokens_len` 存入 `CorpusChunk`）。空语料时 `avgdl` 取 1.0 且不抛异常。依据：data-model.md §3/§2.2
- [X] T019 [US1] 实现 BM25 打分与关键词检索：`backend/retrieve/lexical.py` 的 `search(index, question, limit)` —— IDF 用 `ln(1 + (n_docs − df + 0.5)/(df + 0.5))`（**恒正**，R1 锁定），`k1`/`b` 从配置传入，返回 `[(chunk_index, bm25)]` 降序。依据：research.md R1、plan.md 关键约束 3
- [X] T020 [US1] 把关键词路接入编排：改 `backend/retrieve/service.py` 的 `search` —— 两路各取 `RETRIEVAL_CANDIDATES` 条，交 `fuse.merge`，按 R7 的纳入条件（`cosine >= threshold` **或** `lexical_rank <= lexical_admit_rank`）过滤后取 `top_k`。**`score` 字段 MUST 保余弦，关键词路独有命中取 `0.0`**。依据：contracts/retrieval.md §2、research.md R6/R7
- [X] T021 [US1] 启动期构建索引：改 `backend/serve.py` —— 在"加载 BGE-M3"之后插入索引构建步骤；预热 jieba；打印语料条数、是否与 manifest 一致、检索配置生效值。Milvus 不可达 / 语料为空 → 非 0 退出；与 manifest 不符 → 告警但不退出。依据：contracts/integration.md §8、plan.md 关键约束 6/7
- [X] T022 [US1] 打通 CLI 的 `--no-semantic` / `--no-lexical` 与 `--candidates`，使"混合 vs 单路"可测量。依据：contracts/cli.md §2
- [X] T023 [US1] 验证：跑 `quickstart.md` §3.2（术语命中）、§3.3（混合不劣于单路）、§4（可复现性 `diff` 两次 `--json` 输出）。断言首条引用原文包含该术语，并确认余弦 < 阈值时 `below_threshold=True` 是**正确行为**而非故障

**Checkpoint**: 术语照抄能命中；两路融合生效；同问题两次结果逐字节一致

---

## Phase 5: User Story 3 - 引用片段在界面上可见且可核对 (Priority: P1)

**Goal**: 检索结果经既有 SSE 通道送达前端，引用区展示可核对的卡片。

**Independent Test**: 浏览器提交问题，断言引用区渲染出与服务端返回**同序、同数量**的卡片，字段逐项一致；且 `git diff --stat frontend/` 无输出。

- [X] T024 [US3] 回传查询向量：改 `backend/api/capture.py` 的 `capture_question()` 返回类型为 `list[float] | None`。**MUST NOT 向调用方抛异常的既有契约不变**；落盘失败但编码成功时**仍返回向量**（两个独立的失败域）。依据：research.md R8、contracts/integration.md §2
- [X] T025 [US3] 透传：改 `backend/api/routes.py` —— `query_vector = capture_question(...)` 并传给 `stream_answer(..., query_vector=...)`。`_SSE_HEADERS` 三条头 MUST 保持不变。依据：contracts/integration.md §3
- [X] T026 [US3] 新增过渡文案：改 `backend/api/__init__.py` —— 加 `RETRIEVAL_READY_NOTICE`，措辞满足 contracts/integration.md §5 的四条要求（传达"找到了资料"+"生成未就绪"+引导看引用+MUST NOT 含诊疗建议），并写明**移除时机 = I-06 接入时**。用户可见文案只此一处
- [X] T027 [US3] 接入检索到事件流：改 `backend/api/stream.py` —— `stream_answer` 增加 `query_vector` 参数；在 `status` 帧之后、`citations` 帧之前执行检索；`query_vector is None` 时留 ERROR 日志**且不调 search**（MUST NOT 静默退化为只跑关键词路）；捕 `RetrievalError` 后留 ERROR 日志 + 走拒答（**不冒泡成 500**）；`_body_chunks` 按 contracts/integration.md §5 的三路径产出正文。**事件顺序 `status → citations → token* → done` 与既有 `completed` / 取消分支 MUST NOT 改动**。依据：contracts/integration.md §4/§5、research.md R9
- [X] T028 [US3] 验证：跑 `quickstart.md` §5（启动服务 + 浏览器验证 + 日志检查）。重点确认：引用卡片六项齐全、展开原文可用、SC-003 的 ≤2s、`grep "检索完成"` 六列齐全、**`git diff --stat frontend/` 无输出（SC-008）**。若前端需改动，MUST 改服务端

**Checkpoint**: 端到端可用 —— 浏览器能看到真实引用片段，前端零改动

---

## Phase 6: User Story 4 - 检索可脱离服务单独自检 (Priority: P2)

**Goal**: 运维人员在无 Web 服务、无浏览器的情况下验证检索逻辑与语料一致性。

**Independent Test**: 跑 `selfcheck`（不连库、不加载权重）9 条断言全 PASS；跑 `corpus` 报告与 manifest 一致。二者退出码均可断言。

- [X] T029 [US4] 实现 `selfcheck` 子命令的纯函数断言（SC-1 ~ SC-9，见 contracts/cli.md §2）：BM25 IDF 恒正、分数对词频单调不减、空语料不抛异常；RRF 双路加权 > 单路、`k` 增大时头部差距缩小、同分排序稳定、同 chunk 去重合并名次；`below_threshold` 四象限；`tokenize` 幂等。写在 `backend/retrieve_search.py` + 断言函数放 `backend/retrieve/report.py`。依据：contracts/cli.md §2、R11（替代 pytest）
- [X] T030 [US4] 实现 `corpus` 子命令：连库拉语料并报告条数 / 与 manifest 一致性 / `chunk_id` 唯一性 / 空 `text` 数 / 空 `file_name` 数 / `page_start > page_end` 数 / 平均文档长度 / 高频词 top10。三项检查任一不符即非 0 退出。依据：contracts/cli.md §2
- [X] T031 [US4] 验证：跑 `quickstart.md` §1、§2，确认退出码与预期一致；跑 §6 的边界用例（纯标点问题、Milvus 运行中挂掉、空 collection）

**Checkpoint**: 四个用户故事全部独立可验证

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T032 [P] **回写 `docs/05_接口设计.md` §4.2** —— 补两处契约扩展：① `search()` 签名增加 `query_vector` 参数（以及为何不内部编码）；② 混合检索下 `is_empty` / `below_threshold` / `top_score` 三字段的新语义与阈值适用范围（R7）。**MUST NOT 静默改契约**。依据：contracts/retrieval.md 开头、research.md R7
- [X] T033 [P] 回填 `TODO(SIMILARITY_THRESHOLD)`：把 T016/T023 标定出的阈值写入 `.specify/memory/constitution.md` 与 `docs/02_架构图.md` §10。依据：constitution 的 TODO 项、quickstart.md §3.4
- [X] T034 [P] 更新 `docs/02_架构图.md` §10 与 `.env.example`，补齐本特性新增的 5 个配置项说明。依据：constitution「密钥与配置」
- [X] T035 行数检查（FR-022 / SC-007）：`find backend/retrieve backend/retrieve_search.py -name "*.py" | xargs wc -l`，确认**每个文件 ≤ 300 行**；超出则拆分模块。依据：plan.md Project Structure 的行数预估
- [X] T036 只读性检查（plan.md 关键约束 9）：在 `backend/retrieve/` 内 grep `insert|delete|upsert|create_collection|drop_collection`，确认**无命中**
- [X] T037 密钥与日志检查（constitution 原则 III）：确认新代码无硬编码密钥、无裸 `python` 调用（`grep -rn "python\b" backend/retrieve*` 只应命中绝对路径）、检索日志不含 `query_vector` 数值与片段全文
- [X] T038 端到端回归：完整走一遍 `quickstart.md` §7 的 12 项自检清单，逐项打勾

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
   └─▶ Phase 2 (Foundational)  ⚠️ 阻塞全部用户故事
          ├─▶ Phase 3 (US2 语义路)
          │      └─▶ Phase 4 (US1 关键词路 + RRF 双路)   ← 依赖 US2，见下
          │             └─▶ Phase 5 (US3 上屏)
          └─▶ (US3 的 T024/T025/T026 可与 Phase 3/4 并行 —— 不同文件)
                        Phase 6 (US4 CLI 自检)  ← 依赖 Phase 4（断言对象是两路与融合）
                              └─▶ Phase 7 (Polish)
```

### 为什么 US1 依赖 US2（**唯一一处故事间依赖**）

RRF 融合需要两路同时存在。US2 交付的语义路是关键词路之前能独立成立的那一半，因此先建。**优先级未变**（US1 与 US2 同为 P1）—— 变的只是构建顺序。若团队有两人，US3 的 API 改动（T024–T026）与 Phase 3/4 完全并行（不同文件）。

### Within Each User Story

- 模型（T007）→ 服务 → CLI 接线 → 验证
- `fuse.py`（T008）是纯函数，必须先于 T020 的双路接入完成
- 每个 phase 结尾都有独立的验证任务（T016 / T023 / T028 / T031）

### Parallel Opportunities

| 机会 | 任务 | 理由 |
|---|---|---|
| Setup 并行 | T003, T004, T005 | 不同文件 |
| Foundational 并行 | T008, T009 | `fuse.py` 与 `report.py` 无交集 |
| 与 Phase 3/4 并行 | T024, T025, T026 | `backend/api/` 的三个文件，不被检索包触及 |
| Polish 并行 | T032, T033, T034 | 不同文档 |

---

## Parallel Example: Foundational 阶段

```bash
# 三个互不依赖的任务可同时开工（不同文件）：
Task: "T008 实现 RRF 融合纯函数 in backend/retrieve/fuse.py"
Task: "T009 实现报告渲染 in backend/retrieve/report.py"
Task: "T011 配置转正与新增字段 in backend/api/config.py"
```

## Parallel Example: US3 与 US1 并行

```bash
# API 侧改动（US3）与检索包改动（US1）互不冲突：
Task: "T024 回传查询向量 in backend/api/capture.py"
Task: "T026 新增过渡文案 in backend/api/__init__.py"
Task: "T019 实现 BM25 打分与关键词检索 in backend/retrieve/lexical.py"
```

---

## Implementation Strategy

### MVP First

**最小可交付 = Phase 1 + Phase 2 + Phase 3（US2）** —— 一条能给出真实引用的语义检索链路，可独立演示。

但**本特性的差异化价值在 Phase 4（US1）** —— 没有它，这就是一个普通的向量检索，与"混合检索"的需求不符。建议 MVP 一次做到 **Phase 4 结束**。

### Incremental Delivery

| 里程碑 | 完成到 | 能演示什么 |
|---|---|---|
| M1 | Phase 2 | 能连库拉语料、RRF 纯函数可单测 |
| M2 | Phase 3 (US2) | CLI 上，口语化提问能返回相关引用 |
| M3 | Phase 4 (US1) | **术语照抄能精确命中** —— 混合检索成立 |
| M4 | Phase 5 (US3) | 浏览器里能看到引用卡片（前端零改动） |
| M5 | Phase 6 (US4) | 全部验收手段就位，可标定阈值 |
| M6 | Phase 7 | 契约回写完成，可交付 |

### 验收顺序的建议

T029 的 `selfcheck`（US4）虽然属 P2，但**值得在 Phase 4 之前就写** —— 它是纯函数断言，不依赖任何外部状态。若实现 BM25/RRF 时先有它，后面排查会省很多事。tasks 里仍按 US4 排在后面，是遵循 spec 的优先级组织；实现者可视情况提前。

---

## Notes

- [P] 任务 = 不同文件、无未完成依赖
- **前端 `frontend/**` 全程零改动**（SC-008）—— 任何时候发现需要改，MUST 改服务端
- 每个 phase 结束都停在 Checkpoint 上独立验证，不要把验证攒到最后
- 三条容易写错的硬约束，实现时逐条自检：
  1. `score` 字段恒为余弦（关键词路独有命中取 `0.0`），RRF 分只进日志 —— contracts/retrieval.md §2
  2. 全仓只有一个 `jieba.lcut` 调用点 —— research.md R2
  3. `backend/retrieve/` 内无任何 Milvus 写操作 —— plan.md 关键约束 9
