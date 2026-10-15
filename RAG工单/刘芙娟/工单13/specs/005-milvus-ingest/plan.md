# Implementation Plan: 入库（S6 步骤）

**Branch**: `005-milvus-ingest` | **Date**: 2026-09-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/005-milvus-ingest/spec.md`

## Summary

把 `data/embeddings/{doc_id}.npy`（float32[N,1024]）+ `{doc_id}.rows.jsonl` + `data/chunks/{doc_id}.chunks.jsonl` + `{doc_id}.fingerprint.json` 写入 Milvus v2.6.9（`http://localhost:19530`），并写出 `data/index_manifest.json`。

技术路线：`pymilvus` 的 `MilvusClient` 高层 API；`create_collection(schema=…, index_params=…)` 一次性建表并自动建索引 + load；写入采用「**先取旧 → 删 → 插 → 校验 → 失败则用旧数据回滚**」的补偿式原子提交（Milvus 2.6 无客户端可见事务，这是唯一可达的原子性）；写入前用 `pipeline_config_hash`（D2）做版本门禁。

三项**由本步骤实测确定、且与既有文档的字面表述有出入**的技术事实，已在 `research.md` 中逐条记录理由：

1. Milvus 的 VARCHAR `max_length` 按**字节**计（上限 65535）。实测最长 `text` 为 **7,794 字节**，按字符数估算会漏判。
2. `delete` / `query` 要求 collection **已 load**；而 load 又要求向量字段**已有索引**。因此索引必须在**空集合**上先建 —— 这是平台强制的顺序。
3. `docs/04 §9.2` 字面要求"写入完成后统一建索引"，在 Milvus 上**无法实现**（建索引前无法 load，未 load 无法 delete/query）。本计划按 §9.2 的**意图**（避免索引看到中途状态）处置：索引建在**空集合**上，之后才发生写入，中途状态从未被索引看到。

## 索引选型：为什么是 `FLAT` + `COSINE`

**本节不是新做的决定。** `FLAT` + `COSINE` 由三处既有文档锁定：

| 出处 | 原文 |
|---|---|
| `docs/04_数据管线设计.md:527` | **索引配置（V1 锁定）**：`FLAT` 索引 + `COSINE` 度量。 |
| `docs/04_数据管线设计.md:517` | `\| vector \| FLOAT_VECTOR(1024) \| .npy 第 i 行 \| COSINE 度量 \|` |
| `docs/02_架构图.md` §7 参数表 | `\| 索引类型 \| FLAT（暴力检索） \| 语料规模小，召回率优先于速度 \|` |
| `docs/02_架构图.md` §7 参数表 | `\| 距离度量 \| COSINE \| 语义检索标准做法，与归一化后的 BGE-M3 输出匹配 \|` |
| `docs/05_接口设计.md:136` | `\| Milvus 检索（FLAT，< 100 chunk） \| 0.1 s \| …` |

本节做的是**补上这三处没有写出的论证**，并给出**换索引的判据**——否则将来语料变大时，没人知道当初为什么选 FLAT、也就不知道什么时候该换。

### 1. Milvus 索引的构成

Milvus 的索引不是一个东西，是三件套的拼装：

- **数据结构**（粗筛）：`IVF`（倒排聚类）/ 图结构（`HNSW`、`DiskANN`）/ 无（`FLAT`）
- **量化器**（可选，省内存与算力）：`SQ8`（8 位标量量化）/ `PQ`（乘积量化）
- **refiner**（可选，补精度）

类型名即拼装结果：`IVF_PQ` = IVF + 乘积量化，`HNSW_SQ` = HNSW + 标量量化。

### 2. 候选范围：先按度量筛掉一半

度量已锁定 `COSINE`。**COSINE 只被 `FLAT` / `IVF_FLAT` / `IVF_SQ8` / `IVF_PQ` / `HNSW` / `DISKANN`（及 GPU 系列）支持**；`SCANN` 与 `HNSW_SQ` / `HNSW_PQ` / `HNSW_PRQ` 的度量是**内积**，在此步即排除。本机无 GPU（`torch 2.12.1+cpu`，Milvus 为单机 Docker），GPU 系列排除。

剩下的候选：`FLAT`、`IVF_FLAT`、`IVF_SQ8`、`IVF_PQ`、`HNSW`、`DiskANN`。

### 3. 在本项目的规模上比较（65 条 / 1024 维）

| 索引 | 65 条时省下的时间 | 召回损失 | 可调参数个数 | 需训练 |
|---|---|---|---|---|
| **FLAT** | — | **0** | **0** | 否 |
| `HNSW` | **≈ 0 ms** | 有 | 3（`M`/`efConstruction`/`ef`） | 否（建图慢） |
| `IVF_FLAT` | **≈ 0 ms** | 有 | 2（`nlist`/`nprobe`） | 是 |
| `IVF_SQ8` / `IVF_PQ` | ≈ 0 ms | 更大 | 2 + 量化参数 | 是 |
| `DiskANN` | 负（更慢） | 有 | 1（`search_list`） | 是，且需高速 SSD |

**FLAT 每次查询的代价**：65 行 × 1024 维 ≈ **6.7 万次乘加**，CPU 上是**几十微秒**。

**而单次检索的真实耗时由另外两件事支配**：

1. 网络往返（脚本/后端 ↔ Milvus 容器）
2. 返回 payload —— 实测 `text` 最长 **7,794 字节**，`top_k = 3` 时单次返回二十余 KB

**结论：在本项目的规模上，近似索引连 1 毫秒都省不下来，却一定要收走召回率。** 这不是权衡，是**净亏**。

### 4. 召回损失在这个系统里的具体后果（决定性理由）

近似索引漏掉一条本该命中的段落，在这套系统里的表现是：

1. 检索返回的证据不足 → 按**宪法原则 II（无据不答与强制溯源引用）**，系统**拒答**
2. 而这个问题的答案**原文里明明有**

于是拒答率悄悄上升，且**无报错、无日志、无异常**——正是本项目一路在防的"静默失效"。

更糟的是，`docs/01_需求分析.md:277` 已经预言了它会被误诊：

> 若上线后因医疗术语、药品名、指南编号等精确字符串召回不足导致拒答率偏高，**第一优先级是补 BM25，而不是放宽相似度阈值**——放宽阈值会使"无据不答"失效，属于倒退而非优化。

即：**召回损失会把人引去调错旋钮**。FLAT 让这条失效路径根本不存在。

### 5. FLAT 没有可调参数 —— 这是特性，不是简陋

`HNSW` 有 `M` / `efConstruction` / `ef`，`IVF` 有 `nlist` / `nprobe`。每一个都是"**改了以后检索悄悄变差、但不报错**"的旋钮 —— 这正是 §10.2 的 `pipeline_config_hash` 门禁要防的那类漂移。

`FLAT` 一个参数都没有，这类风险**直接归零**。这与本项目"宁可少一个旋钮，也不要一个会静默劣化的旋钮"的一贯取向一致。

### 6. 建索引成本

本流程的索引建在**空集合**上（见 `research.md` R3 —— Milvus 强制要求先有索引才能 load，未 load 不能 delete/query）。

- `FLAT`：**无训练阶段**，建索引即登记，瞬间完成，零成本。
- `HNSW`：建图，慢。
- `DiskANN`：建 Vamana 图 + PQ，更慢。

FLAT 让 R3 这个被迫的顺序调整不产生任何额外代价。

### 7. 为什么不用 `AUTOINDEX`

Milvus 官方把 `AUTOINDEX` 推荐为通用默认，但在本项目上有两个具体问题：

1. **它是黑盒。** 官方文档说明开源 Milvus 中 `AUTOINDEX` 默认落到 **`HNSW`** —— 即给你一个**有召回损失、有三个可调旋钮**的索引，而这里只需要 65 条精确检索。且该选择可能随 Milvus 版本变化。
2. **它让版本门禁失效。** `docs/04_数据管线设计.md:658` 把 `index_type` 列为**触发重建索引**的条件之一，说明这个值是要被**记录并比对**的。若记录 `AUTOINDEX`，记的不是实际索引；若记录服务端实际选中的那个，它可能在你不知情时改变 —— **`pipeline_config_hash` 门禁防的正是这件事**。用一个"服务端会自己变的值"当记录，等于把门禁关掉。

### 8. 换索引的判据

判据不是"感觉慢了"，而是**规模门槛**：

| chunk 数 | 建议 | 说明 |
|---|---|---|
| **< 10 万** | **`FLAT`** | CPU 上仍在几十毫秒以内 |
| 10 万 ~ 100 万 | `FLAT` 仍可用，开始评估 `HNSW` | 需实测延迟 |
| > 10^6，内存装得下 | `HNSW` | **换之前必须先量召回损失** |
| > 10^8，装不下内存 | `DiskANN` | 需高速 SSD |

**换索引前的前置动作**：按 `docs/02_架构图.md:403` 确立的方法构造**已知答案的查询集**，量出候选索引相对 FLAT 的召回损失，再决定是否值得换。**不能凭"官方推荐 HNSW"就换。**

当前规模 **65 条**，距第一个门槛差**三个数量级**。`docs/04:131` 说"语料扩到多文件、chunk 数上千后…"，**上千条对 FLAT 依然毫无压力**。

### 9. 落到实现

```python
client.create_collection(
    collection_name="med_rag_v1",
    schema=schema,                      # docs/04 §9.1 的 11 个字段
    index_params=[{
        "field_name": "vector",
        "index_type": "FLAT",
        "metric_type": "COSINE",
    }],
)
```

`FLAT` 无需任何附加参数。实测需验证：`describe_index` 返回的 `index_type == "FLAT"` 且 `metric_type == "COSINE"`（见 `quickstart.md` 第 4.2 步）。

## Technical Context

**Language/Version**: Python 3.12.14（`D:/zg6_Project/9/med_rag/rag/python.exe`，宪法原则 I 锁定）

**Primary Dependencies**: `pymilvus`（**新增，当前未安装**）+ `numpy`（已有）。MUST NOT 引入 `FlagEmbedding` / `torch` / `transformers` —— 本步骤不加载任何模型。

**Storage**: Milvus v2.6.9 standalone（Docker `milvusdb/milvus:v2.6.9`，实测 `docker ps`），`http://localhost:19530`，**无认证**，当前 collection 列表为空。

**Testing**: 无测试框架（沿用本项目既有约定：`docs/superpowers/plans` 的 Global Constraints 禁止引入 pytest，以**逐项对拍 + 边界构造**做验证）。验证输出落在 `.smoke_out/`。

**Target Platform**: Windows 11，单机 Docker Milvus。

**Project Type**: CLI 单入口脚本（离线数据管线的一步），非服务。

**Performance Goals**: 65 个 chunk 的完整入库（含建表、建索引、load、flush、校验）在 **60 秒内**完成。这是运维步骤，无并发展开需求。

**Constraints**:

- 入口文件 **≤ 300 行**（用户要求）；超出拆入 `backend/index/` 子模块，**对外入口只有一个**。
- **脚本交付后不自行运行**（用户要求），须给出可复制的运行命令。
- 不新增依赖目录、不新建 venv（宪法原则 I）；依赖须先经人工确认再装（宪法「Development Workflow」）。
- 索引 **FLAT + COSINE**，维度 1024（`docs/04 §9.1`，V1 锁定）。

**Scale/Scope**: 当前 1 份文档 / 65 个 chunk / 1 个 collection。设计上支持多文档（位置参数留空 = 全部，D3）。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 / 约束 | 门禁问题 | 结论 |
|---|---|---|
| **I. 环境锁定与依赖治理**（NON-NEGOTIABLE） | 是否只用 `rag/python.exe`？新依赖是否只进 `rag/` 并回写 `requirements.txt`？ | ✅ 通过。计划中所有命令写绝对路径；`pymilvus` 的安装命令交人工执行后回写 `requirements.txt`；不新建 venv、不动系统 Python。 |
| **II. 无据不答与强制溯源引用**（NON-NEGOTIABLE） | 是否可能让"库里数据不可溯源"？ | ✅ 通过，且**本步骤是该原则的数据前提**。强制校验 `page_start`/`page_end` 存在且有序（FR-007）、`text` 存原文而非 `text_for_embedding`（FR-010）、行对齐逐条校验（FR-006）。页码或对齐任一错误，运行时的引用卡片就是编造。 |
| **III. 密钥零硬编码**（NON-NEGOTIABLE） | 凭据是否只走环境变量？日志是否脱敏？ | ✅ 通过。地址走参数（默认 `http://localhost:19530`）；凭据只从 `MILVUS_TOKEN` 读，未设置即按无凭据连接（实测该实例无认证）；错误信息只报变量名不回显值。 |
| **IV. 紧急症状前置响应**（NON-NEGOTIABLE） | 是否产生面向用户的回答文本？ | ➖ 不适用。 |
| **V. 面向群众的医疗安全边界**（NON-NEGOTIABLE） | 是否产生诊断/处方/剂量？ | ➖ 不适用。 |
| **附加约束 · 代码规范** | `snake_case`？类型注解？是否吞掉失败条件？ | ✅ 通过。FR-028 强制；FR-024 禁止 `try/except` 吞掉校验失败、连接失败、行数不符。补偿式回滚的 `except` MUST 重新抛出并落盘证据，不得静默。 |
| **附加约束 · 知识与数据边界** | 是否新增语料？检索参数是否被记录且不依赖库默认值？ | ✅ 通过。不新增语料，只索引 S1 已准入的那一份 PDF。`top_k`/阈值不参与 hash（FR-019），写入 manifest 的 `app_config` 段（记录用途）。取值见「待你确认」第 2 项。 |
| **开发工作流 · 先规格后实现** | 是否经过 specify→plan→tasks？ | ✅ 通过。本文件为 plan 产物，之后经 `/speckit-tasks` 才进入实现。 |
| **开发工作流 · 依赖安装需人工确认** | 是否私自安装？ | ✅ 通过。只给出命令，由用户执行。 |

**门禁结论：无违规项。** Complexity Tracking 为空（无需要辩护的复杂度）。

## Project Structure

### Documentation (this feature)

```text
specs/005-milvus-ingest/
├── spec.md              # 规格（含 D1–D4 裁决）
├── plan.md              # 本文件
├── research.md          # Phase 0：技术事实与决策
├── data-model.md        # Phase 1：字段、校验规则、状态机
├── quickstart.md        # Phase 1：可跑通的验证场景
├── contracts/
│   └── cli.md           # Phase 1：命令行契约 + 退出码 + index_manifest 结构
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2（/speckit-tasks 产出，不由本命令创建）
```

### Source Code (repository root)

沿用 S2–S5 的既有模式（薄入口 + 同名子包，见 `backend/embed_chunks.py` + `backend/embed/`）：

```text
backend/
├── index_milvus.py           # ★ 唯一入口：argparse / 编排 / 报告 / 退出码  (≤300 行)
└── index/
    ├── __init__.py           # 常量：COLLECTION / DIM / 各字段 max_length / 退出码 / IndexError
    ├── inputs.py             # 读四份产物 + E1–E8 校验（写操作之前全部完成）
    ├── versions.py           # pipeline_config_hash 的计算、比对与差异定位（D2）
    ├── store.py              # MilvusClient 封装：连接 / 建表 / 门禁 / 原子写入 / 补偿回滚 / 计数
    └── manifest.py           # index_manifest.json 的读-按 doc_id 合并-写（D4）

data/
├── index_manifest.json       # 产出物（§9.3）
└── index_backup/
    └── {doc_id}.rollback.jsonl   # 仅在需要删除旧数据时写；回滚失败时作为人工恢复的唯一依据
```

**Structure Decision**: 单项目布局，不新增顶层目录。入口命名 `backend/index_milvus.py` 对齐 `docs/05 §5` 的子命令名 `index`，并与既有四个入口（`parse_pdf.py` / `clean_parsed.py` / `chunk_clean.py` / `embed_chunks.py`）的命名风格一致。

**模块划分的唯一原则**：**按"是否会写库"切分**。`inputs.py` / `versions.py` / `manifest.py` 是纯函数式的读写与计算，可独立验证；只有 `store.py` 持有 Milvus 连接并执行写入。这让"写操作之前完成全部校验"（FR-005）在结构上可见，而不是靠代码顺序的自觉。

## Complexity Tracking

无。Constitution Check 无违规项。

## 已确定与仍待确认的事项

### 已确定 1：`max_length` — **`text` = 16384**（人工裁决于 2026-09-27）

`max_length` **按 UTF-8 字节计**（Milvus 语义，见 `research.md` R2），上限 65535。取值与实测余量：

| 字段 | 取值 | 实测最大 | 字节余量 |
|---|---|---|---|
| `text` | **16384** | 7,794 | 2.1× |
| `section` | 512 | 110 | 4.7× |
| `file_name` | 512 | 50 | 10× |
| `chunk_id` / `doc_id` / `pipeline_config_hash` | 64 | 17 / 12 / 64 | 有余 |
| `source_hash` | 64 | 64 | 恰好（固定 64 位十六进制） |
| `block_type` | 32 | 13 | 2.5× |

**注意余量的换算**：本语料的字节/字符比实测为 **1.9**（表格含 ASCII 表线与英文药名），不是纯中文的 3。所以"2.1 倍字节余量"折算成**字符数**只有 1.33×（纯中文最坏情况）～ 2.1×（与当前字符构成相同）。**16384 是"够用"而非"宽裕"，这个认识 MUST 随代码一起交付**，以便将来换语料撞限时能立即判断是该调 `max_length` 还是该查 chunk 异常。

**为什么不能更小**：`docs/04 §7` 规定表格整表成块、不参与合并，因此单个块的**长度没有可推导的上界**——它完全取决于语料里最大的那张表。取"贴近实测值"（如 8192，仅 1.05 倍余量）等于下一份文档就可能撞限。

**撞限时的处置**：`release` → `alter_collection_field(max_length=新值)` → `load`，改常量后重跑。**不重建、不重灌、不重启 Milvus**（`contracts/cli.md` 的 E8 会在写入前拦下并指名是哪条 chunk）。该 `alter` 路径在 pymilvus 2.6 有文档支持，但社区有"创建后不可改"的不同说法——**实现时 MUST 先实测一次再写进交付说明**。

### 已确定 2：`app_config` 写 `{"top_k": 3, "similarity_threshold": 0.6}`，**且不触碰宪法**（人工裁决于 2026-09-27）

- **阈值取 0.6**（人工给定）。它比 `docs/04 §9.3` 示例的 0.45 **更严格、更容易拒答**，方向在医疗场景下是保守安全的（`docs/01_需求分析.md:277`：拒答率偏高时第一优先级是补 BM25，不是放宽阈值）。
- **只写进 `index_manifest.json`，不改 `.specify/memory/constitution.md`。** 两条理由：
  1. 该项目文档要求该阈值必须经**标定**产生 —— `docs/02_架构图.md:403` 给出方法（"构造 30 条语料内问题 + 30 条语料外问题，统计两类得分分布，取分离点"），`docs/03_技术选型说明.md:481` 列为验收项 V5，`docs/02:551` 列为 M3 阻塞验收项。**0.6 未走该流程，因此它是"当前取值"，不是"标定结论"。**
  2. 宪法中的 `TODO(SIMILARITY_THRESHOLD)` 位于**第 1–42 行的注释块**（Sync Impact Report）内，其定义不在正文；正文第 66 行与第 172 行引用该名（第 172 行要求 `TODO(...)` MUST 回填）。将来标定出结果时，改的是变更日志，应走 §Governance 的"提出修订 → 说明动机与影响面 → 递增版本 → 写 Sync Impact Report"，**不应由入库脚本顺手代改**。
- `app_config` **不参与 `pipeline_config_hash`**（§10.2 末段），且属查询期参数，改它**无需重建索引**（§10.3 末行）。

### 已确定 3：实现层面的 10 项自定取值 — **全部按原样采纳**（人工确认于 2026-09-27）

以下为规划阶段自行选取、经人工逐条过目后**全部采纳**的项：

| # | 自定内容 | 备选 |
|---|---|---|
| 1 | 入口名 `backend/index_milvus.py` + `backend/index/` 下 5 个模块 | 别的命名 / 单文件 |
| 2 | 回滚备份落盘 `data/index_backup/{doc_id}.rollback.jsonl` | 别的路径 / 不落盘（则回滚失败时无法人工恢复） |
| 3 | 用 `MilvusClient` 而非 ORM `Collection` | — |
| 4 | `enable_dynamic_field = False` | `True`（允许动态字段） |
| 5 | 报告分六段 + 汇总（格式见 `contracts/cli.md §4`） | JSON 输出 |
| 6 | 新增退出码 `4`（门禁拒绝）/ `5`（回滚失败，需人工介入） | 并入既有 `2`/`3` |
| 7 | 多文档时退出码取最严重的一份 | 按首个失败退出 |
| 8 | 校验项 V1–V12 清单（见 `data-model.md §4`） | 增减项 |
| 9 | 忽略 `{doc_id}.npy.tmp.npy`（S5 遗留中间产物） | 报错要求先清理 |
| 10 | 位置参数留空 = 全部文档（已由 D3 裁决为 A，此处仅登记） | — |

---

**规划阶段结论**：技术上下文无 NEEDS CLARIFICATION 残留，Constitution Check 无违规。索引选型已具论证（见上）。三项已全部裁决：`max_length`、`app_config`、上表 10 项。**可以进入 `/speckit-tasks`。**
