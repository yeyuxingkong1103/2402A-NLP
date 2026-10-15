# Phase 1 Data Model: 入库（S6 步骤）

**Feature**: `005-milvus-ingest` | **Date**: 2026-09-27

---

## 1. Collection：`med_rag_v1`

来源 `docs/04 §9.1`。维度、度量、索引类型为 V1 锁定值。

| 字段 | Milvus 类型 | 约束 | 来源 | 说明 |
|---|---|---|---|---|
| `chunk_id` | `VARCHAR(64)` | **主键**，`auto_id=False` | `chunks.jsonl` | 幂等键（块级） |
| `vector` | `FLOAT_VECTOR(1024)` | 度量 **COSINE** | `.npy` 第 `row_index` 行 | FLAT 索引 |
| `text` | `VARCHAR(16384)` | 非空 | `chunks.jsonl`.`text` | **展示用原文**，非 `text_for_embedding` |
| `doc_id` | `VARCHAR(64)` | | `chunks.jsonl`.`doc_id` | **删除键与幂等键**（D1） |
| `file_name` | `VARCHAR(512)` | | 同上 | 引用卡片 |
| `page_start` | `INT32` | 必填 | 同上 | 引用卡片 |
| `page_end` | `INT32` | `≥ page_start` | 同上 | 跨页合并时与 `page_start` 不同 |
| `section` | `VARCHAR(512)` | 可为空串 | 同上 | 引用卡片显示章节 |
| `block_type` | `VARCHAR(32)` | 四类之一 | 同上 | 前端差异化渲染 |
| `source_hash` | `VARCHAR(64)` | | 同上 | **上游产物标识，不用于幂等**（D1） |
| `pipeline_config_hash` | `VARCHAR(64)` | 逐行同值 | 本步骤计算 | 版本门禁载体（§10.2） |
| **`chunk_meta`** | **`JSON`** | ≤ 65536 字节；键名 `[A-Za-z0-9_]+` | `chunks.jsonl` 的**剩余字段** | 见 §1.1 |

`enable_dynamic_field = False` —— 未声明的字段**报错**而不是被静默收进动态字段。这是刻意的：schema 是 `docs/04 §9.1` 的契约，多一个字段必须是显式决定。

**`max_length` 一律按 UTF-8 字节计**（R2），上限 65535。取值见 `plan.md`「已确定 1」。

**索引**：`FLAT` + `COSINE`，在 `create_collection` 时与 schema 一并给出（R3）。

### 1.1 `chunk_meta`（JSON 字段）

**内容规则**：`{k: chunk[k] for k in sorted(chunk) if k not in SCALAR_FIELDS}` ——
即「**S4 整条记录减去已单独成列的字段**」。**派生而非手写白名单**：S4 新增字段会自动进来，
S6 不用改、不用重建 collection。键按字典序，保证同一份产物每次生成的 JSON 完全一致。

实测（`d6da41b5d356`，65 条）装的是这 6 个键：

| 键 | 类型 | 为什么必须存 |
|---|---|---|
| `text_for_embedding` | str | **真正喂给 BGE-M3 的文本**。`text` 列是不含章节路标的展示原文，两者不同 |
| `heading_path` | list[str] | `text_for_embedding` 的路标部分。单独存才能看出结构，不必反拆字符串 |
| `block_ids` | list[str] | 溯回 S3 的块，可逐字核对 |
| `sub_index` | int | 同一 `block_id` 被切分时的子块序号 |
| `char_len` | int | 非空白字符数（分块参数判据） |
| `chunk_rule_version` | str | 产出该 chunk 的分块规则版本 |

**为什么用 JSON 而不是逐个加列**：Milvus **没有数组类型**，`heading_path` 与 `block_ids`
无论如何都得自己 `json.dumps` / `json.loads`；既然反正要编解码，整条塞进一个字段比加
4 个 VARCHAR + 1 个 INT32 更省事，且对未来免疫。pymilvus 读 JSON 字段会**自动还原成 dict**，
读取方不用手写解码。

**两个约束**（`inputs.py` 的 V12 在写库前拦）：

1. Milvus **JSON 键只允许字母、数字、下划线**。
2. **单个 JSON 字段上限 65536 字节**（实测最大约 8 KB，8 倍余量）。

**已知坑**：pymilvus 的 JSON 值若含 **numpy 类型**会插入失败，且报错信息指向错误
（[#2886](https://github.com/milvus-io/pymilvus/issues/2886)）。本管线取值自 `json.load`，
全是原生类型；将来若注入向量运算结果，须先转原生类型。

---

## 2. 输入产物（只读）

### 2.1 `data/embeddings/{doc_id}.npy`

| 项 | 约束 |
|---|---|
| dtype | `float32` |
| shape | `(N, 1024)`，N ≥ 1 |
| 行模长 | 每行 `‖v‖₂ == 1.0 ± 1e-5`（S5 承诺 L2 归一化；COSINE 下非归一化向量会让打分失真） |
| 与 rows 的关系 | `N == len(rows)` |
| 忽略项 | 同目录的 `{doc_id}.npy.tmp.npy`（S5 遗留中间产物，E9） |

### 2.2 `data/embeddings/{doc_id}.rows.jsonl`

| 字段 | 用途 |
|---|---|
| `row_index` | **`.npy` 的行号**。行对齐的唯一依据 |
| `chunk_id` | 必须等于 `chunks[row_index].chunk_id` |
| `char_len` | 仅用于报告 |
| `vector_norm` | 仅用于报告（实测值，非硬写 1.0） |

### 2.3 `data/chunks/{doc_id}.chunks.jsonl`

**全部 15 个字段都会被消费**：其中 9 个单独成列（§1 表的"来源"列），`chunk_rule_version` 额外作为 `pipeline_config_hash` 的输入，**其余全部进 `chunk_meta`**（§1.1）。

> 原本的设计是「`text_for_embedding` 不落库，检索用的文本由运行时按同一口径现算」，依据是
> `docs/04 §9.1` 的"存展示原文"与 `specs/003`/`specs/004` 的"口径只有一处定义"。
> **该设计在 2026-09-27 由用户指示修正**：只存展示原文会导致"这条 1024 维向量到底是从哪串
> 文本算出来的"无法核对——重拼口径一旦与 S5 不一致，**你不会知道**。这正是 `docs/04 §8`
> 要防的静默漂移。现改为把 S4 记录整体留存于 `chunk_meta`，让向量可复现、可审计。
>
> 注意这**不改变**检索路径：查询时用户问题仍需按同一口径编码，`chunk_meta` 是**留档**，
> 不是查询时的替代品。

### 2.4 `data/embeddings/{doc_id}.fingerprint.json`

整份纳入 `pipeline_config_hash`（R5）。实测 10 个字段：`model_dir`、`config_sha256`、`weight_file`、`weight_bytes`、`max_length`、`pooling`、`normalization`、`dtype`、`dim`、`rule_version`。

---

## 3. 产出物：`data/index_manifest.json`

来源 `docs/04 §9.3`。**读-按 `doc_id` 合并-写**（D4 / R8）。

```json
{
  "built_at": "<ISO-8601 带时区，本次运行时刻>",
  "collection": "med_rag_v1",
  "metric_type": "COSINE",
  "index_type": "FLAT",
  "dim": 1024,
  "total_chunks": 65,
  "documents": [
    {
      "doc_id": "d6da41b5d356",
      "file_name": "国家基层高血压防治管理指南2025版.pdf",
      "source_hash": "39ced98cf4e59a0e4a068d1d7aba69f01ec2d209a939c2ee02f48de3def79bb8",
      "chunk_count": 65,
      "page_count": 15
    }
  ],
  "pipeline_config_hash": "<sha256 hex>",
  "pipeline_config": { "…见 R5 的输入 dict…" },
  "app_config": { "top_k": 3, "similarity_threshold": 0.45 }
}
```

**字段规则**：

| 字段 | 规则 |
|---|---|
| `built_at` | 本次运行时刻（ISO-8601，带时区）。重复入库时**刷新** |
| `total_chunks` | `documents[].chunk_count` 之和。**写前向库核实**（R8） |
| `documents[]` | 按 `doc_id` 去重；计数为 0 的条目**移除** |
| `page_count` | 该文档 chunk 的 `max(page_end)`。实测 15，与 PDF 页数一致 |
| `pipeline_config_hash` | R5 计算值 |
| `pipeline_config` | R5 的**输入 dict 本身**（不只哈希）—— 供门禁失败时逐项定位差异（FR-019a） |
| `app_config` | **不参与 hash**（§10.2 末段）。取值见 `plan.md`「待你确认 2」 |

**注**：S4 的分块规则版本串是 `chunk-v1+bge-m3`（含模型名），因为 D2 裁决让分块依赖 BGE-M3 的相似度判定。这个命名不是笔误，MUST 原样保留。

---

## 4. 校验规则（写操作之前必须全部通过）

对应 spec 的 E1–E8。任一条不通过 → **退出码 2，且不发生任何写操作**。

| # | 规则 | 判据 | 为什么必须拦下 |
|---|---|---|---|
| V1 | npy 行数 == rows 行数 | `shape[0] == len(rows)` | 少了/多了向量 |
| V2 | **逐条**行对齐 | `rows[i].chunk_id == chunks[i].chunk_id`，**i 从 0 递增逐条比** | **最危险**：集合相等但顺序错位时，向量会配上错误的文本，检索命中后引用指向别的段落 |
| V3 | `chunk_id` 无重复 | `len(set) == len` | 主键冲突 / 静默覆盖 |
| V4 | `row_index` 连续且与行号一致 | `rows[i].row_index == i` | 索引取值的依据 |
| V5 | 每行 L2 模长 ≈ 1 | `abs(‖v‖-1) ≤ 1e-5` | COSINE 度量下非归一化向量让打分失真 |
| V6 | 页码完整有序 | `page_start ≥ 1` 且 `page_end ≥ page_start` | **引用卡片的全部依据**；页码错 = 编造出处 |
| V7 | `source_hash` 全行同值 | `len(set) == 1` | 一份文档的 chunk 必须同属一份源文件 |
| V8 | `chunk_rule_version` 全行同值 | 同上 | 拼凑的产物 |
| V9 | `text` 非空非全空白 | `text.strip()` 非空 | 空文本入库后检索无意义 |
| V10 | 字段字节长度 ≤ `max_length` | `len(s.encode("utf-8"))`（**字节**，R2） | 报出是哪条 `chunk_id` 的哪个字段，而非让 Milvus 在插入时抛一句笼统的错 |
| V11 | `.npy` 可读且 shape[1] == 1024 | — | 维度不符会让整个 schema 失效 |
| V12 | `chunk_meta` 三查：必需键齐备 / 键名合法 / 序列化 ≤ 65536 字节 | 见 `data-model.md §1.1` | Milvus 对 JSON 键的字符集与大小的硬约束；在写库前拦住才能**指名是哪条 chunk** |

V1–V9、V11、V12 在 `inputs.py`；V10 用 `__init__.py` 的 `MAX_LENGTHS` 表。

---

## 5. 状态机：单份文档的写入

`pending` → 全部校验通过后进入 `writing` → 依 `research.md R4` 的六步。

```text
  [校验]  V1–V12 全通过
     │  任一失败 ──────────────────────────→ 退出码 2（库未被触碰）
     ▼
  [门禁]  库内 pipeline_config_hash vs 本次
     │  不一致 ────────────────────────────→ 退出码 4（库未被触碰）
     ▼
  [取旧]  N_old = count(doc_id)
     │    N_old > 0 → 落盘 data/index_backup/{doc_id}.rollback.jsonl
     ▼
  [删除]  delete(filter="doc_id == '<id>'")
     ▼
  [插入]  insert(rows) → insert_count
     ▼
  [校验]  flush；Strong 一致性 count(doc_id) == N_expected ?
     ├─ 是 ──────────────────────────────→ committed
     └─ 否 → [补偿] delete(doc_id) → 回填旧行 → 复验 == N_old ?
                ├─ 是 ────────────────────→ 退出码 2（"已回滚，库回到运行前"）
                └─ 否 ────────────────────→ 退出码 5（库状态不确定；备份文件路径已给出）
```

**不变量**：

- `[删除]` 之后若不进入 `committed`，则库 MUST 回到 `[取旧]` 时的状态——或**明确报告回不去**（退出码 5），MUST NOT 沉默。
- 多份文档时**每份独立走这套状态机**：一份失败不影响其它份的既成结果，但整体 MUST 非 0 退出并**逐份**报出结果。

---

## 6. 关系

```text
data/source_data/*.pdf
   │ sha256 → doc_id (前12位)                    ← 删除键与幂等键（D1）
   │
   ├─S2→ data/parsed/{doc_id}/…_origin.pdf
   │        │ sha256 → source_hash                ← 上游标识，仅落库（D1）
   │        │
   │        ├─S3→ data/clean/{doc_id}.blocks.jsonl
   │        └─S4→ data/chunks/{doc_id}.chunks.jsonl ──┐
   │                                                  │
   └─S5→ data/embeddings/{doc_id}.npy  ───────────────┤
         data/embeddings/{doc_id}.rows.jsonl ─────────┤ row_index ↔ chunk_id
         data/embeddings/{doc_id}.fingerprint.json ───┤
                                                       │
                                        S6（本特性）───┤
                                                       ▼
                              Milvus collection med_rag_v1 (65 rows)
                                                       │
                                    data/index_manifest.json（身份证，D4）
```

运行时检索接口（`docs/05 §4`）消费 collection 与 manifest；本特性不实现它。
