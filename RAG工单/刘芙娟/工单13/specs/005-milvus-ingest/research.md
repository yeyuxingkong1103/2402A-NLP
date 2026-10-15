# Phase 0 Research: 入库（S6 步骤）

**Feature**: `005-milvus-ingest` | **Date**: 2026-09-27

本文件解决规划阶段的技术未知。每条给出**决定 / 理由 / 备选**。所有"实测"均为本次会话在
`D:/zg6_Project/9/med_rag` 上实际执行的结果，不是推断。

---

## R0 — 索引选型：`FLAT` + `COSINE`（论证见 `plan.md`）

**决定**：`FLAT` + `COSINE` + 1024 维。**这不是本计划的决定** —— 由 `docs/04:527`（"索引配置（V1 锁定）"）、`docs/04:517`、`docs/02_架构图.md` §7 参数表、`docs/05:136` 四处锁定。本计划补的是**论证与换索引判据**，完整论证写在 **`plan.md` 的「索引选型：为什么是 `FLAT` + `COSINE`」一节**（含候选范围按度量筛选、近似索引在 65 条规模上是净亏、召回损失会导致"调错旋钮"的失效路径、FLAT 零可调参数与门禁的关系、AUTOINDEX 为何不适用、换索引的规模门槛）。

**理由摘要**：65 条 × 1024 维下 FLAT 每次查询约 6.7 万次乘加（几十微秒），而单次检索耗时由**网络往返 + 返回 payload**（`text` 最长 7,794 字节 × `top_k=3`）支配 —— 近似索引**省不下 1 毫秒**，却必然收走召回率；召回损失在本系统里的表现是"原文有答案却拒答"，且 `docs/01:277` 已预言它会被误诊为阈值问题。

**备选与否决理由**：见 `plan.md` 同节的逐项对照表（`HNSW` / `IVF_FLAT` / `IVF_SQ8` / `IVF_PQ` / `DiskANN` / `AUTOINDEX`）。`SCANN` 与 `HNSW_SQ/PQ/PRQ` 的度量是内积，**在按 COSINE 筛选这一步即被排除**。

---

## R1 — Milvus 服务端版本与客户端选型

**决定**：服务端 `milvusdb/milvus:v2.6.9`（Docker `milvus-standalone`）；客户端用 `pymilvus` 的 **`MilvusClient`** 高层 API，不使用 ORM 风格的 `Collection`。

**理由**：

- 版本由实测 `docker ps` 得到：`milvusdb/milvus:v2.6.9`，端口 `19530`（gRPC/REST）与 `9091`（metrics）已映射。
- `MilvusClient` 的 `create_collection(schema=…, index_params=…)` 能**一次性完成建表 + 建索引 + load**，正好绕开 R3 里那个"必须先在空集合上建索引"的强制顺序。ORM 的 `Collection` 需要手写 `create_index` + `load` 三步，且在 FLAT 上有已知的 `index not found` 报错路径。
- 无需 `connections.connect()` 的全局连接状态，便于脚本化。

**备选与否决理由**：

- *ORM `Collection`*：多三步 API、全局连接状态、FLAT 上的已知报错路径。无一利。
- *Milvus 的 REST v2 接口*（`curl` 已验证可达）：需手工拼 JSON、错误处理更笨重、Python 侧无类型。仅适合健康检查。
- *Milvus Lite（本地文件）*：与用户给定的 Docker 部署不符。

**待实现时确认**：`pymilvus` 的版本须与 2.6.9 服务端兼容（2.6.x）。安装命令里不锁小版本，装完由 `--print-env` 报告实测版本，与 R1 的期望值比对。

---

## R2 — VARCHAR `max_length` 按**字节**计（不是字符）

**决定**：所有 VARCHAR 字段的 `max_length` 一律**按 UTF-8 字节数**计算余量，绝不按字符数。取值已裁决为 `text=16384`（人工裁决 2026-09-27），并附**字节余量折算成字符余量只有 1.33×～2.1×** 的说明 —— 见 `plan.md` 的「已确定 1」。

**理由**：

- 实测服务端 API 文档与多个 SDK 参考一致：`max_length` 是"max byte length"，上限 **65535**（字节）。CJK 字符 3 字节/字。
- 本语料的实测差异极大：最长 `text` 是 **4,094 字符 / 7,794 字节**（差 1.9 倍）。若按字符数算余量，一个"看起来 4000 字符、留 4096 余量"的配置会在**写入时**被 Milvus 拒绝，且错误信息只报字段不报是哪条 chunk。

**备选与否决理由**：

- *按字符数估算*：直接错，见上。**这是本步骤最容易踩的坑。**
- *一律取上限 65535*：正确但每行缓冲区按上限分配；且它会让"超长"这件事**永不报错**，与 §9.2 的"禁止部分成功"精神相反——宁可让换语料时撞限并被拦下，也不要让它悄悄通过（虽然 65535 下不会截断，但会掩盖"某条 chunk 异常巨大"这个信号）。

**由此派生的实现要求**：`inputs.py` 的 E8 校验 MUST 用 `len(s.encode("utf-8"))` 判长度，并在超长时报出**是哪条 `chunk_id` 的哪个字段、实际多少字节、上限多少**。

---

## R3 — 索引必须在空集合上先建（`docs/04 §9.2` 的顺序在 Milvus 上不可实现）

**决定**：`create_collection(schema=…, index_params=FLAT/COSINE)` —— 建表的同时建索引并 load，**此时集合是空的**。之后才做"取旧 → 删 → 插 → 校验"。

**理由**：

1. `delete(filter=…)` 与 `query(filter=…)` 要求 collection **已 load**（Milvus 的明确要求，未 load 会报错）。
2. `load_collection` 要求**向量字段已有索引**（"an index must be created on vector fields before a collection can be loaded"）。
3. 两条合起来的必然顺序是：**建索引 → load → 才能 delete/query**。

而 `docs/04 §9.2` 写的是"索引构建：写入完成后统一建索引，不在写入过程中建"，理由是"避免索引在中途状态被查询使用"。该要求在 Milvus 上无法按字面实现——不建索引就无法 load，不 load 就无法 delete/query，整个流程走不动。

**按意图处置**：§9.2 的担心是"索引看到写入的中途状态"。本方案里索引建在**空集合**上（零条数据），之后的所有写入都发生在索引已就绪之后，**"中途状态"从未被索引看到**；且 FLAT 是暴力检索、无训练阶段，不存在"训练集只覆盖了一半数据"的问题。**意图满足，字面顺序被平台强制改写。**

**这一取舍 MUST 记入交付说明**，因为它是对既有设计文档字面条款的偏离。

**备选与否决理由**：

- *建表不建索引 → load → 报错*：走不通（第 2 条）。
- *建表 → 建索引 → load → delete/query/insert*：这就是本方案，只是把"建索引"和"建表"合并成一次调用。
- *每轮插入后 drop 索引再重建*：破坏性、无收益、且 drop 索引期间 load 状态失效，反而制造出 §9.2 想避免的中途状态。

---

## R4 — 原子性：Milvus 2.6 无客户端可见事务，只能补偿

**决定**：「**先取旧 → 删 → 插 → 校验 → 失败则用旧数据回滚**」的补偿式提交，配合落盘备份。

**理由**：

- Milvus 的 WAL 内部确有事务消息（`BeginTxn`/`CommitTxn`/`RollbackTxn`），但那是**服务端内部机制**，**不暴露给客户端**。客户端可见的保证是："单文档操作原子；**批量操作部分失败时已成功的不会回滚**；不支持跨 collection 事务"。
- 而 `docs/04 §9.2` 的「禁止部分成功」要求的是**跨"删除 + 插入"两个操作**的原子性。这在客户端只能靠补偿实现。

**具体流程**（每份文档）：

```text
1. 写前计数 N_old = count(doc_id)
2. 若 N_old > 0：查回该 doc_id 的全部行 → 落盘 data/index_backup/{doc_id}.rollback.jsonl
3. delete(filter="doc_id == '<id>'")
4. insert(rows)   → 得到 insert_count
5. flush；以 Strong 一致性 count(doc_id) = N_new
6. 若 N_new != N_expected：
     a. delete(filter="doc_id == '<id>'")          # 清掉可能进去的半份
     b. 若 N_old > 0：insert(备份的旧行)            # 还原
     c. 复验 count(doc_id) == N_old；失败则退出码 5 并指明备份文件路径
     d. 退出码 2，报告"已回滚"
```

**关键点**：第 2 步的备份是回滚能力的**唯一来源**。没有它，"删了旧的、没进去新的"就无法挽救。备份文件同时是回滚本身也失败时**人工恢复的唯一依据**。

**备选与否决理由**：

- *先插后删*：`chunk_id` 是主键，重复主键插入会失败或产生重复行，走不通。
- *insert 到临时 collection 再换名*：Milvus 无 collection 重命名；且跨 collection 无事务，问题只是被搬了个地方。
- *依赖 `upsert`*：`upsert` 按主键覆盖，能解决"同 `chunk_id` 更新"，但**解决不了"新版本少了一些 chunk"**——那些消失的 `chunk_id` 不会被 upsert 删掉，会留下陈旧数据。与 D1 选 `doc_id` 作删除键的理由同源。
- *不做补偿，失败就报错*：违反 §9.2 的「禁止部分成功」。

**由 R4 派生的退出码**：`5` = 写入校验失败且回滚也失败（库状态不确定，**需人工介入**）。`0`/`1`/`2`/`3` 沿用 `docs/05 §5`；`4` 用于版本门禁拒绝（需人工决定是否重建）。详见 `contracts/cli.md`。

---

## R5 — `pipeline_config_hash` 的输入与计算（落地 D2）

**决定**：`sha256` over 一个**确定性序列化**的 dict，键按字典序、值原样：

```python
{
  "clean_rule_version": "clean-v1",              # from backend.clean.CLEAN_RULE_VERSION（import）
  "chunk_rule_version": "chunk-v1+bge-m3",       # from chunks.jsonl 逐行取值，须全行一致
  "chunk_min_chars": 300,                        # from backend.chunk.core.LOW（import）
  "chunk_max_chars": 700,                        # from backend.chunk.core.HIGH（import）
  "min_ratio": 0.6,                              # from backend.chunk.core.MIN_RATIO（import）
  "chunk_overlap": 0,                            # S4 无滑窗重叠（实测 S4 规格：无 overlap）
  "embed": { … fingerprint.json 的全部 10 个字段 … },
}
```

**理由**：

- **数值从上游代码 import，不从文档抄**。实测确认 `backend/chunk/core.py` 只 import `os` / `re` 与包内 `RULE_VERSION`，**不拉 torch/transformers**，所以 `from chunk.core import LOW, HIGH, MIN_RATIO` 是零成本的，且**彻底消除了 D2 发现的文档漂移**——S6 与 S4 共用同一个常量源，改一处两边同时变。
  - 若将来 `core.py` 引入重依赖，这个 import 会变贵但不会变错；届时应改为读取一个轻量常量模块。
- **`chunk_rule_version` 逐行取值并校验全行一致**：实测 65 行全为 `chunk-v1+bge-m3`。不一致说明产物是拼凑的，必须报错（并入 `inputs.py` 的校验）。
- **`fingerprint.json` 整份纳入**：S5 已把"影响向量数值的全部参数"（权重哈希、`max_length`、池化、归一化、dtype、维度）收进指纹。整份纳入既不漏也不挑。
- **`chunk_overlap: 0`**：S4 的分块规则是"在边界处切"，无滑窗重叠；产物里也没有 overlap 字段。写 0 是记录事实。
- **查询期参数不入 hash**：`top_k` / 相似度阈值不参与（§10.2 末段）。

**门禁读取路径**：`pipeline_config_hash` 是 schema 里的**逐行字段**（§9.1），因此无需额外的元数据 collection——从任意一行读回即可：

```python
rows = client.query(collection, filter="", output_fields=["pipeline_config_hash"], limit=1)
```

集合为空 → 无门禁（首次入库）。不一致 → 退出码 4，并从 `data/index_manifest.json` 的 `pipeline_config` 段读取旧值以**报出差异的具体参数**（US3-2）。若 manifest 缺失，则如实说明"无法逐项比对，只能报出两个哈希值"——**不假装知道差异在哪**。

**备选与否决理由**：

- *只 hash 规则版本串*：`chunk_rule_version` 不编码 `LOW`/`HIGH`（实测 `chunk-v1+bge-m3`），只改数值不改版本串时门禁静默失效。这是 D2 明确否决的方案 B。
- *把 hash 存进 collection 的描述（description）*：Milvus 的 collection description 是可变字符串、无版本语义，且改动它需要对集合加锁；逐行字段已经够用。
- *另开一个元数据 collection*：为 64 字节引入一张表、一次跨表读、一次跨表一致性问题。否决。

---

## R6 — 一致性级别与 `flush`

**决定**：写入后 `flush()`，随后所有**校验性**的 `query` 一律带 `consistency_level="Strong"`。

**理由**：Milvus 默认一致性是 Bounded（有界滞后），写入后立刻 `count(*)` 可能读到旧值——那会让 R4 的第 5 步**误判**为"行数不符"并触发一次不必要的回滚。`flush` 把数据落到段，Strong 保证读到自己写的。65 行规模下 flush 的代价可忽略。

**备选与否决理由**：*用默认一致性 + 重试*：需要自己写轮询与超时，且"读到旧值"与"真的少写了"难以区分。Strong 是这两者唯一干净的区分方式。

---

## R7 — 幂等键：用 `doc_id`（落地 D1）

**决定**：删除与幂等一律按 `doc_id` 过滤；`source_hash` 字段照实存上游值，不参与删除或幂等。

**理由**（实测支撑）：

- `doc_id` = `data/source_data/国家基层高血压防治管理指南2025版.pdf` 的 sha256 前 12 位。实测：该 PDF 的 sha256 = `d6da41b5d3565c4969150009a6a5d00bb45de19b361cfa3ec9a0f51d25ef9bbd`，前 12 位正是 `d6da41b5d356` ✅ —— 与 `docs/04 §10.1` 的说法一致。
- `source_hash` = MinerU `_origin.pdf` 副本的 sha256。实测：`data/parsed/d6da41b5d356/…/…_origin.pdf`（1,247,676 字节）的 sha256 = `39ced98cf4e59a0e4a068d1d7aba69f01ec2d209a939c2ee02f48de3def79bb8`，与 chunks.jsonl 中 65 行完全一致 ✅ —— 而它与源 PDF 的哈希**不是同一个值**（源 PDF 1,246,571 字节）。依据 `backend/chunk/loader.py:26 resolve_source_hash`：它 walk 到 `_origin.pdf` 重算。
- 结论：**用 `source_hash` 作删除键，MinerU 重跑解析就会换键，旧数据删不掉**。

**备选与否决理由**：*新增 `pdf_sha256` 字段（原方案 C）*：语义更全，但要改 `docs/04 §9.1` 的 schema、改将来的后端、且 `doc_id` 已经是同一个哈希的前缀——多存一份 52 位十六进制换不来任何能力。否决。

---

## R8 — `index_manifest.json` 以**库的实际内容**为准重建 `documents`（落地 D4 / FR-022a）

**决定**：写入成功后，收集 `doc_id` 全集 = （已有 manifest 的 `documents[].doc_id`）∪（本次处理的 `doc_id`），对每个 `doc_id` 用 `count(*)` 向库核实：计数为 0 的条目**从 manifest 中移除**，计数与预期不符的**如实写入实际值**并计入报告。

**理由**：

- §9.3 把该文件定义为"索引的身份证"与后端启动校验的对象。一份与库不符的身份证比没有身份证更糟。
- 用逐 `doc_id` 的 `count(*)` 而不是 `group_by_fields`：后者在 `query` 上的支持面较窄，且这里文档数量是**个位数**，逐条查询的代价可忽略，换来的是不依赖较新 API。

**备选与否决理由**：*直接用已有 manifest 覆盖写*：会把"库里有、manifest 没有"的文档永久藏起来。*用 `group_by`*：见上。

---

## R9 — 入口与模块划分

**决定**：`backend/index_milvus.py`（唯一入口，≤300 行）+ `backend/index/`（5 个模块）。

**理由**：与 `backend/embed_chunks.py` + `backend/embed/` 同构（既有模式，见 `specs/004`）；入口名对齐 `docs/05 §5` 的子命令 `index`。模块**按"是否会写库"切分**，使 FR-005「一切校验先于任何写操作」在结构上可见。详见 `plan.md` 的 Structure Decision。

**备选与否决理由**：*单文件*：300 行装不下四份产物的校验 + 补偿式回滚 + 门禁 + manifest 合并。*建 `backend/pipeline` 调度器*（`docs/05 §5` 描述的那个）：该调度器不存在，S2–S5 都各有独立入口；在本特性里造它属于范围蔓延，已记入 spec 的「未纳入本特性的已知文档漂移」。

---

## R10 — 依赖

**决定**：唯一新依赖 `pymilvus`。安装命令由交付说明给出，**由用户执行**。

```text
D:/zg6_Project/9/med_rag/rag/python.exe -m pip install pymilvus
D:/zg6_Project/9/med_rag/rag/python.exe -c "import pymilvus; print(pymilvus.__version__)"
```

装完后回写 `requirements.txt`（宪法原则 I）。`numpy` 已装（S5 在用）。

**理由**：宪法「Development Workflow」明确"涉及依赖安装的动作 MUST 先获得人工确认"。实测当前 `pymilvus` 未安装（`ModuleNotFoundError`）。
