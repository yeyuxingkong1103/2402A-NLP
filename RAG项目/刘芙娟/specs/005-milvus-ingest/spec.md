# Feature Specification: 入库（S6 步骤）

**Feature Branch**: `005-milvus-ingest`

**Created**: 2026-09-27

**Status**: Draft

**Input**: User description: "生成入库脚本，保存到 `d:\zg6_Project\9\med_rag\backend`，脚本不超过 300 行，如果脚本太长就分模块写，但是脚本的入口只设计一个。入库方式选用 Milvus，Docker 里已部署好且已打开，连接地址为 `http://localhost:19530`。脚本设计好之后先不要运行，告诉我运行方式让我自己运行。如果有不明确的情况，告诉我并推荐 2–3 个解决方案、说明推荐哪个，让我做决策，严禁私自做决策。"

## 背景与定位

本特性实现 `docs/04_数据管线设计.md` §9 的 **S6 入库**，是 `specs/004-bge-embedding`（S5 向量化）的下游，也是整条离线管线的**最后一步**：把磁盘上的三份产物变成 Milvus 里一个可被检索的 collection，供运行时检索接口（`docs/05 §4`）消费。

上游三份产物：

| 产物 | 内容 | 本步骤怎么用 |
|---|---|---|
| `data/embeddings/{doc_id}.npy` | float32[65, 1024]，逐行 L2 归一化 | `vector` 字段 |
| `data/embeddings/{doc_id}.rows.jsonl` | 65 行，`row_index → chunk_id` 对齐表 | **行对齐的唯一依据** |
| `data/chunks/{doc_id}.chunks.jsonl` | 65 行，chunk 全量元数据 | `text` 及各标量字段 |
| `data/embeddings/{doc_id}.fingerprint.json` | 模型指纹 | `pipeline_config_hash` 的输入之一 |

**本步骤是整条管线里唯一会写坏"运行时状态"的环节**。S2–S5 只产出磁盘文件，跑错了重跑即可；S6 写的是**正在被查询的库**。因此本规格的第一议题是**"怎么保证写不坏"**——写入要么整份文档全成，要么完全不留痕（§9.2「禁止部分成功」）；参数一变就必须拒绝写入而不是静默混入（§10.2）。

---

## 实测基线

### 输入产物（实测，`doc_id = d6da41b5d356`）

| 项 | 实测值 |
|---|---|
| chunk 数 | **65** |
| 向量矩阵 | `(65, 1024)` float32，266,368 字节 |
| `text` 字节长度（UTF-8） | 中位约 600，**最大 7,794**，最小 14 |
| `text` 字符长度 | 最大 **4,094**（与 S4 报告里的 3,160「字」口径不同：S4 统计的是非空白字符） |
| `section` 最长字节 | 110 |
| `block_type` 取值 | `text` / `table` / `image` / `page_footnote` 四类 |
| `page_start` / `page_end` 范围 | 1–15，`page_end ≥ page_start` |
| `source_hash` | **全 65 行同一个值** `39ced98c…79bb8` |
| `chunk_rule_version` | 全 65 行 `chunk-v1+bge-m3` |

### Milvus 侧（实测）

| 项 | 实测值 |
|---|---|
| 连通性 | `http://localhost:19530` **可达**，`POST /v2/vectordb/collections/list` 返回 `{"code":0,"data":[]}` |
| 认证 | **无**（未启用用户名/密码，请求无 token 即通） |
| 现有 collection | **空**，`med_rag_v1` 尚不存在 —— 这是**首次入库** |
| `pymilvus` | **未安装**（`ModuleNotFoundError`） |

### 向量化是 CPU 产物，入库不涉及模型

S6 **不加载任何模型**，只搬运字节。所以本步骤的运行时间由网络往返决定，与 S2/S5 的重模型加载无关。

---

## ⚠️ 两处「文档与现实不符」（已实测确认，直接影响本步骤的正确性）

本规格在实测阶段发现两处上游文档与现实不一致。它们不是风格问题，而是**会让本步骤静默写错库**的问题，必须在实现前裁决。

### 发现的 A：`source_hash` 的真实含义与 `docs/04 §10.1` 的说法不符

`docs/04 §10.1` 写：

> **文档级** `source_hash`（PDF 内容 sha256）… **注意 `doc_id` 与 `source_hash` 的关系**：`doc_id` 是 `source_hash` 的前 12 位。

**实测结果与此矛盾**：

| 对象 | sha256 |
|---|---|
| `data/source_data/国家基层高血压防治管理指南2025版.pdf`（1,246,571 字节） | `d6da41b5d356`5c49…9bbd ← **`doc_id` 确实是它的前 12 位** ✅ |
| `data/parsed/d6da41b5d356/…/…_origin.pdf`（1,247,676 字节，MinerU 的副本） | `39ced98cf4e5`9a0e…79bb8 ← **chunks.jsonl 里 `source_hash` 的取值** |
| chunks.jsonl 中的 `doc_id` | `d6da41b5d356` |

即：**`doc_id` 是源 PDF 的哈希前缀，而 `source_hash` 字段是 MinerU `_origin.pdf` 副本的哈希**（依据 `backend/chunk/loader.py:26 resolve_source_hash`，它 walk 到 `_origin.pdf` 重算）。两者**不是同一个哈希的不同长度，而是两个不同的文件**。

**为什么这件事在 S6 会变成问题**：`docs/04 §9.2` 规定「先删同 `source_hash` 的旧数据」，§10.1 把 `source_hash` 定为文档级幂等键。若 S6 照字面用这个 `source_hash` 当删除键，则：

- MinerU 换版本重跑解析 → `_origin.pdf` 被重新生成、字节可能不同 → `source_hash` 变了 → **旧数据删不掉，新旧两份并存**。检索会同时命中同一段落的两个版本，引用页可能指向已作废的排版。
- 反之，若用 `doc_id` 当删除键（= 源 PDF 哈希前缀），则"同一份 PDF 重新走一遍管线"这件事天然幂等，符合 `docs/04 §10.1` 想表达的语义。

**裁决结果：D1（见文末「已裁决决定」）—— 删除键与幂等键用 `doc_id`，`source_hash` 字段照实存上游值。**

### 发现的 B：`docs/04 §9.3` 里 `pipeline_config` 的参数值已被 S4 裁决作废，但文档未回写

`docs/04 §9.3` 的 `index_manifest.json` 示例中写着：

```json
"chunk_min_chars": 300,
"chunk_max_chars": 500,
```

而 `specs/003-semantic-chunking` 的 **D1 裁决**（2026-09-23）已经把目标长度改为 **500–700 字**，并明确写了：

> `docs/04 §7` 原定的「300–500 字」与「跨章节合并禁止」**作废并须回写 `docs/04 §7`**。

该回写**至今未执行**（实测：`docs/04 §7` 仍写着 300–500，§9.3 示例仍是 300/500）。`docs/04 §7` 是 S4 分块的设计来源，因此**文档里的分块参数与磁盘上真实产物的分块参数不一致**。

**为什么这件事在 S6 会变成问题**：`pipeline_config_hash` 是"影响产物的全部参数的哈希"，而它的作用是**门禁**——参数变了就拒绝写入（§10.2）。若 S6 把 `docs/04 §9.3` 的示例值（300/500）当权威输入，算出的 hash 反映的是一套**从未真正用过的参数**；将来真的改了分块参数，hash 可能反而不变，门禁失效。**一个恒定的假 hash 比没有门禁更危险**，因为它会让人以为门禁在工作。

**裁决结果：D2（见文末「已裁决决定」）—— 以实际生效的代码常量为准（`LOW, HIGH = 300, 700`），并把 `docs/04 §7`、`§9.3` 回写正确。**

> 补充实测（裁决前追加）：`backend/chunk/core.py:10-13` 记载下限"由 500 降为 300"，理由是 80% 的 section 不到 500 字、而 D1 又只允许同父章节合并，两者叠加会让 62% 的 chunk 落在下限之外。因此**实际生效的是 300–700**，与 D1 字面的"500–700"、`docs/04 §7` 的"300–500"**三者互不相同**。`chunk_rule_version`（`chunk-v1+bge-m3`）**不编码 LOW/HIGH**，只改数值不改版本串时规则串拼出的 hash 不会变——这是 D2 必须把数值纳入 hash 的原因。

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 一份文档的 65 个分块完整进入检索库 (Priority: P1)

运维人员跑一条命令，把 `d6da41b5d356` 的 65 个分块及其向量写入 Milvus。跑完后，用同一个 chunk 的文本去检索，能命中它自己，且返回的 `page_start`/`section` 与 `chunks.jsonl` 中的记录逐字一致。

**Why this priority**: 这是本特性的存在理由。没有它，S2–S5 的全部产物都停在磁盘上，运行时检索接口无库可查。

**Independent Test**: 清空 collection 后跑一次，断言 collection 行数 == 65，且逐条抽查 `chunk_id`/`text`/`page_start` 与磁盘产物一致。

**Acceptance Scenarios**:

1. **Given** collection 不存在、三份产物齐备且互相一致，**When** 运行入库命令，**Then** 命令以退出码 0 结束，collection 存在且行数 **恰好等于 65**
2. **Given** 入库完成，**When** 用某个 chunk 的 `text` 与该 chunk 的向量做检索，**Then** 第一条命中的 `chunk_id` 就是它自己，`score ≥ 0.999`
3. **Given** 入库完成，**When** 按 `chunk_id` 查询任一行，**Then** `text`/`doc_id`/`file_name`/`page_start`/`page_end`/`section`/`block_type` 七项与 `chunks.jsonl` 中该 `chunk_id` 的记录**逐字相同**（`text` 是原文，不是 `text_for_embedding`）
4. **Given** 入库完成，**When** 检查 collection 的索引配置，**Then** 索引类型为 `FLAT`、度量类型为 `COSINE`、维度为 1024

---

### User Story 2 - 同一份文档重复入库，库里内容不变 (Priority: P1)

运维人员不确定上次是否跑成功，于是又跑了一遍一模一样的命令。库里仍然是 65 行，`chunk_id` 集合与第一遍完全相同，没有任何重复。

**Why this priority**: 幂等是"敢重跑"的前提。若重跑会变成 130 行，运维就只能靠人工 drop collection 来补救，而这正是 §10.2 想避免的"破坏性操作被日常化"。

**Independent Test**: 连续跑两次同样的命令，两次都以退出码 0 结束，且第二次后行数仍为 65、无重复 `chunk_id`。

**Acceptance Scenarios**:

1. **Given** 该文档已入库 65 行，**When** 用完全相同的输入再跑一次，**Then** 退出码 0，行数仍为 **65**（不是 130）
2. **Given** 重复入库后，**When** 对比两次的 `chunk_id` 集合，**Then** 两个集合**完全相同**
3. **Given** 重复入库后，**When** 检索同一个 chunk，**Then** 结果与第一次入库后**逐条相同**，不出现同一段落的两条命中

---

### User Story 3 - 参数变更时拒绝写入，而不是静默污染 (Priority: P1)

有人只改了分块参数就重跑 S4→S6。此时磁盘上的向量与库里已有的向量**不在同一个语义空间**（分块边界变了，`chunk_id` 对应的文本也变了）。脚本必须**拒绝写入并报错退出**，把"要不要重建"这个决定权交回给人。

**Why this priority**: 这是 §10.2 的核心。与 S5 的"模型漂移"同构——混用不会报错，只会让检索悄悄变差。**静默污染比拒绝服务更糟**，因为它不可发现。

**Independent Test**: 手工构造一个与库内不一致的参数版本，确认命令非 0 退出、库内容一字未改，且错误信息说明了"哪里变了"。

**Acceptance Scenarios**:

1. **Given** 库内已有该 collection 且其参数版本与本次输入不符，**When** 运行入库命令，**Then** 命令**非 0 退出**，且 collection 的行数与内容**一字未改**
2. **Given** 上述拒绝情形，**When** 读错误信息，**Then** 信息中明确给出了**两个版本的值及差异点**（哪个参数变了），而不是一句笼统的"参数不匹配"
3. **Given** 库内参数版本与本次输入相符，**When** 运行入库命令，**Then** 正常写入，退出码 0
4. **Given** 运维人员**显式**发起重建，**When** 命令带上重建确认开关，**Then** 才允许清空该文档/该 collection 后重建；**不带该开关时，脚本 MUST NOT 执行任何删除**

---

### User Story 4 - 索引的「身份证」被落盘，供后端启动校验 (Priority: P2)

入库完成后，`data/index_manifest.json` 被写出或更新，记录了本次入库的时间、collection 名、度量类型、索引类型、维度、总块数、每份文档的块数与页数，以及本次的参数版本哈希。

**Why this priority**: `docs/04 §9.3` 把它定义为 S6 的产出物、后端启动时校验的对象。没有它，"库里的数据对应哪套参数"就只能靠猜。

**Independent Test**: 入库后读该文件，逐项核对字段齐备、数值与库和磁盘产物一致。

**Acceptance Scenarios**:

1. **Given** 首次入库完成，**When** 读 `data/index_manifest.json`，**Then** 文件存在且含 `built_at`/`collection`/`metric_type`/`index_type`/`dim`/`total_chunks`/`documents`/`pipeline_config_hash`/`pipeline_config`/`app_config` 全部字段
2. **Given** 上述文件，**When** 核对数值，**Then** `total_chunks == 65`、`dim == 1024`、`metric_type == "COSINE"`、`index_type == "FLAT"`、`documents[0].source_hash` 与 `chunk_count == 65`、`page_count == 15` 均与库和磁盘产物一致
3. **Given** 同一份文档重复入库，**When** 再次读该文件，**Then** `documents` 数组中该文档**只有一条记录**（不重复追加）
4. **Given** 另一份文档随后入库，**When** 读该文件，**Then** `documents` 数组中**两份文档都在**，`total_chunks` 为其和

---

### User Story 5 - 失败不留半份，错误可定位 (Priority: P2)

入库中途 Milvus 掉线、或插入行数与预期不符。脚本必须让库回到**运行前的状态**，而不是留下一份"看起来有数据但引用会指错页"的半成品。

**Why this priority**: §9.2「禁止部分成功」——半份文档入库会让引用指向不存在的页码，这比"没入库"危险得多，因为前者会被用户当成可用结果。

**Independent Test**: 在插入过程中人为制造失败（如中途断连），确认命令非 0 退出且 collection 行数回到运行前的值。

**Acceptance Scenarios**:

1. **Given** collection 已有该文档的旧数据，**When** 插入中途失败，**Then** 命令非 0 退出，且该文档的数据回到**运行前的完整状态**（不是"删了旧的、没进去新的"）
2. **Given** 插入返回的成功行数与预期 chunk 数**不相等**，**When** 脚本检测到，**Then** **立即回滚并报错非 0 退出**，MUST NOT 报告成功
3. **Given** Milvus 不可达，**When** 运行命令，**Then** 报错信息指明是连接问题、给出连接地址，并以**退出码 3** 结束（`docs/05 §5` 约定：3 = 外部依赖不可用）
4. **Given** 任何失败情形，**When** 读错误信息，**Then** 信息包含足够定位的上下文（哪一步、哪个 `chunk_id` 或哪一批次、原始异常），且**不回显凭据**

---

### Edge Cases

| # | 情形 | 期望处置 |
|---|---|---|
| E1 | `.npy` 行数与 `.rows.jsonl` 行数不等 | 校验失败，退出码 2，指出两边行数 |
| E2 | `.rows.jsonl` 的 `chunk_id` 与 `chunks.jsonl` 的 `chunk_id` **顺序错位**（不是集合不等，而是第 i 行对不上） | 校验失败，退出码 2，指出第一个错位的 `row_index` —— **这是最危险的输入错误：向量会配错文本** |
| E3 | `chunks.jsonl` 内 `chunk_id` 重复 | 校验失败，退出码 2 |
| E4 | `.npy` 的第 i 行模长不是 1（S5 承诺 L2 归一化） | 校验失败，退出码 2 —— 度量是 COSINE，非归一化向量会让打分失真 |
| E5 | 某 chunk 的 `page_start` 缺失或为 0 | 校验失败，退出码 2（`docs/04`：页码缺失的 chunk 数必须为 0） |
| E6 | 某 chunk 的 `source_hash` 与其他行不同 | 校验失败，退出码 2（一份文档的 chunk 必须同属一份源文件） |
| E7 | `text` 为空或全空白 | 校验失败，退出码 2 |
| E8 | 某个字段的字节长度超过 collection 的 `max_length` | 写入前校验并指出是哪条 `chunk_id` 的哪个字段超长；**MUST NOT** 靠 Milvus 在插入时抛错 |
| E9 | `data/embeddings/` 下存在 S5 遗留的 `{doc_id}.npy.tmp.npy` | **忽略**它（只认精确文件名 `{doc_id}.npy`），且在报告中不误报为第二份产物 |
| E10 | 产物目录下有多份 `doc_id` 的产物，未指定 `--doc-id` | 见 FR-002 的处置 |
| E11 | collection 存在但 schema 与 §9.1 不符（如维度不是 1024、主键不是 `chunk_id`） | 校验失败并明确报出差异字段；**MUST NOT** 静默按现有 schema 写入 |
| E12 | `data/index_manifest.json` 存在但 `pipeline_config_hash` 与库内不一致 | 提示两个值并要求人工确认，MUST NOT 自动覆盖 |
| E13 | Milvus 版本不支持 `FLAT` 索引或 `COSINE` 度量 | 报错并给出实测的服务端版本；**MUST NOT** 自动降级为别的索引类型 |

---

## Requirements *(mandatory)*

### 功能需求

**入口与调用**

- **FR-001**: 脚本 MUST 以 **`D:/zg6_Project/9/med_rag/rag/python.exe`** 的绝对路径调用（宪法原则 I）。脚本自身与说明文档 MUST NOT 出现裸 `python` / `python3` / `py`。
- **FR-002**: 脚本 MUST 有**且只有一个**命令入口。文档标识 MUST 以**位置参数** `doc_ids` 传入（沿用 `backend/embed_chunks.py:219` 的既有约定），**留空 = 处理 `data/chunks/` 下的全部文档**；**MUST NOT** 通过"猜一个默认文档"来隐式选择。
- **FR-003**: 入口文件的代码行数 MUST ≤ 300 行；超出部分 MUST 拆到 `backend/` 下的子模块，但**对外仍只有这一个入口**（与 `backend/embed_chunks.py` + `backend/embed/` 的既有模式一致）。
- **FR-004**: 脚本 MUST NOT 被自动运行。交付时 MUST 给出可直接复制的运行命令、退出码含义、前置依赖安装命令。

**输入校验（写库之前的闸门）**

- **FR-005**: 脚本 MUST 在**任何写操作之前**完成 E1–E7 的全部校验，并把校验结果计入报告。任一不通过 MUST 以退出码 2 结束，且**不得发生任何写操作**。
- **FR-006**: 行对齐校验 MUST 是**逐条**的（`rows[i].chunk_id == chunks[i].chunk_id`），MUST NOT 退化为集合比较（E2 的错位会因此漏过）。
- **FR-007**: 脚本 MUST 校验每个 chunk 的 `page_start`/`page_end` 存在且 `page_end ≥ page_start`。
- **FR-008**: `source_hash` 字段 MUST 照实存 `chunks.jsonl` 中的取值（即 MinerU `_origin.pdf` 的 sha256），MUST NOT 改写或重算；同时**删除键与幂等键 MUST 用 `doc_id`**（D1）。

**写入**

- **FR-009**: collection schema MUST 与 `docs/04 §9.1` 一致：`chunk_id`(VARCHAR, 主键)、`vector`(FLOAT_VECTOR, 1024, COSINE)、`text`、`doc_id`、`file_name`、`page_start`(INT32)、`page_end`(INT32)、`section`、`block_type`、`source_hash`、`pipeline_config_hash`(VARCHAR)、**`chunk_meta`(JSON)**。
- **FR-009a**: `text` 列存的是**展示用原文**，而向量由 `text_for_embedding`（章节路标 + 正文）算出。为使"这条向量从哪串文本算出来"**可核对**，MUST 把 S4 记录中未单独成列的字段整体存入 `chunk_meta`(JSON)，规则为「**S4 整条记录减去 `SCALAR_FIELDS` 中已有的列**」—— MUST NOT 手写白名单（白名单会与 schema 脱节）。MUST 至少包含 `text_for_embedding`、`heading_path`、`block_ids`、`sub_index`、`char_len`、`chunk_rule_version` 六项。
- **FR-009b**: 写入前 MUST 校验 `chunk_meta`：① 键名只含字母/数字/下划线（Milvus 对 JSON 键的硬约束）；② 序列化后 ≤ **65536 字节**（Milvus 单个 JSON 字段上限）；③ 六个必需键齐备。任一不过 MUST 在**写操作之前**以退出码 2 结束，并指名是哪条 `chunk_id`。
- **FR-009a**: collection 不存在时 MUST **自动创建**（D3），无需额外开关；创建时 MUST 一次性带上索引配置（FLAT + COSINE）与 load 状态，不留"建了但没索引/没 load"的中间态。collection **已存在**但 schema 与 §9.1 不符时 MUST 报错退出（E11），MUST NOT 自动改 schema。
- **FR-010**: `text` 字段 MUST 存 chunk 的 `text`（展示用原文），MUST NOT 存 `text_for_embedding`。
- **FR-011**: 向量 MUST 取 `.npy` 的**第 `row_index` 行**，且该行 MUST 与 `rows.jsonl` 中 `row_index` 对应的 `chunk_id` 绑定 —— MUST NOT 依赖"读文件的先后顺序恰好一致"。
- **FR-012**: 写入 MUST 以**单份文档为原子单位**（§9.2）：一份文档要么 65 行全进，要么完全不留痕。MUST NOT 出现半份文档入库。
- **FR-013**: 脚本 MUST 在插入返回后核对**成功行数 == 预期 chunk 数**，不等则回滚并非 0 退出（§9.2「禁止部分成功」）。
- **FR-014**: 删除操作 MUST 只按 `doc_id` 过滤（D1）；MUST NOT 出现无过滤条件的 `delete` / `drop`。
- **FR-015**: `FLAT` 索引 MUST 在**写入完成之后**统一创建（§9.2：避免索引在中途状态被查询使用）。
- **FR-016**: collection MUST 在写入后 load，使数据可被检索。

**幂等与版本门禁**

- **FR-017**: 同一份输入重复运行 MUST 得到相同的库内容（行数与 `chunk_id` 集合均不变）。
- **FR-018**: 写入前 MUST 计算本次的 `pipeline_config_hash` 并与库内已记录的版本比对；不一致时 MUST **拒绝写入**并非 0 退出，错误信息 MUST 给出**差异的具体参数**（§10.2）。
- **FR-019**: `pipeline_config_hash` 的输入 MUST 为（D2）：**数值参数以实际生效的代码常量为准** —— `chunk_min_chars=300`、`chunk_max_chars=700`、`chunk_overlap=0`、`min_ratio=0.6`（`backend/chunk/core.py`），加上 S5 `fingerprint.json` 的**全部字段**，加上清洗/分块规则版本串（`clean-v1`、`chunk-v1+bge-m3`）。**数值 MUST 参与 hash**，不得只依赖规则版本串。查询期参数（`top_k`、相似度阈值）MUST NOT 参与该 hash（§10.2 末段）。
- **FR-019a**: `pipeline_config_hash` 的计算 MUST 是**确定性**的（同一组输入恒得同一值），且 MUST 记录**输入清单本身**（不只记录哈希），以便不一致时报出差异的具体参数（US3-2）。
- **FR-020**: 破坏性操作（清空 collection / 全量重建）MUST 需要调用方**显式**给出确认开关才执行；缺少该开关时，脚本 MUST 只报出"需要重建"这一事实，MUST NOT 执行删除。
- **FR-021**: 脚本 MUST NOT 因"参数变了"而自动触发全量重建（§10.2：自动重建会在无人知晓时清空索引）。

**产出物与报告**

- **FR-022**: 入库成功后 MUST 写出/更新 `data/index_manifest.json`（D4），字段与 `docs/04 §9.3` 一致。**首次创建**；后续入库 MUST **读入已有文件、按 `doc_id` 覆盖该文档条目**、保留其它文档条目、更新 `total_chunks`/`built_at`/`pipeline_config_hash`。同一 `doc_id` 重复入库 MUST 覆盖而非追加（US4-3）。
- **FR-022a**: 脚本 MUST 处置"库里有、manifest 里没有"与"manifest 里有、库里没有"这两类不一致：MUST 以**库的实际内容**为准重建 `documents`（可用 `query` 按 `doc_id` 聚合 `count(*)` 得到），并把修正结果计入报告，MUST NOT 静默继承一份过期的 manifest。
- **FR-023**: 运行结束 MUST 在终端给出可读报告：本次写入行数、collection 当前总行数、索引配置、参数版本哈希、以及每一项校验的通过情况。报告 MUST 脱敏（不含凭据）。
- **FR-024**: 失败时 MUST 以非 0 退出码结束，并提供足够的定位信息（哪一步、哪个 `chunk_id`/批次、原始异常）。MUST NOT 用 `try/except` 吞掉校验失败、连接失败、行数不符这类失败条件（宪法「代码规范」）。
- **FR-025**: 退出码 MUST 与 `docs/05 §5` 的约定一致：`0` 成功 / `1` 参数错误 / `2` 校验失败 / `3` 外部依赖不可用。若本步骤需要区分更多情形，MUST 复用 `backend/embed/__init__.py` 的既有风格（在模块内集中定义常量）而**不得**与既有含义冲突。

**连接与凭据**

- **FR-026**: Milvus 连接地址 MUST 可配置（命令行参数），默认值取 `http://localhost:19530`。
- **FR-027**: 若 Milvus 启用了认证，凭据 MUST 只从环境变量读取（宪法原则 III）。MUST NOT 硬编码进源码、配置、日志或错误信息。当前实例实测**无认证**，因此该路径 MUST 支持"未设置时按无凭据连接"，但 MUST NOT 静默忽略"设置了却无效"的情形。

**代码规范（宪法）**

- **FR-028**: 标识符 MUST 为 `snake_case`；函数签名 MUST 带类型注解，公开函数 MUST 带返回类型注解。
- **FR-029**: 新增依赖 MUST 只装进 `rag/`，且 MUST 同步登记到仓库根 `requirements.txt`。

### 关键实体

- **Chunk 记录**：一个可检索的最小单元。由 `chunk_id` 唯一标识，携带展示原文、来源文档标识、页码范围、所属章节、块类型，以及指向向量矩阵某一行的引用关系。它的 **页码与章节是引用卡片的全部依据**——错了就等同于编造出处。
- **Collection**：Milvus 中承载全部 chunk 的容器，固定 1024 维、COSINE 度量、FLAT 索引。
- **索引清单（`index_manifest.json`）**：索引的身份证。记录"库里的数据是用哪套参数产出的"，是版本门禁的持久化载体。
- **参数版本哈希（`pipeline_config_hash`）**：影响产物的**全部**参数的哈希。它是"参数变了没有"的唯一判据，也是拒绝写入的依据。

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 首次入库后，collection 行数**恰好 65**，其中每一条的 `chunk_id` 都能在 `chunks.jsonl` 中找到，且 `text` 逐字相同（抽 20 条 100% 命中）。
- **SC-002**: 连续两次入库后，行数仍为 **65**，`chunk_id` 集合与第一次**完全相同**（零重复）。
- **SC-003**: 构造一处参数版本不一致，脚本**必然**拒绝写入并非 0 退出；退出后 collection 行数与内容与运行前**逐条相同**（抽查 3 种不一致：分块规则、embedding 指纹、清洗规则）。
- **SC-004**: 构造一次插入中途失败，脚本**必然**回滚并非 0 退出，collection 回到运行前状态；**不出现**"旧数据已删、新数据未进"的中间态。
- **SC-005**: E1–E13 十三项边界情形**逐项**有明确的、可断言的处置，无一项被静默忽略；其中 E1/E2/E3/E4/E5/E6/E7 七项在**发生写操作之前**被拦下。
- **SC-006**: 从零开始的使用者，仅依据入口脚本头部的说明，可完成「装依赖 → 跑入库 → 核对报告」全流程，无需阅读其它文件。
- **SC-007**: 用一个 chunk 的 `text` 做检索能命中它自己（`score ≥ 0.999`），且返回的 `page_start`/`section` 与磁盘产物一致 —— 即**库里的数据确实可被运行时检索消费**。
- **SC-008**: 入口脚本行数 ≤ **300** 行；超出部分拆入子模块后，对外入口仍只有一个。
- **SC-009**: `data/index_manifest.json` 的每个字段都能与库或磁盘产物对上，无一项是手工填写或推测值。

---

## Assumptions

- **本特性只做 S6 入库**，不做查询检索接口（`docs/05 §4` I-08/I-04 等）、不做后端服务、不做 `verify`/`rebuild` 子命令的完整实现（若 Q3/Q4 裁决要求，仅做本步骤必需的最小集）。
- **S5 产物已通过验收**（`specs/004`，`.smoke_out/s5_accept.txt` 17 PASS / 0 FAIL），本步骤不重新校验模型指纹的**内部**一致性，只把它作为 hash 的输入。
- **运行环境**：`D:/zg6_Project/9/med_rag/rag/python.exe`（Python 3.12.14），不新建虚拟环境（宪法原则 I）。
- **依赖安装需人工确认**（宪法「Development Workflow」）：本步骤需要 `pymilvus`，当前**未安装**。脚本交付时 MUST 给出确切的安装命令，MUST NOT 自行安装。安装后 MUST 回写 `requirements.txt`。
- **Milvus 为单机 Docker 部署、无认证、当前 collection 为空**（均实测确认）。本规格不覆盖集群部署、TLS、RBAC 的配置方式。
- **`text` 字段的 `max_length`**：Milvus 的 VARCHAR 以**字节**计，实测最大 7,794 字节。本规格按"留足余量但不无谓放大"取值，**具体数值见 Q5**（Milvus 上限为 65,535）。
- **`data/source_data` 语料范围尚未确定**（宪法 TODO(SOURCE_DATA_SCOPE)），本特性只处理已存在的这一份 PDF。
- **`docs/04 §7` 与 `§9.3` 中被 S4 的 D1 裁决作废的旧参数值（300–500 字）尚未回写**。本特性 MUST NOT 依赖这些旧值；是否回写由 Q2 裁决一并决定。
- **`docs/05 §5` 描述的 `backend.pipeline <子命令>` 统一调度器尚不存在**；S2–S5 实际各自有独立入口（`parse_pdf.py` / `clean_parsed.py` / `chunk_clean.py` / `embed_chunks.py`）。本特性**沿用实际做法**（独立入口，命名对齐 §5 的子命令名 `index`）。该文档漂移不在本特性范围内。

---

## 已裁决决定（人工裁决于 2026-09-27）

四项全部选 **A**。以下为固化后的决定，实现 MUST 照此执行。

### D1 — `source_hash` 的取值来源与删除键：**选项 A**

**决定**：

- **删除键与幂等键 = `doc_id`**（= 源 PDF 的 sha256 前 12 位，实测 `d6da41b5d356` 与 `data/source_data/*.pdf` 的哈希前缀一致）。删除用 `doc_id == "<id>"` 过滤（FR-014）。
- **`source_hash` 字段照实存** `chunks.jsonl` 里的值（= MinerU `_origin.pdf` 的 sha256，实测 `39ced98c…79bb8`），**不改写、不重算、不用它做幂等**。

**理由**：`doc_id` 由源 PDF 内容决定，"同一份 PDF 重跑整条管线"因此天然幂等；`source_hash` 依赖 MinerU 副本字节，换版本重跑解析就会变，用它做删除键会**新旧版本并存**（同一段落两条命中，引用指向已作废的排版）。两者语义不同，各自保留原义即可，无需新增字段。

**须回写**：`docs/04 §10.1` 现称"`source_hash`（PDF 内容 sha256）… `doc_id` 是它的前 12 位"——与实测矛盾，须更正为：`doc_id` = 源 PDF sha256 前 12 位（文档级幂等键）；`source_hash` = MinerU `_origin.pdf` 的 sha256（上游产物标识，不用于幂等）。

### D2 — `pipeline_config_hash` 的输入：**选项 A**

**决定**：hash 输入 = 下列三组，**数值参数必须参与**。

1. **数值参数（以 `backend/chunk/core.py` 实际生效的常量为准）**：`chunk_min_chars=300`、`chunk_max_chars=700`、`chunk_overlap=0`、`min_ratio=0.6`。
2. **S5 `fingerprint.json` 的全部字段**：`model_dir`、`config_sha256`、`weight_file`、`weight_bytes`、`max_length`、`pooling`、`normalization`、`dtype`、`dim`、`rule_version`。
3. **规则版本串**：`clean-v1`、`chunk-v1+bge-m3`、`embed-v1+bge-m3`。

**理由**：`chunk_rule_version`（`chunk-v1+bge-m3`）**不编码 `LOW`/`HIGH`**。若 hash 只由规则串拼成，只改数值不改版本串时门禁不触发——而"改了分块参数"正是 §10.3 重跑矩阵里必须重跑 S4→S6 的那一行。**把数值纳入 hash 是让门禁真正生效的唯一办法。**

**须回写**：

- `docs/04 §7`：现写"300–500 字 + 禁止跨章节合并"，已被 S4 的 D1 作废且实际生效值为 **300–700**（下限 500→300 的理由见 `backend/chunk/core.py:10-12`），须更正并注明有限跨章节合并规则。
- `docs/04 §9.3`：`pipeline_config` 示例中的 `chunk_max_chars: 500` 须改为 **700**；`chunk_min_chars: 300` 本就正确，保留。

### D3 — 未指定文档标识 / collection 不存在时的处置：**选项 A**

**决定**：

- 文档标识为**位置参数** `doc_ids`，**留空 = 处理 `data/chunks/` 下的全部文档**（沿用 `backend/embed_chunks.py:219` 的既有约定）。
- 多份文档时，**每份文档各自原子提交**（FR-012），一份失败不影响其它份的既成结果，但整体 MUST 非 0 退出并逐份报出结果。
- **collection 不存在时自动创建**（按 §9.1 的 schema + FLAT/COSINE），无需额外开关。

**理由**：S2–S5 四个入口（`parse_pdf.py`/`clean_parsed.py`/`chunk_clean.py`/`embed_chunks.py`）调用方式一致，S6 保持一致比"额外保守"更有价值——运维不必为最后一步换一套调用习惯。创建 collection 是**幂等的非破坏性**动作，把它做成需要显式开关的额外概念，收益不抵认知成本。

### D4 — `index_manifest.json` 的写入职责与合并语义：**选项 A**

**决定**：**本脚本负责写** `data/index_manifest.json`。

- 首次创建；后续入库读入已有文件，**按 `doc_id` 覆盖该文档条目**，保留其它文档条目，更新 `total_chunks` / `built_at` / `pipeline_config_hash` / `pipeline_config`。
- **不一致以库的实际内容为准**（FR-022a）：按 `doc_id` 从 collection 聚合 `count(*)` 重建 `documents`，修正结果计入报告。

**理由**：`docs/04 §9.3` 已把该文件定义为 S6 的产出物与后端启动校验的对象；测得的 US4 三条验收场景（字段齐备、重复入库不追加、多文档都保留）在 A 下均可断言。

---

## 未纳入本特性的已知文档漂移（记录，不在本次修复范围）

- `docs/05 §5` 描述的 `backend.pipeline <子命令>` 统一调度器**不存在**；实际是各步独立入口。本特性沿用实际做法，不建调度器。



---

## Constitution Check（宪法 1.0.0 逐条自检）

| 原则 | 结论 | 依据 |
|---|---|---|
| **I. 环境锁定与依赖治理** | ✅ 通过 | FR-001 锁定 `rag/python.exe` 绝对路径；FR-029 要求新依赖只进 `rag/` 并回写 `requirements.txt`；Assumptions 明确"不自行安装、给出命令并由人工确认"。 |
| **II. 无据不答与强制溯源引用** | ✅ 通过 | 本步骤**不生成任何知识性陈述**，只搬运原文。且 FR-007/FR-010 保证写进库的 `text` 是原文、页码完整——这正是"可回原文核对"的**数据前提**：库里页码错了，运行时的引用就是编造。US3-3 的实时核对场景即此原则在数据管线侧的落地。 |
| **III. 密钥零硬编码** | ✅ 通过 | FR-026/FR-027：地址可配、凭据只走环境变量、日志脱敏；FR-024 明确错误信息不回显凭据。 |
| **IV. 紧急症状前置响应** | ➖ 不适用 | 本步骤不产生任何面向用户的回答文本。 |
| **V. 面向群众的医疗安全边界** | ➖ 不适用 | 同上；本步骤不产生诊断、处方或剂量内容。 |
| **附加约束 · 代码规范** | ✅ 通过 | FR-028 强制 `snake_case` 与类型注解；FR-024 明确禁止用 `try/except` 吞掉校验/连接/行数不符这类失败条件。 |
| **附加约束 · 知识与数据边界** | ✅ 通过 | 本步骤不新增语料，只索引已存在的、经 S1 准入的那一份 PDF；语料范围未定这一 TODO 未被本步骤扩大。检索参数（`top_k`/阈值）不参与 `pipeline_config_hash`（FR-019），符合"不依赖库默认值"的记录要求——它们由运行时接口显式传入（`docs/05 §4`）。 |
| **开发工作流 · 先规格后实现** | ✅ 通过 | 本文件即 `/speckit-specify` 的产出；后续 MUST 依次经 `/speckit-plan` → `/speckit-tasks` 才进入实现。 |
| **开发工作流 · 依赖安装需人工确认** | ✅ 通过 | 见 Assumptions；`pymilvus` 的安装命令随交付一并给出，由用户自行执行。 |

**门禁结论**：无违规项，无需 Complexity Tracking。进入 `/speckit-clarify`（裁决上述 4 项）前的规格自检通过。
