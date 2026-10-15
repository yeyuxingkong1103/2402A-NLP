# Feature Specification: 向量化（S5 步骤）

**Feature Branch**: `004-bge-embedding`

**Created**: 2026-09-23

**Status**: Draft

**Input**: User description: "生成 S5 向量化脚本，保存到 `d:\zg6_Project\9\med_rag\backend`，300 行以内，超出则分模块。输入 `data/chunks/d6da41b5d356.chunks.jsonl`，输出到 `data` 目录，模型 `E:\资料\BAAI--bge-m3`。生成后让用户查看并说明运行方法，不擅自运行。遇到不明确要求时汇报并给出 2-3 个方案供用户决定。"

## 背景与定位

本特性实现 `docs/04_数据管线设计.md` §8 的 **S5 向量化**，是 `specs/003-semantic-chunking` 的下游：把 `data/chunks/{doc_id}.chunks.jsonl` 变成 `data/embeddings/{doc_id}.npy` + `{doc_id}.rows.jsonl`，供 S6 入库。

**这是整条管线里唯一会静默失效的环节**（`docs/04 §8`）。若入库与查询用了不同版本的权重、不同的归一化方式，甚至只是不同的截断长度，**检索会变差但不报错**——系统照常返回结果，只是相关性和拒答率悄悄劣化，排查时很难想到是模型版本问题。因此本规格把"防漂移"当作第一议题，而非附属要求。

---

## 实测基线

| 项 | 实测值 |
|---|---|
| 输入 chunk 数 | **65** |
| 模型权重 | `E:\资料\BAAI--bge-m3`，`pytorch_model.bin` **2.27 GB**，架构 `XLMRobertaModel`，隐藏层 **1024** |
| 运行设备 | **CPU**（`torch 2.12.1+cpu`，**无 CUDA**） |
| `text_for_embedding` token 数（BGE-M3 tokenizer 实测） | 中位 **283**，最大 **3260**，最小 25 |

### 截断长度是必须裁决的事（实测分布）

| 阈值 | 超过阈值的 chunk 数 |
|---|---|
| **> 512 token**（多数库的默认值） | **9 / 65（14%）** |
| > 1024 token | 2 / 65 |
| > 2048 token | 1 / 65 |
| > 4096 token | **0** |

最长的一条是 `d6da41b5d356:0028`（表格「表4 基层常用降压药物的用法、适应证、禁忌证及不良反应」），**3260 token / 3160 字**。

**若采用默认的 `max_length=512`，9 个 chunk 会被静默截断，其中 1 个丢掉 84% 的内容。** 丢的正是那张表的后半张——`docs/04 §7` 专门论证过"切了表头，后半张表的列就失去含义"，而截断会以完全相同的方式毁掉它，且**不留任何痕迹**。

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 把分块产物变成向量矩阵，且行序严格对齐 (Priority: P1)

运维人员对一份已分块的文档运行向量化命令，得到一份 `[N, 1024]` 的 `float32` 矩阵和一份逐行描述表。

**Why this priority**: 这是 S6 入库的直接输入。`docs/04 §8` 指出**类型不匹配的批量插入是 Milvus 的主要崩溃原因**——把 `float64` 或形状错误的数组插进去，库要么报错要么写入垃圾。

**Independent Test**: 对 65 个 chunk 跑一次，断言 `len(rows) == npy.shape[0] == 65` 且 `npy.shape[1] == 1024`。

**Acceptance Scenarios**:

1. **Given** 一份分块产物，**When** 运行向量化命令，**Then** 生成 `.npy` 与 `.rows.jsonl`，退出码 0
2. **Given** 向量化完成，**When** 断言，**Then** `len(rows) == npy.shape[0]`，且 `rows[i].chunk_id` 与分块产物第 i 行的 `chunk_id` **完全一致**
3. **Given** 向量化完成，**When** 检查 `npy.dtype`，**Then** 为 `float32`，`npy.shape[1] == 1024`

---

### User Story 2 - 每一行都真正做了 L2 归一化 (Priority: P1)

矩阵的每一行模长都等于 1.0。

**Why this priority**: `docs/04 §8` 规定归一化后写入，使**余弦相似度 = 内积**，与 Milvus 的 `COSINE` 度量一致。漏归一化不会报错，只会让检索结果悄悄变差——与漂移同类。

**Independent Test**: 逐行算 `‖v‖₂`，全部落在 `1.0 ± 1e-5`。

**Acceptance Scenarios**:

1. **Given** 向量化完成，**When** 逐行计算模长，**Then** 全部在 `1.0 ± 1e-5` 内
2. **Given** 归一化完成，**When** 检查 `rows[i].vector_norm`，**Then** 该字段如实反映实测模长（而非硬写 1.0）
3. **Given** 某行模长偏离 1.0 超出容差，**When** 运行命令，**Then** 报错并非 0 退出，**不得**静默写出一份"看起来正常"的矩阵

---

### User Story 3 - 模型指纹落盘，挡住静默漂移 (Priority: P1)

每次向量化都产出当前模型与参数的指纹，供后端启动时校验。

**Why this priority**: `docs/04 §8` 把漂移列为"整条管线唯一会静默失效的环节"，并给出三重防护。第一重就是**能力探测**：把模型的可辨识特征算成指纹。没有这一步，后两重防护无从谈起。

**Independent Test**: 连续跑两次向量化，两次输出的指纹**逐字符相同**；把权重文件改名后再跑，指纹**必须改变或命令报错**。

**Acceptance Scenarios**:

1. **Given** 一次成功的向量化，**When** 查看产物，**Then** 存在记录模型指纹的字段，且包含：权重目录、`config.json` 哈希、权重文件大小
2. **Given** 同一模型与参数连续跑两次，**When** 比较指纹，**Then** 完全一致
3. **Given** 权重文件被替换或截断，**When** 运行命令，**Then** 指纹改变（或命令直接报错），**不得**沿用旧指纹

---

### User Story 4 - 截断与异常必须显式暴露，绝不静默 (Priority: P1)

超过截断上限的 chunk、空文本 chunk、数量不符，全部以非 0 退出码或显式报告暴露。

**Why this priority**: 用户与项目都把"静默失效"列为最危险的缺陷。截断尤其隐蔽——它不报错、不留痕，只是让向量少了一截内容。

**Independent Test**: 构造一条超长 chunk 与一条空 chunk，分别确认前者被报告、后者被拒绝。

**Acceptance Scenarios**:

1. **Given** 某 chunk 的 token 数超过截断上限，**When** 向量化完成，**Then** 该 chunk 被记入**超长清单**并在报告中列出（含 `chunk_id` / token 数 / 丢失比例）
2. **Given** 某 chunk 的文本为空或纯空白，**When** 运行命令，**Then** 报错并非 0 退出，**不得**生成零向量（`docs/04 §8`：零向量与任何文本的余弦相似度都是 0，会静默污染检索）
3. **Given** 分块产物行数与预期不符（如写入中途被截断），**When** 运行命令，**Then** 不复用旧产物、重新全量计算

---

### Edge Cases

- **输入分块产物不存在** → 报错非 0 退出，提示先跑 S4
- **权重目录不存在或缺 `pytorch_model.bin`** → 报错非 0 退出，指出具体缺失项
- **权重文件大小与预期严重不符（下载不完整）** → 报错非 0 退出
- **`text_for_embedding` 字段缺失**（旧版本分块产物）→ 报错非 0 退出，**不得**退化为用 `text`
- **某 chunk 的 token 数超上限** → 记入超长清单（见 Q1 裁决），不得静默
- **某 chunk 文本为空** → 报错非 0 退出
- **CPU 无 CUDA** → 正常在 CPU 上跑，不报错、不尝试 GPU
- **同一 `doc_id` 重复向量化** → 覆盖旧产物
- **批内 padding 造成长度爆炸**（一个批里混进 3260 token 的长 chunk，会把同批其余 15 个都 padding 到 3260）→ 须按 token 长度排序分批再还原顺序
- **磁盘空间不足导致 `.npy` 写一半** → 不留可被误认为完整的产物

---

## Requirements *(mandatory)*

### Functional Requirements

**通用**

- **FR-001**: 脚本 MUST 位于 `backend/`，**总行数 ≤ 300**；超出则拆到子目录，但**只留一个入口脚本**
- **FR-002**: 脚本 MUST 接受文档标识作为输入；未指定时处理 `data/chunks/` 下全部分块产物
- **FR-003**: 脚本 MUST 以 `data/embeddings/{doc_id}.npy` 与 `{doc_id}.rows.jsonl` 为输出（`docs/04 §4` 产物契约）
- **FR-004**: 脚本 MUST 加载 `E:\资料\BAAI--bge-m3` 的**本地**权重，MUST NOT 联网下载
- **FR-005**: 脚本 MUST NOT 硬编码密钥或凭据（宪法原则 III）
- **FR-006**: 脚本 MUST 以退出码区分成功 / 输入问题 / 模型问题 / 空文本
- **FR-007**: 脚本 MUST NOT 用宽泛异常捕获吞掉失败

**编码口径（`docs/04 §8` 锁定）**

- **FR-008**: 编码输入 MUST 是 chunk 的 **`text_for_embedding`**，MUST NOT 用 `text`
- **FR-009**: 向量 MUST 为 **1024 维**，dtype MUST 为 **`float32`**
- **FR-010**: 每行 MUST 做 **L2 归一化**
- **FR-011**: 批大小 MUST 显式配置（默认 16），MUST NOT 依赖库默认值
- **FR-012**: 向量矩阵的行序 MUST 与 `rows.jsonl` 行序**严格一致**，且与分块产物的 `chunk_id` 顺序一致
- **FR-013**: 空或纯空白文本 MUST 报错，MUST NOT 生成零向量

**产出契约**

- **FR-014**: `.rows.jsonl` 每行 MUST 含 `row_index`、`chunk_id`、`char_len`、`vector_norm`（`docs/04 §8`）
- **FR-015**: `vector_norm` MUST 是**实测模长**，MUST NOT 硬写 1.0
- **FR-016**: 脚本 MUST 在写出前断言 `len(rows) == npy.shape[0]`，不符即报错

**防漂移（`docs/04 §8` 第一重防护）**

- **FR-017**: 脚本 MUST 产出**模型指纹**，构成见 Q2 裁决
- **FR-018**: 指纹 MUST 写入产物，供 S6 的 `index_manifest.json` 与后端启动校验使用
- **FR-019**: 同一模型与参数连续两次运行，指纹 MUST 逐字符相同（确定性）
- **FR-020**: 权重文件被替换/截断时，指纹 MUST 改变或命令报错

**可观测性**

- **FR-021**: 脚本 MUST 输出报告：chunk 数、向量形状、dtype、模长区间、耗时、模型指纹
- **FR-022**: 脚本 MUST 输出**超长 chunk 清单**（超过截断上限的），含 `chunk_id`、token 数、字符数、丢失比例
- **FR-023**: 脚本 MUST 输出 token 长度分布（中位/最大/最小）

### Key Entities

- **分块 (Chunk)**: `chunks.jsonl` 一行。向量化的输入单元；其中 `text_for_embedding` 是编码输入，`chunk_id` 是行序对齐的锚点。
- **向量矩阵 (Embedding Matrix)**: `[N, 1024]` 的 `float32` 数组，逐行 L2 归一化。与 `rows.jsonl` 靠**行号**绑定。
- **行对齐表 (Rows)**: 第 i 行描述矩阵第 i 行。它存在的唯一目的是**让"行号 ↔ chunk_id"的对应关系可被断言**——一旦错位，引用就会指向错误的原文，且不会报错。
- **模型指纹 (Model Fingerprint)**: 模型与编码参数的可辨识摘要。它是三道防漂移防线里唯一在**写入侧**生效的那道。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 对 65 个 chunk（中位 283 token）在 **CPU 上 5 分钟内**完成
- **SC-002**: `len(rows) == npy.shape[0] == 65`，且 `npy.shape[1] == 1024`、`dtype == float32`
- **SC-003**: **100%** 的行满足 `|‖v‖₂ − 1.0| ≤ 1e-5`
- **SC-004**: `rows[i].chunk_id` 与分块产物第 i 行**逐条一致**，抽样 10 条 100% 命中
- **SC-005**: 同一输入连续跑两次，模型指纹**逐字符相同**
- **SC-006**: 全部 11 项边界情况均以非 0 退出码或显式报告处置，**没有任何一种被静默忽略**
- **SC-007**: 报告必然包含超长 chunk 清单；清单为空时**显式显示"0 条"**（区别于"未检查"）
- **SC-008**: 从零开始的使用者，仅依据脚本头部说明，可完成"向量化 → 校验形状 → 看待办"全流程
- **SC-009**: 脚本总行数 **≤ 300**

## Assumptions

- **模型已就位**：`E:\资料\BAAI--bge-m3` 含完整的 `pytorch_model.bin`（2.27 GB）。HF 缓存里那份是**不完整的**（只有 22 MB 的 config/tokenizer），故必须用本地目录。
- **加载方式**：直接用环境已有的 `transformers 4.57.6` 加载 `AutoModel`，**不引入 `FlagEmbedding`**（BGE-M3 是标准 `XLMRobertaModel`）。
- **只用稠密向量**：`docs/04 §8` 锁定 V1 只用 1024 维稠密向量，不使用 BGE-M3 的稀疏/ColBERT 表示。
- **本特性只做 S5**，不做入库（S6）、不做后端查询编码（除非 Q3 裁决要求）。
- **不确定"歧义医学表述"的处理**：向量化不涉及语义判断，不引入 LLM。
- 运行环境为 `D:/zg6_Project/9/med_rag/rag/python.exe`（Python 3.12.14）。

---

## 已裁决决定（人工裁决于 2026-09-23）

### D1 — 截断长度：**选项 C（4096）**

**决定**：`max_length = 4096`。

- 实测最长为 3260 token（表4），**4096 之下零截断**——内容零丢失。
- CPU 上比 512 慢，但 65 条的量级仍是分钟级。
- FR-022 的超长清单**仍必须实现**：换语料后可能出现超 4096 的 chunk，届时必须显式报出而非静默截断。

### D2 — 模型指纹：**选项 B（§8 原定义 + 编码参数）**

**决定**：指纹包含以下全部字段。

| 字段 | 取值 |
|---|---|
| `model_dir` | 权重目录的绝对路径 |
| `config_sha256` | `config.json` 的完整 sha256 |
| `weight_file` / `weight_bytes` | 权重文件名与字节数 |
| **`max_length`** | 4096（D1）|
| **`pooling`** | `cls` |
| **`normalization`** | `l2` |
| **`dtype`** | `float32` |

后四项是对 `docs/04 §8` 原定义的**扩展**，须回写文档。理由：§8 只覆盖 `config.json` 哈希与权重文件大小，**挡不住"改了截断长度却没重建库"**——而这正是 §8 自己列为头号风险的漂移。

### D3 — S5 的范围：**选项 B（离线批量 + 可复用编码模块）**

**决定**：编码口径抽成独立模块 `backend/embed/model.py`，**批处理脚本与将来的后端查询共用同一份代码**。

- 模块暴露 `Encoder.encode_documents()` 与 `Encoder.encode_query()` 两个方法。
- BGE-M3 的稠密检索对 query 与 passage 用**同一套编码**（CLS + L2，无指令前缀），故两个方法当前实现一致——但**接口分开**，以便将来换模型（如需要 query 前缀的 bge-large-zh）时不必改调用方。
- **本规格不实现后端**，只保证"口径只有一处定义"。
