# Feature Specification: 语义分块（S4 分块步骤）

**Feature Branch**: `003-semantic-chunking`

**Created**: 2026-09-23

**Status**: Draft

**Input**: User description: "设计 S4 分块脚本，保存到 `d:\zg6_Project\9\med_rag\backend`，约 300 行。输入 `data/clean/*.blocks.jsonl`，按语义分块，块过小则合并，目标每块 500-700 字。遇到不确定是否要分块的情况必须让用户裁决，禁止私自决策。"

## 背景与定位

本特性实现 `docs/04_数据管线设计.md` §7 的 **S4 分块**，是 `specs/002-parse-output-clean` 的下游：把 `data/clean/{doc_id}.blocks.jsonl` 变成 `data/chunks/{doc_id}.chunks.jsonl`，供 S5 向量化。

**本规格的第一议题是「目标长度与现有设计的冲突」**：用户要求 500–700 字，而 `docs/04 §7` 锁定的是 300–500 字。对样例文档实测后，这个冲突比看上去严重得多——见下方实测数据。

---

## 实测基线（来自 `data/clean/d6da41b5d356.blocks.jsonl`，254 块）

| 指标 | 实测值 |
|---|---|
| 正文块数 / 正文字符（非空白） | 180 块 / **27,115 字** |
| **正文块长度中位数** | **84 字**（180 块里 103 块 <100 字） |
| 最长块 / 最短块 | 3,160 字（表4）/ 5 字 |
| section 总数 | 65 |
| **section 长度中位数** | **186 字** |
| **< 500 字的 section** | **52 / 65（80%），合计 9,708 字（占全文 36%）** |
| 最大 section | 4,493 字（`5.4.3 药物治疗方案`，4 块） |
| 表格 | 8 个，合计 7,399 字；最大的表 3,160 字 |

### 这组数字意味着什么

1. **正文块中位数只有 84 字** —— 要凑到 500–700 字，平均每个 chunk 得合并 **6–8 个**块。合并规则是本步骤的主体，不是配角。
2. **80% 的 section 不到 500 字** —— 若同时禁止跨章节合并（`docs/04 §7`），这 52 个 section **永远凑不到 500 字下限**。目标长度与「禁止跨章节合并」两条规则**在数学上互斥**，必须二选一或放宽其一。
3. **表4 有 3,160 字** —— `docs/04 §7` 规定表格整表成一块。它会是目标的 **4.5 倍**。这是既定规则的自然结果，不是缺陷，但须向用户明确。

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 把清洗产物切成可检索、可引用的语义块 (Priority: P1)

运维人员对一份已清洗的文档运行分块命令，得到一份 `chunks.jsonl`：每块是**一段或多段完整的正文**，携带页码区间、章节路径、来源块 ID。

**Why this priority**: 这是向量化的直接输入。`docs/04 §7` 指出切分单位必须是**段落对齐**而非固定字数滑窗——滑窗会把段落拦腰截断，引用卡片就只能展示半句话给用户。

**Independent Test**: 对 `data/clean/d6da41b5d356.blocks.jsonl` 跑一次分块，检查每个 chunk 的 `text` 都能在原文中**逐字找到**（无截断、无拼接痕迹）。

**Acceptance Scenarios**:

1. **Given** 一份清洗产物，**When** 运行分块命令，**Then** 生成 `data/chunks/{doc_id}.chunks.jsonl`，退出码 0
2. **Given** 分块完成，**When** 抽查任一 chunk，**Then** 其 `text` 由若干**完整块**拼接而成，绝不出现半句话
3. **Given** 分块完成，**When** 检查 `block_ids`，**Then** 每块都能据此在清洗产物中定位到全部来源

---

### User Story 2 - 小段合并到目标长度，且不跨越章节 (Priority: P1)

块太小时，脚本把**同一章节内相邻的块**合并，直到接近目标长度。

**Why this priority**: 实测正文块中位数仅 84 字。不合并会产生上百个低信息碎块，检索质量直接崩塌（`docs/04 §7` 把「低于下限的 chunk 占比 >15%」列为异常）。

**Independent Test**: 分块后统计长度分布，确认落在目标区间的 chunk 占比达标，且**没有任何 chunk 跨越 section 边界**。

**Acceptance Scenarios**:

1. **Given** 同一 section 内有多个小段，**When** 分块完成，**Then** 它们被合并，合并后的 chunk 不超上限
2. **Given** 相邻两段分属不同 section，**When** 分块完成，**Then** 它们**不会**出现在同一个 chunk 中（除非用户显式授权跨章节合并）
3. **Given** 某 section 总字数不足下限，**When** 分块完成，**Then** 按本规格的裁决结果处置，且该情况**必须出现在报告里**

---

### User Story 3 - 不确定的边界必须由人裁决，不得私自决定 (Priority: P1)

脚本遇到"切还是不切说不准"的情况时，**停下来把问题交给用户**，而不是自己选一个。

**Why this priority**: 用户显式要求，且与宪法原则 V（医疗安全边界）一致——分块决定了引用粒度和知识边界，"系统自己猜"会让知识库的边界不可审计。

**Independent Test**: 构造一个边界模糊的输入，确认脚本**没有**自行切分，而是产出了待裁决项并以非 0 退出码结束。

**Acceptance Scenarios**:

1. **Given** 出现无法用既定规则判定的边界，**When** 运行分块命令，**Then** 该边界被记录为待裁决项，且命令以非 0 退出码结束
2. **Given** 存在待裁决项，**When** 人工给出答复后重跑，**Then** 该边界按人工答复处置
3. **Given** 无待裁决项，**When** 运行分块命令，**Then** 正常产出，退出码 0

---

### User Story 4 - 产出可人工复核的分块质量报告 (Priority: P2)

分块完成即输出一份指标报告，人工据此判断参数是否合理。

**Why this priority**: `docs/04 §7` 列了 5 项自检指标。没有它们，调参只能靠盲猜。

**Independent Test**: 报告必然包含 `docs/04 §7` 的全部 5 项指标，且页码缺失数为 0。

**Acceptance Scenarios**:

1. **Given** 分块完成，**When** 查看报告，**Then** 可见：chunk 总数、长度中位数、低于下限占比、超长 chunk 清单、页码缺失数
2. **Given** 页码缺失数不为 0，**When** 查看报告，**Then** 该项被标红并以非 0 退出码结束

---

### Edge Cases

- **某 section 只有 1 个小段且远低于下限** → 处置见 Q1 裁决；无论如何 MUST 出现在报告中
- **超长段落（> 上限）** → 按**句边界**二次切分，子块共享 `block_id` 并以 `sub_index` 区分（`docs/04 §7`）
- **单个块自身就超过上限且无法按句切分**（如大表格 3,160 字）→ 表格整表成一块不切（`docs/04 §7`）；MUST 记入报告的「超长 chunk 清单」
- **表格块** → 整表成一块，MUST NOT 切开（切了表头，后半张表就失去列含义）
- **图片块（`image`，2 个降级占位）** → 不单独成 chunk，并入相邻正文；若相邻无正文则记入报告
- **`page_footnote` 块（1 个）** → 并入其所属 section 的相邻块
- **英文段落（Abstract，2,032 字）** → 长度按字符数计，不做语言区分
- **chunk 跨越多个页码** → `page_start` / `page_end` 构成区间，引用卡片显示为"第 12–13 页"
- **清洗产物不存在** → 报错非 0 退出，提示先跑 S3
- **同一 `doc_id` 重复分块** → 覆盖旧产物

---

## Requirements *(mandatory)*

### Functional Requirements

**通用**

- **FR-001**: 脚本 MUST 位于 `backend/`，单文件，**约 300 行**（允许 ±20%）
- **FR-002**: 脚本 MUST 接受文档标识或产物目录作为输入；未指定时处理 `data/clean/` 下全部文档
- **FR-003**: 脚本 MUST 以 `data/chunks/{doc_id}.chunks.jsonl` 为输出（`docs/04 §7` 产物契约）
- **FR-004**: 脚本 MUST 使用**本地** BGE-M3 权重做分块判定（D2），MUST NOT 联网下载、MUST NOT 复用 S5 的向量产物。权重缺失时报错非 0 退出并给出下载指令
- **FR-005**: 脚本 MUST 以退出码区分成功 / 输入问题 / 有待裁决项 / 页码缺失
- **FR-006**: 脚本 MUST NOT 用宽泛异常捕获吞掉失败

**分块规则**

- **FR-007**: 切分 MUST 以**块**为单位（段落对齐），MUST NOT 按字符数滑窗切割，MUST NOT 在句中断开
- **FR-008**: 标题块 MUST NOT 单独成 chunk；其文本 MUST 进入 `heading_path` / `section`
- **FR-009**: chunk 的重叠 MUST 为 **0**（`docs/04 §7`：重叠会破坏引用唯一性）
- **FR-010**: 同 section 内相邻块 MUST 合并至接近目标长度；合并 MUST NOT 超出上限
- **FR-011**: 跨 section 合并 MUST 按 **D1** 执行：仅当两 section 的 `heading_path[:-1]` 相同（同父章节）且合并后不超上限时允许；父章节不同是硬边界
- **FR-012**: 跨文件合并 MUST 禁止（引用必须定位到单一文件名）
- **FR-013**: 表格块 MUST 整表成一块，MUST NOT 切分
- **FR-014**: 超过上限的正文块 MUST 按**句边界**二次切分；子块共享来源 `block_id` 并用 `sub_index` 区分；MUST NOT 在句中截断
- **FR-015**: 图像 / 页脚注释等非正文块 MUST 并入相邻正文块，MUST NOT 静默丢弃

**裁决机制**

- **FR-016**: 脚本 MUST 在遇到本规格未覆盖的边界判定时**停止并请求人工裁决**，MUST NOT 自行选择
- **FR-017**: 待裁决项 MUST 以可读形式输出（终端 + 落盘），每条 MUST 携带：`doc_id`、来源 `block_ids`、页码、文本摘要、**为什么无法自动判定**
- **FR-018**: 人工答复 MUST 可被脚本读入并重跑（答复载体见 Q3 裁决）
- **FR-019**: 存在未答复的待裁决项时，脚本 MUST 以非 0 退出码结束，MUST NOT 写出"看起来完整"的产物

**产出契约**

- **FR-020**: 每条 chunk MUST 含 `docs/04 §7` 的全部字段：`chunk_id`、`doc_id`、`file_name`、`text`、`text_for_embedding`、`page_start`、`page_end`、`section`、`heading_path`、`block_type`、`block_ids`、`sub_index`、`char_len`、`source_hash`、`chunk_rule_version`
- **FR-021**: `text` MUST 是**纯正文原文，一字不改**；`text_for_embedding` MUST 为 `heading_path` 拼接 + 正文，**仅用于向量化，不展示、不入库**（`docs/04 §7` 明确要求把这一区分写进代码注释）
- **FR-022**: `chunk_id` MUST 形如 `{doc_id}:{序号4位}`，分块参数不变时稳定
- **FR-023**: `source_hash` MUST 为完整 64 位（非 `doc_id` 的 12 位截断），用于幂等入库
- **FR-024**: 每块 MUST 携带 `chunk_rule_version`

**质量门槛**

- **FR-025**: 报告 MUST 输出 `docs/04 §7` 的全部 5 项自检指标
- **FR-026**: 页码缺失的 chunk 数 MUST 为 **0**；非 0 时 MUST 非 0 退出
- **FR-027**: 脚本 MUST 输出 chunk 长度分布与超长 chunk 清单

### Key Entities

- **清洗后块 (Clean Block)**: `blocks.jsonl` 一行。分块的输入单元，携带 `block_id`、`page`、`heading_path`、`is_heading`、`char_len`。
- **分块单元 (Chunk Unit)**: 一个或多个**同 section 相邻**块构成的待分块序列。合并、切分都在这个层级进行。
- **分块 (Chunk)**: 输出的一行。携带两份文本（`text` 展示用 / `text_for_embedding` 检索用）与完整溯源链（`block_ids`）。
- **待裁决项 (Pending Decision)**: 规则无法判定的边界。它是本规格的**一等产出物**，不是错误日志——它的存在是"系统没有私自决策"的证据。
- **章节边界 (Section Boundary)**: `heading_path` 的变化点。它是合并的硬边界（除非用户授权跨越）。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 对样例文档（27,115 字 / 约 180 个块）一次分块在 **CPU 上 90 秒内**完成（含模型加载与推断）
- **SC-002**: chunk 的 `text` 能在清洗产物中**逐字找到**，抽样 20 条 100% 命中（无滑窗截断、无改写）
- **SC-003**: 落入 500–700 字区间的 chunk 占比 **≥ 50%**；低于下限的 chunk 占比 ≤ 25%，且每一个都有一个对应的待裁决项或已裁决理由
- **SC-004**: 页码缺失的 chunk 数为 **0**
- **SC-005**: 全部 12 项边界情况（见 Edge Cases）均按规则处置，**没有任何一种被静默忽略**
- **SC-006**: 遇到规则未覆盖的边界时，脚本**必然**产出待裁决项并非 0 退出；抽查 3 个人工构造的模糊边界，3/3 被正确拦下
- **SC-007**: 从零开始的使用者，仅依据脚本头部说明，可完成"分块 → 看报告 → 答复待裁决项 → 重跑"全流程
- **SC-008**: 脚本行数在 **240–360 行**之间

## Assumptions

- **与 `docs/04 §7` 的关系**：§7 原定的 300–500 字与「禁止跨章节合并」**均已由 D1 作废**，须回写 `docs/04 §7`。
- **BGE-M3 权重状态（实测）**：HF 缓存目录存在但**只有 config 与 tokenizer（22 MB），没有权重**（`blobs` 为 0 字节）。须下载约 2.2 GB。加载走环境已有的 `transformers`，**不需要装 `FlagEmbedding`**。运行设备为 CPU（`torch 2.12.1+cpu`，无 CUDA）。
- 运行环境：`D:/zg6_Project/9/med_rag/rag/python.exe`（Python 3.12.14），不新建虚拟环境（宪法原则 I）。
- 输入产物已通过 `specs/002` 的验收（23/23 通过）。
- `source_hash` 可从 `doc_id` 反查——`doc_id` 即 PDF 内容 sha256 前 12 位，但**完整 64 位需要重新计算或从上游读取**。若上游未保存完整哈希，本步骤可能需要读 `data/parsed/` 下的源 PDF 重算，或退化为记录 `doc_id`。**这是实现阶段的待查项**。
- 本特性只做 S4 分块，不做向量化、入库。

---

## 已裁决决定（人工裁决于 2026-09-23）

### D1 — 目标长度与跨章节合并：**选项 D**

**决定**：目标 **500–700 字**；**有限跨章节合并** —— 仅当两个相邻 section **同属一个父章节**（`heading_path[:-1]` 相同）、且合并后不超上限时，才允许跨 section 合并。

- `docs/04 §7` 原定的「300–500 字」与「跨章节合并禁止」**作废并须回写文档**。
- 父章节不同时，**硬边界**，绝不合并。
- 即使同父章节，合并后若超 700 字，**不合并**，各自成块。

### D2 — 「按语义分块」的含义：**选项 B（嵌入式）**

**决定**：用 **BGE-M3 稠密向量**计算相邻块的余弦相似度，在**语义断崖**处切分。

**关键实现约束**：

- **加载方式**：直接用环境里已有的 `transformers` 加载 `AutoModel`（BGE-M3 是标准 `XLMRobertaModel`，隐藏层 1024 维）。**不引入 `FlagEmbedding`** —— 少装一个包，少一层版本耦合。
- **稠密向量**：取 `last_hidden_state[:, 0]`（CLS）+ **L2 归一化**，与 `docs/04 §8` 的向量化口径保持一致。
- **只用于分块决策**，MUST NOT 落盘、MUST NOT 传给 S5（S5 会自己再算一遍，避免跨步骤的向量口径耦合）。
- **输入文本**：用 `heading_path 拼接 + 块正文`（即 chunk 的 `text_for_embedding` 口径）。理由：脱离标题的段落语义模糊（`docs/04 §7`），算相似度时同样如此。
- **设备**：CPU（实测 `torch 2.12.1+cpu`，无 CUDA）。27,115 字约 180 个块，CPU 上应在 1 分钟内完成。
- **漂移风险**：分块结果从此**依赖模型版本**。MUST 把模型指纹写入 `chunk_rule_version`，模型一变即视为需全量重建（`docs/04 §8` 的"向量空间漂移"风险由此提前到分块阶段）。**这是选 B 的已知代价。**

### D3 — 裁决机制：**选项 A**

**决定**：待裁决项写 `data/chunks/{doc_id}.decisions.jsonl`，回答写 `data/chunks/{doc_id}.answers.json`，存在未答复项时**非 0 退出码**。

**「什么算不确定」—— 采用规格中提出的四类**（用户未提出修改）：

1. section 字数不足下限的 60%，合并它必须跨章节
2. 某块在句子中间结束，且找不到可续接的相邻块
3. 表格块超过上限（实测表4 = 3,160 字）
4. 同 section 内相邻块的余弦相似度落在**阈值 ± 容差带**内 —— 即"切不切都说得过去"

**第 4 类是对 D2 的必然补充**：既然判定依据是相似度阈值，那么**贴近阈值**的边界就是天然的不确定区，必须交人裁决，MUST NOT 由脚本按 `> 阈值?` 硬判。默认容差带 `±0.05`。
