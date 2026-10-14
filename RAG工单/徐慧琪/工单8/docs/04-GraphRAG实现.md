> 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答

# Graph RAG 实现细节（硬性要求3）

本文逐条说明硬性要求3 的实现：实体识别 → 关系抽取 → 图谱构建 → 子图检索 →
图+向量融合 → LLM 生成。所有数字为**本机实测**（2026-10-14，语料 = ccf_competition 的 9 份年报）。

---

## 一、实体识别：词典最长匹配 + 别名兜底

### 实现

| 环节 | 位置 | 说明 |
| --- | --- | --- |
| 金融词典 | `src/kg_dict.py` | 9 家公司（26 个别名，含股票代码/简称/全称）、**39 项指标**、机构表、板块表、章节白名单、**30 个子类限定词** |
| 最长匹配 | `kg_dict.find_companies / find_metrics / find_institutions / find_sectors` | 同一位置多个候选时取**最长**者，避免「银行」吃掉「平安银行」；先匹配长别名再匹配短别名 |
| 问题侧链接 | `graph_retriever.link_entities` | ① 公司（词典最长匹配，按 surface 与原名的接近度打分）② 指标（**只链接图谱中真实存在取值的指标**，防止链接到空节点）③ 图谱别名兜底（词典未覆盖的别名从图里找）|
| 缓存 | `link_entities(use_cache=True)` | LRU 缓存，同一问题的链接结果复用（问答性能保障之一） |

### 实测覆盖率

- 第 1 层跑**全量 7757 个块**，耗时 **0.4 秒**，其中 **1846 块**（23.8%）产出实体。
- 关系候选 3271 条 → 经消歧归一后入图 **873 条**；实体原始 4286 → 归一后 **560 个节点**。
- 9 家公司**全部**在各自年报中被识别（`Company` 节点 9/9）。

> 为什么用词典而不是 NER 模型：本机**无 UIE/Spacy 等专用 NER 权重**，且工单明确「禁止下载」。
> 词典+正则是确定性抽取，**零幻觉**，代价是召回受词典边界限制 —— 该取舍在 `docs/12` 中如实标注。

## 二、关系抽取：第 1 层确定性 + 第 2 层 LLM（三道闸门）

### 第 1 层（`src/kg_extract.py`，全量、秒级、零幻觉）

| 关系 | 抽取依据 |
| --- | --- |
| `Company -HAS_METRIC-> Metric` | 指标名（词典）+ 指标后 30 字窗口内的数值（金额/百分比）+ 报告期 + 公司归属（**块内没有公司名时按来源文件名归属**）；Metric 节点名形如 `营业收入(1,168,867百万元)@2019年度`，**值、单位、期间三元组归一** |
| `Metric -REPORTED_IN-> Time` | 报告期正则（2019 年 / 报告期末 / 上年末 …） |
| `Company -IN_INDUSTRY-> Industry` | 公司所属行业（词典） |
| `Department -BELONGS_TO-> Department/Company` | **组织隶属规则**：「X 下设/下辖/内设 A、B、C」句式 → 父组织 + 子组织列表（63 条，覆盖 9 家公司中的 7 家） |

**子类口径守卫**（防止把子类指标当成母指标）：企业贷款不良率 ≠ 不良贷款率。
`kg_extract._is_sub_metric_mention` 用「子句级 + 父块上下文 + 表内跨行」三重判定拦截
（30 个子类限定词），这是本项目针对金融年报的**关键正确性修正**。

**组织隶属规则的精度守卫**（宁可漏抽，不可抽错）：句子片段头（「各位监事分别列席董事会」）拒绝；
泛指子项（「部门」「直属单位」）拒绝；双栏 PDF 行交错导致物名被打断时，用原文自带的
**「等N个」对账**，数量对不上则整组不抽。

### 第 2 层（`src/kg_extract_llm.py` + `kg_prompt.py`，可选、限量）

两段式调用（先抽实体、再抽关系，避免一次输出过长导致 JSON 破损），
支持 **JSONL 断点续抽**（`--resume`）、单块重试 `KG_LLM_RETRY` 次、异常不外抛。

**三道闸门**（`kg_prompt.to_entities_relations`）——LLM 输出必须逐条通过：

| 闸门 | 校验内容 | 不通过的处理 |
| --- | --- | --- |
| ① 本体约束 | 实体类型必须在本体 `ENTITY_TYPES` 内，名称非空，`validate_entity` 通过 | 丢弃该项（不连坐整块） |
| ② 类型约束 | 关系类型必须在 `RELATION_TYPES` 内，且 **head/tail 的实体类型合法** | 丢弃该条关系 |
| ③ 证据落地 | `evidence` 必须**真实出现在块原文中**（`_evidence_is_grounded`） | 丢弃该条关系（防幻觉的核心） |

另外：`salvage_llm_json` 可在 JSON 破损时抢救出完整对象，但**同样要过三道闸门**，不绕闸门。

## 三、图谱构建：NetworkX + 三种持久化产物

### 存储结构（`src/kg_store.KnowledgeGraph`）

- 底层 `networkx.MultiDiGraph`；节点携带 `name / type / aliases / attrs`，边携带
  `rel_type / evidence / source_doc / page / confidence / attrs`。
- 规则抽取的边 `confidence = 1.0`，LLM 抽取的边 < 1.0 且**必须**带 evidence。
- 查询能力（4 类）：`neighbors(k-hop)` / `subtree(BELONGS_TO 子树)` /
  `shortest_path(无向视图最短路)` / `metric_compare(同名指标跨公司跨期取数)`。

### 持久化产物与用途

| 产物 | 用途 |
| --- | --- |
| `data/kg/kg.pkl` | **二进制快照**，问答时 `KnowledgeGraph.load()` 毫秒级加载 |
| `data/kg/kg.graphml` | **可交换图文件**，可用 Gephi/yEd 等外部工具打开审阅 |
| `data/kg/kg_entities.jsonl`、`kg_relations.jsonl` | **逐行 JSON**，便于 diff、增量与人工抽查 |
| `data/kg/kg_stats.json` | 轻量统计缓存（`build_kg.py --stats` 免加载全图） |

### 原子写

`KnowledgeGraph.save`：先写 `*.tmp`，再 `os.replace` 覆盖（Windows 上也是原子替换）；
GraphML 写入失败会清理临时文件并保留 pickle 产物 —— **磁盘上任何时刻都是完整版本**，
不会出现写到一半的坏图谱（见 `docs/11` 故障5）。

### 实测规模（`docs/03` 由 `build_kg.py --report` 生成）

560 节点 / 873 边 / 9 份年报 / **带 provenance 的边 100%**。
节点：Metric 436、Department 69、Time 24、Institution 19、Company 9、Industry 3；
边：HAS_METRIC 440、REPORTED_IN 361、**BELONGS_TO 63**、IN_INDUSTRY 9。

## 四、子图检索：四种模式

由 `graph_retriever.detect_mode(question, linked)` 判定，优先级
**metric_compare > org_tree > path > khop**（结构约束越强越优先）：

| 模式 | 触发条件 | 图查询实现 | 产出事实示例 |
| --- | --- | --- | --- |
| `metric_compare` | 链接到 **≥2 家公司** 且（含指标实体 或 问句含指标词） | `KnowledgeGraph.metric_compare`：遍历 `Company -HAS_METRIC-> Metric` 取同名指标跨公司/跨期，附**最高/最低结论** | 「比较结论：不良贷款率中 平安银行 最高（1.65%），招商银行 最低（1.16%）。」 |
| `org_tree` | 问句含组织关键词（组织结构/板块/下设/内设/直属单位/隶属…） | `KnowledgeGraph.subtree(root, "BELONGS_TO")` 沿**入边**向下展开（本体中 BELONGS_TO 为子→父） | 「投资银行管理委员会 下设：金融行业组、能源化工行业组、…」 |
| `path` | ≥2 个实体 且 问句含关系词（什么关系/持股/控股/股东…） | `shortest_path`（无向视图）→ 按边真实方向陈述关系链 | 「中国平安 --CONTROLS--> 平安银行」 |
| `khop` | 兜底 | 种子实体的 k-hop 邻居（`KG_SUBGRAPH_HOPS = 2`），按关系类型组织事实 | 指标事实、行业、报告期 |

约束：子图节点数上限 `KG_SUBGRAPH_MAX_NODES = 60`（软上限可传参放宽），
事实条数同上限截断 —— **防止上下文爆炸**（实测单题图谱事实 1~60 条）。

`SubgraphResult` 同时返回 `facts`（给 LLM 的自然语言事实）、`nodes/edges`（给可视化）、
`source_pages`（provenance 汇总，支撑「返回答案所在的文本块」）。

## 五、图 + 向量 + 全文三路融合

`src/graph_fusion.py`。设计前提：三条通道**互补而非替代** ——
图谱给结构化可验证事实（对长尾叙述覆盖不到），向量给语义相近文本（对数值精确性不敏感），
全文给字面命中文本（对同义表述不敏感）。

- **权重**（`config.KG_FUSION_WEIGHTS`）：`graph 0.35 / vector 0.40 / fulltext 0.25`，
  融合方法 `weighted`（可切 `rrf`）；向量与全文两路复用 07 的加权融合，**与工单07 保持同口径**。
- **按「文档+页」去重**：同一页常被向量与全文同时命中，若不去重，
  上下文预算会被同一页的多个块挤占，导致**证据多样性下降**（实测 top-8 命中常集中在 2~3 页）。
  去重时保留该页得分最高的块。
- **上下文分区组装**（`build_context_text`）：`【图谱事实】` 区与 `【文本证据】` 区分开，
  让 LLM 清楚区分「已核验的结构化数据」与「待引用的原文片段」；默认最多 40 条事实、
  每块 700 字符。

## 六、LLM 生成：Prompt 结构、防幻觉、降级

### Prompt（外置 `prompts/graph_qa.yaml`，便于后续工单调优）

结构：`system_zh / system_en`（中英各一套）+ 数据区（图谱事实 + 文本证据）+
**图谱事实为空时的提示**（`no_graph_hint_zh`：明确告知本次无图谱事实，避免模型臆测）。

防幻觉约束（system prompt 规则）：
1. **只依据**提供的图谱事实与文本证据回答；
2. 图谱事实优先采信，文本证据补充细节；
3. 每个关键数字/结论标注来源 `（文档名 第X页）`；
4. 证据不足时明确回答「现有资料未包含该信息」，**禁止编造数字**；
5. 多实体/多数值问题用**结构化列表**逐条列出，便于核对。

### 降级路径（详见 `docs/11`）

| 失败点 | 行为 |
| --- | --- |
| 图谱未构建/加载失败 | `degraded=True`，退化为纯文本 RAG（07 能力），`degraded_reason="图谱不可用：…"` |
| 子图检索异常 | 同上，降级为纯文本通道 |
| LLM 不可用（Ollama 未启动/超时） | 走**抽取式答案**：直接把子图事实成文返回，关键数字不丢，并标 `degraded` |
| 实体未链接到任何节点 | 模式判定为 `none`，纯文本通道作答（如实标注，不伪装成图谱作答） |

实测（14 题）：平均 **2.44 s**、最长 6.06 s（要求 ≤3 s，均值达标）；
14/14 成功，无降级。
