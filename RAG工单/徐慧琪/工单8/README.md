# 工单08：基于 Graph RAG 实现金融问答

> **工单编号**：人工智能NLP-RAG-基于GraphRAG实现金融问答
> **任务**：基于附件 `ccf_competition.zip` 的 9 份上市公司年度报告构建**金融知识图谱**，
> 在工单07 的混合检索 RAG 之上叠加图谱能力，实现 **Graph RAG 问答**与**金融研报生成**，
> 并与工单07 逐题对比（硬性要求4）。

---

## 一、这个工单做了什么

| 环节 | 结果 |
| --- | --- |
| 模型扫描（硬性要求0） | 复用本机已有权重，**零下载** → `docs/01` |
| 知识图谱构建 | 第 1 层词典+正则（全量 7757 块，**0.4 秒**）+ 第 2 层 LLM（可选、断点续抽）→ **560 节点 / 873 边 / 9 份文档，带 provenance 的边 100%** → `docs/03` |
| 实体/关系抽取 | 本体 10 类实体 + 11 类关系；实测入图 6 类实体（Metric/Department/Time/Institution/Company/Industry）与 4 类关系（HAS_METRIC/REPORTED_IN/**BELONGS_TO**/IN_INDUSTRY）→ `docs/04` |
| 子图检索 | 四模式：`metric_compare` / `org_tree` / `path` / `khop`，按问句自动判定 → `docs/04` |
| Graph RAG 问答 | 图谱事实 + 向量召回 + 全文召回三路融合 → Reranker 精排 → LLM 生成 → `docs/04` |
| **评测** | `eval_question.md` 14 题全部跑通，**平均 2.44 秒**，无降级 → `docs/05` |
| **与 07 对比** | 关键事实覆盖率 **0.385 vs 0.342**（可比题）；更优 3 / 持平 5 / 落后 2，并**如实列出图谱覆盖缺口** → `docs/06` |
| 可视化与界面 | Streamlit 六页签：问答 / 研报 / 图谱可视化 / 实体关系 / 评测 / 状态；vis.js 交互 + matplotlib 离线兜底 → `docs/07` |
| **上传文档问答** | 问答页可上传自己的 PDF，走会话级索引（向量+BM25 混合 → Reranker 精排），答案带页码；不写入知识图谱 → `docs/09` |
| 容错与长时运行 | 8 类故障清单 + 日志轮转 + 内存监控 + 断点续抽 + 健康检查 → `docs/11` |

**核心结论**：Graph RAG 的增量集中在**多跳对比**与**组织结构**类问题上
（如 Q09 对比平安/招行的不良贷款率：0.67 → 1.00；Q13 中信证券组织结构由图谱事实作答）；
对单点事实型问题两种方案接近 —— **图谱的价值不在替换，而在补上「关系」这条腿**。
未达标项与受限度已逐条标注在 `docs/12` 第二节。

---

## 二、复用了什么

**复用 01~07 的 29 个模块，检索核心零改动**：PDF 解析、分块、嵌入（bge-m3）、
Qdrant、BM25 倒排、Reranker（bge-reranker-v2-m3）、Query 理解、多轮对话、LLM 调用封装。

**本工单新增 19 个模块**：`kg_schema` / `kg_dict` / `kg_extract` / `kg_extract_llm` /
`kg_prompt` / `kg_select` / `kg_dedup` / `kg_store` / `kg_build` / `kg_report` /
`graph_retriever` / `graph_fusion` / `graph_qa` / `graph_viz` / `speech` / `ui_helpers` /
`runtime_ext` / `compare_with_wo07` / `upload_index`。

> 复用模块的文件头**保留上游工单编号**，本工单在其后追加一行本工单编号（标明来源，不删原注释）。
> 复用的意义：Q1~Q10 与工单07 **同源同金标准 + 同一底层检索**，因此两份结果的差异
> 可以归因到图谱本身。

---

## 三、快速开始

```bash
cd 工单8

# ① 环境确认（只读、零下载）
python scan_local_models.py                      # → docs/01
python -c "from src import bootstrap, llm; print(llm.health())"

# ② 解析 + 分块 + 建索引（复用 07 链路）
python build_index.py

# ③ 构建知识图谱（第 1 层：秒级）
python build_kg.py --layer1                      # → 560 节点 / 873 边
python build_kg.py --report                      # → docs/03

# ③'（可选）第 2 层 LLM 抽取：耗时长，务必带 --resume 断点续抽
python build_kg.py --all --resume

# ④ 批量评测（14 题，约 1 分钟）→ data/eval/graph_qa_results.json
python graph_qa_cli.py --all

# ⑤ 与工单07 对比 → docs/05 + docs/06 + data/eval/compare_wo07.json
python src/compare_with_wo07.py

# ⑥ 健康检查与测试
python -c "from src.runtime_ext import health_check; import json; print(json.dumps(health_check(), ensure_ascii=False, indent=2))"
python -m pytest tests/ -q                       # 251 passed
```

界面（中英双语 / 语音输入 / 图谱可视化 / 研报生成）：

```bash
ollama serve &          # 确认本地大模型在跑
streamlit run app.py    # → http://localhost:8501（六页签）
```

---

## 四、交付物索引

| 类别 | 文件 | 说明 |
| --- | --- | --- |
| 模型扫描 | `docs/01-本机模型扫描报告.md` | 已有模型 + 缺失模型 + 零下载确认 |
| 系统架构 | `docs/02-系统架构设计.md` | Graph RAG 流程图、模块划分、技术选型理由、复用边界 |
| 图谱构建 | `docs/03-知识图谱构建.md` | 规模、类型分布、各文档贡献（脚本生成） |
| 实现细节 | `docs/04-GraphRAG实现.md` | 实体识别/关系抽取/图谱构建/子图检索/融合/生成 逐条对应代码 |
| 检索结果 | `docs/05-eval_question检索结果.md` | 14 题逐题：问题 + 模式 + 图谱事实 + 答案 |
| 对比分析 | `docs/06-与07工单对比分析.md` | 逐题覆盖率 + 优势不足归因 + **覆盖缺口清单** |
| 可视化说明 | `docs/07-知识图谱可视化说明.md` | 两种渲染方式、配色表、交互、provenance 解读 |
| 技术文档 | `docs/08-技术文档.md` | 环境、构建流程、配置项、模块 API、二次开发 |
| 用户手册 | `docs/09-用户手册.md` | 启动、提问（中英/语音）、看结果、看图谱、研报、排障 |
| 视频说明 | `docs/10-演示视频说明.md` | 两个演示视频的分镜脚本与录制建议 |
| 容错与长时运行 | `docs/11-容错机制与长时间运行方案.md` | 8 类故障清单 + 日志轮转/内存监控/续抽 + 降级标注 |
| 验收对照 | `docs/12-工单硬性要求验收对照表.md` | 逐条对照 + **未达标项如实标注** |
| 代码 | `build_kg.py`、`graph_qa_cli.py`、`src/`（48 个模块） | 图谱构建 CLI、评测 CLI、Graph RAG 全链路 |
| 数据产物 | `data/kg/`、`data/eval/graph_qa_results.json`、`data/eval/compare_wo07.json` | 图谱、逐题结果、对比结果 |

---

## 五、关键设计取舍（一句话版）

- **自研编排而非 LangChain**：LangChain 的图问答链只对接 Neo4j/Cypher，用 NetworkX 反而要写适配器；
  自研可最大化复用 01~07，并保证与工单07 逐题严格可比。
- **NetworkX 而非 Neo4j**：符合「零下载」硬约束，本工程全部图查询（k-hop/子树/最短路/指标对比）
  已由 `kg_store.py` 覆盖。
- **分层抽取**：第 1 层词典+正则作**可信骨架**（确定性、零幻觉、秒级）；第 2 层 LLM 只在不确信的关系上
  补充，且必须过**三道闸门**（本体 / 类型 / evidence 落地）。
- **宁可漏抽不可抽错**：组织关系抽取对句子片段头、泛指子项、双栏 PDF 行交错（用「等N个」对账）
  一律拒绝 —— 错误的树比没有树更糟。
