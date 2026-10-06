# PDF 文档的表格解析及检索优化（工单 03）

**工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化**

在工单 01/02 的问答系统之上，完成本工单的两个关键变化：

1. **多文档**：新增《招股说明书2.pdf》（武汉力源信息技术股份有限公司，350 页，创业板），
   系统同时索引《招股说明书1.pdf》（武汉兴图新科电子股份有限公司，548 页，科创板）；
2. **表格作为一等公民**：表格不再"整表塞进一个块"，而是**结构化**为「表头块 + 行级块」，
   检索能定位到「哪个 PDF / 哪一页 / 哪个表 / 哪一行」，生成时把**整表**喂给 LLM。

目标：14 道必测题（id 1~4 的答案**全在表格里**）回答准确率 ≥90%、响应时间 ≤3 秒。

### 实测结果（本机 RTX 4060 Laptop）

| 指标 | 工单03（表格优化） | 工单03 消融（无表格优化） | 工单01/02 基线 | 纯 LLM |
| --- | --- | --- | --- | --- |
| **14 题全对数量** | **13 / 14** | 11 / 14 | 7 / 14 | 0 / 14 |
| LLM 裁判准确率（0/0.5/1） | **0.821** | 0.75 | 0.679 | 0.5 |
| 关键事实命中率 | **0.929** | 0.837 | 0.633 | 0.036 |
| **表格题 id 1~4 事实命中率** | **1.000** | **0.678** | 0.05 | 0.00 |
| 平均响应时间 | 2.48s（中位 2.47s） | 2.32s | 7.67s | 0.85s |
| ≤3 秒占比 | 0.79 | 0.71 | 0.00 | 1.00 |
| Hit Rate@5 / MRR@5 / NDCG@5 | 1.00 / 0.929 / 0.674 | 1.00 / 0.917 / 0.651 | 0.857 / 0.714 / 0.531 | — |

> 「消融」= **同一份索引**，仅把表格结构化块从检索中排除，因此它量化的是表格结构化的
> **净增益**：表格题 **1.000 − 0.678 = +0.322（+32 个百分点）**，
> 不掺入分词、模型、语料规模的差异。
> 检索侧另有 `docs/06-答案定位报告.md`：14 题的关键事实 **14/14 全部被检索到（100%）**，
> 逐题给出所在 PDF / 页码 / 表格 / 行号。
> 完整数据与逐题明细见 `docs/03-优化前后对比分析.md`。

> **硬约束**：全流程只使用本机已下载的模型，任何环节都不触发模型下载。

---

## 快速开始

```bash
# 1) 确认 Ollama 在跑且本机已有 qwen2.5:3b
ollama list

# 2) 建库（双文档：解析 → 表格结构化 → 向量化 → Qdrant）
python build_index.py

# 3) 启动中英双语界面
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可提问（中文/英文），回答带**来源文档 + 页码**，
表格题额外给出**表格名与命中的行号**。

---

## 系统架构

```
【离线建库】
  招股说明书1.pdf（复用已有 MinerU 产物）      招股说明书2.pdf（本工单新解析，239s）
        └────────────────────┬────────────────────────┘
                             ▼
     MinerU 3.4.5 pipeline（独立 venv D:\model\mineru-venv，模型本地快照）
       content_list.json：text / table(table_body=HTML) / image / header …
                             ▼
   ┌── 文本链路（沿用工单02）──────┐  ┌── 表格链路（★工单03 新增）─────────────┐
   │ 标题层级 + 父子块              │  │ table_parser   HTML→矩阵(rowspan/colspan)│
   │ 子块 300–450 字（检索）        │  │   ├ 表头识别 / 字段-值表识别             │
   │ 父块 ≤1800 字（上下文）        │  │   └ 质量分 → 低质量表走 PaddleOCR-VL 兜底 │
   │                               │  │ table_chunker  表头块 + 行级块            │
   │                               │  │   行级描述："表：X | 发行股数：1,670万股" │
   └───────────┬───────────────────┘  └───────────────┬────────────────────────┘
               ▼                                      ▼
      embedder bge-m3（D:\model\bge-m3，1024 维，CUDA，local_files_only）
               ▼
      Qdrant（本机无 Docker → local 嵌入式模式；collection `zhaogu_v3_table`）
      payload：source / block_type / page_idx / heading_path /
               table_id / table_html / table_header / row_index …

【在线问答】
  提问（中文/英文）
   └─ doc_router ★新   公司名 → 文档路由（"武汉力源"→招股2，"兴图新科"→招股1）
       └─ query_rewrite  意图/消歧/分解/扩展（规则版，<1ms）
           └─ 混合检索  向量 top-30 + BM25 top-30 → RRF(k=60) 融合
               └─ 精排  bge-reranker-v2-m3（top-20 → 6）
                   └─ context ★改  命中表格行 → 按 table_id 还原**整表**喂 LLM
                       └─ 生成  qwen2.5:3b + 三重校验重试 + 抽取式降级
                           → 答案 + [n] 引用 + 表格定位（PDF/页/表/行）
```

**生成后三重校验 → 定向重试一次**（工单3 针对表格题新增，均为实测失败模式驱动）：
① 数字：答案里的数字必须在片段中，或由**答案自己写出的两个数**一步算出（表格题的比例）；
② 表格行：列举型问题漏列了表格中的行（如 id 4 漏掉 2 家企业）；
③ 书名号名称：与片段不一致的名称简写（如把标准全称写短）。

**关键设计**：表格复用 01/02 已有的「子块检索 + 父块上下文」机制——
`parent_id = table_id`，`child.text = 行级描述`，`parent_text = 整表 Markdown`。
于是 `context.expand_parents` 无需特判即可把"命中某一行"自动还原为"整张表"。

---

## 目录结构

```
工单3/
├── src/
│   ├── bootstrap.py        # 离线环境 + Windows 栈溢出修复（必须最先导入）
│   ├── config.py           # 全部配置（多文档 / 表格参数 / 模型路径）
│   ├── pdf_parser.py       # 解析入口（MinerU 可选 → PyMuPDF 增强回退，保留表格 HTML）
│   ├── table_parser.py     # ★新 表格 HTML → 结构（矩阵/表头/字段-值表/质量分/OCR 兜底）
│   ├── table_chunker.py    # ★新 表格 → 表头块 + 行级块（含 8 项元数据）
│   ├── doc_router.py       # ★新 多文档路由（公司名 → PDF）
│   ├── answer_location.py  # ★新 答案定位报告（14 题 → PDF/页/表/行）
│   ├── chunker.py          # 标题层级 + 父子块（表格转交 table_chunker）
│   ├── pymupdf_parser.py   # PyMuPDF 五遍扫描引擎（回退）
│   ├── ocr_fallback.py     # PaddleOCR-VL 兜底（独立 venv 子进程）
│   ├── embedder.py         # bge-m3 向量化
│   ├── vector_store.py     # Qdrant（local/server 双模式；表格元数据入 payload）
│   ├── retriever.py        # 向量+BM25+RRF 混合检索（支持 source / 块类型过滤）
│   ├── reranker.py         # bge-reranker-v2-m3 精排
│   ├── query_rewrite.py    # 查询改写（意图/消歧/分解/扩展）
│   ├── context.py          # 上下文组装（表格按 table_id 还原整表）
│   ├── llm.py              # Ollama 封装
│   ├── rag.py              # 全链路编排（路由/计时/缓存/数字校验/降级）
│   └── baseline.py         # 「无表格优化」基线桥接（只读调用工单1）
├── build_index.py          # 建库（多文档 + 表格结构化 + OCR 兜底）
├── app.py                  # Streamlit 中英双语界面（问答/对比/表格/知识库）
├── evaluate_compare.py     # 四路对比（03 / 03消融 / 01·02基线 / 纯LLM）
├── evaluate_ragas.py       # RAGAS 评估（本地裁判，离线）
├── scripts/                # MinerU 调用、基线子进程
├── tests/                  # pytest 单元测试
├── data/                   # 解析/表格/分块/向量库/评估产物
└── docs/                   # 见下方文档索引
```

---

## 主要命令

| 命令 | 说明 |
| --- | --- |
| `python build_index.py` | 双文档建库（解析→表格结构化→向量化→入库） |
| `python build_index.py --rebuild` | 强制重新解析 |
| `python build_index.py --source 招股说明书2.pdf` | 只建某个文档 |
| `streamlit run app.py` | 启动双语问答界面 |
| `python evaluate_compare.py` | 四路对比评估（含表格消融） |
| `python evaluate_compare.py --skip-baseline` | 跳过工单1子进程（更快） |
| `python evaluate_ragas.py` | RAGAS 评估（本地裁判） |
| `python -m src.answer_location` | 生成答案定位报告 |
| `python -m pytest tests/ -v -m "not slow"` | 单元测试 |

---

## 表格专项：为什么这样设计

| 工单要求 | 本项目实现 |
| --- | --- |
| 表格识别，保留 MinerU 的 HTML | `pdf_parser.normalize_item` 保留 `table_html`；`table_parser` 解析 |
| 表头必须单独提取 | 表头块（`table_header`）独立入库，支持"按列名检索" |
| 表格结构化表示，不直接丢 HTML | 行级自然语言描述：`表：本次发行概况 \| 发行股数：1,670万股 \| 页码：3` |
| 原始 HTML 作为元数据存 payload | 每个表格 chunk 都带 `table_html`（BM25 语料构建时排除以省内存） |
| 表格元数据 8 项 | `block_type/page_idx/source/table_id/table_html/table_header/row_index/heading_path` |
| 识别质量差的表格用 PaddleOCR-VL 兜底 | `table_parser.ocr_fallback_tables`，按页去重、限量、超时可退 |
| 检索结果可追溯到页/表/行 | `citation_map` 输出 `table_id/table_caption/row_indices` |
| 生成时把表格结构一起喂进去 | `context.expand_parents` 命中行 → 还原整表（表头+全部行），不做行级裁剪 |

**实测发现并修正的两个坑**（详见 `docs/05-表格解析与检索专项说明.md`）：

1. 招股书的「本次发行概况」是**字段-值表**（2 列，没有列名）。若把首行当列名，
   列名会变成"发行股票类型/人民币普通股(A股)"，**后续每一行的描述全部错位**。
   → 新增 `kv` 模式识别（2 列表 + "释义表"的 `指` 连接词列）。
2. 财务表的日期表头（`2010-6-30 | 2009-12-31`）会被宽松的数值正则当成数据行，
   导致列名退化成"列1/列2/列3"。→ 收紧数值判定为正则 `^-?[\d,]+(?:\.\d+)?%?$`。

---

## 文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/03-优化前后对比分析.md` | 四路对比表（准确率/时间/HitRate/MRR/NDCG），重点对比表格题 |
| `docs/04-RAGAS评估报告.md` | RAGAS 四指标 |
| `docs/05-表格解析与检索专项说明.md` | 表格如何识别/结构化/入库/检索 + 踩坑记录 |
| `docs/06-答案定位报告.md` | 14 题逐题的 PDF / 页码 / 表格 / 行号定位 |
| `docs/07-演示录制脚本.md` | 演示视频的分镜与口播（视频需自行录制） |
| `docs/08-工单验收对照表.md` | **逐条对照工单的验收结果**（达标/部分达标/未达标，含实测数据与未达标原因） |
| `docs/技术文档.md` | 架构、选型、开发流程、使用方法、工程问题记录 |
| `docs/用户手册.md` | 界面操作、上传文档、常见问题 |
