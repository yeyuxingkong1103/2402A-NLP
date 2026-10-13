# 工单07：功能测试及评估

> **工单编号**：人工智能NLP-RAG-功能测试及评估
> **任务**：基于 01~06 工单实现的 RAG 系统，对附件 `ccf_competition.zip` 中的
> 9 份上市公司年度报告（2812 页）完成 **RAG 检索测试与评估**，输出 10 个问题的
> 检索结果、量化评估指标与**逐题归因分析**。

---

## 一、这个工单做了什么

| 环节 | 结果 |
| --- | --- |
| 模型扫描（硬性要求0） | 四类角色本机**零缺口**，**未下载任何模型** → `docs/01` |
| 语料准备 | 解压附件（还原 GBK 中文文件名）→ 9 份年报 / 2812 页 → `data/corpus/` |
| 问题整理 | 由 `sample_questions.pdf` 整理出 **10 个可核验的测试问题**，覆盖 5 种题型 → `docs/02` |
| 建库 | PyMuPDF 解析 → **7225 块**（text 5124 / table 2101）→ Qdrant + 自研倒排 + BM25 |
| RAG 测试 | 10 题 × **3 种检索策略**（vector / fulltext / hybrid）全量留痕 → `docs/04` |
| 指标评估 | Precision@k / Recall@k / Hit Rate / MRR / NDCG@k / Answer Accuracy → `docs/05` |
| RAGAS 评估 | Faithfulness / Answer Relevancy / Context Precision / Recall → `docs/07` |
| **归因分析** | 逐题定位失败环节，归纳 7 类失败模式，给 P0/P1/P2 改进建议 → **`docs/06`** |

**核心结论**：混合检索在 6 项检索指标中 5 项最优（事实覆盖 0.680 vs vector 0.480 /
fulltext 0.400）。主要缺陷集中在**精排排序**、**上下文组装丢证据**、
**图像解析未泛化**、**口径选择**四处，均已定位到具体上游工单与改进建议。

---

## 二、被测系统：复用 01~06，不重写检索逻辑

`src/`（30 个模块）**原样复用**上游 01~06 实现的 RAG 系统，
**检索核心零改动**（`strategies.py` / `fulltext.py` / `fusion.py` / `rerankers.py` /
`embedder.py` / `vector_store.py`）。

本工单的改动**仅限语料配置**，逐条列于 `docs/03-测试环境说明.md` 第一节：
`config.py`（9 家公司语料 / collection 名 / 关闭招股书专用组织图模块）、
`dialogue.py`（公司映射改为引用 config）、`rag.py`（提示词的语料身份与年报取数规则）、
`tokenizer.py`（领域保护词）、`pdf_parser.py`（页眉噪声词）、
`pymupdf_parser.py`（**表格抽取修复**，见下）。

> **表格抽取修复**：上游 PyMuPDF 默认策略在年报上会把整张表压成 2~3 列、行值错位
> （实测「主要会计数据」36 行 × 3 列且内容错乱）。已改为 `strategy="text"` +
> 网格清洗 + 单位抽取，修复后同一张表 **40 行 × 7 列**，
> 数值与单位与 PDF 原文一致。这是"让复用系统能被正确测试"的必要前提。

---

## 三、快速开始

```bash
cd 工单7

python scan_local_models.py                     # ① 模型扫描（只读、零下载）→ docs/01
python scripts/setup_project.py                 # ② 解压附件 + 语料规范化 + 自检

python build_index.py --rebuild --fallback-only --no-ocr --no-embed --skip-orgchart
                                                # ③ 解析 9 份年报 → 分块（约 5 分钟）
python build_index.py --no-ocr --skip-orgchart  # ④ 向量化 + 建索引（约 3 分钟）

python scripts/validate_ground_truth.py         # ⑤ 金标准闸门（评估前必跑）
python run_rag_questions.py --strategy hybrid   # ⑥ 10 题留痕 → docs/04
python run_rag_questions.py --strategy vector
python run_rag_questions.py --strategy fulltext
python evaluate_rag07.py                        # ⑦ 指标汇总 → docs/05
python evaluate_ragas.py --strategy hybrid      # ⑧ RAGAS → docs/07
```

调试用界面（中英双语 / 策略切换 / 多轮对话）：

```bash
streamlit run app.py
```

---

## 四、交付物索引

| 文档 | 内容 |
| --- | --- |
| `docs/01-本机模型扫描结果.md` | 已有模型清单 + 缺失分析（零缺口） |
| `docs/02-数据准备与问题整理.md` | 语料清单 + 10 个问题 + 金标准自检记录 |
| `docs/03-测试环境说明.md` | 系统复用说明 + 策略/Reranker 配置 + 解析选型 |
| `docs/04-RAG检索结果-{hybrid,vector,fulltext}.md` | **逐题**：Top-K 片段原文 + 引用来源 + 答案 + 耗时 |
| `docs/05-RAG评估结果.md` | **逐题指标** + 策略汇总表 + 关键事实命中明细 |
| **`docs/06-检索结果问题分析报告.md`** | **重点**：逐题归因 + 7 类失败模式 + P0/P1/P2 改进建议 |
| `docs/07-RAGAS评估报告.md` | 四项 RAGAS 指标 + 与检索指标的交叉验证 |
| `docs/08-技术文档.md` | 测试方法 + 指标定义 + 使用方法 + 增量扩展 |
| `docs/09-容错机制与长时间运行方案.md` | 六类容错清单 + 长跑方案 |
| `docs/10-工单硬性要求验收对照表.md` | 8 条硬性要求 + 输出格式 + 验收标准逐条对照 |

**代码**

| 文件 | 作用 |
| --- | --- |
| `scripts/setup_project.py` | 解压附件（cp437→gbk 还原中文名）+ 语料规范化 + 工程自检 |
| `scripts/validate_ground_truth.py` | 金标准自检闸门（防"金标写错导致指标假性偏低"） |
| `scripts/grep_corpus.py` | 语料关键词检索（人工核对金标准用） |
| `build_index.py` | 解析 → 分块 → 向量化 → Qdrant / 倒排 / BM25 |
| `run_rag_questions.py` | 10 题逐题留痕（片段原文 / 引用 / 答案 / 耗时） |
| `evaluate_rag07.py` | 检索与生成指标汇总（复用上游指标实现） |
| `evaluate_ragas.py` | RAGAS 评估（本地 Ollama 裁判，离线） |
| `src/` | **01~06 复用的 RAG 系统本体** |

---

## 五、目录结构

```
工单7/
├─ ccf_competition.zip            附件原件（只读，勿删）
├─ ccf_competition/{pdf,txt}/     解压后原始语料（文件名已还原为中文）
├─ sample_questions.pdf           示例问题来源
├─ src/                           ★ 01~06 复用的 RAG 系统（30 模块）
├─ scripts/                       数据准备与质检脚本
├─ build_index.py                 建库
├─ run_rag_questions.py           10 题留痕
├─ evaluate_rag07.py              指标汇总
├─ evaluate_ragas.py              RAGAS 评估
├─ app.py                         Streamlit 界面（中英双语）
├─ data/
│  ├─ corpus/{pdf,txt}/           规范化语料（各模块实际读取位置）
│  ├─ chunks/chunks.json          7225 块
│  ├─ qdrant/                     向量库（collection ccf_v7_annual_report）
│  ├─ index/fulltext.pkl          自研倒排索引
│  └─ eval/                       ground_truth / rag_results_* / metrics_07 / ragas_results
├─ docs/                          全部交付文档（见上表）
└─ logs/                          分阶段运行日志
```

---

## 六、两个实测运维约束

1. **Qdrant local 模式单进程占用**：嵌入式存储同一目录只能被一个进程打开，
   并发跑多个评估脚本会互斥 —— **串行执行**，或把 `QDRANT_MODE` 切 `server`。
2. **进程退出时的 Qdrant 析构告警**：日志末尾可能出现
   `Exception ignored in: <function QdrantClient.__del__> … ImportError: sys.meta_path is None`。
   这是解释器关闭阶段的**无害告警**（退出码 0、数据已落盘），可忽略。
