# 基于 PDF 文档的问答系统优化（RAG）

> **工单编号**：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
> **上游工单**：01-基于 PDF 文档的问答系统
> **数据源**：`data/招股说明书1.pdf`（武汉兴图新科电子股份有限公司招股意向书）
> **优化目标**：问答准确率 ≥ 90%、提问到返回答案 ≤ 3 s

本项目在 01 工单（基础 RAG 问答）之上，从 **PDF 解析处理 / 分块 / 检索 / 答案合成** 四个层面
做检索准确率优化，并保留**可复现的优化前基线**，形成 before/after 量化对比。

---

## 一、优化前后对比结论

| 指标 | 优化前（基线，01 工单） | 优化后 | 提升 |
|---|---|---|---|
| 答案命中率（Accuracy） | 60.0%（6/10） | **100.0%（10/10）** | **+40 pt** |
| 检索 Top-1 命中率 | 60.0% | 70.0% | +10 pt |
| 检索 MRR | 0.750 | 0.784 | +0.034 |
| 平均响应时间 | 3.532 s | **1.122 s** | 提速 ≈ 3× |
| 最大响应时间 | — | **2.181 s** | ✅ ≤ 3 s |

> 详细逐题对比见 [`output/evaluation_report.md`](output/evaluation_report.md)。

---

## 二、优化方案总览

| 层面 | 优化前（基线） | 优化后 | 解决的问题 |
|---|---|---|---|
| **1. PDF 解析** | 直接抽取全文文本 | 版式清洗（剔除页眉/页脚/水印）+ 表格结构化（Markdown）+ 行合并与空白规整 + 章节标题识别 | 噪声污染语义、表格数值被拆散 |
| **2. 分块** | 固定 500/80 字符递归切片 | 结构感知分块（按标题切语义小节）+ 表格原子块 + 父子块 + 过短合并/超长切分 | 跨小节"串味"、表头与数据分离、上下文不足 |
| **3. 检索** | 单路查询 + 向量/BM25 RRF 融合 | 多路查询（核心问句/关键词/实体增强）+ 双路召回 + 加权线性重排（关键词覆盖/数值/实体/表格加成） | 公司名等套话稀释语义、排序不准 |
| **4. 答案合成** | 本地 `deepseek-r1:1.5b` 生成 | 抽取式答案合成（类型感知打分 + 对比词/套话惩罚 + 页码引用） | 推理型模型慢（>10 s）、易幻觉、不可溯源 |

---

## 三、目录结构

```
工单二/
├── app.py                      # Streamlit 演示界面
├── requirements.txt
├── .env.example
├── data/
│   ├── 招股说明书1.pdf          # 知识库源文档
│   └── questions.json          # 10 道待评估题目 + 参考答案
├── src/
│   ├── config.py               # 全局配置（分块/检索/重排参数）
│   ├── pdf_parser.py           # 【优化点 1】PDF 解析与版式清洗
│   ├── chunking.py             # 【优化点 2】结构感知分块 / 表格原子块 / 父子块
│   ├── query_understanding.py  # 【优化点 3-1】Query 理解与多路查询
│   ├── knowledge_base.py       # 向量化与 FAISS 知识库管理
│   ├── retriever.py            # 【优化点 3-2】混合检索（向量 + BM25 + RRF）
│   ├── reranker.py             # 【优化点 3-3】加权线性重排
│   ├── answer_builder.py       # 【优化点 4】抽取式答案合成
│   ├── qa_engine.py            # 问答引擎（优化链路 / 基线链路）
│   └── evaluate_keys.py        # 评估关键线索（脚本与界面共用）
├── scripts/
│   ├── build_kb.py             # 构建向量库（--all 同时构建优化前/后）
│   ├── evaluate.py             # 优化前后对比评估
│   ├── make_figures.py         # 生成对比图表
│   └── make_video.py           # 合成系统演示视频（MP4）
├── vector_db/
│   ├── optimized/              # 优化后 FAISS 索引
│   └── baseline/               # 优化前 FAISS 索引
└── output/                     # 评估报告、图表、演示产物
```

---

## 四、环境准备

```bash
# 1) 创建/激活环境（本项目使用 conda 环境 langchain2，Python 3.10）
conda activate langchain2
pip install -r requirements.txt

# 2) 启动本地 Ollama 并拉取模型
ollama serve
ollama pull bge-m3:567m
ollama pull deepseek-r1:1.5b
```

---

## 五、运行步骤

```bash
# 1) 构建向量库（优化前基线 + 优化后，首次约 1~3 分钟）
python scripts/build_kb.py --all

# 2) 运行优化前后对比评估（自动预热，避免冷启动计入耗时）
python scripts/evaluate.py
#   产物：output/evaluation_report.json、output/evaluation_report_markdown.txt

# 3) 生成对比图表
python scripts/make_figures.py

# 4) 合成演示视频（用截图与图表生成 MP4，产物 output/demo_video.mp4）
python scripts/make_video.py

# 5) 启动演示界面
streamlit run app.py
```

---

## 六、验收对照

| 验收项 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 问答准确性 | ≥ 90% | 100%（10/10） | ✅ |
| 响应时间 | ≤ 3 s | 平均 1.122 s / 最大 2.181 s | ✅ |
| 优化方案对比分析 | 需提供 | `output/evaluation_report.md` + `output/figures/` | ✅ |
| 交互友好性 | 清晰简洁界面 | Streamlit 界面（单题问答 / 前后对比 / 批量评估） | ✅ |
| 多语言支持 | 中英文 | 中文为主，英文提问同样走同一链路（Query 理解兼容英文分词） | ✅ |
| 容错机制 | 处理常见异常 | 空问题兜底、向量检索异常降级、LLM 调用异常捕获、FAISS 中文路径桥接 | ✅ |

---

## 七、容错与稳定性设计

- **空输入**：`QueryUnderstanding.analyze` 对空串返回空分析结果，答案合成输出兜底话术；
- **检索异常**：向量检索/BM25 任一失败时降级为另一路结果，不中断问答；
- **模型异常**：`QAEngine._invoke` 捕获 LLM 调用异常并返回可读提示；
- **路径兼容**：Windows 下 FAISS 无法处理中文路径，`knowledge_base.save_local/load_local` 使用临时 ASCII 目录桥接；
- **长时运行**：向量库与 BM25 索引进程内构建一次并复用，`Streamlit` 以 `st.cache_resource` 缓存引擎；
- **可溯源**：所有答案附带页码引用，避免大模型幻觉。

---

## 八、代码注释规范

所有源码文件头部均标注工单编号：
`工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化`