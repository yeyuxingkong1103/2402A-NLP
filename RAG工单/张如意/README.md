# 张如意 · RAG 工单作业（工单一 ~ 工单八）

> 项目名称：RAG（检索增强生成）
> 工单来源：北京八维信息集团 · 八维文化与产业研究院
> 创建人：王洪荣　|　工单版本：V1.1（20250206）/ V1.0（20250826）
> 完成日期：2026-10-03

---

## 一、提交内容与工单对应关系

| 本目录 | 工单名称 | 核心能力 |
|---|---|---|
| [`工单一-基于PDF文档的问答系统/`](工单一-基于PDF文档的问答系统/) | 基于 PDF 文档的问答系统 | PDF 解析 + 向量检索 + LLM 生成 + Web 界面 |
| [`工单二-问答系统检索优化/`](工单二-问答系统检索优化/) | 问答系统检索优化 | 分块优化 + 检索优化 + Prompt 优化，准确率 ≥90% |
| [`工单三-表格解析及检索优化/`](工单三-表格解析及检索优化/) | 表格解析及检索优化 | pdfplumber 表格抽取 → Markdown → 整表入库 |
| [`工单四-图像内容解析及检索优化/`](工单四-图像内容解析及检索优化/) | 图像内容解析及检索优化 | 多模态（CLIP + VLM）图表语义解析 |
| [`工单五-Query理解优化-多轮对话/`](工单五-Query理解优化-多轮对话/) | Query 理解优化（多轮对话） | 意图识别 / 消歧 / 分解 / 多轮指代消解 |
| [`工单六-混合检索/`](工单六-混合检索/) | 混合检索 | 向量 + BM25 全文 + 三种融合 + 三种重排 |
| [`工单七-功能测试及评估/`](工单七-功能测试及评估/) | 功能测试及评估 | 测试集构建 + 跑批 + RAGAS 评估 + 问题分析 |
| [`工单八-GraphRAG金融问答/`](工单八-GraphRAG金融问答/) | Graph RAG 金融问答 | 知识图谱构建 + 社区检测 + 图谱检索 + 可视化 |

各工单按顺序递进，能力在前序成果上叠加：

```
工单一  基础版：fixed 分块 → 纯向量检索 → LLM 生成
  ↓
工单二  structure 分块（章节路径注入）+ TF-IDF 重排 + Prompt 优化
  ↓
工单三  + 表格解析（pdfplumber 整表转 Markdown 入库）
  ↓
工单四  + 图像多模态解析（CLIP + VLM）
  ↓
工单五  + Query 理解（意图 / 消歧 / 分解 / 多轮指代消解）
  ↓
工单六  + BM25 全文检索 + 混合融合（weighted / rrf / vote）+ 级联重排
  ↓
工单七  → 在 9 份金融年报上做功能测试与 RAGAS 评估
  ↓
工单八  → Graph RAG：知识图谱构建 + 图谱检索 + 可视化
```

### 语料

| 文档 | 内容 | 用于工单 |
|---|---|---|
| 《招股说明书1.pdf》 | 武汉兴图新科电子股份有限公司（军工电子信息） | 工单一、工单二、工单四（可选）、工单五 |
| 《招股说明书2.pdf》 | 武汉力源信息技术股份有限公司（IC 分销） | 工单三、工单四、工单五 |
| `ccf_competition/` 9 份年报 | 平安银行、中国平安、招商银行、邮储银行、中信证券、中国人寿、中国太保、招商证券、国泰君安 | 工单七、工单八 |
| `sample_questions.pdf` | 4 个示例问题及参考答案 | 工单七 |

---

## 二、目录结构

```
张如意/
├── README.md                        ← 本文件
├── requirements.txt                 依赖清单
├── rag_core/                        ★ 共享核心库（各工单共用，约 5000 行）
├── 工单一-基于PDF文档的问答系统/
│   ├── src/                         可运行脚本（build_index / qa_cli / serve / run_evaluation ...）
│   ├── docs/                        技术文档 / 用户手册 / 验收对照表
│   ├── demo/                        演示录制脚本 + 截图说明
│   └── results/                     运行产出说明（字段含义与生成方式）
├── 工单二-问答系统检索优化/            （目录结构同上）
├── 工单三-表格解析及检索优化/
├── 工单四-图像内容解析及检索优化/
├── 工单五-Query理解优化-多轮对话/
├── 工单六-混合检索/
├── 工单七-功能测试及评估/
└── 工单八-GraphRAG金融问答/
```

> 说明：各工单的脚本会通过 `Path(__file__).resolve().parents[2]` 把项目根目录加入
> `sys.path` 并导入 `rag_core`，因此请**保持本目录整体结构不变**（`rag_core/` 与
> 各工单目录同级），脚本即可直接运行。

---

## 三、共享核心库 `rag_core`

各工单的能力构建在同一套核心库上，通过配置切换能力开关，避免重复实现。

| 模块 | 职责 | 主要类 / 函数 |
|---|---|---|
| `config.py` | 全局配置、路径、超参、评测问题集 | `PRESETS`、`QUESTIONS_*` |
| `llm.py` | LLM 客户端（DeepSeek 主 / Ollama 备） | `chat`、`chat_json`、`get_usage` |
| `embed.py` | 向量嵌入，支持多模型切换 | `encode`、`set_model`、`MODEL_REGISTRY` |
| `pdf_parse.py` | PDF 三层解析（文字 / 表格 / 图像） | `parse_pdf`、`extract_table_blocks`、`extract_images` |
| `chunk.py` | 四种分块策略 | `chunk_blocks`：fixed / recursive / semantic / structure |
| `vectorstore.py` | 向量库（Chroma 主 / NumPy 备） | `VectorStore` |
| `bm25.py` | 倒排索引 + BM25 全文检索 | `BM25Retriever`：布尔 / 短语 / 模糊 / 多字段 |
| `rerank.py` | 三种重排 + 级联 | `LLMReranker`、`TFIDFReranker`、`AdaptiveReranker`、`CascadeReranker` |
| `retriever.py` | 统一检索，三种融合算法 | `Retriever.retrieve`：weighted / rrf / vote |
| `query_understand.py` | Query 理解与多轮指代消解 | `understand`、`resolve_coreference`、`Turn` |
| `generator.py` | 答案生成与引用 | `generate_rag`、`generate_llm_only`、`generate_multi_hop` |
| `evaluate.py` | RAGAS 四大指标 + 检索指标 | `evaluate_records`、`context_precision`、`context_recall` |
| `image_parse.py` | 图像多模态语义解析 | `CLIPImageParser`、`VLMImageParser` |
| `graph_rag.py` | 知识图谱构建与图谱检索 | `KnowledgeGraph`、`GraphExtractor`、`GraphRAG` |
| `pipeline.py` | 流水线编排 | `Pipeline`、`PRESETS` |
| `api.py` | FastAPI 服务 + 内置 Web 界面 | `app` |

---

## 四、环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt
# 国内加速：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置 LLM API Key（DeepSeek）
export DEEPSEEK_API_KEY="sk-xxxxxxxx"          # Linux / macOS
$env:DEEPSEEK_API_KEY="sk-xxxxxxxx"            # Windows PowerShell

# 3. 可选：本地模型后端（离线场景）
ollama pull qwen3:4b                    # 生成模型
ollama pull dengcao/bge-large-zh-v1.5   # 嵌入模型
```

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（**答案生成 / 评估必需**） | — |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `RAG_LLM_MODEL` | 生成模型名 | `deepseek-chat` |
| `RAG_EMBED_MODEL` | 嵌入模型 | `BAAI/bge-large-zh-v1.5` |
| `RAG_VLM_MODEL` | 多模态模型名（工单四） | 自动探测 |
| `OLLAMA_BASE_URL` | 本地模型服务地址 | `http://localhost:11434` |
| `HF_ENDPOINT` | HuggingFace 镜像（国内） | — |

---

## 五、运行方法（以工单一为例）

```bash
# 1. 建立索引：解析《招股说明书1.pdf》→ 分块 → 向量索引 + BM25 索引
python 工单一-基于PDF文档的问答系统/src/build_index.py

# 2. 启动服务（FastAPI + 内置 Web 界面）
python 工单一-基于PDF文档的问答系统/src/serve.py
# 浏览器访问 http://localhost:8000
```

各工单的主要脚本：

| 工单 | 主要脚本 |
|---|---|
| 工单一 | `build_index.py`（建索引）、`qa_cli.py`（命令行问答）、`serve.py`（Web 服务）、`run_evaluation.py`（评测）、`compare_rag_vs_llm.py`（RAG vs 纯 LLM 对比）、`speech_input.py`（语音输入） |
| 工单二 | `optimize_chunking.py` / `optimize_retrieval.py` / `optimize_prompt.py`（三项优化）、`run_full_optimization.py`（整体跑批）、`serve.py` |
| 工单三 | `table_extractor.py`（表格抽取）、`build_index_with_tables.py`（整表入库建索引）、`table_qa.py`（表格问答）、`compare_with_without_tables.py`（有无表格对比）、`serve.py` |
| 工单四 | `image_extractor.py`（图像抽取）、`image_semantic_parse.py`（CLIP + VLM 语义解析）、`build_multimodal_index.py`（多模态建索引）、`image_qa.py`（图像问答）、`compare_with_without_images.py`、`serve.py` |
| 工单五 | `query_understanding.py`（意图 / 消歧 / 分解）、`multi_turn_chat.py` + `chat_ui.py`（多轮对话）、`ablation_multiturn.py`（消融对比）、`serve.py` |
| 工单六 | `vector_search.py` / `fulltext_search.py`（双通道）、`hybrid_search.py`（融合检索）、`rerank_comparison.py`（三种重排对比）、`retrieval_strategy_comparison.py`、`serve.py` |
| 工单七 | `prepare_corpus.py`（语料准备）、`build_questions.py`（测试集构建）、`run_rag_test.py`（跑批）、`run_evaluation.py`（RAGAS 评估）、`analyze_problems.py`（问题分析）、`serve.py` |
| 工单八 | `prepare_corpus.py`、`build_graph.py`（图谱构建）、`graph_qa.py`（图谱问答）、`visualize_graph.py`（可视化）、`research_report.py`（研报生成）、`compare_with_wo07.py`、`serve.py` |

---

## 六、说明

1. **代码注释规范**：每个工单的源码文件头部注释均标注该工单编号，符合各工单「备注」中的要求。
2. **演示视频**：受交付形式限制，各工单 `demo/演示脚本.md` 提供完整的分镜录制脚本
   （镜头号 / 时长 / 操作步骤 / 预期画面 / 解说词），配套 `demo/screenshots/` 收集实际运行截图。
3. **结果数据**：各工单 `results/` 目录存放脚本实际运行产生的结果文件，
   字段含义见各目录下的 `README.md`。
   **交付时 `results/` 仅含说明文件，未预置跑批结果**——
   指标数值需实际运行脚本产生，避免出现「未经运行就写死的指标」。
4. **纯检索实验不需要 API Key**：PDF 解析、分块对比、向量嵌入与检索、
   BM25 全文检索、TF-IDF 重排、检索层指标（Hit Rate / MRR / Recall@k）均可离线运行；
   答案生成、RAGAS 评估、图谱抽取需要 `DEEPSEEK_API_KEY`。
5. 工单原文中的目录名（如 `工单01-基于PDF文档的问答系统/`）在本提交中按班级要求命名，
   源码内容未作改动。
