# 基于 PDF 文档的问答系统（RAG）技术文档

- **项目名称**：基于 PDF 文档的问答系统（RAG）
- **文档类型**：技术文档
- **日期**：2025 年

---

## 1. 系统概述

本系统针对《招股说明书1.pdf》（约 10.47 MB）构建了一个基于大语言模型（LLM）和检索增强生成（RAG）技术的问答系统。用户输入自然语言问题后，系统从招股说明书中检索相关内容，并由 LLM 生成准确、简洁的回答。

### 核心功能

- **Query 理解**：将用户的长问题改写为检索友好的短查询
- **检索与生成**：向量检索 + BM25 关键词检索的混合召回，结合 LLM 生成回答
- **交互界面**：基于 Gradio 的问答界面，支持问题输入、答案展示和反馈

---

## 2. 系统架构

```text
┌─────────────────────────────────────────────────────┐
│                   用户（Gradio 界面）                 │
└──────────────────────┬──────────────────────────────┘
                       │ 问题
                       ▼
┌─────────────────────────────────────────────────────┐
│              Query 理解（LLM 改写查询）               │
└──────────────────────┬──────────────────────────────┘
                       │ 改写后的查询
                       ▼
┌─────────────────────────────────────────────────────┐
│                   混合检索                            │
│  ┌──────────────────┐    ┌──────────────────┐        │
│  │    向量检索       │    │  BM25 关键词检索  │        │
│  │ (Chroma + bge)   │    │  (jieba 分词)    │        │
│  └────────┬─────────┘    └────────┬─────────┘        │
│           └────────┬──────────────┘                  │
│                    ▼                                 │
│              合并去重（取前 8 块）                     │
└──────────────────────┬──────────────────────────────┘
                       │ 上下文
                       ▼
┌─────────────────────────────────────────────────────┐
│           LLM 生成（Qwen2.5-7B-Instruct）             │
└──────────────────────┬──────────────────────────────┘
                       │ 回答
                       ▼
┌─────────────────────────────────────────────────────┐
│            Gradio 界面（答案 + 来源片段）              │
└─────────────────────────────────────────────────────┘
```

---

## 3. 技术选型

| 组件 | 选型 | 说明 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04 | AutoDL 容器环境 |
| Python | 3.12 | 兼容新版依赖 |
| PyTorch | 2.5.1 + CUDA 12.4 | GPU 加速 |
| 大语言模型 | Qwen2.5-7B-Instruct | 中文能力强，4-bit 量化后约 5GB 显存 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | 中文语义嵌入，轻量高效 |
| 向量数据库 | ChromaDB 0.5.23 | 本地持久化，无需额外服务 |
| PDF 解析 | pdfplumber（LangChain PDFPlumberLoader） | 支持文字和表格提取 |
| 检索策略 | 向量检索 + BM25 混合 | 兼顾语义相似与精确关键词 |
| 中文分词 | jieba | BM25 中文分词 |
| RAG 框架 | LangChain 0.3.30 | 检索与生成编排 |
| 评估框架 | RAGAS 0.3.2 | RAG 质量评估 |
| Web 界面 | Gradio 5.42.0 | 快速搭建交互界面 |

---

## 4. 目录结构

```text
/root/autodl-tmp/projects/RAG/
├── 0_raw_data/
│   └── 招股说明书1.pdf          # 原始 PDF
├── 1_trans_data/
│   └── 招股说明书1.txt          # PDF 解析后的纯文本
├── 2_vector_db/                 # Chroma 向量库持久化目录
├── parse_pdf.py                 # PDF 解析脚本
├── build_db.py                  # 向量库构建脚本
├── rag_engine.py                # RAG 引擎核心
├── app.py                       # Gradio 问答界面
├── test_rag.py                  # 单问题测试脚本
├── batch_test.py                # 批量测试脚本
├── ground_truth.py              # 标准答案（评估用）
├── evaluate_rag.py              # RAG 评估脚本
└── evaluate_pure_llm.py         # 纯 LLM 评估脚本
```

---

## 5. 开发流程

### 5.1 环境准备

```bash
# 创建虚拟环境
conda create -n rag python=3.12 -y
conda activate rag

# 配置 pip 镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 PyTorch（与镜像版本一致）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 安装核心依赖
pip install langchain==0.3.30 langchain-community==0.3.30 langchain-core==0.3.86
pip install chromadb==0.5.23 sentence-transformers==3.3.1
pip install pdfplumber bitsandbytes==0.44.1
pip install jieba rank_bm25
pip install ragas==0.3.2 datasets==2.21.0
pip install gradio==5.42.0
```

### 5.2 模型下载

```bash
# 使用 modelscope 下载（国内速度快）
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models')
snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir='/root/autodl-tmp/models')
"
```

### 5.3 PDF 解析

```bash
python parse_pdf.py
```

输出：`1_trans_data/招股说明书1.txt`

### 5.4 构建向量库

```bash
python build_db.py
```

输出：`2_vector_db/` 目录（1214 个文本块）

### 5.5 启动问答界面

```bash
python app.py
```

在 AutoDL 控制台将 6006 端口映射为自定义服务，通过浏览器访问。

---

## 6. 关键技术实现

### 6.1 文本分块

- 分块大小：500 字符
- 重叠长度：50 字符
- 分隔符优先级：`\n\n` → `\n` → `。` → `；` → `，`

### 6.2 Query 理解

用 LLM 将用户长问题改写为短查询，去掉冗余实体和客套词，提升检索命中率。

**示例：**

- 原问题：报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？
- 改写后：公司报告期内军用领域收入

### 6.3 混合检索

- **向量检索**：基于 Chroma + bge-small-zh-v1.5，取 top 6
- **BM25 检索**：基于 jieba 分词，取 top 6
- **合并策略**：向量结果优先，BM25 结果补充，去重后取前 8 块

### 6.4 LLM 生成

- **模型**：Qwen2.5-7B-Instruct
- **量化**：4-bit（bitsandbytes NF4）
- **生成参数**：do_sample=False（确定性输出），max_new_tokens=512

### 6.5 Prompt 设计

Prompt 中包含 5 条关键规则：

1. 区分母公司与子公司
2. 子公司数值不能当作母公司数值
3. 军用收入占比需用"直接+间接"合计值
4. 军用收入四个年度数值按原文顺序对应
5. 找不到信息时明确回答"上下文未提供相关信息"

---

## 7. 性能指标

| 指标 | 数值 |
| --- | --- |
| 向量库文本块数 | 1214 |
| 平均响应时间 | 2.7 秒 |
| faithfulness | 1.0000 |
| answer_relevancy | 0.8411 |
| 10 题准确率 | 100% |

---

## 8. 依赖版本清单

```text
Python 3.12
torch==2.5.1+cu124
transformers==4.46.3
accelerate==1.0.1
bitsandbytes==0.44.1
langchain==0.3.30
langchain-community==0.3.30
langchain-core==0.3.86
chromadb==0.5.23
sentence-transformers==3.3.1
pdfplumber
jieba
rank_bm25
ragas==0.3.2
datasets==2.21.0
gradio==5.42.0
```
