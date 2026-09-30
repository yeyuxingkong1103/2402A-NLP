# 技术文档 — 基于 PDF 文档的问答系统

> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

## 一、系统架构

用户问题 → Query 理解 → 向量检索 → 答案生成 → Web 展示

## 二、技术选型

| 模块 | 技术 | 版本 |
|------|------|------|
| PDF 解析 | PyMuPDF | 1.23.8 |
| 表格提取 | pdfplumber | 0.10.3 |
| 向量模型 | BAAI/bge-base-zh-v1.5 | 768 维 |
| 向量框架 | sentence-transformers | 6.1.0 |
| 索引 | faiss-cpu | 1.8.0 |
| Web | Gradio | 6.28.0 |
| Python | 3.12 | - |

## 三、核心模块

### 3.1 pdf_parser.py

- extract_text() — 按页提取 PDF 文字
- extract_tables() — 提取所有表格
- chunk_text() — 按 300 字符 + 80 字符重叠切块

### 3.2 vector_retriever.py

- 加载 bge-base-zh-v1.5（safetensors 模式）
- build_index() — 批量向量化 + FAISS 索引
- retrieve(query, top_k=10) — 检索 Top-K 相关块

### 3.3 query_understanding.py

- 意图识别（数值/实体/标准查询等）
- 消歧（报告期、公司等替换）
- 分解（并列结构、时间维度拆分）
- 关键词提取

### 3.4 llm_generator.py

- Mock 模式（默认）：智能拼接检索原文
- 关键词命中排序 + 去重
- 支持切换 DeepSeek API

### 3.5 rag_evaluator.py

- 关键词覆盖率
- 检索相似度评分
- RAG vs 纯 LLM 对比

## 四、数据流

1. 加载 PDF → 548 页文本 + 477 表格
2. 文本切分 → 2010 块（300/80）
3. 表格转文本 → 455 块
4. 合并 → 2465 块
5. 向量化 → 2465 × 768 矩阵
6. FAISS 索引构建
7. 用户提问 → Query 理解 → 检索 Top-10 → 生成答案

## 五、性能

| 阶段 | 耗时 |
|------|:---:|
| PDF 解析 | ~5s |
| 文本切分 | <1s |
| 模型加载 | ~2s |
| 向量化 2465 块 | ~4s |
| 索引构建 | <1s |
| 单次查询 | 0.01~0.02s |

## 六、代码规范

所有 .py 文件头部统一包含工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

## 七、已知限制

1. Mock 模式无法跨块汇总（如军用收入各年度分散在 3~4 块）
2. 表格结构化程度有限
3. 未接真实 LLM 时，对复杂问题的回答不如 API

## 八、扩展性

- 换模型：改 vector_retriever.py 中 model_name 参数
- 换 LLM：改 llm_generator.py 中 _call_* 方法
- 换数据：把 PDF 放到 data/，改路径
- 多文档：build_index() 支持 chunks 列表
