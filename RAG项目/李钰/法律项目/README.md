# 法律RAG问答系统

基于中国裁判文书网法律数据集（500条）的检索增强生成（RAG）问答系统，支持混合检索、重排序和多轮对话。

## 项目架构

```
法律项目/
├── build_kb.py                      # 离线建库模块（PDF解析→分块→向量化→Milvus存储→BM25索引）
├── rag_chain.py                     # 在线问答模块（混合检索→RRF融合→重排序→多轮对话→LLM生成）
├── web_app.py                       # Web启动模块（HTTP服务器→聊天界面→接口服务）
├── 中国裁判文书网法律数据集_500条.pdf  # 法律数据源
├── bm25_index.pkl                   # BM25稀疏索引（运行build_kb.py后自动生成）
├── README.md                        # 项目文档
└── 思维导图.md                      # 系统思维导图
```

## 核心功能

### 1. 离线建库（build_kb.py）

| 功能 | 说明 |
|------|------|
| PDF解析 | 支持三种方法：PyMuPDF（默认）、PaddleOCR（扫描件）、MinerU（版面分析） |
| 文本分块 | 滑动窗口分块，块大小500字符，重叠80字符，智能句末截断 |
| 向量入库 | bge-m3模型生成1024维向量，批量写入Milvus |
| BM25索引 | rank-bm25 + jieba分词构建稀疏检索索引，持久化到磁盘 |

**解析方法切换**：修改 `build_kb.py` 中 `PARSER_METHOD` 变量
- `auto`：优先PyMuPDF，失败降级PaddleOCR
- `pymupdf`：文本型PDF专用
- `paddleocr`：扫描件/图片型PDF
- `mineru`：需要版面分析的复杂文档

### 2. 在线问答（rag_chain.py）

| 功能 | 说明 |
|------|------|
| 稠密检索 | Milvus向量检索（COSINE相似度），返回Top-20 |
| 稀疏检索 | BM25关键词检索，返回Top-20 |
| 混合融合 | RRF（Reciprocal Rank Fusion）倒数排名融合，K=60 |
| 重排序 | CrossEncoder（bge-reranker-base）精排，无依赖时降级 |
| 多轮对话 | Redis存近期对话 + Milvus存全部历史（语义召回） |
| LLM生成 | qwen2.5:1.5b模型，法律RAG提示词模板 |

**RAG流程**：
```
用户问题 → 查询向量化 → 混合检索（稠密+稀疏）→ RRF融合 → 重排序
         → 语义召回历史 + 近期历史 → LLM生成 → 记录对话
```

### 3. Web启动（web_app.py）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回聊天界面HTML |
| `/status` | GET | 返回系统初始化状态 |
| `/ask` | POST | 接收问题，返回答案+参考来源 |

**特性**：
- 零前端依赖，HTML/CSS/JS内嵌于Python字符串
- 后台线程初始化RAG，前端轮询状态
- 法律蓝主题，聊天气泡式交互
- 串行化RAG调用，避免Ollama并发冲突

## 环境准备

### 1. 安装Ollama并拉取模型

```bash
# 安装Ollama后执行
ollama pull bge-m3:567m    # 嵌入模型（1024维）
ollama pull qwen2.5:1.5b   # 对话模型
```

### 2. 安装Python依赖

```bash
# 核心依赖（必需）
pip install pymilvus milvus-lite pymupdf langchain-ollama rank-bm25 jieba

# 重排序（推荐）
pip install sentence-transformers

# 多轮对话Redis（可选，不装自动降级为内存存储）
pip install redis

# OCR解析（可选，按需安装）
pip install paddleocr paddlepaddle

# MinerU解析（可选，按需安装）
pip install magic-pdf
```

## 运行方式

### 第一步：离线建库

```bash
python build_kb.py
```

输出：
- Milvus数据库：`C:\Users\ETERNITY\milvus_legal_rag.db`
- BM25索引：`bm25_index.pkl`

### 第二步：启动Web问答

```bash
python web_app.py
```

浏览器自动打开 `http://localhost:8000/`，等待"系统就绪"后即可问答。

## 配置说明

两份代码文件顶部均有配置区，关键参数：

| 参数 | build_kb.py | rag_chain.py | 说明 |
|------|-------------|--------------|------|
| MILVUS_DB | ✓ | ✓ | Milvus数据库路径（ASCII路径） |
| EMBED_MODEL | ✓ | ✓ | 嵌入模型名 |
| LLM_MODEL | - | ✓ | 对话模型名 |
| CHUNK_SIZE | ✓ | - | 文本块大小 |
| TOP_K_DENSE | - | ✓ | 稠密检索返回数 |
| TOP_K_FINAL | - | ✓ | 最终返回文档数 |
| RRF_K | - | ✓ | RRF融合常数 |
| HISTORY_TURNS | - | ✓ | 保留对话轮数 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 向量数据库 | Milvus Lite（本地文件模式） |
| 嵌入模型 | bge-m3:567m（Ollama） |
| 对话模型 | qwen2.5:1.5b（Ollama） |
| 稀疏检索 | rank-bm25 + jieba分词 |
| 重排序 | sentence-transformers CrossEncoder |
| 会话存储 | Redis + Milvus |
| Web服务 | Python http.server标准库 |
| PDF解析 | PyMuPDF / PaddleOCR / MinerU |

## 数据集说明

数据源：`中国裁判文书网法律数据集_500条.pdf`

包含500条裁判文书，每条含：
- 案件标题与案号
- 当事人信息
- 审理法院
- 案件详情

案件类型：民事、刑事、行政、知识产权。
