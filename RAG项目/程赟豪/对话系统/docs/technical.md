# 技术设计文档（RAG 角色扮演系统）

## 1. 技术架构

```
┌─────────────┐      ┌──────────────────────────────────────────┐
│  Web 前端    │ HTTP │  FastAPI 应用 (main.py)                    │
│  static/    │─────▶│  /api/chat  /api/chat/stream              │
└─────────────┘      │  /api/role/*  /api/knowledge/*            │
                     └───┬──────────┬──────────┬────────────┬────┘
                         │          │          │            │
              ┌──────────▼───┐ ┌────▼─────┐ ┌──▼──────┐ ┌───▼────┐
              │  检索 (rag)   │ │  记忆    │ │  角色    │ │  大模型 │
              │ retrieval.py │ │ short/long│ │ manager │ │  llm/  │
              └──────┬───────┘ └────┬─────┘ └─────────┘ └───┬────┘
                     │              │                        │
        ┌────────────▼─────────┐  ┌─▼──────┐   ┌─────────────▼──────┐
        │  Milvus 向量库        │  │ Redis  │   │ DeepSeek / 本地 vLLM│
        │  + 本地 BM25 稀疏索引  │  └────────┘   └────────────────────┘
        └──────────────────────┘
```

## 2. 模块功能设计

| 模块 | 文件 | 职责 |
|------|------|------|
| 文档解析 | `src/rag/document_loader.py` | PyMuPDF 提取 PDF 文本/图片、pdfplumber 提取表格、txt/md/json 读取 |
| 文本分块 | `src/rag/text_chunking.py` | fixed/sentence/paragraph/semantic 四种分块 |
| 向量化 | `src/rag/embedding.py` | 加载本地 BGE-m3（sentence-transformers），1024 维，归一化 |
| 向量库 | `src/rag/vector_store.py` | Milvus 集合（id/向量/原文/来源/页码/摘要/创建/修改时间），增删查 |
| BM25 稀疏 | `src/rag/bm25.py` | 客户端 Okapi BM25，中文二元组分词，作为第二路召回 |
| 检索 | `src/rag/retrieval.py` | 混合检索：向量 + BM25 多路召回 → RRF 融合 → 相似度过滤 → 重排序 |
| 重排序 | `src/rag/rerank.py` | BGE-rerank 精排（无模型时退回本地关键词精排） |
| Query 改写 | `src/rag/query_rewriter.py` | 结合历史共指消解、多查询扩写 |
| 生成 | `src/rag/generator.py` | 提示词构建、调用大模型、后处理（正则）、流式 |
| 知识库服务 | `src/rag/knowledge.py` | 文档→分块→向量化→入库；统计/列表/删除 |
| 短期记忆 | `src/memory/short_term.py` | Redis List 保存会话历史，TTL 过期 |
| 长期记忆 | `src/memory/long_term.py` | Milvus 保存用户画像/对话摘要，按用户隔离 |
| 角色管理 | `src/role/` | 预设模板 + JSON 文件持久化角色 |
| 大模型 | `src/llm/` | 在线 API（OpenAI 兼容）+ 本地 vLLM |
| 日志 | `src/utils/logger.py` | logging + RotatingFileHandler |

## 3. 混合检索设计（关键）

文档.txt 要求「混合检索 / 多路召回 / BM25」。本实现采用：

1. **向量召回**：BGE-m3 把问题编码成 1024 维向量，Milvus IVF_FLAT（IP 内积，等价余弦）检索 top_k×3。
2. **BM25 召回**：本地 Okapi BM25 索引（中文按字符二元组分词）检索 top_k×3。
3. **RRF 融合**：`score = Σ 1/(k + rank + 1)`，同一分块同时被两路召回时得分累加。
4. **相似度过滤**：向量距离低于阈值（默认 0.5）的结果剔除。
5. **重排序**：BGE-rerank 精排（不可用时退回关键词重叠精排），取 top_k。

> 设计取舍：BM25 采用客户端索引而非 Milvus 内建 BM25 Function，原因是 Milvus 默认分词器对中文按空格切分召回差，且客户端实现可做中文二元组分词、无需改动现有集合 schema。数据规模在本项目 KB 范围内完全可承受。

## 4. 数据模型

Milvus `roleplay_knowledge` 集合字段（对应文档.txt 第 37 行）：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT64 主键自增 | 唯一性 |
| chunk_id | VARCHAR | 分块 ID |
| text | VARCHAR(65535) | 原文 |
| vector | FLOAT_VECTOR(1024) | 向量 |
| source | VARCHAR | 文档来源 |
| page_number | INT32 | 页码 |
| metadata | VARCHAR | JSON 元信息 |
| create_time / modify_time | INT64 | 创建/修改时间 |
| summary | VARCHAR | 分块摘要 |

## 5. 关键配置

见 `config/config.yaml`：`llm`（provider/API key/model）、`embedding`（本地 BGE-m3 路径、device）、
`rerank`（method/model/device）、`milvus`（host/port/collection）、`redis`、`retrieval`（top_k/阈值/use_bm25/query_rewrite）、`document`（分块方式）。
