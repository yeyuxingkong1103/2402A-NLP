# 设计文档 · 基于 RAG 的角色扮演系统

## 1. 技术架构

```
┌────────────────────────── 客户端 ──────────────────────────┐
│   Streamlit Web   │   HTTP API (FastAPI)   │   CLI (main.py) │
└──────────────┬────────────────┬────────────────┬───────────┘
               │                │                │
               └────────────────┼────────────────┘
                                ▼
                    ┌─────────── services.py ────────┐
                    │  answer() / chat() / 知识库管理 │
                    └────┬───────────────┬───────────┘
                         │               │
              ┌──────────┘               └──────────────┐
              ▼                                          ▼
   ┌──── generation.py (生成+流式) ──┐         ┌──── memory_store.py ────┐
   │  query_rewrite → 检索 → 重排  │         │  Redis 短期记忆          │
   │  → LangChain / OpenAI         │         │  回退：进程内字典        │
   └────┬───────────────┬──────────┘         └─────────────────────────┘
        │               │
        ▼               ▼
  ┌──────────┐   ┌────────────────────────────────────┐
  │ rag.py   │   │ vector_store.py (Milvus 混合检索)    │
  │ BM25     │   │  ensure_collection / upsert / search│
  └────┬─────┘   └────┬───────────────────────────────┘
       │                │
       ▼                ▼
   ┌───────────────────────────────────┐
   │  ingest.py（动态更新 / 分块 / PDF）│
   │   embeddings.py (BGE-m3)         │
   │   rerank.py (BGE-rerank)         │
   └───────────────────────────────────┘

持久层：
   MySQL/SQLite（用户/角色/会话/消息/知识文档）
   Redis（短期记忆 + 缓冲）
   Milvus（长期知识库向量）
   BM25 索引 pickle（缓存）
```

### 1.1 分层职责

| 层 | 模块 | 职责 |
| --- | --- | --- |
| 入口 | app.py / api.py / main.py | Web / HTTP / CLI |
| 服务 | services.py | 会话管理、记忆读写、生成编排 |
| 生成 | generation.py / langchain_chain.py / query_rewrite.py | 多轮生成、流式、Query 改写、LangChain |
| 检索 | rag.py / vector_store.py / rerank.py / embeddings.py | BM25、向量召回、重排、向量化 |
| 数据接入 | ingest.py | 文件解析、分块、PDF 去水印、入 BM25 + Milvus |
| 记忆 | memory_store.py | Redis List 短期记忆、缓冲、回退 |
| 持久 | database.py / auth.py | 用户/角色/会话/消息/知识文档、注册登录 |
| 配置 | config.py / logger.py | 环境变量、日志 |

### 1.2 技术选型

- 大模型：兼容 OpenAI 协议，可接 DeepSeek / 豆包 / 硅基流动 / 千问 / Claude / ChatGPT / Gemini；本地 vLLM / SGLang / xInference。
- 编排：LangChain（Prompt + Model + Retriever + Memory + Chain + Agent 工具描述）。可选 LlamaIndex / Dify / LightRAG / RagFlow。
- 向量库：Milvus（IVF_FLAT + COSINE；字段 id/vector/text/source/summary/created_at/updated_at）。
- 关系库：MySQL（生产）/ SQLite（开发），SQLAlchemy 2.0 ORM。
- 缓存 / 短期记忆：Redis（List 数据类型，O(1) LTrim/RPush，Key-Value 键值对）。
- 向量化：BGE-m3（1024 维）。
- 重排：BGE-reranker-base。
- PDF：PyMuPDF（fitz）去水印 + PDFPlumber（表格）；可选 PaddleOCR / 多模态大模型。
- 评测：RAGAS。

## 2. 功能设计

### 2.1 用户与认证
- 表 `users(id, username, password_hash, created_at)`。
- 密码：`PBKDF2-HMAC-SHA256`，120000 轮，盐 `secrets.token_hex(16)`。
- Token：`{user_id}:{nonce}:{hmac_sha256_sig}`，HMAC-SHA256 验签。
- API 中间件 `current_user` 解析 `Authorization: Bearer <token>`。

### 2.2 角色与提示词
- 表 `roles(id, code, name, domain, description)`。
- `roles.py` 内置 9 个角色及其 `system_prompt`，初始化时写入 MySQL。
- `get_role(code)` 未命中时回退 teacher。

### 2.3 会话与消息
- `chat_sessions(id, user_id, role_code, title, created_at, updated_at)`
- `chat_messages(id, session_id, role, content, created_at)`
- `ensure_session` 复用既有 session 或新建；每轮 user+assistant 两条消息持久化。

### 2.4 知识库与动态更新
- 表 `knowledge_docs(id, source, summary, chunk_count, created_at, updated_at)`
- 文件类型分发：
  - `.tsv`：按 4 列加载句对
  - `.pdf`：`read_pdf` = PyMuPDF 去水印 + 文本；文本为空回退 `_read_pdf_tables`（PDFPlumber）
  - `.txt`/其他：UTF-8 文本
- 分块策略：段落数组 → 累积到 `MAX_CHUNK=400` → 超长段落硬切；每个 chunk 封装为 `SentencePair`。
- 入库：BM25 索引重建 + Milvus `upsert_texts` + KnowledgeDoc 记录。
- 缓存：`.cache/teacher_bm25.pkl` + `.cache/extra_docs.pkl`，按 TSV mtime 失效。

### 2.5 检索与重排
- `TranslationRetriever.search`：jieba+英文分词 → BM25Okapi → 排序 → 阈值过滤。
- `hybrid_search`：BM25 ∪ 向量召回 → 简单分数融合。
- `rewrite_query`：大模型扩写检索词（无 KEY 时原样返回）。
- `rerank`：BGE-rerank 计算相关性得分重排（未启用时按混合得分截断）。

### 2.6 多轮对话与记忆
- 短期记忆：Redis List，Key `chat:{user_id}:{role_code}:{session_id}`；`RPush` + `LTrim(-2N, -1)` + `Expire(7d)`；O(1) 常数时间复杂度。
- 长期记忆：Milvus 持久向量库 + MySQL `chat_messages` 落库。
- 回退：Redis 不可用时使用进程内 `defaultdict(list)`，保证开发可用。

### 2.7 生成与流式
- `build_messages`：system_prompt + 历史截断（最近 `SHORT_MEMORY_TURNS*2`） + 「检索资料 + 用户问题」。
- `chat`：优先 LangChain，失败回退原生 OpenAI；未配置 KEY 走 `fallback_answer`（仅讲解检索结果）。
- `chat_stream`：OpenAI `stream=True` 逐 chunk yield，配合 FastAPI `StreamingResponse`。

### 2.8 缓冲
- `memory_store.cache_get/cache_set`：通用缓存（TTL 默认 300s）。
- BM25 pickle 缓存按数据 mtime 失效。
- Milvus collection 单例 `ensure_collection`。

## 3. Milvus Collection 设计

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | INT64 PK auto_id | 主键唯一自增 |
| vector | FLOAT_VECTOR(dim=1024) | BGE-m3 向量 |
| text | VARCHAR(2000) | 混合检索原文 |
| source | VARCHAR(256) | 文档来源 |
| summary | VARCHAR(512) | 摘要 |
| created_at | VARCHAR(32) | 创建时间 |
| updated_at | VARCHAR(32) | 修改时间 |

- 索引：IVF_FLAT + COSINE，`nlist=128`。
- 检索：`metric_type=COSINE`，`nprobe=10`，输出 `text/source/summary`。

## 4. 数据库设计（MySQL/SQLite）

| 表 | 主要字段 |
| --- | --- |
| users | id, username(unique), password_hash, created_at |
| roles | id, code(unique), name, domain, description |
| chat_sessions | id, user_id(FK), role_code, title, created_at, updated_at |
| chat_messages | id, session_id(FK), role, content, created_at |
| knowledge_docs | id, source, summary, chunk_count, created_at, updated_at |

## 5. RAG 优化策略

| 方向 | 实现 |
| --- | --- |
| 混合检索 | BM25 + Milvus 分数融合 |
| 重排序 | BGE-rerank |
| 多路召回 | Milvus / MySQL / Redis / Neo4J / MongoDB / ClickHouse / 互联网（预留） |
| 分块优化 | 段落 + 固定长度 + 句子；预留语义/标题/父子块 |
| Query 改写 | 大模型扩写检索词 |
| 得分过滤 | SCORE_THRESHOLD + 余弦相似度过滤 |
| Embedding 优化 | BGE-m3（可微调） |
| 数据优化 | 去重、删除低质量文档（KnowledgeDoc） |
| 生成结果优化 | 后处理、校验（角色 prompt 约束） |
| 缓冲 | Redis cache_get/cache_set + BM25 pickle |
| 流式输出 | OpenAI stream + StreamingResponse |
| 大模型优化 | 多 provider 切换、temperature=0.4 |
| 硬件优化 | 负载均衡 nginx/http-proxy、横向扩展 |

## 6. 环境

- 开发：`APP_ENV=development`，本地 SQLite + 进程内字典 + BM25
- 测试：`APP_ENV=testing`，pytest + httpx + RAGAS
- 生产：`APP_ENV=production`，MySQL + Redis + Milvus + nginx 负载均衡
