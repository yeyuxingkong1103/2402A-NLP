# 技术设计文档

## 1. 技术架构

```text
客户端 / Postman / Apipost
            |
        Nginx / 负载均衡
            |
        FastAPI API 层
            |
  RAGService 编排层
   |       |       |       |
解析分块  检索重排  Prompt/LLM  记忆
   |       |       |       |
PyMuPDF  Milvus   OpenAI    Redis
pdfplumber 本地索引兼容 API   MySQL 长期记录
PaddleOCR-VL     vLLM/SGLang
```

## 2. 代码模块

- `app/core/config.py`：环境变量和默认配置。
- `app/core/logging.py`：JSON 日志、轮转文件、请求 ID。
- `app/core/database.py`：SQLAlchemy AsyncEngine 和初始化。
- `app/storage/models.py`：角色、文档、用户、消息表。
- `app/rag/parsers.py`：PDF、文本和 OCR 解析器。
- `app/rag/chunking.py`：标题/段落优先的分块器。
- `app/rag/embeddings.py`：BGE-m3/句向量/本地 fallback。
- `app/rag/vector_store.py`：本地 JSON 向量索引和 Milvus 适配器。
- `app/rag/retrievers.py`：向量召回、BM25 风格召回、RRF。
- `app/rag/reranker.py`：BGE-Reranker 和 fallback scorer。
- `app/rag/prompt.py`：角色模板和上下文注入。
- `app/rag/memory.py`：Redis 短期记忆和内存降级。
- `app/rag/llm.py`：Mock、OpenAI 兼容同步/流式调用。
- `app/rag/service.py`：离线入库和在线聊天编排。
- `app/api/routes.py`：角色、文档、搜索、聊天、评测接口。

## 3. 数据模型

### Role

`id`、`name`、`category`、`description`、`system_prompt`、`knowledge_scope`、创建时间、修改时间。

### Document

`id`、`filename`、`source`、`content_hash`、`status`、`chunk_count`、`error_message`、创建时间、修改时间。

### Milvus chunk 记录

`id` 主键、`vector` 向量、`text` 原文、`document_id` 文档 ID、`source` 来源、`metadata` JSON。metadata 保存页面、解析器、role_id、分块方式等字段，便于过滤和追踪。

### ConversationMessage

`id`、`user_id`、`role_id`、`conversation_id`、`message_type`、`content`、`citations_json`、创建时间。

## 4. 关键设计决策

1. 默认 fallback：避免开发机必须先安装 GPU、Milvus、Redis 和模型权重。
2. 适配器切换：环境变量控制外部服务，保持 API 与上层编排不变。
3. 混合检索：向量擅长语义，BM25 风格匹配擅长专有名词、法条编号和疾病名称；RRF 组合两者排名。
4. 角色范围：入库 metadata 带有 `role_id`，角色可设 `global` 或专属范围。
5. 引用优先：回答上下文中显式携带 source 和 score，接口返回引用，便于评测与审计。
6. 长短期记忆分离：Redis 保存最近消息，数据库保存完整消息，避免把全部历史塞入 Prompt。

## 5. 可演进方向

- 将 lexical fallback 替换为 rank-bm25 / Elasticsearch / Milvus BM25 原生全文检索。
- 增加父子块、摘要块、文档去重、低质量文档过滤和 Query 扩写。
- 增加 Neo4j 关系召回、MySQL 结构化召回、MongoDB/ClickHouse 业务数据召回和互联网搜索召回。
- 接入 MinerU 进行复杂 PDF 版面、公式和表格抽取。
- 增加内容安全分类、PII 脱敏、用户鉴权、配额、熔断、重试、缓存和 Nginx 限流。
- 使用 vLLM/SGLang 暴露 OpenAI 兼容网关；大模型可选 Qwen、DeepSeek 或本地量化模型。
