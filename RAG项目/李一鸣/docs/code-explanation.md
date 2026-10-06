# 代码解释文档

## 1. 总体架构

项目是一个 FastAPI 后端，核心链路可以概括为：

```text
HTTP 请求
  -> FastAPI 路由
  -> RAGService
  -> 记忆读取 / Query 改写
  -> 向量召回 + BM25 召回
  -> RRF 混合融合
  -> 重排序
  -> 提示词模板
  -> 大模型或 Mock LLM
  -> 正则后处理
  -> 保存消息和引用
  -> 返回 JSON 或 SSE
```

离线知识库链路是：

```text
上传文件
  -> DocumentParser 解析
  -> DocumentChunker 分块
  -> EmbeddingService 向量化
  -> LocalVectorStore 或 Milvus
  -> HybridRetriever 建立词法索引
```

## 2. 程序入口

### `run.py`

读取配置并调用 Uvicorn：

1. 从 `app.core.config` 获取 `Settings`。
2. 使用配置中的 host 和 port 启动 `app.main:app`。

### `app/main.py`

应用初始化和生命周期管理：

1. 创建目录：`data`、上传目录、索引目录和日志目录。
2. 配置 JSON 日志。
3. 初始化数据库表和轻量级 SQLite 字段迁移。
4. 创建默认角色。
5. 创建 `RAGService` 并挂到 `app.state.rag`。
6. 注册 HTTP request id 中间件。
7. 关闭时释放数据库连接和 RAG 资源。

## 3. 配置层

### `app/core/config.py`

`Settings` 使用 Pydantic Settings 从 `.env` 和环境变量读取配置。重要配置包括：

- 路径：`PROJECT_ROOT`、`DATA_DIR`、`UPLOAD_DIR`、`LOCAL_VECTOR_INDEX`、`LOG_DIR`。
- 存储：`DATABASE_URL`、`REDIS_URL`、`MILVUS_URI`。
- 模型：`EMBEDDING_PROVIDER`、`EMBEDDING_MODEL`、`RERANKER_MODEL`、`LLM_BASE_URL`、`LLM_MODEL`。
- 检索：`TOP_K_VECTOR`、`TOP_K_BM25`、`TOP_K_FINAL`、`VECTOR_SCORE_THRESHOLD`。
- 分块：`CHUNK_SIZE`、`CHUNK_OVERLAP`。
- 评测：`RAGAS_ENABLED`。

`get_settings()` 使用缓存，保证同一进程内使用同一份配置。

### `app/core/logging.py`

配置控制台日志和文件日志。日志格式是 JSON，包含 timestamp、level、logger、message、request_id 和业务扩展字段，便于按 trace 排查完整链路。

## 4. API 层

### `app/api/routes.py`

提供以下接口：

- `GET /api/v1/health`：检查数据库、记忆、向量库、模型和 Embedding 状态。
- `GET /api/v1/roles`：获取角色列表。
- `POST /api/v1/roles`：创建自定义角色。
- `GET /api/v1/documents`：获取知识库文档。
- `POST /api/v1/documents/upload`：上传并完成入库。
- `POST /api/v1/search`：执行混合检索和重排。
- `POST /api/v1/chat`：普通或流式聊天。
- `POST /api/v1/evaluate`：离线代理或真实 RAGAS 评测。

路由层负责参数校验、依赖注入、HTTP 状态码、SSE 封装和响应模型转换。RAG 业务编排集中在 `RAGService`，避免路由函数变得复杂。

### `app/storage/schemas.py`

定义 Pydantic 请求和响应模型，例如 `ChatRequest`、`ChatResponse`、`RoleCreate`、`EvaluationRequest`。这些模型自动生成 OpenAPI 文档并执行输入校验。

## 5. 数据持久化层

### `app/core/database.py`

创建 SQLAlchemy Async Engine 和异步 Session：

- 默认：SQLite + `aiosqlite`。
- 生产：MySQL + `aiomysql`。

### `app/storage/models.py`

定义数据库表：

- `users`：用户信息。
- `roles`：角色名称、性格、专业领域、说话风格、安全策略、提示词和知识范围。
- `documents`：文件名、来源、状态、哈希、分块数和错误信息。
- `conversation_messages`：用户和助手消息、会话、引用及时间。

### `app/storage/repositories.py`

封装角色、文档和聊天消息的数据库操作，包括创建角色、查询角色、保存消息、更新文档状态和创建默认角色。

## 6. 离线知识库处理

### `app/rag/parsers.py`

`DocumentParser` 按扩展名处理文件：

- PDF：优先使用 PyMuPDF 提取正文。
- PDF 表格：使用 pdfplumber 补充表格文本。
- 扫描件：预留 PaddleOCR-VL 降级入口。
- TXT/MD/CSV/JSON：使用文本读取和结构化序列化。

输出统一为 `ParsedDocument`，让后续分块逻辑不依赖具体文件格式。

### `app/rag/chunking.py`

`DocumentChunker` 采用结构感知和固定长度重叠策略：

1. 先识别标题、段落和空行。
2. 保留文本顺序和基础 metadata。
3. 超过 `chunk_size` 时切分。
4. 使用 `chunk_overlap` 保留相邻上下文。

每个结果是一个 `ChunkRecord`，包含 chunk id、document id、正文、来源、metadata 和向量。

### `app/rag/embeddings.py`

`EmbeddingService` 提供可替换的向量化实现：

- `hash`：离线确定性向量，适合开发和测试。
- `sentence_transformers`：使用 BGE-m3 等 Sentence Transformers 模型。
- `bge_m3`：预留 FlagEmbedding 原生接口。

文档和查询必须使用同一个向量维度和模型配置。

### `app/rag/vector_store.py`

提供两个后端：

- `LocalVectorStore`：JSON 文件保存向量和原文，开发环境无需启动 Milvus。
- `MilvusVectorStore`：生产环境写入 Milvus collection，并保留结构化 metadata。

`build_vector_store()` 根据 `MILVUS_ENABLED` 选择后端。

## 7. 在线检索链路

### `app/rag/retrievers.py`

`HybridRetriever` 同时执行两路召回：

1. Dense retrieval：查询向量与文档向量计算余弦相似度。
2. Lexical retrieval：使用 `rank_bm25`；不可用时使用轻量词法重叠降级。
3. RRF：按照两路结果的排名进行 Reciprocal Rank Fusion。
4. 过滤：支持按 `role_id` 限制知识范围。

日志会记录 `dense`、`lexical` 和 `fused` 数量。

### `app/rag/reranker.py`

`RerankerService` 对融合候选进行精排：

- 开启配置且模型可用时使用 BGE-Reranker。
- 否则使用查询与候选文本的词法重叠分数。

最终只保留 `top_k_final` 条引用，减少提示词长度。

## 8. 记忆、提示词和生成

### `app/rag/memory.py`

`MemoryService` 负责短期记忆：

- Redis 模式使用 `user_id:role_id:conversation_id` 作为 key。
- Redis 不可用时使用进程内字典降级。
- 保存最近若干轮消息，最多由 `MAX_MEMORY_MESSAGES` 控制。

长期消息由 `RAGService._save_chat_turn()` 保存到数据库。

### `app/rag/prompt.py`

根据角色系统提示词、历史消息、召回引用和当前问题构造大模型 messages。提示词要求模型：

- 遵循角色性格和说话风格。
- 优先依据知识库上下文。
- 信息不足时明确说明。
- 高风险领域给出边界和专业求助提示。

### `app/rag/llm.py`

`LLMService` 兼容 OpenAI Chat Completions 协议：

- `complete()` 返回完整回答。
- `stream_complete()` 解析 `data:` SSE 增量内容。
- `mock` 模式不调用外部服务，用于离线验证。

因此 DeepSeek、千问、豆包兼容网关、vLLM 和 SGLang 都可以通过 `LLM_BASE_URL` 切换。

### `app/rag/postprocess.py`

生成后处理负责清理多余空白、重复标记、异常引用格式和不希望返回给用户的文本。它是生成结果的最后一道轻量校验。

## 9. `RAGService` 主编排

### `app/rag/service.py`

`RAGService` 是核心业务服务，主要方法：

- `ingest_file()`：完成解析、分块、向量化、写入索引和更新数据库状态。
- `chat()`：完成记忆读取、Query 改写、混合召回、重排、提示词构建、生成、后处理、保存消息。
- `stream_chat()`：与 `chat()` 使用同样的上下文准备逻辑，以 SSE 逐块返回回答。
- `rewrite_query()`：短问题结合上一轮用户问题进行补全。

`chat()` 会记录 memory、query rewrite、retrieval、rerank、llm 和 total 等耗时字段，便于性能分析。

## 10. 评测模块

### `app/rag/evaluation.py`

评测有两种模式：

- `local_proxy`：使用词法重叠近似 faithfulness、relevancy 和 correctness，完全离线。
- `ragas`：显式开启 `RAGAS_ENABLED=true`，使用 OpenAI 兼容模型执行 RAGAS 评测。

RAGAS 导入包含兼容 shim，用于处理 RAGAS 0.4.x 与 `langchain-community` Vertex AI 模块路径变化。

## 11. 启动与依赖目录

### `run_zhuangao6.ps1`

这是 Windows 当前唯一推荐启动脚本。它固定使用：

```text
D:\develop_tool1\anaconda3\envs\zhuangao6\python.exe
```

并设置 D 盘依赖覆盖目录。项目根目录中的 `conda_packages` 不是源码，而是当前 Conda 环境由于权限限制采用的运行依赖目录，不能删除。

### `requirements.txt`

记录项目需要的 Python 依赖。新增或升级依赖后，应使用 D 盘目标目录重新安装：

```powershell
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "D:\rag-roleplay-system\conda_packages"
conda run -n zhuangao6 python -s -m pip install --ignore-installed --target D:\rag-roleplay-system\conda_packages -r requirements.txt
```

## 12. 测试模块

- `tests/test_chunking.py`：分块边界和重叠。
- `tests/test_memory.py`：短期记忆顺序和隔离。
- `tests/test_postprocess.py`：回答清洗。
- `tests/test_retrieval.py`：向量和词法混合召回。
- `tests/test_evaluation.py`：离线评测默认行为。

测试只验证业务代码，不会删除数据库或向量索引。
