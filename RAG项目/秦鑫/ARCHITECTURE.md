# LAW-RAG 系统整体架构

本文档依据 2026 年 9 月 2 日工作区中的实际代码整理，描述当前系统的模块划分、请求链路、数据流和部署方式。

## 1. 系统定位

LAW-RAG 是面向法律场景的检索增强生成系统，主要能力包括：

- 基于公共法律知识库的法律问答；
- 基于用户上传材料的私有文档检索；
- 多轮会话、短期记忆和可选长期记忆；
- 检索过程流式展示；
- 法律解决方案生成和 PDF 导出；
- 用户账号、工作区文件、活动记录和注销申请管理；
- 公共数据清洗、解析、切分、向量化和 Milvus 入库。

系统采用前后端分离的逻辑结构，但运行时可以由同一个 FastAPI 进程同时提供 API 和 `frontend` 静态页面。

## 2. 总体结构

```text
浏览器 / 桌面客户端
        |
        | HTTP / JSON / multipart / SSE
        v
FastAPI 应用（backend/app/main.py）
        |
        +-- 系统与日志：system.py
        +-- 认证与用户：auth.py
        +-- 工作区文件：workspace/
        +-- 记忆服务：memory/
        +-- RAG 问答：rag/
        |
        +-- MySQL：账号、文件、历史、长期记忆、用户状态
        +-- Redis：认证会话、短期记忆、缓存、限流辅助数据
        +-- Milvus：公共法律库和用户私有文档向量
        +-- 外部模型服务：LLM、Embedding、Reranker、多模态/OCR
        +-- Celery 或本地线程：后台文件处理和记忆任务
```

## 3. 代码组织

```text
backend/app/
├── main.py              # FastAPI 应用、生命周期、中间件和路由注册
├── config.py            # 环境变量和运行参数
├── core.py              # 通用安全、标识和公共集合定义
├── auth.py              # 注册、登录、会话、用户资料和注销申请
├── system.py            # 服务装配、健康检查、结构化日志和请求上下文
├── tasks.py             # Celery 任务及本地线程任务兜底
├── models/              # LLM、Embedding、Reranker 和多模态模型网关
├── storage/             # MySQL、Redis、Milvus 存储适配器
├── workspace/           # 文件上传、解析、快照、索引和私有检索
├── memory/              # 短期记忆、长期记忆、冲突和指代消解
└── rag/                 # 查询理解、检索规划、证据处理、上下文和答案生成

data_pipeline/
├── load.py              # 文本、PDF、DOCX、XLSX、JSON 和图片加载/OCR
├── parse.py             # 法律文档类型识别和结构化解析
├── clean.py             # 数据清洗和质量校验
├── chunk.py             # 语义切分、父子块和结构化记录切分
├── embedding.py         # 批量向量化和向量格式处理
└── index.py             # 公共数据构建、私有文档处理和 Milvus 入库

frontend/
├── index.html           # 前端入口
└── src/                 # 前端交互逻辑和样式
```

## 4. 应用启动和服务装配

### 4.1 FastAPI 生命周期

`backend/app/main.py` 创建 FastAPI 应用，并在 lifespan 中完成以下工作：

1. 读取 `Settings`；
2. 调用 `create_services(settings)` 创建所有基础设施和领域服务；
3. 将服务集合放入 `app.state.services`；
4. 注册请求日志中间件、CORS 中间件和统一异常处理器；
5. 注册系统、认证、用户、工作区、记忆和 RAG 路由；
6. 应用退出时关闭日志资源。

核心服务实例包括：

```text
mysql -> UserStore / FileStore / ConversationHistoryStore / LongTermMemoryStore
redis -> AuthSessionStore / ShortMemoryRedisStore / RAG 方案缓存
milvus -> 公共法律集合 / 用户私有文档集合
model -> ModelGateway
memory -> MemoryOrchestrator
workspace -> WorkspaceService
rag -> RagWorkflow
```

### 4.2 后台任务

`run.py` 启动服务时会尝试启动 Celery worker。Celery 不可用或 worker 启动失败时，系统回退到进程内线程池，保证文件处理仍可执行。

当前任务主要包括：

- `law_rag.process_workspace_upload`：用户文件后台解析、切分、向量化和入库；
- `law_rag.process_user_document`：通用用户文档入库；
- `law_rag.ocr_document`：OCR 任务；
- `law_rag.daily_memory_job`：长期记忆提取任务。

## 5. RAG 问答链路

一次 `/api/v1/legal/ask` 请求的主要处理过程如下：

```text
请求进入
  -> 读取可选登录用户和会话
  -> 限流检查
  -> 查询理解 QueryUnderstanding
  -> 生成检索计划 SearchRouter
  -> 并行访问公共库、私有材料和可选 Web
  -> 多路结果合并（RRF）
  -> Reranker 重排
  -> EvidenceBuilder 构造证据
  -> ContextBuilder 控制上下文预算
  -> AnswerGenerator 生成法律回答
  -> 保存会话历史和短期记忆
  -> 返回答案、来源和处理元数据
```

### 5.1 查询理解

`rag/understand.py` 将用户自然语言问题转换为可检索的结构化信息，包含问题类型、关键词、实体、时间范围、法律领域和检索意图等。对于登录用户，还会结合会话记忆和指代消解后的问题。

问题理解结果还包含上层问答路线 `route`，由大模型优先识别，并由规则进行保守兜底。当前路线包括：

```text
direct / law_only / case_law / evidence_strategy / procedure_guide
current_law_web / document_review / calculation / full_retrieval
```

其中 `document_review` 优先检索用户上传材料并结合相关法律依据，`calculation` 优先召回赔偿、补偿、利息、违约金等计算依据；两者默认不联网。明确要求最新法规、政策或官方信息时使用 `current_law_web`。

### 5.2 检索规划和召回

`rag/plan.py` 根据查询理解结果和 `route` 决定召回渠道、目标集合和集合级检索参数。`rag/search.py` 负责公共集合、私有工作区和联网结果的检索、合并、重排及证据构建。

公共集合当前包括：

```text
civil_code_articles
civil_interpretations
civil_cases
civil_elements
civil_evidence
civil_processes
civil_questions
civil_citations
```

用户上传材料使用按 Embedding 维度命名的私有集合，例如 `user_upload_documents_1024`，并通过 `user_id`、`document_id` 过滤。

### 5.3 证据和答案生成

`rag/context.py` 按上下文 token 预算组织证据、历史和用户画像。`rag/answer.py` 调用模型生成回答和法律解决方案，并提供 Markdown 到 PDF 的渲染能力。低置信度结果会记录结构化告警，便于后续人工复核。

### 5.4 流式输出

`RagWorkflow.stream` 将检索状态、思考摘要、步骤、答案片段和最终结果转换为 SSE 事件。当前事件类型为：

```text
thinking -> 检索/生成过程摘要
status   -> 当前处理状态
step     -> 当前步骤
chunk    -> 最终回答片段
complete -> 完整结果
error    -> 流式处理异常
```

## 6. 用户工作区链路

工作区服务位于 `backend/app/workspace/`，主要由以下模块组成：

- `api.py`：文件上传、批量上传、列表和删除接口；
- `service.py`：文件生命周期、处理调度和状态更新；
- `file_utits.py`：扩展名、文件名、路径、抽取元数据和本地存储工具；
- `search.py`：用户私有材料检索；
- `__init__.py`：公共导出和兼容入口。

文件处理状态通常经历：

```text
上传保存 -> processing -> ready
                    \-> stored（原文件保留，但暂未建立向量索引）
                    \-> failed
```

处理步骤为：保存原文件、抽取文本、OCR/多模态处理、识别文档类型、切分文本、Embedding、写入私有 Milvus 集合、保存处理快照和更新 MySQL 文件状态。

当后台队列不可用、索引暂时失败或文本不可检索时，系统保留原文件并返回降级状态，避免上传请求直接丢失用户材料。

## 7. 记忆架构

`MemoryOrchestrator` 位于 `backend/app/memory/api.py`，统一编排以下能力：

```text
short_memory.py  -> Redis 短期对话、锁、压缩和会话摘要
long_memory.py   -> 长期事实提取、检索、合并、删除和冲突处理
resolver.py      -> 对“他、这个、上述”等指代进行上下文消解
storage/mysql.py -> 原始消息、摘要、长期记忆和冲突记录持久化
```

每轮问答会保存用户消息和助手消息，并尝试同步更新短期记忆。加载上下文时，服务按需读取最近消息、短期摘要、原始消息回查、用户画像和已启用的长期记忆。

会话删除只清理短期记忆和会话级派生状态，当前实现保留历史记录、长期记忆、工作区向量和原始材料，防止误删用户数据。

长期记忆默认关闭，用户开启后才参与检索和每日记忆提取；冲突记录可由用户选择接受新值、保留旧值或丢弃。

## 8. 数据存储设计

### 8.1 MySQL

当前存储层覆盖以下数据：

- `users`：账号和密码哈希；
- `user_sessions`：认证会话；
- `user_files`：上传文件和处理状态；
- `user_settings`、`user_profiles`：用户设置和画像；
- `user_activity`：用户活动记录；
- `user_deletion_requests`：账号注销状态；
- `chat_messages`：会话消息历史；
- `conversation_summaries`、`conversation_messages`：会话摘要和消息索引；
- `long_term_memory`：长期用户记忆；
- `memory_conflicts`、`memory_jobs`：记忆冲突和任务状态；
- `case_memories`：兼容旧版案件记忆数据。

### 8.2 Redis

Redis 主要用于认证会话、登录失败计数、短期记忆、问答方案缓存和过期数据。会话历史及短期记忆有效期由 `HISTORY_TTL` 等配置控制。

### 8.3 Milvus

Milvus 保存公共法律知识和用户上传材料的向量。公共集合按照法律数据类型拆分；私有集合按用户和文档过滤，避免不同用户之间的数据串读。

### 8.4 文件系统

文件系统保存上传原文件、处理快照、公共数据处理结果、质量报告和结构化日志。用户处理快照位于配置的数据目录下，并按用户 ID 和文档 ID 隔离。

## 9. 数据处理流水线

`data_pipeline` 将原始法律数据或用户文件转换为可检索记录：

```text
load
  -> parse
  -> clean / validate
  -> chunk
  -> embedding
  -> index
```

### 9.1 加载

`load.py` 支持文本、Markdown、CSV/TSV、JSON、PDF、DOCX、XLSX 和常见图片。图片通过 OCR 提取中文和英文文本。

### 9.2 解析和清洗

`parse.py` 识别法律条文、司法解释、案例、证据、流程、问题和引用等数据类型，并构造结构化记录。`clean.py` 负责占位数据过滤、字段规范化和质量检查。

### 9.3 切分和向量化

`chunk.py` 对普通长文档进行按句切分，对较长文本构造父子块；对结构化法律记录保留单条记录边界。`embedding.py` 使用批量请求、缓存和维度校验生成向量。

### 9.4 入库

`index.py` 将公共记录写入对应公共集合，将用户文档写入私有集合，并生成处理快照和质量报告。

## 10. 安全和可观测性

- 密码只保存哈希值，不通过用户接口返回；
- Cookie 默认使用 `HttpOnly` 和 `SameSite=Lax`；
- 私有检索同时校验用户 ID 和文档 ID；
- 文件名、文档 ID 和快照路径经过规范化，阻断路径穿越；
- 注册、登录失败、上传、后台任务、检索低置信度和异常均写入结构化日志；
- 日志支持轮转、记录截断和哈希链完整性校验；
- 请求上下文包含 request ID、trace ID、user ID、session ID 和客户端地址；
- RAG 接口对匿名用户和登录用户使用不同的滑动窗口限流。

## 11. 部署配置

主要运行参数来自环境变量，当前默认值包括：

| 配置项 | 默认值 | 作用 |
|---|---|---|
| `SERVER_HOST` | `0.0.0.0` | 服务监听地址 |
| `SERVER_PORT` | `7294` | 服务监听端口 |
| `EXPOSE_DOCS` | `true` | 是否开放 OpenAPI 文档 |
| `EMBEDDING_DIM` | `1024` | 向量维度 |
| `EMBEDDING_BATCH_SIZE` | `32` | Embedding 批量大小 |
| `MAX_UPLOAD_BYTES` | `20 MB` | 单文件上传上限 |
| `WORKSPACE_BACKGROUND_WORKERS` | `4` | 工作区后台线程数 |
| `HISTORY_TTL` | `604800` 秒 | 会话/缓存有效期 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 默认大语言模型 |

生产环境至少需要正确配置 MySQL、Redis、Milvus、模型服务地址和 API Key，并根据部署方式设置 CORS 和 `AUTH_COOKIE_SECURE`。
