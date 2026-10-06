# 技术架构设计

> 项目：MentalHeal RAG  
> 版本：v0.1

## 1. 总体架构

这个项目可以理解成 4 层：

1. 前端页面：用户聊天、心理健康助手、知识库管理。
2. 后端服务：接口、业务逻辑、RAG 编排、权限和日志。
3. AI 能力层：PDF 解析、OCR、Embedding、Rerank、大模型。
4. 数据存储层：MySQL、Redis、Milvus。

```mermaid
flowchart TD
    U[用户浏览器] --> F[Vue 前端]
    F --> A[FastAPI 后端]

    A --> C[心理健康聊天业务]
    A --> R[RAG 编排服务]
    A --> I[文档导入服务]

    I --> P[PDF 解析]
    I --> O[OCR / MinerU]
    I --> E[BGE-m3 向量化]
    E --> M[Milvus]

    R --> Q[问题改写/扩写]
    Q --> V[向量检索]
    Q --> B[BM25 关键词检索]
    V --> RR[BGE-rerank]
    B --> RR
    RR --> L[DeepSeek 大模型]
    L --> S[安全校验和后处理]

    A --> DB[(MySQL)]
    A --> RD[(Redis)]
    A --> M[(Milvus)]
```

## 2. 离线知识库链路

离线链路负责把 PDF 变成可检索的知识。

```mermaid
flowchart LR
    A[本地 PDF] --> B[文档类型判断]
    B --> C[PyMuPDF]
    B --> D[pdfplumber]
    B --> E[MinerU]
    B --> F[PaddleOCR]
    C --> G[文本清洗]
    D --> G
    E --> G
    F --> G
    G --> H[分块]
    H --> I[摘要/元数据]
    I --> J[BGE-m3]
    J --> K[Milvus]
    I --> L[MySQL]
```

关键点：

- 普通 PDF 优先用 PyMuPDF。
- 表格多的 PDF 用 pdfplumber 辅助。
- 扫描版或图片型 PDF 必须走 OCR。
- 复杂版面可以接 MinerU。
- 分块不要太碎，也不要太长。
- 每个 chunk 都要保留来源、页码、文档 ID、心理健康分类。

## 3. 在线问答链路

在线链路负责回答用户问题。

```mermaid
sequenceDiagram
    participant User as 用户
    participant Web as 前端
    participant API as FastAPI
    participant Redis as Redis短期记忆
    participant Milvus as Milvus知识库
    participant MySQL as MySQL元数据
    participant Rerank as BGE-rerank
    participant LLM as DeepSeek

    User->>Web: 输入问题
    Web->>API: POST /api/v1/chat
    API->>Redis: 读取最近聊天记录
    API->>Milvus: 向量检索
    API->>MySQL: 查询助手/文档元数据
    API->>Rerank: 精排候选片段
    API->>LLM: 提示词 + 上下文 + 问题
    LLM-->>API: 生成回答
    API->>API: 后处理和安全校验
    API->>Redis: 保存本轮对话
    API-->>Web: 返回回答和来源
    Web-->>User: 展示回答
```

## 4. 数据存储分工

## 4.1 MySQL

MySQL 保存结构化数据：

- 用户信息。
- 心理健康助手配置。
- 文档信息。
- 文档分块元数据。
- 聊天会话信息。
- 聊天消息记录。
- RAG 评测记录。

## 4.2 Redis

Redis 保存短期、高频、临时数据：

- 最近聊天记录。
- 热门问题缓存。
- 文档解析任务状态。
- 临时验证码或 token。

## 4.3 Milvus

Milvus 保存语义向量：

- 知识库 chunk 向量。
- 长期记忆向量。

## 5. 后端模块设计

```text
backend/app/
  api/          只放接口路由
  core/         配置、日志、异常、安全基础能力
  db/           MySQL、Redis、Milvus 连接
  models/       SQLAlchemy 数据库模型
  schemas/      Pydantic 请求和响应结构
  services/     业务服务
  rag/          检索、重排、提示词、生成、后处理
  ingestion/    PDF 解析、OCR、清洗、分块、向量化
  memory/       短期记忆和长期记忆
  security/     心理危机检测、内容安全
```

建议原则：

- `api` 不写复杂业务，只接收请求、调用 service、返回结果。
- `services` 负责业务流程。
- `rag` 负责 RAG 链路。
- `ingestion` 负责知识库入库链路。
- `db` 只负责连接和底层封装。

## 6. 前端页面设计

一期可以做 4 个页面：

1. 登录页：简单登录或游客进入。
2. 心理健康聊天页：核心演示页面。
3. 知识库管理页：上传心理健康 PDF、查看解析状态。
4. RAG 评测页：查看测试问题和评测结果。

后续再加：

- 心理健康助手配置页。
- RAG 评测页。
- 系统监控页。

## 7. 心理健康安全策略

系统需要内置最基本的安全策略：

- 不做诊断。
- 不替代医生。
- 不鼓励危险行为。
- 检测到自杀、自伤、伤害他人时，优先安全引导。
- 建议用户联系专业机构、紧急服务、可信任的人。
- 回答语气必须温和，不责备用户。

## 8. 可扩展设计

本项目不做多行业角色扩展，只在心理健康范围内做配置化扩展：

- 心理健康助手有独立 prompt。
- 心理健康助手绑定心理健康知识库。
- 不同心理健康场景可以配置不同语气，例如压力管理、睡眠健康、焦虑自助。
- 心理危机安全规则始终启用。
- 如果用户问非心理健康问题，系统应礼貌说明当前只支持心理健康相关内容。

## 9. 部署架构

一期开发环境：

```text
本机浏览器
  -> 前端 Vite dev server
  -> FastAPI
  -> 本机 MySQL
  -> Docker Redis
  -> Docker Milvus
  -> DeepSeek API
  -> MinerU API
```

生产环境：

```text
Nginx
  -> 前端静态资源
  -> FastAPI 多进程
  -> MySQL
  -> Redis
  -> Milvus
  -> 日志目录
  -> 外部大模型 API / 本地模型服务
```

## 10. 一期最小可运行版本

最小可运行版本不要一开始追求特别复杂，先做到：

1. 后端能启动。
2. 能连接 MySQL、Redis、Milvus。
3. 能导入一个 PDF。
4. 能解析 PDF 和 OCR。
5. 能向量化并写入 Milvus。
6. 用户能问问题。
7. 系统能检索知识库并调用 DeepSeek 回答。
8. 前端能显示聊天内容和来源。

这个版本跑通后，再逐步优化混合检索、RAGAS、压力测试、部署脚本。
