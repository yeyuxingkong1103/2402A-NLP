# 法律 RAG 离线 Pipeline 与 MySQL 入库设计

## 1. 总体架构

法律 RAG 包含两个业务 Pipeline：

1. **离线 Pipeline**：把授权官方法律数据加工为稳定、可审计的数据包；
2. **检索 Pipeline**：根据用户问题执行查询理解、召回、重排、上下文扩展和回答生成。

两个 Pipeline 通过 MySQL 文档数据和 Milvus 向量索引衔接。入库脚本与索引脚本是持久化工具，不定义为第三个 Pipeline。

当前阶段以以下主链路为核心，并预留认证、记忆和混合检索的完整边界：

```text
离线 Pipeline → 标准数据包 → MySQL 权威存储与关键词索引 → Milvus 向量索引
认证用户 → Redis 短期记忆 / Milvus 长期记忆 → 受权限约束的检索融合
```

Milvus 部署后再实现向量索引脚本，之后开发检索 Pipeline。

## 2. 当前状态

项目已有相互独立且经过单元测试的采集、原始 HTML 保存、解析、清洗、父子分块和 Embedding 客户端，但尚无统一离线 Pipeline、标准数据包、MySQL 持久化和检索 Pipeline。

本阶段复用现有处理组件，不修改解析与分块算法。

## 3. 本阶段范围

### 3.1 包含

- 串联采集、解析、清洗、分块和校验的离线 Pipeline；
- 可重复消费的标准数据包及 manifest；
- 数据包原子发布和完整性校验；
- SQLAlchemy 2.x MySQL 数据层；
- 文档、版本、采集记录、分块和导入记录模型；
- 独立 MySQL 入库脚本；
- 幂等导入和事务回滚；
- 用户、登录凭证、会话和权限基础模型；
- Redis 短期记忆与会话状态设计；
- Milvus 法律混合检索索引和用户长期记忆设计；
- SQLite 内存单元测试；
- 不含密钥的环境变量示例和依赖声明。

### 3.2 不包含

- 未经授权来源的正式法律数据采集；
- 绕过登录、验证码、WAF 或其他访问控制；
- 生产级分布式锁、任务编排和部署编排；
- 具体大模型供应商的生产 SLA；
- 与法律知识库混用的用户记忆 collection。

## 4. 离线 Pipeline

### 4.1 职责

离线 Pipeline 只处理文档，不连接 MySQL、Redis 或 Milvus：

```text
授权官方 URL
  → 白名单、robots.txt 与来源限速校验
  → 下载并保存原始文件
  → 解析与文本清洗
  → 法律条文父子分块
  → 数据质量校验
  → 生成标准数据包
```

Pipeline 通过依赖注入接收 crawler、解析器、分块器、时钟和 ID 生成器，不依赖隐藏的全局客户端。

### 4.2 标准数据包

每次成功运行生成一个不可变目录：

```text
artifacts/<package_id>/
├── manifest.json
├── documents.jsonl
├── document_versions.jsonl
├── document_chunks.jsonl
├── crawl_records.jsonl
└── raw/
    └── <content_hash>.<extension>
```

文件职责：

- `manifest.json`：Schema 版本、Pipeline 版本、`package_id`、生成时间、来源、记录数和各文件 SHA-256；
- `documents.jsonl`：稳定文档标识、来源 URL 和标题；
- `document_versions.jsonl`：内容哈希、媒体类型、清洗正文和原始文件相对路径；
- `document_chunks.jsonl`：父子关系、条号、顺序、正文和检索增强文本；
- `crawl_records.jsonl`：采集结果、HTTP 状态和来源信息；
- `raw/`：原始 HTML、PDF 或文本文件。

数据包不得包含 API Key、数据库密码或其他连接凭据。

### 4.3 原子发布

Pipeline 先写入同级临时目录，完成以下校验后再原子重命名为最终目录：

- 必需文件均存在；
- JSON/JSONL Schema 合法；
- manifest 记录数与实际一致；
- 文件 SHA-256 一致；
- 文档、版本和 chunk 引用关系完整；
- 所有子块均引用同版本中的父块；
- `chunk_id` 在数据包内唯一；
- 原始文件路径不能逃逸数据包目录。

失败时不发布最终目录，也不删除 Pipeline 启动前已存在的文件。

## 5. 数据标识与版本

- `document_id`：由规范化 `source_url` 稳定生成；
- `content_hash`：由原始响应内容生成 SHA-256；
- `document_version_id`：由 `document_id + content_hash` 稳定生成；
- `chunk_id`：沿用现有跨文档唯一、长度受限的稳定 ID；
- `package_id`：唯一标识一次完整 Pipeline 产物；
- `schema_version`：定义数据包结构兼容性；
- `pipeline_version`：标识清洗、解析和分块规则版本。

Pipeline 不查询数据库，因此不判断 MySQL 中的 `new`、`updated` 或 `unchanged`。这些状态由入库脚本根据已有数据判定。

## 6. MySQL 数据层

### 6.1 配置与连接

- 生产环境通过 `DATABASE_URL` 注入 MySQL 连接；
- 配置模块不打印密码或完整连接 URL；
- 测试显式注入 SQLite 内存 URL；
- Repository 接收 Session，不创建全局连接；
- 使用 SQLAlchemy 2.x typed declarative；
- 登录认证使用短时 JWT access token 和可撤销的 refresh token；
- 密码只保存经过强哈希的密码摘要，不保存明文密码；
- 所有记忆和检索请求必须从认证上下文获得 `user_id`，禁止由客户端任意指定归属用户。

### 6.2 数据模型

#### documents

- `id`：数据库主键；
- `document_key`：稳定业务标识，唯一；
- `source_url`：规范化官方来源 URL，唯一；
- `title`；
- `current_version_id`，可为空；
- `created_at`、`updated_at`。

#### document_versions

- `id`；
- `version_key`：稳定版本标识，唯一；
- `document_id`；
- `content_hash`；
- `raw_file_path`；
- `media_type`；
- `cleaned_content`；
- `version_status`：`new` 或 `updated`；
- `processing_status`：`awaiting_embedding`、`indexed` 或 `failed`；
- `created_at`。

`document_id + content_hash` 建立唯一约束。

#### crawl_records

- `id`；
- `package_id`；
- `source_url`；
- `document_id`，可为空；
- `document_version_id`，可为空；
- `status`：`success`、`unchanged` 或 `failed`；
- `http_status`；
- `content_hash`；
- `raw_file_path`；
- `failed_stage`；
- `error_summary`；
- `created_at`。

错误摘要必须截断、脱敏，不保存完整响应正文。

#### document_chunks

- `id`；
- `chunk_id`：稳定业务 ID，唯一；
- `document_version_id`；
- `parent_chunk_id`，父块为空；
- `chunk_type`：`parent` 或 `child`；
- `article_number`；
- `sequence`；
- `content`；
- `retrieval_text`；
- `created_at`。

MySQL 不保存 1024 维向量。

#### import_records

- `id`；
- `package_id`：唯一；
- `schema_version`；
- `pipeline_version`；
- `status`：`processing`、`imported`、`already_imported` 或 `failed`；
- `document_version_id`，可为空；
- `error_summary`；
- `created_at`、`updated_at`。

该表用于审计和数据包级幂等，不替代后续 Redis 短期任务状态。

### 6.3 认证与用户隔离

MySQL 负责身份和授权基础数据，至少包括 `users`、`user_sessions` 或等价模型：

- 用户名或邮箱唯一；
- 密码保存为强哈希摘要，支持账户停用；
- access token 短时有效，refresh token 可撤销并记录过期时间；
- 普通用户只能访问自己的会话、短期记忆和长期记忆；
- 管理员管理账户状态，但默认不能读取用户私密记忆正文；
- 认证失败、账户停用和 token 失效均返回统一的非敏感错误信息。

`user_id` 是所有记忆隔离的强制字段。服务端不得信任请求体中的 `user_id`，必须从已验证的认证上下文注入。

### 6.4 短期记忆与长期记忆

短期记忆使用 Redis，长期记忆使用独立的 Milvus `user_memories` collection：

```text
登录用户
  → 认证上下文 user_id
  → Redis 读取当前会话最近消息/摘要
  → Milvus 按 user_id 过滤召回长期记忆
  → 与法律知识混合检索结果合并
  → 重排与回答
```

Redis key 必须同时包含 `user_id` 和 `session_id`，并设置 TTL。短期记忆包括最近消息、会话摘要、临时任务状态和检索缓存；过期不影响长期记忆。

长期记忆写入前必须经过摘要、去重、敏感信息过滤和存储价值判断。Milvus 记录至少包含 `memory_id`、`user_id`、可选 `role_id`/`session_id`、`memory_type`、`content` 或摘要、`created_at`、`deleted` 和 embedding。检索始终过滤当前 `user_id` 与 `deleted=false`；删除时按 `memory_id` 删除或标记为不可检索。

长期记忆不写入 MySQL，MySQL 不承担长期记忆正文或向量的第二份存储。不得把密码、API Key、身份证号等不必要的敏感信息写入记忆。

### 6.5 混合检索数据流

法律文档同时进入 MySQL 和 Milvus，不能只写 SQL：

1. MySQL 保存文档、版本、清洗正文、父子 chunk、来源和状态，作为权威数据源；
2. MySQL 为 `content` 和 `retrieval_text` 建立关键词检索能力，生产环境优先使用 `FULLTEXT`；
3. 索引阶段从 MySQL 读取待索引 chunk，生成 dense embedding；
4. Milvus 独立的 `legal_documents` collection 保存法律 chunk 向量和必要过滤字段；
5. 在线检索同时执行 MySQL 关键词召回和 Milvus 向量召回，再合并去重、过滤、融合排序；
6. Milvus 返回 `chunk_id` 后，从 MySQL 获取权威正文、父块和来源信息，再执行 rerank 与引用组装。

`legal_documents` 与 `user_memories` 必须是两个独立 collection。法律知识检索默认不混入用户记忆；需要融合时也必须分别召回、分别过滤、分别标注来源。

## 7. MySQL 入库脚本

命令边界：

```bash
python -m app.cli.import_mysql --package artifacts/<package_id>
```

脚本按以下顺序执行：

1. 定位并读取 manifest；
2. 校验 Schema 版本、记录数、文件哈希和路径安全；
3. 校验文档、版本和父子 chunk 引用；
4. 检查 `package_id` 是否已成功导入；
5. 开启 MySQL 事务；
6. 按 `source_url` 查找或创建逻辑文档；
7. 根据内容哈希判定 `new`、`updated` 或 `unchanged`；
8. 对新版本写入版本、父子 chunk 和采集记录；
9. 将新版本标记为 `awaiting_embedding`；
10. 更新当前版本并记录导入结果；
11. 提交事务并输出不含敏感信息的报告。

入库脚本不重新爬取、解析、清洗或分块，也不修改数据包。

## 8. 幂等与事务规则

- 相同 `package_id` 已成功导入时返回 `already_imported`；
- 相同 `source_url + content_hash` 返回 `unchanged`，不创建重复版本或 chunk；
- 同一 URL 首个内容版本标记为 `new`；
- 同一 URL 的新内容哈希标记为 `updated`；
- `chunk_id` 唯一，重复数据不得产生重复块；
- 新版本及其所有 chunk 成功写入后才更新 `current_version_id`；
- 核心记录任一写入失败时回滚整个事务；
- 数据包和原始文件保持不变，以便修复后安全重试。

## 9. 错误处理

### 9.1 Pipeline 阶段

固定阶段：`crawl`、`parse`、`chunk`、`validate`、`publish`。

失败时返回阶段和脱敏错误，不发布最终数据包，不伪造成功状态。

### 9.2 入库阶段

固定阶段：`load`、`validate`、`persist`。

- 加载或校验失败时不连接或不修改数据库；
- 持久化失败时回滚事务；
- 导入失败不得留下 `imported` 状态；
- 报告不得包含密码、API Key 或完整数据库 URL。

## 10. 测试策略

遵循 TDD，先写失败测试再实现。

### 10.1 数据包测试

使用 Fake crawler 和 pytest 临时目录覆盖：

- 爬取、解析、清洗、分块到数据包的完整流程；
- manifest 记录数和文件哈希；
- 父子 chunk 引用完整性；
- 非法路径和被篡改文件被拒绝；
- 失败不发布半成品；
- 测试数据不污染项目正式 `data` 或 `artifacts` 目录。

### 10.2 Repository 与入库测试

使用 SQLite 内存数据库覆盖：

- 首次导入创建 `new` 版本；
- 相同内容导入返回 `unchanged`；
- 内容变化创建 `updated` 版本；
- 父子 chunk 完整保存；
- 重复 `package_id` 返回 `already_imported`；
- 重复 `chunk_id` 不产生重复数据；
- 中途异常时事务整体回滚；
- 安全摘要不泄漏数据库密码。

### 10.3 验证命令

- 定向单元测试；
- 完整 `pytest`；
- `python -m compileall -q app tests`；
- MySQL 可用后再运行显式启用的真实集成测试。

## 12. 后续向量索引与检索 Pipeline

索引阶段为法律知识生成 dense embedding，并写入 Milvus 向量索引：

```text
MySQL 查询 awaiting_embedding 的子块
  → 生成 dense embedding
  → Milvus legal_documents 按 chunk_id upsert
  → MySQL 将版本或 chunk 标记为 indexed
```

用户长期记忆独立写入 `user_memories` collection，不进入法律文档 collection：

```text
Redis 短期消息/摘要
  → 摘要、去重、敏感信息过滤
  → 生成长期记忆 embedding
  → Milvus user_memories 按 user_id 写入
```

之后实现在线检索 Pipeline：

```text
已认证用户问题
  → 查询清洗与改写
  → MySQL 关键词召回
  → Milvus 法律向量召回
  → 按 user_id 过滤召回长期记忆
  → Redis 读取短期记忆
  → 合并关键词结果、向量结果和记忆结果
  → 以 chunk_id 从 MySQL 获取权威正文、父块和来源
  → 分别标注法律知识与用户记忆来源
  → Reranker 重排
  → 上下文组装
  → LLM 回答与来源引用
```

索引失败时 MySQL 保持 `awaiting_embedding`，确保能够重试。检索 Pipeline 只消费已成功索引的数据，并且必须在认证和权限过滤完成后调用记忆检索。

## 13. 本阶段验收标准

使用一份授权官方法律文档完成受控验证后，应能：

- 通过一个离线 Pipeline 命令生成完整、原子发布的数据包；
- 在不连接 MySQL 的情况下检查清洗正文和父子 chunk；
- 通过独立脚本把数据包事务导入 MySQL；
- 查询到逻辑文档、版本、清洗正文、采集记录和完整父子块；
- 新版本状态为 `awaiting_embedding`，不会在尚未完成 Milvus 索引时伪造 `indexed`；
- 认证用户只能访问自己的会话和记忆，未认证请求不能访问记忆接口；
- Redis 短期记忆按 `user_id`/`session_id` 隔离并遵守 TTL；
- Milvus 建立独立的 `legal_documents` 和 `user_memories` collection；
- 法律数据支持 MySQL 关键词检索 + Milvus 向量检索，且返回结果能回查 MySQL 权威正文；
- 长期记忆只写入 `user_memories`，按 `user_id` 过滤，删除后不可被召回；
- 重复处理或导入不会产生重复版本和 chunk；
- 篡改或不完整的数据包不能进入数据库；
- 所有认证、记忆和检索失败均返回脱敏错误，不泄露凭据或其他用户内容。
