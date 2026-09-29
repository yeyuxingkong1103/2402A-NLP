# 法律 RAG 收敛路线实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 按已确认的首期边界，依次打通官方采集、离线标准数据包、MySQL 文档入库、人工审核发布、Embedding/Milvus 索引和单一法律知识助手在线问答闭环。

**架构：** 系统分为离线知识处理 Pipeline 和在线问答 Pipeline。离线链路先生成不可变数据包，再由独立导入服务写入 MySQL；只有人工审核并发布的版本才允许进入 Embedding 和 Milvus。在线链路从 Redis 读取短期会话记忆，使用 Milvus 召回、MySQL 做权限与法律状态过滤，再经 Reranker 和 LLM 生成带引用的 SSE 回答。

**技术栈：** Python 3.11+、FastAPI、SQLAlchemy 2.x、PyMySQL、Alembic、pytest、Redis、Celery、Milvus、BGE-M3、BGE-Reranker、SSE、Next.js + TypeScript、Docker Compose。

**规格：** `docs/需求文档.md`、`docs/技术栈与架构文档.md`、`docs/接口文档.md`、`docs/superpowers/specs/2026-09-15-legal-rag-p0-corrections-design.md`、`docs/superpowers/specs/2026-09-15-legal-rag-mysql-offline-pipeline-design.md`

## 全局约束

- 首期只实现一个预设法律知识助手。
- 首期法域限定为中国大陆全国层面，专题限定为劳动法，法源为全国性法律法规和司法解释。
- 正式法律数据只能通过授权官方来源采集。
- 未经人工审核并发布的文档版本不得参与检索。
- 首期只启用 Redis 短期会话记忆，不实现 Milvus 长期记忆。
- 案例材料、其他法律角色、用户自定义角色和多云部署不进入首期验收。
- MySQL 保存用户、权限、文档、版本、审核状态、父子 chunk 和任务元数据；不保存 1024 维向量。
- Milvus 只保存已发布版本的检索索引及必要的稳定标识和过滤元数据。
- 离线 Pipeline 不连接 MySQL、Redis 或 Milvus；MySQL 导入服务不重新采集、解析、清洗或分块。
- 测试数据只能写入 pytest 临时目录、内存数据库或测试容器，不污染正式 `data`、`artifacts` 和用户数据库。
- 所有新功能遵循 TDD：先写最小失败测试，再实现最少代码，再运行定向测试和回归测试。
- 每个新增或修改的 Python 文件不超过 300 行；需要拆分职责时创建专注模块，不通过超长文件承载多个子系统。
- 新增注释、用户可见错误和文档使用简体中文；技术标识符保留原文。
- API Key、密码和完整数据库连接 URL 不得硬编码、打印、写入数据包、测试输出或记忆。
- 测试采集器时不访问真实官方站点；真实采集只在来源授权、目标目录和执行目的明确后进行。
- 当前工作区不是 Git 仓库，不执行 commit、branch、merge 或 worktree 操作；每个阶段以测试报告和范围检查作为交付记录。
- 不把金融 RAG、历史 SSE 压测或其他项目的 Milvus 数据作为本项目完成证据。

---

## 文件总览

### 第一阶段：采集可靠性

- 修改：`backend/app/crawler/rate_limiter.py`：增加可测试的等待计算或预约接口，保持来源隔离。
- 修改：`backend/app/crawler/requester.py`：处理 robots 状态、缓存键、有限重试和 50 MB 有界读取。
- 修改：`backend/app/crawler/robots_policy.py`：明确 robots `404/410` 的安全默认语义。
- 测试：`backend/tests/unit/test_rate_limiter.py`、`backend/tests/unit/test_requester.py`、`backend/tests/unit/test_robots_policy.py`。

### 第二阶段：标准数据包与离线 Pipeline

- 已有：`backend/app/pipeline/package_models.py`：不可变数据包记录模型和 JSONL 序列化。
- 创建：`backend/app/pipeline/package_validator.py`：Schema、哈希、记录数、引用和路径安全校验。
- 创建：`backend/app/pipeline/package_writer.py`：临时目录写入、manifest 生成和原子发布。
- 创建：`backend/app/pipeline/offline_pipeline.py`：串联采集、解析、分块和数据包发布。
- 修改：`backend/app/pipeline/__init__.py`：导出公共接口。
- 创建：`backend/tests/unit/test_package_validator.py`、`test_package_writer.py`、`test_offline_pipeline.py`。

### 第三阶段：MySQL 文档入库

- 创建：`backend/app/config.py`：安全读取数据库配置。
- 创建：`backend/app/database/base.py`、`session.py`、`models.py`：SQLAlchemy Base、Engine、Session 和 ORM。
- 创建：`backend/app/database/repository.py`：数据包级事务和幂等写入。
- 创建：`backend/app/importers/mysql_importer.py`：先校验数据包、再事务导入。
- 创建：`backend/app/cli/import_mysql.py`：MySQL 导入 CLI。
- 创建：`backend/tests/unit/test_config.py`、`test_database_repository.py`、`test_mysql_importer.py`、`test_cli.py`。
- 创建：`backend/tests/integration/test_mysql_import.py`：显式启用、默认跳过的真实 MySQL 测试。

### 第四阶段：审核与发布

- 创建：`backend/app/review/schemas.py`、`service.py`：审核状态和发布状态转换。
- 修改：`backend/app/database/models.py`：增加审核、发布和索引状态字段或关联表。
- 创建：`backend/app/api/review.py`：管理员审核和发布接口。
- 创建：`backend/tests/unit/test_review_service.py`、`backend/tests/test_review_api.py`。

### 第五阶段：Embedding 与 Milvus 索引

- 创建：`backend/app/indexing/contracts.py`：Embedding、VectorStore 和索引结果协议。
- 创建：`backend/app/indexing/embedding_indexer.py`：读取已发布子块、批量向量化和状态更新。
- 创建：`backend/app/indexing/milvus_store.py`：集合初始化、幂等 upsert 和查询。
- 创建：`backend/app/cli/build_index.py`：索引命令入口。
- 创建：`backend/tests/unit/test_embedding_indexer.py`、`test_milvus_store.py`、`test_build_index_cli.py`。
- 创建：`backend/tests/integration/test_milvus_index.py`：显式启用的 Milvus 测试。

### 第六阶段：最小在线检索和 SSE

- 创建：`backend/app/retrieval/contracts.py`、`query_processor.py`、`service.py`：查询、召回、过滤和上下文协议。
- 创建：`backend/app/models/reranker.py`、`backend/app/models/llm.py`：Reranker 与 LLM 适配器协议和 Mock 实现。
- 创建：`backend/app/memory/redis_store.py`：短期会话记忆读写。
- 创建：`backend/app/chat/service.py`：单一法律知识助手问答编排。
- 创建或修改：`backend/app/api/chat.py`、`backend/app/main.py`：SSE 接口和依赖注入。
- 创建：`backend/tests/unit/test_retrieval_service.py`、`test_chat_service.py`、`test_redis_store.py`、`backend/tests/test_chat_api.py`。

### 第七阶段：首期验收与部署

- 修改：`backend/pytest.ini`、`backend/pyproject.toml`：集成标记和依赖配置。
- 创建：`frontend/`：Next.js 首期聊天和管理员审核页面，仅在后端闭环通过后实施。
- 创建：`docker-compose.yml`、`backend/Dockerfile`、`frontend/Dockerfile`、`nginx/`：Ubuntu 部署编排。
- 创建：`scripts/verify_deployment.*`：健康检查、数据包校验、索引和问答验收脚本。
- 修改：`docs/测试文档.md`、`docs/部署文档.md`：记录真实验证结果，不写入凭据。

---

### 任务 1：完成采集器可靠性收尾

**文件：**
- 修改：`backend/app/crawler/rate_limiter.py`
- 修改：`backend/app/crawler/requester.py`
- 修改：`backend/app/crawler/robots_policy.py`
- 测试：`backend/tests/unit/test_rate_limiter.py`
- 测试：`backend/tests/unit/test_requester.py`
- 测试：`backend/tests/unit/test_robots_policy.py`

- [ ] **步骤 1：编写限速等待和来源隔离失败测试**

增加测试：同一来源在间隔内返回下一次允许时间或明确等待秒数，不同来源互不影响；请求器在测试时钟推进后才发出第二次页面请求。测试替身记录 URL、调用次数和 timeout。

- [ ] **步骤 2：运行限速定向测试确认旧实现失败**

运行：`python -m pytest tests/unit/test_rate_limiter.py tests/unit/test_requester.py -k "wait or retry or cache" -v`

预期：新等待接口、重试和缓存隔离用例失败，现有基础行为测试保持可收集。

- [ ] **步骤 3：实现最小限速等待协议**

在 `SourceRateLimiter` 中增加不执行真实睡眠的 `seconds_until_allowed(source_id) -> float`，保留 `can_request()` 和 `record_request()`；请求器使用注入的 `sleep` 或调用方策略，测试注入零等待实现。robots 请求不记录页面限速。

- [ ] **步骤 4：编写 robots 和有界读取失败测试**

覆盖：robots `404/410` 按安全默认策略处理；相同 `source_id` 在不同 scheme/netloc 下不复用缓存；响应声明超过 50 MB 时拒绝；无 `Content-Length` 时累计读取超过 50 MB 立即失败且不返回部分正文。

- [ ] **步骤 5：实现 robots 缓存隔离和 50 MB 流式读取**

缓存键使用 `source_id + scheme + netloc`；读取前检查 `Content-Length`，读取时按固定块累计，超过上限抛出阶段明确的采集异常；失败结果不包含正文。

- [ ] **步骤 6：编写有限重试失败测试并实现退避**

对超时、`URLError` 和 HTTP 5xx 覆盖最多 3 次尝试、指数退避和最终失败；HTTP 4xx 不重试；每次测试注入 `sleep`，断言退避次数和 URL，不访问网络。

- [ ] **步骤 7：运行采集回归和编译检查**

运行：`python -m pytest tests/unit/test_rate_limiter.py tests/unit/test_requester.py tests/unit/test_robots_policy.py tests/unit/test_runner.py -q`；运行：`python -m compileall -q app tests`。

验收：采集测试通过，首次页面采集成功，重复页面遵守间隔，robots 缓存隔离，读取有上限，重试有限。

### 任务 2：完成标准数据包校验和原子发布

**文件：**
- 创建：`backend/app/pipeline/package_validator.py`
- 创建：`backend/app/pipeline/package_writer.py`
- 修改：`backend/app/pipeline/__init__.py`
- 测试：`backend/tests/unit/test_package_validator.py`
- 测试：`backend/tests/unit/test_package_writer.py`

- [ ] **步骤 1：编写篡改、路径逃逸和引用错误测试**

构造一个最小合法 `PackageData`，覆盖：JSONL 文件被篡改、manifest 哈希错误、`../secret` 原始路径、缺少 raw 文件、重复 `chunk_id`、子块引用不存在父块、版本或文档关联错误。

- [ ] **步骤 2：运行校验测试确认模块缺失或行为失败**

运行：`python -m pytest tests/unit/test_package_validator.py -v`；预期：校验器接口尚未完整实现时失败，失败信息不包含正文。

- [ ] **步骤 3：实现 `PackageManifest` 和 `validate_package()`**

校验必需文件、`schema_version == "1.0"`、受管文件 SHA-256、记录数、文档/版本/chunk/crawl 关系、chunk 唯一性、父子引用、相对路径和普通文件约束，返回 `ValidatedPackage`。

- [ ] **步骤 4：编写原子发布失败测试**

覆盖：成功发布生成 `manifest.json` 和四个 JSONL；临时目录不残留；最终目录已存在时拒绝覆盖；中途异常不留下最终目录；已有最终目录不被破坏；manifest 不包含绝对路径或敏感字段。

- [ ] **步骤 5：实现 `publish_package()`**

使用同级临时目录写入 raw 文件和 JSONL，计算哈希并写 manifest，调用 `validate_package()`，通过后使用 `Path.replace()` 发布；异常时只清理本次临时目录。

- [ ] **步骤 6：运行数据包阶段回归**

运行：`python -m pytest tests/unit/test_package_models.py tests/unit/test_package_validator.py tests/unit/test_package_writer.py -q`。

验收：数据包可往返读取、可验证、不可被半成品发布、路径和哈希错误均被拒绝。

### 任务 3：打通离线采集到数据包 Pipeline

**文件：**
- 创建：`backend/app/pipeline/offline_pipeline.py`
- 修改：`backend/app/pipeline/__init__.py`
- 测试：`backend/tests/unit/test_offline_pipeline.py`

- [ ] **步骤 1：编写成功链路失败测试**

注入 Fake crawler、临时 HTML、`parse_document`、`chunk_document`、固定时钟和固定 package ID；断言调用顺序为 crawler → parser → chunker → writer，输出包含文档、版本、crawl 和父子 chunk。

- [ ] **步骤 2：编写阶段失败和隔离测试**

覆盖采集失败、缺少原始文件、解析失败、空正文、空 chunk、发布失败；断言最终目录不存在、错误阶段属于 `crawl/parse/chunk/validate/publish`，错误不包含正文，Pipeline 不导入数据库模块。

- [ ] **步骤 3：实现 `OfflinePipeline` 和结果类型**

实现 `run(crawl_id, source_id, source_url, document_title) -> OfflinePipelineResult`；使用 `stable_document_id()` 和 `stable_version_id()`；从现有 `CrawlResult` 映射 `PackageData`；不连接数据库、Redis 或 Milvus。

- [ ] **步骤 4：运行离线链路和既有处理回归**

运行：`python -m pytest tests/unit/test_offline_pipeline.py tests/unit/test_runner.py tests/unit/test_parser.py tests/unit/test_cleaner.py tests/unit/test_chunker.py -q`。

验收：使用测试替身可以生成通过校验的数据包，失败无半成品，既有解析和分块行为不回归。

### 任务 4：实现 MySQL 配置、ORM、事务和幂等导入

**文件：**
- 创建或修改：`backend/pyproject.toml`、`backend/.env.example`
- 创建：`backend/app/config.py`
- 创建：`backend/app/database/base.py`、`session.py`、`models.py`、`repository.py`
- 创建：`backend/app/importers/mysql_importer.py`
- 创建：`backend/app/cli/import_mysql.py`
- 测试：`backend/tests/unit/test_config.py`、`test_database_repository.py`、`test_mysql_importer.py`、`test_cli.py`
- 测试：`backend/tests/integration/test_mysql_import.py`

- [ ] **步骤 1：编写 SQLite 内存 Schema 和配置失败测试**

断言 `DatabaseSettings` 拒绝空 URL及不支持的 scheme；`safe_summary()` 隐藏用户名和密码；SQLite 内存 Engine 可创建 `documents`、`document_versions`、`crawl_records`、`document_chunks`、`import_records`。

- [ ] **步骤 2：实现 SQLAlchemy 配置和 ORM**

使用 `Mapped[...]`、`mapped_column()`、外键和表级唯一约束；文档、版本、采集记录、父子 chunk 和导入审计记录均保存业务字段，不添加向量字段；SQLite 测试启用 `StaticPool`。

- [ ] **步骤 3：编写新建、重复、更新和回滚失败测试**

同一文档首次导入返回 `new`；同内容新 package 返回 `unchanged`；相同 package 返回 `already_imported`；新 content hash 返回 `updated`；冲突 chunk 或中途异常后不产生部分版本、chunk 或成功导入记录。

- [ ] **步骤 4：实现 `PackageRepository.import_package()`**

按 source URL 获取或创建文档；按 document 和 content hash 判断版本；按父块后子块顺序写入；新版本状态设为 `awaiting_review`；成功后写审计记录并更新当前版本；由数据库唯一约束保障幂等。

- [ ] **步骤 5：编写校验先于事务测试**

向 `MysqlPackageImporter` 注入计数 Session factory，传入非法数据包，断言返回阶段 `validate` 且 Session 创建次数为 0；持久化异常包装为 `persist`，不泄漏 URL、密码或正文。

- [ ] **步骤 6：实现 `MysqlPackageImporter` 和 CLI**

`import_path()` 先调用 `validate_package()`，成功后才创建 Session，并在事务上下文中调用 Repository；CLI 只接受显式 package 路径，成功返回 0，参数或业务失败返回 1。

- [ ] **步骤 7：运行 MySQL 阶段测试**

运行：`python -m pytest tests/unit/test_config.py tests/unit/test_database_repository.py tests/unit/test_mysql_importer.py tests/unit/test_cli.py -q`；运行默认跳过集成测试：`python -m pytest tests/integration/test_mysql_import.py -q`。

验收：SQLite 内存测试通过，非法包不打开数据库，重复导入幂等，内容变化生成新版本，失败事务完整回滚。

### 任务 5：补齐人工审核和发布状态边界

**文件：**
- 创建：`backend/app/review/schemas.py`、`backend/app/review/service.py`
- 修改：`backend/app/database/models.py`
- 创建：`backend/app/api/review.py`
- 修改：`backend/app/main.py`
- 测试：`backend/tests/unit/test_review_service.py`、`backend/tests/test_review_api.py`

- [ ] **步骤 1：编写状态转换失败测试**

覆盖 `awaiting_review → approved → published` 成功路径；普通用户不能审核；重复发布、从 `rejected` 直接发布、已发布版本修改原文均失败；发布旧版本时新版本标记 `superseded`。

- [ ] **步骤 2：实现状态模型和审核服务**

固定状态集合 `awaiting_review`、`approved`、`rejected`、`published`、`superseded`；审核操作写入审核人、时间和原因；发布操作只允许已批准版本，并保证同一文档只有一个当前发布版本。

- [ ] **步骤 3：实现管理员审核接口**

增加查询待审核版本、批准、驳回和发布接口；接口依赖现有认证权限；响应不返回密钥、完整内部路径或正文全文；非法状态返回统一业务错误。

- [ ] **步骤 4：运行审核 API 回归**

运行：`python -m pytest tests/unit/test_review_service.py tests/test_review_api.py tests/test_auth.py tests/test_permissions.py -q`。

验收：未发布版本无法被索引服务选取，状态转换和权限边界由测试保护。

### 任务 6：实现已发布子块的 Embedding 与 Milvus 索引

**文件：**
- 创建：`backend/app/indexing/contracts.py`、`embedding_indexer.py`、`milvus_store.py`
- 创建：`backend/app/cli/build_index.py`
- 测试：`backend/tests/unit/test_embedding_indexer.py`、`test_milvus_store.py`、`test_build_index_cli.py`
- 测试：`backend/tests/integration/test_milvus_index.py`

- [ ] **步骤 1：编写未发布过滤和维度校验失败测试**

构造包含 `awaiting_review`、`approved`、`published` 三种状态的子块；断言只读取 `published`；Embedding 返回非 1024 维时任务失败且不写入 Milvus；空批次不创建集合写入。

- [ ] **步骤 2：定义 Embedding、VectorStore 和索引结果协议**

定义批量 `embed(texts: Sequence[str]) -> Sequence[Sequence[float]]`、`upsert(records)`、`search(vector, filters, limit)` 和 `IndexResult`；协议不绑定具体供应商，复用现有 Embedding 客户端适配器。

- [ ] **步骤 3：实现批量索引服务**

从 MySQL 查询已发布子块，按固定批大小调用 Embedding，检查数量和 1024 维，写入 `chunk_id/document_id/version_id` 与法律过滤元数据；成功后更新索引状态，重复执行使用稳定 chunk ID 幂等处理。

- [ ] **步骤 4：实现 Milvus Store**

初始化知识库集合和密集向量字段；将集合操作封装在 Store 内；写入前验证稳定标识和版本；查询只返回召回所需的 chunk ID、相似度和过滤字段，不把 Milvus 作为业务主库。

- [ ] **步骤 5：实现索引 CLI 和受控集成入口**

CLI 支持显式 knowledge base 或版本范围，失败输出阶段和脱敏摘要；真实 Milvus 测试仅在 `RUN_MILVUS_INTEGRATION=1` 且 URI 明确时运行，否则安全跳过。

- [ ] **步骤 6：运行索引阶段测试**

运行：`python -m pytest tests/unit/test_embedding_indexer.py tests/unit/test_milvus_store.py tests/unit/test_build_index_cli.py -q`；默认运行：`python -m pytest tests/integration/test_milvus_index.py -q`。

验收：只有发布版本进入 Milvus，Embedding 维度严格为 1024，重复索引不产生重复业务块。

### 任务 7：实现最小在线检索、Redis 短期记忆和 SSE

**文件：**
- 创建：`backend/app/retrieval/contracts.py`、`query_processor.py`、`service.py`
- 创建：`backend/app/models/reranker.py`、`backend/app/models/llm.py`
- 创建：`backend/app/memory/redis_store.py`
- 创建：`backend/app/chat/service.py`
- 创建或修改：`backend/app/api/chat.py`、`backend/app/main.py`
- 测试：`backend/tests/unit/test_retrieval_service.py`、`test_chat_service.py`、`test_redis_store.py`
- 测试：`backend/tests/test_chat_api.py`

- [ ] **步骤 1：编写检索过滤和无证据失败测试**

注入 Milvus 召回结果和 MySQL 元数据，包含未发布版本、失效版本、无权限知识库和有效版本；断言在线服务只保留有效发布内容，召回不到证据时返回明确的“无法基于现有法源确认”结果，不调用 LLM 生成确定性结论。

- [ ] **步骤 2：定义查询、召回、精排、上下文和引用类型**

定义 `QueryRequest`、`RetrievedChunk`、`Citation`、`AnswerContext` 和 `StreamEvent`；子块用于召回，根据 `parent_id` 补齐父块；上下文包含法规名称、条文号、版本状态、生效信息和来源 URL。

- [ ] **步骤 3：实现最小检索服务**

执行鉴权后的权限过滤、问题标准化、Embedding 查询、Milvus 召回、MySQL 父块与法律元数据补全、去重和 Reranker 排序；限制最终上下文数量，禁止未发布版本混入。

- [ ] **步骤 4：实现 Redis 短期记忆**

以 `user_id + session_id` 隔离会话，保存最近有限条消息并设置 TTL；Redis 不可用时返回明确的可重试错误或按明确定义的无记忆模式处理，不写入 Milvus 长期记忆。

- [ ] **步骤 5：实现 Mock LLM、引用校验和 SSE 编排**

`ChatService.stream_answer()` 先完成检索和上下文校验，再调用 LLM；输出 `status`、`citation`、`token`、`result` 或 `error` 事件；回答必须包含风险提示，引用必须来自实际上下文。

- [ ] **步骤 6：实现 `/api/v1/chat` SSE 接口**

校验登录用户、session 和预设法律知识助手；将服务异常映射为统一错误事件；不在日志或响应中输出凭据、内部数据库 URL 或完整异常堆栈。

- [ ] **步骤 7：运行在线闭环测试**

运行：`python -m pytest tests/unit/test_retrieval_service.py tests/unit/test_chat_service.py tests/unit/test_redis_store.py tests/test_chat_api.py -q`；再运行全量：`python -m pytest -q`。

验收：单一法律助手可以基于已发布劳动法源返回 SSE；权限、版本、生效状态、引用和短期记忆行为可验证。

### 任务 8：完成首期部署和业务验收

**文件：**
- 修改：`backend/pyproject.toml`、`backend/pytest.ini`
- 创建：`frontend/` 首期聊天和审核页面
- 创建：`docker-compose.yml`、`backend/Dockerfile`、`frontend/Dockerfile`、`nginx/`
- 创建：`scripts/verify_deployment.ps1`、`scripts/verify_deployment.sh`
- 修改：`docs/测试文档.md`、`docs/部署文档.md`

- [ ] **步骤 1：编写部署配置和健康检查失败测试**

为 API、Worker、MySQL、Redis、Milvus、前端和 Nginx 定义健康检查；测试配置中禁止把真实凭据写入 compose 文件；未配置模型服务时使用 Mock 而不是访问未知地址。

- [ ] **步骤 2：实现后端、Worker、前端和反向代理编排**

服务通过环境变量注入配置，数据目录使用显式 volume；API、Worker 和前端分别构建；Nginx 只代理已定义的 API 和 SSE 路径；不自动清空数据库或 Milvus 集合。

- [ ] **步骤 3：实现首期前端最小页面**

只实现登录、法律知识助手聊天、引用展示和管理员审核列表/操作；不加入其他角色、长期记忆或用户自定义知识库入口。

- [ ] **步骤 4：运行代码、配置和安全检查**

运行：`python -m pytest -q`；运行：`python -m compileall -q app tests`；检查新增 Python 文件行数；扫描源码、compose、文档和测试输出中的密钥、密码和完整生产 URL。

- [ ] **步骤 5：执行受控真实离线链路**

从当前白名单配置中选择已确认授权的官方劳动法页面，使用用户指定目录运行 `build_ingestion_package`；只报告来源域名、哈希前缀、正文字符数、父子块数量和校验结果。

- [ ] **步骤 6：执行 MySQL、审核、索引和问答验收**

在用户确认目标库和服务可连接后依次执行：数据包导入 → 管理员审核 → 发布 → Embedding/Milvus 索引 → 单问题 SSE；重复导入和重复索引均验证幂等；不删除或清空用户数据。

- [ ] **步骤 7：按真实证据更新测试和部署文档**

记录测试通过数、集成测试跳过原因、真实来源、数据包统计、MySQL 导入状态、Milvus 索引状态和 SSE 结果；任何未实际执行的步骤明确标记为未验证。

---

## 阶段停止条件

- 阶段 1 未通过前，不进入数据包 Pipeline。
- 阶段 2 未生成并验证标准数据包前，不进入 MySQL 导入。
- 阶段 3 未实现事务幂等前，不进行真实数据库写入。
- 阶段 4 未完成审核发布状态前，不执行正式 Embedding 或 Milvus 写入。
- 阶段 5 未验证只索引已发布版本前，不进入在线问答。
- 阶段 6 未通过 Mock 闭环测试前，不进行真实模型服务压测。
- 阶段 7 未完成全量测试、编译检查、敏感信息检查和受控业务验收前，不声称首期完成。

## 计划自检结果

- 需求覆盖：认证与权限由现有模块和任务 5、7、8 接续；单一法律助手由任务 7；官方采集由任务 1、3；审核发布由任务 5；文档解析、清洗、分块由现有模块和任务 3；Embedding/Milvus 由任务 6；Redis 短期记忆由任务 7；SSE 由任务 7；Ubuntu Docker Compose 由任务 8。
- 架构边界：MySQL 是业务与文档元数据主库，Milvus 是已发布内容的检索索引，Redis 只保存短期会话，不实现长期记忆。
- 安全边界：采集遵守白名单、robots 和限速；数据包校验路径和哈希；数据库导入先校验再事务；在线检索过滤发布状态和权限；凭据不进入代码、输出或数据包。
- 范围检查：没有加入多角色、案例材料、用户自定义知识库、Milvus 长期记忆、金融 RAG 或多云部署。
- 验证检查：每个任务均有定向测试和阶段验收；最终使用全量 pytest、compileall、敏感信息扫描及受控真实链路验证。
