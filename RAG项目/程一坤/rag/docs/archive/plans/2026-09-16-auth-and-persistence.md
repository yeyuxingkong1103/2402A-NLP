# 认证与基础持久化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `subagent-driven-development`（推荐）或 `executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 建立可审计的法律数据包校验、MySQL 权威存储、用户登录和认证基础能力。

**架构：** 离线数据包先经过 manifest、哈希、引用和路径校验，再通过 SQLAlchemy 事务导入 MySQL。MySQL 保存法律文档、版本、chunk、采集记录、导入审计、用户和 refresh token 元数据；认证上下文统一提供 `user_id`。

**技术栈：** Python、SQLAlchemy 2.x typed declarative、PyMySQL、SQLite 内存测试、FastAPI、PyJWT、密码强哈希库。

**规格：** `docs/superpowers/specs/2026-09-15-legal-rag-mysql-offline-pipeline-design.md`

## 全局约束

- 遵循 TDD：先写失败测试，再实现最少代码。
- MySQL 保存法律文档、版本、chunk 和采集记录。
- 入库顺序为：校验数据包 → 检查幂等 → 开启事务 → 导入 → 提交。
- 不修改或删除原始 HTML、旧哈希文件和已发布数据包。
- 错误摘要必须脱敏、截断，不保存完整正文、密码、API Key 或完整连接 URL。
- 当前目录不是 Git 仓库，不执行或声称执行 commit、分支或 Git 收尾。

---

### 任务 1：恢复数据包校验与原子发布

**文件：**
- 创建：`backend/app/pipeline/package_validator.py`
- 创建：`backend/app/pipeline/package_writer.py`
- 修改：无
- 测试：`backend/tests/test_package_module.py`

- [ ] **步骤 1：运行现有失败测试**

运行：`python -m pytest tests/test_package_module.py -q`
预期：因缺少 `app.pipeline.package_validator` 或 `app.pipeline.package_writer` 失败。

- [ ] **步骤 2：实现 `package_validator.py`**

实现 `MANIFEST_FILE`、`SCHEMA_VERSION`、`PackageValidationError`、`validate_package()`，完成必需文件检查、JSONL 读取、manifest 记录数校验、SHA-256 校验、文档/版本/chunk/crawl 引用校验、父块存在性校验和 raw 路径不能逃逸根目录校验；异常文本只包含文件名、字段名和阶段。

- [ ] **步骤 3：实现 `package_writer.py`**

实现 `PIPELINE_VERSION`、`publish_package()`：先创建同级临时目录，写入 JSONL 和 raw 文件，生成 manifest，调用 `validate_package()`，再使用原子重命名发布；目标目录已存在或任一步失败时保留既有目录且清理本次临时目录。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_package_module.py -q`
预期：该测试文件中的用例全部通过。

---

### 任务 2：建立 SQLAlchemy MySQL 数据模型

**文件：**
- 创建：`backend/app/db/base.py`
- 创建：`backend/app/db/sql_models.py`
- 创建：`backend/app/db/engine.py`
- 修改：`backend/requirements.txt`
- 测试：`backend/tests/test_sql_models.py`

- [ ] **步骤 1：编写失败测试**

测试使用 `sqlite+pysqlite:///:memory:` 创建 engine，验证 metadata 能创建 `documents`、`document_versions`、`document_chunks`、`crawl_records`、`import_records`、`users` 和 `user_sessions`；验证 `document_key`、`version_key`、`chunk_id`、`package_id` 和用户名具备唯一约束。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_sql_models.py -q`
预期：因 SQLAlchemy 模型或依赖不存在而失败。

- [ ] **步骤 3：实现最少模型代码**

使用 `DeclarativeBase`、`Mapped` 和 `mapped_column` 定义 typed declarative 模型。为 `document_chunks` 增加 MySQL `FULLTEXT` 索引所需的 `content`/`retrieval_text` 索引声明；SQLite 测试必须能够创建表。连接模块从显式传入的 URL 创建 engine，不打印 URL。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_sql_models.py -q`
预期：表、外键和唯一约束测试通过。

---

### 任务 3：实现事务型数据包入库 Repository

**文件：**
- 创建：`backend/app/db/repositories.py`
- 创建：`backend/app/db/import_service.py`
- 测试：`backend/tests/test_mysql_import_service.py`

- [ ] **步骤 1：编写失败测试**

覆盖首次导入返回 `new`、相同 `source_url + content_hash` 返回 `unchanged`、内容变化返回 `updated`、重复 `package_id` 返回 `already_imported`、父子 chunk 正确保存，以及中途异常后文档/版本/chunk/import 记录全部回滚。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_mysql_import_service.py -q`
预期：因 Repository 和导入服务不存在而失败。

- [ ] **步骤 3：实现导入事务**

实现 `import_package(session, validated_package)`：先检查成功的 `package_id`，按 `source_url` 查找或创建文档，按 `content_hash` 判定状态；新版本写入版本、chunk、crawl，设置 `awaiting_embedding`，成功后写 `imported`，异常时回滚并写入脱敏 `failed` 记录。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_mysql_import_service.py -q`
预期：所有幂等、状态和回滚测试通过。

---

### 任务 4：实现 MySQL 入库 CLI

**文件：**
- 创建：`backend/app/cli/__init__.py`
- 创建：`backend/app/cli/import_mysql.py`
- 修改：`backend/app/core/config.py`
- 测试：`backend/tests/test_import_mysql_cli.py`

- [ ] **步骤 1：编写失败测试**

测试 CLI 能拒绝不存在目录、拒绝校验失败数据包、使用显式数据库 URL 导入测试包，并输出不含密码和正文的状态摘要。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_import_mysql_cli.py -q`
预期：因 CLI 入口不存在而失败。

- [ ] **步骤 3：实现 CLI**

实现 `python -m app.cli.import_mysql --package <package_dir> --database-url <url>`；加载并校验 package，再创建 session 执行导入；生产默认从 `DATABASE_URL` 读取，测试显式传入 SQLite URL；退出码区分成功、已导入和失败。

- [ ] **步骤 4：运行定向验证**

运行：`python -m pytest tests/test_import_mysql_cli.py -q` 和 `python -m compileall -q app tests`
预期：测试通过，编译命令退出码为 0。
