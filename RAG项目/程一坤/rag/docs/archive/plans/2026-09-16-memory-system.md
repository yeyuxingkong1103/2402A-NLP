# 认证与记忆系统实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `subagent-driven-development`（推荐）或 `executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现登录用户隔离下的 Redis 短期记忆和 Milvus 长期记忆。

**架构：** MySQL 管理用户身份和 refresh token 元数据；Redis 保存带 TTL 的会话消息、摘要和临时状态；Milvus 独立的 `user_memories` collection 保存长期记忆向量及用户过滤字段。所有记忆操作从认证上下文获得 `user_id`。

**技术栈：** FastAPI、PyJWT、密码强哈希、Redis 5.x、pymilvus、pytest、Milvus Lite 或显式集成环境。

**规格：** `docs/superpowers/specs/2026-09-15-legal-rag-mysql-offline-pipeline-design.md`

## 全局约束

- 未认证请求不能访问记忆接口。
- 服务端不信任请求体中的 `user_id`。
- Redis key 必须包含 `user_id` 和 `session_id`，并设置 TTL。
- 长期记忆写入前执行摘要、去重、敏感信息过滤和价值判断。
- `user_memories` 与 `legal_documents` 必须是独立 collection。
- 删除或标记删除后的记忆不得被召回。
- 不保存密码、API Key、身份证号等不必要敏感信息。

---

### 任务 1：实现认证核心服务

**文件：**
- 创建：`backend/app/auth/passwords.py`
- 创建：`backend/app/auth/tokens.py`
- 创建：`backend/app/auth/service.py`
- 测试：`backend/tests/test_auth_service.py`

- [ ] **步骤 1：编写失败测试**

覆盖密码哈希不可逆、正确密码验证、错误密码拒绝、access token 包含用户标识并过期、refresh token 可撤销，以及停用用户不能登录。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_auth_service.py -q`
预期：因认证实现不完整而失败。

- [ ] **步骤 3：实现最少代码**

使用强哈希保存密码摘要；使用短时 JWT access token；refresh token 使用随机不可预测值并只保存其哈希和过期/撤销状态；认证服务从 Repository 获取用户，不在 token 中放入敏感信息。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_auth_service.py -q`
预期：全部通过。

---

### 任务 2：实现登录 API 与认证依赖

**文件：**
- 修改：`backend/app/auth/router.py`
- 修改：`backend/app/api/dependencies.py`
- 修改：`backend/app/main.py`
- 测试：`backend/tests/test_auth_api.py`

- [ ] **步骤 1：编写失败测试**

使用 `httpx` 覆盖注册、登录、当前用户、退出登录、无 token 拒绝和停用用户拒绝；断言响应不泄露密码、完整 token 存储值或内部异常。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_auth_api.py -q`
预期：因路由和认证依赖尚未接通而失败。

- [ ] **步骤 3：实现最少 API 代码**

增加注册/登录/刷新/退出和当前用户接口；认证依赖解析 access token 并注入 `AuthenticatedUser(user_id, roles)`；所有错误使用统一非敏感响应。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_auth_api.py -q`
预期：全部通过。

---

### 任务 3：实现 Redis 短期记忆 Repository

**文件：**
- 创建：`backend/app/memory/short_term.py`
- 创建：`backend/app/memory/keys.py`
- 测试：`backend/tests/test_short_term_memory.py`

- [ ] **步骤 1：编写失败测试**

使用 fake Redis 覆盖最近消息追加、会话摘要读写、TTL 设置、用户与会话隔离、删除会话记忆；构造另一个用户访问同一 session 字符串，必须读不到原数据。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_short_term_memory.py -q`
预期：因短期记忆 Repository 不存在而失败。

- [ ] **步骤 3：实现最少代码**

定义 `ShortTermMemoryStore`，所有 key 使用 `short_memory:{user_id}:{session_id}:...`；写入消息和摘要时设置 TTL；读取和删除方法只接受认证上下文提供的 `user_id`。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_short_term_memory.py -q`
预期：全部通过。

---

### 任务 4：实现 Milvus 长期记忆 Repository

**文件：**
- 创建：`backend/app/memory/long_term.py`
- 创建：`backend/app/memory/milvus_schema.py`
- 测试：`backend/tests/test_long_term_memory.py`

- [ ] **步骤 1：编写失败测试**

使用 fake Milvus client 覆盖写入前敏感信息过滤、`user_id` 强制写入、按用户过滤召回、删除后不可召回、与 `legal_documents` collection 名称隔离。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_long_term_memory.py -q`
预期：因长期记忆 Repository 和 schema 不存在而失败。

- [ ] **步骤 3：实现最少代码**

定义 `user_memories` schema：`memory_id`、`user_id`、可选会话/角色字段、类型、摘要/正文、时间戳、`deleted` 和 embedding；写入前调用脱敏与去重策略；查询表达式固定包含当前 `user_id` 和 `deleted == false`；删除使用标记删除或明确删除接口，并让召回排除已删除记录。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_long_term_memory.py -q`
预期：全部通过；没有可用 Milvus 集成环境时不伪造真实连接成功。
