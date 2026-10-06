# LAW-RAG 接口文档

本文档依据当前工作区中的实际代码整理，适用于当前 FastAPI 服务和前端客户端。

## 1. 服务信息

- 服务框架：FastAPI
- 默认地址：`http://127.0.0.1:7294`
- OpenAPI：`/openapi.json`
- Swagger UI：`/docs`
- ReDoc：`/redoc`
- 健康检查：`GET /health`
- 前端入口：`GET /`

当 `EXPOSE_DOCS=false` 时，OpenAPI、Swagger UI 和 ReDoc 会关闭。

## 2. 通用约定

### 2.1 请求头和认证

登录成功后，服务端默认写入 `HttpOnly` Cookie：`lawrag_auth`。同时支持以下方式传递认证令牌：

```http
Authorization: Bearer <token>
```

Cookie 名称、令牌有效期和 `Secure` 属性均可通过服务配置调整。需要登录的接口在缺少有效会话时返回 HTTP `401`。

### 2.2 成功响应

常规接口使用以下结构：

```json
{
  "success": true,
  "data": {}
}
```

部分历史兼容接口还会附带 `message` 或其他业务字段，客户端应优先读取 `success` 和 `data`。

### 2.3 错误响应

HTTP 异常统一转换为：

```json
{
  "success": false,
  "message": "错误说明",
  "code": "HTTP_400",
  "data": null
}
```

请求体校验失败时状态码为 `422`，错误码为 `VALIDATION_ERROR`。常见状态码如下：

| 状态码 | 含义 |
|---|---|
| `400` | 参数或业务内容不合法 |
| `401` | 未登录、令牌失效或账号不存在 |
| `403` | 当前账号无权执行操作 |
| `404` | 资源不存在 |
| `409` | 资源冲突，例如用户名或邮箱已存在 |
| `422` | 请求体字段校验失败 |
| `429` | 登录或问答请求过于频繁 |
| `500` | 服务内部错误 |
| `503` | 依赖服务未初始化或暂不可用 |

## 3. 接口总览

| 模块 | 方法 | 路径 | 认证 |
|---|---|---|---|
| 系统 | GET | `/health` | 否 |
| 系统 | GET | `/api/v1/system/logs/integrity` | 审计用户或审计令牌 |
| 认证 | POST | `/api/v1/auth/register` | 否 |
| 认证 | POST | `/api/v1/auth/login` | 否 |
| 认证 | POST | `/api/v1/auth/logout` | 可选 |
| 认证 | GET | `/api/v1/auth/me` | 是 |
| 用户 | GET/PUT | `/api/v1/user/settings` | 是 |
| 用户 | GET/PUT | `/api/v1/user/profile` | 是 |
| 用户 | GET | `/api/v1/user/activity` | 是 |
| 用户 | GET/POST/DELETE | `/api/v1/user/deletion-request` | 是 |
| 工作区 | POST | `/api/v1/legal/files` | 是 |
| 工作区 | POST | `/api/v1/legal/files/batch` | 是 |
| 工作区 | GET | `/api/v1/legal/files` | 是 |
| 工作区 | DELETE | `/api/v1/legal/files/{document_id}` | 是 |
| 记忆 | GET/PUT | `/api/v1/memory/settings` | 是 |
| 记忆 | GET | `/api/v1/memory/conflicts/pending` | 是 |
| 记忆 | POST | `/api/v1/memory/conflicts/{conflict_id}/resolve` | 是 |
| 记忆 | GET/DELETE | `/api/v1/memory` | 是 |
| 记忆 | GET/DELETE | `/api/v1/memory/{memory_id}` | 是 |
| 记忆 | GET | `/api/v1/memory/{memory_id}/explain` | 是 |
| RAG | POST | `/api/v1/legal/ask` | 否，可选登录 |
| RAG | POST | `/api/v1/legal/ask/stream` | 否，可选登录 |
| RAG | POST | `/api/v1/legal/solution` | 否，可选登录 |
| RAG | POST | `/api/v1/legal/solution/pdf` | 是 |
| RAG | GET | `/api/v1/legal/sessions` | 是 |
| RAG | DELETE | `/api/v1/legal/sessions/{session_id}` | 是 |

## 4. 系统接口

### 4.1 获取系统健康状态

```http
GET /health
```

返回 MySQL、Redis、Milvus、模型、工作区和 RAG 相关状态，以及已建立的公共集合和索引记录数。典型字段包括：

```json
{
  "success": true,
  "data": {
    "status": "ok",
    "rag": "ok",
    "mysql": {"connected": true},
    "redis": {"connected": true},
    "milvus": {"connected": true, "public_collections": [], "indexed_records": 0},
    "model": {"connected": true},
    "workspace": {"enabled": true},
    "llm_provider": "deepseek",
    "llm_model": "deepseek-chat",
    "embedding_model": "Pro/BAAI/bge-m3",
    "reranker_model": "Pro/BAAI/bge-reranker-v2-m3"
  }
}
```

### 4.2 校验日志完整性

```http
GET /api/v1/system/logs/integrity
```

该接口用于审计 JSON 日志的哈希链完整性。访问需要配置的审计令牌，或当前用户属于 `LOG_AUDIT_USERS`。返回日志文件路径、记录数量、首尾哈希和校验结果。

## 5. 认证和用户接口

### 5.1 注册

```http
POST /api/v1/auth/register
Content-Type: application/json
```

```json
{
  "username": "demo_user",
  "email": "demo@example.com",
  "password": "abc12345"
}
```

校验规则：用户名为 3 至 32 位字母、数字或下划线；密码为 6 至 20 位，必须同时包含字母和数字；邮箱需要符合基本邮箱格式。成功后创建账号、会话并写入认证 Cookie。

### 5.2 登录

```http
POST /api/v1/auth/login
```

请求体与注册接口相同，但只使用 `username` 和 `password`。同一 IP 和用户名组合默认累计 5 次失败后，在 15 分钟内暂时拒绝继续登录。

### 5.3 登出和当前用户

```http
POST /api/v1/auth/logout
GET  /api/v1/auth/me
```

`logout` 删除服务端会话并清理 Cookie。`me` 返回公开用户信息、用户设置、用户画像和账号注销状态，不返回密码哈希等敏感字段。

### 5.4 用户设置、画像和活动

```http
GET /api/v1/user/settings
PUT /api/v1/user/settings
GET /api/v1/user/profile
PUT /api/v1/user/profile
GET /api/v1/user/activity?limit=20
```

设置和画像更新接口接收 JSON 对象。默认设置包括回答详略程度、是否默认联网、是否显示检索过程、是否自动展开专业内容、长期记忆开关和主题配置。`limit` 用于限制活动记录数量。

### 5.5 账号注销申请

```http
GET    /api/v1/user/deletion-request
POST   /api/v1/user/deletion-request
DELETE /api/v1/user/deletion-request
```

三个接口分别用于查询注销状态、发起注销申请和取消注销申请。注销状态会在认证校验阶段参与账号访问控制。

## 6. 用户工作区接口

### 6.1 单文件上传

```http
POST /api/v1/legal/files
Content-Type: multipart/form-data
```

表单字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | 文件 | 是 | 当前允许的文本、PDF、DOCX、XLSX、JSON 和图片等格式 |
| `session_id` | 字符串 | 否 | 将材料绑定到指定会话 |

服务会先保存原文件，再进行解析、切分、Embedding 和 Milvus 私有集合入库。文本文件通常同步返回；音频和视频等较慢媒体在开启异步处理时返回 `processing` 状态，由 Celery 或本地线程继续处理。

### 6.2 批量上传

```http
POST /api/v1/legal/files/batch
Content-Type: multipart/form-data
```

表单字段为多个 `files`、可选 `session_id` 和可选 `upload_password`。返回每个文件的处理结果，以及 `uploaded_count`、`failed_count`、`processing_count`、`stored_count` 和 `ready_count`。

### 6.3 文件列表

```http
GET /api/v1/legal/files?session_id=<session_id>
```

不传 `session_id` 时返回当前用户的全部文件；传入后只返回该会话关联材料。每条记录包含文件标识、文件名、大小、状态、抽取方式、OCR/多模态状态和向量分块数量等信息。

### 6.4 删除文件

```http
DELETE /api/v1/legal/files/{document_id}?session_id=<session_id>
```

删除会清理用户文件记录、原始文件、处理快照和私有向量。若传入 `session_id` 与文件所属会话不一致，接口返回 `session_mismatch`，不会删除文件。

## 7. 记忆接口

长期记忆接口全部要求登录。长期记忆未启用时，删除接口返回 `locked=true`，不会执行删除。

### 7.1 设置长期记忆

```http
GET /api/v1/memory/settings
PUT /api/v1/memory/settings
```

更新请求体：

```json
{"enable_long_memory": true}
```

### 7.2 查询、解释和删除记忆

```http
GET    /api/v1/memory?limit=100&offset=0
GET    /api/v1/memory/{memory_id}/explain
DELETE /api/v1/memory/{memory_id}
DELETE /api/v1/memory
```

长期记忆按用户隔离。解释接口返回记忆来源和相关会话信息；删除接口支持删除单条或删除当前用户全部长期记忆。

### 7.3 处理记忆冲突

```http
GET  /api/v1/memory/conflicts/pending?limit=50
POST /api/v1/memory/conflicts/{conflict_id}/resolve
```

请求体：

```json
{"action": "accept_new"}
```

`action` 支持 `accept_new`、`keep_old` 和 `discard`。

## 8. RAG 法律问答接口

### 8.1 普通问答

```http
POST /api/v1/legal/ask
Content-Type: application/json
```

请求体：

```json
{
  "question": "劳动合同到期后继续工作，是否视为续签？",
  "session_id": "session-001",
  "include_web": false,
  "thinking_enabled": false
}
```

`question` 和兼容字段 `query` 二选一，服务端会优先使用 `question`。问题不能为空。匿名请求不会使用会话记忆；登录用户可通过 `session_id` 关联对话历史和工作区材料。

处理流程包括查询理解、检索规划、公共法库检索、用户私有材料检索、可选联网检索、RRF 融合、重排、证据构建、上下文压缩和答案生成。

### 8.2 流式问答

```http
POST /api/v1/legal/ask/stream
Accept: text/event-stream
```

响应为 Server-Sent Events，事件格式如下：

```text
event: thinking
data: {"status":"running","summary":"...","step":"..."}

event: status
data: {"status":"..."}

event: step
data: {"step":"..."}

event: chunk
data: {"delta":"文本片段"}

event: complete
data: {"answer":"...","sources":[],"meta":{}}

event: error
data: {"message":"错误说明"}
```

实际事件顺序由当前检索和生成过程决定。`chunk` 用于逐段输出最终回答；`complete` 表示本轮结束并携带完整结果。

匿名用户每 60 秒最多 5 次问答请求，登录用户每 60 秒最多 20 次。限制器按用户标识或客户端 IP 统计。

### 8.3 生成法律解决方案

```http
POST /api/v1/legal/solution
Content-Type: application/json
```

请求体：

```json
{
  "question": "案件事实和用户问题",
  "answer": {"answer": "问答结果"},
  "sources": [{"title": "法条标题", "content": "证据内容"}],
  "session_id": "session-001",
  "force_refresh": false
}
```

`answer` 可为字符串或对象；`sources` 最多 20 条。服务端根据用户、会话、问题、回答和来源生成 SHA-256 缓存键，默认使用 Redis 缓存解决方案；`force_refresh=true` 时跳过旧缓存并覆盖缓存。

### 8.4 导出解决方案 PDF

```http
POST /api/v1/legal/solution/pdf
Content-Type: application/json
```

请求体：

```json
{"markdown": "# 法律解决方案\n\n..."}
```

接口要求登录，Markdown 长度不超过 100000 个字符，且生成前至少包含 50 个字符。成功返回 `application/pdf` 文件流，并设置附件下载响应头。

### 8.5 会话列表和会话删除

```http
GET    /api/v1/legal/sessions?limit=20
DELETE /api/v1/legal/sessions/{session_id}
```

会话列表返回当前用户最近会话，`limit` 自动限制在 1 至 20。会话删除接口清理短期记忆和会话级派生状态，但当前实现保留历史记录、长期记忆、工作区向量和原始材料，因此该接口属于前端会话状态删除，不是数据物理全删除。

## 9. 数据和安全边界

- 公共法律库与用户私有材料使用不同检索路径和向量集合。
- 私有检索必须同时绑定 `user_id` 和允许的 `document_id`，避免跨用户读取。
- 用户认证信息通过服务端会话管理，密码只保存哈希值。
- 上传文件名、文档标识和快照路径经过规范化，避免路径穿越。
- 关键请求、登录失败、文件处理和检索异常写入结构化日志。
- 生产环境建议关闭公开调试文档、启用 HTTPS、设置 `AUTH_COOKIE_SECURE=true`，并限制 CORS 来源。
