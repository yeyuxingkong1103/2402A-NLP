# 04 — 接口文档

> 版本：1.0.0 ｜ 基础地址：`http://127.0.0.1:8000` ｜ 前缀：`/api/v1`
> 在线 Swagger：`http://127.0.0.1:8000/docs`（ReDoc：`/redoc`，OpenAPI JSON：`/openapi.json`）
> 本文与代码实现对齐（`src/api/v1/`、`src/schemas/`），如有出入以代码为准。

---

## 1. 通用约定

### 1.1 统一响应体

所有接口返回：

```json
{
  "code": 0,
  "message": "success",
  "data": { }
}
```

- `code = 0` 表示成功；非 0 表示业务错误，HTTP 状态码同步返回。
- `data` 可为 `null`。

### 1.2 鉴权

- 方式：`Authorization: Bearer <access_token>`（JWT HS256）。
- Access Token 有效期 `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`（默认 1440 分钟）；Refresh Token 有效期 `JWT_REFRESH_TOKEN_EXPIRE_MINUTES`（默认 10080 分钟）。
- 免鉴权接口：`GET /`、`GET /health`、`POST /auth/register`、`POST /auth/login`、`POST /auth/refresh`、`GET /personas`、`GET /personas/{id}`、`/docs` 等文档接口。

### 1.3 限流

`POST /chat` 与 `POST /chat/stream` 受单用户限流（`RATE_LIMIT_PER_MINUTE`，默认 30 次/分钟）。Redis 不可用时放行（有意降级）。`POST /auth/login` 另按 IP 限流（`LOGIN_RATE_LIMIT_PER_MINUTE`，默认 10 次/分钟），防暴力破解。所有响应携带 `X-Request-ID` 头，服务端日志全链路携带同一 rid。

### 1.4 错误码

| HTTP | code | 异常类 | 典型场景 |
| :--- | :--- | :--- | :--- |
| 400 | 400 | `AppError` | 不支持的文件类型、参数业务校验失败 |
| 401 | 401 | `AuthError` | 未登录、token 过期/伪造、密码错误 |
| 403 | 403 | `PermissionError_` | 需要管理员权限、访问他人会话 |
| 404 | 404 | `NotFoundError` | 用户/角色/会话/文档不存在 |
| 409 | 409 | `ConflictError` | 用户名/邮箱/角色编码重复 |
| 429 | 429 | `RateLimitError` | 触发限流、角色未开放（`status != 1`） |
| 500 | 500 | 兜底 | 未捕获异常 |
| 502 | 502 | `ExternalServiceError` | 大模型调用失败、未配置 `LLM_API_KEY` |

---

## 2. 系统接口

### 2.1 GET `/` — 服务信息

免鉴权。返回服务名称、版本与免责声明。

### 2.2 GET `/health` — 健康检查

免鉴权。探测 MySQL / Redis / Milvus 连通性。

```json
{
  "code": 0, "message": "success",
  "data": { "status": "ok", "mysql": true, "redis": true, "milvus": true, "detail": null }
}
```

---

## 3. 认证接口（`/auth`）

### 3.1 POST `/auth/register` — 用户注册（U-01）

免鉴权。

请求体：

| 字段 | 类型 | 必填 | 校验 |
| :--- | :--- | :--- | :--- |
| username | string | 是 | 3~64 字符，全局唯一 |
| password | string | 是 | 6~128 字符，bcrypt 加密存储 |
| email | string | 否 | 需包含 `@`，全局唯一 |
| phone | string | 否 | ≤32 字符 |
| nickname | string | 否 | ≤64 字符 |

```json
// 请求
{"username": "demo001", "password": "demo123456", "nickname": "演示用户"}
// 响应
{"code": 0, "message": "注册成功", "data": {"user_id": 1, "username": "demo001"}}
```

### 3.2 POST `/auth/login` — 登录（U-02）

免鉴权。请求体 `{"username": "...", "password": "..."}`。

响应 `data`（`TokenResponse`）：

```json
{
  "access_token": "eyJhbGciOi...", "refresh_token": "eyJhbGciOi...",
  "token_type": "bearer", "expires_in": 86400,
  "user_id": 1, "username": "demo001", "roles": ["user"]
}
```

登录成功会写 `login_logs`（IP、User-Agent）与审计日志。

### 3.3 POST `/auth/refresh` — 刷新 Token

免鉴权。请求体 `{"refresh_token": "..."}`。响应同登录（新 access/refresh token）。

### 3.4 POST `/auth/logout` — 退出登录

需登录。写审计日志（action=logout），响应 `data: null`。

### 3.5 GET `/auth/me` / PUT `/auth/me` — 当前用户

需登录。GET 返回用户资料（含 `roles`）；PUT 接收 `UserUpdateRequest`（见 4.2）。

---

## 4. 用户接口（`/users`）

### 4.1 GET `/users/me` — 当前用户资料（U-03）

需登录。返回 id、username、nickname、avatar、email、phone、status、roles、默认角色偏好等。

### 4.2 PUT `/users/me` — 修改资料（U-04）

需登录。所有字段可选（传 null/缺省即不改）：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| nickname | string | ≤64 |
| avatar | string | ≤512，图片 URL |
| email / phone | string | 邮箱需含 `@` |
| default_persona_id | int | 默认心理医生角色 ID |

### 4.3 POST `/users/me/password` — 修改密码

需登录。`{"old_password": "...", "new_password": "..."}`（新密码 6~128）。成功后旧 token 不强制失效（无状态 JWT）。

### 4.4 GET `/users/me/preferences` — 我的角色偏好（U-08）

需登录。返回 `[{persona_id, persona_code, persona_name, is_default}]`。

### 4.5 POST `/users/me/preferences` — 设置默认心理医生（P-03）

需登录。`{"persona_id": 2, "is_default": true}`。响应 `{"default_persona_id": 2}`。

### 4.6 GET `/users/me/login-logs` — 我的登录日志（U-07）

需登录。Query：`limit`（1~200，默认 20）。

---

## 5. 心理医生角色接口（`/personas`）

### 5.1 GET `/personas` — 角色列表（P-01）

免鉴权。Query：`include_inactive`（默认 false，仅返回 `status=1`）。

响应：`{"items": [...], "total": n}`；列表项不含 `system_prompt`（需详情获取）。角色字段见 5.2。

### 5.2 GET `/personas/{persona_id}` — 角色详情（P-02）

免鉴权。返回完整字段：

| 字段 | 说明 |
| :--- | :--- |
| id / persona_code / name | 主键、编码（`humanistic_lin`/`cbt_chen`/`mindfulness_zhou`）、名称 |
| title / therapy_type / style / methods | 头衔、流派、风格、核心方法 |
| greeting | 开场白 |
| system_prompt | 完整角色提示词 |
| knowledge_scope / avatar / status | 知识范围、头像、上下架状态 |
| model_params | 如 `{"temperature": 0.8, "max_tokens": 2048, "top_p": 0.9}` |
| safety_boundary | 安全边界文本 |

### 5.3 POST `/personas` — 新增角色（P-08，管理员）

必填：`persona_code`（2~64，唯一）、`name`、`system_prompt`；其余字段同详情。注意：动态新增的角色不会自动获得 `KNOWLEDGE_DIRS` 知识库目录映射，持久化角色建议走 `scripts/seed_personas.py`。

### 5.4 PUT `/personas/{persona_id}` — 编辑角色（管理员）

字段全部可选。

### 5.5 POST `/personas/{persona_id}/status` — 角色上下架（P-07，管理员）

Query：`status`（1=上架，0=下架）。下架后 `/chat` 返回 429（"该心理医生暂未开放"）。

---

## 6. 会话接口（`/conversations`）

### 6.1 POST `/conversations` — 新建会话（C-01）

需登录。`{"persona_id": 1, "title": "可选"}`。响应含会话信息与角色 `greeting`（开场白）。会话绑定 `(user_id, persona_id)`，同一用户可对不同角色各建独立会话。

### 6.2 GET `/conversations` — 会话列表（C-08）

需登录。Query：`persona_id`（可选过滤）、`limit`（1~500，默认 100）。仅返回本人未删除会话，按更新时间倒序。

### 6.3 GET `/conversations/{conversation_id}` — 会话详情

需登录。访问他人会话返回 403。

### 6.4 GET `/conversations/{conversation_id}/messages` — 历史消息（C-05）

需登录。Query：`limit`（1~500，默认 200）、`offset`（默认 0）。响应：`{"conversation_id", "total", "items": [{id, role, content, tokens, refs, created_at}]}`；`refs` 为该条助手消息引用的知识片段。

### 6.5 DELETE `/conversations/{conversation_id}` — 删除会话（C-07）

需登录。逻辑删除（`status=0`），`data: null`。

### 6.6 POST `/conversations/{conversation_id}/save-memory` — 手动保存长期记忆

需登录。强制把当前会话摘要写入 Milvus 长期记忆（C-04）。响应 `{"conversation_id", "summary"}`；无内容时 `data: null`。

---

## 7. 对话接口

### 7.1 POST `/chat` — 非流式聊天（R-01 ~ R-10）

需登录。请求体（`ChatRequest`）：

| 字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| persona_id | int | 是 | 心理医生角色 ID |
| message | string | 是 | 1~4000 字符 |
| conversation_id | int | 否 | 不传则自动新建会话 |
| stream | bool | 否 | 忽略（流式请用 `/chat/stream`） |

响应 `data`（`ChatResponse`）：

```json
{
  "conversation_id": 1001,
  "persona_id": 1,
  "answer": "听起来你这段时间很不容易……",
  "references": [{"doc_id": 10, "chunk_id": 3, "source": "心理学与生活.pdf", "score": 0.87}],
  "tokens": 256,
  "finish_reason": "stop",
  "crisis_detected": false,
  "crisis_notice": null
}
```

- `crisis_detected=true` 时 `answer` 末尾附带危机转介提示（热线 `CRISIS_HOTLINE` 与 120/110）。
- 处理链路：限流 → Query 改写 → BGE-M3 向量化 → Milvus 混合检索 → 重排 → 提示词拼接（知识+长短期记忆）→ DeepSeek 生成 → 后处理 → 落库（MySQL + Redis）。

### 7.2 POST `/chat/stream` — SSE 流式聊天（C-06）

需登录。请求体同 7.1。响应 `Content-Type: text/event-stream`，事件类型（阶段 4 首 token 优化：`meta` 在建会话后、检索前立即发出；引用列表随 `done` 下发）：

| 事件 | data 字段 | 说明 |
| :--- | :--- | :--- |
| `meta` | conversation_id, persona_id, persona_name, crisis_detected | 会话元信息（首事件，检索前发出） |
| `delta` | content | 增量文本片段（可多次） |
| `done` | conversation_id, tokens, finish_reason, crisis_detected, references | 结束标记 |
| `error` | message | 出错（如大模型失败、会话并发锁占用） |

示例（curl）：

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"persona_id":2,"message":"我总觉得自己做得不够好。"}'
```

事件帧格式：

```
event: meta
data: {"conversation_id": 1001, "persona_id": 2, "persona_name": "陈认知医生", "crisis_detected": false}

event: delta
data: {"content": "我们一起看看"}

event: done
data: {"conversation_id": 1001, "tokens": 180, "finish_reason": "stop", "crisis_detected": false, "references": []}
```

---

## 8. 知识库接口（`/knowledge`，管理员）

### 8.1 POST `/knowledge/upload` — 上传文档并入库（K-01 ~ K-09）

`multipart/form-data`：

| 字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| persona_id | int | 是 | 归属角色 |
| strategy | string | 否 | 分块策略：`fixed`/`sentence`/`paragraph`（默认）/`heading`/`semantic` |
| file | file | 是 | 仅支持 pdf / txt / md / docx |

响应 `{"doc_id", "title", "persona_id", "status", "chunk_count", "stored"}`。文件落 `data/uploads/{时间戳}_{原文件名}`；解析失败（如扫描件无文本层）时文档状态记 `failed` 并带 `error_msg`。

### 8.2 GET `/knowledge/docs` — 文档列表（K-10）

Query：`persona_id`（可选）。返回文档元数据（status：pending/processing/done/failed）。

### 8.3 DELETE `/knowledge/docs/{doc_id}` — 删除文档（K-11）

同时删除 Milvus 中对应向量与 MySQL 元数据。

### 8.4 POST `/knowledge/rebuild` — 按角色重建索引（K-11/K-12）

请求体：`{"persona_id": 2 或 null(全部), "drop_existing": false}`。仅处理 `KNOWLEDGE_DIRS` 已配置的角色（三个种子角色）。响应 `{"results": [...]}`。

### 8.5 POST `/knowledge/search` — 检索测试（R-01 ~ R-06）

管理员。`{"persona_id": 1, "query": "如何缓解失眠", "top_k": 5}`。响应含 `rewritten_query`（LLM 改写结果）与 `hits`（doc_id/chunk_id/source/text/score）。

### 8.6 GET `/knowledge/stats` — 知识库统计

各角色 chunks 数与 Milvus 向量数。

---

## 9. 管理后台接口（`/admin`，管理员）

| 接口 | 方法 | Query / Body | 说明 |
| :--- | :--- | :--- | :--- |
| `/admin/users` | GET | `page`(默认1), `page_size`(≤100), `keyword`, `status` | 用户列表（U-05） |
| `/admin/users/{user_id}/status` | POST | `{"status": 0或1}` | 禁用/启用用户，写审计日志 |
| `/admin/logs/login` | GET | `user_id`, `limit`(≤500) | 登录日志（U-07） |
| `/admin/logs/audit` | GET | `user_id`, `limit`(≤500) | 审计日志 |
| `/admin/conversations` | GET | `user_id`, `limit`(≤200) | 全量会话审计 |
| `/admin/monitor` | GET | — | 系统监控：MySQL/Redis/Milvus 健康、模型加载状态、各角色向量数与知识库目录 |

---

## 10. 评测接口

### 10.1 POST `/eval/ragas` — RAGAS 评测（管理员）

请求体（`RagasEvalRequest`）：

| 字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| persona_id | int | 是 | 被评测角色 |
| dataset_path | string | 否 | 自定义 jsonl 数据集路径，默认 `data/eval/ragas_dataset.jsonl` |
| limit | int | 否 | 评测样本数上限，默认 10 |

数据集每行：`{"persona_code": "cbt_chen", "question": "...", "ground_truth": "可选参考答案"}`。

响应 `{"persona_id", "samples", "metrics", "report_path"}`：

- `metrics`：`faithfulness` / `answer_relevancy` / `context_precision` / `context_recall`（0~1）。
- `engine`：`builtin_llm_judge`（默认回退引擎）或 `ragas`（当前环境 ragas 0.4.3 与 langchain-community 0.4.2 冲突不可导入，故实际走内置 Judge，指标口径一致）。
- 报告 JSON 落 `data/eval/reports/ragas_report_persona{id}_{时间戳}.json`。

---

## 11. 调用示例（端到端）

```bash
# 1) 登录
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo001","password":"demo123456"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['data']['access_token'])")

# 2) 新建会话（林知暖）
curl -s -X POST http://127.0.0.1:8000/api/v1/conversations \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"persona_id":1}'

# 3) 多轮聊天
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"persona_id":1,"conversation_id":1001,"message":"我最近总是失眠，心里很慌。"}'
```
