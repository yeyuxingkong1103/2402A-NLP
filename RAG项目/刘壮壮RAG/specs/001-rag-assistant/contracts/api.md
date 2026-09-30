# API 接口契约

Base URL: `/api`。除 `health`、`auth/register`、`auth/login` 外，所有接口需 `Authorization: Bearer <JWT>`。

约定：
- 请求/响应 JSON（除文件上传 multipart、SSE 流）。
- 错误返回 `{"detail": "<消息>"}`。
- 401 = 未认证/凭证无效；403 = 无权限（本系统用 404 兜底越权，避免资源存在性泄露）；404 = 资源不存在；409 = 冲突；422 = 校验失败。

---

## Health

### `GET /api/health`
响应 200: `{"status": "ok"}`

---

## Auth

### `POST /api/auth/register`
请求:
```json
{"username": "alice", "password": "secret123"}
```
响应 201:
```json
{"id": 1, "username": "alice", "created_at": "2026-09-22T00:00:00"}
```
- username 3-64，password 6-128；重名返回 409。

### `POST /api/auth/login`
请求:
```json
{"username": "alice", "password": "secret123"}
```
响应 200:
```json
{"access_token": "<jwt>", "token_type": "bearer"}
```
- 用户名或密码错返回 401（不区分哪个错）。

### `GET /api/auth/me`
响应 200:
```json
{"id": 1, "username": "alice", "created_at": "..."}
```

---

## Characters

### `GET /api/characters/templates`
响应 200（数组）:
```json
[{"key": "tcm_doctor", "name": "中医", "description": "...", "system_prompt": "..."}]
```

### `POST /api/characters`
请求:
```json
{
  "name": "我的中医",
  "system_prompt": "你是中医（可空，配 template_key 用模板）",
  "model_name": "gpt-4o-mini",
  "base_url": null,
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 2048,
  "template_key": "tcm_doctor"
}
```
响应 201: CharacterRead（含 id, created_at, updated_at）。

### `GET /api/characters`
响应 200: CharacterRead 数组（仅当前用户的角色）。

### `GET /api/characters/{id}`
响应 200: CharacterRead。非本人/不存在返回 404。

### `PATCH /api/characters/{id}`
请求: CharacterUpdate（部分字段，见下）。响应 200: CharacterRead。
```json
{"temperature": 0.3, "system_prompt": "新的提示词"}
```

### `DELETE /api/characters/{id}`
响应 204。级联清理会话/知识库/记忆。

---

## Knowledge（角色知识库）

### `POST /api/characters/{character_id}/knowledge`（multipart）
字段: `file`（UploadFile）。响应 201: KnowledgeFileRead。
- 支持 txt/md/pdf/png/jpg/jpeg/webp/bmp；不支持格式或超过 100MB 返回 400。
- 落盘 + DB `status=pending`，异步投递 `process_file`。

### `GET /api/characters/{character_id}/knowledge`
响应 200: KnowledgeFileRead 数组。

### `DELETE /api/characters/{character_id}/knowledge/{file_id}`
响应 204。级联清理磁盘 + DB + Milvus 向量。

---

## Conversations（角色下的会话）

### `POST /api/characters/{character_id}/conversations`
请求:
```json
{"title": "第一次聊天"}
```
响应 201: ConversationRead（含 id, character_id, title, created_at）。

### `GET /api/characters/{character_id}/conversations`
响应 200: ConversationRead 数组（按 id 倒序）。

### `DELETE /api/characters/{character_id}/conversations/{conversation_id}`
响应 204。同时清空 Redis 短期记忆。

---

## Chat（SSE 流式对话）

### `POST /api/characters/{character_id}/conversations/{conversation_id}/chat`
请求:
```json
{"message": "你好"}
```
响应 200: `Content-Type: text/event-stream`。每个事件:
```
data: {"delta": "你"}
data: {"delta": "好"}
data: [DONE]
```
- 每轮：检索知识库+长期记忆 → 组装 → 流式生成 → 写回 Redis 短期记忆。
- 达阈值（10 条用户消息）异步投递 `extract_memory`。

---

## Schemas 参考

**CharacterRead**:
```json
{
  "id": 1, "name": "...", "system_prompt": "...", "model_name": "...",
  "base_url": null, "temperature": 0.7, "top_p": 1.0, "max_tokens": 2048,
  "created_at": "...", "updated_at": "..."
}
```

**KnowledgeFileRead**:
```json
{"id": 1, "filename": "a.txt", "file_type": "text", "status": "pending", "created_at": "..."}
```

**ConversationRead**:
```json
{"id": 1, "character_id": 2, "title": "...", "created_at": "..."}
```
