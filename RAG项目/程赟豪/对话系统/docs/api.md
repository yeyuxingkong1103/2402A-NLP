# 接口文档（HTTP / JSON）

Base URL：`http://localhost:8000`，交互格式均为 JSON（流式接口为 SSE）。
启动后可在浏览器打开 `/docs` 查看自动生成的 Swagger 文档。

## 1. 聊天

### POST /api/chat
单轮问答（内部支持多轮记忆）。

请求体：
```json
{
  "message": "同仁堂前身始创于哪一年？",
  "session_id": "sess-xxx",
  "role_id": "doc-001",
  "use_history": true
}
```

响应体：
```json
{
  "message": "同仁堂前身……",
  "session_id": "sess-xxx",
  "sources": [
    {"text": "……", "source": "同仁堂研报.pdf", "page": 3}
  ],
  "role": {"role_id": "doc-001", "name": "王医生"}
}
```

### POST /api/chat/stream
流式输出（SSE）。请求体同 `/api/chat`。事件流：
```
data: {"delta": "同"}
data: {"delta": "仁"}
...
data: {"done": true}
```

## 2. 角色管理

### POST /api/role/create
```json
{"role_type": "doctor", "role_id": "doc-002", "name": "李医生", "custom_prompt": null}
```
响应：`{"role_id": "doc-002", "name": "李医生", "description": "专业的医疗健康顾问"}`

### GET /api/role/list
响应：`{"roles": [...], "available_types": ["doctor","lawyer",...]}`

### GET /api/role/{role_id}
响应角色详情（含 system_prompt、personality、speaking_style 等）。

## 3. 知识库（动态更新）

### POST /api/knowledge/add
```json
{"file_path": "data/raw/test.txt", "source_name": "test.txt", "chunking_method": "semantic"}
```
响应：`{"status":"ok","message":"已添加 test.txt，共 1 个分块","source":"test.txt","chunks":1}`

### POST /api/knowledge/build
批量索引 raw 目录。请求体：`{"raw_dir": null, "drop_existing": false}`

### GET /api/knowledge/stats
响应：`{"total_docs": 3, "total_chunks": 433, "sources": [...]}`

### GET /api/knowledge/list
响应：`{"sources": ["test.txt", "同仁堂研报.pdf", ...]}`

### DELETE /api/knowledge/{source}
按来源删除文档全部分块。响应：`{"status":"ok","source":"test.txt","deleted":true}`

## 4. 会话 & 健康检查

### DELETE /api/session/{session_id}
清除会话短期记忆。响应：`{"status":"ok","message":"会话 xxx 已清除"}`

### GET /api/health
响应：`{"status":"ok","timestamp":"2026-09-16T..."}`
