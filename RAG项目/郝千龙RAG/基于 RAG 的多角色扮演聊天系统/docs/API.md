# 接口文档 · 基于 RAG 的角色扮演系统

- 基础地址：`http://<host>:8000`
- 请求/响应：HTTP JSON（除流式接口外）
- 认证：除 `/health` 外，业务接口需 Header `Authorization: Bearer <token>`
- 字符集：UTF-8

## 1. 健康检查

### GET /health
- 请求：无
- 响应：`200 OK`
```json
{ "status": "ok", "env": "development", "kb_size": 128000 }
```

## 2. 认证

### POST /api/auth/register
- 请求体：
```json
{ "username": "alice", "password": "alice1234" }
```
- 响应 `200`：
```json
{ "user_id": 2, "username": "alice", "token": "2:xxxx:yyyy" }
```
- 错误：`400 用户名已存在`

### POST /api/auth/login
- 请求体：
```json
{ "username": "alice", "password": "alice1234" }
```
- 响应 `200`：
```json
{ "user_id": 2, "username": "alice", "token": "2:xxxx:yyyy" }
```
- 错误：`401 用户名或密码错误`

## 3. 角色列表

### GET /api/roles
- 请求：无
- 响应 `200`：
```json
{
  "roles": [
    { "code": "teacher", "name": "英语教师", "domain": "英语学习", "description": "..." },
    { "code": "doctor", "name": "医生", "domain": "医疗", "description": "..." }
  ]
}
```

## 4. 对话

### POST /api/chat
- Header：`Authorization: Bearer <token>`
- 请求体：
```json
{
  "query": "I miss you 怎么翻译？",
  "role_code": "teacher",
  "session_id": null,
  "top_k": 6,
  "stream": false
}
```
- 响应 `200`（非流式）：
```json
{
  "session_id": 12,
  "answer": "I miss you 可译为「我很想你」。",
  "retrieved": "1. [4.21] I miss you. <-> 我很想你。",
  "hit_count": 3
}
```
- 响应 `200`（流式，`stream=true`）：`text/plain; charset=utf-8`，逐 chunk 返回 token。

## 5. 知识库动态更新

### POST /api/kb/upload
- Header：`Authorization: Bearer <token>`
- 请求：`multipart/form-data`，字段 `file`（txt/pdf/tsv）
- 响应 `200`：
```json
{ "filename": "law.pdf", "chunks": 32 }
```

## 6. 错误码

| 状态码 | 含义 |
| --- | --- |
| 200 | 成功 |
| 400 | 参数错误 / 用户名已存在 |
| 401 | 未登录 / token 无效 |
| 422 | 请求体校验失败 |
| 500 | 服务内部错误 |

## 7. 字段速查

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| username | string(2~32) | 用户名 |
| password | string(4~64) | 密码 |
| query | string | 用户问题 |
| role_code | string | 角色 code（teacher / doctor / lawyer / …） |
| session_id | int? | 会话 id；首次传 null，由响应返回后复用 |
| top_k | int? | 检索条数，默认 6 |
| stream | bool | 是否流式输出，默认 false |
| answer | string | 模型回答 |
| retrieved | string | 检索资料格式化文本 |
| hit_count | int | 召回条数 |
| kb_size | int | 当前知识库条目数 |
| chunks | int | 本次上传分块数 |

## 8. 调用示例（curl）

```bash
# 注册
curl -X POST http://127.0.0.1:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"alice1234"}'

# 登录拿 token
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"alice1234"}' | python -c "import sys,json;print(json.load(sys.stdin)['token'])")

# 对话（非流式）
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"I miss you 怎么翻译？","role_code":"teacher"}'

# 流式
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"go to sleep","stream":true}'

# 上传知识库
curl -X POST http://127.0.0.1:8000/api/kb/upload \
  -H "Authorization: Bearer $TOKEN" -F "file=@law.pdf"
```
