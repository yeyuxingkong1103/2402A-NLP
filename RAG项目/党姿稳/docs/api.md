# 接口文档

服务默认监听 `0.0.0.0:8080`，启动后可在 `/docs` 查看自动生成的交互式文档。

```bash
bash scripts/start.sh api      # 只启动 API
```

## 1. 普通对话

一次性返回完整回答，适合脚本调用与批量测试。

```
POST /chat
```

**请求体**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `user_id` | string | 否 | 用户标识，用于隔离记忆，默认 `default` |
| `message` | string | 是 | 用户问题 |

```json
{
  "user_id": "user_001",
  "message": "劳动合同到期不续签，公司需要支付经济补偿吗？"
}
```

**响应**

```json
{
  "reply": "根据《劳动合同法》第四十六条……",
  "domain": "legal",
  "role": "法律顾问",
  "rewritten_query": "劳动合同到期不续签，公司需要支付经济补偿吗？",
  "sources": [
    {
      "index": 1,
      "file": "劳动法常见问题.pdf",
      "page": 5,
      "score": 0.9123,
      "preview": "用人单位依法解除劳动合同的，应当向劳动者支付经济补偿……"
    }
  ],
  "grounded": true,
  "elapsed": 3.21
}
```

| 字段 | 说明 |
|---|---|
| `domain` | 识别到的领域：`legal` / `medical` / `english` / `chat` |
| `role` | 路由到的角色名，`chat` 时为「通用助手」 |
| `rewritten_query` | Query 改写后的检索用查询，无历史上下文时与原文相同 |
| `sources` | 引用的知识库片段。文件名与页码分开给：`file` 是文件名，`page` 是页码（0 表示纯文本导入、没有页码） |
| `grounded` | 本轮是否有知识库资料支撑。为 `false` 时回答以「资料中未找到相关内容」开头，来源标为「通用知识」 |

`grounded` 为 `false` 时 `sources` 只有一条占位：

```json
[{"index": 0, "type": "general", "file": "通用知识", "page": 0, "score": 0.0, "preview": ""}]
```

## 2. 流式对话（SSE）

以 Server-Sent Events 逐段推送，适合网页端实时展示。

```
POST /chat/stream
```

**请求体**：与 `/chat` 完全一致。

**响应**：`text/event-stream`，每帧形如 `event: <类型>` + `data: <JSON>`。

事件类型依次为：

| 事件 | 时机 | 关键字段 |
|---|---|---|
| `meta` | 检索完成、开始生成前 | `domain` `role` `sources` `rewritten_query` `grounded` `prepare_seconds` |
| `delta` | 每产出一段文本 | `content` |
| `done` | 生成结束 | `reply` `domain` `role` `sources` `grounded` `elapsed` |
| `error` | 中途失败 | `message` |

`meta` 与 `done` 都带 `grounded`：本轮没有检索到任何知识库资料时为 `false`，前端可以据此提示「本轮回答没有资料支撑」。

```
event: meta
data: {"type":"meta","domain":"medical","role":"医疗咨询","sources":[...],"grounded":true}

event: delta
data: {"type":"delta","content":"普通感冒"}

event: delta
data: {"type":"delta","content":"多由病毒引起"}

event: done
data: {"type":"done","reply":"普通感冒多由病毒引起……","grounded":true,"elapsed":4.02}
```

**前端示例**

```javascript
const response = await fetch("/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ user_id: "u_5ee443c5", message: "感冒要吃抗生素吗" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const frames = buffer.split("\n\n");
  buffer = frames.pop();
  for (const frame of frames) {
    const line = frame.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    const event = JSON.parse(line.slice(6));
    if (event.type === "meta") renderSources(event.sources, event.grounded);
    if (event.type === "delta") appendText(event.content);
    if (event.type === "error") showError(event.message);
  }
}
```

## 3. 导入知识库 PDF

按已有路径导入：

```
POST /kb/import
{ "domain": "legal", "pdf_path": "data/legal/劳动法常见问题.pdf", "strategy": "parent_child" }
```

上传并导入（`multipart/form-data`），文件会保存到 `data/<domain>/` 后立即入库：

```
POST /kb/import/upload
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `domain` | string | 是 | `legal` / `medical` / `english` |
| `pdf_path` | string | 仅 `/kb/import` | PDF 路径，**必须位于项目的 `data` 目录下** |
| `file` | file | 仅 `/kb/import/upload` | PDF 文件 |
| `strategy` | string | 否 | 分块策略，默认 `parent_child` |

可选策略：`fixed` `sentence` `paragraph` `heading` `semantic` `parent_child`

```bash
curl -X POST http://127.0.0.1:8080/kb/import/upload \
  -F "domain=medical" -F "file=@常见疾病诊疗指南.pdf"
```

**响应**（两个接口格式相同）

```json
{
  "file": "劳动法常见问题.pdf", "domain": "legal", "strategy": "parent_child",
  "pages": 10, "raw_chunks": 142, "filtered_chunks": 138, "stored": 152
}
```

`stored` 大于 `filtered_chunks` 是因为父子块策略下父块与子块都会入库。

## 4. 知识库检索

只做检索不生成回答，用于调试召回效果。

```
POST /kb/search
```

```json
{ "query": "经济补偿金怎么算", "domain": "legal", "top_k": 5 }
```

**响应**

```json
{
  "query": "经济补偿金怎么算",
  "domain": "legal",
  "count": 3,
  "results": [
    {
      "text": "经济补偿按劳动者在本单位工作的年限……",
      "source": "劳动法常见问题.pdf",
      "page": 5,
      "chunk_type": "parent_child",
      "score": 0.412,
      "rerank_score": 0.9871
    }
  ]
}
```

`score` 是余弦相似度（用于阈值过滤），`rerank_score` 是重排分数（用于排序）。
`source` 只是文件名，页码单独放在 `page`（`0` 表示纯文本导入、没有页码）。

## 5. 知识库统计

```
GET /kb/stats
```

```json
{
  "legal": { "collection": "kb_legal", "chunks": 152, "documents": 3 },
  "medical": { "collection": "kb_medical", "chunks": 98, "documents": 2 },
  "english": { "collection": "kb_english", "chunks": 0, "documents": 0 }
}
```

## 6. 查看与清空用户记忆

```
GET    /memory/{user_id}
DELETE /memory/{user_id}
```

```json
{
  "user_id": "u_5ee443c5", "rounds": 6,
  "short_term": [
    { "role": "user", "content": "什么情况下可以要经济补偿", "ts": 1758000000 },
    { "role": "assistant", "content": "……", "ts": 1758000003 }
  ],
  "long_term": [
    { "id": 12, "text": "用户问题：什么情况下可以要经济补偿……",
      "fact_type": "summary", "domain": "legal", "role": "法律顾问", "created_at": 1758000003 }
  ]
}
```

`short_term` 最多保留最近 5 轮对话（10 条消息）；`rounds` 是累计对话轮次，用于判断是否到了提炼关系记忆的时机。
`long_term` 里 `fact_type` 为 `summary` 的是每轮沉淀的问答摘要，为 `profile` 的是每 5 轮提炼一次的偏好与已确认事实；`domain` 用于按领域隔离召回。
`DELETE` 会同时清掉短期、长期与轮次计数。

## 7. 查看所有角色

```
GET /roles
```

```json
{
  "roles": [
    { "name": "法律顾问", "domain": "legal", "identity": "你是一名执业多年的中国法律顾问……" },
    { "name": "医疗咨询", "domain": "medical", "identity": "你是一名临床经验丰富的全科医生……" },
    { "name": "英语学习助手", "domain": "english", "identity": "你是一名资深英语教师……" }
  ],
  "chat_role": "通用助手"
}
```

## 8. 健康检查

```
GET /            返回运行模式、当前模型、各知识库条数与配置告警
GET /health/llm  发一次最小请求验证模型连通性，返回延迟与样例输出，用于排查密钥或地址问题
```

```json
{
  "status": "ok", "local_mode": false, "embedding_backend": "bge-m3",
  "llm": { "provider": "deepseek", "model": "deepseek-v4-flash",
           "base_url": "https://api.deepseek.com/v1", "api_key_configured": true },
  "kb": { "legal": { "chunks": 152, "documents": 3 } },
  "config_warnings": []
}
```

## 错误响应

所有接口在出错时返回标准 HTTP 状态码与 `detail` 字段：

```json
{ "detail": "未知领域：xxx" }
```

| 状态码 | 含义 |
|---|---|
| 400 | 参数错误（领域不存在、路径越界、文件类型不符） |
| 404 | 文件不存在 |
| 500 | 检索或模型调用失败 |
