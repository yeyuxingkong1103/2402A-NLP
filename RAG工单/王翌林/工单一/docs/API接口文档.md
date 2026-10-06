# API 接口文档

> **工单编号**：人工智能NLP-RAG-基于PDF文档的问答系统
> **版本**：v1.0.0
> **Base URL**：`http://127.0.0.1:8001`
> **协议**：HTTP/1.1, JSON (SSE 为例外)
> **OpenAPI**：`http://127.0.0.1:8001/docs` （FastAPI 自动生成 Swagger UI）

---

## 一、接口总览

| # | 方法 | 路径 | 说明 | 认证 |
|---|------|------|------|------|
| 1 | `GET` | `/api/health` | 健康检查 | 无 |
| 2 | `GET` | `/api/stats` | 系统统计 | 无 |
| 3 | `POST` | `/api/ask` | 同步问答（RAG / LLM） | 无 |
| 4 | `POST` | `/api/ask_stream` | SSE 流式问答 | 无 |
| 5 | `POST` | `/api/feedback` | 提交反馈 | 无 |
| 6 | `GET` | `/api/questions` | 示例问题列表 | 无 |

---

## 二、通用说明

### 2.1 请求格式

- Content-Type：`application/json`
- 字符编码：`UTF-8`
- 时间：毫秒（ms）
- 数字：浮点数

### 2.2 响应格式

**成功**：直接返回 Pydantic 模型定义的 JSON 对象

**失败**：
```json
{
  "detail": "错误描述"
}
```

### 2.3 错误码

| HTTP Code | 说明 |
|-----------|------|
| `200 OK` | 成功 |
| `400 Bad Request` | 请求参数校验失败（Pydantic 自动） |
| `422 Unprocessable Entity` | 请求体字段校验失败 |
| `500 Internal Server Error` | 后端异常（detail 含 str(e)） |
| `503 Service Unavailable` | RAG Engine / 依赖服务未就绪 |

### 2.4 限流 / 超时

| 项目 | 值 |
|------|-----|
| 请求体上限 | 2000 字符（question） |
| top_k 范围 | 1-20 |
| 建议超时 | 30s（RAG 含 reranker + LLM） |
| SSE 建议超时 | 60s |

---

## 三、接口详情

### 3.1 健康检查

```
GET /api/health
```

**说明**：返回 MySQL / Milvus / Embedding / LLM 状态，用于负载均衡探活和系统自检。

**请求参数**：无

**响应示例**：
```json
{
  "status": "ok",
  "mysql": "ok (8.0.46-0ubuntu0.22.04.4)",
  "milvus": "ok (remote)",
  "embedding_model": "/home/dabaie/models/bge-m3",
  "llm_model": "deepseek-v4-flash",
  "uptime_seconds": 14.3
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 固定 `ok`（只要 API 本身活着） |
| mysql | string | `ok (版本号)` / `failed: 原因` |
| milvus | string | `ok (mode)` mode=remote/lite | `failed: 原因` |
| embedding_model | string | 嵌入模型路径或名称 |
| llm_model | string | LLM 模型名称 |
| uptime_seconds | float | API 运行秒数 |

---

### 3.2 系统统计

```
GET /api/stats
```

**说明**：返回文档数、Chunk 数、QA 日志数、反馈数、向量实体数。

**请求参数**：无

**响应示例**：
```json
{
  "total_documents": 1,
  "total_chunks": 1303,
  "total_qa_logs": 39,
  "total_feedback": 5,
  "collection_entities": 1303
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| total_documents | int | MySQL documents 表记录数 |
| total_chunks | int | MySQL chunks 表记录数 |
| total_qa_logs | int | MySQL qa_logs 表记录数 |
| total_feedback | int | MySQL feedbacks 表记录数 |
| collection_entities | int \| null | Milvus collection 实体数（可能为 null） |

---

### 3.3 同步问答（核心）

```
POST /api/ask
Content-Type: application/json
```

**说明**：支持两种模式——RAG 检索增强（默认）和纯 LLM 问答。内部经过：查询改写 → 混合检索（向量+BM25）→ RRF 融合 → reranker 重排 → LLM 生成。

**请求体**：

```json
{
  "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
  "top_k": 5,
  "use_rag": true,
  "doc_id": null
}
```

| 字段 | 类型 | 必填 | 默认 | 校验 | 说明 |
|------|------|------|------|------|------|
| question | string | ✅ | - | 1 ≤ len ≤ 2000 | 用户问题 |
| top_k | int | ❌ | 5 | 1 ≤ top_k ≤ 20 | 最终返回的引用数量 |
| use_rag | bool | ❌ | true | - | true=RAG, false=纯 LLM |
| doc_id | string | ❌ | null | - | 限制检索范围（预留） |

**响应示例**（RAG 模式）：

```json
{
  "mode": "rag",
  "answer": "武汉兴图新科电子股份有限公司的法定代表人是**程家明**。\n\n依据：\n- [资料1]“一、发行人的基本情况”中列明：法定代表人：程家明。",
  "qa_log_id": 42,
  "references": [
    {
      "page": 52,
      "chunk_id": "_p52_c0",
      "score": 8.4453,
      "preview": "一、发行人的基本情况 法定代表人：程家明 成立日期：2015年..."
    },
    {
      "page": 52,
      "chunk_id": "_p52_c1",
      "score": 7.8120,
      "preview": "武汉兴图新科电子股份有限公司（以下简称“发行人”）..."
    }
  ],
  "latency_ms": 2100.0,
  "token_usage": {
    "prompt_tokens": 1834,
    "completion_tokens": 126,
    "total_tokens": 1960
  },
  "query_understanding": {
    "original": "法定代表人是谁？",
    "rewrite": "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "intent": "factual_query",
    "synonyms": ["法人代表", "法定负责人"]
  },
  "breakdown": {
    "vector_latency_ms": 120.0,
    "bm25_latency_ms": 45.0,
    "rerank_latency_ms": 180.0,
    "llm_latency_ms": 1700.0
  },
  "rag_answer": "...同 answer...",
  "rag_latency_ms": 2100.0,
  "llm_answer": null,
  "llm_latency_ms": null
}
```

**响应示例**（纯 LLM 模式 `use_rag=false`）：

```json
{
  "mode": "llm",
  "answer": "作为一个 AI，我没有实时数据库...",
  "qa_log_id": null,
  "references": [],
  "latency_ms": 1200.0,
  "token_usage": {"prompt_tokens": 52, "completion_tokens": 78, "total_tokens": 130},
  "query_understanding": null,
  "breakdown": null,
  "rag_answer": null,
  "rag_latency_ms": null,
  "llm_answer": "...同 answer...",
  "llm_latency_ms": 1200.0
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| mode | string | `"rag"` 或 `"llm"` |
| answer | string | 最终答案（Markdown 格式） |
| qa_log_id | int \| null | MySQL qa_logs 主键，可用于提交反馈 |
| references | ReferenceItem[] | 引用溯源列表 |
| latency_ms | float | 端到端耗时 |
| token_usage | Dict[str, int] | LLM token 用量 |
| query_understanding | Dict \| null | 查询改写结果 |
| breakdown | Dict[str, float] \| null | 各阶段耗时细分 |
| rag_answer | string \| null | RAG 模式下同 answer |
| rag_latency_ms | float \| null | RAG 模式下同 latency_ms |
| llm_answer | string \| null | 纯 LLM 模式下同 answer |
| llm_latency_ms | float \| null | 纯 LLM 模式下同 latency_ms |

**ReferenceItem**：

| 字段 | 类型 | 说明 |
|------|------|------|
| page | int \| null | 原文 PDF 页码（1-based） |
| chunk_id | string \| null | Milvus 主键 `{doc_id}_p{page}_c{idx}` |
| score | float \| null | 融合后归一化分数（越大越好） |
| preview | string \| null | 原文片段（最长 120 字） |

---

### 3.4 SSE 流式问答

```
POST /api/ask_stream
Content-Type: application/json
Accept: text/event-stream
```

**说明**：Server-Sent Events 流式输出。前端可边生成边展示，提升用户感知速度。内部流程与 `/api/ask` 相同，但答案逐字推送（每 3 字符一个 token 事件）。

**请求体**：同 `POST /api/ask`

**响应格式**：`text/event-stream`，每个事件一条 `data: {...}\n\n`，事件之间空行分隔。

**事件类型**：

| type | 字段 | 说明 |
|------|------|------|
| `status` | `msg` | 状态提示（如"正在检索..."） |
| `retrieved` | `count`, `latency_ms`, `references` | 检索完成，含 top-k 引用列表 |
| `token` | `text` | 逐字答案片段（每 3 字符推送一次） |
| `done` | `latency_ms`, `token_usage` | 答案完成 |
| `error` | `msg` | 发生错误（提前终止） |
| - | `[DONE]` | 流结束标记（data 值为字符串 `[DONE]`） |

**原始流示例**：

```
data: {"type": "status", "msg": "正在检索..."}

data: {"type": "retrieved", "count": 5, "latency_ms": 345.2, "references": [{"page": 52, "chunk_id": "_p52_c0", "score": 8.44, "preview": "一、发行人的基本情况 法定代表人..."}]}

data: {"type": "token", "text": "武汉"}

data: {"type": "token", "text": "兴图"}

data: {"type": "token", "text": "新科"}

...

data: {"type": "token", "text": "程家明。"}

data: {"type": "done", "latency_ms": 2100.0, "token_usage": {"prompt_tokens": 1834, "completion_tokens": 126, "total_tokens": 1960}}

data: [DONE]
```

**curl 测试**：

```bash
curl -N -X POST http://127.0.0.1:8001/api/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"question":"法定代表人是谁？","top_k":3,"use_rag":true}'
```

**前端 JavaScript（EventSource）**：

```javascript
// EventSource 只支持 GET，POST 需用 fetch
const resp = await fetch('/api/ask_stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({question: '法定代表人是谁？', top_k: 5, use_rag: true})
});
const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buffer = '';
while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, {stream: true});
  const lines = buffer.split('\n\n');
  buffer = lines.pop();
  for (const line of lines) {
    const data = line.replace(/^data: /, '').trim();
    if (!data) continue;
    if (data === '[DONE]') return;
    const evt = JSON.parse(data);
    if (evt.type === 'token') appendToAnswer(evt.text);
    else if (evt.type === 'done') onComplete(evt);
    else if (evt.type === 'retrieved') showRefs(evt.references);
  }
}
```

---

### 3.5 提交反馈

```
POST /api/feedback
Content-Type: application/json
```

**说明**：对某条问答结果提交评分，用于后续质量分析和 RAG 优化。建议在用户点击"👍"或"👎"时调用。

**请求体**：

```json
{
  "qa_log_id": 42,
  "rating": 5,
  "is_correct": 1,
  "comment": "非常准确！"
}
```

| 字段 | 类型 | 必填 | 默认 | 校验 | 说明 |
|------|------|------|------|------|------|
| qa_log_id | int | ✅ | - | 存在性 | 来自 `/api/ask` 返回的 qa_log_id |
| rating | int | ✅ | - | 1 ≤ rating ≤ 5 | 1=很差, 5=很好 |
| is_correct | int | ❌ | null | 0 / 1 | 答案是否事实正确 |
| comment | string | ❌ | null | ≤ 500 字 | 文字反馈 |

**响应示例**：
```json
{
  "id": 15,
  "ok": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | feedbacks 表自增主键 |
| ok | bool | 固定 true（成功写入） |

**错误场景**：
```json
// qa_log_id 不存在
HTTP 500: {"detail": "(1452, \"Cannot add or update a child row: a foreign key constraint fails\")"}
```

---

### 3.6 示例问题

```
GET /api/questions
```

**说明**：返回 5 个示例问题，前端首页展示引导用户点击。

**请求参数**：无

**响应示例**：
```json
[
  {"id": 1, "text": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
  {"id": 2, "text": "公司的主要风险因素有哪些？"},
  {"id": 3, "text": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
  {"id": 4, "text": "公司本次发行的股票数量和每股面值是多少？"},
  {"id": 5, "text": "报告期内公司的净利润分别是多少？"}
]
```

---

## 四、curl 快速测试集

### 4.1 健康检查

```bash
curl http://127.0.0.1:8001/api/health
```

### 4.2 系统统计

```bash
curl http://127.0.0.1:8001/api/stats
```

### 4.3 同步问答（RAG）

```bash
curl -X POST http://127.0.0.1:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "top_k": 5,
    "use_rag": true
  }'
```

### 4.4 同步问答（纯 LLM）

```bash
curl -X POST http://127.0.0.1:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是向量数据库？", "use_rag": false}'
```

### 4.5 提交反馈

```bash
curl -X POST http://127.0.0.1:8001/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"qa_log_id": 42, "rating": 5, "is_correct": 1, "comment": "准确"}'
```

### 4.6 SSE 流式（-N 禁止缓冲）

```bash
curl -N -X POST http://127.0.0.1:8001/api/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"question": "军用领域收入是多少？", "top_k": 3, "use_rag": true}'
```

### 4.7 示例问题

```bash
curl http://127.0.0.1:8001/api/questions
```

---

## 五、Postman 集合

OpenAPI 自动生成的 Swagger UI：`http://127.0.0.1:8001/docs`

可在 Swagger UI 中直接：
1. 测试每个接口
2. 查看完整 Schema
3. 一键导出 Postman Collection

---

## 六、字段枚举值

| 字段 | 值 | 说明 |
|------|-----|------|
| `AskResponse.mode` | `"rag"` | RAG 检索增强模式 |
| `AskResponse.mode` | `"llm"` | 纯 LLM 问答模式 |
| `FeedbackRequest.rating` | 1-5 | 1 很差 / 5 很好 |
| `FeedbackRequest.is_correct` | 0 / 1 | 事实是否正确 |
| `ReferenceItem.score` | 0-10 浮点 | 归一化分数（bge-reranker 0-10 区间） |

---

## 七、性能参考

基于 1303 chunks（招股说明书1.pdf, 548 页）的实测：

| 接口 | 典型延迟 | P95 延迟 | 瓶颈 |
|------|---------|---------|------|
| GET /api/health | < 10ms | < 20ms | - |
| GET /api/stats | < 50ms | < 100ms | MySQL COUNT |
| POST /api/ask (RAG) | 2,000-4,000ms | 8,000ms | LLM 生成 + reranker |
| POST /api/ask (LLM) | 1,000-1,500ms | 2,500ms | LLM 生成 |
| POST /api/ask_stream | 同 /ask（逐字推送减少感知延迟） | - | LLM 生成 |
| POST /api/feedback | < 50ms | < 100ms | MySQL INSERT |

**RAG 延迟分解**（典型值）：

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 查询改写 | ~10ms | 同义词替换 + 意图识别 |
| 向量召回 (Milvus) | ~80ms | top-50, ef=64 |
| BM25 召回 | ~30ms | jieba + BM25Okapi |
| RRF 融合 + 去重 | ~5ms | chunk_id 作主键 |
| bge-reranker 重排序 | ~180ms | top-30 → top-5, FP16 |
| LLM 生成 | ~1,700ms | deepseek-v4-flash |
| **合计** | **~2,000ms** | - |

**diskcache 缓存命中**（重复查询）：`/api/ask` 可降至 **~1,200ms**（跳过检索链路）

---

## 八、部署端口说明

| 环境 | 端口 | 说明 |
|------|------|------|
| 默认开发 | **8001** | 8000 常被其他项目占用 |
| 覆盖方式 | `export APP_PORT=8000` | 启动前设置环境变量 |
| Nginx 反代 | 80 → /api/ | 生产环境 |

---

*本文档为工单交付物 | 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统 | 接口实现见 src/api.py | Schemas 见 src/schemas.py*
