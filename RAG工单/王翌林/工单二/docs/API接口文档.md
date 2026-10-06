# API 接口文档

> **工单编号**：人工智能NLP-RAG-基于PDF文档的问答系统优化
> **版本**：v2.0.0（工单二优化版）
> **Base URL**：`http://127.0.0.1:8000`
> **协议**：HTTP/1.1, JSON（SSE 端点为 `text/event-stream`）
> **OpenAPI / Swagger**：`http://127.0.0.1:8000/docs`（FastAPI 自动生成）
> **接口实现**：[src/api.py](file:///home/dabaie/code/工单/工单二/src/api.py)
> **Pydantic 模型**：[src/schemas.py](file:///home/dabaie/code/工单/工单二/src/schemas.py)

---

## 一、接口总览

| # | 方法 | 路径 | 说明 | 认证 |
|---|------|------|------|------|
| 1 | `GET` | `/api/health` | 健康检查（MySQL / Milvus / 模型状态） | 无 |
| 2 | `GET` | `/api/stats` | 系统统计（文档数 / Chunk 数 / QA 日志 / 反馈数） | 无 |
| 3 | `POST` | `/api/ask` | 同步问答（RAG 优化链路 / RAG 基线链路 / 纯 LLM） | 无 |
| 4 | `POST` | `/api/ask_stream` | SSE 流式问答（逐字推送） | 无 |
| 5 | `POST` | `/api/feedback` | 提交反馈（点赞 / 点踩 / 评论） | 无 |
| 6 | `GET` | `/api/questions` | 示例问题列表（5 条预置问题） | 无 |

---

## 二、通用说明

### 2.1 请求格式

- Content-Type：`application/json`
- 字符编码：`UTF-8`
- 时间单位：毫秒（ms）
- 数字类型：浮点数

### 2.2 响应格式

**成功**：返回 Pydantic 模型定义的 JSON 对象，HTTP 200

**失败**：

```json
{
  "detail": "错误描述"
}
```

### 2.3 错误码

| HTTP Code | 说明 | 触发场景 |
|-----------|------|----------|
| `200 OK` | 成功 | 正常请求 |
| `422 Unprocessable Entity` | 请求体字段校验失败 | 空问题 / 问题超 2000 字 / top_k 越界 / rating 越界 |
| `500 Internal Server Error` | 后端异常 | LLM 超时 / Milvus 异常 / MySQL 写入失败（detail 含 `str(e)`） |
| `503 Service Unavailable` | RAG Engine 未就绪 | 模型加载中或初始化失败 |

### 2.4 限流 / 超时

| 项目 | 值 |
|------|-----|
| question 长度 | 1 ~ 2000 字符 |
| top_k 范围 | 1 ~ 20 |
| rating 范围 | -1 / 0 / 1（工单二：-1=点踩 / 0=仅评论 / 1=点赞） |
| comment 长度 | ≤ 500 字 |
| 建议超时（RAG） | 30s（含 reranker + LLM 生成） |
| 建议超时（SSE） | 60s |
| 建议超时（纯 LLM） | 15s |

---

## 三、接口详情

### 3.1 健康检查

```
GET /api/health
```

**说明**：返回 MySQL / Milvus / Embedding / LLM 运行状态，用于负载均衡探活和系统自检。FastAPI `lifespan` 启动时自动探测并缓存状态。

**请求参数**：无

**响应模型**：`HealthResponse`

**响应示例**：

```json
{
  "status": "ok",
  "mysql": "ok (8.0.46-0ubuntu0.22.04.4)",
  "milvus": "ok (remote)",
  "embedding_model": "BAAI/bge-m3",
  "llm_model": "deepseek-v4-flash",
  "uptime_seconds": 14.3
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 固定 `"ok"`（只要 API 进程存活） |
| mysql | string | `"ok (版本号)"` 或 `"failed: 原因"` |
| milvus | string | `"ok (remote)"` / `"ok (lite)"` 或 `"failed: 原因"` |
| embedding_model | string | 嵌入模型名称（如 `BAAI/bge-m3`） |
| llm_model | string | LLM 模型名称（如 `deepseek-v4-flash`） |
| uptime_seconds | float | API 运行时长（秒） |

---

### 3.2 系统统计

```
GET /api/stats
```

**说明**：返回文档数、Chunk 数、QA 日志数、反馈数、Milvus 向量实体数。

**请求参数**：无

**响应模型**：`StatsResponse`

**响应示例**：

```json
{
  "total_documents": 1,
  "total_chunks": 1565,
  "total_qa_logs": 39,
  "total_feedback": 5,
  "collection_entities": 1565
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| total_documents | int | MySQL documents 表记录数 |
| total_chunks | int | MySQL chunks 表记录数 |
| total_qa_logs | int | MySQL qa_logs 表记录数 |
| total_feedback | int | MySQL feedbacks 表记录数 |
| collection_entities | int \| null | Milvus collection 实体数（远程不可用时为 null） |

---

### 3.3 同步问答（核心）

```
POST /api/ask
Content-Type: application/json
```

**说明**：工单二优化版核心接口，支持三种链路：

1. **优化链路**（默认）：语义缓存 → LLM 查询改写 → 混合召回（向量 + BM25 + 表格专项）→ RRF 融合 → bge-reranker 重排序 → 父子块映射 → DeepSeek 生成
2. **基线链路**（`chain=baseline`）：工单一原版 RAG（BM25 + 向量 + RRF + reranker + LLM），供前端对比展示
3. **纯 LLM**（`use_rag=false`）：跳过检索，直接 DeepSeek 生成

**请求模型**：`AskRequest`

**请求体**：

```json
{
  "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
  "top_k": 8,
  "use_rag": true,
  "doc_id": null,
  "lang": null,
  "chain": null
}
```

| 字段 | 类型 | 必填 | 默认值 | 校验规则 | 说明 |
|------|------|------|--------|----------|------|
| question | string | ✅ | — | 1 ≤ len ≤ 2000 | 用户问题 |
| top_k | int | ❌ | 5 | 1 ≤ top_k ≤ 20 | 最终返回引用数量 |
| use_rag | bool | ❌ | true | — | true=RAG 问答，false=纯 LLM |
| doc_id | string \| null | ❌ | null | — | 限制检索范围（预留字段） |
| lang | string \| null | ❌ | null | `"zh"` / `"en"` / null | 工单二：回答语言；null=自动检测 |
| chain | string \| null | ❌ | null | `"optimized"` / `"baseline"` / null | 工单二：链路选择；null 或 `"optimized"`=优化链路，`"baseline"`=基线链路 |

**响应模型**：`AskResponse`

**响应示例**（优化链路 RAG 模式）：

```json
{
  "mode": "rag",
  "answer": "武汉兴图新科电子股份有限公司的法定代表人是**程家明**。\n\n依据：\n- [资料1]\"一、发行人的基本情况\"中列明：法定代表人：程家明。",
  "qa_log_id": 42,
  "references": [
    {
      "page": 52,
      "chunk_id": "doc1_p52_c0",
      "score": 0.9847,
      "preview": "一、发行人的基本情况 法定代表人：程家明 成立日期：2015年..."
    }
  ],
  "latency_ms": 1766.0,
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
    "vector_latency_ms": 90.0,
    "bm25_latency_ms": 35.0,
    "rerank_latency_ms": 180.0,
    "llm_latency_ms": 1350.0
  },
  "cache_hit": false,
  "rag_answer": "...同 answer...",
  "rag_latency_ms": 1766.0,
  "llm_answer": null,
  "llm_latency_ms": null
}
```

**响应示例**（基线链路 `chain=baseline`）：

```json
{
  "mode": "rag",
  "answer": "法定代表人是程家明。",
  "qa_log_id": 43,
  "references": [...],
  "latency_ms": 13520.0,
  "token_usage": {...},
  "query_understanding": null,
  "breakdown": null,
  "cache_hit": null,
  "rag_answer": "...同 answer...",
  "rag_latency_ms": 13520.0,
  "llm_answer": null,
  "llm_latency_ms": null
}
```

**响应示例**（纯 LLM 模式 `use_rag=false`）：

```json
{
  "mode": "llm",
  "answer": "作为一个 AI，我没有实时数据库...",
  "qa_log_id": 44,
  "references": [],
  "latency_ms": 1200.0,
  "token_usage": {"prompt_tokens": 52, "completion_tokens": 78, "total_tokens": 130},
  "query_understanding": null,
  "breakdown": null,
  "cache_hit": null,
  "rag_answer": null,
  "rag_latency_ms": null,
  "llm_answer": "...同 answer...",
  "llm_latency_ms": 1200.0
}
```

**响应字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| mode | string | `"rag"`（RAG 模式）或 `"llm"`（纯 LLM 模式） |
| answer | string | 最终答案（Markdown 格式） |
| qa_log_id | int \| null | MySQL qa_logs 主键，用于提交反馈 |
| references | ReferenceItem[] | 引用溯源列表 |
| latency_ms | float | 端到端耗时 |
| token_usage | Dict[str, int] | LLM token 用量（prompt/completion/total） |
| query_understanding | Dict \| null | 工单二：查询改写结果（仅优化链路返回） |
| breakdown | Dict[str, float] \| null | 工单二：各阶段耗时细分（仅优化链路返回） |
| **cache_hit** | bool \| null | **工单二新增**：语义缓存是否命中（仅优化链路返回） |
| rag_answer | string \| null | RAG 模式下同 answer |
| rag_latency_ms | float \| null | RAG 模式下同 latency_ms |
| llm_answer | string \| null | 纯 LLM 模式下同 answer |
| llm_latency_ms | float \| null | 纯 LLM 模式下同 latency_ms |

**ReferenceItem**：

| 字段 | 类型 | 说明 |
|------|------|------|
| page | int \| null | 原文 PDF 页码（1-based） |
| chunk_id | string \| null | 分块标识符 `{doc_id}_p{page}_c{idx}` |
| score | float \| null | 融合后归一化分数（越大越相关） |
| preview | string \| null | 原文片段预览（最长 120 字） |

**breakdown 字段**（仅优化链路）：

| 子字段 | 说明 |
|--------|------|
| vector_latency_ms | 向量召回耗时 |
| bm25_latency_ms | BM25 召回耗时 |
| rerank_latency_ms | 重排序耗时 |
| llm_latency_ms | LLM 生成耗时 |

---

### 3.4 SSE 流式问答

```
POST /api/ask_stream
Content-Type: application/json
Accept: text/event-stream
```

**说明**：Server-Sent Events 流式输出，前端可边生成边展示，提升用户感知速度。内部流程与 `/api/ask` 基线链路一致（检索 → LLM 逐字推送）。

**请求体**：同 `POST /api/ask`（`AskRequest`）

**响应格式**：`text/event-stream`，每个事件一条 `data: {...}\n\n`，事件间空行分隔。

**事件类型**：

| type | 携带字段 | 说明 |
|------|----------|------|
| `status` | `msg` | 状态提示（如 `"正在检索..."`） |
| `retrieved` | `count`, `latency_ms`, `references` | 检索完成，含 top-k 引用列表 |
| `token` | `text` | 逐字答案片段 |
| `done` | `latency_ms`, `token_usage` | 答案完成 |
| `error` | `msg` | 发生错误（提前终止） |
| — | `[DONE]` | 流结束标记（data 值为字符串 `[DONE]`） |

**原始流示例**：

```
data: {"type": "status", "msg": "正在检索..."}

data: {"type": "retrieved", "count": 5, "latency_ms": 345.2, "references": [{"page": 52, "chunk_id": "doc1_p52_c0", "score": 8.44, "preview": "一、发行人的基本情况 法定代表人..."}]}

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
curl -N -X POST http://127.0.0.1:8000/api/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"question":"法定代表人是谁？","top_k":3,"use_rag":true}'
```

**前端 JavaScript（fetch + ReadableStream）**：

```javascript
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

**说明**：对某条问答结果提交评分，用于后续质量分析和 RAG 优化。工单二优化版将 rating 语义调整为：+1=点赞 / -1=点踩 / 0=仅评论（与前端 👍/👎 按钮对齐）。

**请求模型**：`FeedbackRequest`

**请求体**：

```json
{
  "qa_log_id": 42,
  "rating": 1,
  "is_correct": 1,
  "comment": "非常准确！"
}
```

| 字段 | 类型 | 必填 | 默认值 | 校验规则 | 说明 |
|------|------|------|--------|----------|------|
| qa_log_id | int | ✅ | — | 外键约束 | 来自 `/api/ask` 返回的 qa_log_id |
| rating | int | ✅ | — | -1 ≤ rating ≤ 1 | 工单二：1=点赞 / -1=点踩 / 0=仅评论 |
| is_correct | int \| null | ❌ | null | 0 / 1 | 答案是否事实正确 |
| comment | string \| null | ❌ | null | ≤ 500 字 | 文字反馈 |

**响应模型**：`FeedbackResponse`

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
| ok | bool | 固定 `true`（成功写入） |

**错误场景**：

| 场景 | HTTP | detail |
|------|------|--------|
| qa_log_id 不存在（外键约束） | 500 | `"Cannot add or update a child row: a foreign key constraint fails"` |
| rating 越界（如 5） | 422 | Pydantic 自动校验 |
| MySQL 不可用 | 500 | `str(e)` |

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
curl http://127.0.0.1:8000/api/health
```

### 4.2 系统统计

```bash
curl http://127.0.0.1:8000/api/stats
```

### 4.3 同步问答（优化链路 RAG）

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "top_k": 8,
    "use_rag": true
  }'
```

### 4.4 同步问答（基线链路，前端对比用）

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "top_k": 5,
    "use_rag": true,
    "chain": "baseline"
  }'
```

### 4.5 同步问答（纯 LLM）

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是向量数据库？", "use_rag": false}'
```

### 4.6 同步问答（指定中文回答）

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the registered capital?", "top_k": 5, "lang": "zh"}'
```

### 4.7 提交反馈（点赞）

```bash
curl -X POST http://127.0.0.1:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"qa_log_id": 42, "rating": 1, "is_correct": 1, "comment": "准确"}'
```

### 4.8 提交反馈（点踩）

```bash
curl -X POST http://127.0.0.1:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"qa_log_id": 42, "rating": -1, "comment": "答案不完整"}'
```

### 4.9 SSE 流式（-N 禁止缓冲）

```bash
curl -N -X POST http://127.0.0.1:8000/api/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"question": "军用领域收入是多少？", "top_k": 3, "use_rag": true}'
```

### 4.10 示例问题

```bash
curl http://127.0.0.1:8000/api/questions
```

---

## 五、Swagger UI

FastAPI 自动生成的 OpenAPI 文档：`http://127.0.0.1:8000/docs`

在 Swagger UI 中可以：
1. 在线测试每个接口
2. 查看完整 Pydantic Schema
3. 一键导出 Postman Collection / OpenAPI JSON

---

## 六、字段枚举值

| 字段 | 可选值 | 说明 |
|------|--------|------|
| `AskRequest.use_rag` | `true` / `false` | true=RAG 检索增强，false=纯 LLM |
| `AskRequest.lang` | `"zh"` / `"en"` / `null` | 回答语言；null=自动检测 |
| `AskRequest.chain` | `"optimized"` / `"baseline"` / `null` | 链路选择；null=优化链路（默认） |
| `AskResponse.mode` | `"rag"` / `"llm"` | 实际使用的问答模式 |
| `AskResponse.cache_hit` | `true` / `false` / `null` | 语义缓存命中（仅优化链路） |
| `FeedbackRequest.rating` | `-1` / `0` / `1` | -1=点踩 / 0=仅评论 / 1=点赞 |
| `FeedbackRequest.is_correct` | `0` / `1` / `null` | 答案事实是否正确 |
| SSE event type | `status` / `retrieved` / `token` / `done` / `error` | SSE 事件类型 |

---

## 七、性能参考

基于 1565 子块（招股说明书 1.pdf, 548 页）的实测数据：

| 接口 | 典型延迟 | P95 延迟 | 瓶颈 |
|------|---------|---------|------|
| GET /api/health | < 10ms | < 20ms | — |
| GET /api/stats | < 50ms | < 100ms | MySQL COUNT |
| POST /api/ask（优化链路 RAG） | 1,700~2,000ms | 3,000ms | LLM 生成 |
| POST /api/ask（优化链路 缓存命中） | ~2ms | ~5ms | 纯缓存读取 |
| POST /api/ask（基线链路 RAG） | 13,500ms | 15,000ms | 无缓存 + 全量检索 |
| POST /api/ask（纯 LLM） | 1,000~1,500ms | 2,500ms | LLM 生成 |
| POST /api/ask_stream | 同 /ask（逐字推送减少感知延迟） | — | LLM 生成 |
| POST /api/feedback | < 50ms | < 100ms | MySQL INSERT |

**优化链路延迟分解**（典型值）：

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 语义缓存检查 | ~5ms | DiskCache + bge-m3 余弦相似度 ≥ 0.92 |
| LLM 查询改写 | ~10ms | 同义词替换 + 意图识别 |
| 向量召回（Milvus HNSW） | ~90ms | top-50, ef=64 |
| BM25 召回 | ~35ms | jieba + BM25Okapi |
| RRF 融合 + 去重 | ~5ms | chunk_id 主键去重 |
| bge-reranker 重排序 | ~180ms | top-30 → top-k, FP16 |
| 父子块映射 | ~10ms | 子块命中 → 父块合并 |
| LLM 生成 | ~1,350ms | deepseek-v4-flash |
| **合计（非缓存）** | **~1,766ms** | — |
| **缓存命中** | **~2ms** | 跳过检索 + 生成全链路 |

---

## 八、部署端口说明

| 环境 | 端口 | 启动脚本 |
|------|------|----------|
| 工单二优化版（默认） | **8000**（FastAPI）+ **8502**（Streamlit） | `bash scripts/start_optimized.sh` |
| 基线版（工单一） | 8001（FastAPI）+ 8501（Streamlit） | `bash scripts/start.sh` |
| 自定义端口 | `APP_PORT=xxxx STREAMLIT_PORT=xxxx` | 启动前 export 环境变量 |
| Swagger 文档 | `http://<host>:<port>/docs` | — |

---

## 九、工单二 v1.0 → v2.0 变更记录

| 变更项 | v1.0（基线） | v2.0（优化版） |
|--------|-------------|---------------|
| Base URL 端口 | 8001 | **8000** |
| AskRequest.lang | 无 | 新增 `"zh"` / `"en"` / null |
| AskRequest.chain | 无 | 新增 `"optimized"` / `"baseline"` / null |
| AskResponse.cache_hit | 无 | 新增 `bool \| null` |
| AskResponse.breakdown | 无 | 新增各阶段耗时细分 |
| AskResponse.query_understanding | 无 | 新增查询改写结果 |
| FeedbackRequest.rating | 1~5（5 星制） | **-1 / 0 / 1**（点赞/点踩/评论） |
| top_k 默认值 | 5 | 5（前端建议 8） |
| 缓存机制 | 无 | 语义缓存（DiskCache + 余弦 ≥ 0.92） |
| 检索 | BM25 + 向量 + RRF | + 表格专项召回 + HyDE |
| 分块 | 固定长度 | + 语义分块 + 父子块 |

---

*本文档为工单交付物 | 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化 | 接口实现见 [src/api.py](file:///home/dabaie/code/工单/工单二/src/api.py) | Schemas 见 [src/schemas.py](file:///home/dabaie/code/工单/工单二/src/schemas.py) | 部署见 [05_部署文档.md](05_部署文档.md)*
