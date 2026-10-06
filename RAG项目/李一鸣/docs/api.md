# 接口文档

默认地址：`http://localhost:8000`

Swagger：`GET /docs`

## GET /api/v1/health

检查应用、记忆、向量库、模型配置。

## GET /api/v1/roles

返回角色列表。响应字段包括 `id`、`name`、`category`、`description`、`system_prompt`、`knowledge_scope`。

## POST /api/v1/roles

请求：

```json
{
  "name": "科研导师",
  "category": "scientist",
  "description": "帮助用户梳理科学问题",
  "personality": "严谨、耐心、鼓励质疑",
  "expertise": "科学方法、论文阅读",
  "speaking_style": "先定义概念，再给分析路径",
  "safety_policy": "不替代专业实验或医疗建议",
  "knowledge_scope": "global"
}
```

## POST /api/v1/documents/upload

`multipart/form-data`：

- `file`：PDF/TXT/MD/CSV/JSON 文件。
- `source`：来源标签，默认 `upload`。
- `role_id`：可选；用于设置该文档的知识范围。

响应示例：

```json
{
  "id": "document-id",
  "filename": "guide.pdf",
  "source": "国家卫健委",
  "status": "ready",
  "chunk_count": 12,
  "error_message": ""
}
```

## GET /api/v1/documents

查看文档入库状态、分块数量和错误信息。

## POST /api/v1/search

请求：

```json
{
  "query": "高血压家庭测量",
  "top_k": 6,
  "role_id": null
}
```

响应包括每个 chunk 的内容、来源、综合分数、dense_score、lexical_score 和检索方式。

## POST /api/v1/chat

非流式请求：

```json
{
  "user_id": "u-001",
  "role_id": "角色 ID",
  "conversation_id": "c-001",
  "message": "我最近测量血压总是偏高，应该先记录哪些信息？",
  "stream": false,
  "top_k": 6
}
```

响应：

```json
{
  "answer": "...",
  "role_id": "角色 ID",
  "conversation_id": "c-001",
  "citations": [
    {
      "chunk_id": "chunk-id",
      "document_id": "document-id",
      "filename": "guide.pdf",
      "content": "...",
      "score": 0.41,
      "retrieval_method": "hybrid"
    }
  ],
  "query": "...",
  "trace_id": "trace-id"
}
```

流式请求只需将 `stream` 设为 `true`，返回 `text/event-stream`。事件格式：

```text
data: {"type":"meta","trace_id":"...","citations":[]}

data: {"type":"token","content":"基于"}

data: {"type":"done","answer":"完整回答","timings":{"total_ms":123}}
```

## POST /api/v1/evaluate

请求：

```json
{
  "samples": [
    {
      "question": "高血压管理包括什么？",
      "answer": "包括生活方式管理和规范测量。",
      "contexts": ["高血压管理包括生活方式干预、规范测量。"],
      "ground_truth": "生活方式管理和规范测量。"
    }
  ]
}
```

默认 `RAGAS_ENABLED=false` 时，接口返回离线本地代理指标。配置非 Mock 的 OpenAI 兼容模型、`LLM_API_KEY`，并设置 `RAGAS_ENABLED=true` 后，接口执行真实 RAGAS `Faithfulness` 和 `ContextRelevance` 评测；外部模型不可用时会记录异常并自动回退。
