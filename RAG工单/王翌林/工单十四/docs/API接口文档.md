# 工单十四：API 接口文档

## 基础信息

- Base URL: `http://localhost:8014`
- 工单：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失

## 接口列表

### 1. 健康检查

**GET** `/health`

返回服务状态和 Milvus 统计信息。

**响应示例：**
```json
{
  "status": "ok",
  "work_order": "工单十四",
  "milvus": {
    "collection": "rag_v14_chunks",
    "exists": true,
    "mode": "lite",
    "num_entities": 12
  }
}
```

### 2. 问答接口

**POST** `/api/ask`

**请求体：**
```json
{
  "query": "该静电除尘器的发明人是？"
}
```

**响应示例（文本问题）：**
```json
{
  "answer": "根据专利文本，该静电除尘器的发明人是 P·吉特勒。",
  "mode": "text_rag",
  "latency_ms": 1234.5,
  "route": {
    "query_type": "text",
    "page": null,
    "needs_vlm": false
  },
  "references": [
    {"page": 1, "chunk_id": "CN100342976C_p1_c0", "score": 0.85, "preview": "...发明人 P·吉特勒..."}
  ],
  "image_path": null,
  "image_desc": null,
  "token_usage": {"prompt_tokens": 450, "completion_tokens": 30, "total_tokens": 480}
}
```

**响应示例（图像问题）：**
```json
{
  "answer": "部件4位于部件5的左侧。",
  "mode": "image_rag",
  "latency_ms": 2345.6,
  "route": {
    "query_type": "image",
    "page": 7,
    "needs_vlm": true
  },
  "references": [],
  "image_path": "./data/pages/rendered/page_07.png",
  "image_desc": "根据图纸观察，部件4位于部件5的左侧...",
  "token_usage": {"prompt_tokens": 600, "completion_tokens": 25, "total_tokens": 625}
}
```

### 3. 获取测试问题

**GET** `/api/questions`

返回工单指定的 6 个测试问题及选项。

**响应示例：**
```json
{
  "questions": [
    {
      "id": 1,
      "question": "根据文本信息，该静电除尘器的发明人是？",
      "type": "text",
      "options": ["A. P·吉特勒", "B. 未知", "C. 静电除尘器研究小组", "D. 该专利未提及发明人"]
    }
  ]
}
```

## 错误码

| HTTP 状态码 | 说明 |
|------------|------|
| 200 | 成功 |
| 422 | 请求参数错误 |
| 500 | 服务器内部错误 |
