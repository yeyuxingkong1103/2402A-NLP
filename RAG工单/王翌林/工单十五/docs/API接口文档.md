# 工单十五：API 接口文档

## 服务信息
- FastAPI 服务地址：`http://<host>:8015`
- 交互式文档：`http://<host>:8015/docs`
- Streamlit 界面：`http://<host>:8515`

## 接口列表

### 1. 健康检查
`GET /health`

响应：
```json
{
  "status": "ok",
  "milvus": {"collection": "rag_v15_chunks", "exists": true, "mode": "remote"}
}
```

### 2. 根信息
`GET /`

响应：
```json
{"work_order": "工单十五", "task": "优化技术图纸与文本的跨模态检索流程", "status": "ok"}
```

### 3. 问答接口
`POST /ask`

请求体：
```json
{"question": "在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？"}
```

响应：
```json
{
  "mode": "crossmodal_rag",
  "answer": "编号13的部件位于编号12的部件内部。",
  "references": [{"page": 11, "chunk_id": "...", "score": 0.83}],
  "figure": {
    "fig_no": 3, "page": 11,
    "caption": "图3 块状散料分配装置结构示意图",
    "description": "...",
    "image_path": "./data/pages_hd/page_11.png"
  },
  "latency_ms": 1670.5,
  "route": {
    "query_type": "image", "needs_vlm": true,
    "fig_no": 3, "page": 11, "parts": [13, 12],
    "has_visual_ref": true
  },
  "token_usage": {"prompt_tokens": 1200, "completion_tokens": 38, "total_tokens": 1238}
}
```

字段说明：
- `mode`：`text_rag`（纯文本）或 `crossmodal_rag`（图文）
- `route`：查询理解结果，含视觉引用元数据
- `figure`：命中的图纸结构化描述（仅图文问题）
- `references`：检索返回的 Top-K 文本片段
- `latency_ms`：端到端耗时

## 调用示例

```bash
curl -X POST http://localhost:8015/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"根据文件中第11页图3，散料从部件14进入后，下一步会经过哪个部件？"}'
```
