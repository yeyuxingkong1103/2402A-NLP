# 招股说明书 RAG 问答系统

## 快速启动

1. 安装并启动 Docker Desktop。
2. 复制配置：`copy .env.example .env`，填写 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`MINERU_API_KEY`；远程 MinerU 使用 `MINERU_API_URL`，百炼视觉模型使用 `DASHSCOPE_BASE_URL` 和 `DASHSCOPE_VL_MODEL`。
3. 确认模型目录存在：
   - `D:\桌面\项目\legal-rag\model\bge-m3`
   - `D:\桌面\项目\legal-rag\model\bge-reranker-large`
4. 将 `招股说明书1-无水印.pdf` 放入 `data/uploads/`，或打开页面后使用上传按钮。
5. 执行 `docker compose up -d --build`。
6. 打开 `http://localhost:5173`。

默认管理员口令为 `.env` 中的 `ADMIN_TOKEN`，仅上传和删除文档需要。

## 架构

- React + Vite + Ant Design：问答、引用、上传和反馈。
- FastAPI：文档、问答、SSE、反馈和健康检查 API。
- Celery + Redis：异步解析与向量入库。
- 远程 MinerU：主解析器；PyMuPDF：本地兜底。
- 本地 `bge-m3`：Embedding；本地 `bge-reranker-large`：重排。
- Milvus Standalone + etcd + MinIO：向量库及持久化。
- PostgreSQL：文档、问答和反馈记录。
- DeepSeek：Query Plan 和最终回答；阿里云百炼 OpenAI 兼容 API 上的 `qwen-vl-max`：按需处理复杂图片/表格。

百炼配置：

```env
DASHSCOPE_API_KEY=你的百炼API Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_VL_MODEL=qwen-vl-max
```

默认使用百炼公共兼容端点；如果使用指定业务空间或其他地域，将 `DASHSCOPE_BASE_URL` 替换为百炼控制台提供的对应 `compatible-mode/v1` 地址。

## API

- `GET /health`、`GET /ready`
- `GET /api/documents`
- `POST /api/documents/upload`（`X-Admin-Token`）
- `GET /api/documents/{document_id}/task`
- `POST /api/ask`
- `POST /api/ask/stream`
- `POST /api/feedback/{request_id}`

## 评估

在服务启动并完成文档入库后运行：

```bash
python scripts/evaluate.py
```

结果写入 `data/exports/evaluation.json`。当前脚本负责批量记录 10 个问题的答案、引用和延迟；Ground Truth、Ragas 评分和纯 LLM 基线应在确认标准答案后补充。
