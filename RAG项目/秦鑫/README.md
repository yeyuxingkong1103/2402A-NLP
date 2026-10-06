# LAW-RAG

一个面向用户使用的本地法律 RAG 服务。后端使用 FastAPI，业务和长期记忆保存在 MySQL，近期缓存和任务队列使用 Redis，公共法律库与用户私有材料向量保存在 Milvus。

## 快速启动

1. 确保本地 MySQL、Redis、Milvus 服务可以连接。
2. 在 `.env` 中配置模型 API 和数据库连接。
3. 本地运行：`L:\anaconda\python.exe run.py`
4. 如需调整短期记忆压缩预算，可在 `.env` 中配置 `SHORT_MEMORY_COMPRESS_TOKEN_LIMIT`，默认 8000。
5. 浏览器打开 `http://你的局域网IP:7294`，接口文档位于 `http://你的局域网IP:7294/docs`。

也可以使用 Docker：

```powershell
docker compose up --build
```

如果你想把向量库单独放到另一组容器或另一台机器上，可以拆开启动：

```powershell
docker compose -f docker-compose.vector.yml up -d
docker compose -f docker-compose.app.yml up --build
```

这时把应用侧的 `MILVUS_URI` 指到外部 Milvus 即可，例如 `http://host.docker.internal:19530`。

## 已实现

- 用户注册、登录、Cookie 会话。
- 八个公共法律集合检索、RRF 融合和重排。
- 本地文件保存到 `data/uploads`，MySQL 保存文件路径，私有 Milvus 保存向量。
- PDF、Word、Excel、图片等材料抽取；图片 OCR 使用 Tesseract OCR，中文识别需要 `chi_sim` 语言包。
- DeepSeek 回答、SiliconFlow Embedding/Reranker、可选 Tavily 联网检索。
- SSE、引用来源、完整对话窗口记忆。
- 健康检查、诊断接口和基础测试。
- 项目主要分为四个业务目录：`data_pipeline/` 处理资料，`rag/` 负责法律问答，`memory/` 负责记忆，`workspace/` 管理用户材料；`models/`、`storage/` 是支撑代码。

## 部署说明

- `docker-compose.yml` 仍然保留为全量启动，适合本地一条命令拉起整套服务。
- `docker-compose.app.yml` 只包含应用、MySQL 和 Redis，便于把应用单独部署。
- `docker-compose.vector.yml` 只包含 Milvus、etcd 和 MinIO，便于把向量库单独部署。
- 前端“偏好设置”里新增了服务地址输入框，适合 exe/WebView 客户端指向不同后端。
- 如果前端和后端不是同源，记得在后端配置 `CORS_ALLOWED_ORIGINS` 或 `CORS_ALLOWED_ORIGIN_REGEX`。
- `desktop-client/` 是一个可打包成 exe 的 Electron 桌面壳，默认优先使用本机局域网地址打开后端。
- 桌面壳的运行和打包命令写在 [`desktop-client/README.md`](L:/law-rag/desktop-client/README.md)。

## 运行说明

日志写入 `data/processed/system_logs`，不会在项目根目录额外生成 `logs/`。OCR 语言数据默认放在 `data/processed/ocr/tessdata`，如果本机 Tesseract 不在 PATH 中，可以在 `.env` 中配置 `TESSERACT_CMD` 和 `TESSDATA_PREFIX`。

## 评估系统

评估数据放在 `evaluation/datasets`，报告输出到 `evaluation/reports`。评估系统覆盖检索命中、排序质量、答案关键词覆盖、禁用词检查、引用来源一致性和延迟统计。

最小数据格式：

```json
[
  {
    "id": "case-001",
    "question": "借款到期不还怎么办？",
    "expected": ["civil_code_articles:675"],
    "returned": ["civil_code_articles:675", "civil_questions:q1"],
    "answer": {
      "answer": "可以先固定借款合同、转账记录和催告记录。",
      "citations": [{"source_id": "civil_code_articles:675"}]
    },
    "expected_answer_keywords": ["借款合同", "转账记录"],
    "expected_citations": ["civil_code_articles:675"],
    "forbidden_answer_terms": ["保证胜诉"],
    "latency_ms": {"total_ms": 1200, "retrieval_ms": 180}
  }
]
```

已生成两个基础数据集：

- `evaluation/datasets/public_law_golden.json`：104 条公共法律库黄金问题，用于真实调用本地 RAG 服务回归。
- `evaluation/datasets/offline_smoke.json`：8 条离线 smoke 数据，用于验证评估工具自身是否正常。

离线评估：

```powershell
L:\anaconda\python.exe tools\evaluate.py evaluation\datasets\offline_smoke.json --k 1,3,5,10 --output evaluation\reports\offline-smoke-report.json
```

调用本地服务评估：

```powershell
L:\anaconda\python.exe tools\evaluate.py evaluation\datasets\public_law_golden.json --ask-url http://127.0.0.1:7294/api/v1/legal/ask --fail-under retrieval.recall@5=0.8 --max-latency-ms 8000
```
