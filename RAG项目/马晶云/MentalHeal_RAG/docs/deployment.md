# 本地与 Ubuntu 部署说明

## 1. 准备环境

```bash
cp .env.example .env
source .venv/bin/activate
./scripts/install/check_env.sh
```

请在 `.env` 中填写 `DEEPSEEK_API_KEY`、`MINERU_API_KEY` 和 MySQL/Redis/Milvus 连接配置。不要把真实密钥写入代码或提交。

## 2. 初始化数据库

先备份现有数据库，再执行完整 schema 或按迁移顺序执行：

```bash
mysql -u root -p < backend/app/db/schema.sql
mysql -u root -p mentalheal_rag_dev < backend/app/db/migrations/001_auth_roles.sql
mysql -u root -p mentalheal_rag_dev < backend/app/db/migrations/002_knowledge_evaluation.sql
```

`002_knowledge_evaluation.sql` 会创建 `knowledge_bases` 和 `rag_evaluation_runs`，不会删除已有文档、会话或消息。

## 3. 启动服务

分别执行：

```bash
./scripts/start/backend.sh
./scripts/start/frontend.sh
```

也可以后台一键启动：

```bash
./scripts/start/all.sh
```

后端默认 `http://127.0.0.1:8000`，前端默认 Vite 地址 `http://localhost:5173`。也可以直接访问 `http://127.0.0.1:8000/docs` 查看 OpenAPI。

## 4. 健康检查与停止

```bash
./scripts/ops/health_check.sh
./scripts/ops/stop_local.sh
```

健康检查同时验证 MySQL、Redis、Milvus 和 `knowledge_chunks` collection；任一依赖不可用时返回 HTTP 503，并只返回错误类型，不返回密码或 API Key。

## 5. 文档和评测管理

管理员登录后可调用或在前端管理中心使用：

- `POST /api/v1/documents/upload`
- `GET /api/v1/documents`
- `POST /api/v1/documents/{document_id}/parse`
- `POST /api/v1/documents/{document_id}/embed`
- `GET /api/v1/documents/jobs`
- `POST /api/v1/evaluations/ragas`
- `GET /api/v1/evaluations/ragas`
- `GET /api/v1/evaluations/ragas/{run_id}`

文档上传限制为 PDF 和 `UPLOAD_MAX_SIZE_MB`；同名文件内容相同时复用记录，内容不同时自动生成哈希后缀，不静默覆盖。解析和向量化均记录 `queued/running/success/failed` 状态及失败原因。

RAGAS 任务使用 `data/evaluation/ragas_dataset.json`，真实执行 Milvus 检索、DeepSeek 生成和五项指标计算，结果写入 `rag_evaluation_runs`。

## 6. Ubuntu 生产建议

- 使用 Python 3.10+ 虚拟环境和 Node.js LTS。
- 用 systemd、Supervisor 或 Docker 运行 Uvicorn，生产环境关闭 `--reload`。
- 用 Nginx 反向代理 `/api/` 和 `/health` 到 Uvicorn，将前端静态文件指向 `frontend/dist`。
- 为 MySQL、Redis、Milvus 和 `data/` 设置备份与磁盘监控。
- 生产日志集中保存，并对 Authorization、API Key、密码和用户原始危机文本脱敏。
