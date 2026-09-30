# 部署指南

本项目当前定位为内部测试和生产化收口版本。对外部署前必须显式配置持久化存储、集中式请求控制、模型路径和密钥。

## 本地一键启动与停止

Windows PowerShell 执行：

```powershell
.\scripts\windows\start-dev.ps1
.\scripts\windows\stop-dev.ps1
```

Linux/macOS 或 Git Bash 执行：

```bash
./scripts/linux/start.sh
./scripts/linux/shutdown.sh
```

启动脚本会自动启动 MySQL、Redis、Milvus、etcd 和 MinIO，执行数据库迁移，并在后台启动 API 与 Celery worker。停止脚本会停止应用和 Docker 依赖，但保留数据卷。只停止应用、保留依赖服务时，Windows 使用 `-KeepDependencies`，Linux/macOS 使用 `--keep-dependencies`。

脚本运行日志和 PID 文件位于 `runtime/logs` 与 `runtime/pids`，这些运行时文件不会进入 Git。

## 本地依赖

脚本内部执行以下 Docker Compose 服务启动：

```bash
docker compose up -d mysql redis etcd minio milvus
```

默认端口：

- MySQL：`3306`
- Redis：`6380` 映射到容器内 `6379`
- Milvus：`19530`

`docker-compose.yml` 已为这些依赖声明本地卷，避免容器重建后丢失开发数据。Compose 不再内置 MySQL root 或 MinIO 管理员默认密码，启动前必须在未纳入 Git 的 `.env` 中提供 `MYSQL_ROOT_PASSWORD`、`MINIO_ROOT_USER` 和 `MINIO_ROOT_PASSWORD`。依赖端口默认只绑定 `127.0.0.1`；需要由受控网关访问时，应通过防火墙和显式 `*_BIND_ADDRESS` 配置调整，而不是直接暴露到所有网卡。正式环境应把数据卷迁移到受控存储。

## 后端环境变量

生产化联调建议至少设置：

```bash
ENVIRONMENT=production
DATABASE_URL=mysql+pymysql://root:root@127.0.0.1:3306/legal_rag
AUTH_STORE_BACKEND=sql
OTP_STORE_BACKEND=redis
REQUEST_CONTROL_BACKEND=redis
REDIS_URL=redis://127.0.0.1:6380/0
CELERY_BROKER_URL=redis://127.0.0.1:6380/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:6380/0
MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION=legal_material_chunks
RAG_FRAMEWORK=langchain
APP_MASTER_KEY=<至少 32 字节随机密钥>
APP_HMAC_KEY=<至少 32 字节随机密钥>
DEEPSEEK_API_KEY=<生产密钥>
```

本地模型路径默认指向仓库内 `models/bge-m3` 和 `models/bge-reranker-large`。如果部署路径不同，需要通过配置或环境注入保持一致。

`RAG_FRAMEWORK` 默认值为 `langchain`，使用 LangChain Retriever/LCEL 和 `ChatPromptTemplate` 编排，同时继续使用项目原有的 BGE-M3、Milvus、BGE Reranker、DeepSeekClient 和法律安全治理。`legacy` 保留为显式回退模式；设置为 `shadow` 时旧链路负责返回，新 LangChain 链只执行脱敏结果对比，不调用 DeepSeek。`langchain` 适配层不可用时会返回安全的依据不足结果，不会静默生成无依据回答。

当 `ENVIRONMENT=production` 时，应用启动会拒绝内存认证、内存 OTP、内存请求控制、缺失持久化依赖、缺失密钥、同源静态前端和空 `CORS_ORIGINS`。生产部署必须显式设置 `AUTH_STORE_BACKEND=sql`、`OTP_STORE_BACKEND=redis`、`REQUEST_CONTROL_BACKEND=redis`、`SERVE_FRONTEND=false` 及受控的 `CORS_ORIGINS`。

## 数据库迁移

启动 API 前执行迁移：

```bash
alembic upgrade head
```

回滚或清理生产数据前必须先备份并确认导出作业、记忆和聊天记录的保留策略。

## 后端启动

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8010
```

多 worker 或多实例部署时必须使用：

- `AUTH_STORE_BACKEND=sql`
- `OTP_STORE_BACKEND=redis`
- `REQUEST_CONTROL_BACKEND=redis`

否则登录态、验证码、限流和并发控制只在单进程内有效。

## Celery Worker

```bash
celery -A backend.app.workers.celery_app.celery_app worker --loglevel=info --pool=solo
```

Windows 本地联调可使用 `solo` pool；Linux 生产环境可根据任务类型选择更合适的 pool。

## 导出安全

用户数据导出接口只在下载响应中返回加密 ZIP 密文。ZIP 临时口令必须通过独立接口二次验证后领取，且服务端只返回一次。生产环境应把过期导出清理加入计划任务：

```bash
python -m backend.scripts.cleanup_exports
```

该脚本会删除已过期或已撤销的导出作业。导出密文应排除在常规备份之外，或使用单独的短保留周期备份策略。

## 验证建议

依赖启动后建议依次执行：

```bash
python -m pytest backend/tests/unit/test_request_control.py backend/tests/api/test_user_data_api.py -q
python -m pytest backend/tests -q
npm run test:e2e:chromium
```

真实模型、Milvus 和 DeepSeek 联调仍应在具备 CUDA、本地模型和有效密钥的环境中单独执行。

## 真实后端浏览器 E2E

真实浏览器联调前先启动依赖和后端：

```bash
docker compose up -d mysql redis etcd minio milvus
set ENVIRONMENT=development
set DEV_AUTH_BYPASS=true
set SERVE_FRONTEND=true
set AUTH_STORE_BACKEND=sql
set OTP_STORE_BACKEND=redis
set REQUEST_CONTROL_BACKEND=redis
set DATABASE_URL=mysql+pymysql://root:root@127.0.0.1:3306/myrag
set REDIS_URL=redis://127.0.0.1:6380/0
set MILVUS_URI=http://localhost:19530
set MILVUS_COLLECTION=legal_material_chunks
set APP_MASTER_KEY=<至少 32 字节本地联调密钥>
set APP_HMAC_KEY=<至少 32 字节本地联调密钥>
uvicorn backend.app.main:app --host 127.0.0.1 --port 8010
```

后端启动后运行：

```bash
npm run test:e2e:real
```

该脚本只执行 `frontend/tests/e2e/chat-real.spec.js`，通过同源 `http://127.0.0.1:8010/` 访问真实 FastAPI 托管的前端页面，并使用本地开发身份头完成真实 API 调用。当前真实用例覆盖高风险家暴咨询的紧急指引链路，不依赖 DeepSeek 输出。

真实知识库 RAG 回归需要额外启动 Milvus、MySQL、Redis、本地 BGE 模型，配置 `APP_MASTER_KEY`、`APP_HMAC_KEY` 和 DeepSeek 密钥，并确保知识库已完成索引。依赖就绪后运行：

```bash
npm run test:e2e:real:rag
```

LangChain 已完成真实 shadow 多场景验证：离婚/抚养权、抚养/探望、夫妻财产/债务三类可回答场景均出现 `comparison_status=match`；无命中和高风险请求分别按事实补充和紧急安全分流，不进入 RAG 对比。若需要回退，可显式设置 `RAG_FRAMEWORK=legacy`；若需要继续观测，可设置 `RAG_FRAMEWORK=shadow`。

可选环境变量：

- `REAL_RAG_QUERY`：覆盖默认婚姻家庭咨询问题。
- `REAL_RAG_EXPECT_TEXT`：覆盖回答中必须出现的文本，默认 `抚养`。
- `REAL_RAG_EXPECT_CITATION`：指定引用区域必须出现的文本；为空时只校验引用可见。

## 备份与恢复

上线前必须完成一次可审计的全量恢复演练：

- 备份导出应启用一致性选项，例如 `mysqldump --single-transaction --routines --events --hex-blob --skip-extended-insert`。
- 恢复校验必须在独立临时库或独立实例执行，不得覆盖正在使用的业务库；推荐使用 MySQL Shell 的 SQL 模式导入：`mysqlsh --sql --uri root@127.0.0.1:3306 --database <临时库> < backup.sql`。
- Windows 原生 `mysql` CLI 在导入包含法律正文特殊反斜杠转义的 dump 时可能失败；正式灾备验收建议使用 MySQL Shell、Linux/容器化 MySQL 客户端或经过验证的备份工具链。
- 恢复完成后需核对 `alembic_version`、`knowledge_materials`、`documents`、`document_chunks`、`conversations` 和 `messages` 等核心表数量。
- 临时恢复库和临时备份文件必须在演练结束后清理。

## 监控与生产验收

应用提供轻量存活、就绪和内部指标端点：

```bash
curl http://127.0.0.1:8010/health/live
curl -i http://127.0.0.1:8010/health/ready
```

生产环境如需暴露 Prometheus 指标，设置 `METRICS_ENABLED=true`，并只通过内部网关或采集网络开放 `/metrics`。指标只包含 HTTP 方法、路径、状态码、耗时和依赖健康状态，不包含用户文本、密钥或连接串。该实现是单进程指标快照，多实例环境应由每个实例分别采集并在 Prometheus 侧聚合。

基础设施健康基线压测：

```bash
python -m backend.scripts.health_load_test --requests 1000 --concurrency 20
```

该脚本仅用于健康端点和网络/编排基线，不代表真实聊天、模型推理或 RAG 性能。真实聊天 API、多实例 Redis/Celery/Milvus/MySQL 和下游模型仍需使用独立压测方案验收。

备份与独立恢复校验：

```bash
python -m backend.scripts.backup_database --output backups/legal-rag.sql
python -m backend.scripts.verify_database_backup \
  --database-url mysql+pymysql://<user>:<password>@<host>:3306/legal_rag_restore \
  --backup backups/legal-rag.sql
```

恢复命令要求目标库名称明确为独立临时库，脚本不会接受 `legal_rag`、`myrag`、`production` 或 `mysql` 等业务库名称。详细人工验收项见 `docs/10-production-checklist.md`。


当前测试环境使用 `fastapi.testclient.TestClient`，项目已在 pytest 配置中精确过滤 Starlette 对 `httpx` 的已知弃用提示。后续升级到 `httpx2` 前，应先确认 FastAPI、Starlette 和 httpx2 的兼容版本组合，再移除该过滤规则。
