# Legal RAG Assistant

法律客服助手：原生 HTML、CSS、JavaScript 前端 + FastAPI 后端，面向婚姻家庭法律咨询 MVP、知识库治理、RAG 检索和个人数据导出场景。

## 当前状态

当前版本已完成 MVP 验收和第一轮生产化收口，可用于本地开发和内部测试；不应直接作为对外生产法律服务上线。

已验证能力：

- 后端单元、API、集成回归：`183 passed, 2 skipped`。
- Playwright Chromium mock E2E：`6 passed, 2 skipped`。
- 真实后端浏览器冒烟 E2E：`1 passed`。
- 真实知识库 RAG 浏览器 E2E：`1 passed`。
- Redis 请求控制、SQL 持久化 repository、Redis OTP、Milvus、Celery、本地 BGE 模型和 DeepSeek 客户端已具备联调路径。
- 用户数据导出已改为加密密文下载 + 独立二次验证一次性口令领取，并提供过期/撤销导出清理脚本。

详见：

- `evaluation/ACCEPTANCE.md`
- `docs/09-test-report.md`
- `docs/08-deployment-guide.md`

## 快速启动

安装 Python 和 Docker Desktop 后，在 Windows PowerShell 中执行：

```powershell
.\scripts\windows\start-dev.ps1
```

Linux/macOS 或 Git Bash 中执行：

```bash
./scripts/linux/start.sh
```

启动脚本会自动启动 MySQL、Redis、etcd、MinIO 和 Milvus，执行 `alembic upgrade head`，然后在后台启动 API 和 Celery worker。API 地址：

```text
http://127.0.0.1:8010/
```

停止全部服务：

```powershell
.\scripts\windows\stop-dev.ps1
```

```bash
./scripts/linux/shutdown.sh
```

停止脚本会结束 API、Celery 和 Docker 依赖，但保留 Docker 数据卷。只停止应用、保留依赖服务时，使用 `-KeepDependencies` 或 `--keep-dependencies`。

默认开发配置会同源托管 `frontend`，并允许本地开发身份头；生产环境必须关闭开发免登录并配置真实认证、密钥和持久化存储。

## 本地依赖

启动脚本也支持跳过依赖和迁移，适合依赖已经运行的场景：

```powershell
.\scripts\windows\start-dev.ps1 -SkipDependencies -SkipMigrations
```

```bash
./scripts/linux/start.sh --skip-dependencies --skip-migrations
```

生产化联调建议至少配置：

```bash
ENVIRONMENT=production
AUTH_STORE_BACKEND=sql
OTP_STORE_BACKEND=redis
REQUEST_CONTROL_BACKEND=redis
SERVE_FRONTEND=false
CORS_ORIGINS=https://legal.example.com
DATABASE_URL=mysql+pymysql://<user>:<password>@<host>:3306/legal_rag
REDIS_URL=redis://<host>:6379/0
MILVUS_URI=http://<host>:19530
RAG_FRAMEWORK=langchain
APP_MASTER_KEY=<至少 32 字节随机密钥>
APP_HMAC_KEY=<至少 32 字节随机密钥>
DEEPSEEK_API_KEY=<生产密钥>
```

生产 Compose 依赖还必须在未纳入 Git 的 `.env` 中提供 `MYSQL_ROOT_PASSWORD`、`MINIO_ROOT_USER` 和 `MINIO_ROOT_PASSWORD`；Compose 端口默认只绑定 `127.0.0.1`。应用在 `ENVIRONMENT=production` 下会拒绝内存存储、缺失密钥、同源静态前端和空 CORS 配置。

## 测试

后端回归：

```bash
python -m pytest backend/tests -q
```

后端编译检查：

```bash
python -m compileall -q backend
```

前端浏览器 E2E：

```bash
npm run test:e2e:chromium
```

真实后端浏览器冒烟：

```bash
npm run test:e2e:real
```

真实知识库 RAG 浏览器回归：

```bash
npm run test:e2e:real:rag
```

运行真实 E2E 前需先启动 `http://127.0.0.1:8010` 后端。冒烟用例覆盖真实前端页面、真实 FastAPI API 和高风险家暴咨询紧急指引链路；RAG 用例还需要 Milvus、MySQL、Redis、本地模型、有效密钥和已索引知识库。

## 导出清理

清理过期或已撤销的用户数据导出作业：

```bash
python -m backend.scripts.cleanup_exports
```

建议生产环境把该命令接入计划任务，并确保导出密文排除在常规长期备份之外，或使用单独短保留周期策略。

## 生产边界

上线前仍需完成：

- 正式安全审查和法律服务人工验收。
- 生产密钥管理、默认密码替换和灾备恢复演练。
- Redis、Celery、Milvus、MySQL 多实例压测和监控告警。
- 真实知识库 RAG 的多场景浏览器回归。
- FastAPI、Starlette 与 `httpx2` 兼容版本评估后再迁移测试依赖。
