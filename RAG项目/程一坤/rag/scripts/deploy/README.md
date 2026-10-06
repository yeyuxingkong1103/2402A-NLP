# 部署交付与敏感配置保护

## 打包

从项目根目录执行交付包脚本：

```bash
bash scripts/deploy/build_release_package.sh
```

脚本生成 `legalrag-release-YYYYMMDD.tar.gz`，压缩包根层直接包含 `backend/`、`frontend/`、`evaluation/`、`scripts/`、`data/`、`docs/` 等目录，不套额外项目目录层级。脚本会硬性断言包内不含 `.env`、`frontend/.env*`、`.next`、`node_modules`、`__pycache__`、`.pytest_cache`、`*.log`、`*.backup-*`、`*.bak` 或 `.workbuddy/`；命中任一项即退出失败。

上传到 Ubuntu 后，在服务器目标目录解压：

```bash
scp legalrag-release-YYYYMMDD.tar.gz <user>@<server>:/opt/
ssh <user>@<server>
mkdir -p /opt/rag
tar -xzf /opt/legalrag-release-YYYYMMDD.tar.gz -C /opt/legal-rag
cd /opt/legal-rag
```

解压后应直接看到：

```text
/opt/legal-rag/backend/
/opt/legal-rag/frontend/
/opt/legal-rag/evaluation/
/opt/legal-rag/scripts/
/opt/legal-rag/data/
/opt/legal-rag/docs/
```

不要把 `.env`、`frontend/.env` 或任何真实密钥文件加入压缩包。`install.sh` 解压后会检查安装目录和 `frontend/` 下是否存在 `.env`，发现时醒目警告，必须移除后再继续交付。

## 本地端口约定

- 后端默认监听 `8001`。
- 本机 `8000` 通常由 attu（Milvus 管理界面）占用，不能配置为前端后端地址。
- 前端 `BACKEND_ORIGIN` 必须指向真实后端，例如 `http://127.0.0.1:8001`。
- 生产经 nginx 反代时，填写 nginx 能访问的内部后端地址。
- `run.sh` 启动前会检查后端端口；若被其他进程占用，会明确提示换端口或修改 `BACKEND_ORIGIN`，不会静默启动。

## Redis 持久化

安装脚本启动 `rag-redis` 时使用 Redis AOF：`appendonly yes`，并挂载 Docker volume `rag_redis_data:/data`。
已有 `rag-redis` 容器也会执行 `CONFIG SET appendonly yes` 与 `CONFIG REWRITE`，用于保证登录会话和 Redis 短期记忆不会因容器重启直接丢失。

## 演示账号清理

默认只做 dry-run 统计：

```bash
python scripts/migrations/remove_demo_accounts.py --dry-run
```

确认输出无误后才执行删除：

```bash
python scripts/migrations/remove_demo_accounts.py --execute
```

脚本只匹配以下完整邮箱，不使用通配符：
`legaladmin@qq.com`、`legaluser@qq.com`、`928421739@qq.com`。

## 安装后校验

```bash
./scripts/deploy/install.sh --method archive --archive ./legal-rag-release.tar.gz
./scripts/deploy/run.sh --env production
```

安装后应确认压缩包不含 `.env`，并检查：

```bash
curl http://127.0.0.1:8001/health/live
curl http://127.0.0.1:3000/
```
