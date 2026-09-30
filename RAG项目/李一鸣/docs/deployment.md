# 部署文档

## 1. Win11 + WSL + Ubuntu

Windows 本机开发使用 Conda 环境 `zhuangao6`，解释器位于 `D:\develop_tool1\anaconda3\envs\zhuangao6`。PowerShell 启动：

```powershell
Set-Location D:\rag-roleplay-system
.\run_zhuangao6.ps1
```

项目的 `.env` 已将数据库、日志、上传目录和本地索引全部配置到 D 盘。

在 WSL Ubuntu 中进入项目目录：

```bash
cd /mnt/d/rag-roleplay-system
cp .env.example .env
sudo bash scripts/install_ubuntu.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

访问 `http://localhost:8000/docs`。如果端口被占用，修改 `.env` 的 `PORT`。

## 2. Docker Compose

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f app
```

停止：

```bash
bash shutdown.sh
```

Compose 默认启动 MySQL、Redis、Milvus Standalone。生产环境要替换默认密码，并将密钥放入安全的环境管理系统。

## 3. Ubuntu 云服务器

```bash
git clone <your-repository-url> /opt/rag-roleplay-system
cd /opt/rag-roleplay-system
sudo bash scripts/install_ubuntu.sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
vim .env
nohup bash run.sh > logs/stdout.log 2>&1 &
```

建议使用 systemd 管理进程、Nginx 提供 HTTPS 和反向代理，示例：

```nginx
server {
    listen 80;
    server_name your-domain.example;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
    }
}
```

## 4. GPU 大模型

本项目的 `LLM_BASE_URL` 只需要指向 OpenAI 兼容服务，例如 vLLM/SGLang：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_API_KEY=local-key
LLM_MODEL=Qwen/Qwen3-8B
```

Qwen 27B BF16 需要约 54 GB 显存，实际还要为 KV Cache、运行时和并发预留空间；GPTQ-int4/AWQ 可显著降低显存需求，但应根据显卡、上下文长度和 QPS 做压测。不要仅凭参数量估算生产容量。

## 5. 环境分层

- 开发环境：SQLite、本地索引、hash embedding、mock LLM。
- 测试环境：独立 MySQL/Redis/Milvus，使用固定测试数据和真实/模拟 OpenAI 兼容网关。
- 生产环境：高可用 MySQL、Redis 持久化、Milvus 集群、模型服务、Nginx、HTTPS、监控、日志集中化和备份。

## 6. 健康检查与排障

```bash
curl http://127.0.0.1:8000/api/v1/health
tail -f logs/app.log
docker compose ps
docker compose logs milvus
```

若 BGE-m3、BGE-Reranker 或 PaddleOCR-VL 模型加载失败，应用会记录异常并降级；生产环境应将这些 fallback 视为告警，而不是最终质量方案。
