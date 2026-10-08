# 工单十：Docker 部署文档

> 工单编号：人工智能NLP-RAG-金融问答系统部署

本文档描述金融问答系统（FastAPI v6 + Streamlit v6）的 Docker 容器化部署方法。

## 一、前置条件

| 依赖 | 说明 |
| --- | --- |
| Docker Engine + Compose 插件 | 本机验证：Docker 29.5.3 / Compose v5.1.4（Docker Desktop WSL2 集成） |
| 宿主机 Milvus | 原生运行并监听 `*:19530`（本系统复用现有 `rag_chunks` 集合，9019 chunks） |
| 模型目录 `/home/dabaie/models/` | 含 `bge-m3`、`bge-reranker-v2-m3`、`chinese-clip-vit-base-patch16` |
| `.env` | 项目根目录，含 `DEEPSEEK_API_KEY/BASE_URL/MODEL`（参照工单七 `.env`） |
| 宿主机端口 8006/8506 空闲 | 如有原生 v6 服务占用，先 `bash scripts/stop_v6.sh` 停止 |

## 二、快速部署

```bash
cd /home/dabaie/code/工单/工单十
bash scripts/start_docker_v10.sh     # 构建镜像 + 启动容器 + 健康检查（首次构建约10-20分钟）
```

启动完成后：

- FastAPI 接口：`http://localhost:8006/api/v6/health`（文档 `/docs`）
- Streamlit 界面：`http://localhost:8506`

停止（保留数据卷）：

```bash
bash scripts/stop_docker_v10.sh            # 容器删除，rag-v10-data 卷保留
bash scripts/stop_docker_v10.sh --prune    # 连带删除数据卷（数据丢失，慎用）
```

## 三、手动命令（等价于脚本）

```bash
docker compose build          # 构建镜像 rag-v10:latest
docker compose up -d          # 后台启动 rag-v10-api / rag-v10-ui
docker compose ps             # 查看状态
docker logs -f rag-v10-api    # 跟踪 API 日志
docker compose down           # 停止并删除容器（卷保留）
```

## 四、容器与端口

| 容器 | 服务 | 端口映射 | 启动命令 |
| --- | --- | --- | --- |
| rag-v10-api | FastAPI v6 | 8006:8006 | `uvicorn src.api_v6:app --host 0.0.0.0 --port 8006` |
| rag-v10-ui | Streamlit v6 | 8506:8506 | `streamlit run app/streamlit_app_v6.py --server.port 8506` |

API 主要端点：`GET /api/v6/health`、`POST /api/v6/ask`（body: `{"question": ..., "doc_id": ...}`）。

## 五、卷与数据持久化（验收二-②）

| 挂载 | 类型 | 用途 |
| --- | --- | --- |
| `rag-v10-data` → `/app/data` | named volume | 关键数据持久化：用户反馈 `feedback_v6/`、运行日志、检索缓存、图像/表格资源；**rag-api 与 rag-ui 同挂一卷，天然支持容器间数据共享** |
| `/home/dabaie/models` → 同名路径 | bind mount（只读） | 复用宿主机模型权重，镜像不含数 GB 模型文件 |

验证持久化：

```bash
docker exec rag-v10-api sh -c 'echo hello > /app/data/.persist_test'
docker compose down && docker compose up -d
docker exec rag-v10-api cat /app/data/.persist_test    # 仍输出 hello
```

## 六、网络配置（验收二-③）

- 自定义 bridge 网络 `rag-v10-net`：容器间按服务名互访，如 `docker exec rag-v10-ui python -c "import requests;print(requests.get('http://rag-api:8006/api/v6/health').status_code)"` → 200
- `extra_hosts: host.docker.internal:host-gateway`：容器访问**宿主机 Milvus(19530)**（环境变量 `MILVUS_HOST=host.docker.internal`）
- 外网：容器默认经宿主机访问 DeepSeek API（`DEEPSEEK_BASE_URL`）

## 七、环境变量

compose 通过 `env_file: .env` 注入，关键项：

| 变量 | 容器内值 | 说明 |
| --- | --- | --- |
| `MILVUS_HOST` | `host.docker.internal`（compose 覆盖） | 指向宿主机 Milvus |
| `MILVUS_PORT` | `19530` | |
| `DEEPSEEK_API_KEY` | 来自 `.env` | LLM 密钥，**不进镜像** |
| `RAG_EMBED_DEVICE` | `cpu` | 容器内强制 CPU 推理 |
| `HF_HUB_OFFLINE` | `1` | 禁止容器内联网下载模型 |

## 八、验收测试

```bash
# 容器运行状态下执行（自动完成：健康检查 + 3 道金融问答 + 日志检查 +
# 卷持久化 down/up 实证 + 容器互访 + 容器→宿主机 Milvus 连通）
/home/dabaie/code/my_project/.venv/bin/python scripts/test_deployed_v10.py
# 结果：docs/deploy_v10_test_results.json，全部 PASS 退出码 0
```

静态单测（无需 docker daemon）：`pytest tests/test_docker_deploy_v10.py -q`（12 用例）。

## 九、FAQ

1. **构建慢/内存不足**：torch CPU wheel 约 200MB，构建期 pip 需约 2GB 内存；WSL2 可在 `%UserProfile%\.wslconfig` 调大 `memory`。
2. **首次健康检查超时**：容器首次问答需加载 bge-m3/reranker/全文索引（1-2 分钟），`healthcheck.start_period=120s` 已放宽；`start_docker_v10.sh` 最长等待 10 分钟。
3. **Milvus 连不上**：确认宿主机 Milvus 监听 `*:19530` 而非仅 `127.0.0.1`；容器内 `python -c "import socket;socket.create_connection(('host.docker.internal',19530),5)"` 自检。
4. **模型找不到**：确认 bind mount 生效 `docker exec rag-v10-api ls /home/dabaie/models/bge-m3/config.json`。
5. **镜像体积**：约 2.5GB（python:3.10-slim + torch CPU + transformers 等）；模型与数据均在卷/挂载中，不随镜像分发。
