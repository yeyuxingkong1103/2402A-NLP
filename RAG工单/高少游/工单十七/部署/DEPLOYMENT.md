# 部署文档（DEPLOYMENT）· RAG API 服务

工单编号: 人工智能 NLP-RAG-解决API服务并发瓶颈与资源泄漏
版本: V1.0 · 日期: 2026-10-09

本文件指导运维人员完成 RAG API 服务（优化版）的部署，含**两种模式**：
① Docker Compose 生产拓扑（推荐）；② 本地直跑（无 Docker 环境）。

---

## 1. 环境要求

| 项 | 要求 |
|----|------|
| 操作系统 | Linux / Windows（本工单在 Windows 11 + Python 3.10 验证） |
| Python | ≥ 3.10 |
| 内存 | ≥ 8GB（优化版稳态约 0.4GB RSS，基线泄漏前约 1.2GB） |
| 依赖服务 | Ollama（嵌入 + LLM）；生产另需 Redis（可选，无则内存队列回退） |
| 端口 | API 9380（容器）/ 8000·8001（本地双版本）；Prometheus 9090；Grafana 3000 |

Ollama 所需模型：

```bash
ollama pull bge-m3:567m
ollama pull bge-large:latest
ollama pull nomic-embed-text:latest
ollama pull deepseek-r1:1.5b
```

## 2. 方式一：Docker Compose 部署（推荐）

```bash
cd 交付/实训一工单/工单十七/部署
docker compose up -d --build

# 查看状态
docker compose ps
docker compose logs -f api-server
```

将启动 5 个容器：`api-server` / `task-executor` / `redis` / `prometheus` / `grafana`。

**资源限制**（工单备注 2）：`api-server` 与 `task-executor` 均限制 `cpus=2.0`、`memory=4g`，
以模拟生产约束并提前暴露问题。

验证：

```bash
curl http://localhost:9380/health
curl http://localhost:9380/metrics | head
```

访问：Grafana `http://localhost:3000`（admin/admin，内置面板「RAG API 性能与资源监控」）。

## 3. 方式二：本地直跑（无 Docker）

```bash
cd 交付/工单代码/工单十七/研发
pip install -r requirements.txt

# 优化版（推荐，端口 8001）
python run_server.py --variant optimized --port 8001

# 基线版（缺陷复现，端口 8000）
python run_server.py --variant baseline --port 8000
```

独立 Task Executor 进程（等价 RAGFlow 的 TaskExecutor 容器）：

```bash
# 终端 A：API Server
python run_server.py --variant optimized --port 8001
# 终端 B：Task Executor（独立进程消费同一 Redis 队列）
REDIS_URL=redis://localhost:6379/0 python -m service.executor
```

> 生产命令等价形式：
> `uvicorn api.app_optimized:app --host 0.0.0.0 --port 9380 --workers 1`

## 4. 关键环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `REDIS_URL` | `redis://localhost:6379/0` | 任务队列地址 |
| `QUEUE_BACKEND` | `auto` | `auto`（Redis 优先回退内存）/ `memory` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | 本地模型服务 |
| `QUERY_CONCURRENCY` | `8` | 问答检索并发上限（信号量） |
| `PARSE_CONCURRENCY` | `2` | 文档解析并发上限 |
| `SESSION_MAX` / `SESSION_TTL` | `512` / `1800` | 会话缓存容量 / 存活秒数 |
| `HTTP_POOL_CONNECTIONS` / `HTTP_POOL_MAXSIZE` | `32` / `64` | 连接池常驻 / 最大连接 |
| `EMBED_CACHE_MAX` | `2048` | 查询编码 LRU 缓存容量（基线为 0=无界） |
| `BASELINE_RSS_LIMIT_MB` | `3072` | 基线看门狗内存上限（模拟 OOM Killer） |

## 5. 上线检查清单

- [ ] Ollama 模型齐全且可访问（`curl $OLLAMA_BASE_URL/api/tags`）
- [ ] `vector_db/` 三个索引文件完整（`chunks.jsonl` / `emb_*.npy` / `inverted_index.pkl`）
- [ ] `/health` 返回 `status=ok` 且 `engine_ready=true`
- [ ] `/metrics` 可抓取，Prometheus Targets 为 UP
- [ ] 容器资源限制已配置（`--memory` / `--cpus`）
- [ ] 压测脚本可运行（`tools/loadtest.py`），基线/优化数据可对比

## 6. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| 启动慢（~10s） | 单例预热（KB 加载 + 分词 + 向量） | 正常，属一次性成本；可加大健康检查 `start_period` |
| `/metrics` 无队列深度 | 未启用 Redis 或队列为空 | 检查 `QUEUE_BACKEND`；深度为 0 属正常 |
| 上传返回 task_id 但查不到 | 未启动 Executor | 启动 `python -m service.executor` |
| 基线服务启动后 ~数分钟退出(137) | 看门狗触发（持续泄漏） | 预期行为，用于复现 OOM；优化版无此问题 |

## 7. 回滚

Docker 模式：`docker compose down` 后切换镜像标签重新 `up`。
本地模式：`--variant baseline` 与 `--variant optimized` 互不影响，直接切换启动参数即可。