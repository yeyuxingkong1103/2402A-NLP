# 部署与运维文档（DEPLOYMENT.md / OPERATION.md）

**工单编号**：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏工单

## 一、部署

```bash
# 1. 构建并启动完整服务（API Server + Task Executor + Redis + Prometheus + Grafana）
docker compose up -d --build

# 2. 查看服务状态
docker compose ps

# 3. 验证
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Docker 资源限制已在 `docker-compose.yml` 中配置（`memory: 8g`、`cpus: 4.0`），
通过环境变量 `CONTAINER_MEMORY` / `CONTAINER_CPUS` 可覆盖，用于模拟生产约束。

## 二、日常监控

| 面板 | 地址 | 关注指标 |
|------|------|----------|
| Grafana | http://localhost:3000 | 延迟 p95/p99、吞吐、错误率 |
| Prometheus | http://localhost:9090 | 指标查询 |
| 容器资源 | `docker stats` | CPU、内存 |
| GPU | `nvidia-smi` | 显存占用与释放 |

## 三、运维要点

1. **内存泄漏排查**：观察 `/metrics` 的 `rag_process_rss_mb` 是否随时间持续上涨；
   12 小时 RSS 增长应 ≤20%。
2. **队列堆积**：`redis-cli INFO stats` 观察任务队列长度；堆积说明 Task Executor 消费不足。
3. **限流**：`config.RATE_LIMIT_PER_SEC` 控制每秒最大请求数，超限返回 429。
4. **连接池**：`DB_POOL_SIZE` / `VECTOR_POOL_SIZE` 控制连接池大小，连接用后必须 `release`。

## 四、故障预案

- 服务 OOM → 提高 `CONTAINER_MEMORY` 或排查泄漏（用 memory-profiler/objgraph 定位未释放对象）；
- 响应延迟升高 → 检查 `rag_latency_p95` 与队列长度，确认是否触发限流或线程池打满；
- GPU 显存不释放 → 确认 VLM 是否按单例复用（`SINGLETON_VLM=True`）。
