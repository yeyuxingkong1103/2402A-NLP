# 运维文档（OPERATION）· RAG API 服务

工单编号: 人工智能 NLP-RAG-解决API服务并发瓶颈与资源泄漏
版本: V1.0 · 日期: 2026-10-09

本文件指导运维人员完成服务的**日常监控、告警、巡检与故障处置**。

---

## 1. 监控体系总览

| 层 | 工具 | 指标 | 文件 |
|----|------|------|------|
| 应用内 | Prometheus `/metrics` | QPS、延迟直方图、在飞请求、RSS/commit、队列深度、任务计数 | `service/metrics.py` |
| 进程外 | `tools/monitor_resources.py` | USS、CPU%、线程数、句柄数 | 本目录 |
| 可视化 | Grafana | 上述指标面板 + 优化前后对比 | `部署/grafana/rag_dashboard.json` |
| 压测 | `tools/loadtest.py` / JMeter | 请求数、成功率、RPS、P50/P95/P99、内存曲线 | `测试/压测脚本/` |

### 1.1 Prometheus 指标清单

| 指标 | 类型 | 含义 | 告警阈值建议 |
|------|------|------|--------------|
| `rag_api_requests_total` | Counter | 请求总数（endpoint/status 维度） | 5xx 比例 > 1% |
| `rag_api_request_latency_ms_bucket` | Histogram | 延迟分布（算 P95/P99） | P95 > 3000ms |
| `rag_api_inflight` | Gauge | 在飞请求数 | 持续 ≈ `QUERY_CONCURRENCY` 表示饱和 |
| `rag_process_rss_bytes` | Gauge | 常驻内存 | 12h 增长 > 20% |
| `rag_process_private_bytes` | Gauge | 私有提交内存（泄漏核心口径） | 稳态斜率持续为正 |
| `rag_task_queue_depth` | Gauge | 队列积压 | 持续 > 100 |
| `rag_tasks_total` | Counter | 任务计数（queued/done/failed） | `failed` 增速异常 |

关键 PromQL：

```promql
# 吞吐
sum(rate(rag_api_requests_total[1m]))
# P95 延迟
histogram_quantile(0.95, sum(rate(rag_api_request_latency_ms_bucket[5m])) by (le))
# 内存（泄漏监控）
rag_process_private_bytes
# 队列积压
rag_task_queue_depth
```

## 2. 日常巡检（建议每日 / 每次压测前）

```bash
# 1) 健康检查
curl -s http://127.0.0.1:8001/health | python -m json.tool

# 2) 关键指标快照
curl -s http://127.0.0.1:8001/metrics | grep -E "rag_process|rag_api_inflight|rag_task_queue"

# 3) 队列消费确认（场景 B 相关）
curl -s http://127.0.0.1:8001/api/v1/datasets/loadtest/documents | python -m json.tool

# 4) 进程外资源采样（10 分钟）
python tools/monitor_resources.py --pid <PID> --duration 600 --interval 3 \
       --out output/loadtest/monitor_manual.json
```

## 3. 告警与处置

| 告警 | 可能原因 | 处置步骤 |
|------|----------|----------|
| P95 > 3s | 并发饱和 / Ollama 慢 / 索引冷 | 1) 看 `rag_api_inflight` 是否触顶；2) 查 Ollama 延迟；3) 确认单例已预热 |
| RSS/commit 持续上涨 | 缓存无界 / 连接泄漏 | 1) 查 `SESSION_MAX`/`EMBED_CACHE_MAX` 是否为 0；2) 查句柄数是否增长；3) 对比 `/health` 的 sessions/evictions |
| 队列深度持续 > 100 | Executor 不足 / 解析过慢 | 1) 增大 `EXECUTOR_WORKERS`；2) 检查 `PARSE_CONCURRENCY`；3) 扩容 Executor |
| 句柄数持续增长 | 连接未归还 | 确认使用共享 `ResourceManager.http()`；排查自定义 `requests.Session()` |
| 5xx 上升 | 下游不可用 | 查 Ollama/Redis 连通性；查 Executor 日志 |

## 4. 性能基线（本工单实测，供对照）

| 场景 | 版本 | 吞吐 | P95 | 失败 | 内存增长 |
|------|------|------|-----|------|----------|
| A（20 并发） | 基线 | 2.96 req/s | 8271ms | 0 | RSS +738% |
| A（20 并发） | 优化 | **14.02 req/s** | **1850ms** | 0 | commit +2.62% |
| B（10 并发混合） | 基线 | 0.33 req/s | 64287ms | 5 | RSS +387% |
| B（10 并发混合） | 优化 | **3.99 req/s** | **2570ms** | 0 | commit +20.45%（含合法入库） |

> 完整数据与图表见 `测试/性能测试报告.md` 与 `测试/原始数据/`。

## 5. 扩容与调优建议

1. **水平扩展**：API Server 无状态，可直接增加副本（`--workers` 或多容器），前置负载均衡；
2. **Executor 扩容**：解析瓶颈时增加 `task-executor` 副本（共享同一 Redis 队列）；
3. **限流调参**：`QUERY_CONCURRENCY` 与 `PARSE_CONCURRENCY` 按下游 Ollama 容量调优，
   原则是「保护下游、稳定 P95」而非一味放大并发；
4. **缓存调参**：`SESSION_MAX` / `EMBED_CACHE_MAX` 依据内存预算设置，务必**有界**。

## 6. 日志与排障

- API 日志：单例预热、连接池就绪、队列后端、会话清理；
- Executor 日志：解析完成/失败、队列剩余深度；
- 建议采集：`docker compose logs -f api-server task-executor`；
- 排障顺序：`/health` → `/metrics` → 队列状态 → 下游（Ollama/Redis）连通性。

## 7. 定期维护

| 周期 | 事项 |
|------|------|
| 每日 | 巡检健康检查与关键指标 |
| 每周 | 检查内存/句柄趋势，确认无缓慢泄漏 |
| 每月 | 复核限流与缓存参数；必要时重跑压测回归 |
| 变更后 | 必须重跑场景 A/B 回归，对比 P95 与内存曲线 |