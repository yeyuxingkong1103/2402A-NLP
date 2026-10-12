# 工单十七 · 运维文档（OPERATION）

## 1. 日常巡检
```bash
curl http://127.0.0.1:8017/health        # 服务健康 + 当前 RSS
curl http://127.0.0.1:8017/metrics       # Prometheus 指标
docker stats rag-api                     # 容器 CPU/内存
nvidia-smi                               # GPU 显存（如启用 GPU 模型）
redis-cli INFO stats                     # 队列/连接情况
```

## 2. 关键告警阈值（Grafana/Prometheus）
| 指标 | 阈值 | 处置 |
|------|------|------|
| `rag_request_duration_seconds` P95 | > 3s 持续 2min | 检查 LLM 并发与队列堆积 |
| `rag_process_rss_mb` | 环比 10min 上涨 > 20% | 排查缓存/连接泄漏，重启实例 |
| `rag_inflight_requests` | 持续 > LLM 并发上限 | 限流生效，考虑扩容 |
| `rag_queue_depth` | 持续 > 0 不回落 | 检查文档解析消费者存活 |

## 3. 长稳测试（12h）
```bash
# 低并发持续压，观察 RSS 是否持续上涨（应 ≤ 20%）
python scripts/load_test.py --mode optimized --port 8017 \
  --out loadtest/stability.json --concurrency 5 --duration 43200
```

## 4. 故障处置
- **响应变慢**：先看 `/metrics` 的 in-flight 与队列深度；若 in-flight 打满，调大 LLM 并发上限或扩容副本。
- **内存上涨**：确认缓存命中淘汰正常（`rag_cache_size` 不应超 capacity）；必要时滚动重启。
- **服务崩溃**：`docker-compose logs rag-api` 查 OOM；调高 `memory` 限制或排查泄漏回归。

## 5. 发布/回滚
- 滚动发布：`docker-compose up -d --no-deps rag-api`。
- 回滚：替换镜像 tag 后 `docker-compose up -d`。
