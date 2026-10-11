# OPERATION.md（工单 17 · 运维监控文档）

> 面向日常运维：监控指标、巡检清单、告警阈值、故障处置 SOP。

## 1. 监控指标与面板

| 指标 | 来源 | 告警阈值（建议） |
|---|---|---|
| 问答 P95 响应时间 | Prometheus `ragflow_request_latency_seconds` | > 3s（持续 5min） |
| 请求失败率 | `ragflow_requests_total{status!="200"}` | > 1% |
| in-flight 并发 | `ragflow_inflight_requests` | 接近限流上限 |
| 解析队列堆积 | `redis-cli XINFO GROUPS` pending / `ragflow_queue_pending` | pending 持续增长 |
| 进程 RSS | `ragflow_process_rss_mb` | 12h 增长 > 20% |
| 容器内存 | `docker stats` | 接近 memory limit |
| GPU 显存 | `nvidia-smi dmon` / `ragflow_gpu_mem_mb` | 接近上限 |

Grafana 建议面板：**时延-QPS-并发**、**队列 lag**、**RSS/GPU 曲线**、**错误率**。

## 2. 日常巡检清单（每日/每周）

- [ ] `docker compose ps` 全部 Up (healthy)
- [ ] 队列 `pending` 是否回落（不持续增长）
- [ ] RSS 曲线是否平稳（无单调上升）
- [ ] 磁盘 / MinIO 容量
- [ ] 错误日志：`docker logs ragflow-server | grep -iE "error|exception|oom"`

## 3. 故障处置 SOP

| 故障 | 排查 | 处置 |
|---|---|---|
| 响应变慢 | 看 in-flight、队列 lag、CPU/内存 | 限流、扩容 worker、重启服务 |
| 内存缓慢增长 | `memory_probe.py` + objgraph | 定位泄漏对象 → 打补丁；临时重启 |
| 解析任务不消费 | `XINFO GROUPS` lag、task_executor 日志 | 重启 task_executor；检查 Redis 连接 |
| OOM 重启 | `docker events` 看 OOMKilled | 提高 limit / 修泄漏 / 降低并发 |
| GPU 显存不释放 | `nvidia-smi` | 检查模型 keep_alive / 单例复用 |

## 4. 变更与回滚

- 所有配置改动记入变更单；`docker-compose.override.yml` 与 `.env` 纳入版本管理；
- 回滚：`docker compose down && git checkout <prev> && docker compose up -d`。

## 5. 压测复现（回归）

```bash
bash 01_代码/py_spy_profile.sh <pid> 30
python 01_代码/stress_test.py --scenario A --users 20 --duration 600 --url <...> --out after_A.json
```
