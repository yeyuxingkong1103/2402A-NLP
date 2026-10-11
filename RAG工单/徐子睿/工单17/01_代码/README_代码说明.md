# 工单 17 · 代码与脚本说明

| 文件 | 作用 |
|---|---|
| `stress_test.py` | Python 压测脚本（场景 A/B），输出 P50/P95/P99、失败率；`--self-test` 可离线自检 |
| `ragflow_stress.jmx` | JMeter 压测计划（参数化 `-Jconcurrency -Jduration`），场景 A；场景 B 复制线程组改上传采样器 |
| `questions.csv` | JMeter 用的题目数据集 |
| `py_spy_profile.sh` | py-spy CPU 热点采样（火焰图 + dump 找重复初始化） |
| `memory_probe.py` | RSS 内存增长探针（定位泄漏） |
| `singleton_patch.py` | 优化补丁：组件单例 + 并发限流 + 连接池复用 + 解析异步化 |
| `prometheus_metrics.py` | Prometheus 监控埋点（时延/并发/队列/内存/显存） |

## 自检（本地可跑）

```bash
python stress_test.py --self-test      # 压测统计逻辑
python singleton_patch.py              # 单例补丁
python prometheus_metrics.py           # 埋点（无 prometheus_client 时降级）
python memory_probe.py --self-test     # 内存探针
```

## 接入 RAGFlow 的位点

- 单例：`api/db/services/dialog_service.py :: get_models()`（rerank/embd/chat 模型）
- 限流/异步：`api/apps/*` 请求入口；`api/db/services/document_app` 上传接口只投队列
- 监控：`api/apps/__init__.py` 注册 `/metrics`
- 解析异步：确认上传只 `queue_tasks`（投 Redis），解析由 `rag/svr/task_executor.py` 消费
