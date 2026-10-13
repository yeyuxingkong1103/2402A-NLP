# 工单17 - API 并发瓶颈与资源泄漏修复

目录包含 Prometheus 指标、请求限流/计时实现和 JMeter 场景。`concurrency_demo.py` 用标准库模拟并发请求，检查限流行为和资源计数是否回收。

```powershell
python concurrency_demo.py --requests 20 --limit 4 --output concurrency.json
```

真实 API 压测使用 `scripts/loadtest/ragflow_load_test.jmx`，需先部署 RAGFlow 服务。

