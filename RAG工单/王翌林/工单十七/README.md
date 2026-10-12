# 工单十七 · 解决 API 服务并发瓶颈与资源泄漏

> 基于工单十六（VLM 微调后）进入试运行，模拟 ≥10 并发时出现：响应延迟飙升、内存持续增长致 OOM、纯文本问答时 GPU 显存占用不释放。本工单完成压测复现 → 根因定位 → 优化修复 → 对比验证 → 监控运维。

## 目录
```
src/
  api.py            优化版 FastAPI 服务（单例/连接池/异步队列/限流/Prometheus）
  api_baseline.py   基线（泄漏）版，用于前后对比
  rag_engine.py     RAG 引擎（Embedder/VectorStore/LLMClient/LRUCache/ConnectionPool/单例）
  metrics.py        Prometheus 指标定义
loadtest/
  locustfile.py     Locust 压测脚本（场景A 问答 / 场景B 混合）
  jmeter_rag.jmx    JMeter 测试计划
  result_*.json     压测原始数据
scripts/
  load_test.py      自包含压测（输出延迟分位/吞吐/内存曲线）
  run_loadtest.sh   一键跑基线+优化
  gen_report.py     生成对比图表
monitoring/
  prometheus.yml    Prometheus 抓取配置
  grafana-dashboard.json  监控面板
docker-compose.yml  API+Redis+Prometheus+Grafana 一键编排（含资源限制）
docs/
  问题排查分析报告.md
  01_性能测试报告.md
  DEPLOYMENT.md / OPERATION.md / 演示视频脚本.md
  screenshots/      优化前后对比图（PNG）
```

## 快速开始
```bash
cd 工单十七
export PYTHONPATH=$(pwd)
uvicorn src.api:app --port 8017          # 优化版
bash scripts/run_loadtest.sh             # 跑基线+优化压测
python scripts/gen_report.py             # 出对比图
```

## 关键结论（20 并发 / 45s）
| 指标 | 优化前 | 优化后 |
|------|:----:|:----:|
| 平均延迟 | 1104 ms | 876 ms |
| 吞吐量 | 18.1 rps | 22.8 rps |
| 内存增长 | +915% | +7.8% |
| 失败数 | 0 | 0 |

详细结论见 `docs/01_性能测试报告.md`。
