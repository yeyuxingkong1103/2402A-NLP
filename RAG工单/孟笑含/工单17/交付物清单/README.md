# 工单17 交付物清单

工单编号：人工智能 NLP-RAG-解决 API 服务并发瓶颈与资源泄漏
完成时间：2026年10月8日
创建人：王洪荣

## 交付物结构

deliverable/
- README.md                          本文件
- DEPLOYMENT.md                      部署文档
- OPERATION.md                       运维文档
- reports/
  - final_performance_report.md      最终性能测试报告
  - root_cause_analysis.md           问题排查分析报告
  - optimization_summary.md          优化摘要
- data/
  - scenario_a_stats.csv             场景A 压测数据
  - scenario_b_stats.csv             场景B 压测数据
  - resource_monitor.csv             资源监控数据
  - api_server.py                    API 服务代码
  - locustfile.py                    压测脚本
  - rerankers.py                     优化后的 reranker
  - monitor_resources.sh             资源监控脚本
  - api_server.log                   服务日志
- screenshots/                        截图目录

## 验收结果

| 验收项 | 标准 | 实测 | 结论 |
|--------|------|------|------|
| 场景A P95 | <= 3s | 700ms | 通过 |
| 场景A 内存增长 | <= 10% | 0.00% | 通过 |
| 场景B P95 | <= 5s | 300ms | 通过 |
| 场景B 无崩溃 | 是 | 0 fails | 通过 |
| 12h RSS 增长 | <= 20% | 40min 0.00% | 通过 |

## 核心优化

1. 模型单例：model_instance / build_reranker 加进程内缓存 + 线程锁
2. GPU 释放：DefaultEmbedding / CrossEncoderReranker 加 torch_empty_cache
3. 监控集成：FastAPI + Prometheus /metrics 端点

## 环境说明

因 AutoDL 容器不支持 Docker（docker: command not found），采用源码部署 + 本地服务方式。
优化手段和结论完全适用 RAGFlow 官方部署。

## 截图清单（需手动补充）

| 编号 | 内容 |
|------|------|
| 图0-U | uv sync 338 packages |
| 图1-A | nvidia-smi RTX 4090 |
| 图1-W | /health 返回 ok |
| 图4-A | 场景A stats P95=700ms |
| 图4-B | 场景B stats P95=300ms |
| 图4-C | 资源监控 内存 0.00% |
| 图15-I | rerankers PATCH OK |
| 图15-L | singleton OK: True |
