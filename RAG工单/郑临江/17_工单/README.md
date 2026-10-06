# 工单 17：解决 API 服务并发瓶颈与资源泄漏

**工单编号**：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏工单

## 一、故障现象

模拟 10+ 用户并发时：
- 响应延迟飙升：`/api/v1/chats_openai/{chat_id}/chat/completions` 平均响应从约 2s 陡增至 10s+；
- 内存泄漏与崩溃：API 容器内存随时间缓慢增长，数小时后触发 OOM Killer 崩溃重启；
- GPU 资源利用不当：仅文本问答时 GPU 显存也被大量占用且释放不及时。

## 二、根因分析（假设-验证-修改）

| 编号 | 类型 | 根因 | 代码位置 |
|------|------|------|----------|
| 瓶颈一 | 资源争用 | 每请求独立初始化 DeepDoc/VLM，并发时初始化延迟+显存争用 | 解析器/VLM 加载处 |
| 瓶颈二 | 阻塞堆积 | 无队列/限流，耗时检索阻塞线程导致请求堆积 | API 请求处理入口 |
| 泄漏点一 | 内存泄漏 | 中间对象/推理张量/会话上下文未正确释放 | 全局缓存与 session |
| 泄漏点二 | 连接泄漏 | DB/向量库连接未归还连接池 | 连接池调用处 |

定位工具：py-spy（CPU 热点）、memory-profiler / objgraph（对象留存）、
nvidia-smi + docker stats（GPU/内存）、redis-cli INFO stats（队列堆积）。

## 三、优化方案（resource_manager.py / api_server.py）

1. **资源池化与单例**：DeepDoc、ReRank、VLM 改造为全局单例（`get_parser/get_reranker/get_vlm`），
   避免重复初始化；DB/向量连接使用 `ConnectionPool` 并 `with` 正确归还。
2. **异步队列与限流**：检索+生成走 `ThreadPoolExecutor` 工作线程池 + 令牌桶限流，
   避免 API Server 阻塞与请求堆积；文档解析由 Task Executor 经 Redis 队列异步处理。
3. **监控**：`monitoring.py` 暴露 `/metrics`（请求量、p95/p99、吞吐、错误率、RSS、GPU 显存），
   接入 Prometheus + Grafana。

## 四、优化前后对比（基准测试报告）

| 指标 | 优化前 | 优化后 | 验收标准 |
|------|--------|--------|----------|
| 场景A 20并发 P95 | ~10s+ | ≤3s | **≤3s** |
| 场景A 请求失败 | 有超时 | 0 | 无失败 |
| 场景B 10并发混合 P95 | 崩溃/超时 | ≤5s | ≤5s |
| 内存增长(10min) | 持续上涨 | ≤10% 波动 | ≤10% |
| 12小时 RSS 增长 | >20% | ≤20% | ≤20% |

（数据为离线占位，真实值以 `python load_test.py` 与 `/metrics` 观测为准。）

## 五、目录结构

```
17_工单/
├── config.py          # 并发/连接池/限流/资源限制
├── resource_manager.py # 单例 + 连接池（修复泄漏）
├── api_server.py       # 优化后的 API 服务（异步队列+限流）
├── monitoring.py       # Prometheus 监控
├── jmeter_script.jmx   # JMeter 压测脚本（场景A/B）
├── load_test.py        # Python 压测脚本
├── docker-compose.yml  # 完整服务部署 + 资源限制
├── prometheus.yml
├── DEPLOYMENT.md       # 部署与运维文档
├── requirements.txt
└── README.md
```

## 六、验收对照

- 场景A（20并发问答）10分钟 P95≤3s、无失败、内存 10% 内波动；
- 场景B（10并发混合）无崩溃、解析任务入队被消费、P95≤5s；
- 12 小时 RSS 增长 ≤20%；
- 部署运维文档完整（DEPLOYMENT.md）；
- 代码注释含工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏工单。
