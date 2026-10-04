# 工单 13：RAG 性能瓶颈识别与优化

**工单编号**：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化

参考：https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-1-production-rag-foundations/rag-performance-bottlenecks

## 一、任务目标

RAG 检索结果返回时间优化到 **3 秒以内**；分析性能瓶颈原因并给出优化前后对比。

RAG 系统的延迟 = 各组成部分延迟之和 + 阶段间数据传输/排队开销。
吞吐量由处理能力最低的阶段决定。

## 二、RAG 流水线 5 个阶段（逐段计时）

| 阶段 | 说明 | 常见瓶颈 |
|------|------|----------|
| 1. 查询处理与增强 | 分词、意图/指代消解 | 分词/嵌入耗时 |
| 2. 检索阶段 | 向量/全文召回 | **全量扫描、无索引** |
| 3. 上下文组装与提示工程 | 拼接上下文 | 上下文过长 |
| 4. LLM 生成 | 调用大模型 | **模型推理延迟** |
| 5. 后处理与响应格式化 | 组装返回 | 序列化 |

## 三、识别缓慢之处的工具与技术

| 类别 | 工具 | 对应文件 |
|------|------|----------|
| 性能分析（应用级） | Python cProfile + snakeviz | `profiling.py` |
| 性能分析（系统级） | Linux perf（CPU/系统调用/I/O） | 见备注 |
| 日志记录 | 结构化日志（请求ID + 时间戳 + 指标） | `observability.py` |
| 分布式追踪 | OpenTelemetry / Jaeger / Zipkin | `tracing.py` |
| 监控与告警 | Grafana / Prometheus | `observability.py` + `docker-compose.yml` |
| 负载测试 | k6 / Locust / JMeter | `load_test.py` |
| 基准测试 | 组件原始性能测量 | `benchmark.py` |

## 四、优化方案

| 瓶颈 | 原因 | 优化 |
|------|------|------|
| 检索阶段慢 | 每次查询全量扫描所有文档 | **预构建倒排索引**，O(关键词) 召回 |
| 重复查询慢 | 相同 query 反复检索+生成 | **结果缓存**，命中缓存直接返回 |
| LLM 生成慢 | 模型推理延迟占比高 | 换更小/更快模型、流式输出、批处理 |
| 上下文过长 | 拼接过多文本 | 限制 Top-K、截断上下文 |

## 五、目录结构

```
13_工单/
├── config.py           # 配置（目标延迟 3s、测试查询、优化开关）
├── pipeline.py         # RAG 流水线（5 阶段计时 + 索引/缓存优化）
├── observability.py    # 结构化日志 + Prometheus 指标
├── tracing.py          # OpenTelemetry / Jaeger / Zipkin
├── profiling.py        # cProfile + snakeviz
├── benchmark.py        # 组件基准测试
├── load_test.py        # k6 / Locust 负载测试脚本
├── main.py             # 优化前后对比主程序
├── docker-compose.yml  # Prometheus + Grafana + Jaeger
├── prometheus.yml      # 监控抓取配置
├── requirements.txt
└── README.md
```

## 六、运行

```bash
pip install -r requirements.txt
python main.py              # 输出优化前后耗时对比、瓶颈分析、监控指标、profile
snakeviz profile.prof       # 可视化性能分析
python load_test.py         # 生成 Locust / k6 脚本
docker compose up -d        # 启动 Prometheus/Grafana/Jaeger
```

## 七、监控 KPI

- 延迟（平均值、中位数、p95/p99 百分位数）
- 吞吐量（每秒请求数）
- 错误率
- 资源利用率（CPU、内存、GPU、网络、磁盘 I/O）
- 队列长度

## 八、验收对照

- 分析检索性能慢的原因：5 阶段逐段计时 + cProfile + 结构化日志 + 追踪；
- 给出优化方案：预建索引、结果缓存等；
- 给出优化前后检索时间对比：`main.py` 输出加速比与耗时对比；
- 每个会话返回检索结果 ≤ 3 秒：通过缓存 + 索引优化满足（离线模拟 LLM 延迟 0.3s）；
- 代码注释含工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。
