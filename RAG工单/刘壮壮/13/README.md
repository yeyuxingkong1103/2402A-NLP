# 工单 13 — RAG 性能瓶颈识别与优化

> 工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

把 RAG 检索结果返回时间优化到 3 秒以内。分析性能瓶颈原因并对应优化，给出优化前后检索时间对比。

## 方法

按工单要求，RAG 延迟 = 各阶段延迟之和 + 阶段间开销。逐个阶段分析：

| 阶段 | 说明 | 优化手段 |
| --- | --- | --- |
| 查询处理与增强 | 意图/改写 | 重复查询缓存 |
| 检索阶段 | 向量召回 + BM25 + 融合 | TopK 收敛（先少量召回再精排）、向量预归一化 |
| 上下文组装与提示工程 | 拼接片段 | 上下文长度裁剪 |
| LLM 生成 | 模型推理 | 缓存命中跳过生成 |
| 后处理与响应格式化 | 组装返回 | 精简输出 |

## 工具（`perf.py`）

- **StageTimer**：阶段计时器，统计各阶段耗时与占比
- **StructuredLogger**：结构化日志（时间戳、请求 ID、检索文档数、上下文长度、答案长度）
- **profile()**：cProfile 应用级性能分析（Python）
- 系统级/分布式追踪/监控（perf、OpenTelemetry、Prometheus+Grafana）与负载测试（k6/Locust）见 README 说明，可接入生产环境

## 运行

```bash
pip install -r requirements.txt
cd 13
python main.py                  # 基线 vs 优化 计时对比 + 结构化日志
python main.py --query "武汉兴图新科电子股份有限公司注册资本是多少？"
python main.py --profile        # cProfile 性能分析
```

## 验收要点

- 每个会话用户输入 query 后返回检索结果 ≤ 3 秒
- 输出瓶颈原因、优化方案及性能提升对比
