# 工单十三：RAG 性能瓶颈识别与优化

> 工单编号：人工智能NLP-RAG-RAG性能瓶颈识别与优化（V1.0-20250826）
> 项目路径：`/home/dabaie/code/工单/工单十三`
> 基线系统：工单六 RAGEngineV6 可配置混合检索引擎

## 项目简介

对现有 RAG v6 混合检索系统进行性能瓶颈识别与优化：通过分阶段结构化日志、请求 ID 全链路追踪、cProfile 剖析、基准测试与负载测试定位瓶颈，实施五项优化（召回候选缩减、重排截断、查询嵌入 LRU 缓存、启动预热、OMP 线程限制），将检索结果返回时间优化到 **3 秒以内**（工单验收标准）。

## 核心结果

| 场景 | 指标 | 优化前 | 优化后 | 提升 |
| --- | --- | --- | --- | --- |
| 单用户 | 检索 p50 | 808.6 ms | **300.6 ms** | 2.7 倍 |
| 单用户 | 检索 p95 | 1 491.7 ms | **1 080.0 ms** | 1.4 倍 |
| 负载(4并发) | 检索 p95 | 62 924.7 ms | **1 489.8 ms** | **42.2 倍** |
| 负载(4并发) | 吞吐量 | 0.40 rps | **2.04 rps** | 5.1 倍 |

**验收结论**：优化后单用户与 4 并发负载下，检索结果返回时间 p50/p95 均稳定在 3 秒以内，满足工单验收标准。

## 优化项

| # | 优化项 | 开关 | 效果 |
| --- | --- | --- | --- |
| 1 | 召回候选 24+24 → 12+12 | `RetrievalConfig(vector_recall_k/fulltext_recall_k)` | 重排推理量减半 |
| 2 | 重排输入截断 1500 → 512 字符 | `RAG_V13_RERANK_MAX_CHARS`（默认 1500） | rerank p50 721→247ms |
| 3 | 查询嵌入 LRU 缓存 | `RAG_V13_EMBED_CACHE=1` | embed p50 70→5.5ms |
| 4 | 启动预热（模型+索引+真实查询） | `engine.warmup()` | 消除冷启动 24.6s |
| 5 | OMP/MKL 线程限制为 8 | `OMP_NUM_THREADS=8` | 负载 rerank p95 40.3s→0.79s |

## 快速开始

```bash
# 优化前基准（12 题 × 2 轮 + 冷启动）
python scripts/benchmark_v13.py --phase before --mode bench

# 优化后基准
python scripts/benchmark_v13.py --phase after --mode bench

# cProfile 剖析
python scripts/benchmark_v13.py --mode profile

# 负载测试（4 并发 × 36 请求）
python scripts/benchmark_v13.py --mode load
```

## 目录结构

```
工单十三/
├── src/
│   ├── perf_v13.py                    # 分阶段打点：请求ID/计时器/JSONL日志
│   ├── rag_engine_v6.py               # v6 引擎（五阶段插桩）
│   ├── retrieval/hybrid_retriever_v6.py # 混合检索器（子阶段插桩）
│   ├── embedding.py                   # bge-m3 封装（+查询 LRU 缓存）
│   └── retrieval/rerankers.py         # 重排器（+RERANK_MAX_CHARS 开关）
├── scripts/
│   ├── benchmark_v13.py               # 基准/剖析/负载四模式脚本
│   └── render_v13_screenshots.py      # 截图渲染脚本
├── data/benchmark_questions_v13.json  # 12 题基准问题集
├── logs/perf_v13.jsonl                # 379 条分阶段打点记录
├── docs/
│   ├── 00_工单十三任务说明.md
│   ├── 14_工单十三性能瓶颈识别与优化.md  # 主文档（瓶颈+方案+对比）
│   ├── v13_benchmark_before.json / v13_benchmark_after.json
│   ├── v13_load_test.json / v13_profile_stats.txt
│   └── screenshots/                   # 9 张测试截图
└── README.md
```

## 工单要求对照

| 工单要求 | 完成情况 |
| --- | --- |
| 检索结果返回时间 ≤ 3S | 单用户 p50 300.6ms / p95 1080.0ms；4 并发负载 p95 1489.8ms，全部达标 |
| 五流程运行时间分析 | 五阶段插桩（query_routing / retrieval / context_assembly / llm_generation / post_processing），见主文档 §2 |
| 瓶颈定位工具 | cProfile + 结构化日志 + 请求 ID 追踪 + 监控埋点 + 负载测试 + 基准测试，六种手段全部落地 |
| 瓶颈原因分析 | 重排 CPU 密集、OMP 超订、冷启动懒加载、嵌入重复编码、LLM 外部 IO（见主文档 §4） |
| 优化方案 | 五项优化，全部 env/配置开关化，可回滚（见主文档 §5） |
| 性能提升对比 | 单用户检索 2.7~2.9 倍、负载 p95 42 倍、吞吐 5.1 倍（见主文档 §6） |
| 过程与问题记录文档 | `docs/14_工单十三性能瓶颈识别与优化.md` §7/§8 |

## 截图索引（docs/screenshots/）

| 截图 | 内容 |
| --- | --- |
| 01_perf_log.png | 分阶段结构化性能日志（单请求全链路 13 阶段） |
| 02_cprofile.png | cProfile 剖析 Top15（LLM 网络等待占 70.7%） |
| 03_benchmark_before.png | 优化前基准汇总 |
| 04_benchmark_after.png | 优化后基准汇总 |
| 05_load_test.png | 负载测试对比汇总 |
| 06_optimize_code.png | 关键优化代码片段 |
| 07_acceptance.png | 3 秒验收标准对照表（5 项全部达标） |
| 08_retrieval_chart.png | 检索阶段优化前后对比图 |
| 09_load_chart.png | 负载测试对比图（对数刻度） |
