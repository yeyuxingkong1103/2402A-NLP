# 工单06 演示截图目录

**工单编号：人工智能NLP-RAG-混合检索任务**

本目录存放演示视频与验收材料的截图。运行对应脚本后，按下表截取关键画面，
文件名建议与表格一致（PNG，宽 ≥1280px）。

| 文件名 | 来源命令 | 截图内容 | 证明的验收点 |
| --- | --- | --- | --- |
| `01-index-stats.png` | `python src/build_index.py` | 索引完成输出：向量条数、BM25 词条数、多字段权重 1.0/1.6 | 倒排索引 + 多字段检索 |
| `02-vector-models.png` | `python src/vector_search.py` | bge-large-zh-v1.5 与 m3e-base 的同题 Top-5 对比与重合度 | 支持多种嵌入模型 |
| `03-vector-rerank.png` | `python src/vector_search.py` | none/tfidf/llm/adaptive 四档重排输出与耗时 | 召回 + 重排；≥3 种重排算法 |
| `04-fulltext-bm25.png` | `python src/fulltext_search.py` | 普通 BM25 查询结果表 | 倒排索引 + BM25 |
| `05-fulltext-boolean-phrase-fuzzy.png` | 同上 | 布尔 AND/OR/NOT、短语匹配、通配符/编辑距离 | 布尔/短语/模糊匹配 |
| `06-fulltext-field-weights.png` | 同上 | 多字段加权消融排名对比 | 多字段检索（正文/标题） |
| `07-hybrid-fusion.png` | `python src/hybrid_search.py` | weighted/rrf/vote 三算法同题对比 | 融合算法（加权平均/投票等） |
| `08-hybrid-alpha-sweep.png` | 同上 | alpha 0→1 的 Top-3 与目标片段排名变化 | 权重调整 |
| `09-strategy-comparison.png` | `python src/retrieval_strategy_comparison.py` | 五种策略准确率/召回率表 | 策略配置对比 |
| `10-strategy-evidence.png` | 打开 `results/strategy_comparison.md` | 达成证据：准确率≥90%、召回率≥95% | 准确率/召回率验收 |
| `11-retrieval-answers.png` | 同上第五节 | 测试问题的 Top-3 真实检索片段与页码 | 检索问题的答案 |
| `12-rerank-comparison.png` | `python src/rerank_comparison.py` | 五档重排 HitRate/MRR/Recall/耗时表 | 重排对比与 3 秒性能 |
| `13-adaptive-feedback.png` | `python src/adaptive_feedback_demo.py` | 反馈前后 Top-10 排序变化 | 用户反馈交互 |
| `14-web-multiturn.png` | `python src/serve.py` | Web 界面多轮问答（含指代消解） | 交互友好性/多轮对话 |
| `15-web-feedback.png` | 同上 | 点击「有帮助/没帮助」后的提示 | 用户反馈 |
| `16-api-strategy-switch.png` | `curl /api/ask`（见演示脚本） | 同一问题切换 strategy/reranker 的 trace | 策略在线配置 |
| `17-latency-trace.png` | `return_trace=true` 响应 | 各阶段耗时明细 | 响应时间 ≤3 秒 |
| `18-metrics.png` | `GET /api/metrics` | 请求量、P95 耗时、LLM 用量 | 资源消耗观测 |

命名规范：`序号-内容.png`；若同一画面需多张，追加 `-a/-b` 后缀。
