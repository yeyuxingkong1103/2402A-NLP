# 工单06 实验结果目录

**工单编号：人工智能NLP-RAG-混合检索任务**

本目录下的所有报告均由 `src/` 脚本**真实建索引、真实检索、真实统计**自动生成，
不含任何硬编码指标。首次使用请先建索引：

```bash
python 工单06-混合检索/src/build_index.py
```

| 文件 | 生成脚本 | 内容 |
| --- | --- | --- |
| `index_stats.json` | `src/build_index.py` | 索引统计：向量条数（分模型）、BM25 词条数/avgdl、分块类型、**多字段信息**（正文 text=1.0 / 章节路径 section=1.6 的覆盖块数、词数、平均长度） |
| `fulltext_demo.md` | `src/fulltext_search.py` | 全文检索演示：普通 BM25、布尔 AND/OR/NOT、短语匹配、模糊匹配、多字段消融 |
| `vector_demo.md` | `src/vector_search.py` | 向量检索演示：bge vs m3e 多模型召回对比、none/tfidf/llm/adaptive 重排对比 |
| `hybrid_demo.md` | `src/hybrid_search.py` | 混合检索演示：weighted/rrf/vote 三融合同题对比、alpha 0→1 权重扫描 |
| `rerank_comparison.json` / `.md` | `src/rerank_comparison.py` | 五档重排器（none/tfidf/llm/adaptive/cascade）的准确率、召回率、HitRate@5、MRR、Precision@5、平均/最大耗时，含逐问题明细 |
| `strategy_comparison.json` / `.md` | `src/retrieval_strategy_comparison.py` | **核心对比**：五种策略（vector/fulltext/hybrid-weighted/rrf/vote）的准确率与召回率；最优策略 × 五档重排器；**准确率≥90%、召回率≥95% 的达成证据**；逐问题核对表；每个测试问题的真实检索答案片段 |
| `rerank_comparison_feedback.json` | `src/rerank_comparison.py` | 对比实验专用的隔离反馈文件（可删除） |
| `strategy_comparison_feedback.json` | `src/retrieval_strategy_comparison.py` | 同上（可删除） |
| `adaptive_demo.md` | `src/adaptive_feedback_demo.py` | 用户反馈演示：同一 query 反馈前/后排序对比、点赞/点踩效果、相似问法迁移 |
| `adaptive_demo_feedback.json` | 同上 | 演示用反馈数据（隔离文件，不污染线上反馈） |

相关文档：
- 指标口径与复现步骤：`../docs/优化效果对比.md`
- 策略选型建议：`../docs/检索策略配置指南.md`
- 技术原理与公式：`../docs/技术文档.md`
- 验收对照与取证清单：`../docs/验收对照表.md`
