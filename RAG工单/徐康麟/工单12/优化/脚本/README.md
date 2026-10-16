# 优化脚本索引（工单12）

> **工单编号：人工智能NLP-RAG 项目-LightRAG 优化**

优化不新增独立脚本，全部复用 `研发\scripts\` 下的可复现命令入口（避免同一逻辑两份实现）：

| 脚本 | 作用 | 产出 |
| --- | --- | --- |
| `研发\scripts\20_build_graph.py` | 小样本抽样建图（抽取 + 归并 + 可视化） | `研发\data\graph\`、`测试\测试截图\知识图谱可视化.png` |
| `研发\scripts\21_graph_stats.py` | 图谱统计（类型分布 / 别名 / 约束相容率） | `研发\data\graph\graph_stats.json` |
| `研发\scripts\30_compare_rag_vs_lightrag.py` | 16 题 × 3 后端检索 + 生成（公平对比） | `研发\评估\逐题结果.jsonl`、`优化\评估结果\对比汇总.json` |
| `研发\scripts\31_ragas_eval.py` | RAGAS 四项（自实现 `_kw` 降级口径） | `优化\评估结果\RAGAS指标.json` |
| `研发\scripts\32_make_screenshots.py` | 中文命名截图 + 对比报告 | `测试\测试截图\*`、`优化\评估结果\对比报告.md` |
| `部署\脚本\核验交付物.ps1` | 交付物核验（扫描真实文件系统） | `部署\交付物核验.md` |

跑法见 `部署\README.md` §1/§1.1；优化点与前后对比见 `优化\LightRAG优化说明.md`。
