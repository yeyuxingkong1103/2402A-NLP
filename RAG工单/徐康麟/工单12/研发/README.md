# 工单12 研发目录说明

**工单编号**：人工智能NLP-RAG 项目-LightRAG 优化

---

## 1 目录结构

```
研发/
├── lightrag/                     ← 核心实现（12 个模块，自研 LightRAG 链路）
│   ├── config.py                 配置加载（部署/配置/lightrag.json）
│   ├── corpus.py                 语料与分块（3013 块）
│   ├── graph_extract.py          实体/关系抽取（Ollama qwen2.5:3b）
│   ├── entity_schema.py          实体与关系类型 Schema（17 类 / 16 种关系）
│   ├── graph_store.py            图谱存取（nodes/edges/graphml/json）
│   ├── graph_viz.py              图谱可视化
│   ├── vector_index.py           本地向量 + BM25 索引（轻量自实现，不依赖工单7 AppConfig）
│   ├── double_retrieval.py       双层检索（实体级 + 关系级）+ RRF 融合
│   ├── routing.py                查询路由（按问题形态选检索深度）
│   ├── qa.py                     问答链路（检索 → 生成）
│   ├── ollama_client.py          Ollama 调用封装
│   └── logging_util.py           结构化日志
├── scripts/                      ← 可复现脚本（按序号排列）
│   ├── 10_build_corpus.py        建语料（3013 块）
│   ├── 11_build_index.py         建向量 + BM25 索引
│   ├── 20_build_graph.py         建知识图谱（抽实体/关系）
│   ├── 21_graph_stats.py         图谱统计
│   ├── 30_compare_rag_vs_lightrag.py  16 题三口径对比
│   ├── 32_make_screenshots.py    生成测试截图
│   └── 01/02/03_*.py             语料探查辅助
├── data/
│   ├── processed/chunks.jsonl    语料分块
│   ├── index/                    向量 + BM25 索引（扁平格式）
│   ├── graph/                    图谱产物（nodes/edges/graphml/json/raw_extractions）
│   ├── graph_基线88块/            建图修复前的基线快照（审计用）
│   ├── reports/                  报告产物
│   └── state.json                断点续跑状态
├── 评估/                          16 题对比产物（对比汇总.json / 逐题结果.jsonl）
└── _未接线_app_继承自工单7/         ⚠️ 见 §2
```

## 2 ⚠️ 关于 `_未接线_app_继承自工单7/`

该目录原为 `研发/app/`，是**从工单7 基座复制**的完整 RAG 应用（51 个文件，30+ 模块）。
**它对本工单不可用**，原因：

| 项 | app 期望（工单7 格式） | 工单12 实际（lightrag 自研） |
| --- | --- | --- |
| 索引目录 | `data/index/bge-m3_1024/` | `data/index/`（扁平） |
| 必需文件 | `chunks_indexed.jsonl`、`faiss.index` 等 5 件 | `vectors.npy`、`bm25.pkl`、`ids.json`（**缺前两个**） |
| 实测结果 | — | 服务启动即抛 `IndexMissingError` |

→ **已移出并保留**（不删，作为工单4/5/6/7 迭代的审计线索）。
详见 `_未接线_app_继承自工单7/README_为何未接线.md`。

**如需 Web 演示界面**：请用 **工单13 的 app**（其索引与工单7 同构，实测 HTTP 200 可用），
或在 `lightrag/double_retrieval.py::hybrid_retrieve` 之上另写轻量界面。

> 已核实：`研发/app/` **无任何外部引用**（21 处 `app.` 引用全在其自身目录内部），
> 移出后 `pytest 测试` 仍 **27 passed**、`import lightrag` 正常。

## 3 复现命令

```powershell
$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
Set-Location 'E:\gao6gongdan\工单12'

& $py 研发/scripts/10_build_corpus.py            # 建语料
& $py 研发/scripts/11_build_index.py             # 建索引
& $py 研发/scripts/20_build_graph.py --sample-size 200 --time-budget 1200   # 建图谱（LLM 抽取，慢）
& $py 研发/scripts/21_graph_stats.py             # 图谱统计
& $py 研发/scripts/30_compare_rag_vs_lightrag.py # 16 题对比
& $py 研发/scripts/32_make_screenshots.py        # 截图
& $py -m pytest 测试 -q --no-header -p no:cacheprovider   # 27 passed
```

**重活必须包在监察器里**（资源纪律 §5.5）：

```powershell
pwsh -NoProfile -File E:\gao6gongdan\_共享工具\安全跑.ps1 `
  -WorkDir "E:\gao6gongdan\工单12" `
  -Command "$py 研发/scripts/20_build_graph.py --sample-size 200 --time-budget 1200" `
  -LogName 工单12建图 -ChildLimitMB 3000
```

## 4 实测口径（截至 2026-10-16）

| 项 | 值 |
| --- | --- |
| 语料分块 | 3013 块（3 份招股说明书） |
| 图谱规模 | **664 节点 / 386 边**（采样 145/200 块，时间预算收束） |
| 抽取失败块 | **0**（修复前为 20/108 = 18.5%） |
| 关系相容率 | **32.6%**（126/386） |
| 实体类型覆盖 | 17/17 类全部用上 |
| 16 题对比 | RAG 2/16 = 12.5%｜LightRAG 3/16 = 18.8%｜LightRAG(strict) 3/16 = 18.8% |
| 测试 | **27 passed** |

> ⚠️ **重要负面结论（如实保留）**：图谱从 88 块扩到 145 块（节点 291→664、边 167→386）后，
> **16 题答对率逐项完全未变**。即**扩图不提升答对率** —— 瓶颈在图谱**质量与使用方式**，
> 不在规模。且 RAG 在上下文精度/召回上反而优于 LightRAG（见 `评估/对比汇总.json`）。
