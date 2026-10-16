# 工单9 优化脚本目录

> 工单编号：**人工智能NLP-RAG-Graph RAG 优化任务**

本目录放**优化过程中用到的诊断/对照脚本**。它们不是交付主链路（主链路在
`研发/scripts/` 与 `研发/评估/`），而是**「现象 → 根因」定位过程留下的可复现工具**——
本工单文档里每一条「实测踩坑」都可以用这些脚本复现。

---

## 1. 诊断脚本（全部只读，不修改任何产物）

| 脚本 | 作用 | 典型输出 |
| --- | --- | --- |
| `诊断_上下文缺失关键词.py` | 逐题列出**哪些参考答案关键词没被上下文覆盖**、以及当时的证据块清单 | Q03 缺失 `['不良贷款率','绿色信贷','赤道原则']` |
| `诊断_逐题指标一览.py` | 优化前后逐题「精度/召回/块数/缺失数」一张表 + 证据块通道分布 | 定位「哪些题拉低了宏平均」 |
| `诊断_关键词在语料中的页码.py` | 把每个缺失关键词在**全语料**里定位到「公司 + 页码」 | `'赤道原则' → 平安银行 p48`（1 块） |
| `诊断_命中页内块构成.py` | 打印某公司某页的**全部块**及每块命中了哪些关键词 | 发现「国泰君安 p4-x1 装着答案但没进上下文」 |
| `诊断_英文提问路由.py` | 英文提问的语料路由、英中术语映射、证据块清单与答案 | 验证 `Ping An Bank → 平安银行` 路由生效 |
| `诊断_引擎单题调试.py` | 直接跑 `OptimizedEngine.retrieve/answer` 并打印完整 traceback 与诊断 | 排查代码异常 |
| `诊断_图谱产物字段.py` | 查看图谱产物的字段结构 | 了解 `entities.jsonl` / `relations.jsonl` 契约 |

**运行**（在工单根目录）：

```powershell
cd E:\gao6gongdan\工单9
$env:PYTHONDONTWRITEBYTECODE='1'
pwsh -NoProfile -File run_py.ps1 优化/脚本/诊断_逐题指标一览.py
```

> ⚠️ 这些脚本会读取 `优化/评估结果/*.jsonl` 与 工单7 索引（**只读**），
> 不会写入任何产物，也不会调用 LLM。

---

## 2. 三层面优化的**主实现**在哪里

| 层面 | 实现文件 | 关键函数 |
| --- | --- | --- |
| 实体/关系构建层 | `研发/app/graphrag_opt/抽取层.py` | `merge_aliases` / `merged_entities` / `normalize_relations` / `prune_noise` / `refine_graph` |
| 检索层 | `研发/app/graphrag_opt/检索层.py` | `link_entities` / `adaptive_hop` / `expand_pruned` / `fuse_scores` / `rerank_trim` / `page_expand` / `field_probe` / `select_evidence` |
| 生成层 | `研发/app/graphrag_opt/生成层.py` | `build_triples_block` / `refusal_guard` / `compose_answer` / `template_confidence` / `llm_polish` |
| 编排 | `研发/app/graphrag_opt/引擎.py` | `OptimizedEngine`（`profile="before"` 走 工单8，`profile="after"` 走本工单三层优化） |
| 阈值 | `研发/app/graphrag_opt/配置.py` + `部署/配置/优化配置.json` | 全部阈值集中在此，**不硬编码** |

---

## 3. 复现「优化前后对比」的最短路径

```powershell
cd E:\gao6gongdan\工单9
$env:PYTHONDONTWRITEBYTECODE='1'

# 抽取层优化（约 0.6 s）
pwsh -NoProfile -File run_py.ps1 研发/scripts/构建优化图谱.py

# 只跑检索（不调 LLM，约 10 s）—— 快速迭代用
pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval_opt.py --profiles before,after --no-llm

# 完整对比（含 LLM 慢路径，约 2~4 分钟）
pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval_opt.py --profiles before,after

# 单层消融（关掉某一层，看该层的净贡献）—— 阈值全部可从环境变量覆盖
$env:GRAPHRAGOPT_EXTRACT__ENABLED='false'      # 关抽取层
$env:GRAPHRAGOPT_RETRIEVAL__FIELD_PROBE='false' # 关字段锚点召回
$env:GRAPHRAGOPT_GENERATION__TEMPLATE_FIRST='false'  # 关模板快路径（回退到全 LLM）
pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval_opt.py --profiles after --no-llm --tag _消融
```

> **消融结果示例（已实测，见 `优化/优化方案与效果.md`）**：
> * 关 `field_probe`：上下文召回从 0.8739 掉回约 0.78；
> * 关 `evidence_focus_pages`：国泰君安 Q15 的召回从 1.000 掉到 0.800；
> * 关 `template_first`：生成段平均耗时从 0 ms 回到约 3500 ms，SLA 达标率从 100% 掉到约 35%。
