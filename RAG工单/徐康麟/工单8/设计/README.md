# 工单8 设计文档索引（README）

> 工单编号：**人工智能NLP-RAG-基于Graph RAG 实现金融问答**
> 目录：`E:\gao6gongdan\工单8\`

---

## 文档地图

| 文件 | 内容 |
| --- | --- |
| `工单需求原文.md` | PDF 抽取的需求原文存档（不改写） |
| `需求分析.md` | 需求逐条拆解、语料分析、**关键设计取舍**、非目标、验收映射 |
| `GraphRAG系统架构.md` | 总体架构图、模块清单、一次提问的数据流、降级矩阵、资源纪律 |
| `知识图谱Schema设计.md` | 9 类实体 + 11 类关系定义表、落盘结构、实测规模与质量、剪枝记录、自动化校验 |
| `接口设计.md` | Python API、命令行接口、HTTP 接口（8124）、产物文件契约 |
| `技术选型.md` | 选型总览表 + 关键选型论证（规则 vs LLM 抽取、为何不用 Neo4j/networkx、依赖清单） |
| `环境事实.md` | 解释器/语料/索引/内存约束/Ollama/沙箱 的**实测事实**（含 MemoryError 修复记录） |
| `验收标准.md` | 工单原文 7 条验收项 + 9 条自定义标准 + 真实评估结果 + **未达标项如实登记** |

## 一句话方案

从 工单7 已解析的 **8847 个分块**（文本 7325 / 表格 1417 / 图像 105）出发，
用**规则 + 词典 + 正则**抽取 **9817 个实体 / 4861 条关系 / 22681 条提及边**构成金融知识图谱；
检索时在 工单7 的「bge-m3 向量 + BM25 → RRF → 轻量重排」之上叠加**图谱第三通道**
（问句实体命中 → 一跳/两跳邻域扩展 → 邻居实体的提及块），
按可审计的「加分 + 名额交错」规则融合，交给 Ollama `qwen2.5:3b` 生成，
最终返回**答案所在的文本块**（chunk_id / 页码 / 文件名 / 文本）与**图谱结构缩略**，
并与 工单7 的冻结评估结果做 17 题并排对比。

## 快速开始

```powershell
cd E:\gao6gongdan\工单8

# ① 构建知识图谱（规则抽取，全量 7430 块，实测 ~7 s）
pwsh -NoProfile -File run_py.ps1 研发/scripts/build_graph.py *> 部署\日志\build_graph.log

# ② 生成图谱可视化（PNG + 自包含 SVG/HTML）
pwsh -NoProfile -File run_py.ps1 研发/scripts/render_graph_viz.py

# ③ 17 题评估 + 与工单7 对比（含 qwen2.5:3b 生成，约 25 分钟）
pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval.py *> 部署\日志\run_eval.log
pwsh -NoProfile -File run_py.ps1 研发/scripts/build_comparison.py

# ④ 生成全部测试截图（中文命名，真实数字）
pwsh -NoProfile -File run_py.ps1 测试/生成截图.py

# ⑤ 跑测试
pwsh -NoProfile -File run_py.ps1 -m pytest 测试 -q

# ⑥ 启动问答界面（端口 8124）
pwsh -NoProfile -File 部署/脚本/启动界面.ps1
```

更省事的方式（一键全流程）：`pwsh -NoProfile -File 部署/脚本/一键跑通.ps1`

## 交付目录

```
工单8/
├── 设计/     本目录（8 份设计文档 + 本 README）
├── 研发/     app/（graphrag 新增 + core 复用）、scripts/、评估/、data/（raw + graph）、requirements.txt、README.md
├── 测试/     pytest 用例 + 测试数据/ + 测试截图/（中文命名 PNG）+ 测试报告.md
├── 优化/     图谱抽取优化说明.md + 脚本/ + 评估结果/
└── 部署/     脚本/ 配置/ 日志/ README.md 交付物核验.md
```

## 纪律声明

1. **真实数据**：所有报告/截图中的数字都能追溯到 `研发/data/graph/图谱构建统计.json`、
   `测试/测试数据/*.json(l|csv)`、`部署/日志/*.log`；
2. **工单编号注释**：所有源码头部含 `# 工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答`；
3. **阈值配置化**：全部集中在 `研发/app/graphrag/config.py`，可用 `GRAPHRAG_*` 环境变量覆盖；
4. **工单7 只读**：未写入 工单7 任何文件（索引与评估结果均为只读引用）；
5. **资源纪律**：单进程、无进程池、Ollama 单模型驻留（`keep_alive="0s"`）、LLM 抽取只抽样。
