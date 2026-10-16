# 工单8 研发说明（README）

> 工单编号：**人工智能NLP-RAG-基于Graph RAG 实现金融问答**

---

## 1. 本工单做了什么

在 工单7 的「bge-m3 向量 + BM25 → RRF → 轻量重排」基础检索之上，**新增知识图谱层**：

1. **图谱构建**（`研发/scripts/build_graph.py`）：从 工单7 已解析的 7430 个分块里，
   用规则/词典/正则抽取实体与关系，产出 `entities.jsonl` / `relations.jsonl` / `graph.json` / `graph.graphml`；
2. **图谱可视化**（`研发/scripts/render_graph_viz.py`）：PIL 自绘力导向布局 PNG（≥120 节点 + 关系 + 图例）
   + 自包含 SVG/HTML；
3. **Graph RAG 检索**（`研发/app/graphrag/engine.py`）：向量+BM25 召回 → 问句实体命中 → 一跳/两跳邻域扩展
   → 邻居实体的提及块（跳数衰减分）→ 与基础命中融合 → 生成 → **返回答案所在文本块**；
4. **问答界面**（`研发/app/ui/serve_graph_web.py`）：纯标准库 Web（端口 8124），
   一屏展示「问题 / 图谱结构 / 检索结果 / 答案」；
5. **评估对比**（`研发/评估/run_eval.py` + `研发/scripts/build_comparison.py`）：
   17 题实跑 + 消融 + 与 工单7 冻结结果并排对比。

## 2. 目录结构

```
研发/
├── app/
│   ├── graphrag/            ★ 本工单新增（全部带工单编号注释）
│   │   ├── config.py        路径契约 + 全部阈值（GRAPHRAG_* 环境变量可覆盖）
│   │   ├── schema.py        9 类实体 + 11 类关系（唯一真源，pytest 校验）
│   │   ├── extractor.py     规则/词典/正则抽取 + 数据自证剪枝
│   │   ├── store.py         邻接表 / 一跳两跳 BFS / GraphML 导出
│   │   ├── llm.py           Ollama 客户端（生成 + 关系补抽，硬超时降级）
│   │   ├── grounded.py      落地式生成（确定性草稿 + 数值原子保全护栏）
│   │   ├── engine.py        Graph RAG 检索与问答主流程
│   │   └── visualize.py     力导向布局 + PIL/SVG/HTML 渲染
│   ├── core/                复用 工单7（检索/生成/语言/引用/日志）
│   ├── storage/ prompts/    复用 工单7
│   └── ui/
│       ├── serve_graph_web.py  ★ 本工单界面（8124）
│       └── serve_fallback.py   复用 工单7（8123，未改动）
├── scripts/
│   ├── build_graph.py       ★ 图谱构建
│   ├── render_graph_viz.py  ★ 图谱可视化
│   ├── build_comparison.py  ★ 与工单7 对比
│   └── build_index.py 等     复用 工单7（**本工单不执行**，索引直接引用其产物）
├── 评估/
│   ├── run_eval.py          ★ 17 题评估（含消融）
│   └── 判分口径.py          ★ 转发 工单7 的纯函数判分模块（口径逐位一致）
├── data/
│   ├── raw/                 9 份年报 PDF（从 C 盘复制，只读语料）
│   ├── graph/               ★ 图谱产物（entities/relations/graph.json/graphml/svg/html/统计）
│   └── 测试问题集.jsonl      17 题（来源：工单7 研发/评估/测试问题集.jsonl）
├── requirements.txt
└── README.md
```

## 3. 快速使用

```powershell
cd E:\gao6gongdan\工单8
$env:PYTHONDONTWRITEBYTECODE='1'

# 构建图谱（全量 7430 块，规则抽取，约 5 s）
pwsh -NoProfile -File run_py.ps1 研发/scripts/build_graph.py --llm-sample 0

# 图谱可视化（PNG + 自包含 SVG/HTML）
pwsh -NoProfile -File run_py.ps1 研发/scripts/render_graph_viz.py

# 单题问答（Python API）
pwsh -NoProfile -File run_py.ps1 -c "import sys; sys.path.insert(0, r'研发'); from app.graphrag.engine import GraphRAGEngine; e=GraphRAGEngine(warmup=True); a=e.answer('平安银行2019年营业收入是多少？', top_k=10); print(a.answer); print([c['chunk_id'] for c in a.evidence_chunks])"

# 启动界面
pwsh -NoProfile -File 部署\脚本\启动界面.ps1
```

## 4. 依赖

`requirements.txt` 只做登记：**本工单未新增任何第三方依赖**（全部复用 工单1 venv 已装包）。
关键依赖：`numpy` / `faiss` / `jieba` / `pymupdf` / `Pillow` / `requests` / `pytest`。

## 5. 纪律声明

* 所有源码头部含 `# 工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答`；
* 阈值集中在 `app/graphrag/config.py`，环境变量 `GRAPHRAG_<段>__<字段>` 可覆盖；
* **工单7 目录只读**：索引、评估结果均为只读引用，未写入任何文件；
* 单进程、无进程池；Ollama 单模型驻留（`keep_alive="120s"`，评估收尾显式卸载）；
* 长日志重定向到 `部署\日志\`；结束前清理 `__pycache__` / `.pytest_cache`。

## 6. 已知限制

见 `设计/验收标准.md` §4 与 `优化/图谱抽取优化说明.md` §5（如实登记：
eval_question.md 未随附件提供 ⇒ 采用 工单7 的 17 题；语音输入与演示视频未实现；
SLA 3 s 未达标，根因是 qwen2.5:3b 在本机的生成耗时）。
