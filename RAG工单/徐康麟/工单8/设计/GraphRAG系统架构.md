# 工单8 GraphRAG 系统架构

> 工单编号：**人工智能NLP-RAG-基于Graph RAG 实现金融问答**

---

## 1. 总体架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ① 语料层（只读复用）                                                          │
│     9 份 A 股年报 PDF  →  工单7 已解析分块 8847 块（文本 7325 / 表格 1417 / 图像 105）  │
│     E:\gao6gongdan\工单7\研发\data\index\bge-m3_1024\chunks_indexed.jsonl      │
│     E:\gao6gongdan\工单7\研发\data\index\bge-m3_1024\{vectors.npy,faiss.index, │
│                                                    ids.json,bm25.pkl}         │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │ 只读
┌───────────────────────────▼──────────────────────────────────────────────────┐
│  ② 图谱构建层  研发/scripts/build_graph.py + 研发/app/graphrag/extractor.py    │
│     规则/词典/正则抽取（8 类实体 · 10 类关系）                                   │
│       ├─ 实体：company/institution/city/metric/amount/ratio/date/product/person│
│       ├─ 关系：发行 披露 持股 任职 归属于 同比变动 位于 母子公司 拥有 指标取值      │
│       └─ 提及边：实体 → 证据块（图谱扩展召回块的主通道）                          │
│     数据自证剪枝：高频剪枝（>8% 块） + 低频剪枝（仅 1 次出现的非主体实体）           │
│     可选：Ollama qwen2.5:3b 抽样补抽（≤400 块，单独记账 extractor=llm_*）        │
│     产物：entities.jsonl / relations.jsonl / graph.json / graph.graphml       │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────────┐
│  ③ 检索层  研发/app/graphrag/engine.py                                        │
│     问句 ──┬─► 基础召回：工单7 retriever.retrieve（bge-m3 向量 + BM25 → RRF → 重排）│
│            └─► 图谱召回：实体命中 → 一跳/两跳邻域 → 邻居实体的提及块（跳数衰减分）    │
│     融合：基础命中加分 + 图谱独有候选按名额交错（graph_slots=2,5,9，可审计）        │
│     证据补全：命中页同类块（枚举完整性，与 工单7 同口径）                          │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────────┐
│  ④ 生成层  研发/app/graphrag/llm.py（Ollama qwen2.5:3b）                       │
│     证据块编号化 → 提示词强制「只依据证据作答 + 标注块号」                        │
│     失败/超时 → 抽取式兜底（直接返回证据原文，不编造）                            │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────────┐
│  ⑤ 交付层                                                                     │
│     · 界面（纯标准库 Web，端口 8124）：问题 / 图谱结构 / 检索结果 / 答案          │
│     · 可视化：PIL 自绘力导向 PNG + 自包含 SVG/HTML                             │
│     · 评估：17 题实跑 + 与 工单7 冻结结果并排对比                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 2. 模块清单

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| 配置 | `研发/app/graphrag/config.py` | 路径契约 + **全部阈值**（环境变量 `GRAPHRAG_*` 可覆盖） |
| Schema | `研发/app/graphrag/schema.py` | 实体类型表 / 关系类型表 / 词典（唯一真源，pytest 校验） |
| 抽取器 | `研发/app/graphrag/extractor.py` | 规则·词典·正则抽取 + 两条数据自证剪枝 + LLM 补抽合并 |
| 存储 | `研发/app/graphrag/store.py` | 邻接表 / 名字索引 / 一跳两跳 BFS / GraphML 导出 |
| LLM | `研发/app/graphrag/llm.py` | Ollama 生成 + 关系补抽（硬超时、显式降级） |
| 引擎 | `研发/app/graphrag/engine.py` | Graph RAG 检索与问答主流程 |
| 可视化 | `研发/app/graphrag/visualize.py` | PIL 力导向 PNG + 自包含 SVG/HTML |
| 构建脚本 | `研发/scripts/build_graph.py` | 图谱构建一键入口 |
| 可视化脚本 | `研发/scripts/render_graph_viz.py` | 图谱可视化产物生成 |
| 评估脚本 | `研发/评估/run_eval.py` | 17 题评估 + 消融 + 与 工单7 对比 |
| 判分口径 | `研发/评估/判分口径.py` | **直接复用 工单7 的 `判分.py`**（保证口径逐位一致） |
| 对比脚本 | `研发/scripts/build_comparison.py` | 并排对比表（json/csv） |
| 截图脚本 | `测试/生成截图.py` | 5 张真实数字 PNG（中文命名） |
| Web 界面 | `研发/app/ui/serve_graph_web.py` | 纯标准库界面（端口 8124） |
| 复用基线 | `研发/app/core/**`、`研发/app/storage/**`、`研发/app/prompts/**` | 工单7 的检索/生成/语言/引用等模块（只在本工单内运行，不改语义） |

## 3. 数据流（一次提问）

```
用户问题（中/英）
  │
  ├─(1) 语言判定：app/core/language.resolve_language（复用 工单7）
  │      英文问句 → LLM 译成中文检索式（结构保全校验 + 重试 1 次 + 缓存）
  │
  ├─(2) 问句理解：app/core/query_understanding.analyze（语料路由 / 答案形态 / 关键词）
  │
  ├─(3) 基础召回：retriever.retrieve(use_cache=False)
  │      · 向量通道 bge-m3（FAISS IndexFlatIP / numpy 等价兜底）
  │      · BM25 通道
  │      · RRF 融合 → 轻量重排 → 语料软加权 → 同页表格补位
  │
  ├─(4) 图谱召回：engine.graph_expand
  │      · 问句实体命中（长名优先，min_entity_name_len=2）
  │      · BFS 一跳/两跳（hop=2，max_expand_nodes=40，min_confidence=0.55）
  │      · 邻居实体的提及块 + 跳数衰减分（0.055 / 0.035 / 0.022）
  │      · 相关性闸门：图谱独有块必须含问句实词或命中实体名
  │
  ├─(5) 融合排序：基础分 + graph_bonus → 图谱独有候选按 graph_slots 交错进 Top-K
  │
  ├─(6) 证据补全：expand_with_page_context（命中页同类块，max_extra=6）
  │
  ├─(7) 生成：llm.answer（Ollama qwen2.5:3b，证据块编号化，强制标注块号）
  │      失败 → 抽取式兜底（_extractive_answer）
  │
  └─(8) 输出：答案 + 证据块（chunk_id / 页码 / 文件名 / 文本）+ 局部子图 + 分段耗时
```

## 4. 关键机制说明

### 4.1 图谱双重角色

| 角色 | 机制 | 价值 |
| --- | --- | --- |
| **召回通道** | 实体的「提及块」构成实体→块的反向索引，沿边走到邻居实体即可召回**向量/BM25 没排上来的块** | 提升 recall（尤其跨文档对比题） |
| **结构解释** | 一次问答返回**局部子图**（命中实体 + 一跳/两跳节点 + 途经关系），界面直接展示 | 满足「输出解析出来的知识图谱结构」 |

### 4.2 为什么 `提及` 关系不单独落盘

`提及` 边数量等于「实体 × 提及块」（本图谱 22681 条）。若单独写进 `relations.jsonl`，
会让关系文件膨胀 5 倍且全是低信息量记录。实现上把提及块存放在 `entities.jsonl` 的 `chunks` 字段里，
`graph.json` 的节点也带 `chunks`；`store.build_store` 由此派生 `chunk_meta`（chunk_id → 页码/文件名），
保证图谱扩展召回的块**永远能给出页码与文件名**（这是「返回答案所在文本块」的硬要求）。

### 4.3 降级矩阵（任何一环缺失都不许崩）

| 缺失项 | 检测点 | 降级行为 | 留痕 |
| --- | --- | --- | --- |
| 图谱文件缺失/损坏 | `store.load_graph` | 空图谱 `ready=False`，检索退化为纯向量+BM25 | `graph.missing` / `graph.load_failed` WARNING |
| FAISS 不可用 | `app/core/vector_store` | numpy 精确检索（结果等价） | `vector.faiss_build_skipped` |
| 可用内存不足建 FAISS | `_memory_headroom_mb()` 试分配 < 120 MB | 跳过现场重建，退回 numpy | 同上（含实测余量） |
| Ollama 不可用/模型缺失 | `llm.available()` | 抽取式兜底（返回证据原文，不编造） | `llm.probe_failed` / `qa.backend_failed` |
| 单块抽取异常 | `extractor.extract_chunk` | 跳过该块并计数 | `graph.chunk_failed` |
| 分块行 JSON 损坏 | `iter_chunks` | 跳过该行并计数 | `graph.bad_jsonl_line` |

### 4.4 资源纪律（2026-XX 事故后强制）

本工单严格单进程、无 `multiprocessing` / 进程池 / `joblib`：

* 图谱构建：单进程流式扫描 7430 块，**5.2 s**，峰值内存 < 400 MB；
* 评估：单进程串行 17 题，逐题 append 到 jsonl；
* Ollama 调用：单模型串行，`keep_alive="0s"`（用完即卸载，避免 2 个模型同时驻留）；
* 索引加载的内存坑与修复见 `环境事实.md` §4。

## 5. 目录结构

```
工单8/
├── 设计/            本目录（需求/架构/Schema/接口/选型/环境/验收/README）
├── 研发/
│   ├── app/
│   │   ├── graphrag/        本工单新增：config/schema/extractor/store/llm/engine/visualize
│   │   ├── core/            复用 工单7 的检索与生成模块
│   │   ├── storage/         复用 工单7 的 SQLite 存储
│   │   ├── prompts/         复用 工单7 的提示词
│   │   └── ui/              serve_graph_web.py（8124）+ 复用的 serve_fallback.py
│   ├── scripts/             build_graph / render_graph_viz / build_comparison / 评估
│   ├── 评估/                run_eval.py + 判分口径.py
│   ├── data/
│   │   ├── raw/             9 份年报 PDF（从 C 盘复制，只读语料）
│   │   ├── graph/           ★ 图谱产物（entities/relations/graph.json/graphml/svg/html）
│   │   └── 测试问题集.jsonl  17 题（来源：工单7）
│   ├── config/              阈值配置（JSON，可覆盖默认值）
│   └── requirements.txt
├── 测试/             测试脚本 + 测试数据/ + **测试截图/**（中文命名 PNG）
├── 优化/             图谱抽取优化说明 + 脚本/ + 评估结果/
└── 部署/             脚本/ 配置/ 日志/ README.md 交付物核验.md
```
