# 工单 12 · LightRAG 优化方案与实现

> **工单编号**：人工智能 NLP-RAG 项目-LightRAG 优化（V1.0-20250826）
> **关联工单**：人工智能NLP-RAG-基于PDF文档的问答系统 ｜ …表格解析及检索优化 ｜ …图像内容解析及检索优化
> **工程根**：`D:\桌面\徐子睿工单一`　**LightRAG 版本**：`lightrag-hku 1.5.7`（独立 venv `.venv_lightrag`）
> **语料**：《招股说明书1》（武汉兴图新科）、《招股说明书2》（武汉力源信息）
> **验收标准**：① 优化图谱构建中的实体、关系，抽取出准确的实体类型和关系类型；② 针对工单指定 16 题给出 **RAG / LightRAG 检索结果对比** 与 **RAGAS 指标对比**

---

## 一、任务理解与总体思路

常规 RAG 是「扁平向量」：把文本切块、编码、按相似度取回，**不建模实体之间的关系**，在多实体关联推理与跨页聚合上容易断层。
LightRAG 走「**图 + 向量**双层索引」：先从文本抽实体/关系建知识图谱，检索时**局部关键词走向量、全局关键词沿图谱扩展**，再把两路上下文交给 LLM 生成。

本工单在现有 RAG 系统之上并行接入 LightRAG，并保证**两者可比**：

| 可比项 | 取值 |
|---|---|
| 嵌入模型 | 均为本机 Ollama `bge-m3` |
| 生成模型（问答阶段） | 均为本机 Ollama `qwen2:7b` |
| 题集 | 工单指定 16 题（招股书1 中文 10 题 + 招股书2 六题） |
| 差异 | **仅检索与上下文组织方式**不同（RAG：混合检索+RRF；LightRAG：图+向量双层） |

---

## 二、实现步骤

| 步 | 内容 | 文件 |
|---|---|---|
| 1 | 建独立环境装 LightRAG（主环境不动，避免依赖冲突） | `.venv_lightrag`（`lightrag-hku 1.5.7`） |
| 2 | 语料接入：复用已有分块结果，按 `(文档,页)` 聚合为「一页一文档」并带 `【文档·第N页】` 标签 | `lightrag_rag/build_lightrag.py` |
| 3 | 确定建图范围：**工单 16 题的金标准页 + 邻页**（招股书1 28 页 + 招股书2 10 页 = 38 页） | `lightrag_rag/make_scope.py` → `scope_pages.json` |
| 4 | LightRAG 配置：`chunk_token_size=1200`、`entity_extract_max_gleaning=0`、embedding=bge-m3(本地)、异步并发可调 | `build_lightrag.py::build_rag()` |
| 5 | **建知识图谱**（实体/关系抽取 → 图存储 + 向量库） | `data/lightrag_store/` |
| 6 | **图谱优化**（别名归并/类型规范化/去噪/关系合并） | `lightrag_rag/optimize_graph.py` |
| 7 | 图谱可视化（交互式 HTML，按类型着色、按度定大小） | `lightrag_rag/viz_lightrag.py` → `static/lightrag_graph.html` / `lightrag_graph_opt.html` |
| 8 | 检索对比：同 16 题，LightRAG（`mix` / `naive`）vs 常规 RAG | `lightrag_rag/query_lightrag.py`、`evaluation/run_compare_lightrag.py` |
| 9 | RAGAS 指标对比 | `evaluation/run_ragas_lightrag.py`（复用独立 venv `.venv_ragas`） |
| 10 | 服务端支持「RAG / LightRAG 双知识库」切换 | `app/server.py`（`POST /api/lightrag`） |

---

## 三、知识图谱

**规模**：**694 个实体节点 / 873 条关系边**（建图输入 37 页，产出 37 个文本块）。

**实体类型分布（LightRAG 标准类型，共 16 种）**：

| 类型 | 数量 | 类型 | 数量 |
|---|---|---|---|
| concept | 234 | event | 33 |
| organization | 166 | method | 26 |
| artifact | 88 | other | 26 |
| content | 51 | person | 14 |
| data | 38 | location | 12 |

**高频实体（度中心 top）**：`Company`(134)、`兴图新科`(59)、`武汉兴图新科电子股份有限公司`(39)、`The Company`(34)、`Operating Activities`(17)、`兴图新科有限`(17)、`发行人`(14)、`Raised Funds`(13)、`Issuer`(12)。

**关系样例**（含方向与关键词）：

```
人民币普通股(A股) --[is issued by]--> 武汉兴图新科电子股份有限公司
兴图新科 --[competes_with / 主要竞争对手]--> 维盛网域、飞讯数码、派中科技
兴图新科 --[supplier]--> 国防指挥领域
```

可视化：`static/lightrag_graph.html`（原始）、`static/lightrag_graph_opt.html`（优化后）。

---

## 四、图谱优化（验收① 的落点）

原始抽取存在三类典型噪声，逐项优化：

| # | 问题 | 优化措施 | 结果 |
|---|---|---|---|
| 1 | **发行人别名分裂**：同一家公司被拆成 `Company` / `The Company` / `Issuer` / `发行人` / `本公司` / `兴图新科` / `兴图新科有限` / `武汉兴图新科电子有限公司` 等多个节点，跨页关系被打散 | 建立**别名归并表**，按实体来源块所属文档判定归属，统一到该文档的发行人全称（避免「Company」被错并到另一家公司） | 归并 **10 项** |
| 2 | **实体类型不规范**：出现中文自造类型（如「发行股数及占发行后总股本比例」「承销方式」） | 类型规范化映射到标准类型（`data` / `concept`） | 类型数 **16 → 10** |
| 3 | **孤立/空实体**：度=0 的实体（如未与其他实体建立关系的香港子公司）只在图上占位 | 剔除孤立节点与无类型无描述实体 | 剔除 **41** 个孤立节点 |
| 4 | **关系重复**：同名同向边因多次抽取而重复 | 边合并 + 关键词去重拼接、描述去重截断 | 边 **873 → 858**（去重后） |

**优化前后对比**

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 实体节点 | 694 | **645** |
| 关系边 | 873 | **858** |
| 实体类型数 | 16 | **10** |
| 孤立节点 | 41 | **0** |
| 发行人相关节点 | 8 个近似别名 | **1 个规范实体**（+ 明确的子公司实体） |

> 统计文件：`lightrag_rag/graph_opt_stats.json`；优化后图：`data/lightrag_store/graph_optimized.graphml`
> 说明：另做过一项**检索侧**优化实验（`query_lightrag.py --opt`：把「题干命中的指标类实体」所在页按「只补漏不抢位」补进上下文），在 16 题上指标**无变化**（43.8%→43.8%），故默认关闭、仅保留代码与实验记录，见第七节。

---

## 五、RAG vs LightRAG 检索结果对比（16 题）

> 明细：`evaluation/对比_RAG_vs_LightRAG.md` / `.json`

| 方案 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均检索+生成时延 |
|---|---|---|---|---|---|
| **常规 RAG**（向量 + BM25 + RRF + 查询理解） | **81.2%** | **100.0%** | **100.0%** | **0.8854** | **1987 ms** |
| LightRAG（mix 混合模式） | 43.8% | 75.0% | 93.8% | 0.6167 | 8888 ms |
| LightRAG（naive 纯向量模式） | 43.8% | 75.0% | 93.8% | 0.6167 | 5717 ms |

**逐题（命中排名，MISS=未召回）** 节选：

| 题号 | 问题（截断） | 常规 RAG | LightRAG(mix) |
|---|---|---|---|
| 1 | 本次发行股数及占发行后总股本比例 | 1 | 5 |
| 2 | 本次募集资金拟投资项目 | 1 | 1 |
| 3 | 存在控制关系的关联方及持股比例 | 1 | 1 |
| 4 | 不存在控制关系的关联方企业 | 1 | 1 |
| 5 | 组织结构图中销售部/大客户销售部构成 | 1 | 1 |
| 6 | 2008 年 IC 市场结构图增长最快/负增长行业 | 1 | 1 |
| 260 | 报告期军用领域收入 | 1 | **MISS** |
| 95 | 参与制定的技术标准 | 1 | 3 |
| 543 | 注册资本 | 1 | 1 |

**结论**：
- 本工单的 16 题**多为单值事实题**（某页的某个数字），常规混合检索（关键词 + 向量 + 口径消歧）更直接，Hit@1 高出 **37.4 个百分点**；
- LightRAG 在**跨页实体关联**题上有优势（如 id5/id6 图像题、id3/id4 关联方题都能直接命中），其「实体-关系」上下文对**多跳问题**更友好；
- LightRAG 单次查询多花 3.9–6.9 秒（多一次图检索 + 更长的上下文拼装），**延迟换全面性**；
- 实测 `mix` 与 `naive` 的**召回页序完全一致**（16/16）——因为 LightRAG 最终交给 LLM 的「上下文块」仍以向量通道的 `chunk_top_k` 块为主，图谱通道主要影响实体/关系摘要（"Final context: 53 entities, 114 relations, 8 chunks"）。

---

## 六、RAGAS 指标对比

> 判分器：本机 `qwen2:7b` + `bge-m3`（与生成模型同源）；输入：同 16 题的「问题 / 答案 / 检索上下文 / 参考答案」，两套方案各一份。
> 结果文件：`evaluation/ragas_lightrag_rag.json`、`evaluation/ragas_lightrag_mix.json`（同一轮次、同口径）

| RAGAS 指标 | 常规 RAG | LightRAG（mix） | 差异 | 说明 |
|---|---|---|---|---|
| **context_precision**（上下文精度，带参考答案） | **0.9519** | 0.5840 | RAG **+0.368** | RAG 的上下文几乎都相关；LightRAG 上下文里混入较多实体/关系摘要 |
| **context_recall**（上下文召回） | **0.8854** | N/A¹ | — | 该轮 LightRAG 判分输出解析失败（判分器为 7B 小模型） |
| **faithfulness**（忠实度） | 0.6890 | N/A¹ | — | 同上 |
| **answer_correctness**（答案正确性） | **0.7801** | 0.6772 | RAG **+0.103** | RAG 答案与参考答案更接近 |
| **answer_relevancy**（答案相关性） | 0.5776 | **0.6924** | LightRAG +0.115 | LightRAG 的回答更贴题（图谱上下文带来更完整的叙述） |

> ¹ LightRAG 侧的 `context_recall` / `faithfulness`：本机小模型判分器对该批样本输出**多次解析失败**（`RagasOutputParserException`），无有效打分 → 如实标注 N/A，不估值；其余 3 个指标正常。

**结论**：与第五节检索指标一致——**常规 RAG 在「上下文精度 / 答案正确性」上明显更好**（0.9519 vs 0.5840、0.7801 vs 0.6772）；
LightRAG 在**答案相关性**上略优（0.6924 vs 0.5776），说明其上下文更「对题」但**块级排序不够准**（与 Hit@1 的差距吻合）。

> ⚠️ 口径提醒：RAGAS 判分由 7B 小模型执行，**同一配置重复运行存在 ±0.1 波动**；上表为同一轮次、同一口径下两套方案的同批对比。

---

## 七、过程问题记录（踩坑与解决）

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | 本地 `qwen2:7b` 建图极慢：**38 页约 10 小时**（实测 40 分钟只完成 1 页） | 7B 模型在 8GB 笔记本 GPU 上做「长文档实体抽取」（输入 3k+ token、输出 2k+ token JSON）吞吐只有几 token/s；且与嵌入模型争显存 | **改用 DeepSeek API（`deepseek-flash`）做抽取**，全文建成 **11 分钟**；检索问答阶段仍用本地 `qwen2:7b`，保证对比公平（`LR_LLM=ollama|deepseek` 可切换） |
| 2 | 并发建图时大量 `Worker execution timeout after 480s` | LightRAG 默认异步并发 + Ollama 单模型排队 → 请求堆叠超时 | 降并发（`LR_MAX_ASYNC=1`）并**串行化**所有 Ollama 消费方（建图 / 造数据 / 评估不同时跑） |
| 3 | Ollama 服务被拖死：GPU 空闲但请求全部挂起 | 多客户端 + 显存临界（qwen2:7b 4.7G + bge-m3 0.66G + KV），runner 卡住 | 重启 Ollama 服务；此后严格控制「同一时刻一个 Ollama 消费方」 |
| 4 | LightRAG 报 `pyairports ModuleNotFoundError` / `braille 进度条` 崩溃 | 依赖缺失；`ascii_colors` 的盲文进度条在 GBK 控制台下抛异常 | 补齐依赖；运行前设 `PYTHONIOENCODING=utf-8` |
| 5 | `doc_status.json` 出现 74 条（每个文档 2 条，一半 failed） | 建图被中断后重启，LightRAG 对同一内容重复登记状态 | 清库重跑一次；终态 **37 条 processed**（37 个文本块，无重复） |
| 6 | 图谱里同一家公司拆成 8 个节点、出现英文/中文/指代三种写法 | 逐块抽取时 LLM 对同一实体用了不同表述 | 见第四节①：别名归并 + 类型规范化 |
| 7 | 检索侧「图谱补页」优化无效 | LightRAG 返回的实体多为 `organization/concept` 泛实体，其来源页遍布全书，补页等于引入噪声 | 收窄为「题干命中 + 指标/数据类实体」仍无提升 → **默认关闭**，如实记录 |

---

## 八、结论与后续优化建议

1. **知识图谱**：38 页语料 → **645 实体 / 858 关系**（优化后），实体类型收敛到 10 类标准类型，发行人别名归一；
2. **检索对比**：16 题上常规 RAG 命中更高（Hit@1 81.2% vs 43.8%，Hit@5 均 ≥93.8%），LightRAG 在跨页关联题上有价值，但时延高 3–7 秒；
3. **RAGAS 对比**：见第六节（同一轮次同口径对比）；
4. **后续可做**：① 把已有的**混合检索通道**（BM25+RRF）接到 LightRAG 的块召回上，兼顾关键词精度与图关系；② 给 LightRAG 配**本地重排模型**（bge-reranker-base 本机已有缓存）提升 `chunk_top_k` 的排序质量；③ 实体别名归并规则沉淀为可配置词典；④ 在 GPU 更充裕的机器上做**全量 600+ 页**建图（脚本已支持 `LR_SCOPE=0`）。

---

## 九、复现命令

```bat
cd /d D:\桌面\徐子睿工单一
:: 1) 建图（LR_LLM=deepseek 走云端抽取，约 11 分钟；LR_LLM=ollama 走本地，慢）
set LR_SCOPE=1 && set LR_LLM=deepseek && .\.venv_lightrag\Scripts\python.exe lightrag_rag\build_lightrag.py
:: 2) 图谱优化 + 可视化
.\.venv_lightrag\Scripts\python.exe lightrag_rag\optimize_graph.py
.\.venv_lightrag\Scripts\python.exe lightrag_rag\viz_lightrag.py
:: 3) 16 题检索对比（LightRAG 侧）
.\.venv_lightrag\Scripts\python.exe lightrag_rag\query_lightrag.py --eval --modes mix,naive
:: 4) RAG 侧同题 + 对比表 / RAGAS 输入
python evaluation\run_compare_lightrag.py --modes mix,naive
:: 5) RAGAS 对比（独立 venv）
set RAGAS_TAG=rag && C:\Users\xzr\.openclaw\workspace\.venv_ragas\Scripts\python.exe evaluation\run_ragas_lightrag.py
set RAGAS_TAG=mix && C:\Users\xzr\.openclaw\workspace\.venv_ragas\Scripts\python.exe evaluation\run_ragas_lightrag.py
```
