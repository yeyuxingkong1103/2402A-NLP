# 工单06：混合检索（向量检索 / 全文检索 / 混合检索）

> **工单编号**：人工智能NLP-RAG-混合检索任务
> （工单 PDF 原文：`人工智能NLP-RAG-混合检索任务任务`，见仓库根目录 PDF）
> **语料**：《招股说明书1.pdf》（武汉兴图新科电子股份有限公司）、
> 《招股说明书2.pdf》（武汉力源信息技术股份有限公司）
> **机器**：Windows 11 / Python 3.12.10 / torch 2.13.0+cu126 / RTX 4060 Laptop 8GB
> **验收对照**：`docs/09-工单验收对照表.md`（逐条对照工单硬性要求）

---

## 一、这一版加的是什么

上游 01~05 已有：PDF 解析（MinerU + 表格结构化 + 组织机构图矢量解析）、
多文档建库、父子块检索、图像 VLM 描述、多轮 Query 理解。
**工单06 只做检索策略**（需求原文：「不是重新做 PDF 解析」）：

| 检索策略 | 实现 | 关键点 |
| --- | --- | --- |
| **向量检索**（召回 + 重排） | `src/strategies.py` + `src/embedder.py` + `src/rerankers.py` | bge 稠密向量余弦召回 `top_k_recall` 条 → **Reranker 模型真实推理**精排 → `top_k_rerank` 条 |
| **全文检索** | `src/fulltext.py`（**自研倒排索引**） | 位置倒排表 + BM25；布尔 `AND/OR/NOT`、短语 `"…"`、模糊 `词~`、多字段 `title:/body:/summary:` |
| **混合检索**（推荐） | `src/strategies.py` + `src/fusion.py` | 双路召回 → **RRF / 加权平均 / 投票** 融合（权重可调）→ **必须经 Reranker 精排** |

可切换的模型（全部本机已有，**零下载**）：

* 嵌入模型 3 个：`bge-m3`（默认）/ `bge-m3-ft-v1` / `miniLM-L6`
* Reranker 模型 3 个：`bge-reranker-v2-m3`（默认）/ `bge-reranker-v2-m3-ms` / `bge-reranker-base`
* 精排算法 4 种：`model`（生产默认，真实 Reranker 推理）/ `llm`（qwen2.5:3b 打分）/
  `tfidf` / `adaptive`（用户反馈自适应）—— 后三者是**对比基线**，界面可选、报告有实测对比

---

## 二、三步跑起来

```bash
cd 工单6

# ① 本机模型扫描（硬性要求0，只读、零下载）—— 必须先跑，产出 docs/01
python scan_local_models.py

# ② 建库（复用上游解析产物 + 图像 VLM 描述增强 + 倒排/向量/BM25 三套索引）
python build_index.py                      # 约 1.5 分钟（bge-m3，RTX 4060）
python build_index.py --embedding-model bge-m3-ft-v1   # 可选：第二个嵌入模型

# ③ 界面（中英双语 / 策略切换 / 多轮 / 反馈）
streamlit run app.py
```

命令行问一句（无需界面）：

```bash
python -c "from src import bootstrap,rag; bootstrap.run_with_large_stack(lambda: print(rag.ask('武汉兴图新科电子股份有限公司注册资本是多少？', use_cache=False).answer))"
```

---

## 三、评估与报告（可复现）

```bash
python evaluate_retrieval.py --strategies hybrid,vector,fulltext   # 16 题 × 3 策略，逐题指标
python evaluate_retrieval.py --rerank-algorithm tfidf              # 换精排算法对比
python evaluate_compare.py                                         # 优化前后（消融 + 上游归档）
python evaluate_ragas.py --limit 16                                # RAGAS 四指标
python benchmark_load.py --levels 10,50,100 --per-level 20         # 高并发压测
python scripts/acceptance_pdf.py                                   # PDF 工单逐项验收实跑（14 项）
python -m pytest tests/ -q                                         # 单元测试（不加载大模型）
```

产出物落在 `docs/`（逐份清单见 `docs/09-工单验收对照表.md`）。

---

## 四、目录结构

```
工单6/
├─ app.py                     Streamlit 中英双语界面（问答/多轮/检索策略/对比/表格/知识库）
├─ scan_local_models.py       本机模型扫描（硬性要求0）→ docs/01 + data/local_models.json
├─ build_index.py             建库：上游解析产物 → 图像块增强 → 向量化 → Qdrant v6 → 倒排索引
├─ build_orgchart.py          组织结构图解析入库（复用上游 05，30 个块）
├─ evaluate_retrieval.py      16 题检索精确度（答案 + Precision/HitRate/MRR/NDCG + 引用来源）
├─ evaluate_compare.py        优化前后对比（同环境消融 + 上游自报归档）
├─ evaluate_ragas.py          RAGAS 评估（本地 Ollama 评判）
├─ benchmark_load.py          高并发压测（10/50/100）
├─ src/
│  ├─ strategies.py           ★ 三种检索策略调度（本工单核心）
│  ├─ fulltext.py             ★ 自研倒排索引（布尔/短语/模糊/多字段）
│  ├─ fusion.py               ★ 融合算法（RRF/加权平均/投票）
│  ├─ rerankers.py            ★ Reranker 注册表（3 模型 + 4 算法）
│  ├─ runtime.py              运行时设置（界面实时切换策略的单一事实源）
│  ├─ tokenizer.py            中英分词 + 数字口径归一（倒排与 BM25 共用）
│  ├─ image_blocks.py         图像块增强（复用上游 04 的 VLM 描述）
│  ├─ rag.py                  端到端问答（检索→精排→上下文→生成→校验→引用）
│  └─ ...                     上游复用：bootstrap / vector_store / chunker / parser / dialogue …
├─ data/                      语料分块、Qdrant、倒排索引、图像描述、评估结果
├─ docs/                      全部交付文档（扫描报告/精确度/对比/RAGAS/压测/技术文档/用户手册…）
│                             含 12-PDF工单逐项验收实跑记录.md（21 项逐条实跑）
└─ tests/                     单元测试（fulltext / fusion / rerankers / strategies / tokenizer …）
```

---

## 五、硬约束与容错（摘要）

* **零下载**：`src/bootstrap.py` 在进程最早期置 `HF_HUB_OFFLINE=1` 等离线开关，
  所有 `from_pretrained` 显式 `local_files_only=True`；模型全部走本机绝对路径。
  扫描报告见 `docs/01-本机模型扫描结果.md` —— 四类模型角色本机全满足，**未下载任何模型**。
* **精排不静默降级**：Reranker 加载失败会进入冷却期并**在界面/报告中如实标注降级**，
  绝不假装精排过（`analysis.rerank_info.degraded`）。
* **Qdrant local 单进程**：嵌入式模式同一存储目录只能被一个进程占用；
  并发压测在**单进程多线程**下进行（多进程需切 `QDRANT_MODE=server`）。
* 完整容错清单与长时间运行方案见 `docs/07-容错机制与长时间运行方案.md`。
