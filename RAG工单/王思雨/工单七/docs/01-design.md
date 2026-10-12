# 第一步 · 设计

> 工单编号：**人工智能NLP-RAG-功能测试及评估**（工单七）
> 目标：在 01–06 已实现的 RAG 之上，对 `ccf_competition` 语料做**功能测试与检索评估**。

---

## 1. 项目目标与本工单边界

**做什么**：把 `ccf_competition` 的 9 份年报 PDF 建成可检索知识库 → 用《sample_questions.pdf》的示例问题（补足至 10 题）跑通整条 RAG 链路 → 用 RAGAS 评估检索与生成质量 → 分析检索结果的问题 → 给出优化方案 → 提供 FastAPI + Vue 演示。

**不做什么**（属于 01–06 的工作，本工单只调用不重造）：Query 理解改写、表格结构化重建、图像/图表解析、Graph+RAG。

**语料说明（与工单原文的差异，如实标注）**：工单写的是《ccf_competition.zip》，实际附件目录 `RAG 工单/附件/ccf_competition/` 下**没有 zip**，已是解压好的 `pdf/`（9 份，62MB）与 `txt/`（9 份，同源）。本工单按**已解压目录**处理，`app/ingest.py` 直接读 `pdf/`；`txt/` 作为解析兜底。语料内容与工单一致（9 份 A 股上市公司年报）。

| 股票代码 | 公司 | 报告年度 | 页数 | 文本层字符数 |
|---|---|---|---|---|
| 000001 | 平安银行 | 2019 | 265 | 265,776 |
| 601318 | 中国平安 | 2019 | 342 | 341,076 |
| 600036 | 招商银行 | 2019 | 351 | 412,970 |
| 601658 | 邮储银行 | 2019 | 421 | 323,371 |
| 600030 | 中信证券 | 2020 | 370 | 444,865 |
| 601628 | 中国人寿 | 2020 | 235 | 236,634 |
| 601601 | 中国太保 | 2021 | 264 | 254,203 |
| 600999 | 招商证券 | 2021 | 278 | 363,055 |
| 601211 | 国泰君安 | 2021 | 286 | 333,360 |

**⚠️ 一个必须先解决的坑（已实测）**：`pdf/` 下 9 个文件的**中文文件名在磁盘上已被 `U+FFFD` 破坏且不可逆**（例如 `2020-02-14__������...__000001__...pdf`）。直接拿文件名当文档名会得到乱码。**唯一幸存的稳定标识是 6 位股票代码**，因此 `app/ingest.py::doc_meta()` 用股票代码回认公司名（`config.CODE2NAME`），并用文件名前缀的**披露日期**（`2020-02-14`）推算报告年度（披露年 − 1，均为 3 月前披露的年报）。

---

## 2. 整体架构

```
                        ┌──────────── 离线：建库（一次性） ────────────┐
  ccf_competition/pdf/  │                                              │
  9 份年报 PDF ─────────►│ ① 解析（四级降级）                           │
                        │    MinerU 主路径 ──失败──► pymupdf ──失败──► 附件 txt │
                        │ ② 语义分块（bge-m3 句向量 + 语义漂移切块，带重叠）      │
                        │ ③ 向量化（bge-m3，1024 维，归一化）           │
                        └───────────┬──────────────────┬───────────────┘
                                    ▼                  ▼
                            Milvus `gd7_chunks`   data/bm25.pkl（BM25Okapi + 分块表）

                        ┌──────────── 在线：一次问答 ────────────┐
   问题 ──► ① 向量召回 top-30 ─┐                                  │
        └─► ② BM25 召回 top-30 ┴─► ③ 通道内归一化 + 加权融合(0.7/0.3) top-30
                                             │
                                             ▼
                              ④ bge-reranker-v2-m3 精排 → top-8
                                             │
                    ⑤ 拼上下文（带 公司/年份/页码 引用） → DeepSeek 生成
                                             │
                                  Redis 缓存（结构指纹版键，TTL 1 天）
                                             ▼
                                        答案 + 引用片段

                        ┌──────────── 评估：RAGAS ────────────┐
   tests/questions.json（10 题 + ground_truth）
        │
        ├─► 逐题关缓存跑真实链路 → 收集 answer / retrieved_contexts
        │        （先落盘 results/raw_*.json，防长任务中断全丢）
        └─► RAGAS 四指标【逐指标串行】：
              Faithfulness / Answer Relevancy / Context Recall / Context Precision
              → nan 记 None 并单独上报有效样本数
              → results/eval_*.json

                        ┌──────────── 演示 ────────────┐
   Vue3（frontend/index.html） ◄── FastAPI（app/main.py，:8010）
        /health /questions /retrieve /ask /results/latest
```

**分层与文件对应**

| 层 | 文件 | 职责 |
|---|---|---|
| 配置 | `app/config.py` | 路径、阈值、模型与集合名（唯一配置源） |
| 入库 | `app/ingest.py` | ZIP/PDF 解析（四级降级）→ 语义分块 → Milvus + BM25 |
| 检索生成 | `app/rag_core.py` | 向量 + BM25 → 融合 → 重排 → DeepSeek 生成 → Redis 缓存 |
| 评估 | `app/evaluate.py` | 10 题作答落盘 + RAGAS 四指标 |
| HTTP | `app/main.py` | FastAPI 路由（分发层，不含算法） |
| 演示 | `frontend/index.html` | Vue 3 单文件，展示检索/生成/评估全流程 |
| 入口 | `scripts/*.py`、`scripts/start.bat` | 建索引 / 跑评估 / 一键启动 |

**硬性技术选型落地对照**

| 要求 | 实现 |
|---|---|
| Python 3.11+ | 实际用 `D:/myconda1/python.exe`（**Python 3.12.7**，唯一同时具备 CUDA torch + MinerU + ragas 的解释器） |
| FastAPI + Vue | `app/main.py` + `frontend/index.html`（Vue 3 全局构建，本地 `vue.global.prod.js`，不走 CDN） |
| MinerU + 备用 | 四级链：MinerU → pymupdf → 附件 txt；`MINERU_ENABLED=0` 可临时关闭主路径 |
| 语义分块 | `app/ingest.py::sem_chunk()`：句向量与当前块语义中心的余弦相似度跌破 `SEM_THRESHOLD` 且块已够长即切块 |
| bge-m3 | `D:/模型/bge-m3`（1024 维，归一化，COSINE） |
| Milvus | `localhost:19530`，集合 `gd7_chunks`（**独立集合，不污染工单六的 `gd6_chunks`**） |
| BM25 | `rank_bm25.BM25Okapi` + jieba 分词，索引落盘 `data/bm25.pkl` |
| bge-reranker-v2-m3 | `D:/模型/bge-reranker-v2-m3`，FlagReranker + fp16，输入截断 512 字符 |
| DeepSeek | `DEEPSEEK_API_KEY` 走环境变量/`.env`，`https://api.deepseek.com`，`deepseek-chat`，温度 0 |
| Redis | `redis://localhost:6379/0`，**必须 `protocol=2`**（本机 Redis 5.0.14 不支持 RESP3） |
| RAGAS | `ragas 0.4.3`，四指标逐指标串行 |

---

## 3. 评估流程设计

### 3.1 从 `sample_questions.pdf` 提取问题

实测该 PDF 共 **4 个问题**（含参考答案），不足 10 个，因此按工单要求**基于文档内容仿写补齐至 10 个**：

| 题号 | 来源 | 目标文档 | 类型 |
|---|---|---|---|
| Q01 | sample 原题 | 平安银行 2019 | 单事实·致辞归纳 |
| Q02 | sample 原题 | 招商银行 2019 | 单事实·致辞归纳 |
| Q03 | sample 原题 | 平安银行 2019 | 多事实·分析 |
| Q04 | sample 原题 | 跨文档 | 跨文档·对比综合 |
| Q05 | 仿写 | 招商银行 2019 p.53 | 单文档·多指标数值 |
| Q06 | 仿写 | 邮储银行 2019 p.61 | 单文档·单指标数值 |
| Q07 | 仿写 | 中国平安 2019 p.40 | 单文档·多指标数值与同比 |
| Q08 | 仿写 | 中信证券 2020 p.31 | 单文档·业务条线分解 |
| Q09 | 仿写 | 中国人寿 2020 p.12/24 | 单文档·多指标数值 |
| Q10 | 仿写 | 国泰君安 2021 p.16/17 | 单文档·多指标数值与同比 |

**两处原题指代的还原（依据语料原文，非猜测）**：
- Q01 的 `xx银行` = **平安银行**：参考答案里的「零售进一步巩固收入贡献的支柱地位」「对公、资金同业」「逾期60天以上贷款占比和拨备覆盖率」三个特征词组，在 9 份年报中**只**出现在 `000001`（平安银行 2019 董事长致辞，p.5）。
- Q02 的 `xx银行` = **招商银行**：「发起设立『逾越者联盟』、参与咖啡零售、出行预订、影票销售」逐字出现在 `600036`（招商银行 2019 年报董事长致辞，p.5–6），其余 8 份 0 命中。

**ground_truth 的生成口径（可核验）**：全部 10 题的 `ground_truth` **逐字取自对应 PDF 原文**，并逐条核对了页码与数字（含同比）；未使用任何语料外知识、未使用模型推测。原文事实由独立检索脚本按「去空格后再匹配 → 回映射取原文」的方式抽取，规避了 PDF 抽取在中文词间插入空格造成的漏配。

### 3.2 构建 RAGAS 评估集

RAGAS 0.4.3 的 `SingleTurnSample` 四元组 ← 本工单的映射：

| RAGAS 字段 | 来源 | 说明 |
|---|---|---|
| `user_input` | `questions[i].question` | 题面（含原题中的 xx 指代，**不改写**，因为要测的就是系统面对原始问法的表现） |
| `response` | 系统生成的答案 | 关缓存跑真实链路得到 |
| `reference` | `questions[i].ground_truth` | 逐字取自 PDF 的标准答案 |
| `retrieved_contexts` | 重排后 top-8 片段的纯文本 | **喂进去的就是喂给大模型的那 8 段**，保证「评估的检索 = 上线的检索」 |

### 3.3 调用 RAG 生成答案与检索上下文

`app/evaluate.py::collect()` 逐题调用 `app/rag_core.py::answer(question, mode="hybrid", use_cache=False)`：

- **必须关缓存**：否则第一次跑完写进 Redis，第二轮评估会全部命中缓存，测不出链路真实表现；
- **先落盘再算指标**：`results/raw_*.json` 在作答阶段就写出，RAGAS 阶段失败也不丢数据（长任务纪律）；
- 每题同时记录 `retrieval_stat`（向量/BM25 各召回多少条）与 `ctx_meta`（每个片段来自哪个公司、哪一页、由哪个通道召回、重排分多少），**这是第四步「检索结果问题分析」的数据来源**。

### 3.4 计算指标

四个指标、**逐指标串行**（`for name, m in plan: evaluate(ds, metrics=[m], ...)`）：

| 指标 | 衡量什么 | 本工单的判读要点 |
|---|---|---|
| Faithfulness | 答案是否完全由检索片段支撑 | 低 = 幻觉或超出片段作答 |
| Answer Relevancy | 答案与问题的相关程度 | 低 = 答非所问或混入无关内容 |
| Context Recall | 标准答案的要点被召回的比例 | 低 = 检索没找到 → 先查语料是否覆盖 |
| Context Precision | 有用片段在排序中的靠前程度 | 低 = 召回噪声多 / 排序差 |

**并发与容错（本机实测约束）**：`EVAL_MAX_WORKERS=2`。RAGAS 默认 16 并发会打满 DeepSeek 限流，把样本静默打成 `nan`；四个指标混在一次 `evaluate()` 里跑会让并发再翻 4 倍。指标值 `nan` 一律写 `None`，并单独记录 `metric_valid`（如 `9/10`）——**绝不把 nan 当数字上报**。

### 3.5 评估口径与适用前提

- 本评估是**端到端一次性问答**评估（单轮、无多轮改写），`question` 原样入参；
- 指标均值同时给出**有效样本数**；若某指标有效样本 < 10，该指标结论**不得当作达标结论**使用；
- RAGAS 判官模型与生成模型同为 DeepSeek，存在**同源偏差**（判官可能偏好与自己风格相近的答案），因此第四步的问题分析以**逐题人工复核 + `ctx_meta` 通道溯源**为主，RAGAS 分数仅作辅助。

---

## 4. 目录结构

```
工单七/
├─ app/                      # 核心链路（4 个模块共 300 行，逐行中文注释）
│   ├─ config.py             (36)  配置：路径/阈值/模型/集合名
│   ├─ ingest.py             (99)  解析（四级降级）+ 语义分块 + 写 Milvus/BM25
│   ├─ rag_core.py           (97)  向量+BM25 融合 → 重排 → DeepSeek → Redis 缓存
│   ├─ evaluate.py           (68)  10 题作答落盘 + RAGAS 四指标
│   ├─ main.py               (68)  FastAPI 路由层（HTTP 分发，不含算法）
│   └─ __init__.py
├─ scripts/
│   ├─ build_index.py        # 建索引入口
│   ├─ run_eval.py           # 评估入口
│   └─ start.bat             # 一键启动（探活 Redis → uvicorn :8010）
├─ tests/
│   ├─ questions.json        # 10 题 + ground_truth
│   └─ test_pipeline.py      # 自检：解析降级链、融合、缓存、RAGAS 输入构造
├─ frontend/index.html       # Vue 3 单文件演示页（+ 本地 vue.global.prod.js）
├─ data/                     # bm25.pkl / index.json / parsed/（MinerU 产物缓存）
├─ results/                  # raw_*.json（作答原始）/ eval_*.json（含指标）
├─ docs/                     # 01-design ~ 05-deploy
├─ logs/
├─ .env / .env.example / requirements.txt / README.md / CLAUDE.md
```

**300 行约束的落实与口径（写明，不藏）**：
- **核心链路 4 模块（config + ingest + rag_core + evaluate）共 300 物理行**，每行代码都带中文注释，注释为**行尾注释**（不额外占行），因此 300 行内同时装下了代码与注释；
- `app/main.py`（69 行）是工单另外单独要求的 **FastAPI 路由层**，与 Vue 同属演示层，**不并入核心链路行数**；`app/__init__.py`（2 行）为包标记；
- 全部 Python 合计 371 行。若把 `main.py` 也算进来则为 371 > 300——此处口径已在表格中明示，供验收判断；
- 为守住 300 行而**放弃的非核心功能**：关键词判分器（只用 RAGAS + 人工复核）、多轮会话历史、反馈票箱、PDF 图像/表格结构化重建（属 03/04 工单）。这些均已在 `app/*.py` 注释中标注。
