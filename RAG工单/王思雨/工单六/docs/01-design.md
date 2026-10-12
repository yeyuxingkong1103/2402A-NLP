# 阶段一 · 设计文档

> 工单编号：人工智能NLP-RAG-混合检索任务
> 对应工单：《人工智能NLP-RAG项目-06-混合检索任务工单V1.1-20250206.pdf》
> 设计原则：**只实现工单明确写出的要求**。工单没写的（登录注册、权限、知识图谱、微调、OCR 训练、监控、多租户、支付、管理后台）一律不做。

---

## 1. 需求到实现的映射

| 工单要求 | 实现位置 | 落点 |
|---|---|---|
| 向量嵌入把文档与 Query 转向量 | `rag_core.embed` | bge-m3（本地 `D:/模型/bge-m3`），L2 归一化 |
| 余弦相似度在 Milvus 快速召回 | `rag_core.vector_search` + `ingest.ensure_collection` | 向量归一化后内积 == 余弦，Milvus 用 `IP` 度量 |
| 支持多种嵌入模型配置 | `config.EMBED_MODELS` + `embed(texts, name)` | 注册表式，加 `bge`/`m3e` 只需登记本地路径 |
| 至少 3 种重排算法 | `rerank_bge` / `rerank_tfidf` / `rerank_adaptive` | 见 §5 |
| 返回 topK，支持配置 | `retrieve(top_k=...)` + `/config` | 前端滑杆 1~30 |
| 倒排索引实现全文检索 | `rag_core.bm25_build/bm25_search` | rank_bm25（BM25Okapi） |
| 布尔查询、短语匹配、模糊匹配 | `query_parse` + `_apply_mask` + `_fuzzy_expand` | 见 §4 |
| 多字段检索（标题/正文/摘要） | `bm25_build` 三棵索引 + `FIELD_WEIGHTS` | 见 §4 |
| 同时执行向量与全文检索 | `retrieve(mode="hybrid")` | 两路各自召回后融合 |
| 混合检索配置项、权重可调 | `/config`、`/search`、前端 | `vec_weight` / `bm25_weight` |
| 融合算法：加权平均、投票 | `fuse()` 的 `weighted` / `vote` | 另加 `rrf`（工单「如…等」允许扩充） |
| 提升 PDF 检索精度 | 见 `04-optimize.md` 的消融与对比 | 准确率/召回率实测 |
| 基于检索结果调 DeepSeek 生成 | `rag_core.answer` + `llm_chat` | 见 §6 |
| 中英文问答 | `_looks_english` + 语言指令 | 见 §6 |
| 多轮对话，历史存 Redis | `hist_get/hist_add` + `retrieve` 前指代消解 | 见 §7 |
| 用户反馈用于自适应重排 | `/feedback` → `fb_add` → `rerank_adaptive` | 见 §5.3 |
| Vue 页面：模式/权重/topK/重排器 | `frontend/index.html` | 本地内置 Vue 3，不依赖 CDN |
| RAGAS 评估 + 报告 | `app/eval_ragas.py` + `POST /eval` | 见 §8 |
| 响应 ≤3 秒 | 缓存 + 级联重排 + 上下文裁剪 | 见 `04-optimize.md` |

---

## 2. 系统架构

```
                        ┌──────────────────────────────┐
   浏览器 (Vue3 SPA) ──▶ │  FastAPI  app/main.py        │
   /config /search       │  /health /ingest /search     │
   /chat   /feedback     │  /chat  /feedback /eval      │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │  app/rag_core.py  核心层      │
                        │                              │
                        │  ┌────────────┐ ┌──────────┐ │
                        │  │ 向量通道    │ │ 全文通道  │ │
                        │  │ bge-m3编码  │ │ BM25×3字段│ │
                        │  │ Milvus IP   │ │ 布尔/短语 │ │
                        │  │ 余弦召回    │ │ 模糊匹配  │ │
                        │  └─────┬──────┘ └────┬─────┘ │
                        │        └────┬────────┘       │
                        │      ┌──────▼──────┐         │
                        │      │ 融合层       │         │
                        │      │ weighted/    │         │
                        │      │ rrf / vote   │         │
                        │      └──────┬──────┘         │
                        │      ┌──────▼──────┐         │
                        │      │ 级联重排     │         │
                        │      │ tfidf→bge    │         │
                        │      │ /adaptive    │         │
                        │      └──────┬──────┘         │
                        └─────────────┼────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
      │ Milvus        │      │ Redis         │      │ DeepSeek API  │
      │ gd6_chunks    │      │ 答案/检索缓存  │      │ deepseek-chat │
      │ 向量+业务字段  │      │ 会话历史/反馈  │      │ OpenAI 兼容   │
      └───────────────┘      └───────────────┘      └───────────────┘
```

离线链路（`app/ingest.py`）：

```
PDF → MinerU 解析 → 统一块 → 归并同页同章节正文 → 语义分块
                                              → 去页眉/单位标注噪声
                                              → 建 3 字段 BM25 索引（→ bm25.pkl）
                                              → bge-m3 批量编码 → Milvus（gd6_chunks）
```

**降级链（每一处失败都必须如实上报 `degraded`，不静默）**

| 环节 | 主路径 | 降级路径 | 上报字段 |
|---|---|---|---|
| PDF 解析 | MinerU | pymupdf | `ingest` 日志 |
| 向量通道 | Milvus 检索 | 返回空，由 BM25 撑住 | `degraded: ["vector"]` |
| 全文通道 | BM25 | 返回空，由向量撑住 | `degraded: ["bm25"]` |
| 重排 | bge-reranker | 回退融合排序 | `degraded: ["rerank"]` |
| 大模型 | DeepSeek | 只列检索原文，不编造 | `degraded: ["llm"]` |
| 缓存/会话 | Redis | 退化为无缓存、单轮 | `degraded: ["redis"]`（健康检查） |

---

## 3. Milvus Schema（集合 `gd6_chunks`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64 | 自增主键 |
| `vector` | FLOAT_VECTOR(1024) | bge-m3 向量，**已 L2 归一化** |
| `chunk_id` | VARCHAR(64) | 全局唯一块号，如 `招股说明书1-00418` |
| `doc` | VARCHAR(128) | 文档名，用于区分两份招股说明书 |
| `title` | VARCHAR(512) | 章节标题（多字段检索的「标题」字段） |
| `text` | VARCHAR(65535) | 正文（多字段检索的「正文」字段） |
| `abstract` | VARCHAR(2048) | 抽取式摘要，取正文首 150 字（多字段检索的「摘要」字段） |
| `source` | VARCHAR(128) | 工单要求的「来源」字段 |
| `page_no` / `page_end` | INT16 | 起止页，1 起，与 PDF 页码一致 |
| `block_type` | VARCHAR(16) | `text` / `table` / `image` / `chart` |
| `locator` | VARCHAR(256) | 可读定位串，如 `招股说明书1 P129` |

索引：`IVF_FLAT`，度量 `IP`，`nlist=1024`。

> **踩坑（写在触点旁）**
> - 向量必须归一化，否则 `IP` 不等于余弦相似度。
> - `pymilvus` 的 `MilvusClient` **没有 `flush()`**，刚写入立刻查询会读不到；因此 `vector_search` 一律显式 `consistency_level="Strong"`。
> - `get_collection_stats()` 返回的 `row_count` 在未落盘时**恒为 0**，只能用于展示，不能拿来做业务判断（本项目用本地 `bm25.pkl` 的块数作为权威计数）。

---

## 4. 全文检索设计

### 4.1 分词

`tokenize()`：中文走 jieba；英文用正则单独补抽（jieba 对长英文串切得差）；数字额外收录**去千分位**形态，使 `14,414.16` 与 `14414.16` 命中同一批词——财报类问题全靠数字定位。

### 4.2 多字段 BM25

标题、正文、摘要各建一棵 `BM25Okapi`，检索时按字段权重加权求和：

```
score(d, q) = W_title · BM25_title(d,q) + W_body · BM25_body(d,q) + W_abstract · BM25_abstract(d,q)
默认 W_title = 2.0, W_body = 1.0, W_abstract = 0.6（可在 .env 调整）
```

标题权重最高：章节标题是极强的相关性信号。摘要字段是抽取式的，与正文高度相关，其**边际贡献在 `04-optimize.md` 里做了消融实测**，不做无依据的宣称。

### 4.3 查询语法

| 语法 | 含义 | 实现 |
|---|---|---|
| `甲 AND 乙` / `甲 与 乙` | 布尔与：两块都出现 | `must` 词项 ⊆ 块词集 |
| `甲 OR 乙` / `或` | 布尔或（默认语义） | `should` 词项，BM25 打分 |
| `NOT 乙` / `非` | 排除 | `neg` 词项 ∩ 块词集 = ∅ |
| `"连续短语"` | 短语匹配 | 归一化后子串包含 |
| `词~` | 模糊匹配 | rapidfuzz 在 27019 词表里找 `score_cutoff=80` 的近邻，扩召回 |

布尔/短语约束以布尔掩码乘到 BM25 得分上；无约束时**跳过全表扫描**（省一次 O(N) 遍历）。

> **踩坑**：早期实现把原始查询串直接交给 jieba，导致 `AND`/`NOT`/`~` 这些**运算符本身被当成检索词**参与打分。现改为只使用解析后的词项。

---

## 5. 重排算法设计

### 5.1 bge-reranker-v2-m3（交叉编码）
`FlagReranker` 成对打分，`normalize=True` 输出 0~1。**打分输入截断到 512 字**：不截断会按最长序列 padding，把打分耗时放大数倍。

### 5.2 TF-IDF 重排
在候选集上现建 `TfidfVectorizer`（分词器与 BM25 同一套，保证口径一致），算查询与候选的余弦。纯 CPU、毫秒级，适合做级联第一级。

### 5.3 用户反馈自适应重排
Redis 里按**片段级**与**文档级**两个键记票：`gd6:fb:chunk:{id}`、`gd6:fb:doc:{doc}`（`HINCRBY up/down`）。

```
feedback = (up - down) / (up + down + 1)          # 平滑到 (-1, 1)
adapt_score = base_score × (1 + α · feedback)      # α = ADAPTIVE_ALPHA = 0.5
```

### 5.4 级联（重排器顺序）
`rerankers` 是**有序列表**，按顺序级联执行：便宜的 TF-IDF 先截断到 `CASCADE_MID=12`，再由 bge 精排到 `top_k`。顺序不同，耗时不同——这是优化阶段可实测的调参项，而不是装饰。

---

## 6. 融合算法设计

两路分数**量级不可比**，直接加权会把弱权通道架空，因此**先在通道内归一化再融合**：

| 算法 | 公式 | 特点 |
|---|---|---|
| `weighted` 加权平均 | `Σ_ch w_ch · minmax(score_ch)` | 分数敏感，需归一化 |
| `rrf` 倒数排名融合 | `Σ_ch w_ch / (K + rank_ch)`，K=60 | 只看名次，对分数尺度鲁棒 |
| `vote` 投票机制 | `Σ_ch w_ch`（被几路召回）− 0.001·最好名次 | 强调「两路都认可」的块 |

融合结果额外回带 `channels`（命中通道）与 `rank_detail`（两路原始名次），便于对比排序变化——**排序改动必须能打印前后名次**。

---

## 7. 问答与交互设计

- **生成**：DeepSeek `deepseek-chat`，温度 0（招股书问答要可复现），`max_tokens=800`，超时 25s。
- **提示词**：只依据【检索片段】作答、数字必须在片段中能找到、句末标注【来源：文档 第X页】、片段不足时明说无法回答。
- **中英文**：`_looks_english()` 按字母/汉字计数判语种，切换系统指令语言。
- **多轮**：历史存 Redis `gd6:hist:{session}`（`rpush`+`ltrim` 定长窗口，TTL 2h）。检索前若当前问题**过短或含指代词**（他/她/它/该/这/那/其/上述），把上一轮用户问句拼进检索问句完成指代消解。
- **反馈**：`/feedback` 写 Redis，供自适应重排。

---

## 8. 缓存设计

键结构（四段，缺一不可）：

```
gd6:{kind}:{代码指纹}:{知识库指纹}:{内容哈希}
```

- **代码指纹**：`config.py`+`rag_core.py`+`ingest.py` 源码 sha256 前 12 位 → 改代码自动换键，避免旧缓存**静默吞掉新字段**（如 `degraded`）。
- **知识库指纹**：`bm25.pkl` 的 mtime 纳秒 → 重新分块后旧答案立即失效。
- **内容哈希**：问句 + 模式 + topK + 权重 + 融合 + 重排器顺序 + 历史摘要 → 调参也会换键。
- 两级缓存：`kind=answer`（答案）与 `kind=retrieval`（检索结果）。缓存**不含 `latency_ms`**，命中时回填本次真实耗时并置 `cached=true`。

---

## 9. Redis 键空间

| 键 | 类型 | 用途 | TTL |
|---|---|---|---|
| `gd6:answer:*` / `gd6:retrieval:*` | string(JSON) | 两级缓存 | 86400s |
| `gd6:hist:{session}` | list | 多轮历史（定长 12 条） | 7200s |
| `gd6:fb:chunk:{id}` | hash | 片段级 up/down | 86400s |
| `gd6:fb:doc:{doc}` | hash | 文档级 up/down | 86400s |

> **踩坑**：本机 Redis 5.0.14 **不支持 RESP3(HELLO)**，redis-py 8 必须显式 `protocol=2`，否则连接直接失败。

---

## 10. 评估方案

| 验收指标 | 口径 | 来源 |
|---|---|---|
| 准确率 ≥90% | 关键词判分：`expect` 关键字段全含 **且** `expect_not` 负样本一个不出现 | `eval_ragas._correct` |
| 召回率 ≥95% | RAGAS `context_recall` | `eval_ragas._ragas` |
| 响应时间 ≤3s | 逐题 `latency_ms` 的 p50/p95/max 与超 3 秒题数 | `eval_ragas.run` |
| 评估报告 | RAGAS 四指标 + 逐题明细落盘 `results/` | `POST /eval` |

`expect_not` 负样本是必须的：**关键词判分抓不到「多答无关内容」**——只查「关键字段在不在」会把「答对了但也混进另一家公司的数字」判成正确。本套题集专门为两家公司互设负样本（如问兴图时出现「力源信息」「赵马克」即判错）。

题集：`tests/questions.json`，共 **22 题**（中文 15、英文 5、多轮指代 2），覆盖两份招股说明书。

---

## 11. 配置项与陷阱索引

所有可调参数集中在 `app/config.py` + `.env`，其余模块只读不硬编码。**已知陷阱写在配置项旁边**：

| 配置 | 陷阱 |
|---|---|
| `EMBED_MODEL_PATH` / `RERANK_MODEL_PATH` | 路径必须**正斜杠**：`D:\模型\bge-m3` 的 `\b` 会被 .env 解析成退格符 |
| `REDIS_PROTOCOL=2` | Redis 5.x 不支持 RESP3，必须显式 RESP2 |
| `LLM_REASONING_EFFORT` | 推理型模型（名字含 flash/reason）不传 `none` 会把 `max_tokens` 吃光返回空内容 |
| `RERANK_MAX_CHARS=512` | 不截断会按最长序列 padding，重排耗时放大数倍 |
| `MINERU_LANGFIX_DIR` | 中文用户名下 fasttext 打不开默认模型，缺了必失败 |
| `EVAL_MAX_WORKERS=2` | ragas 默认 16 并发会打满 DeepSeek 限流，把指标打成 `nan` |
| `API_PORT=8010` | 8000 已被本机工单五服务占用 |
