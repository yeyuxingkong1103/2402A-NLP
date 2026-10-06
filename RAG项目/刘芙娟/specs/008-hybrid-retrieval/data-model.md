# Phase 1 Data Model: 混合检索（S9）

**Feature**: `008-hybrid-retrieval` | **Date**: 2026-09-28

本文件定义检索特性的全部数据形状：对外的契约模型、内部的中间结构、配置项，以及它们之间的流转。

---

## 1. 对外契约模型（MUST 逐字段匹配 `docs/05` §4.2）

### 1.1 `RetrievedPassage`

一条可展示、可溯源的原文片段。**这是检索的输出单元，也是前端渲染单元的来源。**

| 字段 | 类型 | 含义 | 约束 |
|---|---|---|---|
| `chunk_id` | `str` | 片段唯一标识，形如 `d6da41b5d356:0000` | 非空；MUST 在结果集内唯一（FR-005） |
| `file_name` | `str` | 来源文件名 | 非空；constitution 原则 II 的溯源依据 |
| `page_start` | `int` | 起始页 | ≥ 1 |
| `page_end` | `int` | 结束页 | ≥ `page_start` |
| `section` | `str \| None` | 章节标题 | 可为空串或 `None`（实测库中存在空串） |
| `block_type` | `str` | 块类型（`text` / `table` 等） | 非空 |
| `text` | `str` | **原文全文**（不是摘要、不是模型产出） | 非空；前端只给预览，展开时显示全文 |
| `score` | `float` | **余弦相似度** | `[-1, 1]`；仅被关键词路命中的片段取 `0.0`（R6） |

> ⚠️ `score` 的语义是**硬的**：前端 `frontend/js/transcript.js` 会 `score.toFixed(2)` 直接显示。塞入 RRF 融合分会显示一个用户无法解释的数字。RRF 分只存在于内部结构 `Candidate` 与日志中。

### 1.2 `RetrievalResult`

一次检索的完整结论。**这是检索与下游（生成 I-06、装配 I-07）之间的唯一交接面。**

| 字段 | 类型 | 含义 |
|---|---|---|
| `passages` | `list[RetrievedPassage]` | 按 RRF 降序，长度 ≤ `top_k` |
| `top_score` | `float \| None` | 结果中最大余弦；**全为关键词命中时 `0.0`**；结果为空时 `None` |
| `is_empty` | `bool` | **最终结果为空**（R7 重定义） |
| `below_threshold` | `bool` | 结果非空，**且所有余弦都低于阈值** —— 即"全部靠关键词救回来的"（R7 重定义） |

**`is_empty` 与 `below_threshold` 的四象限**（FR-013 要求二者可区分）：

| 情形 | `is_empty` | `below_threshold` | 含义 | 处置 |
|---|---|---|---|---|
| 库为空 / 一个候选都没有 | `True` | `False` | "完全没命中" | 拒答；查索引是否损坏 |
| 有候选但全部不达标且关键词也没救回 | `True` | `True` | "命中但不够像" | 拒答；考虑调阈值或补语料 |
| 有结果，至少一条余弦 ≥ 阈值 | `False` | `False` | 正常命中 | 进入生成（本期走 D3 过渡文案） |
| 有结果，但全部余弦 < 阈值（靠关键词纳入） | `False` | `True` | "知识库有这个词，语义模型没理解它" | 正常展示引用；**本信号持续上升是模型/语料不匹配的早期征兆** |

> R7 把 `below_threshold` 从"命中但不够像"改成了"靠关键词救回来的"。新定义**更可操作**：旧定义只是重复 `is_empty` 的信息，新定义标记的是一个具体的系统状态。

---

## 2. 内部中间结构（不出现在任何响应中）

### 2.1 `Candidate` —— 单条候选在两路中的名次

融合前的中间结构。**它的存在意义是让「两路各自的原始名次」在融合后仍然可查** —— 融合规则的有效性只能靠回看两路名次来验证（spec Key Entities 的显式要求）。

| 字段 | 类型 | 含义 |
|---|---|---|
| `chunk` | `CorpusChunk` | 该候选的语料条目（见 2.2） |
| `cosine` | `float \| None` | 语义路余弦；未命中语义路时为 `None` |
| `semantic_rank` | `int \| None` | 语义路名次，从 1 起；未命中为 `None` |
| `bm25` | `float \| None` | BM25 原始分；未命中关键词路时为 `None` |
| `lexical_rank` | `int \| None` | 关键词路名次，从 1 起；未命中为 `None` |
| `rrf_score` | `float` | `Σ 1/(k + rank)`；仅内部排序与日志用 |
| `matched_ratio` | `float` | 该候选命中的查询词占全部查询词的比例（0.0–1.0）。**由 `service` 在融合后回填** —— `fuse.merge` 是纯函数、不认识查询 |

**不变量**：`semantic_rank is not None ⟺ cosine is not None`；`lexical_rank is not None ⟺ bm25 is not None`。二者**至少有一个非 None**（否则该候选不该存在）。

### 2.2 `CorpusChunk` —— 内存语料条目

启动期从 Milvus 拉回的全量 chunk。**这是两路共用的唯一语料源**（D1/FR-009）。

| 字段 | 类型 | 来源 |
|---|---|---|
| `chunk_id` | `str` | Milvus `chunk_id` |
| `text` | `str` | Milvus `text` |
| `file_name` | `str` | Milvus `file_name` |
| `page_start` / `page_end` | `int` | Milvus 同名字段 |
| `section` | `str \| None` | Milvus `section` |
| `block_type` | `str` | Milvus `block_type` |
| `tokens` | `list[str]` | **启动期由 `tokenize(text)` 算出并缓存**，不在查询期重算 |
| `tokens_len` | `int` | `len(tokens)`，BM25 的文档长度项 |

> `tokens` 在启动期缓存：查询期的每一次 BM25 打分都要用文档长度，重算分词会把查询期成本乘以语料规模。语料是只读的，缓存没有失效问题。

---

## 3. BM25 索引结构（`lexical.py` 内部）

```
postings: dict[str, list[tuple[int, int]]]   # term → [(doc_index, tf), ...]
doc_len:  list[int]                          # 与语料列表同序
avgdl:    float                              # 平均文档长度
df:       dict[str, int]                     # term → 出现该词的文档数（= len(postings[term])）
n_docs:   int
```

**不变量**：`n_docs == len(doc_len) == len(corpus)`；`df[t] == len(postings[t])`。

**打分公式**（R1，MUST 逐字实现）：

```
IDF(t)   = ln(1 + (n_docs − df(t) + 0.5) / (df(t) + 0.5))            # 恒正
score(d) = Σ_{t ∈ q} IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·doc_len[d]/avgdl))
```

**空语料**：`n_docs == 0` 时 `avgdl` 取 1.0（避免除零），关键词路直接返回空列表 —— MUST NOT 抛异常。

---

## 4. 配置契约

全部为**显式配置**，MUST NOT 依赖任何库的默认值（constitution「知识与数据边界」）。

| 变量名 | 类型 | 默认 | 含义 | 来源 |
|---|---|---|---|---|
| `MILVUS_URI` | `str` | — | 向量库地址 | 既有（本特性转正为必需） |
| `MILVUS_COLLECTION` | `str` | — | collection 名 | 既有（本特性转正为必需） |
| `MILVUS_TOKEN` | `str \| None` | `None` | 认证令牌；实测服务端未启用 | 既有（本特性转正为必需） |
| `EMBED_MODEL_PATH` | `str` | — | BGE-M3 权重目录 | 既有（本特性转正为必需） |
| `SIMILARITY_THRESHOLD` | `float` | — | **语义路**余弦阈值（R7） | 既有（本特性转正为必需）；数值待标定 |
| `TOP_K` | `int` | — | 最终返回条数 | 既有；`docs/05` §8 裁决为 3 |
| `RETRIEVAL_CANDIDATES` | `int` | `20` | **每路**取多少候选再融合 | 本特性新增 |
| `RRF_K` | `int` | `60` | RRF 平滑常数（R5） | 本特性新增 |
| `LEXICAL_ADMIT_RANK` | `int` | `3` | 关键词路名次 ≤ 此值即可绕过语义阈值纳入（R7） | 本特性新增 |
| `LEXICAL_MIN_COVERAGE` | `float` | `0.5` | 关键词准入所需的**查询词覆盖率**下限（2026-09-28 补充裁决，FR-013a） | 本特性新增 |
| `BM25_K1` | `float` | `1.5` | BM25 词频饱和参数（R1） | 本特性新增 |
| `BM25_B` | `float` | `0.75` | BM25 长度归一化参数（R1） | 本特性新增 |

**校验规则**（启动期一次性校验，不合法即启动失败）：
- `TOP_K ≥ 1`
- `RETRIEVAL_CANDIDATES ≥ TOP_K`（否则融合的候选池比要返回的还少，`RETRIEVAL_CANDIDATES` 失去意义）
- `RRF_K ≥ 1`；`LEXICAL_ADMIT_RANK ≥ 1`；`0 ≤ LEXICAL_MIN_COVERAGE ≤ 1`
- `0 ≤ BM25_B ≤ 1`；`BM25_K1 > 0`
- `0 ≤ SIMILARITY_THRESHOLD ≤ 1`

**配置转正**：`backend/api/config.py` 的 `REQUIRED_WHEN_RETRIEVAL_LANDS` 元组 MUST 整体移入 `REQUIRED_NOW`。这是 S7 规格预先标好的转正时机（该元组的注释写着"转正时机：检索模块（I-05）接入时"）。

---

## 5. 检索流程（数据流转）

```
① routes.py:  question ──▶ capture_question()  ──▶ (落盘记录, 查询向量 or None)
                                                    │
②             query_vector ────────────────────────┘
                    │
③ stream.py:  service.search(question, query_vector, top_k, threshold)
                    │
                    ├─▶ store.search_semantic(vec, C)   ──▶ [(chunk_idx, cosine)]  (已按余弦降序)
                    │
                    ├─▶ lexical.search(question, C)     ──▶ [(chunk_idx, bm25)]    (已按 BM25 降序)
                    │
                    ├─▶ fuse.merge(...)  ── 按 chunk_id 去重 → 合并名次 → RRF 打分 → 排序
                    │
                    ├─▶ 阈值过滤（R7）:  cosine ≥ threshold  or  lexical_rank ≤ admit_rank
                    │
                    └─▶ 取前 top_k  →  RetrievalResult
                    │
④ stream.py:  citations 事件 ──▶ 前端 renderCitations（零改动）
              token 事件     ──▶ 正文（命中走 D3 过渡文案；未命中走拒答兜底）
```

**去重发生在融合前**（步骤 ③ 的 `merge` 内），不是在取前 top_k 之后 —— 否则重复项会先占用名额再被丢弃，实际返回条数少于 `top_k`。

---

## 6. 状态与失败处置

| 状态 | 触发条件 | `search()` 行为 | 请求链路表现 | 日志级别 |
|---|---|---|---|---|
| `ok` | 有结果且至少一条过语义阈值 | 返回 `is_empty=False, below_threshold=False` | 引用区 + D3 过渡文案 | INFO |
| `lexical_only` | 有结果但全部靠关键词纳入 | 返回 `is_empty=False, below_threshold=True` | 引用区 + D3 过渡文案 | **WARNING**（R7：值得关注的信号） |
| `below_threshold` | 有候选但一条都没过 | 返回 `is_empty=True, below_threshold=True` | 拒答兜底话术 | INFO |
| `no_candidates` | 两路都无候选（含空库） | 返回 `is_empty=True, below_threshold=False` | 拒答兜底话术 | INFO |
| `vector_unavailable` | 查询向量为 `None`（S8 编码失败） | 抛 `RetrievalError` | ERROR 日志 + 拒答兜底话术 | **ERROR** |
| `store_unavailable` | Milvus 查询失败 | 抛 `RetrievalError` | ERROR 日志 + 拒答兜底话术 | **ERROR** |

**后两种为何不冒泡成 500**：检索发生在 SSE 流已经开始之后（`status` 帧已发出），无法再改 HTTP 状态码。冒泡会让客户端拿到"流中断"，与"网络断了"不可区分（R9）。

**为何不违反「禁止 try/except 吞掉」**：判定标准是"事后能不能查出来"。这里 MUST 留 ERROR 日志（含 `answer_id` 与异常类型），失败**可查**。同一条推理已在 `backend/api/capture.py` 的模块文档字符串中确立。

**`store_unavailable` 在启动期已被挡掉一次**：语料构建在启动期（R3/FR-010），Milvus 不可达则服务直接启动失败。此处的 `store_unavailable` 只覆盖"启动后 Milvus 才挂掉"的情况。

---

## 7. 与既有模型的边界

| 既有模型 | 位置 | 关系 |
|---|---|---|
| `backend.api.schemas.Citation` | 传输层 | 由 `RetrievedPassage` **映射**而来，字段名逐字段一致（`docs/05` §3.1.3）。映射是刻意的：检索层不该依赖 HTTP 传输模型，否则改前端契约会牵动检索实现 |
| `backend.query.service.Encoder` | S8 | **只读复用**（`encode_query`），本特性 MUST NOT 调用它（FR-003 禁止重新编码） |
| `backend.query.gate` | S8 | 启动期已保证查询向量与索引同空间，本特性**不重复校验** |
| `backend.embed.*` | S5 | 不直接接触 |
| `backend.index.*` | S6 | **不直接接触**。collection 名与字段名与 `backend/index/__init__.py` 的契约一致，但本特性不复用其模块 —— 那是入库侧的包，含写库能力，复用会把写能力带进只读路径 |
