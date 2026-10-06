# 契约：检索服务（I-05）

**Feature**: `008-hybrid-retrieval` | **Version**: 1.0 | **Date**: 2026-09-28

上游契约：`docs/05_接口设计.md` §4.2。本文件**不重新定义**该契约，只做两件事：
1. 把 `docs/05` §4.2 留白的部分（混合检索下阈值的适用范围）**补齐**；
2. 把实现必须遵守的不变量写成可断言的形式。

> ⚠️ 本文档 §2 的扩展 MUST 回写 `docs/05_接口设计.md` §4.2 —— 契约变更 MUST NOT 只存在于规格目录里。

---

## 1. 函数签名（与 `docs/05` §4.2 逐字一致）

```python
def search(
    question: str,
    query_vector: list[float],
    top_k: int,
    threshold: float,
) -> RetrievalResult: ...
```

**与 `docs/05` §4.2 的差异说明**：原契约签名为 `search(question, top_k, threshold)`。本特性**增加一个参数 `query_vector`**，理由见 §3。这是本特性对上游契约的**第二处扩展**，同样 MUST 回写 `docs/05`。

**签名不变量**：

| 不变量 | 理由 |
|---|---|
| `top_k` 与 `threshold` MUST 由调用方显式传入，MUST NOT 有默认值 | `docs/05` §4.2 约束 3：便于测试注入不同参数 |
| 函数内 MUST NOT 读 `os.environ` | FR-024：配置在启动期一次性加载 |
| 函数内 MUST NOT 调用任何编码器 | FR-003：查询向量由调用方提供 |
| 函数 MUST NOT 修改入参 | 便于重复调用 |
| 相同入参 + 相同索引 → 相同返回值（逐字段） | SC-006 可复现 |

---

## 2. 返回值的语义澄清（对 `docs/05` §4.2 的扩展，R7）

`docs/05` §4.2 原文：

> `is_empty: bool  # 命中为空 或 最高分低于阈值`
> `below_threshold: bool  # 区分"完全没命中"与"命中但不够像"`

**原文只设想了单路检索。** 混合检索下"命中"横跨两条路，因此三个字段的定义 MUST 按下列语义执行：

```python
passages = [已被纳入的候选，按 rrf_score 降序][:top_k]

纳入条件: (cosine >= threshold)
          or (lexical_rank <= lexical_admit_rank
              and matched_ratio >= lexical_min_coverage)

top_score        = max(cosine for p in passages) if passages else None
                   # 全为关键词命中时 => 0.0（不是 None —— 见 INV-6 的说明）
is_empty         = len(passages) == 0
below_threshold  = 有候选存在，且其中最高余弦仍 < threshold
                   # 注意：独立于 is_empty。无候选时为 False。
```

**为什么这样改**：若阈值只卡余弦（原文的字面执行），用户照抄一个罕见术语时 BM25 的精确命中会被判为不相关并走拒答 —— 而"照抄术语能命中"（US1）恰恰是本特性存在的理由。见 `research.md` R7 的三方案对比。

**为什么准入还需要 `matched_ratio`（查询词覆盖率）** —— 这是 2026-09-28 的补充裁决，修的是一个真实缺陷：

`lexical_rank` **单独不可用**：BM25 对任何查询都必然返回前 N 名，包括与知识库
完全无关的提问。实测（语料 65 chunks）「今天晚饭吃什么好」的 top-3 覆盖率仅
0.20 却占据关键词路前三名，于是被当成了"命中" —— 直接违反 constitution 原则 II。

| 查询 | 关键词路 top-3 覆盖率 | 语义最高余弦 | 应当 |
|---|---|---|---|
| `氢氯噻嗪`（术语照抄） | 1.00, 1.00, 0.50 | 0.5023 | 命中（US1） |
| `今天晚饭吃什么好`（无关） | 0.20, 0.20, 0.20 | 0.5198 | 拒答（SC-004） |

覆盖率的直觉：**关键词准入存在的理由是"用户照抄术语"—— 照抄时查询词基本都在
库里；闲聊式提问的大部分词库里根本没有。** 它与语料规模无关。

**两个被实测否掉的判据**（记录在此，以免后人重复尝试）：

- **IDF 门槛 —— 分不开。** 「晚饭」idf=3.78、「好」idf=2.69，比「高血压」的
  0.42 **还高**。在一个通篇讲高血压的语料里，「高血压」本身就是高频词。
- **余弦下限 —— 分不开。** 无关问题的最高余弦 0.5198 **高于** 术语问题
  （`氢氯噻嗪`）的 0.5023。小语料（单文档、主题均质）下 BGE-M3 的余弦有一个
  很高的地板，两类查询的分布完全重叠。

**`below_threshold` 的新含义**：原定义（"命中但不够像"）在混合检索下被收紧为
「**有候选，但全部不达标**」。`is_empty=True` 的两种情形因此可区分（FR-013）：

| 情形 | `is_empty` | `below_threshold` |
|---|---|---|
| 两路都没有候选 | `True` | `False` |
| 有候选，但全部未过阈值 | `True` | `True` |
| 有结果，至少一条过阈值 | `False` | `False` |
| 有结果，全部靠关键词纳入 | `False` | `True` |

**`lexical_admit_rank`**：默认 3，显式配置。取 3 而非 1 的理由：允许关键词路的前几名都过 —— 只放第 1 名会让"原文在两段里各说一半"这种情况只召回其中一半。

**`lexical_min_coverage`**：默认 0.5（一半以上的查询词要命中），显式配置。调低会让无关提问更容易被放行；调到 1.0 会让关键词路几乎不再放行。

---

## 3. `query_vector` 为何是入参而不是内部计算

**要求**：调用方 MUST 传入查询向量，`search()` MUST NOT 自己编码。

三条理由，按重要性排序：

1. **FR-003 / S8 的 D3 裁决。** `backend/embed/model.py` 是全项目唯一的编码口径定义。检索路径若自己编码，就是**第二个编码入口** —— S8 整篇规格在防的东西。多一个入口，"口径只有一处"这个保证就没了。
2. **向量已经算好且当前被丢弃。** `backend/api/routes.py` 在 `capture_question()` 里刚算过（S8）；`capture.py` 只是没把返回值传出来。重新编码一次 = 每次提问多 100–300 ms 的 CPU 推理。
3. **可测试性。** 向量是入参，测试可以传入手工构造的向量来验证检索逻辑，不需要加载 2.27 GB 权重。

**失败时的传递约定**：调用方在编码失败时（S8 的 `capture_and_embed` 返回 `status=failed`，`vector` 为 `None`）MUST **不调用** `search()`，MUST 直接走拒答路径并留 ERROR 日志。`search()` 的 `query_vector` 类型是非可选的 `list[float]`，不承担 `None` 的分支 —— 把"向量没算出来"这个判断放在有上下文（知道失败原因）的调用方，而不是放在检索里靠猜。

---

## 4. 必须成立的不变量（可实现为自检断言）

| 编号 | 不变量 | 对应 FR |
|---|---|---|
| INV-1 | `len(result.passages) <= top_k` | FR-008 |
| INV-2 | `passages` 中的 `chunk_id` 两两不同 | FR-005 |
| INV-3 | `result.passages` 按 `rrf_score` 严格降序（同分时按 `(-cosine, chunk_id)` 稳定排序） | FR-005, SC-006 |
| INV-4 | 结果中每条 `passage.score == 该 chunk 的余弦相似度`；关键词路独有命中为 `0.0` | FR-007 |
| INV-5 | `result.is_empty == (len(passages) == 0)` | FR-012 |
| INV-6 | `result.top_score is None` ⟺ `is_empty` | `docs/05` §4.2 |
| INV-7 | `below_threshold == (not is_empty and top_score < threshold)` | FR-013 |
| INV-8 | 每条 `passage.text` 非空，且 `file_name` / `page_start` / `page_end` 非空/有效 | FR-004, 宪法原则 II |
| INV-9 | 相同入参连续调用两次，两次返回值逐字段相同 | SC-006 |
| INV-10 | `search()` 内无任何 Milvus 写操作 | 结构约束 |

**INV-6 的一处细节**：`top_score` 在"结果非空但全是关键词命中"时为 `0.0` 而不是 `None`。`None` 保留给"结果为空"。前端 `transcript.js` 对 `score` 做 `toFixed(2)`，`None` 会抛异常 —— 而 `score` 与 `top_score` 必须是同一把尺子。

---

## 5. 异常契约

```python
class RetrievalError(Exception):
    """检索基础设施失败。**不是**「检索为空」。

    ⚠️ 刻意不继承 ValueError —— 与 backend/query/QueryError、backend/index/IngestError
    同一取向：继承内建异常会让 `except ValueError` 意外捕获到它。
    """
```

| 情形 | 行为 |
|---|---|
| 检索结果为空 / 全部不达标 | **正常返回** `is_empty=True`，MUST NOT 抛异常（`docs/05` §4.2 约束 2） |
| Milvus 查询失败 | 抛 `RetrievalError` |
| 索引未构建（服务端忘了启动期构建） | 抛 `RetrievalError`，消息说明是编程错误而非数据问题 |
| `query_vector` 维度不等于 1024 | 抛 `RetrievalError`（**门禁的第二道**：S8 的指纹门禁应已在启动期挡住，这里只是兜底） |

**MUST NOT**：把 `RetrievalError` 降级成 `is_empty=True` 返回。那会让"检索坏了"表现为"知识库里没有" —— 用户会换个问法继续试，而问题在服务端。

---

## 6. 日志契约（FR-019 / FR-020）

每次检索 MUST 产生**一条** INFO 级汇总日志（失败时额外一条 ERROR）：

```
检索完成 answer_id=<id> is_empty=<bool> below_threshold=<bool>
        semantic=<n> lexical=<n> fused=<n> returned=<n>
        top_cosine=<f> top_rrf=<f> elapsed_ms=<int> state=<state>
```

| 字段 | 说明 |
|---|---|
| `semantic` / `lexical` | **两路各自的原始命中数**（去重前）。二者是融合规则有效性的唯一观测点 |
| `fused` | 去重后候选数 |
| `returned` | 最终返回条数 |
| `top_cosine` / `top_rrf` | 首条的余弦与 RRF 分（便于标定 `SIMILARITY_THRESHOLD`） |
| `state` | data-model.md §6 的六种状态之一 |

**FR-020 的落点**：`semantic` / `lexical` 为 `-1` 表示**该路执行失败**（而非无命中，后者为 `0`）。二者在日志里 MUST 可区分 —— 把"查询失败"记成"没找到"会让一次检索故障看起来像一次正常的空结果。

**MUST NOT** 记录：`query_vector` 的数值、`MILVUS_TOKEN`、片段全文（片段原文可达数千字符，且已可通过 `chunk_id` 回查）。
