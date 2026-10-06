# 契约：与既有模块的接缝

**Feature**: `008-hybrid-retrieval` | **Version**: 1.0 | **Date**: 2026-09-28

本特性只改动四处既有文件，全部集中在 `backend/api/` 与 `backend/serve.py`。**前端零改动是硬验收项（SC-008）。**

---

## 1. 改动总览

| 文件 | 改动 | 风险 |
|---|---|---|
| `backend/api/capture.py` | `capture_question()` 的返回类型由 `None` 改为 `list[float] \| None` | 低 —— 唯一调用方是 `routes.py` |
| `backend/api/routes.py` | 接收返回值并透传给 `stream_answer()` | 低 |
| `backend/api/stream.py` | 在 `status` 帧之后、`citations` 帧之前执行检索 | 中 —— 事件顺序是硬契约 |
| `backend/api/__init__.py` | 新增 `RETRIEVAL_READY_NOTICE` 文案 | 低 |
| `backend/api/config.py` | 检索组配置转正为必需项；新增 5 个配置字段 | **中** —— 启动失败条件变严 |
| `backend/serve.py` | 启动序列插入索引构建 | **中** —— 启动时间增加、失败条件增加 |
| `frontend/**` | **零改动** | — |

---

## 2. `capture.py` —— 回传查询向量（R8）

### 改动

```python
# 改前
def capture_question(question: str, answer_id: str) -> None:

# 改后
def capture_question(question: str, answer_id: str) -> list[float] | None:
    """留存一次提问，并返回本次提问的查询向量（编码失败时为 None）。

    ⚠️ 与 R8 的关系：检索（I-05）需要查询向量，而它这里刚算过。
    让调用方重新编码一次是第二编码入口（FR-003）+ 每次提问多 100–300 ms。
    本函数只改返回值，契约「MUST NOT 向调用方抛异常」不变。
    """
```

### 保持不变的契约

- **MUST NOT 向调用方抛异常** —— 这个性质**不变**。失败时返回 `None` 并留 WARNING 日志（既有行为）。
- 失败路径仍按 `status=failed` 落盘，仍可被 `backend/query_embed.py` 的 CLI 重跑。
- `logger.warning("capture_degraded ...")` 保留。

### 新增的返回路径

| 情形 | 返回 |
|---|---|
| 编码成功 | `record["vector"]`（1024 个 float） |
| 编码失败 / 模型不可用 | `None`（同时已有 WARNING 日志） |
| 落盘失败但编码成功 | **仍然返回向量** —— 落盘失败不该让检索也失败（两者是独立的失败域） |

> 最后一条是刻意的：留存的目的是"可回看"，检索的目的是"当场回答"。落盘失败时用户仍应得到检索结果。

---

## 3. `routes.py` —— 透传

```python
question = validate_question(payload.question)
answer_id = str(uuid.uuid4())
logger.info(...)

# 留存 + 拿到查询向量。同一行做两件事：它们本来就是同一次编码的产物。
query_vector = capture_question(question, answer_id)

return StreamingResponse(
    stream_answer(
        question=question,
        answer_id=answer_id,
        config=config,
        query_vector=query_vector,          # 新增
    ),
    media_type="text/event-stream",
    headers=_SSE_HEADERS,
)
```

**`_SSE_HEADERS` 三条头 MUST 保持不变**（`Cache-Control: no-cache` / `X-Accel-Buffering: no` / `Connection: keep-alive`）—— 它们的理由见 `routes.py` 的既有注释，本特性不触碰。

---

## 4. `stream.py` —— 检索插入点

### 事件顺序（硬契约，MUST 保持）

```
status → citations → token* → done
```

**本特性只改 `citations` 与 `token` 的**内容**，不改事件顺序。**

### 改动后的生成器骨架

```python
async def stream_answer(*, question, answer_id, config, query_vector) -> AsyncIterator[str]:
    completed = False
    try:
        yield sse_event(EV_STATUS, {"answer_id": answer_id, "state": STATE_ACCEPTED})

        # ── 新增：检索 ──────────────────────────────────────
        retrieval = None
        if query_vector is None:
            # S8 的编码失败。MUST NOT 静默退化为只跑关键词路（FR-020）。
            logger.error("retrieval_skipped answer_id=%s reason=query_vector_unavailable", answer_id)
        else:
            try:
                retrieval = service.search(
                    question=question,
                    query_vector=query_vector,
                    top_k=config.top_k,
                    threshold=config.similarity_threshold,
                )
            except RetrievalError as exc:
                # 不冒泡：流已经开始，异常无法变成 HTTP 状态码（R9）。
                # 留 ERROR 日志使其可查 —— 这不是"吞掉"。
                logger.error("retrieval_failed answer_id=%s reason=%s", answer_id, exc)

        citations = _citation_payload(retrieval)     # 无结果时为 []
        yield sse_event(EV_CITATIONS, {"citations": citations})

        chunks = _body_chunks(config, retrieval)     # 见 §5
        for chunk in chunks:
            yield sse_event(EV_TOKEN, {"text": chunk})

        body_text = "".join(chunks)
        answer_text = body_text + ANSWER_JOINER + DISCLAIMER
        envelope = not_ready_response(answer_id, answer_text, DISCLAIMER)

        completed = True
        logger.info(
            "回答完成 answer_id=%s is_refusal=%s citations=%d",
            answer_id, envelope.is_refusal, len(envelope.citations),
        )
        yield sse_event(EV_DONE, envelope.model_dump())

    except (asyncio.CancelledError, GeneratorExit) as exc:
        # 既有逻辑不变（见 stream.py 的原始注释：两条取消路径都必须捕、
        # 都必须重新抛出、只在未完成时记 client_disconnected）
        ...
        raise
```

### 必须保持不变的三件事

1. **`status` 永远是第一个事件**，且在检索之前发出 —— 流被掐断时终帧到不了，`answer_id` 若只在终帧，这次提问就失去了可关联标识（S7 research R10）。
2. **`completed` 的置位时机与 `except` 分支**：既有注释详细解释了 `asyncio.CancelledError` 与 `GeneratorExit` 两条取消路径、以及"已完成后被收尾不算断连"。本特性 MUST NOT 改动这段。
3. **`citations` 事件即使为空也照发** —— 省略会让前端的空态分支从"本特性起第一次执行"变成"永远不执行"。

---

## 5. `_body_chunks` 的改动（D3 的落点）

```python
def _body_chunks(config: AppConfig, retrieval: RetrievalResult | None) -> list[str]:
    chunks: list[str] = []
    if config.test_preamble:                # 既有故障注入，位置不变
        chunks.append(config.test_preamble)
        chunks.append(ANSWER_JOINER)

    if retrieval is not None and not retrieval.is_empty:
        chunks.append(RETRIEVAL_READY_NOTICE)   # D3
    else:
        chunks.append(REFUSAL_FALLBACK)         # FR-014

    return chunks
```

**`CAPABILITY_NOT_READY` 的去留**：本特性接入后，它**不再出现在 `_body_chunks` 里**（检索已就绪，不该再说"向量检索能力正在接入中"）。它**保留在 `__init__.py` 中不删**，理由：它是 S7 规格的产物，其存在意义是"能力未就绪"这个状态的唯一文案；未来若检索被临时禁用，它是现成的正确文案。删除会让 S7 的规格与代码对不上。

### 三条路径的用户可见文案

| 路径 | 正文 | `is_refusal` |
|---|---|---|
| 检索命中或靠关键词纳入 | `RETRIEVAL_READY_NOTICE` | `True`（未产出知识性内容） |
| 检索为空 / 全部不达标 / 基础设施失败 | `REFUSAL_FALLBACK` | `True` |
| 既有的注入测试（`test_preamble`） | 注入文本 + 上述之一 | 同上 |

> ⚠️ `is_refusal` 同为 `True` 但**含义不同**（"生成未就绪"vs"知识库没有"），这正是 S7 在 `CAPABILITY_NOT_READY` 的注释里已经确立的区分逻辑，本特性沿用同一取向。

### `RETRIEVAL_READY_NOTICE` 的措辞要求

MUST 满足（constitution 原则 V + D3）：

- 传达"**找到了资料**"—— 否则用户会以为是没找到。
- 传达"**生成能力未就绪**"—— 否则用户会以为这是最终答案。
- 引导用户**看下方的引用原文** —— 引用是本特性真实交付的东西。
- MUST NOT 含任何诊断、剂量、药品、机构建议。
- **MUST 带有明确的移除时机注释**（生成模块 I-06 接入时删除）。

参考措辞（plan 实现时定稿，此处只锁语义）：

> 「已找到与你的问题相关的资料，请阅读下方引用原文。答案生成能力正在接入中，本期暂不能给出成文回答。」

---

## 6. `__init__.py` —— 文案唯一处

新增：

```python
# 检索已就绪、生成未就绪的过渡文案（S9 / D3）。
#
# ⚠️ 生命周期：生成模块（I-06）接入时 MUST 删除本常量与它的使用点。
#    它的存在前提是"检索通了但没生成"这个中间态，那个状态会随 I-06 落地消失。
#
# 为什么不复用 REFUSAL_FALLBACK：后者表示"知识库里没有"，前者表示"有，但生成没做完"。
# 二者对用户的后续动作完全不同（换问法 vs 等能力上线）。
# docs/05 §3.1.4 规则 3 明令禁止把"本该能答但基础设施没就绪"伪装成拒答。
RETRIEVAL_READY_NOTICE = "…"
```

**这个文件是用户可见文案的唯一处**（既有约束）。新增文案 MUST 只加在这里，MUST NOT 写在 `stream.py` 或 `retrieve/` 包内。

---

## 7. `config.py` —— 配置转正

```python
# 改前
REQUIRED_NOW: tuple[str, ...] = ()
REQUIRED_WHEN_RETRIEVAL_LANDS: tuple[str, ...] = (
    "milvus_uri", "milvus_collection", "embed_model_path",
    "similarity_threshold", "top_k",
)

# 改后
REQUIRED_NOW: tuple[str, ...] = (
    "milvus_uri", "milvus_collection", "embed_model_path",
    "similarity_threshold", "top_k",
)
```

这正是 S7 规格预先标好的转正时机（该元组的注释写着"转正时机：检索模块（I-05）接入时"），调用方（`serve.py`、未来的 `/health`）一行都不用改。

**新增字段**（data-model.md §4）：`retrieval_candidates` / `rrf_k` / `lexical_admit_rank` / `bm25_k1` / `bm25_b`，均带默认值，因此不进入必需项。

**校验**：新增一个 `validate_retrieval_config()`（或并入现有校验路径），规则见 data-model.md §4。不合法即启动失败并说明**哪一项、什么范围** —— MUST NOT 回显 `MILVUS_TOKEN` 的值（constitution 原则 III）。

---

## 8. `serve.py` —— 启动序列

```
① 读配置（缺必需项 → 退出码 2）
② 指纹门禁 gate.check_gate()（最便宜的检查最先，S8 既有取向）
③ 加载 BGE-M3 权重（约 10 s，S8 既有）
④ 【新增】构建 BM25 索引  ← 连 Milvus 拉语料 + 分词 + 预热 jieba
⑤ 打印就绪信息，起 HTTP
```

**为什么 ④ 在 ③ 之后**：③ 是既有步骤且最慢；④ 需要 Milvus，两者无依赖。放在后面只是不改动既有顺序，减少 diff。**若将来想缩短启动时间，④ 可以和 ③ 并行**，但那需要线程安全的前提验证（R12），不在本特性范围。

**启动输出 MUST 包含**（否则"关键词路悄悄是空的"这件事在启动时不可见）：
- 语料条数，以及是否与 `index_manifest.json` 的 `total_chunks` 一致
- 分词器预热耗时
- 检索配置的生效值（`top_k` / `threshold` / `candidates` / `rrf_k` / `lexical_admit_rank`）

**失败处置**：

| 失败点 | 退出码 | 提示 |
|---|---|---|
| Milvus 不可达 | 3（沿用 `EXIT_MODEL` 一类的依赖失败语义；实现时与既有常量对齐） | 指出 `MILVUS_URI` 与 `docker ps` 自检命令 |
| 语料为空（拉回 0 条） | 3 | 指出需先运行 S6 入库 |
| 与 manifest 的 `total_chunks` 不符 | **告警但不退出** —— 库可能刚被重跑过而 manifest 是旧的，此时"照常启动 + 明确告警"比"拒绝启动"更有用 |

### S8 行为变更（必须写进启动提示与 quickstart）

**Milvus 从"可选"变为"启动的硬前提"。** 当前 `backend/serve.py` 在 Milvus 未运行时也能正常启动（门禁只读 `index_manifest.json`）。接入本特性后不能 —— 语料构建在启动期完成（FR-010），而"启动成功但每次检索降级"被 spec 的 Edge Case 明令禁止。

---

## 9. 前端：MUST 保持零改动（SC-008）

前端**一行都不改**，因为三处接缝已经就位：

| 前端文件 | 已就位的能力 | 本特性是否触发 |
|---|---|---|
| `frontend/js/sse.js` | `dispatchEvent_` 已处理 `citations` 事件名 | 会 —— 从"恒定空数组"变成真实数据 |
| `frontend/js/transcript.js` | `renderCitations` 已渲染 `file_name` / `page_start` / `page_end` / `score` / `text`，含预览与展开 | 会 |
| `frontend/js/transcript.js` | `renderDone` 已用终帧 `answer_text` 校正流式拼接结果 | 会 |

**若实现过程中发现需要改前端，MUST 改服务端而非前端**（FR-018）。判据：服务端产出的 `citation` 对象形状偏离了 `docs/05` §3.1.3。前端是这个契约的**执行方**，不是它的协商方。

**SC-008 的验证方式**：用 `git diff --stat frontend/` 断言无输出。若本仓库未纳入版本管理，则用文件 mtime 或改动前后逐字节比对。
