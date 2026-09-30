"""入库链路的公共一段：分好块的文本 → （可选）摘要 → 向量化 → 去重 → 写库。

为什么单独抽出来：这条链路此前在三个地方各写一遍（`/knowledge/upload`、
`scripts/ingest.py`、`scripts/seed.py`），而且已经漂移过——seed 那份没有摘要、
没有低质量过滤、没有 chunk 去重，改一处不会同步到另两处。

**只抽真正相同的中段**。三处差异都在两头，仍留在各自调用方，因为它们是有语义的：
- 解析来源不同：HTTP 上传的字节 vs 磁盘文件；
- 分块参数不同：接口按 query 传参、CLI 按 flag、seed 用固定值；
- 去重策略不同：接口不做文档级去重（同 source 重传就是再灌一遍）、CLI 按已登记
  source 跳过（`--re-ingest` 才重灌）、seed 每次都先删旧向量；
- 后置动作不同：接口要 `retriever.invalidate` 让 BM25 缓存失效、CLI 是独立进程
  没有缓存可失效、seed 还要顺手清掉关系库里的重复登记。

这里刻意分成**两步**（build_chunks / store_chunks）而不是一个函数：中间要插
"删旧数据"这个破坏性动作，而它必须等摘要与向量化（会联网、会失败）成功之后再执行。
"""
from __future__ import annotations

from app.core.document.enhance import chunk_hash, summarize_chunks
from app.core.logging_config import get_logger

log = get_logger("ingest")


def build_chunks(
    texts: list[str],
    *,
    source: str,
    title: str,
    embedding,
    summary: bool = False,
    llm=None,
    dedup: bool = True,
) -> list[dict]:
    """分好块的文本 → chunk 记录（含向量）。**这一步会调 LLM/Embedding，可能失败。**

    刻意与 store_chunks 分开：重灌流程里"删旧数据"是破坏性的，必须等这一步
    （摘要 + 向量化）全部成功之后再删——否则 Ollama 一抖动就成了"旧的删了、新的没写，
    文档两边一起消失"（实测过：milvus 2→0、sql 1→0）。
    """
    # 摘要与向量都在这一步做完，且**都不 catch**：这里抛异常时调用方还没删旧数据，
    # 库里保持原样。这正是「先算后删」的意义所在，别在这里加兜底吞异常。
    # 两个列表都按下标与 texts 对齐（summarize_chunks 的等长契约见 enhance.py）。
    summaries = summarize_chunks(llm, texts) if summary else ["" for _ in texts]
    # 整篇一次交给 Embedding 侧，由它按 embed_batch_size 分批发请求：逐条调是 n 次网络往返，
    # 而整批一个请求又会被服务端上限打挂（实测见 embedding.embed_texts 的注释）。它返回条数
    # 不符时会抛 RuntimeError，挡掉下面 zip 静默丢尾块。
    vectors = embedding.embed_texts(texts)

    seen: set[str] = set()
    chunks = []
    for i, (t, v) in enumerate(zip(texts, vectors)):
        fp = chunk_hash(t)
        if dedup and fp in seen:
            continue  # 同批内内容重复只留一份
        seen.add(fp)
        chunks.append(
            {
                "text": t,
                "title": title,
                "source": source,
                # 用的是去重**之前**的下标，所以去重生效时 chunk_index 会跳号。
                # 它只表示「在原文档里的第几块」，不要求连续。
                "chunk_index": i,
                "summary": summaries[i] if summary else "",
                "vector": v,
            }
        )
    return chunks


def store_chunks(chunks: list[dict], *, role_id: str, source: str, title: str, milvus, sql) -> int:
    """写向量库 + 登记关系库，返回写入的 chunk 数。

    两步都做完才算入库成功；登记失败时**补偿删除**刚写的向量——否则会留下
    "检索能召回、列表里看不到"的孤儿数据，再跑一次 ingest 还会重复灌一份。
    （补偿按 (role, source) 删：正常路径下这个 source 的旧向量此前已按需清理，
    删多了也只是把它恢复到"这条 source 没东西"的状态。）

    返回值是**本次提交的条数**，不是向量库回执核对出来的结果（milvus.insert 返回的 id
    列表在这里没被使用）；上传接口回给前端的 chunk_count、日志里的 N 块都是它。
    """
    # 顺序固定：先写向量（检索能用的那份），后登记关系库（列表/去重判据用的那份）。
    # 顺序反过来的话，登记成功而向量写入失败就会留下「列表里有、检索查不到」的空文档，
    # 而且因为已登记，下次 seed/ingest 还会把它当成已入库跳过。
    milvus.insert(role_id, chunks)
    try:
        sql.register_document(role_id, source, title, len(chunks))
    except Exception as exc:  # noqa: BLE001
        log.error(
            "登记关系库失败，回滚刚写入的 %d 条向量（role=%s source=%s）: %s",
            len(chunks), role_id, source, exc,
        )
        try:
            milvus.delete_by_source(role_id, source)
        except Exception as rollback_exc:  # noqa: BLE001
            log.error("回滚也失败了，库里可能留有孤儿向量: %s", rollback_exc)
        raise
    return len(chunks)
