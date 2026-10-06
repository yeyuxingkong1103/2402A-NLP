"""子块 → 父块回填。

存在的理由（FR-3.5 / AC-8）：dense 与 sparse 都只召回子块（款/项），
而生成层要的是**整条法条**——半条原文会让模型补全，补全就是幻觉。
父块与子块同存于 law_chunks 集合（②期设计决定，不用 docstore），
所以回填就是一次 query，不必二次落库。

本模块不导入任何模型代码，也不碰 LangChain。
"""
from __future__ import annotations

from app.db.milvus import COLLECTION, OUTPUT_FIELDS

# 父块要取回的字段：OUTPUT_FIELDS 已覆盖 text/path/article_no/chunk_type，
# 所以这行去重拼接的结果与 OUTPUT_FIELDS 完全相同——保留该写法是防御：
# 将来 OUTPUT_FIELDS 若收窄掉 chunk_type，父块仍能拿到断言身份所需的字段
BACKFILL_FIELDS = list(dict.fromkeys(OUTPUT_FIELDS + ["chunk_type"]))

# 一次 IN 查询的 id 上限。粗排 top50 最多 50 个父块，取 200 留余量；
# 真超了说明上层传错了东西，宁可分批也不要让 Milvus 拒掉整条查询
QUERY_BATCH = 200


def parent_ids_of(hits: list[dict]) -> list[str]:
    """按出现顺序取出去重后的 parent_id，跳过空值。"""
    ids: list[str] = []
    for hit in hits:
        pid = hit.get("parent_id")
        if pid and pid not in ids:
            ids.append(pid)
    return ids


def backfill_parents(client, hits: list[dict],
                     collection: str = COLLECTION) -> list[dict]:
    """取回父块，按各自子块在 hits 中最早出现的位置排序。

    排序只依赖粗排顺序、不需要 RRF 分数：hybrid_search 返回的 hit 里没有
    distance（②期只留了 entity），而"取最高子块分"与"取最早出现的子块"
    给出的父块顺序相同——都是"哪个父块的证据先被召回"。
    """
    ids = parent_ids_of(hits)
    if not ids:
        return []
    rows: list[dict] = []
    for start in range(0, len(ids), QUERY_BATCH):
        chunk = ids[start:start + QUERY_BATCH]
        expr = "chunk_id in [" + ", ".join(f'"{cid}"' for cid in chunk) + "]"
        rows.extend(client.query(collection, filter=expr,
                                 output_fields=BACKFILL_FIELDS))
    by_id = {row["chunk_id"]: row for row in rows}
    blocks: list[dict] = []
    # 按 ids 的顺序（= 子块最早出现顺序）产出，父块在库里对不上就说明
    # 子块指向了不存在的父块，这时跳过比塞个半成品进下游安全
    for order, pid in enumerate(ids):
        row = by_id.get(pid)
        if row is None:
            continue
        block = dict(row)
        block["source"] = "vector"
        block["rank_index"] = order
        block["rerank_score"] = None
        blocks.append(block)
    return blocks
