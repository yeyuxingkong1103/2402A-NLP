#!/usr/bin/env python
# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单06 - 混合检索（向量检索 / 全文检索 / 混合检索）
"""
重建 Milvus 稀疏向量：TF → **全局 IDF 加权的 TF-IDF**。

【为什么必须重建，而不是"以后再说"】
两件独立的事都要求重算：
1. **工单01 写入的稀疏向量是无效的** —— 它用 Python 内置 `hash()` 做 token→维度映射，
   而 `hash()` 对字符串**逐进程随机化**。入库进程与查询进程算出的维度对不上，
   Milvus 稀疏检索会**静默失效**（不报错、召回率≈0）。已改为
   `text_analysis.term_id`（crc32，内容稳定）。
2. **原先只有 TF 没有 IDF** —— 入库是逐文档写的，算不出全局 DF。
   现在全量拉一遍算出 DF，把 IDF 乘进去。这就是工单01 注释里承诺的
   「工单06 再把 IDF 补上」。

【为什么是 delete + insert 而不是 upsert】
建表时用的是 `auto_id=True`，而 Milvus **不支持** auto_id 集合的 upsert
（upsert 由 insert+delete 实现，禁止改主键列）。
所以只能：全量拉出（含 dense 向量）→ 重算 sparse → 删光 → 重插。
**主键会全变**。本项目主键（chunk_id）只用于单次请求内的去重与邻块标记，
没有任何持久化引用，所以安全 —— 这一点由 preflight 的行数校验兜底。

用法：
    PY=~/rag-data/venv/bin/python
    $PY scripts/rebuild_sparse.py            # 默认 collection
    $PY scripts/rebuild_sparse.py --dry-run  # 只看 DF 统计，不写库
    $PY scripts/rebuild_sparse.py --collection rag_chunks_bgelarge
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.text_analysis import term_id, tokenize          # noqa: E402
from app.core.vectorstore import VectorStore, lexical_sparse   # noqa: E402


def build_idf(contents: list[str]) -> tuple[dict[int, float], Counter]:
    """全局 DF → IDF。IDF 用 BM25 的平滑式，保证非负。"""
    df: Counter = Counter()
    for text in contents:
        df.update(set(tokenize(text)))
    n = len(contents) or 1
    idf = {term_id(t): math.log((n - c + 0.5) / (c + 0.5) + 1.0)
           for t, c in df.items()}
    return idf, df


def main() -> int:
    ap = argparse.ArgumentParser(description="重建 Milvus 稀疏向量（TF → TF-IDF）")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    store = VectorStore(collection=args.collection) if args.collection else VectorStore()
    ok, msg = store.health()
    if not ok:
        print(f"Milvus 不可用：{msg}")
        return 2
    name = store.collection

    t0 = time.perf_counter()
    rows = store.fetch_all(with_vectors=True)
    print(f"[{name}] 拉全量 {len(rows)} 行（含 dense/sparse），耗时 {time.perf_counter()-t0:.2f}s")
    if not rows:
        print("库里没有数据，先跑 scripts/ingest_all.py")
        return 1

    idf, df = build_idf([r.get("content") or "" for r in rows])
    print(f"全局 DF：{len(df)} 个词条；IDF 范围 "
          f"[{min(idf.values()):.3f}, {max(idf.values()):.3f}]")

    old_sparse_nonempty = sum(1 for r in rows if r.get("sparse"))
    print(f"现有 sparse 字段非空的行：{old_sparse_nonempty}/{len(rows)}"
          f"（这些是工单01 用随机 hash 写的，维度对不上，必须重算）")

    if args.dry_run:
        print("--dry-run：只统计，不写库。")
        return 0

    def _strip_id(rs: list[dict]) -> list[dict]:
        """插入前必须去掉 `id`：集合是 auto_id，主键由 Milvus 生成，
        带 `id` 会被判为 "unexpected field" 而**整批失败**（实测踩过）。"""
        return [{k: v for k, v in r.items() if k != "id"} for r in rs]

    old_rows = _strip_id(rows)          # 原样保留一份，用于失败回滚
    new_rows = []
    for r in rows:
        row = {k: v for k, v in r.items() if k != "id"}
        row["sparse"] = lexical_sparse(r.get("content") or "", idf=idf)
        new_rows.append(row)

    def _insert_all(rs: list[dict]) -> int:
        n = 0
        for i in range(0, len(rs), args.batch):
            res = store.client.insert(collection_name=name, data=rs[i:i + args.batch])
            n += int(res.get("insert_count", 0))
        store.flush()
        return n

    t = time.perf_counter()
    deleted = store.client.delete(collection_name=name, filter="id >= 0")
    print(f"已删除 {deleted.get('delete_count', '?')} 行，耗时 {time.perf_counter()-t:.2f}s")

    t = time.perf_counter()
    try:
        total = _insert_all(new_rows)
    except Exception as e:  # noqa: BLE001
        # 【这一步是必须的】delete 已经生效、回不去了 —— 若插入再失败，
        # 整个知识库就空了。所以必须**立刻把原始行原样插回去**再抛出。
        # 实测踩过：因为没去掉 `id` 字段，整批插入被 Milvus 拒绝，
        # 集合当场变空（还得重新入库）。
        print(f"!! 插入失败：{e}\n   正在回滚（把原始行原样插回）……")
        restored = _insert_all(old_rows)
        print(f"   已回滚 {restored} 行")
        raise
    print(f"已重插 {total} 行（含原 dense 向量 + 新 TF-IDF sparse），"
          f"耗时 {time.perf_counter()-t:.2f}s")

    # 【必须用 query 数行，不能用 store.count()】count() 读的是 collection stats，
    # 会把**已删除但尚未 compaction** 的行也算进去 —— 实测刚重建完 count() 报 3696
    # 而真实只有 1848，会打出一条吓人的**假警报**。query 尊重删除标记，才是真实行数。
    verify = store.client.query(collection_name=name, filter="id >= 0",
                                output_fields=["id"], limit=16384)
    after = len(verify)
    print(f"重建后行数：{after}（用 query 核实；count() 会因未 compact 的已删行虚高）"
          + ("" if after == len(rows) else f"  ⚠️ 与重建前 {len(rows)} 不一致，请检查"))
    print("注意：主键已全部更换（auto_id 集合不能用 upsert）。"
          "chunk_id 只用于单次请求内的去重，未被持久化引用。")
    return 0 if after == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
