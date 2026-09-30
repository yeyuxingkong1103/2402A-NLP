# -*- coding: utf-8 -*-
"""MySQL 文档结构化召回（多路召回的「路 2」）。

与另两路的分工
--------------
    路1 向量库（Milvus/Qdrant）—— **语义**相似
    路2 MySQL                 —— **文档级**结构匹配（本文件）
    路3 Neo4j                 —— **法条关系**（引用 / 相邻）

为什么需要路 2
--------------
向量检索按语义相似度排序，**不擅长「某份文档在不在库里」这类元数据问题**。
例如问「知识库里有没有《劳动法》」时：
    - 向量路可能返回一堆「看起来相关」的条文（因为有劳动合同相关内容），
      但答不出「《劳动法》本身未收录」这个事实；
    - MySQL 路直接查 `kb_documents`，**精确**判定该文档是否在库。

这正是本项目的真实短板：`qa_set.json` 中标注为「语料缺口」的题
（劳动法 / 劳动合同法）就是这类问题。

实测数据（2026-09-17）
----------------------
`kb_documents` 共 178 条：`kb_medical` 2 份 PDF 指南、`kb_legal` 176 部法律 txt。
"""
import re
import time
from datetime import datetime

from ..core.db import SessionLocal
from ..core.logging import get_logger
from ..models import KbDocument

log = get_logger("doc_store")

# 库里文档名的形态：《X》或全称。从 query 里抽候选名做匹配。
_CITE = re.compile(r"《([^》]{2,40})》")


def _norm(s: str) -> str:
    """归一化文档名：去掉常见前缀与扩展名，便于包含匹配。"""
    s = re.sub(r"\.(txt|pdf|md)$", "", s or "")
    s = s.replace("中华人民共和国", "").replace("中国", "")
    return s.strip()


def _known_names(collection: str | None = None) -> list[tuple[str, str]]:
    """库里所有文档的 (原始文件名, 归一化名)。"""
    db = SessionLocal()
    try:
        q = db.query(KbDocument.source_path)
        if collection:
            q = q.where(KbDocument.collection == collection)
        out = []
        for (p,) in q.all():
            fn = p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            out.append((fn, _norm(fn)))
        return out
    finally:
        db.close()


def _candidates(query: str, collection: str | None = None) -> list[str]:
    """从 query 里找出**库里真实存在**的文档名。

    ⚠️ 不要用「正则猜文档名」（2026-09-17 踩过）：
    初版用 `[一-龥]{2,20}(?:法|规定|...)` 抽候选，贪婪匹配把整句
    「劳动法对试用期是怎么规定」当成了一个文档名，导致 MySQL 永远查不中。

    改为**与库中真实文档名做包含匹配** —— 库里只有 178 份文档，
    逐个比对成本可忽略，且不会猜出不存在的名字。
    """
    names: list[str] = [m.group(1) for m in _CITE.finditer(query)]

    q_norm = query.replace("《", "").replace("》", "")
    for fn, nn in _known_names(collection):
        if len(nn) < 3:
            continue
        # 全名出现在 query 里，或用足够长的前缀（>=5 字）指代
        if nn in q_norm:
            names.append(fn)
            continue
        for cut in range(len(nn), 4, -1):
            if nn[:cut] in q_norm:
                names.append(fn)
                break

    seen, uniq = set(), []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def doc_lookup(query: str, collection: str | None = None,
               limit: int = 5) -> tuple[list[dict], dict]:
    """查 `kb_documents`，判断 query 提到的文档是否在库。

    返回 (hits, meta)：
        hits —— 命中的文档（**不是** chunk，因为本路回答的是「文档在不在」）
        meta —— {"in_library": bool, "matched": [...], "missing": [...],
                 "candidates": [...]}

    `meta` 的价值在于：**「没找到」本身就是一个可用的结论**
    （可以据此告诉模型「该法未收录，请勿编造」），
    而不是像向量检索那样在没收录时硬返回一堆弱相关条文。
    """
    # `_candidates` 返回的就是**库中真实存在的文档文件名**，
    # 因此这里只需按文件名把记录取出来，不必再做模糊匹配。
    candidates = _candidates(query, collection)
    if not candidates:
        return [], {"in_library": False, "matched": [], "missing": [],
                    "candidates": [], "reason": "query 中未出现库内文档名"}

    db = SessionLocal()
    try:
        q = db.query(KbDocument)
        if collection:
            q = q.where(KbDocument.collection == collection)
        docs = q.all()
        by_name = {d.source_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]: d for d in docs}

        hits, missing = [], []
        for name in candidates:
            d = by_name.get(name)
            if d is None:
                missing.append(name)
                continue
            hits.append({
                "doc_id": d.id,
                "collection": d.collection,
                "source": name,
                "chunk_count": d.chunk_count,
                "doc_type": d.doc_type,
                "status": d.status,
                "created_at": d.created_at.strftime("%Y-%m-%d") if d.created_at else None,
            })

        meta = {
            "in_library": bool(hits),
            "matched": [h["source"] for h in hits[:limit]],
            "missing": missing[:limit],
            "candidates": candidates[:limit],
            # 库里文档总数 —— 供上层在「未收录」时告知模型「库中共 N 份文档」，
            # 而不是让它去猜。刻意不塞全量名单（178 个文件名太长，会挤占 prompt）。
            "library_size": len(docs),
        }
        if hits:
            log.info("文档路命中 %d 份：%s", len(hits), meta["matched"][:3])
        else:
            log.info("文档路：候选 %s 未收录（库中共 %d 份文档）",
                     candidates[:3], len(docs))
        return hits, meta
    finally:
        db.close()


def library_summary(collection: str | None = None) -> dict:
    """知识库概览：文档数、块数、按类型分布。供检索调试台与健康检查使用。"""
    db = SessionLocal()
    try:
        q = db.query(KbDocument)
        if collection:
            q = q.where(KbDocument.collection == collection)
        docs = q.all()
        by_coll: dict[str, dict] = {}
        for d in docs:
            e = by_coll.setdefault(d.collection, {"docs": 0, "chunks": 0, "types": {}})
            e["docs"] += 1
            e["chunks"] += d.chunk_count or 0
            e["types"][d.doc_type] = e["types"].get(d.doc_type, 0) + 1
        return {"total_docs": len(docs),
                "total_chunks": sum(d.chunk_count or 0 for d in docs),
                "by_collection": by_coll}
    finally:
        db.close()
