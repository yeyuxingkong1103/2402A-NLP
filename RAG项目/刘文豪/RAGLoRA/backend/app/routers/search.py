# -*- coding: utf-8 -*-
"""检索调试台路由：纯检索（不生成），返回完整召回/精排明细。"""
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core import config
from ..core.db import get_db
from ..core.logging import get_logger
from ..deps import get_current_user
from ..services import rerank, retrieval
from .. import models, schemas

router = APIRouter(prefix="/search", tags=["检索"])
log = get_logger("search")


@router.post("", summary="混合检索（可含精排）")
def do_search(body: schemas.SearchIn, db: Session = Depends(get_db),
              user: models.User = Depends(get_current_user)):
    t0 = time.time()

    hits, collections = retrieval.search(
        body.query, collection=body.collection,
        recall_k=body.recall_k, filters=body.filters,
    )
    recall_count = len(hits)

    t = time.time()
    if body.use_rerank and hits:
        hits = rerank.rerank(body.query, hits, body.top_k)
    else:
        hits = hits[:body.top_k]
    rerank_ms = round((time.time() - t) * 1000)

    elapsed_ms = round((time.time() - t0) * 1000)
    result = {
        "query": body.query,
        "collection": body.collection or "(全部)",
        "collections_searched": collections,
        "recall_count": recall_count,
        "used_rerank": body.use_rerank and rerank.is_available(),
        "rerank_ms": rerank_ms,
        "elapsed_ms": elapsed_ms,
        "hits": [{
            **h,
            "source_label": retrieval.format_source(h),
        } for h in hits],
    }

    # 存历史（失败不影响检索结果）
    try:
        db.add(models.SearchHistory(
            user_id=user.id, query=body.query[:512],
            collection=body.collection, top_k=body.top_k,
            use_rerank=body.use_rerank, result_json=result, elapsed_ms=elapsed_ms,
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        log.warning("检索历史保存失败: %s", e)

    return result


@router.get("/collections", response_model=list[schemas.CollectionOut], summary="集合列表")
def list_collections(db: Session = Depends(get_db),
                     _user: models.User = Depends(get_current_user)):
    from ..services import ingest
    bindings: dict[str, list[str]] = {}
    for ch in db.query(models.Character).all():
        bindings.setdefault(ch.kb_collection, []).append(ch.name)
    return [
        schemas.CollectionOut(name=s["name"], points=s["points"],
                              characters=bindings.get(s["name"], []))
        for s in ingest.collection_stats()
    ]


@router.get("/history", summary="我的检索历史")
def list_history(limit: int = 50, db: Session = Depends(get_db),
                 user: models.User = Depends(get_current_user)):
    rows = (db.query(models.SearchHistory)
            .filter_by(user_id=user.id)
            .order_by(models.SearchHistory.id.desc())
            .limit(limit).all())
    return [{
        "id": r.id, "query": r.query, "collection": r.collection,
        "top_k": r.top_k, "use_rerank": r.use_rerank,
        "elapsed_ms": r.elapsed_ms, "created_at": r.created_at,
        "result": r.result_json,
    } for r in rows]


@router.get("/history/{history_id}", summary="检索历史详情")
def get_history(history_id: int, db: Session = Depends(get_db),
                user: models.User = Depends(get_current_user)):
    r = db.get(models.SearchHistory, history_id)
    if not r or r.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "记录不存在")
    return {"id": r.id, "query": r.query, "created_at": r.created_at, "result": r.result_json}


@router.delete("/history/{history_id}", summary="删除检索历史")
def delete_history(history_id: int, db: Session = Depends(get_db),
                   user: models.User = Depends(get_current_user)):
    r = db.get(models.SearchHistory, history_id)
    if not r or r.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "记录不存在")
    db.delete(r)
    db.commit()
    return {"deleted": history_id}
