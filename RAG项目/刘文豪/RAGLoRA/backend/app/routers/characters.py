# -*- coding: utf-8 -*-
"""角色路由：列表 / 详情 / 新建 / 修改 / 删除。"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core.db import get_db
from ..core.logging import get_logger
from ..deps import get_current_user
from ..seed import PROMPT_TEMPLATE
from ..services import redis_extra
from .. import models, schemas

router = APIRouter(prefix="/characters", tags=["角色"])
log = get_logger("characters")


@router.get("", response_model=list[schemas.CharacterOut], summary="角色列表（可按分类过滤）")
def list_characters(category: str | None = None, db: Session = Depends(get_db),
                    _user: models.User = Depends(get_current_user)):
    q = db.query(models.Character)
    if not category:
        return q.order_by(models.Character.id).all()

    # Redis Set 二级索引（char_cat:{category}）。返回 None 说明 Redis 不可用，
    # 回源 MySQL 过滤——功能等价，只是少了一次索引加速。
    ids = redis_extra.category_ids(category)
    if ids is None:
        return q.filter_by(category=category).order_by(models.Character.id).all()
    if not ids:
        return []
    rows = q.filter(models.Character.id.in_(ids)).all()
    # SMEMBERS 无序，按 id 排序保证响应稳定（也便于测试断言）
    return sorted(rows, key=lambda c: c.id)


# ⚠️ 该路由必须注册在 /{character_id} 之前，否则 "rank" 会被当成路径参数解析成 int 而 422。
@router.get("/rank/hot", summary="角色热度榜（Redis zSet）")
def hot_character_rank(top: int = 10, db: Session = Depends(get_db),
                       _user: models.User = Depends(get_current_user)):
    pairs = redis_extra.hot_characters(max(1, min(top, 50)))
    if not pairs:
        return []
    ids = [pid for pid, _ in pairs]
    by_id = {c.id: c for c in db.query(models.Character)
             .filter(models.Character.id.in_(ids)).all()}
    return [
        {
            "character_id": pid,
            "name": by_id[pid].name,
            "avatar": by_id[pid].avatar,
            "category": by_id[pid].category,
            "score": score,
        }
        for pid, score in pairs if pid in by_id
    ]


@router.get("/{character_id}", response_model=schemas.CharacterOut, summary="角色详情（Hash 缓存）")
def get_character(character_id: int, db: Session = Depends(get_db),
                  _user: models.User = Depends(get_current_user)):
    # read-through：先查 Redis Hash（char:{id}），未命中回源 MySQL 并回填。
    # 角色字段几乎不变，300s TTL + 写路径主动失效已足够保证一致性。
    ch = redis_extra.get_character_cached(character_id, db)
    if not ch:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "角色不存在")
    return ch


@router.post("", response_model=schemas.CharacterOut,
             status_code=status.HTTP_201_CREATED, summary="新建自定义角色")
def create_character(body: schemas.CharacterIn, db: Session = Depends(get_db),
                     _user: models.User = Depends(get_current_user)):
    if db.query(models.Character).filter_by(slug=body.slug).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "slug 已存在")

    data = body.model_dump()
    # 没给模板就套用通用骨架
    data.setdefault("prompt_template", None)
    if not data.get("prompt_template"):
        data["prompt_template"] = PROMPT_TEMPLATE

    ch = models.Character(**data, is_builtin=False)
    db.add(ch)
    db.commit()
    db.refresh(ch)
    # Redis Set 二级索引：新角色归入其分类（失败仅告警，列表查询会回源 MySQL）
    redis_extra.index_character(ch.id, ch.category)
    log.info("新建角色: %s (%s)", ch.name, ch.slug)
    return ch


@router.put("/{character_id}", response_model=schemas.CharacterOut, summary="修改角色")
def update_character(character_id: int, body: schemas.CharacterUpdateIn,
                     db: Session = Depends(get_db),
                     _user: models.User = Depends(get_current_user)):
    ch = db.get(models.Character, character_id)
    if not ch:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "角色不存在")

    old_category = ch.category
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(ch, field, value)
    db.commit()
    db.refresh(ch)
    # 写路径一致性：Hash 缓存失效 + 分类索引随 category 变更同步
    redis_extra.invalidate_character_cache(character_id)
    if old_category != ch.category:
        redis_extra.unindex_character(character_id, old_category)
        redis_extra.index_character(character_id, ch.category)
    return ch


@router.delete("/{character_id}", summary="删除角色")
def delete_character(character_id: int, db: Session = Depends(get_db),
                     _user: models.User = Depends(get_current_user)):
    ch = db.get(models.Character, character_id)
    if not ch:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "角色不存在")
    if ch.is_builtin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "内置角色不可删除")

    used = db.query(models.Conversation).filter_by(character_id=character_id, is_deleted=False).count()
    if used:
        raise HTTPException(status.HTTP_409_CONFLICT, f"该角色下还有 {used} 个会话，无法删除")

    # 先取分类再删库，随后同步清理 Redis（缓存 + 分类索引）
    category = ch.category
    db.delete(ch)
    db.commit()
    redis_extra.invalidate_character_cache(character_id)
    redis_extra.unindex_character(character_id, category)
    return {"deleted": character_id}
