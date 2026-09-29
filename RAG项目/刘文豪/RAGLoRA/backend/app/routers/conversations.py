# -*- coding: utf-8 -*-
"""会话路由：会话 CRUD 与历史消息。

多用户隔离：每个接口都校验 conversation.user_id == 当前用户，
不满足一律 404（而非 403，避免暴露「该会话存在」这一信息）。
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..core.db import get_db
from ..core.logging import get_logger
from ..deps import get_current_user
from ..services import memory as memory_svc
from .. import models, schemas

router = APIRouter(prefix="/conversations", tags=["会话"])
log = get_logger("conversations")


def _owned(conv_id: int, user: models.User, db: Session) -> models.Conversation:
    conv = db.get(models.Conversation, conv_id)
    if not conv or conv.user_id != user.id or conv.is_deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")
    return conv


@router.get("", response_model=list[schemas.ConversationOut], summary="我的会话列表")
def list_conversations(character_id: int | None = None,
                       db: Session = Depends(get_db),
                       user: models.User = Depends(get_current_user)):
    q = db.query(models.Conversation).filter_by(user_id=user.id, is_deleted=False)
    if character_id:
        q = q.filter_by(character_id=character_id)
    # 注意：不能用 nullslast()——它生成的 `NULLS LAST` 是 PostgreSQL 语法，
    # MySQL 不支持（会报 1064 语法错误）。MySQL 中 DESC 本身就把 NULL 排在最后。
    return q.order_by(models.Conversation.last_message_at.desc(),
                      models.Conversation.id.desc()).all()


@router.post("", response_model=schemas.ConversationOut,
             status_code=status.HTTP_201_CREATED, summary="新建会话")
def create_conversation(body: schemas.ConversationCreateIn,
                        db: Session = Depends(get_db),
                        user: models.User = Depends(get_current_user)):
    character = db.get(models.Character, body.character_id)
    if not character:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "角色不存在")

    conv = models.Conversation(
        user_id=user.id,
        character_id=body.character_id,
        title=body.title or f"与{character.name}的对话",
        last_message_at=datetime.now(),
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    log.info("新建会话 id=%d user=%d character=%s", conv.id, user.id, character.slug)
    return conv


@router.get("/{conv_id}/messages", response_model=list[schemas.MessageOut], summary="历史消息")
def list_messages(conv_id: int,
                  limit: int = Query(default=50, ge=1, le=200),
                  before_id: int | None = None,
                  db: Session = Depends(get_db),
                  user: models.User = Depends(get_current_user)):
    conv = _owned(conv_id, user, db)
    q = db.query(models.Message).filter_by(conversation_id=conv.id)
    if before_id:
        q = q.filter(models.Message.id < before_id)
    rows = q.order_by(models.Message.id.desc()).limit(limit).all()
    return list(reversed(rows))


@router.patch("/{conv_id}", response_model=schemas.ConversationOut, summary="重命名会话")
def rename_conversation(conv_id: int, body: schemas.ConversationRenameIn,
                        db: Session = Depends(get_db),
                        user: models.User = Depends(get_current_user)):
    conv = _owned(conv_id, user, db)
    conv.title = body.title
    db.commit()
    db.refresh(conv)
    return conv


@router.delete("/{conv_id}", summary="删除会话（软删除）")
def delete_conversation(conv_id: int, db: Session = Depends(get_db),
                        user: models.User = Depends(get_current_user)):
    conv = _owned(conv_id, user, db)
    conv.is_deleted = True
    db.commit()
    memory_svc.clear(user.id, conv.id)
    return {"deleted": conv_id}


@router.get("/{conv_id}/memory", summary="查看该会话的短期记忆（Redis）")
def get_memory(conv_id: int, db: Session = Depends(get_db),
               user: models.User = Depends(get_current_user)):
    conv = _owned(conv_id, user, db)
    return {
        "redis_available": memory_svc.is_available(),
        "turns_kept": 6,
        "ttl_seconds": 3600,
        "redis": memory_svc.get_recent(user.id, conv.id),
        "from_mysql": memory_svc.rebuild_from_db(user.id, conv.id, db),
    }
