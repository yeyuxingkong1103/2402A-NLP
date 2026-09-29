# app/api/role.py
"""角色与会话接口。

角色是对话的「人设来源」，会话是对话的「容器」，二者都归属于
同一个用户与角色的组合，放在一起便于对照维护。
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.core.memory_service import get_memory
from app.core.role_service import RoleService
from app.core.session_service import SessionService
from app.db import get_db, redis_conn
from app.schemas import Resp

roles_router = APIRouter(prefix="/api/roles", tags=["角色"])
sessions_router = APIRouter(prefix="/api/sessions", tags=["会话"])


# ==================== 角色 ====================
@roles_router.get("", response_model=Resp, summary="角色列表")
async def list_roles(db: Session = Depends(get_db)):
    roles = RoleService.list_roles(db, only_active=True)
    return Resp(data=[r.to_dict() for r in roles])


@roles_router.get("/hot", response_model=Resp, summary="角色热度排行")
async def hot_roles(k: int = 10):
    return Resp(data=[{"role_key": r, "score": s}
                      for r, s in redis_conn.top_roles(k)])


@roles_router.get("/{role_key}", response_model=Resp, summary="角色详情（含人设）")
async def get_role(role_key: str, db: Session = Depends(get_db)):
    role = RoleService.get_by_key(db, role_key)
    if role is None:
        raise HTTPException(status_code=404, detail="角色不存在: %s" % role_key)
    return Resp(data=role.to_dict(with_prompt=True))


# ==================== 会话 ====================
@sessions_router.get("", response_model=Resp, summary="会话列表")
async def list_sessions(user_id: int = Query(1, ge=0),
                        role_key: str = Query(None),
                        # ge=1：负数会被直接拼进 SQL 的 LIMIT，触发语法错误
                        limit: int = Query(50, ge=1, le=200),
                        db: Session = Depends(get_db)):
    rows = SessionService.list_sessions(db, user_id, role_key, limit)
    return Resp(data=[s.to_dict() for s in rows])


@sessions_router.get("/{session_id}/messages", response_model=Resp,
                     summary="会话历史消息")
async def get_messages(session_id: str, limit: int = Query(100, ge=1, le=500),
                       db: Session = Depends(get_db)):
    s = SessionService.get(db, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    msgs = SessionService.get_messages(db, session_id, limit)
    return Resp(data={"session": s.to_dict(),
                      "messages": [m.to_dict() for m in msgs]})


@sessions_router.get("/{session_id}/memory", response_model=Resp,
                     summary="查看短期/长期记忆与会话热状态")
async def get_memory_view(session_id: str, user_id: int = Query(1),
                          role_key: str = Query(...),
                          db: Session = Depends(get_db)):
    """调试用：直观看到三层记忆。

    short_term    Redis List，最近 N 轮原文
    session_state Redis Hash，会话热状态（角色/轮次/最后活跃）——
                  每轮问答结束时由 rag_service 写入，这里读出来给调用方看
    long_term     Milvus 摘要条数
    """
    history = redis_conn.get_recent_turns(user_id, role_key, session_id,
                                          settings.SHORT_TERM_TURNS)
    return Resp(data={"short_term": history,
                      "session_state": redis_conn.get_session_meta(session_id),
                      "long_term": get_memory().stats(str(user_id), role_key)})


@sessions_router.delete("/{session_id}", response_model=Resp, summary="删除会话")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    if not SessionService.delete(db, session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    db.commit()          # 响应发出前落库，理由同 chat 接口
    return Resp(msg="已删除")


ROUTERS = [roles_router, sessions_router]
