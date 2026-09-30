"""src/api/routers/session.py —— 会话管理路由。

在链路中的位置：
    HTTP → 【本文件】 → src/models/database.py（session / message 表）
    src/api/routers/chat.py 在对话前会校验会话归属（逻辑与本文件的 messages 一致）

路由前缀 /api/v1/sessions：
    POST /sessions                 新建会话
    GET  /sessions/{sid}/messages  查看某会话的历史消息

会话是"一个用户在某个角色下的一段连续对话"：
    它把消息组织起来，也是短期记忆的容器（src/memory/short_term.py 按会话存取最近若干轮）。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from src.api.deps import current_user
from src.api.routers.role import get_role_config
from src.models.database import ChatSession, Message, User, db_session
from src.schemas.roleplay import SessionCreate

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("")
def create_session(request: SessionCreate, user: User = Depends(current_user)):
    """新建一个会话。

    参数：
        request: SessionCreate（role_id + title）
        user: 当前用户
    返回：
        {"session_id", "role_id", "title"}

    第一行 get_role_config 是"先校验角色存在"：
        它在角色不存在时抛 404，从而**阻止创建指向不存在角色的会话**。
        少了这一步，用户能建出一堆点进去就报错的空会话 ——
        要建会话，先确认它要绑定的角色是真实存在的。

    注意这个调用没有使用返回值：
        这里只想要它的副作用（校验 + 不存在时报错），
        角色配置本身在对话时才需要。

    db.refresh(session) 是必需的：
        session.id 由数据库自增生成，commit 前是 None，
        不 refresh 就返回不了 session_id。

    tenant_id 取自 user 而不是请求体：
        租户由服务端根据登录身份决定，绝不接受客户端传入 ——
        否则用户可以通过伪造 tenant_id 把会话建到别人租户下。
    """
    get_role_config(request.role_id, user.tenant_id)
    with db_session() as db:
        session = ChatSession(user_id=user.id, role_id=request.role_id, tenant_id=user.tenant_id, title=request.title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return {"session_id": session.id, "role_id": session.role_id, "title": session.title}


@router.get("/{sid}/messages")
def messages(sid: int, user: User = Depends(current_user)):
    """查看某会话的全部历史消息。

    参数：
        sid: 会话 id（路径参数）
        user: 当前用户
    返回：
        {"messages": [{"id", "speaker", "content", "created_at"}, ...]}
    异常：
        会话不存在或不属于当前用户 -> 404。

    越权校验与 chat.py 的 _session 完全一致（不存在/不属于我都返回 404）：
        统一 404 而不区分 403，是为了不泄露"这个 id 是否存在"。

    按 created_at 升序（order_by(Message.created_at)）：
        聊天记录要从早到晚读，倒序会让对话读起来逻辑错乱。

    created_at.isoformat() 转成 ISO 8601 字符串：
        datetime 对象无法直接被 JSON 序列化，
        转成 ISO 格式既满足 JSON 要求，也是前端最容易解析的时间格式。
    """
    with db_session() as db:
        session = db.get(ChatSession, sid)
        if not session or session.user_id != user.id:
            raise HTTPException(status_code=404, detail="会话不存在")
        rows = db.scalars(select(Message).where(Message.session_id == sid).order_by(Message.created_at)).all()
        return {"messages": [{"id": item.id, "speaker": item.speaker, "content": item.content, "created_at": item.created_at.isoformat()} for item in rows]}
