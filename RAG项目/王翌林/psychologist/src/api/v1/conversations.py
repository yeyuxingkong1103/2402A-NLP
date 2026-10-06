"""会话接口：新建、列表、历史消息、删除。

会话（conversation）绑定"用户 + 角色"，是承载聊天记录的容器。
这里所有接口都以 get_current_user 为依赖，且服务层内部会再次校验
"该会话是否属于当前用户"，双重保证用户只能访问自己的会话，防止水平越权。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.deps import get_current_user
from src.core.exceptions import ok
from src.db.mysql import get_db
from src.models import User
from src.schemas import ConversationCreateRequest
from src.services import conversation_service, persona_service

router = APIRouter(prefix="/conversations", tags=["会话"])


@router.post("", summary="新建会话（C-01：选择心理医生角色）")
def create_conversation(payload: ConversationCreateRequest,
                        user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 先校验角色存在，再创建会话；角色信息在服务层按需加载，控制器不接触 SQL。
    persona = persona_service.get_persona(db, payload.persona_id)
    conv = conversation_service.create_conversation(db, user.id, persona.id, payload.title)
    data = conversation_service._conv_to_dict(conv, persona)
    data["greeting"] = persona.greeting
    # 附带角色开场白，前端可在会话建立后立即展示一句欢迎语，提升体验。
    return ok(data, "会话创建成功")


@router.get("", summary="会话列表（C-08：仅本人会话）")
def list_conversations(persona_id: int | None = Query(None), limit: int = Query(100, ge=1, le=500),
                       user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # user.id 来自登录态，而非客户端，天然过滤出"仅本人会话"。
    items = conversation_service.list_conversations(db, user.id, persona_id, limit=limit)
    return ok({"items": items, "total": len(items)})


@router.get("/{conversation_id}", summary="会话详情")
def get_conversation(conversation_id: int, user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    # 传入 user.id，服务层会校验会话归属，非本人会话会抛错（防越权）。
    conv = conversation_service.get_conversation(db, conversation_id, user.id)
    persona = persona_service.get_persona(db, conv.persona_id)
    return ok(conversation_service._conv_to_dict(conv, persona))


@router.get("/{conversation_id}/messages", summary="历史消息（C-05）")
def list_messages(conversation_id: int, limit: int = Query(200, ge=1, le=500),
                  offset: int = Query(0, ge=0),
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # limit/offset 构成分页：limit 控制条数，offset 控制跳过多少条。
    return ok(conversation_service.list_messages(db, conversation_id, user.id, limit, offset))


@router.delete("/{conversation_id}", summary="删除会话（C-07：逻辑删除）")
def delete_conversation(conversation_id: int, user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    # 逻辑删除：数据库里通常只是打标记，不物理删数据，便于追溯与恢复。
    conversation_service.delete_conversation(db, conversation_id, user.id)
    return ok(None, "会话已删除")


@router.post("/{conversation_id}/save-memory", summary="手动保存长期记忆")
def save_memory(conversation_id: int, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    conv = conversation_service.get_conversation(db, conversation_id, user.id)
    # 这里用函数内 import 延迟加载 memory_service，避免模块循环依赖，也减少启动开销。
    from src.services import memory_service
    # force=True 表示不按既定触发条件，而是强制立即总结归档为长期记忆。
    summary = memory_service.maybe_save_long_term(db, conv, user.id, conv.persona_id, force=True)
    if not summary:
        return ok(None, "暂无可保存的对话内容")
    return ok({"conversation_id": conv.id, "summary": summary}, "长期记忆已保存")