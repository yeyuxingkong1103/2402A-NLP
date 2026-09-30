"""管理后台接口：用户管理、会话审计、日志、系统监控。

这是整个后台管理的入口，全部接口都以 get_current_admin 作为依赖，
属于"运维/监管"能力。需要注意：即使 admin.py 里偶尔出现少量 SQL 查询
（如会话审计、监控计数），也只是"只读统计"，真正的业务逻辑仍在服务层。
"""
import os

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.deps import get_client_ip, get_current_admin
from src.core.config import settings
from src.core.exceptions import ok
from src.db import milvus as milvus_db
from src.db import redis as redis_db
from src.db.mysql import get_db, health_check as mysql_health
from src.models import Conversation, Message, User
from src.schemas import UserStatusRequest
from src.services import conversation_service, knowledge_service, persona_service, user_service

router = APIRouter(prefix="/admin", tags=["管理后台"])


@router.get("/users", summary="用户管理列表（U-05）")
def list_users(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
               keyword: str | None = None, status: int | None = None,
               admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    # page/page_size 构成分页，keyword/status 用于按关键词或状态筛选用户。
    return ok(user_service.list_users(db, page, page_size, keyword, status))


@router.post("/users/{user_id}/status", summary="用户禁用/启用（U-05）")
def set_user_status(user_id: int, payload: UserStatusRequest, request: Request,
                    admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    # 禁用/启用用户是高风险操作：改完状态后必须写审计日志，
    # 记录"谁（admin.id）在什么时候对哪个用户做了什么"，保证可追溯。
    user_service.set_user_status(db, user_id, payload.status)
    user_service.add_audit_log(
        db, admin.id, "set_user_status", f"target={user_id} status={payload.status}",
        get_client_ip(request),
    )
    return ok({"user_id": user_id, "status": payload.status}, "状态已更新")


@router.get("/logs/login", summary="登录日志（U-07）")
def login_logs(user_id: int | None = None, limit: int = Query(50, ge=1, le=500),
               admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return ok(user_service.list_login_logs(db, user_id, limit))


@router.get("/logs/audit", summary="审计日志（9.2）")
def audit_logs(user_id: int | None = None, limit: int = Query(50, ge=1, le=500),
               admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return ok(user_service.list_audit_logs(db, user_id, limit))


@router.get("/conversations", summary="会话审计")
def list_all_conversations(user_id: int | None = None, limit: int = Query(50, ge=1, le=200),
                           admin: User = Depends(get_current_admin),
                           db: Session = Depends(get_db)):
    # 管理员可查看任意用户的会话，用于内容安全监管；这里直接用 SQLAlchemy 表达式做只读查询。
    stmt = select(Conversation).order_by(Conversation.id.desc()).limit(limit)
    if user_id:
        stmt = stmt.where(Conversation.user_id == user_id)
    convs = list(db.execute(stmt).scalars().all())
    # 一次性取出所有角色做 id -> 对象 映射，避免在循环里逐条查询（N+1 问题）。
    personas = {p.id: p for p in persona_service.list_personas(db, only_active=False)}
    return ok({
        "items": [conversation_service._conv_to_dict(c, personas.get(c.persona_id)) for c in convs],
    })


@router.get("/monitor", summary="系统监控（数据库/缓存/向量库/模型）")
def monitor(admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    # 函数内 import 延迟加载模型/嵌入器等重资源，避免 admin 模块被导入时立即初始化大模型。
    from src.rag.embedder import get_embedder
    from src.rag.reranker import get_reranker
    from src.services.persona_seed import KNOWLEDGE_DIRS

    # 三个计数都做 int(... or 0) 兜底：避免空表时 scalar() 返回 None 导致类型错误。
    user_count = int(db.execute(select(func.count(User.id))).scalar() or 0)
    conv_count = int(db.execute(select(func.count(Conversation.id))).scalar() or 0)
    msg_count = int(db.execute(select(func.count(Message.id))).scalar() or 0)

    # 逐个角色汇总其向量数量与知识目录，形成"知识库维度"的监控视图。
    knowledge = []
    for persona in persona_service.list_personas(db, only_active=False):
        knowledge.append({
            "persona_id": persona.id,
            "persona_code": persona.persona_code,
            "persona_name": persona.name,
            "vectors": milvus_db.count_knowledge(persona.id),
            "knowledge_dirs": [
                os.path.join(settings.project_root, d) for d in KNOWLEDGE_DIRS.get(persona.persona_code, [])
            ],
        })

    return ok({
        "app": {"name": settings.app_name, "version": settings.app_version, "debug": settings.debug},
        "mysql": {"healthy": mysql_health(), "users": user_count,
                  "conversations": conv_count, "messages": msg_count},
        "redis": {"healthy": redis_db.health_check()},
        "milvus": {"healthy": milvus_db.health_check(),
                   "collection": settings.milvus_collection,
                   "vectors_total": milvus_db.count_knowledge()},
        "models": {
            "embedding": {"path": settings.embedding_model_path, "dim": settings.embedding_dim,
                          "device": settings.embedding_device, "loaded": get_embedder().loaded},
            "reranker": {"path": settings.reranker_model_path,
                         "device": settings.reranker_device, "loaded": get_reranker().loaded},
            "llm": {"provider": settings.llm_provider, "model": settings.llm_model,
                    "base_url": settings.llm_base_url},
        },
        "knowledge": knowledge,
    })