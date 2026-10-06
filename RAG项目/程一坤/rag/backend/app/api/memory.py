"""长期记忆接口（批次 14，接口文档 8.1 / 8.2 / 8.3）。

语义（按用户裁决）：
- 8.1 GET  /api/v1/memories                → 本人记忆分页列表（user_id 取认证上下文）
- 8.2 DELETE /api/v1/memories/{memory_id}  → 软删除；非本人/不存在一律 404（不暴露存在性）
- 8.3 PUT  /api/v1/users/me/memory-settings → 长期记忆开关；关闭后既不写入也不读取

生产装配：记忆集合与开关存储按 settings 直连（与 review.py 同一风格）；
测试通过 app.state 注入替身。
"""

import logging

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from app.auth.current_user import CurrentUser
from app.core.config import settings
from app.errors import not_found_error

router = APIRouter(tags=["memories"])

logger = logging.getLogger(__name__)


def _get_memory_store(request: Request):
    """长期记忆存储（测试可经 app.state.long_term_memory 注入替身）。"""
    store = getattr(request.app.state, "long_term_memory", None)
    if store is None:
        from pymilvus import MilvusClient

        from app.memory.long_term import LongTermMemoryStore
        from app.models.embedding import SiliconFlowEmbeddingClient

        store = LongTermMemoryStore(
            milvus_client=MilvusClient(
                uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
            ),
            embedding_client=SiliconFlowEmbeddingClient(
                api_url=settings.embedding_api_base_url,
                api_key=settings.embedding_api_key,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
            ),
            collection_name=settings.milvus_long_term_collection_name,
            dimension=settings.embedding_dimension,
            dedup_threshold=settings.long_term_memory_dedup_threshold,
        )
        store.ensure_collection()
        request.app.state.long_term_memory = store
    return store


def _get_settings_store(request: Request):
    """记忆开关存储（测试可经 app.state.memory_settings_store 注入替身）。"""
    store = getattr(request.app.state, "memory_settings_store", None)
    if store is None:
        from app.chat.chat_runtime import build_default_session_factory
        from app.memory.memory_settings import MemorySettingsStore

        store = MemorySettingsStore(build_default_session_factory())
        request.app.state.memory_settings_store = store
    return store


@router.get("/api/v1/memories")
def list_memories(
    current_user: CurrentUser,
    request: Request,
    character_id: str | None = Query(default=None, max_length=64),
    page: int = Query(default=1, ge=1, description="页码，1 起始"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页条数，上限 100"),
) -> dict:
    """返回本人长期记忆的分页列表（接口 8.1）。

    - user_id 一律取认证上下文，不收客户端参数（跨用户隔离的接口层兜底）
    - 已软删/已过期不出现在列表里（存储层过滤）
    """
    request_id = request.state.request_id
    store = _get_memory_store(request)
    result = store.list_memories(
        user_id=current_user.user_id,
        character_id=character_id,
        page=page,
        page_size=page_size,
    )
    items = [
        {
            "memory_id": record["memory_id"],
            "character_id": record["character_id"],
            "content": record["content"],
            "summary": record["summary"],
            "importance": record["importance"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "expire_at": record["expire_at"],
            "source_session_id": record["source_session_id"],
        }
        for record in result["items"]
    ]
    return {
        "code": 0,
        "message": "success",
        "data": {
            "items": items,
            "total": result["total"],
            "page": result["page"],
            "page_size": result["page_size"],
        },
        "request_id": request_id,
    }


@router.delete("/api/v1/memories/{memory_id}")
def delete_memory(
    memory_id: str,
    current_user: CurrentUser,
    request: Request,
) -> dict:
    """软删除本人的一条长期记忆（接口 8.2）。

    归属与存在性统一 404：非本人的记忆走同一路径，不暴露"存在但不属于你"。
    """
    request_id = request.state.request_id
    store = _get_memory_store(request)
    deleted = store.soft_delete(user_id=current_user.user_id, memory_id=memory_id)
    if not deleted:
        logger.info(
            "长期记忆删除失败（不存在或非本人）：user_id=%s memory_id=%s",
            current_user.user_id,
            memory_id,
        )
        raise not_found_error("记忆不存在")
    return {
        "code": 0,
        "message": "success",
        "data": {"memory_id": memory_id, "deleted": True},
        "request_id": request_id,
    }


class MemorySettingsRequest(BaseModel):
    """记忆开关请求体（接口 8.3）。"""

    long_term_memory_enabled: bool = Field(strict=True)


@router.put("/api/v1/users/me/memory-settings")
def update_memory_settings(
    payload: MemorySettingsRequest,
    current_user: CurrentUser,
    request: Request,
) -> dict:
    """更新本人长期记忆开关（接口 8.3）；关闭后既不写入也不读取。"""
    request_id = request.state.request_id
    store = _get_settings_store(request)
    updated = store.set_enabled(current_user.user_id, payload.long_term_memory_enabled)
    if not updated:
        raise not_found_error("用户不存在")
    logger.info(
        "长期记忆开关更新：user_id=%s enabled=%s",
        current_user.user_id,
        payload.long_term_memory_enabled,
    )
    return {
        "code": 0,
        "message": "success",
        "data": {
            "long_term_memory_enabled": payload.long_term_memory_enabled,
        },
        "request_id": request_id,
    }
