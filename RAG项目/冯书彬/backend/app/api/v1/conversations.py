from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_current_principal
from backend.app.services.conversation_service import delete_conversation_and_exclusive_memories

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, principal: dict = Depends(get_current_principal)):
    # 删除会话时以认证用户为准，只清理独占记忆；共享记忆保留给其他会话使用。
    try:
        deleted_memory_ids = delete_conversation_and_exclusive_memories(principal["sub"], conversation_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"deleted": True, "deleted_memory_ids": deleted_memory_ids}
