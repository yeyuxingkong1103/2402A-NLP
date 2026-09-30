from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.base import get_db
from ...db.models import Session, User
from ...deps import get_milvus
from ..deps import get_current_user
from ..schemas import ok

router = APIRouter(tags=["memories"])


@router.get("/api/v1/sessions/{session_id}/memories")
async def list_memories(session_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    s = await db.get(Session, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    milvus = get_milvus()
    mems = await milvus.hybrid_search("long_term_memory", "", f"user_id == {user.id} && character_id == {s.character_id}", top_k=50)
    return ok([{"id": m["id"], "type": m.get("memory_type"), "content": m["content"]} for m in mems])


@router.delete("/api/v1/memories/{memory_id}")
async def delete_memory(memory_id: int, user: User = Depends(get_current_user)):
    await get_milvus().delete_by_filter("long_term_memory", f"id == {memory_id}")
    return ok()


@router.post("/api/v1/sessions/{session_id}/extract")
async def trigger_extract(session_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    s = await db.get(Session, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    # 入队 ARQ
    from arq import create_pool
    from ...worker.run import WorkerSettings
    redis = await create_pool(WorkerSettings.redis_settings)
    job = await redis.enqueue_job("extract_memory_task", session_id, user.id, s.character_id)
    await redis.aclose()
    return ok({"job_id": job.job_id})
