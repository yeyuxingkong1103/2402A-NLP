# app/api/chat.py
"""问答接口：普通返回 + SSE 流式返回。"""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from sqlalchemy.orm import Session

from app.core.rag_service import get_rag_service
from app.db import get_db, redis_conn
from app.schemas import AskRequest, Resp
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["问答"])


@router.post("/ask", response_model=Resp, summary="角色扮演问答")
async def ask(req: AskRequest, db: Session = Depends(get_db)):
    """
    完整 RAG 链路：双层记忆 -> 追问改写 -> 混合检索 -> 重排 -> 生成。
    不传 session_id 会自动新建会话并返回，后续带上即可延续多轮对话。
    """
    try:
        result = get_rag_service().chat(
            db=db, user_id=req.user_id, role_key=req.role_key,
            question=req.question, session_id=req.session_id)
        # 显式提交：yield 依赖的 teardown 在响应发出之后才执行，
        # 若只依赖它提交，客户端拿到 200 时数据可能尚未落库
        db.commit()
        redis_conn.incr_role_hot(req.role_key)
        return Resp(data=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.exception("问答失败")
        raise HTTPException(status_code=500, detail="服务内部错误: %s" % e)


@router.post("/stream", summary="角色扮演问答（SSE 流式）")
async def ask_stream(req: AskRequest, db: Session = Depends(get_db)):
    """事件流：meta（会话与来源）-> delta（正文增量）-> done。"""
    def event_gen():
        try:
            for item in get_rag_service().chat_stream(
                    db=db, user_id=req.user_id, role_key=req.role_key,
                    question=req.question, session_id=req.session_id):
                yield "data: %s\n\n" % json.dumps(item, ensure_ascii=False)
            db.commit()             # 流结束后落库，理由同上
            redis_conn.incr_role_hot(req.role_key)
        except Exception as e:                              # pragma: no cover
            logger.exception("流式问答失败")
            yield "data: %s\n\n" % json.dumps(
                {"type": "error", "message": str(e)}, ensure_ascii=False)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


ROUTERS = [router]
