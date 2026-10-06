import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.models.chat import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.security.auth import get_current_user
from app.services.chat_service import get_chat_service

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _sse_events(
    request: ChatRequest,
    current_user: User,
) -> Iterator[str]:
    try:
        events = get_chat_service().stream_answer(
            request.message.strip(),
            request.session_id,
            request.top_k,
            current_user,
            request.role_id,
        )
        for event in events:
            yield f"event: {event['event']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as exc:
        payload: dict[str, Any] = {"message": "在线问答暂时不可用，请稍后重试"}
        if isinstance(exc, (RuntimeError, FileNotFoundError, PermissionError, HTTPException)):
            payload["message"] = getattr(exc, "detail", str(exc))
        yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    try:
        return get_chat_service().answer(
            request.message.strip(),
            request.session_id,
            request.top_k,
            current_user,
            request.role_id,
        )
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="在线问答暂时不可用，请稍后重试") from exc


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        _sse_events(request, current_user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
