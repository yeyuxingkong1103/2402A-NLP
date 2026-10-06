"""对话接口：/chat 与 /chat_stream。两个核心接口"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.logging_conf import log
from app.rag.pipeline import answer, answer_stream
from app.schemas import ChatReq, ChatResp

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResp)
def chat(req: ChatReq) -> dict:
    try:
        return answer(req.user_id, req.role_id, req.message, use_rag=req.use_rag)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error("chat 失败: %s", exc)
        raise HTTPException(500, str(exc)) from exc


@router.post("/chat_stream")
def chat_stream(req: ChatReq) -> StreamingResponse:
    def gen():
        try:
            yield from answer_stream(req.user_id, req.role_id, req.message, use_rag=req.use_rag)
        except ValueError as exc:
            yield f"错误：{exc}"
        except Exception as exc:  # noqa: BLE001
            log.error("chat_stream 失败: %s", exc)
            yield f"错误：{exc}"

    return StreamingResponse(gen(), media_type="text/plain")
