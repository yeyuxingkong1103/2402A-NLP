"""对话接口：RAG 问答（/chat 非流式、/chat/stream 流式 SSE）。

错误码分工（与 pipeline / llm 的分层对应）：EmptyAnswerError → 502（模型答了但答的是空的），
其它异常 → 503（依赖掉线）。流式路径尽量把失败挡在响应头发出「之前」，挡不住的才发 error 事件。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.logging_config import get_logger
from ..core.pipeline import EmptyAnswerError
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])
log = get_logger("api.chat")


def _ensure_role(pipeline, role_id: str):
    """校验角色存在；未知角色返回 404 而不是 500。

    必须在进入流式生成器「之前」调用：响应头一旦发出就改不了状态码，
    在生成器里抛错只会得到一个 200 + 半截流，用户端完全看不出发生了什么。
    """
    try:
        return pipeline.ensure_role(role_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    p = request.app.state.pipeline
    try:
        _ensure_role(p, req.role_id)
        return p.answer(req.question, req.role_id, req.session_id)
    except EmptyAnswerError as exc:
        # 502 而非 200+空串：让调用方能区分「模型答了」和「模型什么都没答」
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # 依赖掉线（LLM/Embedding/SQL/Milvus 运行中挂了）或其它未预期错误：
        # 别裸 500，给调用方一个明确信号；完整 traceback 进日志，不掩盖排查信息。
        log.exception("对话失败 role=%s session=%s", req.role_id, req.session_id)
        raise HTTPException(
            status_code=503,
            detail=f"服务暂时不可用（{type(exc).__name__}），请稍后重试",
        ) from exc


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    p = request.app.state.pipeline
    try:
        _ensure_role(p, req.role_id)  # 未知角色在这里就 404，别等流开始了才炸
        chunks = p.retrieve(req.question, req.role_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # 流还没开始（响应头未发出）依赖就掉了：可以正常返回 JSON 503，
        # 而不是等流开始后只能发半截 error 事件。
        log.exception("流式检索失败 role=%s session=%s", req.role_id, req.session_id)
        raise HTTPException(
            status_code=503,
            detail=f"服务暂时不可用（{type(exc).__name__}），请稍后重试",
        ) from exc
    sources = p.to_sources(chunks)
    final: list[str] = []  # answer_stream 会把后处理过的完整答案 append 进来

    def gen():
        yield _sse("sources", {"sources": sources})
        try:
            for delta in p.answer_stream(
                req.question, req.role_id, req.session_id, chunks=chunks, final=final
            ):
                yield _sse("delta", {"delta": delta})
        except EmptyAnswerError as exc:
            # 模型一个字都没答（思考阶段就把预算用光了）。消息本身是给用户看的，
            # 不加异常类名前缀——那不是用户需要知道的。
            log.warning("模型未产出回答 role=%s session=%s", req.role_id, req.session_id)
            yield _sse("error", {"message": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            # 生成中途出错（模型掉线、超时等）：明确告诉前端，避免静默截断
            log.exception("流式生成失败 role=%s session=%s", req.role_id, req.session_id)
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
            return
        # 流里吐的是原始增量，这里回传清洗后的权威文本，前端用它覆盖显示
        answer = final[0] if final else ""
        yield _sse("done", {"answer": answer})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
