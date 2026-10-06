"""聊天接口：非流式 /chat 与 SSE 流式 /chat/stream。

聊天是本系统最核心的路径，但控制器依旧很薄：限流 -> 校验角色 -> 转交 rag_service。
至于检索、拼接提示词、调用大模型等重活，全部封装在服务层，路由只是入口。
"""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.api.deps import get_current_user
from src.core.exceptions import RateLimitError, ok
from src.db.mysql import get_db
from src.db.redis import check_rate_limit
from src.models import User
from src.schemas import ChatRequest
from src.services import persona_service, rag_service

router = APIRouter(tags=["对话"])


@router.post("/chat", summary="非流式聊天（R-01 ~ R-10）")
def chat(payload: ChatRequest, user: User = Depends(get_current_user),
         db: Session = Depends(get_db)):
    # 限流按用户维度（user.id）计数，放在控制器入口：调用大模型成本高，
    # 必须在进入检索/生成前就拦住高频请求，避免资源被单个用户耗尽。
    if not check_rate_limit(user.id):
        raise RateLimitError("请求过于频繁，请稍后再试")
    persona = persona_service.get_persona(db, payload.persona_id)
    # 角色下架（status != 1）后应停止提供对话服务，这里提前拦截并给出友好提示。
    if persona.status != 1:
        raise RateLimitError("该心理医生暂未开放")

    # answer() 内部完成：检索知识库 -> 组装上下文 -> 调用模型 -> 落库，一次返回完整结果。
    data = rag_service.answer(
        db, user.id, payload.persona_id, payload.message, payload.conversation_id
    )
    return ok(data)


@router.post("/chat/stream", summary="SSE 流式聊天（C-06）")
def chat_stream(payload: ChatRequest, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    if not check_rate_limit(user.id):
        raise RateLimitError("请求过于频繁，请稍后再试")

    # answer_stream 返回一个生成器（generator），模型每生成一段就 yield 一段，
    # 无需等整段答案生成完即可推送给前端，显著降低首字延迟。
    generator = rag_service.answer_stream(
        db, user.id, payload.persona_id, payload.message, payload.conversation_id
    )
    # StreamingResponse 是 FastAPI 处理流式响应的标准方式，配合 media_type 声明为 SSE：
    # 它会把生成器逐块写到响应体，而不是一次性序列化后返回。
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            # 关闭缓存：SSE 是持续输出，客户端/中间层绝不能缓存或提前结束连接。
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关闭 Nginx 的响应缓冲，否则流式内容会被攒成一大块才下发，失去流式意义。
            "X-Accel-Buffering": "no",
        },
    )