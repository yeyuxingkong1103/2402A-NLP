"""FastAPI 接口：/health 与 /chat。"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import admin
import rag
import roles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def verify_admin(x_api_key: str | None = Header(None)):
    """fail-closed 校验 X-API-Key：未配置 ADMIN_KEY 或 key 不匹配一律 401。"""
    admin_key = os.getenv("ADMIN_KEY")
    if not admin_key or x_api_key != admin_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载角色缓存 + 预热 embedder/reranker，避免首个请求冷启动。"""
    roles.load_roles()
    rag.load_embedder()
    logger.info("embedder 预热完成")
    rag.load_reranker()
    logger.info("reranker 预热完成")
    yield


app = FastAPI(title="高血压医生 RAG", lifespan=lifespan)
app.include_router(admin.router, dependencies=[Depends(verify_admin)])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    # 多用户场景必须让客户端传唯一 session_id，否则所有请求共享 default 会话，会串记忆。
    session_id: str = "default"
    # 角色名（如「高血压医生」）；None 用第一个角色（默认医生），不存在则 404。
    role: str | None = None
    # 长期记忆归属；None 时用 session_id 兜底（向后兼容）。
    user_id: str | None = None


class Source(BaseModel):
    content: str
    page: int
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


def _validate_role(role: str | None) -> str | None:
    """校验角色名：明确传入但不存在则 404；None 原样返回。"""
    if role is not None and roles.get_role(role) is None:
        raise HTTPException(status_code=404, detail=f"角色不存在: {role}")
    return role


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message 不能为空")
    role = _validate_role(req.role)
    user_id = req.user_id
    logger.info("收到请求 session_id=%s role=%s user_id=%s message 长度=%d", req.session_id, role, user_id, len(message))
    try:
        if role is None:
            if user_id is None:
                return rag.ask(message, req.session_id)
            return rag.ask(message, req.session_id, user_id=user_id)
        if user_id is None:
            return rag.ask(message, req.session_id, role)
        return rag.ask(message, req.session_id, role, user_id=user_id)
    except Exception as e:  # noqa: BLE001
        logger.exception("处理请求失败 session_id=%s", req.session_id)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message 不能为空")
    role = _validate_role(req.role)
    user_id = req.user_id
    logger.info("收到流式请求 session_id=%s role=%s user_id=%s message 长度=%d", req.session_id, role, user_id, len(message))
    if role is None:
        if user_id is None:
            gen = rag.ask_stream(message, req.session_id)
        else:
            gen = rag.ask_stream(message, req.session_id, user_id=user_id)
    else:
        if user_id is None:
            gen = rag.ask_stream(message, req.session_id, role)
        else:
            gen = rag.ask_stream(message, req.session_id, role, user_id=user_id)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 挂载前端静态文件（放在所有 API 路由之后，/health /chat /docs 等优先匹配）
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
"""
FastAPI 接口模块的作用：对外暴露 HTTP 接口，接收前端对话请求，路由分发，参数校验、鉴权，同时提供健康检测和前端静态资源服务。
第一，`lifespan`是服务生命周期管理器，服务启动的时候预先加载角色配置，预热向量模型和重排模型，这样做的目的是把耗时的模型加载放到服务启动阶段，避免第一个用户请求触发模型加载，造成首请求冷启动超时。
第二，定义请求体`ChatRequest`、返回结构`ChatResponse`和来源文档`Source`这几个 Pydantic 模型，这样做的目的是自动完成入参类型校验，规范请求和返回的数据结构。`/health`健康接口直接返回状态 ok，这样做的目的是给监控、容器编排提供探测端点，用来判断服务实例是否存活。
第三，`/chat`是普通非流式对话接口，校验入参消息不为空，校验角色，把 session_id、role、user_id 透传给 rag.ask，这样做的目的是接收对话请求，把请求参数传递给 rag 模块执行业务逻辑。`/chat/stream`是 SSE 流式接口，参数校验逻辑和普通 chat 一致，调用 rag.ask_stream 拿到生成器，包装成 StreamingResponse 返回，并添加禁用缓存的响应头，这样做的目的是将大模型分段输出的内容以 SSE 流的形式实时推送给前端。最后挂载前端静态文件，放在所有 API 路由后面，这样做的目的是保证 API 路由优先匹配，防止前端路由拦截`/health`、`/chat`这类接口请求。
"""