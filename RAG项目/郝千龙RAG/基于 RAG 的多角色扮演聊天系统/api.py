# -*- coding: utf-8 -*-
"""【HTTP 接口层 · api.py】FastAPI 后端：用户注册登录鉴权、多角色对话（含流式 SSE）、知识库文件上传。"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth import authenticate, create_user, make_token, parse_token
from config import API_HOST, API_PORT, APP_ENV, UPLOAD_DIR, DEFAULT_ROLE_CODE
from database import SessionLocal, User, init_db
from logger import log
from memory_store import get_messages
from services import add_knowledge, answer, ensure_session, get_retriever, list_roles_with_data
from generation import chat_stream

init_db()  # 启动即建表 + 写种子数据（幂等）

# FastAPI 应用 + 全放开的 CORS（便于前端/Postman 联调）
app = FastAPI(title="基于RAG的多角色扮演系统", version="1.0.0", description="支持多角色的RAG增强对话系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 请求体模型（Pydantic 自动校验与文档化）=====
class RegisterIn(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=4, max_length=64)


class LoginIn(BaseModel):
    username: str
    password: str


class ChatIn(BaseModel):
    query: str
    role_code: str = DEFAULT_ROLE_CODE
    session_id: Optional[int] = None  # 首次传 None，之后复用返回的会话 id
    top_k: Optional[int] = None
    stream: bool = False


def current_user(authorization: Optional[str] = Header(default=None)) -> User:
    """鉴权依赖：解析 Authorization: Bearer <token>，返回当前用户。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "缺少 Authorization: Bearer <token>")
    user_id = parse_token(authorization.split(" ", 1)[1].strip())
    if not user_id:
        raise HTTPException(401, "token 无效")
    with SessionLocal() as db:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(401, "用户不存在")
        return user


@app.get("/health")
def health():
    """健康检查：返回环境与知识库规模。"""
    retriever = get_retriever()
    return {"status": "ok", "env": APP_ENV, "kb_size": len(retriever.pairs)}


@app.post("/api/auth/register")
def register(body: RegisterIn):
    """注册：用户名查重 → 建用户 → 直接签发 token。"""
    with SessionLocal() as db:
        if db.query(User).filter(User.username == body.username).first():
            raise HTTPException(400, "用户名已存在")
        user = create_user(db, body.username, body.password)
        return {"user_id": user.id, "username": user.username, "token": make_token(user.id)}


@app.post("/api/auth/login")
def login(body: LoginIn):
    """登录：校验密码 → 签发 token。"""
    with SessionLocal() as db:
        user = authenticate(db, body.username, body.password)
        if not user:
            raise HTTPException(401, "用户名或密码错误")
        return {"user_id": user.id, "username": user.username, "token": make_token(user.id)}


@app.get("/api/roles")
def roles():
    """角色列表（不含 system_prompt），只返回知识库中有数据的角色。"""
    return {"roles": list_roles_with_data()}


@app.post("/api/chat")
def chat_api(body: ChatIn, user: User = Depends(current_user)):
    """对话接口：stream=true 走流式（逐 chunk 文本），否则返回完整 JSON。"""
    if body.stream:
        sid = ensure_session(user.id, body.role_code, body.session_id)
        history = get_messages(user.id, body.role_code, sid)  # 读短期记忆
        retriever = get_retriever()

        def gen():
            """SSE 流式响应生成器：把 chat_stream 产出的每个文本片段逐个 yield 给浏览器，实现打字机效果。"""
            # 生成器逐块产出，StreamingResponse 边生成边发送
            for piece in chat_stream(
                body.query, retriever, history=history, top_k=body.top_k, role_code=body.role_code
            ):
                yield piece

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
    # 非流式：完整 RAG 链路（含记忆写回与消息落库）
    result = answer(user.id, body.role_code, body.query, session_id=body.session_id, top_k=body.top_k)
    return result


@app.post("/api/kb/upload")
def upload_kb(file: UploadFile = File(...), user: User = Depends(current_user)):
    """知识库动态更新：保存上传文件 → 解析分块入库。"""
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(file.file.read())
    n = add_knowledge(dest)
    log.info("user %s uploaded %s chunks=%s", user.username, dest, n)
    return {"filename": file.filename, "chunks": n}


def main():
    """直接运行本文件：启动 uvicorn（开发环境带热重载）。"""
    import uvicorn

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=APP_ENV == "development")


if __name__ == "__main__":
    main()
