# -*- coding: utf-8 -*-
"""
main.py -- 接口层：FastAPI 服务（P4 服务化的对外门户）

接口清单（对应交付物《接口文档》）：

  POST /api/register        注册            {username, password}
  POST /api/login           登录            {username, password} -> {user_id, username}
  GET  /api/roles           角色列表        -> [{id, name, description}]
  POST /api/chat            对话（带记忆）  {user_id, role_id, message} -> {answer, sources, rewritten_query}
  GET  /api/history         查历史          ?user_id=&role_id=
  DELETE /api/history       清空记忆        ?user_id=&role_id=

启动：uvicorn app.main:app --reload --port 8000
文档：浏览器打开 http://127.0.0.1:8000/docs（FastAPI 自动生成的交互文档）
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APITimeoutError, OpenAIError
from pydantic import BaseModel, Field

from app import db
from app.config import load_env

load_env()


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    yield

app = FastAPI(
    title="知愈·通用健康问答",
    version="0.2.0",
    lifespan=lifespan,
    description=(
        "面向常见健康问题的 RAG 医生问答系统；当前可引用的专属知识库主要是"
        "国家基层高血压防治管理指南，其他主题仅提供有限的一般健康科普。"
    ),
)

engine = None                      # RAG 引擎（首次调用时懒加载，避免启动就占内存）
engine_lock = Lock()
logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")


def get_engine():
    global engine
    if engine is None:
        with engine_lock:
            if engine is None:
                from app.rag import RAGEngine

                try:
                    engine = RAGEngine()
                except RuntimeError as e:
                    raise HTTPException(status_code=503, detail=str(e)) from e
    return engine


# ---------- 请求体模型（pydantic 会自动校验参数）----------
class RegisterReq(BaseModel):
    username: str = Field(min_length=2, max_length=20)
    password: str = Field(min_length=4, max_length=50)


class ChatReq(BaseModel):
    user_id: int
    role_id: int
    message: str = Field(min_length=1, max_length=500)


def require_doctor(role_id: int):
    role = db.get_role(role_id)
    if not role or role["name"] != "医生":
        raise HTTPException(status_code=404, detail="医生角色不存在或该角色不可用")
    return role


# ---------- 接口 ----------
@app.get("/")
def root():
    return {"service": "RAG 角色扮演系统", "docs": "/docs", "status": "ok"}


@app.get("/chat", include_in_schema=False)
def chat_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chat/", include_in_schema=False)
def chat_page_canonical():
    # Relative redirect preserves external proxy prefixes.
    return RedirectResponse("../chat", status_code=307)


@app.get("/api/status")
async def service_status():
    """Check the configured model service without initializing embeddings."""
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    model_name = os.environ.get("LLM_MODEL", "Qwen3-0.6B")
    model_ready = False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                base_url + "/models",
                headers={"Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "EMPTY")},
            )
            response.raise_for_status()
            model_ready = any(item.get("id") == model_name for item in response.json().get("data", []))
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass
    knowledge_ready = engine is not None
    return {
        "ready": model_ready and knowledge_ready,
        "provider": os.environ.get("LLM_PROVIDER", "unknown"),
        "model": model_name,
        "model_ready": model_ready,
        "knowledge_ready": knowledge_ready,
        "message": "可以开始提问" if model_ready else "模型服务未连接，请检查网络、API 配置或账户余额",
    }


@app.post("/api/register")
def register(req: RegisterReq):
    uid = db.create_user(req.username, req.password)
    if uid is None:
        raise HTTPException(status_code=400, detail="用户名已存在")
    return {"user_id": uid, "username": req.username, "message": "注册成功"}


@app.post("/api/login")
def login(req: RegisterReq):
    user = db.verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"user_id": user["id"], "username": user["username"], "message": "登录成功"}


@app.get("/api/roles")
def roles():
    return [role for role in db.list_roles() if role["name"] == "医生"]


@app.post("/api/chat")
def chat(req: ChatReq):
    role = require_doctor(req.role_id)
    try:
        result = get_engine().chat(req.user_id, req.role_id, req.message, role["persona_prompt"], role["name"])
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="模型回答超时，请稍后重试。") from exc
    except APIConnectionError as exc:
        logger.warning("Model connection failed: %r", exc.__cause__)
        raise HTTPException(status_code=503, detail="模型服务连接失败，请检查网络和 API 配置。") from exc
    except OpenAIError as exc:
        logger.warning("Model request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="模型未能完成回答，请检查 API Key、余额或服务状态。") from exc
    result["role"] = role["name"]
    return result


@app.get("/api/history")
def history(user_id: int = Query(...), role_id: int = Query(...)):
    require_doctor(role_id)
    text = get_engine().get_history(user_id, role_id)
    return {"user_id": user_id, "role_id": role_id, "history": text}


@app.delete("/api/history")
def clear_history(user_id: int = Query(...), role_id: int = Query(...)):
    require_doctor(role_id)
    get_engine().clear_memory(user_id, role_id)
    return {"message": "该角色的对话记忆已清空"}
