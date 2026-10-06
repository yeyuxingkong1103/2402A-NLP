# -*- coding: utf-8 -*-
"""FastAPI 在线服务：多用户 / 多角色 / 多轮对话 + Redis 短期记忆 + SSE 流式 + RAGAS 评估。

启动方式：
    python run.py
或
    uvicorn rag2.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import RAG2Config, apply_updates, load_config, save_config, validate_updates
from .logging_config import get_logger, setup_logging
from .online_chat import RAGChat
from .roles import Role, load_roles, role_from_dict, save_roles

logger = get_logger("server")

WEB_DIR = Path(__file__).resolve().parent / "webui"
COOKIE_NAME = "rag2_token"


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str


class ChatRequest(BaseModel):
    question: str
    role_id: str
    session_id: str | None = None
    top_k: int | None = None
    mode: str | None = None
    rerank: bool | None = None
    stream: bool = True


class SessionCreate(BaseModel):
    role_id: str = "general"
    title: str = ""


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    retrieved_contexts: list[str] | None = None
    reference: str | None = None
    metrics: str | list[str] | None = None


class RolePayload(BaseModel):
    role_id: str
    name: str = ""
    system_prompt: str = ""
    description: str = ""
    avatar: str = "🙂"
    followups: list[str] = []


def _sse(event: str, data: Any) -> str:
    """SSE 帧：event + data(JSON)。"""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _token_from(request: Request, authorization: str | None) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.cookies.get(COOKIE_NAME, "")


def current_user(request: Request, authorization: str | None = Header(default=None)) -> str:
    """从 Authorization 头或 Cookie 解析登录用户（返回用户名）。"""
    chat: RAGChat = request.app.state.chat
    token = _token_from(request, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="缺少访问令牌，请先登录")
    username = chat.get_memory().resolve_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return username


def current_admin(request: Request, user: str = Depends(current_user)) -> str:
    """管理员校验：用户名需在 config.admin.usernames 里。"""
    cfg: RAG2Config = request.app.state.cfg
    if user not in (cfg.admin.usernames or []):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


def _is_admin(request: Request, username: str) -> bool:
    cfg: RAG2Config = request.app.state.cfg
    return username in (cfg.admin.usernames or [])


# ---------------------------------------------------------------------------
# 自重启（改端口/地址后生效）：run.py 以监督循环运行，把运行中的 uvicorn Server 挂到这里
# ---------------------------------------------------------------------------
_uvicorn_server: Any = None
_restart_pending = False


def bind_server(server: Any) -> None:
    """由 run.py 把运行中的 uvicorn.Server 挂载进来，用于触发优雅重启。"""
    global _uvicorn_server
    _uvicorn_server = server


def request_restart() -> bool:
    """请求重启：置位标志并让 uvicorn 优雅退出；返回是否已触发自动重启。"""
    global _restart_pending
    _restart_pending = True
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True
        return True
    return False


def consume_restart() -> bool:
    """run.py 循环读取并清除重启标志。"""
    global _restart_pending
    flag = _restart_pending
    _restart_pending = False
    return flag


def create_app(config: RAG2Config | None = None) -> FastAPI:
    """构建 FastAPI 应用（懒连接 Redis / Milvus / Ollama）。"""
    cfg = config or load_config()
    chat = RAGChat(cfg)
    roles = load_roles(cfg.roles_file())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(level=cfg.app.log_level, log_dir=cfg.app.log_dir)
        logger.info("启动 %s（%s:%d）", cfg.app.name, cfg.app.host, cfg.app.port)
        yield
        logger.info("服务已停止")

    app = FastAPI(
        title=cfg.app.name,
        description="多用户 / 多角色 RAG 问答服务（Milvus + BM25 + Redis 短期记忆 + SSE + RAGAS）",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.chat = chat
    app.state.cfg = cfg
    app.state.roles = roles

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("未处理异常：%s", exc)
        return JSONResponse(status_code=500, content={"ok": False, "message": f"{type(exc).__name__}: {exc}"})

    # ------------------------------------------------------------------ 健康
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        report: dict[str, Any] = {"app": cfg.app.name, "ok": True, "components": {}}
        try:
            report["components"]["redis"] = {"ok": True, **chat.get_memory().ping()}
        except Exception as exc:  # noqa: BLE001
            report["components"]["redis"] = {"ok": False, "error": str(exc)}
        try:
            retriever = chat.get_retriever()
            ping = retriever.ping()
            rows = retriever.count()
            report["components"]["milvus"] = {
                "ok": bool(ping.get("reachable")), **ping, "rows": rows,
            }
        except Exception as exc:  # noqa: BLE001
            report["components"]["milvus"] = {"ok": False, "error": str(exc)}
        report["components"]["roles"] = {"ok": True, "total": len(roles)}
        report["models"] = {
            "llm": cfg.llm.model,
            "llm_base_url": cfg.llm.base_url,
            "embed_model": cfg.milvus.embed_model,
            "rerank_model": cfg.milvus.rerank_model,
        }
        report["ok"] = all(item.get("ok", False) for item in report["components"].values())
        return report

    # ------------------------------------------------------------------ 角色
    @app.get("/api/roles")
    def list_roles(user: str = Depends(current_user)) -> dict[str, Any]:
        return {"ok": True, "roles": [r.to_public_dict() for r in roles], "user": {"username": user}}

    # ------------------------------------------------------------------ 认证
    @app.post("/api/auth/login")
    def login(payload: LoginRequest, request: Request) -> JSONResponse:
        username = (payload.username or "").strip()
        if not username:
            raise HTTPException(status_code=400, detail="用户名不能为空")
        memory = chat.get_memory()
        user = memory.set_user(username)
        token = memory.create_token(username)
        resp = JSONResponse({
            "ok": True,
            "token": token,
            "user": user,
            "is_admin": _is_admin(request, username),
            "expires_in": cfg.redis.token_ttl,
        })
        resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=cfg.redis.token_ttl)
        return resp

    @app.get("/api/auth/me")
    def me(request: Request, user: str = Depends(current_user)) -> dict[str, Any]:
        return {"ok": True, "user": {"username": user, "display_name": user}, "is_admin": _is_admin(request, user)}

    @app.post("/api/auth/logout")
    def logout(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        token = _token_from(request, authorization)
        if token:
            try:
                chat.get_memory().delete_token(token)
            except Exception:  # noqa: BLE001
                pass
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE_NAME)
        return resp

    # ------------------------------------------------------------------ 问答
    @app.post("/api/chat")
    def chat_endpoint(payload: ChatRequest, user: str = Depends(current_user)):
        if not payload.stream:
            result = chat.answer(
                payload.question, user_id=user, role_id=payload.role_id,
                session_id=payload.session_id, top_k=payload.top_k,
                mode=payload.mode, rerank=payload.rerank,
            )
            return {"ok": True, "result": result}

        def event_stream() -> Iterator[str]:
            try:
                for event in chat.stream_answer(
                    payload.question, user_id=user, role_id=payload.role_id,
                    session_id=payload.session_id, top_k=payload.top_k,
                    mode=payload.mode, rerank=payload.rerank,
                ):
                    yield _sse(str(event.get("type", "message")), event)
            except Exception as exc:  # noqa: BLE001 - 安全网
                logger.exception("SSE 流异常")
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    # ------------------------------------------------------------------ 会话
    @app.get("/api/sessions")
    def list_sessions(user: str = Depends(current_user)) -> dict[str, Any]:
        return {"ok": True, "sessions": chat.get_memory().list_sessions(user)}

    @app.post("/api/sessions")
    def create_session(payload: SessionCreate, user: str = Depends(current_user)) -> dict[str, Any]:
        session_id = chat.get_memory().create_session(user, payload.role_id, title=payload.title)
        return {"ok": True, "session_id": session_id}

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
        memory = chat.get_memory()
        meta = memory.session_meta(session_id)
        if not meta:
            raise HTTPException(status_code=404, detail="会话不存在")
        if str(meta.get("user_id")) != user:
            raise HTTPException(status_code=403, detail="无权访问该会话")
        return {"ok": True, "session": meta, "messages": memory.messages(session_id)}

    @app.get("/api/sessions/{session_id}/messages")
    def session_messages(session_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
        memory = chat.get_memory()
        meta = memory.session_meta(session_id)
        if not meta or str(meta.get("user_id")) != user:
            raise HTTPException(status_code=404, detail="会话不存在或无权访问")
        return {"ok": True, "messages": memory.messages(session_id)}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
        memory = chat.get_memory()
        meta = memory.session_meta(session_id)
        if not meta:
            raise HTTPException(status_code=404, detail="会话不存在")
        if str(meta.get("user_id")) != user:
            raise HTTPException(status_code=403, detail="无权删除该会话")
        memory.delete_session(session_id)
        return {"ok": True, "deleted": session_id}

    # ------------------------------------------------------------------ 评估
    @app.post("/api/evaluate")
    def evaluate(payload: EvaluateRequest, user: str = Depends(current_user)) -> dict[str, Any]:
        contexts = payload.retrieved_contexts
        if contexts is None:
            hits = chat.retrieve(payload.question)
            contexts = [hit.text for hit in hits]
        from .ragas_eval import RagasEvaluator  # 懒加载，避免 import rag2 强依赖 ragas

        evaluator = RagasEvaluator(
            llm_model=cfg.llm.model,
            llm_base_url=cfg.llm.base_url,
            llm_api_key=cfg.llm.api_key,
            embed_model=cfg.milvus.embed_model,
            device=cfg.milvus.device,
        )
        scores = evaluator.evaluate(
            question=payload.question,
            answer=payload.answer,
            retrieved_contexts=contexts,
            reference=payload.reference,
            metrics=payload.metrics,
        )
        return {"ok": True, "scores": scores}

    # ------------------------------------------------------------------ 知识库
    @app.get("/api/kb/status")
    def kb_status(user: str = Depends(current_user)) -> dict[str, Any]:
        retriever = chat.get_retriever()
        return {
            "ok": True,
            "milvus": retriever.ping(),
            "rows": retriever.count(),
            "collection": cfg.milvus.collection,
            "roles": [r.to_public_dict() for r in roles],
        }

    # ------------------------------------------------------------------ 管理员后台
    def _editable_config() -> dict[str, Any]:
        return {
            "app": {"host": cfg.app.host, "port": cfg.app.port},
            "llm": {
                "base_url": cfg.llm.base_url,
                "api_key": cfg.llm.api_key,
                "model": cfg.llm.model,
                "temperature": cfg.llm.temperature,
                "max_tokens": cfg.llm.max_tokens,
            },
            "retrieval": {
                "top_k": cfg.retrieval.top_k,
                "mode": cfg.retrieval.mode,
                "rerank": cfg.retrieval.rerank,
                "rerank_top_k": cfg.retrieval.rerank_top_k,
            },
            "memory": {"short_term_turns": cfg.memory.short_term_turns},
            "redis": {"history_ttl": cfg.redis.history_ttl, "token_ttl": cfg.redis.token_ttl},
            "admin": {"usernames": list(cfg.admin.usernames)},
        }

    @app.get("/api/admin/config")
    def admin_config(user: str = Depends(current_admin)) -> dict[str, Any]:
        return {"ok": True, "config": _editable_config(), "is_admin": True, "user": {"username": user}}

    @app.post("/api/admin/config")
    def admin_config_update(
        payload: dict[str, Any] = Body(...), user: str = Depends(current_admin)
    ) -> dict[str, Any]:
        errors = validate_updates(payload)
        if errors:
            raise HTTPException(status_code=400, detail="；".join(errors))
        apply_updates(cfg, payload)
        chat.apply_config(cfg)
        save_config(cfg)
        logger.info("管理员 %s 更新了配置：%s", user, sorted(payload.keys()))
        return {"ok": True, "config": _editable_config()}

    @app.post("/api/admin/restart")
    def admin_restart(user: str = Depends(current_admin)) -> dict[str, Any]:
        """重启服务（改端口/地址后调用）。仅在 run.py 监督模式下能自动重启。"""
        triggered = request_restart()
        logger.info("管理员 %s 触发了服务重启（自动=%s）", user, triggered)
        return {"ok": True, "restarting": triggered, "hint": "服务将在数秒内重启" if triggered else "未在 run.py 监督模式下运行，请手动重启以生效"}

    @app.get("/api/admin/roles")
    def admin_roles(user: str = Depends(current_admin)) -> dict[str, Any]:
        return {"ok": True, "roles": [r.to_dict() for r in roles]}

    @app.post("/api/admin/roles")
    def admin_role_create(payload: RolePayload, user: str = Depends(current_admin)) -> dict[str, Any]:
        role_id = (payload.role_id or "").strip()
        if not role_id:
            raise HTTPException(status_code=400, detail="role_id 不能为空")
        if any(r.role_id == role_id for r in roles):
            raise HTTPException(status_code=409, detail=f"角色已存在：{role_id}")
        role = role_from_dict(payload.model_dump())
        roles.append(role)
        chat.set_roles(roles)
        save_roles(roles)
        logger.info("管理员 %s 新增角色：%s", user, role_id)
        return {"ok": True, "role": role.to_dict()}

    @app.put("/api/admin/roles/{role_id}")
    def admin_role_update(role_id: str, payload: RolePayload, user: str = Depends(current_admin)) -> dict[str, Any]:
        if not any(r.role_id == role_id for r in roles):
            raise HTTPException(status_code=404, detail=f"角色不存在：{role_id}")
        data = payload.model_dump()
        data["role_id"] = role_id  # 以路径为准
        updated = role_from_dict(data)
        roles[:] = [updated if r.role_id == role_id else r for r in roles]
        chat.set_roles(roles)
        save_roles(roles)
        logger.info("管理员 %s 更新角色：%s", user, role_id)
        return {"ok": True, "role": updated.to_dict()}

    @app.delete("/api/admin/roles/{role_id}")
    def admin_role_delete(role_id: str, user: str = Depends(current_admin)) -> dict[str, Any]:
        if not any(r.role_id == role_id for r in roles):
            raise HTTPException(status_code=404, detail=f"角色不存在：{role_id}")
        if len(roles) <= 1:
            raise HTTPException(status_code=400, detail="至少保留一个角色")
        roles[:] = [r for r in roles if r.role_id != role_id]
        chat.set_roles(roles)
        save_roles(roles)
        logger.info("管理员 %s 删除角色：%s", user, role_id)
        return {"ok": True, "deleted": role_id}

    # ------------------------------------------------------------------ 前端
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(str(WEB_DIR / "index.html"))

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> Response:
            return Response(status_code=204)

    return app


app = create_app()


def main() -> None:  # pragma: no cover - 便于 python -m rag2.server 启动
    import uvicorn

    cfg = load_config()
    setup_logging(level=cfg.app.log_level, log_dir=cfg.app.log_dir)
    uvicorn.run("rag2.server:app", host=cfg.app.host, port=cfg.app.port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
