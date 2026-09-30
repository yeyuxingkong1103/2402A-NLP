from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymilvus import MilvusClient
from sqlalchemy import select, text

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.evaluations import router as evaluations_router
from app.api.memories import router as memories_router
from app.api.auth import router as auth_router
from app.api.history import router as history_router
from app.api.roles import router as roles_router
from app.api.retrieval import router as retrieval_router
from app.core.config import get_settings
from app.memory.redis_memory import get_chat_memory
from app.models.chat import AIRole
from app.db.session import get_engine, get_session_factory

load_dotenv()
settings = get_settings()

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(history_router)
app.include_router(roles_router)
app.include_router(documents_router)
app.include_router(evaluations_router)
app.include_router(memories_router)


def initialize_roles() -> None:
    from app.api.roles import DEFAULT_ROLES
    from datetime import datetime

    db = get_session_factory()()
    try:
        for item in DEFAULT_ROLES:
            if db.scalar(select(AIRole).where(AIRole.role_id == item["role_id"])):
                continue
            db.add(AIRole(**item, is_active=True, created_at=datetime.utcnow(), updated_at=datetime.utcnow()))
        db.commit()
    finally:
        db.close()


initialize_roles()
app.include_router(retrieval_router)


@app.get("/health")
def health() -> JSONResponse:
    checks: dict[str, object] = {
        "mysql": False,
        "redis": False,
        "milvus": False,
        "knowledge_collection": False,
    }
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["mysql"] = True
    except Exception as exc:
        checks["mysql_error"] = type(exc).__name__
    try:
        checks["redis"] = get_chat_memory().ping()
    except Exception as exc:
        checks["redis_error"] = type(exc).__name__
    try:
        client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        checks["milvus"] = bool(client.list_collections() is not None)
        checks["knowledge_collection"] = client.has_collection(
            settings.milvus_collection_knowledge
        )
    except Exception as exc:
        checks["milvus_error"] = type(exc).__name__
    checks["status"] = "ok" if all(
        checks.get(name) is True for name in ("mysql", "redis", "milvus", "knowledge_collection")
    ) else "degraded"
    checks["service"] = "mentalheal-rag"
    return JSONResponse(
        status_code=200 if checks["status"] == "ok" else 503,
        content=checks,
    )
