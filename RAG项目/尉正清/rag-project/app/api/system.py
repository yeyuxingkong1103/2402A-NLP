# app/api/system.py
"""用户与系统接口：注册登录、健康检查、业务统计。"""
import hashlib
import hmac
import json
import os

from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db, get_milvus, get_redis, mysql_conn, redis_conn
from app.models.tables import ChatSession, Message, Role, User
from app.schemas import LoginRequest, RegisterRequest, Resp
import logging

logger = logging.getLogger(__name__)

users_router = APIRouter(prefix="/api/users", tags=["用户"])
system_router = APIRouter(prefix="/api", tags=["系统"])

_ITER = 200_000


def hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITER)
    return "%s$%s" % (salt.hex(), dk.hex())


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt_hex), _ITER)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:                                       # pragma: no cover
        return False


# ==================== 用户 ====================
@users_router.post("/register", response_model=Resp, summary="注册")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    exists = db.execute(select(User).where(
        User.username == req.username)).scalars().first()
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")

    user = User(username=req.username,
                password_hash=hash_password(req.password),
                nickname=req.nickname or req.username)
    db.add(user)
    try:
        # 先查后插不是原子操作，并发下仍可能撞唯一索引；
        # 依赖数据库约束兜底，把它翻译成 400 而不是 500
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="用户名已存在")

    logger.info("新用户注册: %s", req.username)
    return Resp(msg="注册成功", data=user.to_dict())


@users_router.post("/login", response_model=Resp, summary="登录")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(
        User.username == req.username)).scalars().first()
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not user.status:
        raise HTTPException(status_code=403, detail="账号已停用")
    return Resp(msg="登录成功", data=user.to_dict())


@users_router.get("/{user_id}", response_model=Resp, summary="查询用户")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return Resp(data=user.to_dict())


# ==================== 系统 ====================
@system_router.get("/health", response_model=Resp, summary="各依赖健康状态")
async def health():
    status = {"mysql": mysql_conn.healthcheck()}

    try:
        get_redis().ping()
        status["redis"] = True
    except Exception:
        status["redis"] = False

    try:
        get_milvus().list_collections()
        status["milvus"] = True
    except Exception:
        status["milvus"] = False

    status["llm"] = bool(settings.LLM_API_KEY)
    status["llm_model"] = settings.LLM_MODEL
    healthy = all(v for v in status.values() if isinstance(v, bool))
    return Resp(data=status, msg="ok" if healthy else "存在不可用依赖")


CACHE_KEY_STATS = "cache:stats"
STATS_TTL = 30


@system_router.get("/stats", response_model=Resp, summary="知识库与业务统计")
# 结果缓存 30 秒，refresh=true 跳过缓存
async def stats(refresh: bool = False, db: Session = Depends(get_db)):
    """统计要扫 Milvus 全量，结果用 Redis String 缓存 30 秒。

    刚入库完想看最新数字，加 ?refresh=true 跳过缓存。
    """
    if not refresh:
        cached = redis_conn.cache_get(CACHE_KEY_STATS)
        if cached:
            return Resp(data=json.loads(cached), msg="ok (cached)")

    milvus = get_milvus()
    collections = {}
    for name in (settings.MILVUS_COLLECTION, settings.MILVUS_MEMORY_COLLECTION):
        try:
            collections[name] = milvus.count(name)
        except Exception:
            collections[name] = 0

    per_role = {}
    sources_by_role = {}
    try:
        # 用翻页取全量：知识库超过 16384 条时单次查询会被 Milvus 拒绝
        rows = milvus.query_all(settings.MILVUS_COLLECTION, 'role_id != ""',
                                output_fields=["role_id", "source"])
        for r in rows:
            k = r.get("role_id", "unknown")
            per_role[k] = per_role.get(k, 0) + 1
            sources_by_role.setdefault(k, set()).add(r.get("source", ""))
    except Exception:
        pass

    def _count(model):
        return db.execute(select(func.count()).select_from(model)).scalar() or 0

    # 各角色已入库的来源文件：优先读 Redis Set（快），
    # 但 Redis 重启可能丢数据，此时从 Milvus 重建 —— Milvus 才是权威记录
    ingested = {}
    for role, srcs in sources_by_role.items():
        cached = redis_conn.ingested_sources(role)
        if not cached:
            cached = sorted(s for s in srcs if s)
            redis_conn.mark_ingested(role, cached)
        ingested[role] = cached

    data = {
        "collections": collections,
        "knowledge_per_role": per_role,
        "ingested_sources": ingested,
        "users": _count(User),
        "roles": _count(Role),
        "sessions": _count(ChatSession),
        "messages": _count(Message),
    }
    redis_conn.cache_set(CACHE_KEY_STATS,
                         json.dumps(data, ensure_ascii=False), ttl=STATS_TTL)
    return Resp(data=data)


ROUTERS = [users_router, system_router]
