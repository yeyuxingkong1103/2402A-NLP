"""用户管理服务（MySQL）：注册、登录、资料、权限、日志。"""
import datetime as dt
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.core.exceptions import AuthError, ConflictError, NotFoundError
from src.core.logging import get_logger
from src.core.security import (create_access_token, create_refresh_token, decode_token,
                               hash_password, token_expires_in, verify_password)
from src.db import redis as redis_db
from src.models import AuditLog, LoginLog, User
from src.services import persona_service

logger = get_logger("service.user")


def _to_dict(user: User, roles: Optional[List[str]] = None) -> Dict:
    """把 User ORM 对象转成普通字典，剥离 SQLAlchemy 状态，便于 JSON 序列化与写入 Redis 缓存。"""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "phone": user.phone,
        "nickname": user.nickname,
        "avatar": user.avatar,
        "status": user.status,
        "roles": roles or [],
        "created_at": user.created_at.strftime("%Y-%m-%d %H:%M:%S") if user.created_at else None,
        "last_login_at": user.last_login_at.strftime("%Y-%m-%d %H:%M:%S") if user.last_login_at else None,
    }


def add_audit_log(db: Session, user_id: Optional[int], action: str,
                  detail: str = "", ip: Optional[str] = None) -> None:
    """写入审计日志（独立事务）。审计是旁路能力，失败绝不能拖垮登录/注册等主流程，故整体 try/except 兜底。"""
    try:
        db.add(AuditLog(user_id=user_id, action=action, detail=detail[:4000], ip=ip))
        db.commit()
    except Exception as exc:  # 审计失败不影响主流程
        db.rollback()
        logger.warning("写入审计日志失败：%s", exc)


def register(db: Session, username: str, password: str, email: Optional[str] = None,
             phone: Optional[str] = None, nickname: Optional[str] = None,
             is_admin: bool = False) -> User:
    # 事务边界：落库前先显式查重，把"用户名/邮箱已占用"转化为友好业务报错，
    # 而不是等数据库唯一约束抛异常（那样报错信息对用户不友好且难定位）。
    exists = db.execute(select(User).where(User.username == username)).scalars().first()
    if exists:
        raise ConflictError(f"用户名已存在：{username}")
    if email:
        email_exists = db.execute(select(User).where(User.email == email)).scalars().first()
        if email_exists:
            raise ConflictError(f"邮箱已被注册：{email}")

    user = User(
        username=username,
        password_hash=hash_password(password),
        email=email,
        phone=phone,
        nickname=nickname or username,
        status=1,
    )
    # 密码只存哈希（hash_password），绝不存明文；commit 后 refresh 拿到自增主键 id。
    db.add(user)
    db.commit()
    db.refresh(user)
    # 角色与用户分表存储（RBAC），注册后按 is_admin 分配基础角色，后续鉴权依赖它。
    persona_service.assign_role(db, user.id, "admin" if is_admin else "user")
    logger.info("用户注册成功：%s(id=%s)", username, user.id)
    return user


def login(db: Session, username: str, password: str, ip: Optional[str] = None,
          user_agent: Optional[str] = None) -> Dict:
    user = db.execute(select(User).where(User.username == username)).scalars().first()
    # 用户不存在与密码错误统一返回同一文案：避免攻击者通过不同报错枚举出"用户名是否存在"。
    if not user or not verify_password(password, user.password_hash):
        add_audit_log(db, user.id if user else None, "login_failed", f"username={username}", ip)
        raise AuthError("用户名或密码错误")
    # status=1 才是正常账号；被禁用账号即使密码正确也拒绝登录。
    if user.status != 1:
        raise AuthError("账号已被禁用，请联系管理员")

    # 事务边界：更新 last_login_at 与写登录日志放在同一次 commit 中原子提交，保证一致性。
    user.last_login_at = dt.datetime.now()
    db.add(LoginLog(user_id=user.id, ip=ip, user_agent=(user_agent or "")[:512]))
    db.commit()
    db.refresh(user)

    roles = persona_service.get_user_roles(db, user.id)
    # 登录成功即预热 Redis 用户缓存，后续 get_profile 直接命中，减少 MySQL 压力。
    redis_db.cache_user_profile(user.id, _to_dict(user, roles))
    add_audit_log(db, user.id, "login", f"username={username}", ip)
    logger.info("用户登录成功：%s", username)
    return {
        "access_token": create_access_token(user.id, roles),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
        "expires_in": token_expires_in(),
        "user_id": user.id,
        "username": user.username,
        "roles": roles,
    }


def refresh_token(db: Session, refresh_token_str: str) -> Dict:
    import jwt as pyjwt
    try:
        payload = decode_token(refresh_token_str)
    except pyjwt.PyJWTError as exc:
        raise AuthError(f"Refresh Token 无效或已过期：{exc}") from exc
    # 换发前两道校验：token 类型必须是 refresh（防止拿 access token 来换），
    # 且 jti 未被拉黑（登出/泄露后主动失效的 token 不能再换发）。
    if payload.get("type") != "refresh":
        raise AuthError("Token 类型错误")
    if redis_db.is_token_blacklisted(payload.get("jti", "")):
        raise AuthError("Refresh Token 已失效，请重新登录")
    user = get_user(db, int(payload["sub"]))
    roles = persona_service.get_user_roles(db, user.id)
    return {
        "access_token": create_access_token(user.id, roles),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
        "expires_in": token_expires_in(),
        "user_id": user.id,
        "username": user.username,
        "roles": roles,
    }


def get_user(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if not user:
        raise NotFoundError(f"用户不存在：{user_id}")
    return user


def get_profile(db: Session, user_id: int, use_cache: bool = True) -> Dict:
    # 读路径优先 Redis 命中，避免每次请求都打 MySQL；默认角色可能随时被用户切换，
    # 故命中缓存时仍实时补算 default_persona_id，避免返回过期值。
    if use_cache:
        cached = redis_db.get_cached_user_profile(user_id)
        if cached:
            cached["default_persona_id"] = persona_service.get_default_persona_id(db, user_id)
            return cached
    user = get_user(db, user_id)
    data = _to_dict(user, persona_service.get_user_roles(db, user_id))
    data["default_persona_id"] = persona_service.get_default_persona_id(db, user_id)
    redis_db.cache_user_profile(user_id, data)
    return data


def update_profile(db: Session, user_id: int, payload: Dict) -> Dict:
    user = get_user(db, user_id)
    default_persona_id = payload.pop("default_persona_id", None)
    for key, value in payload.items():
        if value is not None and hasattr(user, key):
            setattr(user, key, value)
    db.commit()
    db.refresh(user)
    if default_persona_id:
        persona_service.set_default_persona(db, user_id, default_persona_id)
    # 资料已变更，主动失效 Redis 缓存，下次读回源 MySQL 重建，避免缓存脏数据。
    redis_db.invalidate_user_profile(user_id)
    logger.info("更新用户资料：%s", user_id)
    return get_profile(db, user_id, use_cache=False)


def change_password(db: Session, user_id: int, old_password: str, new_password: str) -> None:
    user = get_user(db, user_id)
    # 先校验原密码，防止会话被他人接管后直接改密；新密码同样只存哈希。
    if not verify_password(old_password, user.password_hash):
        raise AuthError("原密码错误")
    user.password_hash = hash_password(new_password)
    db.commit()
    add_audit_log(db, user_id, "change_password")
    logger.info("用户修改密码：%s", user_id)


def list_users(db: Session, page: int = 1, page_size: int = 20,
               keyword: Optional[str] = None, status: Optional[int] = None) -> Dict:
    stmt = select(User)
    count_stmt = select(func.count(User.id))
    if keyword:
        like = f"%{keyword}%"
        condition = (User.username.like(like)) | (User.nickname.like(like)) | (User.email.like(like))
        stmt = stmt.where(condition)
        count_stmt = count_stmt.where(condition)
    if status is not None:
        stmt = stmt.where(User.status == status)
        count_stmt = count_stmt.where(User.status == status)

    total = int(db.execute(count_stmt).scalar() or 0)
    stmt = stmt.order_by(User.id.desc()).offset((page - 1) * page_size).limit(page_size)
    users = list(db.execute(stmt).scalars().all())
    items = [_to_dict(u, persona_service.get_user_roles(db, u.id)) for u in users]
    return {"total": total, "page": page, "page_size": page_size, "items": items}


def set_user_status(db: Session, user_id: int, status: int) -> User:
    user = get_user(db, user_id)
    user.status = 1 if status else 0
    db.commit()
    db.refresh(user)
    # 状态影响登录与展示，必须同步失效用户缓存，让禁用立即生效。
    redis_db.invalidate_user_profile(user_id)
    logger.info("用户状态变更：%s -> %s", user_id, status)
    return user


def list_login_logs(db: Session, user_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
    stmt = select(LoginLog).order_by(LoginLog.id.desc()).limit(limit)
    if user_id:
        stmt = stmt.where(LoginLog.user_id == user_id)
    return [
        {
            "id": row.id,
            "user_id": row.user_id,
            "ip": row.ip,
            "user_agent": row.user_agent,
            "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None,
        }
        for row in db.execute(stmt).scalars().all()
    ]


def list_audit_logs(db: Session, user_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)
    return [
        {
            "id": row.id,
            "user_id": row.user_id,
            "action": row.action,
            "detail": row.detail,
            "ip": row.ip,
            "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None,
        }
        for row in db.execute(stmt).scalars().all()
    ]