"""密码哈希与 JWT 鉴权。

职责边界：本模块只管"算哈希 / 签发解析 token"，不做数据库查询、不管 HTTP。
这样 core 层不依赖 db 层（唯一例外是吊销时的延迟导入，见 revoke_token），
既能被中间件复用，也方便单元测试脱离数据库运行。
"""
import datetime as dt
import uuid
from typing import Any, Dict, Optional

import bcrypt
import jwt

from src.core.config import settings


def hash_password(password: str) -> str:
    """用 bcrypt 生成不可逆的密码哈希（自带盐，结果已含盐值）。

    bcrypt 每次 gensalt 都产生随机盐，因此同一密码两次哈希结果不同，
    校验必须用 verify_password 而非字符串比较。
    """
    # 夹逼到 [4,16]：轮数太低不安全，太高会让登录接口 CPU 打满（DoS 风险），
    # 这里做一次兜底，防止 .env 被误改成一个极端值。
    rounds = max(4, min(16, settings.bcrypt_rounds))
    salt = bcrypt.gensalt(rounds=rounds)
    # bcrypt 只接受 bytes，故先 encode；返回 str 便于直接存入数据库 VARCHAR。
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """校验明文密码与已存哈希是否匹配，任何异常都返回 False 而不抛出。"""
    # 空值快速失败：避免把 None/空串喂给 bcrypt 触发异常，也防止"空密码"这种脏数据被误判通过。
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # 哈希格式损坏（如库里存了明文/被截断）时 bcrypt 会抛异常；
        # 登录接口不应因此 500，统一按"密码错误"处理更安全（不泄露内部状态）。
        return False


def _create_token(user_id: int, token_type: str, expires_minutes: int, extra: Optional[Dict[str, Any]] = None) -> str:
    """内部通用签发函数：access / refresh 共用，差异只在 type 与有效期。"""
    # 用 UTC 而非本地时间：避免服务器时区变更导致 exp 解析错乱，
    # JWT 标准 (RFC 7519) 也要求基于 UTC 计算 NumericDate。
    now = dt.datetime.now(dt.timezone.utc)
    payload: Dict[str, Any] = {
        # sub 必须是字符串：部分 JWT 库/校验器要求 sub 为 str，用 int 会踩兼容坑。
        "sub": str(user_id),
        "type": token_type,
        "jti": uuid.uuid4().hex,  # 唯一 ID，用于吊销（logout/改密后拉黑）
        "iat": int(now.timestamp()),  # 签发时间
        "exp": int((now + dt.timedelta(minutes=expires_minutes)).timestamp()),  # 过期时间
    }
    # extra 用于附加 roles 等业务声明，放在最后合并进 payload。
    # 注意：update 会覆盖同名键，所以 extra 只由本模块内部构造（外部走 create_access_token），
    # 不接收用户可控输入，避免伪造 sub/exp。
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: int, roles: Optional[list] = None) -> str:
    """签发短期访问令牌，并把角色列表塞进 payload 供鉴权中间件免查库判权限。"""
    return _create_token(
        user_id, "access", settings.jwt_access_token_expire_minutes, {"roles": roles or []}
    )


def create_refresh_token(user_id: int) -> str:
    """签发长期刷新令牌：只允许换取新的 access token，不携带角色等业务声明。"""
    return _create_token(user_id, "refresh", settings.jwt_refresh_token_expire_minutes)


def decode_token(token: str) -> Dict[str, Any]:
    """解析 JWT，失败时抛出 jwt.PyJWTError 子类。

    刻意不在这里 try/except 转成 AuthError：本函数属于底层工具，
    由调用方（鉴权依赖）决定把哪类失败映射成 401。
    注意 jwt.decode 默认会校验 exp，过期即抛 ExpiredSignatureError。
    """
    # algorithms 必须显式白名单：不传会让攻击者用 alg=none 伪造 token（经典 JWT 漏洞）。
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def token_expires_in() -> int:
    """返回 access token 剩余秒数，供前端决定何时静默刷新。"""
    return settings.jwt_access_token_expire_minutes * 60


def revoke_token(payload: Dict[str, Any]) -> bool:
    """按 payload 的 jti 吊销 token（写 Redis 黑名单，TTL=剩余有效期）。

    延迟导入 redis，避免 core 层对 db 层的静态依赖。
    """
    jti = payload.get("jti")
    # 没有 jti 的旧 token 无法加入黑名单，只能靠自然过期，故直接返回 False。
    if not jti:
        return False
    exp = int(payload.get("exp", 0))
    ttl = exp - int(dt.datetime.now(dt.timezone.utc).timestamp())
    # 已过期的 token 无需拉黑（Redis 里也没必要占位），直接视为吊销成功语义。
    if ttl <= 0:
        return False
    # 函数内导入而非模块顶部：切断 core → db 的导入链，防止循环依赖与启动期副作用。
    from src.db.redis import blacklist_token
    return blacklist_token(jti, ttl)