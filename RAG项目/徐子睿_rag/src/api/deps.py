"""src/api/deps.py —— 认证与依赖注入：口令哈希、JWT 签发/校验、当前用户解析。

在链路中的位置：
    各业务路由（src/api/routers/*）通过 FastAPI 的 Depends 机制调用本文件的 current_user，
    由它解出"这次请求是谁发的"，路由再基于这个身份做权限判断。

三块内容：
    口令哈希   hash_password / verify_password —— PBKDF2-HMAC-SHA256
    JWT 签发   create_access_token
    身份解析   current_user（必需认证）/ optional_user（可选认证）

安全设计要点：
    口令用 PBKDF2 加盐哈希 + 常量时间比较，绝不明文存储、也不用裸 SHA256
    JWT 用配置里的密钥签名，过期时间由配置控制
    开发用的 X-User-Id 旁路在 environment == "prod" 时被强制关闭
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from configs.settings import get_settings
from src.models.database import User, db_session

# auto_error=False 让缺少凭证时**不自动报 403**，而是把 None 交给 current_user 处理。
# 这是为了给开发用的 X-User-Id 旁路留出判断空间（见 current_user）。
bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """用 PBKDF2-HMAC-SHA256 对口令加盐哈希。

    参数：
        password: 用户明文口令
    返回：
        形如 "pbkdf2$120000${盐 hex}${摘要 hex}" 的字符串（存入 user.password_hash）。

    参数选择与理由：
        盐     secrets.token_bytes(16) —— 每个用户一个随机盐。
               没有盐的话，相同口令得到相同哈希，攻击者可以用彩虹表批量破解，
               也能从"两个用户哈希相同"推断出他们用了同一个口令
        迭代   120_000 次 —— 故意让计算变慢。
               哈希算法本身很快，而快正是暴力破解的有利条件；
               迭代十万次把单次校验成本抬到毫秒级，对正常登录无感，
               对暴力破解则是数万倍的代价
        算法   SHA-256（pbkdf2_hmac 的第一个参数）

    把参数（算法名、迭代次数、盐）一起编码进结果字符串：
        这样以后提高迭代次数时，老用户仍能用老参数正常校验（见 verify_password 会读出来），
        不需要强制所有人改口令。

    注意迭代次数在代码里出现两次（这里写 120_000，f-string 里写 "120000"），
    且 verify_password 是从字符串里解析出这个数字的 —— 两者必须保持一致，
    改一处忘了另一处会导致新旧口令校验行为不一致。
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"pbkdf2$120000${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """校验口令是否与存储的哈希匹配。

    参数：
        password: 待校验的明文口令
        encoded: 存储的哈希字符串（hash_password 的产物）
    返回：
        匹配返回 True，否则 False（包括格式异常的情况）。

    迭代次数从存储的字符串里解析（int(rounds)）：
        这让"历史口令用老参数、新口令用新参数"能共存 ——
        将来调高迭代次数时，老用户不必重置口令。

    用 hmac.compare_digest 而不是 == ：
        这是防"时序攻击"的标准做法。普通字符串比较在发现第一个不同字符时就返回，
        比较耗时与"前多少个字符相同"成正比 ——
        攻击者据此可以逐字节猜出正确哈希。compare_digest 的耗时与内容无关。

    捕获 (ValueError, TypeError) 返回 False：
        存储值格式不对（缺字段、盐不是合法 hex、轮数不是数字）时，
        应该判定为"校验失败"，而不是让异常冒出去变成 500 ——
        数据损坏不该表现为服务错误。
    """
    try:
        _, rounds, salt, expected = encoded.split("$", 3)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(digest.hex(), expected)
    except (ValueError, TypeError):
        return False


def create_access_token(user: User) -> str:
    """为用户签发 JWT 访问令牌。

    参数：
        user: 用户对象
    返回：
        签好名的 JWT 字符串。

    载荷三个字段：
        sub        用户 id（JWT 规范里的主体字段，注意必须是字符串 ——
                   规范要求 sub 为 StringOrURI，直接放 int 会被严格实现拒绝）
        tenant_id  租户 id。放进令牌里，后续请求就不必再查库取租户，也避免
                   被客户端篡改（载荷有签名保护）
        exp        过期时间（UTC）。用 datetime.now(timezone.utc) 而不是本地时间 ——
                   JWT 的 exp 按 UTC 解释，用本地时间会让有效期随服务器时区偏移。

    过期时间由配置的 access_token_minutes 控制，不硬编码：
        开发时可以调长避免频繁重登，生产环境应调短以缩小令牌泄露的窗口。
    """
    settings = get_settings()
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    return jwt.encode({"sub": str(user.id), "tenant_id": user.tenant_id, "exp": expires}, settings.secret_key, algorithm=settings.jwt_algorithm)


def current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> User:
    """解析当前请求的用户身份（路由用它做依赖注入）。

    参数：
        request: 请求对象（用于读 X-User-Id 头）
        credentials: HTTPBearer 解析出的 Bearer 凭证，可能为 None
    返回：
        数据库里的 User 对象。
    异常：
        无凭证或凭证无效 -> HTTPException 401；用户不存在 -> 401。

    两条解析路径：
        1. 开发旁路（X-User-Id 请求头）—— 只在 allow_dev_header_auth 打开
           且 environment != "prod" 时生效。方便本地用 curl 调试，不必先登录取令牌
        2. 正常路径（Bearer JWT）—— 解码、验签、取出 sub 再去库里确认用户存在

    为什么旁路要加两道门（配置开关 + 环境判断）：
        仅靠配置开关的话，一个配置错误（比如生产环境的 .env 误拷了开发配置）
        就会让任何人伪造一个用户 id 直接通过认证。
        再叠一层"环境必须是 prod 之外"的判断，能让这类失误在生产环境失效。
        `user_id.isdigit()` 是在读库前先筛掉明显非法的输入。

    为什么还要回库里查一次用户（而不是直接信任 JWT 里的 sub）：
        令牌签发后用户可能已被删除。只验签的话，一个已删除用户的旧令牌
        在过期前仍能正常使用，会造成各种"幽灵用户"数据。
        查库这一步把"令牌有效"和"用户真实存在"两件事对齐了。

    捕获 (jwt.InvalidTokenError, KeyError, ValueError)：
        分别对应"签名/过期无效""载荷缺 sub""sub 不是合法整数"三种异常输入，
        统一转成 401 而不是让它们变成 500。
    """
    settings = get_settings()
    if credentials is None:
        user_id = request.headers.get("X-User-Id") if settings.allow_dev_header_auth and settings.environment != "prod" else None
        if user_id and user_id.isdigit():
            with db_session() as db:
                user = db.get(User, int(user_id))
                if user:
                    return user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="需要 Bearer JWT")
    try:
        payload = jwt.decode(credentials.credentials, settings.secret_key, algorithms=[settings.jwt_algorithm])
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="JWT 无效或已过期") from exc
    with db_session() as db:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")
        return user


def optional_user(request: Request) -> User | None:
    """尝试解析当前用户，未登录时返回 None 而不报错。

    参数：
        request: 请求对象
    返回：
        User 对象，或 None。

    用途：
        让某些接口"登录与否都能用"（例如匿名体验、公开的知识库浏览）。
        这类接口拿到 None 时应按匿名用户处理。

    实现上复用 current_user 并吞掉 HTTPException：
        好处是认证逻辑只有一份，不会出现"可选认证"路径漏了某个校验。
        注意这里传 credentials=None —— 意味着本函数**只走开发旁路**，
        不会解析请求头里的 JWT。如果将来需要"可选但有 JWT 就用"，
        这里要改成从 request.headers 里自行解析 Authorization 头。
    """
    try:
        return current_user(request, None)
    except HTTPException:
        return None
