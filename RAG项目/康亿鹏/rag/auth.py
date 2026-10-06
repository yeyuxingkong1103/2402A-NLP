"""用户认证逻辑：注册、登录、JWT 令牌。"""  # 模块说明
import logging  # 运行日志
from datetime import datetime, timedelta  # 时间计算

import bcrypt  # 密码哈希库（原生 API，不依赖 passlib）
import jwt  # PyJWT：编码/解码 JWT 令牌
from fastapi import Depends, HTTPException, status  # FastAPI 异常与依赖
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # Bearer 令牌提取
from sqlalchemy.orm import Session  # SQLAlchemy 会话类型

import config  # 全局配置
from database import User, get_db  # 用户模型与数据库依赖

logger = logging.getLogger("rag.auth")  # 本模块日志器

_security = HTTPBearer()  # Bearer 令牌提取器


def hash_password(password: str) -> str:  # 把明文密码哈希为 bcrypt
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:  # 验证明文密码与哈希是否匹配
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def register_user(db: Session, username: str, password: str) -> User:  # 注册新用户
    """创建用户记录；用户名重复时抛 400。"""
    existing = db.query(User).filter(User.username == username).first()  # 查重
    if existing:
        logger.warning("注册被拒：用户名已存在 username=%s", username)  # 业务失败原因
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "用户名已存在")

    user = User(  # 创建用户对象（普通用户，不再绑定角色）
        username=username,
        password_hash=hash_password(password),
    )
    db.add(user)  # 加入会话
    db.commit()  # 提交到数据库
    db.refresh(user)  # 刷新拿到自增 ID
    logger.info("注册成功 user_id=%s username=%s", user.id, username)  # 成功日志
    return user


def login_user(db: Session, username: str, password: str) -> str:  # 登录并返回 JWT
    """验证密码，成功则返回 JWT 令牌。"""
    user = db.query(User).filter(User.username == username).first()  # 查用户
    if not user or not verify_password(password, user.password_hash):
        logger.warning("登录失败：用户名或密码错误 username=%s", username)  # 只记用户名，绝不记密码
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    logger.info("登录成功 user_id=%s username=%s", user.id, username)  # 成功日志
    return create_token(user)


def create_token(user: User) -> str:  # 生成 JWT 令牌
    payload = {  # 令牌载荷
        "user_id": user.id,
        "username": user.username,
        "exp": datetime.utcnow() + timedelta(hours=config.jwt_expire_hours),
    }
    return jwt.encode(payload, config.jwt_secret, algorithm="HS256")  # 编码并签名


def decode_token(token: str) -> dict:  # 解码并验证 JWT 令牌
    try:
        return jwt.decode(token, config.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "令牌已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "令牌无效")


def get_current_user(  # FastAPI 依赖：从请求头提取 token，返回用户信息
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """从 Bearer token 解析当前用户，返回 {user_id, username}。"""
    payload = decode_token(credentials.credentials)
    return {
        "user_id": payload["user_id"],
        "username": payload["username"],
    }
