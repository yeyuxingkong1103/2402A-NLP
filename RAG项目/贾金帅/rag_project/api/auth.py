"""Authentication storage, dependencies, and routes."""
import hashlib
import re
import secrets
import sqlite3
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src import config
from .dependencies import _AUTH_COOKIE
from .schemas import LoginRequest, RegisterRequest

router = APIRouter(tags=["auth"])


def _auth_db() -> sqlite3.Connection:  # 打开认证数据库连接
    conn = sqlite3.connect(str(config.AUTH_DB_PATH), timeout=10)  # 连接 users.db，避免长时间锁表
    conn.row_factory = sqlite3.Row  # 让查询结果可按列名访问
    return conn  # 返回连接


def _hash_password(password: str, salt: str) -> str:  # PBKDF2 密码哈希
    return hashlib.pbkdf2_hmac(  # 调用标准库 PBKDF2
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000  # 算法/密码/盐/迭代次数
    ).hex()  # 转成十六进制字符串存储


def _init_auth_db() -> None:  # 初始化认证相关数据库表
    """建表；首次启动时自动创建演示账号 admin / 123456。"""
    config.AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # 确保数据库目录存在
    with _auth_db() as conn:  # 打开连接（with 自动提交事务）
        conn.execute(  # 创建用户表（不存在才创建）
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        conn.execute(  # 创建会话表（token 为主键）
            """CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )"""
        )
        conn.execute(  # 创建历史对话表（按用户关联）
            """CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'qa',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        conn.execute(  # 创建历史消息表
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                images TEXT,
                created_at REAL NOT NULL
            )"""
        )
        # 兼容旧库：messages 表缺 images 列时补上（存上传的图片/文件，JSON 数组）
        msg_cols = [row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "images" not in msg_cols:  # 没有该列
            conn.execute("ALTER TABLE messages ADD COLUMN images TEXT")  # 补列
        if config.AUTH_SEED_DEMO:  # 开启演示账号种子时
            demo = conn.execute(  # 查询是否已有 admin
                "SELECT id FROM users WHERE username = ?", ("admin",)
            ).fetchone()
            if demo is None:  # 还没有演示账号
                salt = secrets.token_hex(16)  # 生成随机盐
                conn.execute(  # 插入演示管理员账号
                    "INSERT INTO users (username, password_hash, salt, display_name, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("admin", _hash_password("123456", salt), salt, "演示管理员", time.time()),
                )
                print("[AUTH] 演示账号已创建：admin / 123456")  # 控制台提示


def _create_session(user_id: int) -> str:  # 为用户创建一个登录会话
    token = secrets.token_urlsafe(32)  # 生成 32 字节随机 token
    now = time.time()  # 当前时间戳
    with _auth_db() as conn:  # 打开数据库
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))  # 顺带清理过期会话
        conn.execute(  # 插入新会话记录
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + config.AUTH_TOKEN_TTL_SECONDS),
        )
    return token  # 返回 token 给前端保存


def _session_user(token: str) -> dict | None:  # 根据 token 查当前用户
    if not token:  # 没有 token
        return None  # 返回未登录
    with _auth_db() as conn:  # 打开数据库
        row = conn.execute(  # 联表查询：会话 → 用户，且未过期
            """SELECT u.id, u.username, u.display_name
               FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token = ? AND s.expires_at > ?""",
            (token, time.time()),
        ).fetchone()
    return dict(row) if row else None  # 命中返回用户信息，否则 None


def _extract_token(authorization: str) -> str:  # 从 Authorization 头里取出 token
    if authorization and authorization.lower().startswith("bearer "):  # 形如 "Bearer xxx"
        return authorization[7:].strip()  # 去掉前缀并去除空白
    return ""  # 没有则返回空串


def get_current_user(request: Request) -> dict | None:  # FastAPI 依赖：获取当前用户
    """返回当前登录用户；AUTH_ENABLED 且未登录/已过期时抛出 401。"""
    # 优先取请求头里的 token，其次取 Cookie（登录后自动带上）
    token = _extract_token(request.headers.get("Authorization", "")) or request.cookies.get(
        _AUTH_COOKIE, ""
    )
    user = _session_user(token)  # 解析 token 并查用户
    if user is None and config.AUTH_ENABLED:  # 开了登录校验但没登录
        raise HTTPException(status_code=401, detail="请先登录后使用")  # 返回 401
    return user  # 返回用户信息（未开启登录时可能为 None）


@router.post("/api/auth/register")  # 注册接口
def register(req: RegisterRequest):  # 接收注册数据
    username = req.username.strip()  # 去掉账号首尾空格
    password = req.password  # 原始密码
    if not re.fullmatch(r"[\w\u4e00-\u9fff.-]{3,32}", username):  # 校验账号格式（3-32 位）
        raise HTTPException(status_code=400, detail="账号需为 3-32 位字母、数字、下划线、点或中文")  # 格式不合法
    if not 6 <= len(password) <= 64:  # 校验密码长度
        raise HTTPException(status_code=400, detail="密码长度需在 6-64 位之间")  # 长度不合法
    display_name = req.display_name.strip() or username  # 没有显示名就用账号名
    salt = secrets.token_hex(16)  # 生成随机盐
    try:  # 尝试插入新用户
        with _auth_db() as conn:  # 打开数据库
            conn.execute(  # 插入用户（含密码哈希）
                "INSERT INTO users (username, password_hash, salt, display_name, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, _hash_password(password, salt), salt, display_name, time.time()),
            )
            user_id = conn.execute(  # 取出刚插入的用户 ID
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()["id"]
    except sqlite3.IntegrityError:  # 账号唯一约束冲突
        raise HTTPException(status_code=409, detail="该账号已存在")  # 提示账号重复
    token = _create_session(user_id)  # 注册成功后直接创建登录会话
    response = JSONResponse({  # 返回 token 和用户信息
        "token": token,  # 登录凭证
        "user": {"id": user_id, "username": username, "display_name": display_name},  # 用户信息
    })
    response.set_cookie(  # 同时写入登录 Cookie，服务端据此强制登录
        _AUTH_COOKIE,
        token,
        httponly=True,  # 前端 JS 不可读，防 XSS 窃取
        max_age=config.AUTH_TOKEN_TTL_SECONDS,  # 与 token 有效期一致
        samesite="lax",  # 同站请求携带
    )
    return response


@router.post("/api/auth/login")  # 登录接口
def login(req: LoginRequest):  # 接收登录数据
    username = req.username.strip()  # 去掉账号首尾空格
    with _auth_db() as conn:  # 打开数据库
        row = conn.execute(  # 按账号查用户
            "SELECT id, username, password_hash, salt, display_name FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None or _hash_password(req.password, row["salt"]) != row["password_hash"]:  # 账号不存在或密码不对
        raise HTTPException(status_code=401, detail="账号或密码错误")  # 统一提示，避免泄露账号是否存在
    token = _create_session(row["id"])  # 创建登录会话
    response = JSONResponse({  # 返回 token 和用户信息
        "token": token,  # 登录凭证
        "user": {  # 用户信息
            "id": row["id"],
            "username": row["username"],
            "display_name": row["display_name"],
        },
    })
    response.set_cookie(  # 写入登录 Cookie
        _AUTH_COOKIE,
        token,
        httponly=True,
        max_age=config.AUTH_TOKEN_TTL_SECONDS,
        samesite="lax",
    )
    return response


@router.post("/api/auth/logout")  # 登出接口
def logout(request: Request):  # 接收请求对象
    # 优先取请求头 token，其次取 Cookie
    token = _extract_token(request.headers.get("Authorization", "")) or request.cookies.get(
        _AUTH_COOKIE, ""
    )
    if token:  # 有 token 才需要删
        with _auth_db() as conn:  # 打开数据库
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))  # 删除该会话
    response = JSONResponse({"ok": True})  # 返回成功
    response.delete_cookie(_AUTH_COOKIE)  # 清除登录 Cookie
    return response


@router.get("/api/auth/me")  # 查询当前登录状态接口
def auth_me(request: Request):  # 接收请求对象
    # 优先取请求头 token，其次取 Cookie
    token = _extract_token(request.headers.get("Authorization", "")) or request.cookies.get(
        _AUTH_COOKIE, ""
    )
    user = _session_user(token)  # 查当前用户
    if user:  # 已登录
        return {"authenticated": True, "user": user}  # 返回用户信息
    if config.AUTH_ENABLED:  # 开了登录校验但未登录
        raise HTTPException(status_code=401, detail="未登录")  # 返回 401
    return {"authenticated": False, "user": None}  # 未开启登录时返回未登录状态
