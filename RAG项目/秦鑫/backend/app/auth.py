import logging
import re
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .core import expires_after, hash_password, new_token, verify_password
from .storage.redis import RedisStore
from .storage.mysql import AuthSessionStore, DEFAULT_PROFILE, DEFAULT_SETTINGS, UserStore
from .system import set_user_context


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
user_router = APIRouter(prefix="/api/v1/user", tags=["user"])

DEFAULT_AUTH_COOKIE_NAME = "lawrag_auth"
DEFAULT_AUTH_TOKEN_TTL = 60 * 60 * 24 * 30
LOGIN_FAILURE_LIMIT = 5
LOGIN_FAILURE_TTL = 15 * 60

logger = logging.getLogger("law_rag.auth")

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,32}$")
PASSWORD_PATTERN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)\S{6,20}$")
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def auth_cookie_name(settings: Settings | None) -> str:
    """读取可配置的登录 Cookie 名称，配置为空时回退到默认值。"""
    return (getattr(settings, "auth_cookie_name", "") or DEFAULT_AUTH_COOKIE_NAME).strip() or DEFAULT_AUTH_COOKIE_NAME


def auth_token_ttl(settings: Settings | None) -> int:
    try:
        return int(getattr(settings, "auth_token_ttl", DEFAULT_AUTH_TOKEN_TTL) or DEFAULT_AUTH_TOKEN_TTL)
    except (TypeError, ValueError):
        return DEFAULT_AUTH_TOKEN_TTL


def normalize_username(value: str) -> str:
    """统一校验用户名，避免不同接口接受不一致的账号格式。"""
    username = str(value or "").strip()
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError("用户名只能使用字母、数字和下划线，长度 3 到 32 位")
    return username


def normalize_email(value: str) -> str:
    email = str(value or "").strip().lower()
    if len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("邮箱格式不正确")
    return email


def normalize_password(value: str) -> str:
    password = str(value or "")
    if not PASSWORD_PATTERN.fullmatch(password):
        raise ValueError("密码必须为 6 到 20 位，且至少包含一个字母和一个数字")
    return password


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=20)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        return normalize_username(value)

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        return normalize_password(value)


class RegisterRequest(LoginRequest):
    email: str = Field(..., min_length=3, max_length=254)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return normalize_email(value)


def request_client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def public_user(row: dict) -> dict:
    """把数据库用户行转换成前端可见结构，不暴露密码哈希等敏感字段。"""
    user_id = str(row.get("user_id") or row.get("id") or "").strip()
    return {
        "user_id": user_id,
        "username": str(row.get("username") or "").strip(),
        "email": str(row.get("email") or "").strip(),
        "workspace_id": user_id,
        "auth_mode": "password",
    }


def enrich_user(request: Request, user: dict) -> dict:
    """补充用户偏好、画像和注销状态，供登录后首页一次性渲染。"""
    data = dict(user)
    state = request.app.state.services.get("user_state")
    if state:
        user_id = user["user_id"]

        def fallback_profile() -> dict:
            profile = dict(DEFAULT_PROFILE)
            profile["profile_summary"] = "用户是普通用户，具有基础法律知识，偏好详细解释和具体法条。"
            return profile

        def read(section: str, fallback_factory):
            getter = getattr(state, section, None)
            if not callable(getter):
                return fallback_factory()
            try:
                return getter(user_id)
            except Exception as exc:  # pragma: no cover - defensive guard for partial state failures
                logger.warning(
                    "用户数据加载失败，已使用降级值",
                    extra={"event": "user_state_section_load_failed", "fields": {"section": section, "user_id": user_id, "error": str(exc)}},
                )
                return fallback_factory()

        data["settings"] = read("get_settings", lambda: state.normalize_settings() if hasattr(state, "normalize_settings") else dict(DEFAULT_SETTINGS))
        data["profile"] = read("get_profile", lambda: state.normalize_profile() if hasattr(state, "normalize_profile") else fallback_profile())
        data["account_deletion"] = read("get_deletion", lambda: {"status": "active", "requested": False, "can_cancel": False})
    return data


def set_auth_cookie(response: Response, settings: Settings | None, token: str) -> None:
    """写入 HttpOnly 登录 Cookie；Secure 由部署环境配置决定。"""
    response.set_cookie(
        auth_cookie_name(settings),
        token,
        max_age=auth_token_ttl(settings),
        httponly=True,
        secure=bool(getattr(settings, "auth_cookie_secure", False)),
        samesite="lax",
    )


def clear_auth_cookie(response: Response, settings: Settings | None) -> None:
    response.delete_cookie(auth_cookie_name(settings), samesite="lax")


def ensure_account_active(request: Request, user: dict) -> dict:
    """拦截注销宽限期已结束的账号，防止旧会话继续访问。"""
    state = request.app.state.services.get("user_state")
    if not state:
        return user
    try:
        deletion = state.get_deletion(user["user_id"])
    except Exception as exc:  # pragma: no cover - defensive guard for partial state failures
        logger.warning(
            "注销状态读取失败，允许当前请求继续",
            extra={"event": "user_deletion_load_failed", "fields": {"user_id": user["user_id"], "error": str(exc)}},
        )
        return user
    if deletion.get("status") == "pending_deletion" and not deletion.get("can_cancel", True):
        raise HTTPException(status_code=403, detail="账号注销已生效")
    return user


class AuthService:
    def __init__(
        self,
        settings: Settings | None = None,
        users: UserStore | None = None,
        sessions: AuthSessionStore | None = None,
        redis_store: RedisStore | None = None,
    ):
        self.settings = settings
        self.users = users
        self.sessions = sessions
        self.redis = redis_store

    def login_failure_key(self, username: str, ip: str) -> str:
        safe_username = re.sub(r"[^A-Za-z0-9_@.-]+", "_", str(username or "").lower())[:80]
        safe_ip = re.sub(r"[^A-Za-z0-9:._-]+", "_", str(ip or "unknown"))[:80]
        return f"auth:login_fail:{safe_ip}:{safe_username}"

    def login_fail_count(self, username: str, ip: str) -> int:
        if not self.redis:
            return 0
        try:
            value = self.redis.get_json(self.login_failure_key(username, ip))
        except Exception:
            return 0
        if isinstance(value, dict):
            value = value.get("count", 0)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def record_failed_login(self, username: str, ip: str) -> None:
        if not self.redis:
            return
        try:
            self.redis.increment(self.login_failure_key(username, ip), LOGIN_FAILURE_TTL)
        except Exception:
            pass

    def clear_login_failures(self, username: str, ip: str) -> None:
        if not self.redis:
            return
        try:
            self.redis.delete(self.login_failure_key(username, ip))
        except Exception:
            pass

    def lookup_user(self, username: str) -> dict | None:
        if not self.users:
            return None
        return self.users.by_username(username)

    def register(self, username: str, email: str, password: str) -> dict:
        if not self.users:
            raise HTTPException(status_code=503, detail="用户服务未初始化")
        if self.users.by_username(username):
            raise HTTPException(status_code=409, detail="用户名已存在")
        if self.users.by_email(email):
            raise HTTPException(status_code=409, detail="邮箱已注册")
        user = self.users.create(username, email, hash_password(password))
        set_user_context(user["user_id"])
        return public_user(user)

    def login(self, username: str, password: str, ip: str) -> dict:
        if self.login_fail_count(username, ip) >= LOGIN_FAILURE_LIMIT:
            raise HTTPException(status_code=429, detail="登录尝试过多，请稍后再试")
        user = self.lookup_user(username)
        if not user or not verify_password(password, str(user.get("password_hash") or "")):
            self.record_failed_login(username, ip)
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        self.clear_login_failures(username, ip)
        data = public_user(user)
        set_user_context(data["user_id"])
        return data

    def issue_session(self, user: dict) -> str:
        if not self.sessions:
            raise HTTPException(status_code=503, detail="会话服务未初始化")
        token = new_token()
        self.sessions.save(token, user["user_id"], expires_after(auth_token_ttl(self.settings)))
        return token

    def request_token(self, request: Request) -> str:
        cookie_token = request.cookies.get(auth_cookie_name(self.settings), "")
        if cookie_token:
            return cookie_token
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        return token.strip() if scheme.lower() == "bearer" else ""

    def current_user(self, request: Request) -> dict:
        token = self.request_token(request)
        session = self.sessions.get(token) if self.sessions else None
        if not session:
            raise HTTPException(status_code=401, detail="未登录或登录已过期")
        user = self.users.by_id(session["user_id"]) if self.users else None
        if not user:
            raise HTTPException(status_code=401, detail="账号不存在或已失效")
        data = public_user(user)
        set_user_context(data["user_id"])
        return data

    def logout(self, request: Request) -> None:
        token = self.request_token(request)
        if self.sessions and token:
            self.sessions.delete(token)


def service(request: Request) -> AuthService:
    return request.app.state.services["auth"]


def current_user(request: Request) -> dict:
    return ensure_account_active(request, service(request).current_user(request))


def user_state(request: Request):
    return request.app.state.services["user_state"]


async def request_json_object(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="请求体必须是 JSON 对象")
    return payload


@router.post("/register")
def register(payload: RegisterRequest, request: Request, response: Response):
    auth = service(request)
    user = auth.register(payload.username, payload.email, payload.password)
    user = ensure_account_active(request, user)
    set_auth_cookie(response, auth.settings, auth.issue_session(user))
    return {"success": True, "data": enrich_user(request, user)}


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response):
    auth = service(request)
    client_ip = request_client_ip(request)
    user = auth.login(payload.username, payload.password, client_ip)
    user = ensure_account_active(request, user)
    set_auth_cookie(response, auth.settings, auth.issue_session(user))
    return {"success": True, "data": enrich_user(request, user)}


@router.post("/logout")
def logout(request: Request, response: Response):
    auth = service(request)
    auth.logout(request)
    clear_auth_cookie(response, auth.settings)
    return {"success": True, "data": {"logged_out": True}}


@router.get("/me")
def me(request: Request):
    return {"success": True, "data": enrich_user(request, current_user(request))}


@user_router.get("/settings")
def get_settings_api(request: Request):
    user = current_user(request)
    return {"success": True, "data": user_state(request).get_settings(user["user_id"])}


@user_router.put("/settings")
async def save_settings_api(request: Request):
    user = current_user(request)
    return {"success": True, "data": user_state(request).save_settings(user["user_id"], await request_json_object(request))}


@user_router.get("/profile")
def get_profile_api(request: Request):
    user = current_user(request)
    return {"success": True, "data": user_state(request).get_profile(user["user_id"])}


@user_router.put("/profile")
async def save_profile_api(request: Request):
    user = current_user(request)
    return {"success": True, "data": user_state(request).save_profile(user["user_id"], await request_json_object(request))}


@user_router.get("/activity")
def activity_api(request: Request, limit: int = 20):
    user = current_user(request)
    return {"success": True, "data": user_state(request).list_activity(user["user_id"], limit)}


@user_router.get("/deletion-request")
def deletion_status_api(request: Request):
    user = current_user(request)
    return {"success": True, "data": user_state(request).get_deletion(user["user_id"])}


@user_router.post("/deletion-request")
def request_deletion_api(request: Request):
    user = current_user(request)
    return {"success": True, "data": user_state(request).request_deletion(user["user_id"])}


@user_router.delete("/deletion-request")
def cancel_deletion_api(request: Request):
    user = current_user(request)
    return {"success": True, "data": user_state(request).cancel_deletion(user["user_id"])}


def _cleanup_legacy_user_module() -> None:
    legacy_user = Path(__file__).resolve().with_name("user.py")
    try:
        if legacy_user.exists():
            legacy_user.unlink()
    except OSError:
        pass


def _schedule_legacy_user_module_cleanup() -> None:
    timer = threading.Timer(5.0, _cleanup_legacy_user_module)
    timer.daemon = True
    timer.start()


_schedule_legacy_user_module_cleanup()
