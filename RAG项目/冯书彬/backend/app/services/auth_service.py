import logging
import secrets
from datetime import datetime, timedelta
from uuid import uuid4

from backend.app.core.config import settings
from backend.app.core.crypto import encrypt_text, hmac_digest
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_otp_code,
    hash_refresh_token,
    utc_now,
)
from backend.app.models.user import DeviceSession, User
from backend.app.schemas.auth import OtpRequestResult, TokenPair
from backend.app.services.auth_store import AuthStore, InMemoryAuthStore, OtpState

logger = logging.getLogger(__name__)

_MAX_DEVICE_SESSIONS = 5
_ALLOWED_TEST_ENVIRONMENTS = {"development", "test", "testing", "internal"}


class AuthServiceError(ValueError):
    # 服务层统一抛业务错误，由 API 层映射为 4xx，避免冒泡成 500。
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


_auth_store: AuthStore = InMemoryAuthStore()


def set_auth_store(store: AuthStore) -> None:
    # 测试或后续数据库 repository 可通过此入口替换存储实现。
    global _auth_store
    _auth_store = store


def reset_auth_store() -> None:
    set_auth_store(InMemoryAuthStore())


def get_auth_store() -> AuthStore:
    return _auth_store


def _mask_phone(phone: str) -> str:
    # 日志只展示首尾少量字符，避免泄露完整手机号。
    if len(phone) < 7:
        return "***"
    return f"{phone[:3]}****{phone[-4:]}"


def _safe_client_id(client_id: str) -> str:
    # 日志不输出原始 client_id，仅记录稳定短摘要用于排查同设备问题。
    return hmac_digest(client_id, purpose="client-id-log")[:12]


def _otp_key(phone: str, client_id: str) -> str:
    # OTP 存储 key 使用 HMAC，避免内存快照直接暴露手机号和设备标识。
    return hmac_digest(f"{phone}:{client_id}", purpose="otp-session")


def _ensure_test_phone_config_allowed() -> None:
    # 生产环境必须禁用固定测试验证码，避免上线绕过短信通道。
    test_numbers = getattr(settings, "TEST_PHONE_NUMBERS", {})
    environment = getattr(settings, "ENVIRONMENT", "development")
    if test_numbers and environment not in _ALLOWED_TEST_ENVIRONMENTS:
        logger.error("fixed otp rejected outside internal env", extra={"environment": environment})
        raise AuthServiceError("fixed test phone numbers are disabled in production", status_code=403)


def _generate_otp(phone: str) -> tuple[str, bool]:
    # 内部测试号码使用固定验证码且不发送短信。
    test_numbers = getattr(settings, "TEST_PHONE_NUMBERS", {})
    if phone in test_numbers:
        return test_numbers[phone], False
    # 普通号码生成 6 位随机数字，当前 MVP 仅模拟发送。
    return f"{secrets.randbelow(1_000_000):06d}", True


def _locked_until(key: str) -> datetime | None:
    state = _auth_store.get_otp(key)
    if state and state.locked_until and state.locked_until > utc_now():
        return state.locked_until
    return None


def request_login_code(phone: str, client_id: str) -> OtpRequestResult:
    # 请求入口先校验固定验证码环境限制。
    _ensure_test_phone_config_allowed()
    key = _otp_key(phone, client_id)
    if _locked_until(key):
        logger.warning(
            "otp request blocked by lock",
            extra={"phone_masked": _mask_phone(phone), "client_id_hash": _safe_client_id(client_id)},
        )
        raise AuthServiceError("otp locked", status_code=429)
    code, should_send = _generate_otp(phone)
    expires_at = utc_now() + timedelta(seconds=settings.OTP_TTL_SECONDS)
    _auth_store.save_otp(key, OtpState(code_hash=hash_otp_code(phone, client_id, code), expires_at=expires_at))
    logger.info(
        "otp code prepared",
        extra={"phone_masked": _mask_phone(phone), "client_id_hash": _safe_client_id(client_id), "sent": should_send},
    )
    return OtpRequestResult(sent=should_send, expires_in_seconds=settings.OTP_TTL_SECONDS)


def _get_valid_otp_state(phone: str, client_id: str) -> OtpState:
    # 按脱敏 key 查找验证码状态。
    key = _otp_key(phone, client_id)
    state = _auth_store.get_otp(key)
    if state is None:
        logger.warning("otp missing", extra={"phone_masked": _mask_phone(phone), "client_id_hash": _safe_client_id(client_id)})
        raise AuthServiceError("otp code not requested", status_code=400)
    # 锁定期内拒绝任何验证码尝试。
    if state.locked_until and state.locked_until > utc_now():
        logger.warning("otp locked", extra={"phone_masked": _mask_phone(phone), "client_id_hash": _safe_client_id(client_id)})
        raise AuthServiceError("otp locked", status_code=429)
    # 过期验证码立即删除，减少敏感派生数据停留时间。
    if state.expires_at <= utc_now():
        _auth_store.delete_otp(key)
        logger.warning("otp expired", extra={"phone_masked": _mask_phone(phone), "client_id_hash": _safe_client_id(client_id)})
        raise AuthServiceError("otp expired", status_code=400)
    return state


def _verify_otp(phone: str, code: str, client_id: str) -> None:
    # 校验验证码摘要，避免比较明文存储值。
    key = _otp_key(phone, client_id)
    state = _get_valid_otp_state(phone, client_id)
    expected_hash = hash_otp_code(phone, client_id, code)
    if not secrets.compare_digest(state.code_hash, expected_hash):
        state.attempts += 1
        if state.attempts >= settings.OTP_MAX_ATTEMPTS:
            state.locked_until = utc_now() + timedelta(seconds=settings.OTP_LOCK_SECONDS)
        _auth_store.save_otp(key, state)
        logger.warning("otp invalid", extra={"phone_masked": _mask_phone(phone), "attempts": state.attempts})
        raise AuthServiceError("invalid otp code", status_code=400)
    # 验证成功后删除 OTP，确保验证码只能使用一次。
    _auth_store.delete_otp(key)
    logger.info("otp verified", extra={"phone_masked": _mask_phone(phone), "client_id_hash": _safe_client_id(client_id)})


def _get_or_create_user(phone: str) -> User:
    # 用手机号 HMAC 查找用户，数据库中不保存明文手机号。
    phone_hash = hmac_digest(phone, purpose="phone")
    existing = _auth_store.get_user_by_phone_hash(phone_hash)
    if existing:
        return existing
    encrypted = encrypt_text(phone, purpose="phone")
    now = utc_now()
    user = User(
        id=str(uuid4()),
        encrypted_phone=encrypted.ciphertext,
        phone_nonce=encrypted.nonce,
        phone_encrypted_data_key=encrypted.encrypted_data_key,
        phone_key_version=encrypted.key_version,
        phone_hmac=phone_hash,
        agreement_version="v1",
        privacy_policy_version="v1",
        adult_confirmed=True,
        created_at=now,
        updated_at=now,
    )
    _auth_store.save_user(user)
    logger.info("user created for phone login", extra={"phone_masked": _mask_phone(phone), "user_id": user.id})
    return user


def _revoke_oldest_session_if_needed(user_id: str) -> None:
    # 只统计未吊销的活跃设备会话。
    sessions = [_auth_store.get_session_by_hash(item) for item in _auth_store.get_user_session_hashes(user_id)]
    active = [session for session in sessions if session and not session.revoked]
    if len(active) <= _MAX_DEVICE_SESSIONS:
        return
    # 超过 5 个设备时吊销最早登录的会话。
    oldest = min(active, key=lambda item: item.login_at)
    oldest.revoked = True
    oldest.revoked_at = utc_now()
    _auth_store.save_session(oldest)
    logger.info("oldest device session revoked", extra={"user_id": user_id})


def _create_device_session(user: User, client_id: str, user_agent: str) -> tuple[str, DeviceSession]:
    # 生成 refresh token 明文并立即转换成服务端摘要保存。
    refresh_token, expires_at = create_refresh_token()
    now = utc_now()
    session = DeviceSession(
        id=str(uuid4()),
        user_id=user.id,
        refresh_token_hash=hash_refresh_token(refresh_token),
        device_identifier=client_id,
        user_agent=user_agent,
        login_at=now,
        last_activity_at=now,
        province_location=None,
        expires_at=expires_at,
    )
    _auth_store.save_session(session)
    _revoke_oldest_session_if_needed(user.id)
    logger.info("device session created", extra={"user_id": user.id, "client_id_hash": _safe_client_id(client_id)})
    return refresh_token, session


def login_with_code(phone: str, code: str, client_id: str, user_agent: str) -> TokenPair:
    # 登录先完成验证码校验，再创建用户和设备会话。
    _ensure_test_phone_config_allowed()
    _verify_otp(phone, code, client_id)
    user = _get_or_create_user(phone)
    refresh_token, session = _create_device_session(user, client_id, user_agent)
    access_token = create_access_token(user.id, session.id)
    logger.info("phone login succeeded", extra={"phone_masked": _mask_phone(phone), "user_id": user.id})
    return TokenPair(access_token=access_token, refresh_token=refresh_token, user_id=user.id, token_type="bearer")


def _get_active_session(refresh_token: str) -> DeviceSession:
    # 使用 refresh token 摘要定位会话，不查询或记录 token 明文。
    session = _auth_store.get_session_by_hash(hash_refresh_token(refresh_token))
    if session is None or session.revoked:
        logger.warning("refresh token rejected")
        raise AuthServiceError("invalid refresh token", status_code=401)
    if session.expires_at <= utc_now():
        session.revoked = True
        session.revoked_at = utc_now()
        _auth_store.save_session(session)
        logger.warning("refresh token expired", extra={"user_id": session.user_id})
        raise AuthServiceError("refresh token expired", status_code=401)
    return session


def refresh_access_token(refresh_token: str) -> TokenPair:
    # 刷新时轮换 refresh token，降低旧 token 泄露风险。
    session = _get_active_session(refresh_token)
    old_hash = session.refresh_token_hash
    new_refresh_token, expires_at = create_refresh_token()
    new_hash = hash_refresh_token(new_refresh_token)
    session.refresh_token_hash = new_hash
    session.expires_at = expires_at
    session.last_activity_at = utc_now()
    _auth_store.delete_session_hash(old_hash)
    _auth_store.replace_user_session_hash(session.user_id, old_hash, new_hash)
    _auth_store.save_session(session)
    access_token = create_access_token(session.user_id, session.id)
    logger.info("access token refreshed", extra={"user_id": session.user_id, "session_id": session.id})
    return TokenPair(access_token=access_token, refresh_token=new_refresh_token, user_id=session.user_id, token_type="bearer")


def revoke_refresh_token(refresh_token: str) -> None:
    # 登出只吊销当前 refresh token 对应的设备会话。
    session = _get_active_session(refresh_token)
    session.revoked = True
    session.revoked_at = utc_now()
    _auth_store.save_session(session)
    logger.info("refresh token revoked", extra={"user_id": session.user_id, "session_id": session.id})
