# -*- coding: utf-8 -*-
"""账号体系：注册 / 登录 / 令牌 / 登出（纯标准库，不引入任何新依赖）。

设计决定（写给下游与审查）
==========================

**1. `user_id` 从哪来**
    注册时用户名被**规范化**（``trim`` + ``NFKC`` + 统一小写）后**直接当作 `user_id`**
    （`AC-AU-28`/`D5`）：`Alice` / ` alice ` / `ALICE` 都是同一个账号 `alice`。
    这样做的好处是既有接口（``/chat``、``/sessions``、``/documents``）**一行都不用改** ——
    它们以 `user_id` 作隔离键，而登录态提供的正好就是同一个字符串。
    额外好处：`users.user_id` 的既有语义（任意业务 ID，`/sessions` 会 `upsert_user("u1")`）
    完全不受影响，因此既有 345 项测试零改动全绿。

**2. token 与 user_id 如何绑定**
    令牌是服务端用 ``secrets.token_urlsafe(32)`` 生成的**高熵随机串**，绝不放在
    客户端可读位置（`localStorage` 被明令禁止，`AC-AU-30`）。服务端只存
    ``sha256(token)``（表 `auth_tokens`），因此**拿到数据库也拿不到可用的 bearer 令牌**。
    绑定关系 = `auth_tokens.user_id` 这一行：校验时必须同时满足
    「哈希命中」+「未 ``revoked_at``」+「``expires_at`` 未过期」，
    返回的 `user_id` **只来自该行**，永远不来自请求体。

**3. `REQUIRE_AUTH`/`AUTH_REQUIRED` 打开时如何防止「带着 A 的 token 用 B 的 user_id」**
    校验器（:meth:`AuthService.authorize`）在强制模式下先取登录态 `user_id`，
    再与请求里的 `user_id`（body 或 query）逐字符比对：

    * 不一致 → **403** ``user_mismatch``（直接拒绝，不猜用户想干什么）；
    * 一致 → 放行；
    * 缺令牌/伪造/已登出/过期 → **401** ``unauthenticated``。

    「用登录态覆盖请求体」和「403」两种做法规范都算通过（`AC-AU-29`），这里选 403：
    **静默覆盖会让调用方以为写进了 B 的会话**，而它其实写进了 A 的，排查时是灾难。
    `/chat` 另有一道独立防线（会话归属校验，用的是会话表里的真实 `user_id`），
    即便鉴权被关闭，也写不进别人的会话。

**4. 密码存储**（`AC-AU-21`~`24`）
    只存**每用户随机盐的单向哈希**：``hashlib.scrypt``（``n=2**15, r=8, p=1``，16 字节
    ``secrets.token_bytes`` 盐），存成 PHC 风格串
    ``scrypt$n=32768,r=8,p=1$<b64盐>$<b64哈希>``；校验用 ``hmac.compare_digest``。
    **明文 / 可逆加密 / base64 当加密 / 裸 SHA256 一律禁止**（裸 SHA256 之所以也不行：
    无盐、可穷举、GPU 秒破）。

**5. 「用户名不存在」与「密码错误」不可区分**（`AC-AU-25`~`27`）
    两条分支返回**同一状态码 + 同一 JSON + 同一文案**，且不存在用户时也跑一次
    scrypt 校验（dummy verify）把耗时差抹平，防止用响应时间枚举用户名。
    注意：**注册查重必然泄露存在性**（"这个用户名能不能用"本身就是答案），
    两者不可兼得 —— 处置是给查重接口限速（`AC-AU-36`），登录保持不可区分。详见 docs/API.md §9。

**6. 令牌载体**：``Set-Cookie: lr_session=<token>; HttpOnly; SameSite=Lax; Path=/``
    （本机 http 调试**不加** ``Secure``，加了本地登录直接失效；生产 HTTPS 必须开
    ``AUTH_COOKIE_SECURE=true``）。**响应体里绝不回显令牌**（`AC-AU-31`）。
    为便于脚本/curl 与单测，``GET /auth/me`` 也接受 ``Authorization: Bearer <token>``，
    但浏览器侧走的是 HttpOnly Cookie。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from ..business.store import BusinessStore, is_unique_violation

logger = logging.getLogger(__name__)

#: 用户名分隔用的 NFKC 之外的字符集：小写字母 / 数字 / 下划线 / 连字符
USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 32
USERNAME_PATTERN_TEXT = "[a-z0-9_-]"

#: 密码长度约束（规范 §3.4.1：至少 10 位；同时给一个上限防"超长密码 = CPU 打满"）
PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 256

#: scrypt 参数（与 docs/WEB-UX.md §3.4.2 逐字一致）
SCRYPT_ALGORITHM = "scrypt"
SCRYPT_N = 2 ** 15          # 32768
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16

#: 存储串前缀（PHC 风格，**自描述参数**：将来调参不影响老用户校验）
def phc_prefix(n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P) -> str:
    return f"{SCRYPT_ALGORITHM}$n={n},r={r},p={p}$"


PHC_PREFIX = phc_prefix()

#: 令牌字节数（``token_urlsafe(32)`` ≈ 256 bit 熵）
TOKEN_BYTES = 32

#: 「用户名不存在」分支用的假凭据盐（固定全零）。假凭据的**哈希串**在
#: :meth:`AuthService._dummy_verify` 里按实例参数现算 —— 刻意**不在模块加载时**算：
#: 生产参数一次 scrypt 约 30–80ms，放在 import 上会让每个进程启动都白付这笔钱。
_DUMMY_SALT = b"\x00" * SALT_BYTES

#: 面向用户的统一失败文案（两条分支逐字相同，`AC-AU-25/26`）
INVALID_CREDENTIALS_MESSAGE = "用户名或密码错误"
MESSAGE_USERNAME_TAKEN = "用户名 {username} 已被占用，换一个试试"

#: 找回密码失败的**唯一**文案（`AC-AU-53` 基准文案：不区分"用户名不存在/恢复码错/不匹配"）
INVALID_RECOVERY_MESSAGE = "用户名、恢复码或新密码不正确"
#: 三个凭据没填全时的提示（`AC-RC-3`：400 且无副作用；这是"请求本身的问题"，与存在性无关）
MESSAGE_RESET_FIELDS_REQUIRED = "用户名、恢复码与新密码都必填"

#: 登录失败（含"用户不存在"）的**唯一**错误码
ERROR_INVALID_CREDENTIALS = "invalid_credentials"
ERROR_INVALID_USERNAME = "invalid_username"
ERROR_WEAK_PASSWORD = "weak_password"
ERROR_USERNAME_TAKEN = "username_taken"
ERROR_UNAUTHENTICATED = "unauthenticated"
ERROR_USER_MISMATCH = "user_mismatch"
ERROR_RATE_LIMITED = "rate_limited"
#: 找回密码失败（用户名错 / 恢复码错 / 不匹配三种情形共用）
ERROR_INVALID_RECOVERY = "invalid_recovery"
#: 恢复码强度不足（配置非法时的兜底错误，正常不会出现）
ERROR_RESET_FIELDS = "reset_fields_required"


# ---------------- 用户名规范化与校验 ----------------

def normalize_username(raw: Any) -> str:
    """规范化用户名：``trim`` + ``NFKC`` + 统一小写。

    为什么三步都要（`AC-AU-9`）：

    * ``trim``：用户从别处复制粘贴常带首尾空格，`" alice "` 与 `"alice"` 必须是同一账号；
    * ``NFKC``：全角 ``Ａlice`` 与半角 ``Alice`` 在界面上看起来不一样、语义上却是同一个
      人想注册的名字（也防"用全角字符冒充已占用用户名"绕过唯一约束）；
    * 小写：``Alice`` / ``ALICE`` / ``alice`` 是同一个账号（**落库统一小写**，写进文档）。

    注意：本函数**只用于用户名**。密码**绝不**做任何规范化（`AC-AU-19`：
    含首尾空格的密码必须原样校验通过）。
    """
    return unicodedata.normalize("NFKC", str(raw or "").strip()).lower()


def username_problem(name: str) -> str:
    """用户名格式问题；空字符串表示合法。

    规则（§3.3.1）：3–32 字符、只允许 ``[a-z0-9_-]``、**不允许纯数字**。
    「不许纯数字」是为了避免用户名与数字 ID 撞车（"12345" 这种名字既没人味也容易与
    会话/文档 ID 混淆）。
    """
    if not name:
        return f"用户名不能为空（规范化后为空），请使用 {USERNAME_MIN_LENGTH}-{USERNAME_MAX_LENGTH} 位小写字母、数字、下划线或连字符"
    length = len(name)
    if length < USERNAME_MIN_LENGTH:
        return f"用户名至少 {USERNAME_MIN_LENGTH} 位，当前 {length} 位"
    if length > USERNAME_MAX_LENGTH:
        return f"用户名最多 {USERNAME_MAX_LENGTH} 位，当前 {length} 位"
    if not name.isascii() or not all(
            (ch.isalnum() and ch.isascii()) or ch in "_-" for ch in name):
        return (f"用户名只能包含小写字母、数字、下划线和连字符"
                f"（不允许空格/中文/标点），实际收到：{name!r}")
    if name.isdigit():
        return "用户名不能是纯数字"
    return ""


def password_problem(password: Any) -> str:
    """密码长度问题；空字符串表示合法。

    ``password is None`` 与空串都算「没填」；**不做 ``strip``** ——
    含首尾空格的密码是有意的（`AC-AU-19`），悄悄删掉它会让用户下次登录失败。
    """
    if password is None:
        return f"密码不能为空，至少 {PASSWORD_MIN_LENGTH} 位"
    text = str(password)
    if not text:
        return f"密码不能为空，至少 {PASSWORD_MIN_LENGTH} 位"
    if len(text) < PASSWORD_MIN_LENGTH:
        return f"密码至少 {PASSWORD_MIN_LENGTH} 位，当前 {len(text)} 位"
    if len(text) > PASSWORD_MAX_LENGTH:
        return f"密码最多 {PASSWORD_MAX_LENGTH} 位，当前 {len(text)} 位"
    return ""


def is_valid_username(name: str) -> bool:
    return not username_problem(name)


# ---------------- 密码哈希（stdlib scrypt + 每用户随机盐）----------------

def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _scrypt(password: str, salt: bytes, *, n: int, r: int, p: int,
            dklen: int = SCRYPT_DKLEN) -> bytes:
    """调 ``hashlib.scrypt``，并把 ``maxmem`` 显式写清。

    为什么显式给 ``maxmem``：scrypt 的内存需求约 ``128 * n * r`` 字节（默认参数下是
    32 MiB），而 OpenSSL 的默认上限正是 32 MiB —— 卡在边界上时不同平台/版本会报
    ``memory limit exceeded``。这里给到需求量的两倍再加余量，行为才是确定的，
    而不是"我这台机器能跑、CI 上不行"。
    """
    password_bytes = password.encode("utf-8")
    maxmem = 128 * n * r * 2 + (1 << 20)
    try:
        return hashlib.scrypt(password_bytes, salt=salt, n=n, r=r, p=p,
                              dklen=dklen, maxmem=maxmem)
    except (ValueError, MemoryError):
        # 某些环境对 maxmem 参数敏感：去掉它（回到库默认）再试一次，仍然失败就抛出，
        # 不静默降级到弱参数。
        logger.warning("scrypt 显式 maxmem 被拒（n=%d r=%d），改用库默认 maxmem 重试",
                       n, r)
        return hashlib.scrypt(password_bytes, salt=salt, n=n, r=r, p=p, dklen=dklen)


def hash_password(password: str, *, n: int = SCRYPT_N, r: int = SCRYPT_R,
                  p: int = SCRYPT_P, salt: bytes | None = None) -> str:
    """把明文密码变成**不可逆**的 PHC 风格存储串。

    返回形如 ``scrypt$n=32768,r=8,p=1$<b64 16 字节盐>$<b64 32 字节哈希>``。

    ⚠️ 明文用完即弃：本函数**不打印、不缓存、不返回**明文；调用方也不得把它写进日志。
    """
    salt = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    digest = _scrypt(password, salt, n=n, r=r, p=p)
    return f"{SCRYPT_ALGORITHM}$n={n},r={r},p={p}${_b64encode(salt)}${_b64encode(digest)}"


def parse_phc(stored: str) -> dict | None:
    """解析 PHC 风格存储串 → ``{"n","r","p","salt","hash"}``；无法解析返回 ``None``。"""
    try:
        algorithm, params, salt_b64, hash_b64 = str(stored).split("$")
    except (ValueError, AttributeError):
        return None
    if algorithm != SCRYPT_ALGORITHM:
        return None
    parsed: dict[str, Any] = {}
    for item in params.split(","):
        if "=" not in item:
            return None
        key, _, value = item.partition("=")
        try:
            parsed[key.strip()] = int(value)
        except ValueError:
            return None
    if not {"n", "r", "p"} <= set(parsed):
        return None
    try:
        parsed["salt"] = _b64decode(salt_b64)
        parsed["hash"] = _b64decode(hash_b64)
    except Exception:  # noqa: BLE001 - 任何解码失败都等价于"这个凭据串不可用"
        return None
    return parsed


def verify_password(password: str, stored: str) -> bool:
    """校验密码；恒时间比较（``hmac.compare_digest``）。

    * 参数**从存储串里读**（自描述），因此将来调整 scrypt 参数不影响老用户登录；
    * 存储串不可解析、密码为空、算法不是 scrypt → 一律 ``False``，绝不抛异常：
      登录路径上任何异常都可能变成"用户名枚举的另一条旁路"。
    """
    if not stored or password is None:
        return False
    parsed = parse_phc(stored)
    if parsed is None:
        return False
    try:
        candidate = _scrypt(str(password), parsed["salt"], n=parsed["n"],
                            r=parsed["r"], p=parsed["p"], dklen=len(parsed["hash"]))
    except (ValueError, MemoryError):
        logger.warning("scrypt 校验失败（参数非法或内存不足），按凭据错误处理")
        return False
    return hmac.compare_digest(candidate, parsed["hash"])


def sha256_hex(text: str) -> str:
    """令牌哈希（库里只存这个，不存令牌明文）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------- 限速 ----------------

class SlidingWindowLimiter:
    """进程内滑动窗口限速（**只用标准库**，不引入 Redis 依赖）。

    语义：同一 key 在 ``window`` 秒内最多 ``max_requests`` 次。

    两点如实说明：

    * 多 worker / 多进程部署时**各自计数**，实际阈值会按 worker 数放大 ——
      本接口限速的目的是"别让一个脚本把 CPU 烧掉"，不是精确配额，够用；
    * 进程重启即清零（无持久化）—— 同样够用，且不会因为一个限速器把服务变成
      有状态组件。
    """

    def __init__(self, window_seconds: float, max_requests: int) -> None:
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_requests = max(1, int(max_requests))
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            horizon = moment - self.window_seconds
            while bucket and bucket[0] <= horizon:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(moment)
            return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        moment = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._hits.get(key) or deque()
            if not bucket:
                return 0
            return max(1, int(bucket[0] + self.window_seconds - moment) + 1)

    def reset(self) -> None:
        """清空所有计数（测试用）。"""
        with self._lock:
            self._hits.clear()


# ---------------- 结果对象 ----------------

@dataclass
class FieldError:
    """一个可判定的字段级错误（对应规范里的 ``error`` / ``message`` 两个键）。"""

    error: str
    message: str
    field: str

    def to_dict(self) -> dict:
        return {"error": self.error, "message": self.message, "field": self.field}


@dataclass
class AuthOutcome:
    """一次注册/登录/重置的结果：``user`` 与 ``error`` 至多一个非空。"""

    user: dict | None = None
    error: FieldError | None = None
    token: str | None = None
    expires_at: float | None = None
    #: **一次性恢复码明文**：只在"注册成功"与"重置成功（轮换后的新码）"这两次响应里
    #: 出现，此后任何接口都不得再返回它（`AC-AU-52` / `AC-RC-6`）。
    recovery_code: str | None = None

    @property
    def ok(self) -> bool:
        return self.user is not None


# ---------------- 服务 ----------------

class AuthService:
    """注册/登录/令牌校验的门面。

    ⚠️ 所有方法都是**同步阻塞**的（SQLite/MySQL 读写 + scrypt 计算）；端点里必须经
    ``run_blocking`` 卸载到线程池，否则事件循环会被 scrypt 拖住（本项目的并发约定）。
    """

    def __init__(self, store: BusinessStore, config: Any, *,
                 hash_params: dict | None = None) -> None:
        self.store = store
        self.config = config
        self.auth_config = getattr(config, "auth", None)
        params = dict(hash_params or {})
        self.hash_params = {
            "n": int(params.get("n", SCRYPT_N)),
            "r": int(params.get("r", SCRYPT_R)),
            "p": int(params.get("p", SCRYPT_P)),
        }
        #: 假凭据盐（固定全零，只为让"不存在的用户"分支也跑一次等价 scrypt 占时间）。
        #: 哈希串**每次现算**（见 :meth:`_dummy_verify`）：缓存一件哈希只省内存分配，
        #: 却会引入"缓存参数与实例参数不一致"的整类 bug（已实测踩过一次：
        #: 测试把 n 调小后，缓存的假凭据仍按生产参数跑，两条分支耗时差 116ms）。
        window = float(getattr(self.auth_config, "rate_limit_window_seconds", 60) or 60)
        limit = int(getattr(self.auth_config, "rate_limit_max_requests", 30) or 30)
        self.limiter = SlidingWindowLimiter(window, limit)
        #: 一次性恢复码的熵（字节）与哈希参数（见 config.AuthConfig）
        self.recovery_code_bytes = int(
            getattr(self.auth_config, "recovery_code_bytes", 24) or 24)
        self.recovery_hash_n = int(
            getattr(self.auth_config, "recovery_hash_n", self.hash_params["n"])
            or self.hash_params["n"])
        #: 假凭据哈希（F-A）：**构造时按本实例参数算好一次**，让"用户不存在"分支与
        #: "密码错误"分支都**恰好 1 次 scrypt**。原实现每次现算（= 2 次 scrypt），
        #: 与密码错误分支差 126–136ms，正好踩碎 `AC-AU-27`（<50ms）。
        #: 代价：每个进程多付一次 scrypt（生产 ≈30–80ms，只在构造时付一次）。
        self._dummy_hash = self.hash_password("dummy-password-for-timing")

    # ---------- 开关/令牌参数 ----------

    @property
    def require_auth(self) -> bool:
        return bool(getattr(self.auth_config, "require_auth", False))

    @property
    def cookie_name(self) -> str:
        return str(getattr(self.auth_config, "cookie_name", "lr_session") or "lr_session")

    @property
    def cookie_secure(self) -> bool:
        return bool(getattr(self.auth_config, "cookie_secure", False))

    @property
    def token_ttl_seconds(self) -> int:
        return int(getattr(self.auth_config, "token_ttl_seconds", 7 * 24 * 3600)
                   or 7 * 24 * 3600)

    def cookie_attributes(self) -> dict:
        """``Set-Cookie`` 的属性（**HttpOnly + SameSite=Lax + Path=/**）。

        ``Secure`` 由配置决定：本机 http 调试默认关闭（开了本地登录直接失效），
        生产 HTTPS 必须 ``AUTH_COOKIE_SECURE=true``。
        """
        return {
            "httponly": True,
            "samesite": "lax",
            "path": "/",
            "secure": self.cookie_secure,
        }

    def ensure(self) -> None:
        """建账号表（幂等；只建一次）。"""
        self.store.ensure_auth_schema()

    # ---------- 密码 ----------

    def hash_password(self, password: str) -> str:
        return hash_password(password, **self.hash_params)

    def verify_password(self, password: str, stored: str) -> bool:
        return verify_password(password, stored)

    def _dummy_verify(self, password: str) -> None:
        """对**不存在的用户**跑一次等价校验：**恰好一次 scrypt**（`AC-AU-27`）。

        修法（F-A）：原实现是 ``verify_password(password, self.hash_password(...))`` ——
        每次都要**现算**假凭据哈希，于是"用户不存在"= 2 次 scrypt，而"密码错误"只跑
        1 次（校验库里那行哈希），实测 p50 差 **126–136ms**。
        现在假凭据哈希在**构造时**按本实例参数算好（``self._dummy_hash``），
        两条分支都只有 1 次 scrypt，差值来自噪声而非分支。
        """
        verify_password(password, self._dummy_hash)

    # ---------- 注册 ----------

    def register(self, username: Any, password: Any) -> AuthOutcome:
        normalized = normalize_username(username)
        problem = username_problem(normalized)
        if problem:
            return AuthOutcome(error=FieldError(ERROR_INVALID_USERNAME, problem, "username"))
        problem = password_problem(password)
        if problem:
            return AuthOutcome(error=FieldError(ERROR_WEAK_PASSWORD, problem, "password"))

        self.ensure()
        password_hash = self.hash_password(str(password))
        try:
            # 唯一性**由数据库 UNIQUE 约束兜底**（不是"先查再插"，并发下那种写法会漏）
            self.store.create_auth_user(normalized, normalized, password_hash)
        except Exception as exc:  # noqa: BLE001 - 只吞"唯一冲突"，其余照抛
            if is_unique_violation(exc):
                logger.info("注册被唯一约束拒绝：username=%s", normalized)
                return AuthOutcome(error=FieldError(
                    ERROR_USERNAME_TAKEN,
                    MESSAGE_USERNAME_TAKEN.format(username=normalized),
                    "username"))
            raise
        logger.info("注册成功：username=%s", normalized)
        # 注册即登录：同一次响应里发令牌，省掉"注册完还要再登一次"的多余一步
        token, expires_at = self.issue_token(normalized)
        # 一次性恢复码（`AC-AU-49/52`）：明文**只在这一个响应里**出现，库里只存盐+哈希
        recovery_code, _salt = self.issue_recovery_code(normalized)
        return AuthOutcome(user={"user_id": normalized, "username": normalized},
                           token=token, expires_at=expires_at,
                           recovery_code=recovery_code)

    # ---------- 一次性恢复码（找回密码）----------

    def issue_recovery_code(self, user_id: str) -> tuple[str, str]:
        """生成并**落库**一次性恢复码；返回 ``(明文码, 盐)``。

        * 强度：``secrets.token_urlsafe(recovery_code_bytes)``（默认 24 字节 ≈192 bit，
          远高于规范要求的 ≥128 bit）；
        * 存储：只存「盐 + PHC 风格单向哈希」——**明文不落库、不进日志**；
        * 轮换 = 覆盖同一行，因此**旧码立刻作废**（校验只认这一行）。
        """
        self.ensure()
        code = secrets.token_urlsafe(self.recovery_code_bytes)
        salt = secrets.token_bytes(SALT_BYTES)
        phc = hash_password(code, n=self.recovery_hash_n, r=SCRYPT_R, p=SCRYPT_P, salt=salt)
        self.store.set_recovery_code(user_id, _b64encode(salt), phc)
        logger.info("已为 user_id=%s 生成/轮换一次性恢复码（库中只存盐+哈希）", user_id)
        return code, _b64encode(salt)

    @staticmethod
    def _verify_recovery_code(code: str, stored: str) -> bool:
        """校验恢复码（PHC 自描述参数 + 恒时间比较；解析失败一律 ``False``）。"""
        return verify_password(code, stored)

    def reset_password(self, username: Any, recovery_code: Any,
                       new_password: Any) -> AuthOutcome:
        """用一次性恢复码重置密码（`AC-AU-53..56` / `AC-RC-1..6`）。

        规则（逐条对应条文）：

        * **三者都必填**：缺任意一个 → 400（``AC-RC-3``），且**无副作用**；
        * 用户名格式非法 / 新密码不合规 → 400（请求本身的问题，不涉及存在性）；
        * 「用户名不存在 / 恢复码错 / 用户名与恢复码不匹配」→ **同一状态码 + 同一文案**
          （``AC-AU-53``：``用户名、恢复码或新密码不正确``），不泄露存在性；
          且不存在用户时也跑一次等价 scrypt 抹平耗时；
        * **失败不消耗恢复码**（``AC-AU-54``：校验是只读的，只有成功才覆盖）；
        * 成功 → ① 改密码哈希；② **旧令牌全部失效**（``AC-AU-55①``）；
          ③ **轮换恢复码**并把新码随本次响应回一次（``AC-AU-55④`` / ``AC-RC-6``）；
          ④ **不发令牌**（页面回 ``/login`` 让用户重新登录，`AC-RC-5`）。
        """
        normalized = normalize_username(username)
        code = "" if recovery_code is None else str(recovery_code)
        password = "" if new_password is None else str(new_password)

        if not normalized or not code or not password:
            return AuthOutcome(error=FieldError(ERROR_RESET_FIELDS,
                                                MESSAGE_RESET_FIELDS_REQUIRED, ""))
        problem = username_problem(normalized)
        if problem:
            return AuthOutcome(error=FieldError(ERROR_INVALID_USERNAME, problem, "username"))
        problem = password_problem(password)
        if problem:
            return AuthOutcome(error=FieldError(ERROR_WEAK_PASSWORD, problem, "new_password"))

        self.ensure()
        row = self.store.get_auth_user_by_name(normalized)
        record = self.store.get_recovery_code(str(row["user_id"])) if row else None
        if record is None or not self._verify_recovery_code(code, str(record.get("hash") or "")):
            # 时序抹平：不存在用户也要付一次等价 scrypt（与 F-A 同一思路）
            self._dummy_recovery_verify(code)
            logger.info("重置失败：username=%s（统一文案，不区分原因；恢复码未被消耗）",
                        normalized)
            return AuthOutcome(error=self._invalid_recovery())

        user_id = str(row["user_id"])
        self.store.update_password_hash(user_id, self.hash_password(password))
        revoked = self.store.revoke_all_tokens_for_user(user_id)
        new_code, _salt = self.issue_recovery_code(user_id)
        logger.info("重置成功：user_id=%s（撤销旧令牌 %d 枚，恢复码已轮换）", user_id, revoked)
        return AuthOutcome(user={"user_id": user_id, "username": str(row["username"])},
                           recovery_code=new_code)

    def _dummy_recovery_verify(self, code: str) -> None:
        """不存在用户/无恢复码时的等价耗时（复用 F-A 的假凭据哈希）。"""
        verify_password(code, self._dummy_hash)

    @staticmethod
    def _invalid_recovery() -> FieldError:
        """重置失败的**唯一**返回体（三种凭据错误逐字相同，`AC-AU-53`）。"""
        return FieldError(ERROR_INVALID_RECOVERY, INVALID_RECOVERY_MESSAGE, "")

    # ---------- 登录 ----------

    def login(self, username: Any, password: Any) -> AuthOutcome:
        normalized = normalize_username(username)
        # 从 UX 规范 §3.9 的表格：**格式非法**与**凭据错误**分开（前者 400，后者 401）。
        # 这不算"泄露用户名存在性"：格式合法与否只取决于这次请求本身，与库里有谁无关。
        problem = username_problem(normalized)
        if problem:
            return AuthOutcome(error=FieldError(ERROR_INVALID_USERNAME, problem, "username"))

        self.ensure()
        row = self.store.get_auth_user_by_name(normalized)
        if row is None:
            # 「用户不存在」：跑一次等价 scrypt 再回同样的 401（不可区分 + 时序抹平）
            self._dummy_verify(str(password or ""))
            logger.info("登录失败：username=%s（reason=no_such_user）", normalized)
            return AuthOutcome(error=self._invalid_credentials())

        if not self.verify_password(str(password or ""), row.get("password_hash") or ""):
            logger.info("登录失败：username=%s（reason=bad_password）", normalized)
            return AuthOutcome(error=self._invalid_credentials())

        user_id = str(row["user_id"])
        token, expires_at = self.issue_token(user_id)
        logger.info("登录成功：user_id=%s", user_id)
        return AuthOutcome(user={"user_id": user_id, "username": str(row["username"])},
                           token=token, expires_at=expires_at)

    @staticmethod
    def _invalid_credentials() -> FieldError:
        """登录失败的**唯一**返回体：两条分支逐字相同（`AC-AU-25/26`）。"""
        return FieldError(ERROR_INVALID_CREDENTIALS, INVALID_CREDENTIALS_MESSAGE, "")

    # ---------- 令牌 ----------

    def issue_token(self, user_id: str) -> tuple[str, float]:
        """签发令牌：高熵随机串（服务端生成，不可预测），库里只存它的 sha256。"""
        self.ensure()
        now = time.time()
        token = secrets.token_urlsafe(TOKEN_BYTES)
        expires_at = now + self.token_ttl_seconds
        self.store.create_auth_token(sha256_hex(token), user_id, now, expires_at)
        # ⚠️ 日志里只有 user_id 与过期时间，**绝不打印令牌**（`AC-AU-30` 的要求延伸到日志）
        logger.info("签发令牌：user_id=%s 有效期至 %.0f（库中只存哈希）", user_id, expires_at)
        return token, expires_at

    def resolve(self, token: str | None) -> dict | None:
        """令牌 → 用户信息（``{"user_id","username","expires_at"}``）。

        无效（缺失 / 伪造 / 已登出 / 过期）一律 ``None``，**不区分原因** ——
        调用方因此不可能把"这个令牌过期了"和"没这个令牌"讲成两件事。
        """
        if not token:
            return None
        self.ensure()
        row = self.store.get_auth_token(sha256_hex(str(token)))
        if row is None:
            return None
        if row.get("revoked_at"):
            return None
        expires_at = float(row.get("expires_at") or 0)
        if expires_at <= time.time():
            return None
        user_id = str(row["user_id"])
        account = self.store.get_auth_user(user_id)
        return {
            "user_id": user_id,
            "username": str(account["username"]) if account else user_id,
            "expires_at": expires_at,
        }

    def current_user(self, token: str | None) -> dict | None:
        return self.resolve(token)

    def revoke(self, token: str | None) -> bool:
        """登出：置 ``revoked_at``，**该令牌立即失效**（校验侧看这个字段）。"""
        if not token:
            return False
        self.ensure()
        row = self.store.get_auth_token(sha256_hex(str(token)))
        if row is None or row.get("revoked_at"):
            return False
        changed = self.store.revoke_auth_token(sha256_hex(str(token)))
        logger.info("登出：user_id=%s 令牌已撤销（ revoked=%d）", row.get("user_id"), changed)
        return bool(changed)

    # ---------- 鉴权中间件用 ----------

    def authorize(self, token: str | None, claimed_user_id: str | None) -> dict | None:
        """强制鉴权模式下解析登录态；**防"带 A 的 token 用 B 的 user_id"**。

        * 开关关闭 → 返回 ``None``（调用方继续保持既有行为，一行都不变）；
        * 开关打开 + 令牌无效 → 返回 ``{"error": 401 结构}``；
        * 开关打开 + 令牌有效 + ``claimed_user_id`` 与登录态不一致 → 403 ``user_mismatch``；
        * 开关打开 + 令牌有效 + 一致（或没传 claimed）→ 返回 ``{"user": {...}}``。

        返回结构故意做成"可判定的字典"而不是异常：端点层才决定怎么变成 HTTP 响应，
        这样这条规则可以被单测直接钉住（不必起 HTTP）。
        """
        if not self.require_auth:
            return None
        user = self.resolve(token)
        if user is None:
            return {"error": {"status_code": 401, "error": ERROR_UNAUTHENTICATED,
                              "message": "未登录或登录已过期"}}
        claimed = normalize_username(claimed_user_id) if claimed_user_id else ""
        if claimed and claimed != str(user["user_id"]):
            logger.warning("拒绝越权：登录态 user_id=%s，请求体/查询串声明 user_id=%s",
                           user["user_id"], claimed)
            return {"error": {"status_code": 403, "error": ERROR_USER_MISMATCH,
                              "message": "请求中的 user_id 与登录账号不一致"}}
        return {"user": user}

    # ---------- 用户名查重 ----------

    def check_username(self, username: Any, *, client_key: str = "") -> dict:
        """查重（`AC-AU-15` 的前端预检；服务端注册路径仍会再兜一次）。

        返回 ``{"status": 200|400|429, "body": {...}}``：

        * ``200 {"username": <规范化>, "available": true|false}``
          —— **命中已占用也回 200**：``available`` 才携带信息，
          这样"能不能用"这件事只有一种解析方式（HTTP 状态码统一表示请求本身的问题）；
        * ``400 {"error": "invalid_username", "message": ...}`` 格式非法；
        * ``429 {"error": "rate_limited", "message": ...}`` 超限（窗口内 ≤30 次/60s）。

        只回"可用与否"，**不回**账号是否存在之外的任何信息（不泄露 created_at、
        不泄露 user_id 以外的东西）。
        """
        if client_key:
            allowed, retry_after = self._rate_check(client_key)
            if not allowed:
                return {"status": 429,
                        "body": {"error": ERROR_RATE_LIMITED,
                                 "message": f"查得太频繁了，{retry_after} 秒后再试"},
                        "retry_after": retry_after}
        normalized = normalize_username(username)
        problem = username_problem(normalized)
        if problem:
            return {"status": 400,
                    "body": {"error": ERROR_INVALID_USERNAME, "message": problem}}
        self.ensure()
        taken = self.store.get_auth_user_by_name(normalized) is not None
        return {"status": 200, "body": {"username": normalized, "available": not taken}}

    def _rate_check(self, client_key: str) -> tuple[bool, int]:
        """限速（`AC-AU-36`）；限速器自身异常时**放行**并记 warning。

        为什么异常时放行而不是拒绝：这个接口只是"体验优化"，拒绝服务会把正常用户
        挡在注册门外（规范的 `AC-AU-14` 也要求前端在查重失败时仍允许提交）。
        """
        try:
            if self.limiter.allow(client_key):
                return True, 0
            return False, self.limiter.retry_after(client_key)
        except Exception as exc:  # noqa: BLE001 - 限速器坏掉不该让注册不可用
            logger.warning("用户名查重限速器异常（放行本次请求）：%s: %s",
                           type(exc).__name__, exc)
            return True, 0

    def reset_rate_limits(self) -> None:
        """清空限速计数（测试用；也给运维留一个手动排障入口）。"""
        self.limiter.reset()


def extract_token(cookie_value: str | None, authorization: str | None) -> str | None:
    """从请求里取令牌：**HttpOnly Cookie 优先**（浏览器路径），其次 ``Bearer``（脚本路径）。"""
    if cookie_value:
        return cookie_value
    if authorization:
        text = authorization.strip()
        for prefix in ("Bearer ", "bearer ", "Token ", "token "):
            if text.startswith(prefix):
                return text[len(prefix):].strip() or None
        return None
    return None


def iter_public_user_fields() -> Iterable[str]:
    """对外可见的用户字段（**白名单**，其余字段一律不出现在响应里）。"""
    return ("user_id", "username")
