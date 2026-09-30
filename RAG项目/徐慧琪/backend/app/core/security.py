"""鉴权原语：口令哈希、JWT 签发与校验、RBAC、当前用户上下文。

本模块只出**纯函数与异常**：FastAPI 的 `Depends` 包装在 api/ 侧（任务 4），所以这里
的单测不起 HTTP、不连库 —— 哈希与 token 都是「给定输入算出输出」，把框架或连接塞进
来只会让这些判据的验证变慢、变脆。

边界（与设计 §三 的分层规则相容）：**本模块不 import app.core.factory** —— 鉴权不该
把 2GB 模型加载拖起来。事实上它连 db/ 都不 import：读账号是 api/ 侧的事，本模块只
认「一行记录里那几个字段」（见 CurrentUser.from_row）。这条边由测试按 AST 扫描钉住。
"""
from __future__ import annotations

import base64
import dataclasses
import enum
import hashlib
import hmac
import os
import secrets
import time
from collections.abc import Collection, Mapping

import jwt

# scrypt 参数（设计 §二 第 8 条：标准库实现，不引 passlib/bcrypt）。
# n 是主旋钮：内存 ≈ 128 * n * r 字节，耗时随 n 线性。2**14 / 8 / 1 本机实测约 40ms
# 一次、16 MiB —— 登录路径上无感，而离线爆破每一次猜测也要付同一量级的代价。
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16

# 校验侧的参数上下限。上限的作用是**挡住库里的荒谬值**：这一列可能被人工改坏，若存进
# 一个 n=2**30 的哈希，一次登录就要 128 GiB 内存 —— 等于一个白送的远程打挂。超限
# 一律判否、不去算：「算不出来」与「对不上」在对外行为上本来就该是同一种结果。
# p 单独设限是因为它是**纯 CPU 时间**旋钮：n 被内存上限约束住时，p=100000 仍能让一次
# 登录占住一个核几十秒。
_SCRYPT_LIMITS = {"n": (2, 2 ** 15), "r": (1, 8), "p": (1, 8), "dklen": (16, 64)}
# 显式给 maxmem：不给时 OpenSSL 用 32 MiB 的默认上限，而 n=2**15 恰好要 32 MiB ——
# 本机实测 hashlib.scrypt(n=2**15, r=8) 直接抛 "memory limit exceeded"，即将来调大
# 参数时旧代码连**新记录都算不出来**，而失败还会被上面那条判否逻辑静默吃掉（表现是
# 「所有人的口令突然都不对」）。取 64 MiB 为上限值（n 上限 × r 上限 = 32 MiB）的两倍：
# 实测「恰好等于所需内存」会被拒（n=2**16, r=8 要 64 MiB，maxmem 也给 64 MiB 时抛错），
# 贴着边界给会踩到。**调大 _SCRYPT_LIMITS 的 n 或 r 时必须同时调大这里**，两处一体。
_SCRYPT_MAXMEM = 64 * 1024 * 1024


class MissingSecretError(RuntimeError):
    """签名密钥未配置。**刻意不是 AuthError 的子类**。

    接口层把 AuthError 映射成 404（设计 §二 第 9 条）。它若也算 AuthError，一台没配
    密钥的服务器会把所有鉴权请求回成 404 —— 把「这台机器配置错了」伪装成「你没登录」，
    运维会顺着鉴权查下去，而真正的问题在环境变量。与 llm_router 的 MissingAPIKeyError
    同口径：配置缺失是服务端故障，要响。
    """


class AuthError(Exception):
    """鉴权失败的基类。接口层只需捕获它，细类见下（对内区分、对外不区分）。"""


class TokenExpiredError(AuthError):
    """token 过期。与签名不符分开的理由见 verify_token 的注释。"""


class TokenInvalidError(AuthError):
    """签名不符、被篡改、结构不合法、声明缺失或值不认识。"""


class PermissionDeniedError(AuthError):
    """身份有效但角色不够，或账号已停用。"""


class Role(enum.Enum):
    """所内角色（设计 §五：lawyer / assistant / partner）。

    **普通 Enum，不是 str 混入**：`Role.LAWYER == "lawyer"` 为假、`isinstance(...,
    str)` 为假、`json.dumps(Role.LAWYER)` 直接 TypeError。库里那一列与 JWT 载荷里
    都是普通字符串，所以**读回处一律显式 `Role(...)` 转换**（CurrentUser.from_row
    与 _to_user 各一处），不靠隐式相等：写成 `row["role"] == Role.LAWYER` 会静默为
    假，成对写成 `!=` 时就是静默放行 —— 两种都过类型检查、都要等一次越权才被发现。
    （本 docstring 曾声称「用 str 混入」，与实际相反，已按实现改写。）
    不真去混 str 的方向性理由：误把字符串传进 has_role/require_role 时，纯 Enum 判否
    （fail-closed），而 str 混入会因值相等判成「在集合里」（fail-open）—— 权限判断上
    两个方向的代价不对称（同 _to_user 的口径）。载荷一侧显式取 .value：纯 Enum 下这
    不是风格问题，是 json 能编码它的前提。
    """

    LAWYER = "lawyer"
    ASSISTANT = "assistant"
    PARTNER = "partner"


def hash_password(password: str) -> str:
    """算出可直接存库的自描述哈希串：`scrypt$n$r$p$盐$哈希`。

    参数写进串本身（设计 §五）是为了**将来调参时旧记录仍可校验**：校验只认串里记着
    的那组参数，不必回头猜「这一行是哪一代代码写的」。盐每次随机，且用 secrets 而不是
    random —— random 是可预测的梅森旋转，盐一旦可预测就等于没有盐。
    空口令当场拒：空串一样能算出合法哈希，落到库里就是一个谁都能登录的账号，而它在
    界面上与正常账号长得完全一样。口令**强度**门槛（长度等）不设在这一层：机制与策略
    分开，策略放在能对人给反馈的地方（运维脚本）。
    """
    if not password:
        raise ValueError("口令不能为空：空口令账户能被任何人登录")
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N,
                            r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_DKLEN,
                            maxmem=_SCRYPT_MAXMEM)
    return "$".join(("scrypt", str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P),
                     _b64(salt), _b64(digest)))


def _b64(raw: bytes) -> str:
    """base64 编码。选它而不是 hex：同样字节数下短一半（串要进 VARCHAR(255)）。

    标准字母表里没有 `$`，所以「按 `$` 切分」永远是安全的 —— 这正是自描述格式能成立
    的前提；换成别的编码前要先确认这一点。
    """
    return base64.b64encode(raw).decode("ascii")


def verify_password(password: str, stored: str) -> bool:
    """校验口令是否吻合。**任何畸形输入一律返回 False，不抛异常。**

    为什么判否而不是抛：调用点是登录端点，而 stored 来自库（可能被人工改坏、被截断、
    是上一代格式）。让它抛会把「登录失败」升格成 500，把「这行数据坏了」这个内部事实
    顺着错误响应漏出去；而「对不上」本来就是登录失败的标准答案，调用方无需区分两者。
    比较用 hmac.compare_digest 而不是 ==：== 在第一个不同字节处就返回，耗时里含有
    「猜对了几位」的信息，足以支撑逐字节爆破。
    """
    parsed = _parse(stored)
    if parsed is None:
        return False
    n, r, p, salt, expected = parsed
    try:
        digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                                dklen=len(expected), maxmem=_SCRYPT_MAXMEM)
    except (ValueError, MemoryError, OverflowError):
        # 超限的参数已在 _parse 里挡掉，但限内仍有算不出来的组合（机器内存紧张、
        # dklen 与 n 的某些搭配）。登录路径上算不出来只能是「对不上」。
        return False
    return hmac.compare_digest(digest, expected)


def _parse(stored: str) -> tuple[int, int, int, bytes, bytes] | None:
    """拆自描述串；任何不合格式之处返回 None（由 verify_password 翻成 False）。

    只认 scrypt 这一种前缀：将来真要换算法时，前缀让迁移有据可依；而「不认识的算法」
    只能判否 —— 拿去算是不可能的，假装它通过更不可能。
    """
    if not isinstance(stored, str):
        return None
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return None
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        # validate=True：默认的 b64decode 会**静默丢弃**非法字符，于是「一堆乱码」
        # 可能被解码成短字节串后继续往下走；严格模式让它当场报错、走判否那一支
        salt = base64.b64decode(parts[4], validate=True)
        expected = base64.b64decode(parts[5], validate=True)
    except (ValueError, TypeError):
        return None
    if not _within(n, _SCRYPT_LIMITS["n"]) or not _within(r, _SCRYPT_LIMITS["r"]):
        return None
    if not _within(p, _SCRYPT_LIMITS["p"]) or not _within(len(expected), _SCRYPT_LIMITS["dklen"]):
        return None
    if not salt or not expected:
        return None
    return n, r, p, salt, expected


def _within(value: int, bounds: tuple[int, int]) -> bool:
    """值是否落在闭区间内（含两端：上限本身就是「合法但最大」那一档）。"""
    return bounds[0] <= value <= bounds[1]


@dataclasses.dataclass(frozen=True)
class CurrentUser:
    """当前用户上下文 —— **数据层隔离的唯一注入点**（设计 §六）。

    三层隔离里，数据层这一层在本轮只有「待接点」（案件表尚未落地）。这个对象就是那
    个接点，因此有一条必须写死的约定：**凡将来要按团队过滤的查询函数，签名必须接收
    它，而不是从全局变量或连接里取 team_id**。team_id 只允许从 token 里来（签发时
    取自 users 表那一列，见设计 §五「三层隔离里 team_id 的唯一来源」），谁都不能在
    别处「补」一个默认团队 —— 那等于把隔离条件让给了调用方的默认值。

    frozen：请求处理中途改身份没有任何合法用途；更要紧的是下游若按 id 做了缓存或审计，
    一个可变的身份对象会让「审计里记的身份」与「实际执行的身份」静默分叉。
    """

    id: int
    role: Role
    team_id: str

    def __post_init__(self) -> None:
        """脏身份不许被造出来（空 team_id、未转成枚举的角色都当场拒）。

        放在构造处而不是各调用点：它是这个类型的定义的一部分。空 team_id 尤其要在
        这里拦 —— 它一路传到查询层的表现是「按团队过滤的谓词恰好命中全表」，一个
        最危险的失效形态长得跟正常一样。
        """
        if not isinstance(self.role, Role):
            raise ValueError(f"role 必须是 Role 枚举：{self.role!r}")
        if not isinstance(self.team_id, str) or not self.team_id:
            raise ValueError("team_id 不能为空：它是数据层隔离的过滤依据")

    @classmethod
    def from_row(cls, row: Mapping) -> CurrentUser:
        """从 users 表的一行造上下文（登录端点用）。停用账号在这里就拒。

        is_active 只有在这里看才作数：token 一旦签发，校验就不再查库（见 TOKEN_TTL_S
        的注释），所以「停用」能立刻生效的唯一时机就是登录这一次。
        拒的形式是 PermissionDeniedError（AuthError 的一种）：对外与「口令不对」是同
        一种响应，不告诉对方「这个账号存在只是被停了」。
        """
        if not row.get("is_active"):
            raise PermissionDeniedError("账号已停用")
        return cls(id=int(row["id"]), role=Role(row["role"]),
                   team_id=row["team_id"])


# 密钥只从环境变量读（沿用 llm_router 对 DeepSeek 密钥的既有口径：不写进代码、不进
# 日志）。FL_ 前缀的理由见 core/config.py：本机用户环境变量里已有别的项目留下的同名
# 变量，撞车的后果是用别人的密钥签发 token，而一切看起来都正常。
JWT_SECRET_ENV = "FL_JWT_SECRET"
# 算法白名单写死一个，校验时交给 PyJWT 的 algorithms=。本机实测的两条（PyJWT 2.13）：
#   ①「忘传」不会静默降级 —— algorithms=None 直接抛 DecodeError，不需要我们自己防；
#   ②真正的坑是**照抄 token 头部自称的 alg**（algorithms=[get_unverified_header(t)["alg"]]）：
#     那样一个 HS512 的 token 会被原样接受，白名单等于摆设。测试就用 HS512 与 alg=none
#     各钉一条（前者复现②，后者配合 verify_signature 那一档）。
JWT_ALGORITHM = "HS256"
# token 有效期：一个工作日。它同时是**停用账号的已签发 token 的残留窗口** —— 校验只看
# 签名与 exp，不查库（本任务只做纯逻辑；逐请求查库是否值得由接口层定，见任务 4）。
TOKEN_TTL_S = 8 * 3600
# 密钥长度下限：HS256 的全部安全性押在这个串上，短到可枚举的密钥等于没有签名。
# 32 是「明显不像口令」的下限，不是某种熵的换算。
MIN_SECRET_LEN = 32


def load_secret(env: Mapping[str, str] | None = None) -> str:
    """从环境变量取签名密钥；缺失或过短当场抛。env 可注入（单测不必改进程环境）。

    抛而不是退回一个内置默认值：有默认值的密钥在每台机器上都一样，等于公开 —— 而
    服务带着一个公开密钥「一切正常」地签发 token，比启动失败糟得多。
    报错只提变量名与长度，**不回显取值**：密钥进日志等于泄露。
    """
    raw = ((os.environ if env is None else env).get(JWT_SECRET_ENV) or "").strip()
    # `if not raw` 这一支**不是护栏**：MIN_SECRET_LEN 是 32，空串本来就过不了下面那道
    # 长度检查，删掉它结论不变（实测：把它改成 `if False` 后全部用例照绿）。留着只为
    # 把「未设置」与「配了但太短」这两种情形在日志里分开 —— 它们的修法不同（一个去拿
    # 密钥，一个去换密钥），而报错文案是运维唯一能看到的区别。与 fees.verify_range 里
    # 那两行「提前返回，不是护栏」同一个口径
    if not raw:
        raise MissingSecretError(
            f"环境变量 {JWT_SECRET_ENV} 未设置或为空；签名密钥必须由环境提供")
    if len(raw) < MIN_SECRET_LEN:
        raise MissingSecretError(
            f"环境变量 {JWT_SECRET_ENV} 过短（{len(raw)} 字符 < {MIN_SECRET_LEN}）："
            "短到可枚举的密钥等于没有签名")
    return raw


def sign_token(user: CurrentUser, *, secret: str, ttl_s: int = TOKEN_TTL_S,
               now: int | None = None) -> str:
    """签发 token。now 可注入：过期用例不必 sleep，也不再依赖真实时钟。

    载荷按设计 §二 第 2 条至少含 sub / role / team_id / exp；iat 一并写上是给排查用的
    （「这个 token 是什么时候发的」比「什么时候过期」更常被问到）。
    sub 存字符串（JWT 规范如此，且 PyJWT ≥2.10 会拒绝非字符串的 sub），读回时再转 int；
    其余三个字段都取自 CurrentUser —— 也就是只有一处能决定 token 里的身份。
    """
    issued = int(time.time()) if now is None else int(now)
    payload = {"sub": str(user.id), "role": user.role.value,
               "team_id": user.team_id, "iat": issued, "exp": issued + int(ttl_s)}
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_token(token: str, *, secret: str) -> CurrentUser:
    """校验 token 并还原成当前用户上下文。

    **过期与签名不符在内部是两种异常**，因为两者的性质相反：过期是正常的客户端现象
    （时钟走了一会儿、token 放了一夜），值得单独计数以判断「TTL 是不是太短」；签名
    不符只可能是伪造或篡改，是攻击信号，要能单独报警。对外二者不必区分（设计 §二
    第 9 条：未认证一律 404），所以 api/ 侧捕 AuthError 这一步就够 —— 细分是**对内**
    的，不强迫调用方分级处理，也不改变对外行为。
    """
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM],
                             options={"require": ["exp", "sub", "role", "team_id"]})
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("token 已过期") from exc
    except jwt.InvalidTokenError as exc:
        # 签名不符、alg 不在白名单、结构损坏、声明不合规全落这里
        raise TokenInvalidError("token 不合法") from exc
    return _to_user(payload)


def _to_user(payload: Mapping) -> CurrentUser:
    """把载荷还原成上下文；任何不认识的东西一律判「不合法」。

    为什么不宽松兜底（比如角色认不出就当律师）：token 载荷是**外部输入**（密钥一旦
    泄露、或将来有多签发方，内容是别人写的）。兜底的方向是给一个不存在的人开权限，
    而拒的方向最差也只是让人重新登录一次 —— 两边代价不对称，只能从严。
    """
    try:
        return CurrentUser(id=int(payload["sub"]), role=Role(payload["role"]),
                           team_id=payload["team_id"])
    except (KeyError, TypeError, ValueError) as exc:
        # CurrentUser 的空 team_id/非枚举角色校验也走这一支：脏身份从 token 里来的时候
        # 要翻成「token 不合法」，而不是让它以 ValueError 的形状逃到接口层变成 500
        raise TokenInvalidError(f"token 载荷不合法：{type(exc).__name__}") from exc


def has_role(user: CurrentUser, roles: Collection[Role]) -> bool:
    """角色是否在允许集合内。只查询、不抛——要拦人的场景用 require_role。"""
    return user.role in roles


def require_role(user: CurrentUser, roles: Collection[Role]) -> None:
    """角色不在允许集合内则抛 PermissionDeniedError，否则静默返回 None。

    返回 None 而不是 bool 是有意的：bool 会被写成 `if not require_role(...)`，而那种
    写法在「允许」时同样走进 if 体 —— 放行与拒绝的判断就这样被写反，且两种写法都过
    类型检查、都过绝大多数用例。要一个布尔值的情形用 has_role。
    """
    if not has_role(user, roles):
        raise PermissionDeniedError(
            f"角色 {user.role.value} 不在允许集合 "
            f"{sorted(role.value for role in roles)} 内")
