# token 与上下文的判据（JWT 签发/校验、身份对象、RBAC）。它从 test_security.py 里
# 按**面向的东西**分出来：那边是口令哈希与密钥装载，这边是 token 与权限 —— 两个文件
# 各自都在 300 行以内（单文件行数是本项目硬约束），且改一处只需读一个文件。
import base64
import json
import time

import jwt
import pytest

from app.core import security
from app.core.security import (
    AuthError, CurrentUser, PermissionDeniedError, Role, TokenExpiredError,
    TokenInvalidError,
)

# 68 字符：过 MIN_SECRET_LEN（32）
SECRET = "test-secret-0123456789abcdefghijklmnopqrstuvwxyz-ABCDEFGHIJKLMNOPQ"


def _user(role: Role = Role.LAWYER, team_id: str = "team-a") -> CurrentUser:
    """本文件统一的身份样本。team_id 刻意非空：空值连造都造不出来（见用例）。"""
    return CurrentUser(id=7, role=role, team_id=team_id)


class _Drop:
    """哨兵：表示「这个声明不要」。None 与「缺键」是两件事，必须能分别表达。"""


_DROP = _Drop()


def _craft(secret: str = SECRET, **overrides) -> str:
    """绕过 sign_token 直接造一个 token —— 模拟**外部输入**（别人写的载荷）。

    必须绕过：sign_token 只会产出合法载荷，而这一类用例要验的恰恰是「载荷被改过、
    或按伪造格式发来时怎么办」，只有直接调 PyJWT 才造得出来。
    """
    payload = {"sub": "7", "role": "lawyer", "team_id": "team-a",
               "iat": int(time.time()), "exp": int(time.time()) + 3600}
    for key, value in overrides.items():
        if value is _DROP:
            payload.pop(key, None)
        else:
            payload[key] = value
    return jwt.encode(payload, secret, algorithm="HS256")

# ---- JWT ----

def test_token_roundtrip_carries_the_identity():
    """往返要保住三样东西：id、角色、团队。id 回来必须是 int（要拿去查库/写审计）。"""
    user = _user(role=Role.PARTNER, team_id="team-b")
    back = security.verify_token(security.sign_token(user, secret=SECRET),
                                 secret=SECRET)
    assert back == user
    assert isinstance(back.id, int) and back.id == 7
    assert back.role is Role.PARTNER
    assert isinstance(back.team_id, str) and back.team_id == "team-b"


def test_exp_and_iat_are_integer_seconds():
    """有效期由 ttl_s 唯一决定（免得「有效期」变成随实现漂的量）。

    now 注入一个固定值而不是靠当前时刻：断言才写得成等式。它取 time.time() 的整数秒，
    这样 token 仍是「未过期」的，直接 decode 就能读回声明。
    """
    now = int(time.time())
    token = security.sign_token(_user(), secret=SECRET, now=now, ttl_s=60)
    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert payload["iat"] == now and payload["exp"] == now + 60


def test_default_ttl_is_the_constant_and_the_constant_is_a_workday():
    """TTL 的两层护栏，各挡一种改坏（此前两层都没有，复审实测）。

    ①**默认值必须真的取自常量**：不传 ttl_s 签一个，exp - iat 恰好等于 TOKEN_TTL_S。
    这一层挡的是「默认值被写成另一个字面量」（如 `ttl_s: int = 365*24*3600`）。
    ②**常量本身必须是裁决过的那个数（8 小时）**：只写①是**恒真**的 —— 等号两边同源于
    同一个常量，把 TOKEN_TTL_S 整段改成一年，两边一起变、断言照样成立（本次变异实测
    确认：34 passed 全绿）。而报告 §二 8 的「停用账号残留窗口 = TOKEN_TTL_S」整段押在
    这个数上，故必须有一个不吃同源性的锚：字面量。改这个数= 改报告里那条结论，
    要经这里过一次（TTL 已被列为待裁决项，确认后连同本行一起改，这是有意的动作）。
    """
    now = int(time.time())
    payload = jwt.decode(security.sign_token(_user(), secret=SECRET, now=now),
                         SECRET, algorithms=["HS256"])
    assert payload["iat"] == now, "给了 now 却不认，等式的前提没了"
    assert payload["exp"] - payload["iat"] == security.TOKEN_TTL_S
    assert security.TOKEN_TTL_S == 8 * 3600, "TTL 改了 —— 报告「残留窗口」那段的结论要一并复核"


def test_expired_token_raises_expired_not_invalid():
    """过期与签名不符必须是**两个不同的类**，对内的区分就建立在这上面。

    用 `type(...) is` 而不是 isinstance：后者放过子类，而「过期是「不合法」的一种」
    这种继承一旦成立，接口层想分开统计/告警时就再也分不开了。
    """
    token = security.sign_token(_user(), secret=SECRET,
                                now=int(time.time()) - 7200, ttl_s=3600)
    with pytest.raises(TokenExpiredError) as exc:
        security.verify_token(token, secret=SECRET)
    assert type(exc.value) is TokenExpiredError
    assert isinstance(exc.value, AuthError), "接口层只捕 AuthError 那一步必须成立"


def test_token_signed_with_another_secret_rejected():
    token = security.sign_token(_user(), secret=SECRET + "x")
    with pytest.raises(TokenInvalidError):
        security.verify_token(token, secret=SECRET)


def test_tampered_signature_rejected():
    """改签名末位 —— 最朴素的伪造形态。

    **替换字符必须落在另一个 base64 分组里**（2026-09-29 任务 3 全量跑撞到一次偶发红
    后查实）：43 字符的 HS256 签名里，末位只承载 4 位有效信息，低 2 位是 base64 的
    填充位。把 'A'(0) 改成 'B'(1) 只动了填充位，解出来的签名与原签名**逐字节相同**，
    「篡改」当场变成没篡改 —— 用例以「DID NOT RAISE」偶发失败。实测 400 个不同签名样本：
    末位只会取 16 个值（4 的倍数），原写法有 27/400 ≈ 1/16 的概率空改；换成
    「'A' ↔ 'E'」（末位分组必然不同）后 400/400 都被拒。
    """
    head, payload, signature = security.sign_token(_user(), secret=SECRET).split(".")
    flipped = "E" if signature[-1] == "A" else "A"
    with pytest.raises(TokenInvalidError):
        security.verify_token(f"{head}.{payload}.{signature[:-1]}{flipped}",
                              secret=SECRET)


def test_tampered_payload_rejected():
    """把载荷里的角色改成 partner、签名原样留着 —— 这就是提权攻击的形状。

    JWT 的载荷是明文（只签名不加密），谁都能改；签名是唯一的拦路石，这一条钉的就是它。
    """
    head, payload, signature = security.sign_token(
        _user(role=Role.ASSISTANT), secret=SECRET).split(".")
    raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    forged = raw.replace(b'"assistant"', b'"partner"')
    assert forged != raw, "载荷里没有 assistant 字样，改不动 —— 本用例已失去意义"
    new_payload = base64.urlsafe_b64encode(forged).decode().rstrip("=")
    with pytest.raises(TokenInvalidError):
        security.verify_token(f"{head}.{new_payload}.{signature}", secret=SECRET)


def test_alg_none_token_rejected():
    """alg=none 的经典混淆：不签名、自称「无算法」。算法白名单必须拒它。"""
    header = base64.urlsafe_b64encode(
        b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    claims = {"sub": "7", "role": "partner", "team_id": "team-a",
              "exp": int(time.time()) + 3600}
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    with pytest.raises(TokenInvalidError):
        security.verify_token(f"{header}.{body}.", secret=SECRET)


def test_token_signed_with_an_unlisted_algorithm_rejected():
    """白名单外的算法一律拒。用 HS512 当样本（同一密钥，只是换了算法）——
    它代表的就是整个白名单之外的家族，包括 RS256 那类拿公钥当 HMAC 密钥的混淆。
    """
    payload = {"sub": "7", "role": "lawyer", "team_id": "team-a",
               "exp": int(time.time()) + 3600}
    token = jwt.encode(payload, SECRET, algorithm="HS512")
    with pytest.raises(TokenInvalidError):
        security.verify_token(token, secret=SECRET)


@pytest.mark.parametrize("broken", [
    {"role": _DROP},        # 少了角色
    {"team_id": _DROP},     # 少了团队 —— 三层隔离的过滤依据
    {"exp": _DROP},         # 少了过期时间：token 永不过期
    {"sub": _DROP},         # 少了 id
    {"team_id": ""},        # 空团队：按团队过滤会命中全表
    {"team_id": None},
    {"team_id": 42},        # 类型不对
    {"sub": "abc"},         # id 不是数字
    {"sub": "7.5"},
    {"role": "root"},       # 不认识的角色
    {"role": None},         # 角色类型不对
])
def test_broken_claims_rejected(broken):
    """载荷里的每一样都得站得住，逐个来 —— 一整排里漏一个，就等于留了一条提权路。"""
    with pytest.raises(TokenInvalidError):
        security.verify_token(_craft(**broken), secret=SECRET)


def test_an_accepted_token_always_carries_a_non_empty_team():
    """设计 §六 的反向钉法：被接受的上下文必带非空 team_id。

    只验「坏的被拒」是不够的 —— 实现若把 team_id 从上下文里整个拿掉（CurrentUser
    少一个字段），那些用例会全绿，而数据层隔离的过滤依据就没有了。
    """
    user = security.verify_token(_craft(), secret=SECRET)
    assert user.team_id == "team-a"


@pytest.mark.parametrize("garbage", ["", "abc", "a.b.c", "a.b.c.d", None, 12345, b"x"])
def test_garbage_tokens_are_invalid_not_crashes(garbage):
    """不是 token 的东西必须走 TokenInvalidError，而不是 TypeError/AttributeError
    这类在接口层会变成 500 的形状。"""
    with pytest.raises(TokenInvalidError):
        security.verify_token(garbage, secret=SECRET)


def test_current_user_refuses_a_dirty_identity():
    """脏身份在构造处就拦：空 team_id、没转成枚举的角色都不许被造出来。"""
    with pytest.raises(ValueError):
        CurrentUser(id=1, role=Role.LAWYER, team_id="")
    with pytest.raises(ValueError):
        CurrentUser(id=1, role=Role.LAWYER, team_id=None)
    with pytest.raises(ValueError):
        CurrentUser(id=1, role="lawyer", team_id="team-a")


def test_from_row_maps_the_table_columns_and_refuses_inactive():
    """登录端点从 users 表一行造上下文：列名是这张表与鉴权层之间的契约。

    停用账号在这一步拒（token 一旦签发就不再查库，所以这是「停用」唯一能立刻生效的
    时机），且抛的是与「口令不对」同族的异常 —— 不告诉对方「账号存在只是被停了」。
    """
    row = {"id": 3, "username": "zhangsan", "password_hash": "x",
           "role": "partner", "team_id": "team-z", "is_active": 1}
    user = CurrentUser.from_row(row)
    assert (user.id, user.role, user.team_id) == (3, Role.PARTNER, "team-z")
    with pytest.raises(PermissionDeniedError):
        CurrentUser.from_row({**row, "is_active": 0})


# ---- RBAC ----

def test_role_is_a_plain_enum_and_reads_back_only_by_explicit_conversion():
    """Role 与库里的字符串**不**直接相等 —— 钉住实参语义，防 docstring 再漂一次。

    复审发现 docstring 曾声称「用 str 混入」（`row["role"] == Role.LAWYER` 直接成立），
    而实现是纯 Enum：照那句话写权限判断会静默为假，写成 `!=` 就成了静默放行。本用例
    把**实际**语义钉住：读回处只能走显式 `Role(...)`（from_row 与 _to_user 正是如此，
    各自的用例另钉），若将来有人真把 Role 改成 str 混入，这里先红 —— 那时 docstring
    与 has_role 的 fail-closed 口径要一并复核，是一次有意的动作，不是顺手改。
    """
    assert Role.LAWYER != "lawyer", "Role 成了 str 混入，docstring 与比较口径要一并改"
    assert not isinstance(Role.LAWYER, str), "同上：str 混入会让误传字符串变成 fail-open"
    assert Role("lawyer") is Role.LAWYER, "显式转换这条路必须通（读回处全靠它）"


def test_role_matrix_allows_only_the_matching_role():
    """3×3 矩阵：每个身份角色 × 每个「只含一个角色的集合」的允许/拒绝。"""
    for user_role in Role:
        for allowed in Role:
            user = _user(role=user_role)
            if allowed is user_role:
                assert security.has_role(user, {allowed}) is True
                assert security.require_role(user, {allowed}) is None
            else:
                assert security.has_role(user, {allowed}) is False
                with pytest.raises(PermissionDeniedError):
                    security.require_role(user, {allowed})


def test_role_check_with_multiple_allowed_roles():
    """集合里有多个角色时按「属于」判 —— 与顺序、数量都无关。"""
    user = _user(role=Role.ASSISTANT)
    assert security.has_role(user, {Role.LAWYER, Role.ASSISTANT}) is True
    assert security.require_role(user, [Role.PARTNER, Role.ASSISTANT]) is None
    with pytest.raises(PermissionDeniedError):
        security.require_role(user, (Role.LAWYER, Role.PARTNER))


def test_empty_allowed_set_denies_everyone():
    """空集合 = 谁都不许。它是很容易写出来的手滑（兜底传了 set()），
    若被当成「没有要求」而放行，就成了一个静默的全开。"""
    with pytest.raises(PermissionDeniedError):
        security.require_role(_user(), set())


def test_permission_denied_is_an_auth_error():
    """角色不足与未认证对外走同一个出口（设计 §二 第 9 条：一律 404），
    所以它必须是 AuthError 的一支 —— 接口层只需捕基类。"""
    assert issubclass(PermissionDeniedError, AuthError)
    with pytest.raises(AuthError):
        security.require_role(_user(role=Role.ASSISTANT), {Role.PARTNER})
