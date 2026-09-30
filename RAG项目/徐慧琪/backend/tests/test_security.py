# 口令哈希与密钥装载的判据，**不依赖 HTTP、不连库**：它们都是「给定输入算出输出」，
# 把框架或连接拉进来只会让这些用例变慢变脆。token 与 RBAC 的判据在
# test_security_token.py —— 按**面向的东西**分文件：那边动 token，这边动口令。
import ast
import base64
import hashlib
import hmac
import pathlib

import pytest

from app.core import security
from app.core.security import AuthError, CurrentUser, MissingSecretError

# 样本密钥（68 字符，过 MIN_SECRET_LEN）。token 用的那份在另一个文件里，各持一份 ——
# 动一个文件的样本不该惊动另一个文件
SECRET = "test-secret-0123456789abcdefghijklmnopqrstuvwxyz-ABCDEFGHIJKLMNOPQ"


# core/security.py 允许 import 的全部模块（白名单）。加一项是有意的动作：
# 逐项都有理由 —— 标准库算法件（base64/hashlib/hmac/secrets）、时间与环境、
# 类型与数据类、PyJWT。**没有** app.* / torch / pymysql。
ALLOWED_IMPORTS = {"__future__", "base64", "dataclasses", "enum", "hashlib",
                   "hmac", "os", "secrets", "time", "collections.abc", "jwt"}


def _imported_modules() -> set[str]:
    """按 AST 扫出 security.py 里 import 的模块名（含 from ... import 的模块侧）。"""
    source = pathlib.Path(security.__file__).read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def test_security_imports_only_lightweight_modules():
    """钉住依赖面：本模块只许用标准库与 PyJWT（brief：security 不 import factory）。

    为什么值得一条用例：任务 4 的每个认证路由都会 import 它，一旦有人在这里 import
    app.db / app.core.factory，**每个**鉴权请求都会背上连库或加载 2GB 模型的代价 ——
    而这种修改在功能上完全看不出问题（只是变慢），只有把依赖面钉死才拦得住。
    """
    imported = _imported_modules()
    assert imported <= ALLOWED_IMPORTS, f"多出未裁决的依赖：{imported - ALLOWED_IMPORTS}"
    # 反向也要钉：白名单允许的集合若被删空（比如整份文件被清空），上面那条恒真
    assert {"jwt", "hashlib", "hmac", "os"} <= imported


def test_current_user_docstring_is_the_injection_point():
    """设计 §六 要求当前用户上下文是「数据层隔离唯一注入点」，这句话要留在代码里。

    钉住措辞不是洁癖：案件表落地时，接这个口的人得先知道有这么一个口（以及「签名必须
    接收它、不许从全局/连接里取」这条约束）。文档里的约定一旦从代码里消失，下一个
    加查询函数的人只会看见一个普通的 dataclass。
    """
    doc = CurrentUser.__doc__ or ""
    assert "数据层隔离的唯一注入点" in doc
    assert "签名必须接收" in doc


# ---- 口令哈希 ----

def test_hash_roundtrip_and_wrong_passwords():
    """往返 + 三种「差一点」的错口令。逐字比对齐才看得出末尾空格这类差异。"""
    stored = security.hash_password("pw-123456")
    assert security.verify_password("pw-123456", stored) is True
    assert security.verify_password("pw-123457", stored) is False
    assert security.verify_password("PW-123456", stored) is False
    assert security.verify_password("pw-123456 ", stored) is False
    assert security.verify_password("", stored) is False


def test_hash_is_self_describing():
    """串的形状按设计 §五：scrypt$n$r$p$盐$哈希，参数就是当前这组。"""
    parts = security.hash_password("pw-123456").split("$")
    assert len(parts) == 6 and parts[0] == "scrypt"
    assert (int(parts[1]), int(parts[2]), int(parts[3])) == (
        security.SCRYPT_N, security.SCRYPT_R, security.SCRYPT_P)
    assert len(base64.b64decode(parts[4])) == security.SALT_BYTES
    assert len(base64.b64decode(parts[5])) == security.SCRYPT_DKLEN


def test_two_hashes_of_one_password_differ_in_both_salt_and_digest():
    """盐必须真起作用。只断言「两个整串不同」挡不住「盐写成常量」——
    那时整串照样不同（摘要位仍会变），而彩虹表一次命中全部账号，故把盐单独比。
    """
    first, second = (security.hash_password("same-pw") for _ in range(2))
    salt_a, salt_b = (base64.b64decode(h.split("$")[4]) for h in (first, second))
    assert salt_a != salt_b
    assert first.split("$")[5] != second.split("$")[5]
    assert security.verify_password("same-pw", first) is True
    assert security.verify_password("same-pw", second) is True


def _raw_hash(n: int, r: int, p: int, password: str = "pw-123456",
              dklen: int = 32) -> str:
    """手工按指定参数造一个哈希串（口令正确、值算得出来）—— 用来试边界。

    刻意不走 hash_password：那是被测代码，用它造样本就只能测到它自己那组参数，
    「校验方读的是串里的参数还是模块常量」这类事永远看不出来。
    """
    salt = b"\x07" * 16
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                            dklen=dklen, maxmem=security._SCRYPT_MAXMEM)
    return "$".join(("scrypt", str(n), str(r), str(p),
                     base64.b64encode(salt).decode(),
                     base64.b64encode(digest).decode()))


def test_verification_reads_the_params_recorded_in_the_string():
    """**自描述格式存在的全部意义**：参数变了以后旧记录仍可校验。

    样本用与模块当前参数**不同**的一组（n 加倍，仍在上限内）手工造 —— 校验方若偷偷
    用了常量 security.SCRYPT_N 而不是串里的值，摘要就对不上，本用例红。这种实现在
    结构上完全看不出来，只有这一条能钉住它。
    """
    assert security.SCRYPT_N != 2 ** 15, "样本参数与模块参数相同了，本用例已失去区分力"
    stored = _raw_hash(2 ** 15, 8, 1)
    assert security.verify_password("pw-123456", stored) is True
    assert security.verify_password("pw-123457", stored) is False


@pytest.mark.parametrize("params", [
    (2 ** 16, 4, 1, 32),     # n 超上限（2**16 > 2**15）
    (2 ** 14, 16, 1, 32),    # r 超上限（16 > 8）
    (2 ** 14, 8, 9, 32),     # p 超上限（9 > 8）：纯 CPU 时间那一路
    (2 ** 14, 8, 1, 8),      # dklen 低于下限（8 < 16）：截短的摘要弱得多
])
def test_params_outside_the_limits_are_refused_even_with_the_right_password(params):
    """超限一律判否 —— 哪怕口令是对的、哪怕这组参数其实算得出来。

    上限是安全设计的一部分（见 security._SCRYPT_LIMITS 的注释）：这一列若被写进一个
    n=2**30 或 p=10**5 的哈希，一次登录就能吃掉 128 GiB 内存或一个核几十秒 —— 那是
    白送的远程打挂。四个样本都刻意挑**算得出来**的那一档（比如 n=2**16 配 r=4 只要
    32 MiB，在 maxmem 之内），否则这道上限删掉时，底层「算不动」会把用例兜成绿。
    """
    n, r, p, dklen = params
    assert security.verify_password("pw-123456",
                                    _raw_hash(n, r, p, dklen=dklen)) is False


@pytest.mark.parametrize("stored", [
    "",                                            # 空串
    "not-a-hash",                                  # 完全不是哈希
    "scrypt$16384$8$1$YWJjZA==",                   # 段数不够（截断/旧格式）
    "bcrypt$16384$8$1$YWJjZA==$YWJjZA==",          # 不认识的算法前缀
    "scrypt$0$8$1$YWJjZA==$YWJjZA==",              # n 低于下限
    "scrypt$1073741824$8$1$YWJjZA==$YWJjZA==",     # n=2**30：内存上限挡的就是它
    "scrypt$16384$8$1$YWJjZA==$YWJjZA==",          # dklen=4，短于下限
    "scrypt$abc$8$1$YWJjZA==$YWJjZA==",            # 参数不是整数
    "scrypt$-1$8$1$YWJjZA==$YWJjZA==",             # 负数参数
    "scrypt$16384$8$1$!!!!$!!!!",                  # 不是合法 base64
    "scrypt$16384$8$1$$",                          # 盐与摘要都空
    None,                                          # 列里是 NULL
    12345,                                         # 列里是数字（人工改坏）
])
def test_malformed_stored_hashes_return_false(stored):
    """畸形串一律 False，不抛：调用点是登录端点，抛会把脏数据升格成 500，
    把「这一行坏了」这个内部事实漏给外面。"""
    assert security.verify_password("pw", stored) is False


def test_comparison_goes_through_compare_digest(monkeypatch):
    """比对必须走 hmac.compare_digest，不能用 ==：后者在第一个不同字节处就返回，
    耗时里含「猜对了几位」的信息，足以支撑逐字节爆破。行为上看不出，故用替身记录。
    """
    calls = []
    real = hmac.compare_digest

    def spy(left, right):
        calls.append((left, right))
        return real(left, right)

    monkeypatch.setattr(security.hmac, "compare_digest", spy)
    stored = security.hash_password("pw-123456")
    assert security.verify_password("pw-123456", stored) is True
    assert len(calls) == 1, "比对没有走 hmac.compare_digest"
    # 右侧必须是库里那一段：比错对象（比如拿盐跟自己比）会恒真
    assert calls[0][1] == base64.b64decode(stored.split("$")[5])


def test_empty_password_cannot_be_hashed():
    """空口令不该能进库：它算得出合法哈希，落到库里就是一个谁都能登录的账号。"""
    with pytest.raises(ValueError):
        security.hash_password("")


def test_plaintext_never_appears_in_the_stored_string():
    """最直白的泄露检查：明文与它常见的两种编码形态都不得出现在串里。"""
    stored = security.hash_password("pw-123456")
    assert "pw-123456" not in stored
    assert base64.b64encode(b"pw-123456").decode() not in stored
    assert b"pw-123456".hex() not in stored


def test_the_recorded_salt_is_really_one_of_the_hash_inputs():
    """盐必须真的喂给了 scrypt，而不是只写在串里当装饰。

    按串里记的盐与参数独立算一遍摘要，必须与串里那段逐字节相同。上一类用例只能证明
    「两次哈希的盐不同」，挡不住「盐不参与计算」这种实现（那时摘要只由口令决定，
    换个盐仍是同一摘要，而串看起来一切正常）。
    """
    stored = security.hash_password("same-pw")
    salt = base64.b64decode(stored.split("$")[4])
    expected = hashlib.scrypt(b"same-pw", salt=salt, n=security.SCRYPT_N,
                              r=security.SCRYPT_R, p=security.SCRYPT_P,
                              dklen=security.SCRYPT_DKLEN,
                              maxmem=security._SCRYPT_MAXMEM)
    assert base64.b64decode(stored.split("$")[5]) == expected


# ---- 密钥装载 ----

def test_load_secret_reads_the_environment():
    assert security.load_secret({security.JWT_SECRET_ENV: SECRET}) == SECRET


@pytest.mark.parametrize("value", [None, "", "   ", "short", "x" * 31])
def test_load_secret_rejects_missing_or_short(value):
    """空串算未设置（config._text 的既有判法）；过短等于没有签名，也拒。"""
    env = {} if value is None else {security.JWT_SECRET_ENV: value}
    with pytest.raises(MissingSecretError):
        security.load_secret(env)


def test_missing_secret_error_does_not_leak_the_value():
    """报错只提变量名与长度，**不回显取值** —— 密钥进日志等于泄露。"""
    bad = "s3cr3t-too-short"
    with pytest.raises(MissingSecretError) as exc:
        security.load_secret({security.JWT_SECRET_ENV: bad})
    assert bad not in str(exc.value)


def test_missing_secret_is_not_an_auth_error():
    """配置缺失绝不能被接口层「未认证 → 404」那条路吞掉。

    若它是 AuthError 的子类，一台没配密钥的服务器会把所有鉴权请求回成 404，运维顺着
    鉴权查下去，而真正的问题在环境变量。与 MissingAPIKeyError 同族（RuntimeError）。
    """
    assert not issubclass(MissingSecretError, AuthError)
    assert issubclass(MissingSecretError, RuntimeError)


