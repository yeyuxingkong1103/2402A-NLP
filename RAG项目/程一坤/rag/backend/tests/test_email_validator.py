"""邮箱白名单校验：三份实现合一后的行为锁定测试（批次 20 / 任务书 2.3）。

锁两件事：
1. **行为**：CodeRequest / RegisterRequest / LoginRequest（及其子类 ResetPasswordRequest）
   对非法邮箱一律拒绝，且错误信息与历史一致（"仅支持 QQ 邮箱或 Foxmail 邮箱"）；
   合法邮箱则归一化（去首尾空白 + 转小写）后接受。
   另外锁定"非字符串输入"的错误类型（string_type）——这是字段类型校验先于
   字段校验器执行的既有行为，收敛实现后不得改变。
2. **结构**：四个模型引用的是**同一个校验器函数对象**，防止将来又退回"每个模型抄一份"。
"""
import pytest
from pydantic import ValidationError

from app.auth.schemas import (
    CodeRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    validate_email_domain,
)

# 三种请求模型（+ 继承 RegisterRequest 的重置密码请求）都要走同一套邮箱校验
REQUEST_MODELS = (CodeRequest, RegisterRequest, LoginRequest, ResetPasswordRequest)

# 各模型除 email 外的必填字段（保证校验能走到 email 之外的部分）
EXTRA_FIELDS = {"password": "demo#2026Admin", "code": "123456"}


def _build(model_cls, email):
    return model_cls(email=email, **EXTRA_FIELDS)


@pytest.mark.parametrize("model_cls", REQUEST_MODELS)
@pytest.mark.parametrize(
    "bad_email",
    ["x@gmail.com", "x@qq.com.cn", "x@foxmail.com.cn", "x@qq", "x@", "", "no-at-sign"],
)
def test_request_models_reject_disallowed_email(model_cls, bad_email) -> None:
    """非法邮箱：四个模型都必须拒绝，错误信息与历史一字不差。"""
    with pytest.raises(ValidationError) as excinfo:
        _build(model_cls, bad_email)
    error = excinfo.value.errors()[0]
    assert error["loc"] == ("email",)
    assert error["type"] == "value_error"
    assert error["msg"] == "Value error, 仅支持 QQ 邮箱或 Foxmail 邮箱"


def test_local_part_is_not_validated_legacy_behavior() -> None:
    """既存边界（本次不改，仅登记）：只校验域名、不校验本地部分，故 "@qq.com" 会被放行。

    批次 20 的任务边界是"三份实现合一、行为零变化"，所以这里**故意保留**该行为；
    若将来要补本地部分校验，改这一处即可——此测试就是当时的显式决策点。
    """
    assert CodeRequest(email="@qq.com").email == "@qq.com"


@pytest.mark.parametrize("model_cls", REQUEST_MODELS)
@pytest.mark.parametrize(
    ("raw_email", "normalized"),
    [
        ("foo@qq.com", "foo@qq.com"),
        ("  Foo@QQ.com  ", "foo@qq.com"),
        ("a@foxmail.com", "a@foxmail.com"),
        ("A@FOXMAIL.COM", "a@foxmail.com"),
    ],
)
def test_request_models_accept_and_normalize_allowed_email(
    model_cls, raw_email, normalized
) -> None:
    """合法邮箱：接受并归一化（去空白 + 小写）——否则登录时大小写不一致会登录失败。"""
    assert _build(model_cls, raw_email).email == normalized


@pytest.mark.parametrize("model_cls", REQUEST_MODELS)
def test_non_string_email_keeps_type_error(model_cls) -> None:
    """非字符串输入：仍报 string_type（字段类型校验先于字段校验器，行为不得漂移）。"""
    with pytest.raises(ValidationError) as excinfo:
        _build(model_cls, 123)
    error = excinfo.value.errors()[0]
    assert error["loc"] == ("email",)
    assert error["type"] == "string_type"


def test_shared_validator_function() -> None:
    """共享实现本身可直接调用（模块级函数，单一来源）。"""
    assert validate_email_domain("  Foo@QQ.com ") == "foo@qq.com"
    with pytest.raises(ValueError, match="仅支持 QQ 邮箱或 Foxmail 邮箱"):
        validate_email_domain("x@gmail.com")


def test_all_request_models_share_one_validator_object() -> None:
    """结构锁定：四个模型引用同一个校验器函数对象（不再出现三份相同函数体）。"""
    impls = {
        model_cls.__pydantic_decorators__.field_validators["allowed_email"].func
        for model_cls in REQUEST_MODELS
    }
    assert len(impls) == 1, "邮箱校验实现出现多份，说明又被复制成了多套逻辑"
