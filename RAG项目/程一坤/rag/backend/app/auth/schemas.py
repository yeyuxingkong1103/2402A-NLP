"""认证请求与响应模型。"""

from pydantic import BaseModel, Field, field_validator


# 邮箱域名白名单（唯一来源）
ALLOWED_DOMAINS = {"qq.com", "foxmail.com"}


def validate_email_domain(value: str) -> str:
    """邮箱域名白名单校验：归一化（去空白 + 小写）后校验域名。

    归一化是必须的：用户在手机端容易带上首尾空格、把域名写成大写，
    若不归一化会在后续登录比对时因为大小写不一致而"密码明明对却登不上"。
    """
    normalized = value.strip().lower()
    if normalized.rsplit("@", 1)[-1] not in ALLOWED_DOMAINS:
        raise ValueError("仅支持 QQ 邮箱或 Foxmail 邮箱")
    return normalized


def _reject_disallowed_email(value: object) -> str:
    """三个请求模型共用的校验实现（模块级，可被直接测试）。"""
    return validate_email_domain(str(value))


# 共享校验器：定义一次、被三个模型引用同一对象。
# 此前 CodeRequest / RegisterRequest / LoginRequest 各写了一份逐字相同的
# allowed_email 函数体——三份逻辑会各自漂移（改一处漏两处），故收敛为一份。
# 注意：这里不能再套一层 classmethod，否则该代理对象被第二个类复用时
# 签名自省会失败（pydantic 2.13：Unrecognized field validator function signature）。
_allowed_email_validator = field_validator("email")(_reject_disallowed_email)


class CodeRequest(BaseModel):
    """发送邮箱验证码的请求。"""

    email: str

    # 复用同一校验器对象（行为与错误信息与收敛前完全一致）
    allowed_email = _allowed_email_validator


class RegisterRequest(BaseModel):
    """注册请求：邮箱 + 密码 + 6 位数字验证码。"""

    email: str
    password: str = Field(min_length=8, max_length=128)
    code: str = Field(min_length=6, max_length=6, pattern="^\\d{6}$")

    allowed_email = _allowed_email_validator


class LoginRequest(BaseModel):
    """登录请求：密码长度宽松（历史密码可能是短密码），不做复杂度校验。"""

    email: str
    password: str = Field(min_length=1, max_length=128)

    allowed_email = _allowed_email_validator


class ResetPasswordRequest(RegisterRequest):
    """重置密码请求：字段与注册一致（邮箱 + 新密码 + 验证码）。"""

    pass


class TokenResponse(BaseModel):
    """认证接口统一响应格式。"""

    # 业务错误码，0 表示成功
    code: int = 0
    # 响应消息
    message: str = "success"
    # 响应数据
    data: dict[str, str]
    # 请求追踪 ID
    request_id: str = ""
