from pydantic import BaseModel, Field


class OtpRequestResult(BaseModel):
    # sent=False 表示内部测试固定验证码未发送真实短信。
    sent: bool
    expires_in_seconds: int


class TokenPair(BaseModel):
    # TokenPair 是服务层和 API 层共享的返回结构。
    access_token: str
    refresh_token: str
    user_id: str
    token_type: str = "bearer"


class RequestCodeBody(BaseModel):
    # 仅接收手机号，不在日志和响应中回显。
    phone: str = Field(min_length=11, max_length=20)
    client_id: str = "browser"


class LoginBody(BaseModel):
    # 验证码只用于本次校验，不落库明文。
    phone: str = Field(min_length=11, max_length=20)
    code: str = Field(min_length=4, max_length=8)
    client_id: str = "browser"


class RefreshBody(BaseModel):
    # Refresh Token 明文只从请求进入服务层，服务端只存摘要。
    refresh_token: str


class LogoutBody(BaseModel):
    # 登出通过 refresh token 定位并吊销当前设备会话。
    refresh_token: str
