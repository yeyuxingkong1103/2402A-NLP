"""鉴权相关 Schema（注册 / 登录 / 刷新令牌 / 令牌与用户信息响应）。

本模块定义“认证域”的请求体与响应体，是接口层与业务层之间的数据契约：
- 请求模型（RegisterRequest 等）：校验客户端传入的数据，挡在业务逻辑之前；
- 响应模型（TokenResponse、UserInfo）：规定服务端返回给客户端的数据形状。
字段约束写在 Field(...) 上，非法请求会在进入路由函数前就抛出校验错误。
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
# ConfigDict 用于配置模型行为；field_validator 用于给单个字段挂自定义校验函数


class RegisterRequest(BaseModel):
    """注册请求体。"""
    # 用户名：3~64 字符。下限 3 是为了避免“a”“ab”这类无意义且易撞名的账号；
    # 上限 64 与数据库 username 列的 varchar 长度保持一致，防止入库截断。
    username: str = Field(min_length=3, max_length=64)
    # 密码：6~128 字符。下限 6 是安全基线（太短极易被暴力破解）；
    # 上限 128 是为了防止超长输入拖慢哈希运算（bcrypt 等算法对超长输入是资源消耗攻击面）。
    password: str = Field(min_length=6, max_length=128)
    # 邮箱：可选，可不填；填了就要不超过 128 字符（对齐数据库列宽）。
    email: Optional[str] = Field(default=None, max_length=128)
    # 手机号：可选，以字符串而非数字存储，因为要保留前导 0 与 +86 等国家码。
    phone: Optional[str] = Field(default=None, max_length=32)
    # 昵称：可选，展示用，限制 64 字符避免超长文本破坏前端布局。
    nickname: Optional[str] = Field(default=None, max_length=64)

    @field_validator("email")
    @classmethod
    def check_email(cls, v: Optional[str]) -> Optional[str]:
        """邮箱格式校验：只做“是否含 @”的轻量检查。

        这里刻意不做完整正则校验——邮箱正则很难写全，
        真正的可达性验证应交给“发送验证邮件”流程；
        此处的目的是拦掉明显手误（如漏写 @）的输入。
        """
        # v 为 None 表示用户没填邮箱，属于合法情况，直接放行
        if v and "@" not in v:
            raise ValueError("邮箱格式不正确")
        return v


class LoginRequest(BaseModel):
    """登录请求体。

    这里不像注册那样限制长度：登录阶段长度约束已无意义，
    统一由“用户名/密码错误”这一笼统提示返回，避免泄露账号是否存在。
    """
    username: str
    password: str


class RefreshRequest(BaseModel):
    """刷新令牌请求体：用长期有效的 refresh_token 换取新的 access_token。"""
    refresh_token: str


class TokenResponse(BaseModel):
    """登录/刷新成功后返回的令牌响应体。"""
    # 访问令牌：短期有效，用于调用受保护接口
    access_token: str
    # 刷新令牌：长期有效，仅用于换新 access_token，权限面更小
    refresh_token: str
    # 令牌类型，固定为 bearer（HTTP Authorization: Bearer <token> 约定）
    token_type: str = "bearer"
    # access_token 的剩余有效期（秒），前端据此决定何时静默刷新
    expires_in: int
    # 用户主键，方便前端直接落库/展示，而不必再解码 JWT
    user_id: int
    # 用户名，用于界面展示
    username: str
    # 角色列表（如 admin/user），前端据此控制菜单与按钮的可见性
    roles: List[str] = []


class UserInfo(BaseModel):
    """当前登录用户的资料视图（不含任何敏感字段，如密码哈希）。"""
    # from_attributes=True：允许用 ORM 对象直接构造本模型（按属性名取值），
    # 这样路由里可以直接 `UserInfo.model_validate(user_orm)`，无需手写 dict 转换
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    nickname: Optional[str] = None
    avatar: Optional[str] = None
    # 账号状态：1=正常，0=禁用。默认 1 表示新用户默认可登录
    status: int = 1
    # 角色列表，默认空列表表示普通用户
    roles: List[str] = []