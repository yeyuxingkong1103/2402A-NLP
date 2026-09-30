"""用户管理 Schema（资料更新、修改密码、启停账号、用户列表、角色偏好）。

本模块服务两类场景：
- 用户自助：改自己的资料（UserUpdateRequest）、改自己的密码（PasswordChangeRequest）、
  设置常用心理医生角色（PreferenceRequest / PreferenceItem）；
- 管理端：查看用户列表（UserListItem）、启用/禁用账号（UserStatusRequest）。

安全提示：本模块所有响应模型都不包含 password / password_hash 字段，
密码只在请求模型中短暂出现，且绝不回传。
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class UserUpdateRequest(BaseModel):
    """用户资料更新请求体（部分更新：只改传入的字段）。"""
    # 昵称：可选，最大 64 字符（对齐数据库列宽，同时避免超长昵称撑坏 UI）
    nickname: Optional[str] = Field(default=None, max_length=64)
    # 头像：可选，最大 512 字符——存的是 URL 而非图片本身，故需要更宽的列
    avatar: Optional[str] = Field(default=None, max_length=512)
    # 邮箱：可选，最大 128 字符（与注册时的约束保持一致，避免同一字段两套规则）
    email: Optional[str] = Field(default=None, max_length=128)
    # 手机号：可选，最大 32 字符（字符串类型以保留前导 0 / 国家码）
    phone: Optional[str] = Field(default=None, max_length=32)
    # 默认角色 ID：用户进入应用时默认打开的心理医生，可选（不传表示不修改）
    default_persona_id: Optional[int] = None


class PasswordChangeRequest(BaseModel):
    """修改密码请求体：必须同时校验旧密码，防止会话被盗后被静默改密。"""
    # 旧密码：不给长度约束，因为它的校验方式是“与库中哈希比对”，不是格式检查
    old_password: str
    # 新密码：6~128 字符，与注册时的密码强度要求完全一致，
    # 否则用户可能通过“改密码”绕开注册密码的长度下限
    new_password: str = Field(min_length=6, max_length=128)


class UserStatusRequest(BaseModel):
    """启用/禁用账号请求体（管理端）。"""
    # Field(description=...) 只生成 OpenAPI 文档说明，不做强制校验；
    # 值为 1 表示启用、0 表示禁用（布尔语义用 int 表达，与数据库列类型一致）
    status: int = Field(description="1=启用，0=禁用")


class UserListItem(BaseModel):
    """用户列表中的单条记录（管理端用户管理页）。"""
    model_config = ConfigDict(from_attributes=True)  # 允许从 ORM 对象直接构造

    id: int
    username: str
    nickname: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    # 账号状态：这里没有默认值，因为它是必填展示项——
    # 管理端必须明确看到每个用户是启用还是禁用
    status: int
    # 注册时间（字符串格式，避免前后端时区解析差异）
    created_at: Optional[str] = None
    # 最后登录时间：可选，从未登录过则为 None，可用于识别僵尸账号
    last_login_at: Optional[str] = None
    # 角色列表，默认空列表；
    # 注意：即使列表模型也不外泄密码等敏感字段
    roles: List[str] = []


class PreferenceRequest(BaseModel):
    """设置“我常用的心理医生角色”的请求体。"""
    persona_id: int
    # 是否设为默认角色。默认 True：
    # 这个接口的典型调用场景就是“把某个角色设为默认”，默认值贴合主要用法
    is_default: bool = True


class PreferenceItem(BaseModel):
    """用户偏好（常用角色）列表中的单条记录。"""
    persona_id: int
    # 冗余的编码/名称字段：由后端关联角色表填充，
    # 让前端渲染偏好列表时无需再请求一次角色详情
    persona_code: Optional[str] = None
    persona_name: Optional[str] = None
    # 是否为当前默认角色。这里默认 False 表示“普通收藏项”，
    # 一条偏好记录只有在被显式设默认时才会是 True
    is_default: bool = False