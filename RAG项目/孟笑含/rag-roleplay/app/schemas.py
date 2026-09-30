# -*- coding: utf-8 -*-
"""接口请求 / 响应模型。"""
from datetime import datetime
# 解析：时间类型（消息响应）

from pydantic import BaseModel, ConfigDict, Field
# 解析：pydantic 模型组件（请求校验与响应序列化）


# 注册请求体（用户名/密码）
class RegisterIn(BaseModel):
    # 解析：注册请求
    username: str = Field(min_length=2, max_length=64)
    # 解析：用户名 2~64 字符（框架层校验，不合法返回 422）
    password: str = Field(min_length=6, max_length=64)
    # 解析：密码 6~64 字符


# 登录请求体
class LoginIn(BaseModel):
    # 解析：登录请求
    username: str
    # 解析：用户名
    password: str
    # 解析：密码


# 用户响应（id/用户名/token）
class UserOut(BaseModel):
    # 解析：用户响应
    id: int
    # 解析：用户 ID
    username: str
    # 解析：用户名
    token: str
    # 解析：登录 token（前端存 localStorage）


# 创建角色请求（模板可选）
class RoleCreate(BaseModel):
    # 解析：创建角色请求
    name: str = Field(min_length=1, max_length=64)
    # 解析：角色名
    category: str = Field(min_length=1, max_length=32)
    # 解析：分类
    persona: str = Field(min_length=1)
    # 解析：人设
    prompt_template: str | None = None
    # 解析：提示词模板（可选，缺省用默认模板）


# 更新角色请求（全部字段可选）
class RoleUpdate(BaseModel):
    # 解析：更新角色请求（部分更新语义）
    name: str | None = Field(default=None, min_length=1, max_length=64)
    # 解析：角色名（可选）
    category: str | None = Field(default=None, min_length=1, max_length=32)
    # 解析：分类（可选）
    persona: str | None = Field(default=None, min_length=1)
    # 解析：人设（可选）
    prompt_template: str | None = None
    # 解析：模板（可选）


# 角色响应
class RoleOut(BaseModel):
    # 解析：角色响应
    model_config = ConfigDict(from_attributes=True)
    # 解析：允许从 ORM 对象直接构造（属性映射）

    id: int
    # 解析：角色 ID
    name: str
    # 解析：角色名
    category: str
    # 解析：分类
    persona: str
    # 解析：人设
    prompt_template: str
    # 解析：提示词模板


# 对话请求（use_rag 请求级 RAG 开关）
class ChatIn(BaseModel):
    # 解析：对话请求
    role_id: int
    # 解析：角色 ID
    content: str = Field(min_length=1)
    # 解析：用户输入（非空）
    use_rag: bool = True
    # 解析：请求级 RAG 开关（默认开，闲聊可关）


# 对话响应（reply + sources 引用 + warnings 校验提醒）
class ChatOut(BaseModel):
    # 解析：对话响应
    role_id: int
    # 解析：角色 ID
    reply: str
    # 解析：角色回复
    sources: list[str] = []
    # 解析：引用来源（知识块原文，评测与前端展示用）
    warnings: list[str] = []  # 后处理校验：回答中无知识依据的数值提醒
    # 解析：校验提醒（回答数值无知识依据时生成）


# 历史消息响应
class MessageOut(BaseModel):
    # 解析：历史消息响应
    model_config = ConfigDict(from_attributes=True)
    # 解析：允许从 ORM 对象构造

    id: int
    # 解析：消息 ID
    sender: str
    # 解析：发送方（user/assistant）
    content: str
    # 解析：内容
    created_at: datetime
    # 解析：发送时间
