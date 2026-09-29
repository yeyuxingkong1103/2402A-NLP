"""Pydantic request models used by the API routes."""
from pydantic import BaseModel


class ChatRequest(BaseModel):  # 健康问答请求体
    question: str = ""  # 用户问题文本
    session_id: str = ""  # 浏览器会话 ID（记忆上下文用）
    conversation_id: int | None = None  # 历史记录会话 ID；为空时后端自动创建
    web_search: bool = False  # 是否由本次请求主动启用 Tavily 联网搜索


class ImageRequest(BaseModel):  # 图片识别请求体
    image: str = ""  # base64 编码的图片内容（不含 data: 前缀）
    filename: str = "photo.jpg"  # 图片文件名（用于判断格式）
    question: str = ""  # 用户附加的问题
    conversation_id: int | None = None  # 历史会话 ID


class RegisterRequest(BaseModel):  # 注册请求体
    username: str = ""  # 账号名
    password: str = ""  # 密码
    display_name: str = ""  # 显示名（可选）


class LoginRequest(BaseModel):  # 登录请求体
    username: str = ""  # 账号名
    password: str = ""  # 密码
