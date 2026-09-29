# app/schemas/__init__.py
"""接口出入参模型（Pydantic v2）。"""
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------- 通用 ----------------
class Resp(BaseModel):
    code: int = 200
    msg: str = "ok"
    data: Optional[Any] = None


# ---------------- 问答 ----------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000,
                          description="用户问题")
    role_key: str = Field(..., max_length=64, description="角色标识，如 lawyer")
    user_id: int = Field(1, ge=0, description="用户ID，未接入鉴权时默认 1")
    # 与 sessions.session_id 的列宽一致：不限制长度会让 MySQL 抛
    # Data too long，表现为 500 而不是参数错误
    session_id: Optional[str] = Field(None, max_length=64,
                                      description="会话ID，不传则新建")


# ---------------- 用户 ----------------
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6, max_length=64)
    nickname: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


# ---------------- 入库 / 更新 ----------------
class IngestDatasetRequest(BaseModel):
    role_key: str = Field(..., description="角色标识")
    force: bool = Field(False, description="True 则清空该角色旧数据后重建")


class UpdateDocumentRequest(BaseModel):
    file_path: str = Field(..., description="文档绝对路径")
    role_key: str = Field(..., description="所属角色")
    force: bool = Field(False, description="跳过哈希比对强制重建")


class DeleteDocumentRequest(BaseModel):
    source: str = Field(..., description="文件名，如 legal_articles.jsonl")
    role_key: str
