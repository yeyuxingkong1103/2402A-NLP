# -*- coding: utf-8 -*-
"""
用户登录接口（增量功能，配套「智查 AI」前端改造）

课程演示口径（经确认）：
    - 无密码、无加密、无验证码、无 token 过期逻辑
    - 输入账号即登录；账号不存在则自动创建
    - 账号持久化在 MySQL（users 表），非前端模拟登录
    - 仅本地局域网使用

接口：
    POST /api/auth/login    输入账号登录（不存在则自动注册）
    GET  /api/auth/me       按 user_id 回查用户
    POST /api/auth/logout   退出登录（前端清理本地登录态即可）

边界：
    本文件只读写 users 表，不触碰知识库、检索、生成链路。
"""

from __future__ import annotations

import base64
import re
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.db import mysql
from backend.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["用户"])

USERNAME_MAX = 32
AVATAR_MAX = 8                          # 预设头像标识（emoji）最长字符数
MAX_AVATAR_BYTES = 1024 * 1024          # 头像图片落盘上限（前端已压到 ~10KB，这里只兜底）
MAX_AVATAR_DATA_URL = 2 * 1024 * 1024   # 请求体里 data URL 的长度上限（约 1.5MB 图片）

# 支持从相册选择的图片格式（前端统一压成 jpeg，这里仍兼容 png / webp）
IMAGE_DATA_URL_PATTERN = re.compile(
    r"^data:image/(png|jpe?g|webp);base64,(.+)$", re.IGNORECASE | re.DOTALL
)

# 账号规则：中文、字母、数字、下划线、连字符、点。不含空格与其它标点，
# 避免出现「空格账号」「前后导出的空字符」这类现场说不清的数据。
USERNAME_PATTERN = re.compile(r"^[\w\u4e00-\u9fff.\-]+$", re.UNICODE)


class LoginRequest(BaseModel):
    """登录请求"""

    username: str = Field(
        ...,
        min_length=1,
        max_length=USERNAME_MAX,
        description="登录账号，1-32 字符；不存在则自动创建",
        examples=["张三"],
    )
    avatar: Optional[str] = Field(
        "",
        description="预设头像标识（emoji，如 🐱），可省略；为空时前端用账号首字兜底",
        examples=["🐱"],
    )
    avatar_image: Optional[str] = Field(
        "",
        max_length=MAX_AVATAR_DATA_URL,
        description=(
            "从相册选的自定义头像，data URL 形式（如 data:image/jpeg;base64,…）。"
            "传了它就以它为准：服务端把图片存到 data/avatars/，库里只记地址"
        ),
    )

    @field_validator("username", mode="before")
    @classmethod
    def _validate_username(cls, value: Any) -> Any:
        """校验账号：去空白、长度上限、字符合法性"""
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            raise ValueError("账号不能为空")
        if len(text) > USERNAME_MAX:
            raise ValueError(f"账号过长，请控制在 {USERNAME_MAX} 个字符以内")
        if not USERNAME_PATTERN.match(text):
            raise ValueError("账号只能包含中文、字母、数字、下划线、连字符和点")
        return text

    @field_validator("avatar", mode="before")
    @classmethod
    def _validate_avatar(cls, value: Any) -> Any:
        """头像标识：去空白、限长、禁止控制字符（前端提供固定选项，后端仍做兜底校验）"""
        if value is None or not isinstance(value, str):
            return ""
        text = value.strip()
        if len(text) > AVATAR_MAX:
            raise ValueError("头像标识过长")
        if any(ord(ch) < 32 for ch in text):
            raise ValueError("头像标识不合法")
        return text


class UserInfo(BaseModel):
    """用户信息"""

    user_id: str = Field(..., description="用户 ID，前端后续所有请求都带上它")
    username: str = Field(..., description="登录账号")
    display_name: str = Field("", description="界面展示名")
    avatar: str = Field("", description="自选头像标识（emoji）；为空时前端用账号首字兜底")
    created: bool = Field(False, description="是否本次新建账号（首次登录）")


def _save_avatar_image(user_id: str, data_url: str) -> str:
    """
    把前端传来的自定义头像（data URL）落盘，返回可访问地址。

    为什么落盘而不是把图片塞进数据库：
        头像是二进制静态资源，存文件、库里只记地址是更干净的做法 ——
        登录响应体不会被撑大，浏览器也能正常缓存图片。
    文件名直接用 user_id，天然一人一张；地址带 ?v=时间戳，
    保证换头像之后浏览器不会继续用缓存里的旧图。
    """
    matched = IMAGE_DATA_URL_PATTERN.match((data_url or "").strip())
    if not matched:
        raise ValueError("头像图片格式不支持（仅支持 png / jpg / webp）")

    kind = matched.group(1).lower()
    ext = "jpg" if kind in ("jpeg", "jpg") else kind
    try:
        raw = base64.b64decode(matched.group(2), validate=False)
    except Exception as exc:
        raise ValueError("头像图片内容无法解析") from exc
    if not raw:
        raise ValueError("头像图片内容为空")
    if len(raw) > MAX_AVATAR_BYTES:
        raise ValueError("头像图片过大")

    target_dir = settings.avatars_path
    target_dir.mkdir(parents=True, exist_ok=True)
    # 同一用户换头像时清掉旧文件（扩展名可能不同）
    for old_file in target_dir.glob("%s.*" % user_id):
        try:
            old_file.unlink()
        except Exception:
            pass

    file_path = target_dir / ("%s.%s" % (user_id, ext))
    file_path.write_bytes(raw)
    logger.info("自定义头像已保存 | 用户=%s | %d 字节 | %s", user_id, len(raw), file_path.name)
    return "/avatars/%s?v=%d" % (file_path.name, int(time.time()))


@router.post("/login", response_model=UserInfo, summary="登录（账号不存在则自动创建）")
def login(request: LoginRequest) -> UserInfo:
    """
    输入账号即登录，无需密码。

    账号不存在时自动创建：课程演示场景下最省事，演示前不必预置账号。
    账号一旦建立，其历史记录与收藏都持久化在 MySQL，换浏览器、清缓存都不受影响。
    """
    try:
        user = mysql.get_or_create_user(request.username, avatar=request.avatar or "")
        # 自定义头像（从相册选的图片）优先于 emoji 标识
        if request.avatar_image:
            try:
                saved = _save_avatar_image(user["user_id"], request.avatar_image)
                mysql.update_user_avatar(user["user_id"], saved)
                user["avatar"] = saved
            except ValueError as exc:
                # 头像不合法只影响头像本身，不阻断登录
                logger.warning("头像保存失败（不影响登录）| 用户=%s | %s", user["user_id"], exc)
    except Exception as exc:
        logger.exception("登录失败 | 账号=%r", request.username)
        raise HTTPException(
            status_code=500,
            detail="登录失败，请稍后重试（详细信息见服务端日志）",
        ) from exc

    display = user.get("display_name") or user["username"]
    logger.info(
        "用户登录 | 账号=%s | 用户ID=%s | 新建账号=%s | 头像=%s",
        user["username"], user["user_id"], bool(user.get("created")), user.get("avatar") or "(首字)",
    )
    return UserInfo(
        user_id=user["user_id"],
        username=user["username"],
        display_name=display,
        avatar=user.get("avatar") or "",
        created=bool(user.get("created")),
    )


@router.get("/me", response_model=UserInfo, summary="查询用户信息")
def me(user_id: str) -> UserInfo:
    """按 user_id 回查用户；前端刷新页面后用它校验本地登录态是否仍然有效"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")

    user = mysql.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在，请重新登录")

    return UserInfo(
        user_id=user["user_id"],
        username=user["username"],
        display_name=user.get("display_name") or user["username"],
        avatar=user.get("avatar") or "",
        created=False,
    )


@router.post("/logout", summary="退出登录")
def logout(user_id: str = "") -> Dict[str, Any]:
    """
    退出登录。

    本项目无 token / 无服务端会话凭据，退出动作由前端清理本地登录态完成；
    本接口只记录日志，便于演示时在服务端观察到「谁退出了」。
    """
    logger.info("用户退出 | 用户ID=%s", user_id or "(未提供)")
    return {"logged_out": True, "user_id": user_id}