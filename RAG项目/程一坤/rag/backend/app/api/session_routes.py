"""会话管理 REST 接口：创建 / 本人列表 / 历史消息（任务 5.4）。

职责边界：
- 本模块只做 HTTP 编排：参数校验（Pydantic）、认证（CurrentUser 依赖）、
  调用存储层、把能力层异常翻译为接口语义
- 存储逻辑在 chat_store.py（写）与 chat_history.py（读），本模块不写 SQL
- 归属失败统一 404（错误码 40001）：不区分"会话不存在"与"存在但不属于你"，
  避免攻击者通过状态码差异探测他人会话 ID 是否有效
- 未登录 / 令牌无效由 CurrentUser 依赖统一抛 401（错误码 40100）
"""

# 导入日志（关键操作留痕）
import logging
# 导入 UUID 生成器（服务端生成对外会话 ID）
import uuid

# 导入 FastAPI 路由、请求对象与查询参数（7.5/7.6 分页用）
from fastapi import APIRouter, Query, Request
# 导入 Pydantic 请求模型基类与字段约束
from pydantic import BaseModel, ConfigDict, Field

# 导入认证依赖（未登录 → 401；user_id 一律以认证上下文为准）
from app.auth.current_user import CurrentUser
# 导入写入侧存储的归属异常（API 层捕获后统一转 404）
from app.chat.chat_store import SessionAccessDenied
# 导入标题清洗规则（显式标题与自动标题走同一套规则）
from app.chat.chat_title import derive_title
# 导入持久化编排层（短期记忆懒建 + 写入/读取存储实例的懒建入口）
from app.api.chat_persistence import (
    get_chat_history,
    get_chat_store,
    get_short_term_memory,
)
# 导入统一错误构造器（SessionAccessDenied → 404）
from app.errors import not_found_error

# 模块级日志器（额外字段 request_id 由调用方注入）
logger = logging.getLogger(__name__)
# 会话管理路由：三个端点都挂在前缀 /api/v1/sessions 下
router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    """创建会话的请求体（接口文档 7.1）；两个字段都可省略。

    约束：
    - extra="ignore"：未知字段忽略，保证接口向后兼容
    - character_id 首期固定 legal-assistant，但仍允许显式传入
    - title 传空/不传时由服务端用"新会话 + 时间"兜底，绝不允许空标题
    """

    # 忽略请求体里的未知字段
    model_config = ConfigDict(extra="ignore")

    # 助手角色 ID（默认法律助手）
    character_id: str = Field(default="legal-assistant", min_length=1, max_length=64)
    # 显式标题（可省略；服务端仍会做清洗，长度上限与库字段一致）
    title: str | None = Field(default=None, max_length=256)


@router.post("", status_code=201)
def create_session(
    payload: CreateSessionRequest,
    current_user: CurrentUser,
    request: Request,
) -> dict:
    """创建会话（接口文档 7.1），返回服务端生成的会话 ID。

    归属规则：user_id 一律取认证上下文，不接受请求体指定；
    title 走统一清洗规则（去首尾空白/压连续空白/超长截 20 字加"…"/空则兜底）。

    参数：
    - payload: 请求体（character_id 与 title 均可省略）
    - current_user: 认证上下文（未登录时依赖层直接抛 401）
    - request: FastAPI 请求对象（取 request_id 与存储实例）

    返回：
    - dict: 通用响应结构，data 含 session_id / character_id / title / created_at
    """
    # 从中间件取请求追踪 ID
    request_id = request.state.request_id
    # 对外会话 ID 由服务端生成（客户端在 /chat/stream 里原样回传该值）
    session_key = f"session_{uuid.uuid4().hex[:12]}"
    # 显式标题与自动建档标题走同一套清洗规则，避免脏标题入库
    title = derive_title(payload.title)
    # 建档：不存在则创建；同一用户重复调用只会新建多个不同 ID 的会话
    record = get_chat_store(request).get_or_create_session(
        user_id=current_user.user_id,
        session_key=session_key,
        character_id=payload.character_id,
        title=title,
    )
    # 建档留痕（不记录标题正文，避免日志泄露用户输入）
    logger.info(
        "会话已创建 session_key=%s user_id=%s",
        record.session_key,
        current_user.user_id,
        extra={"request_id": request_id},
    )
    # 201 表示资源创建成功；data 使用对外字段名 session_id
    return {
        "code": 0,
        "message": "success",
        "data": {
            "session_id": record.session_key,
            "character_id": record.character_id,
            "title": record.title,
            "created_at": record.created_at,
        },
        "request_id": request_id,
    }


@router.get("")
def list_sessions(
    current_user: CurrentUser,
    request: Request,
    page: int = Query(default=1, ge=1, description="页码，1 起始"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页条数，上限 100"),
) -> dict:
    """返回本人会话的分页列表（按最近活跃倒序，附带每个会话的消息条数）。

    参数：
    - current_user: 认证上下文（只返回该 user_id 的会话）
    - request: FastAPI 请求对象
    - page: 页码（1 起始；page=0 由 Query 校验直接 422）
    - page_size: 每页条数（1~100，防止全量拉取拖库）

    返回：
    - dict: 通用响应结构，data.items 为会话卡片列表（接口文档 7.5）
    """
    # 从中间件取请求追踪 ID
    request_id = request.state.request_id
    # 读取层 SQL 带 user_id 过滤，他人会话天然不可见；分页在 SQL 层完成
    sessions = get_chat_history(request).list_sessions(
        current_user.user_id, page=page, page_size=page_size
    )
    # 数据库内部字段叫 session_key，对外统一暴露为 session_id
    items = [
        {
            "session_id": item["session_key"],
            "character_id": item["character_id"],
            "title": item["title"],
            "message_count": item["message_count"],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
        }
        for item in sessions
    ]
    # 返回通用响应结构（外层键 items，与接口文档 7.5 契约一致）
    return {
        "code": 0,
        "message": "success",
        "data": {"items": items},
        "request_id": request_id,
    }


@router.get("/{session_id}/messages")
def list_session_messages(
    session_id: str,
    current_user: CurrentUser,
    request: Request,
    page: int = Query(default=1, ge=1, description="页码，1 起始"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页条数，上限 100"),
) -> dict:
    """返回本人指定会话的分页历史消息（时间正序，citations 已反序列化）。

    归属失败处理：会话不存在或不属于本人统一 404，不暴露会话存在性。

    参数：
    - session_id: 路径参数，对外会话 ID
    - current_user: 认证上下文
    - request: FastAPI 请求对象
    - page: 页码（1 起始；page=0 由 Query 校验直接 422）
    - page_size: 每页条数（1~100，防止全量拉取拖库）

    返回：
    - dict: 通用响应结构，data.items 为消息列表（接口文档 7.6；
      assistant 消息的 message_id 与 SSE message_start 一致）

    异常：
    - 404（错误码 40001）：会话不存在或不属于本人
    """
    # 从中间件取请求追踪 ID
    request_id = request.state.request_id
    try:
        # 读取层先做 SQL 层归属校验，再按时间正序取分页消息
        messages = get_chat_history(request).list_messages(
            current_user.user_id, session_id, page=page, page_size=page_size
        )
    except SessionAccessDenied:
        # 两种失败场景对外同一句话，统一转 404（错误码 40001）
        raise not_found_error("会话不存在") from None
    # 返回通用响应结构（外层键 items，与接口文档 7.6 契约一致）
    return {
        "code": 0,
        "message": "success",
        "data": {"items": messages},
        "request_id": request_id,
    }


@router.delete("/{session_id}")
def delete_session(
    session_id: str,
    current_user: CurrentUser,
    request: Request,
) -> dict:
    """删除本人会话：MySQL 会话与消息同事务删除，Redis 短期记忆尽力清理。

    归属失败处理与历史接口一致：不存在或不属于本人统一 404，不暴露存在性。

    参数：
    - session_id: 路径参数，对外会话 ID
    - current_user: 认证上下文（未登录时依赖层直接抛 401）
    - request: FastAPI 请求对象

    返回：
    - dict: 通用响应结构，data 含 session_id 与 deleted 状态

    异常：
    - 404（错误码 40001）：会话不存在或不属于本人
    """
    # 从中间件取请求追踪 ID
    request_id = request.state.request_id
    try:
        # 写入侧同一事务删除会话与全部消息（SQL 层归属过滤）
        get_chat_store(request).delete_session(current_user.user_id, session_id)
    except SessionAccessDenied:
        # 两种失败场景对外同一句话，统一转 404（错误码 40001）
        raise not_found_error("会话不存在") from None
    # Redis 短期记忆尽力清理：MySQL 是事实来源，清理失败不影响删除结果
    try:
        # 复用 chat.py 的懒建逻辑（测试可注入替身）；按 user_id + session_id 清理
        memory = get_short_term_memory(request)
        # 实例存在时清掉消息列表与摘要两个 key
        if memory is not None:
            memory.delete_session_memory(current_user.user_id, session_id)
    except Exception as error:
        # Redis 抖动只记警告：残留短期记忆 key 带 TTL 会自然过期
        logger.warning(
            "删除会话后清理短期记忆失败：%s session_id=%s",
            type(error).__name__,
            session_id,
            extra={"request_id": request_id},
        )
    # 删除留痕（不含用户输入内容）
    logger.info(
        "会话已删除 session_id=%s user_id=%s",
        session_id,
        current_user.user_id,
        extra={"request_id": request_id},
    )
    # 返回通用响应结构
    return {
        "code": 0,
        "message": "success",
        "data": {"session_id": session_id, "status": "deleted"},
        "request_id": request_id,
    }
