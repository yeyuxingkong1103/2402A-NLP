"""统一错误体与异常处理器。

错误体结构严格对齐 `docs/05` §2.3：

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

⚠️ 为什么必须显式注册处理器，而不是依赖框架默认行为：

FastAPI/Starlette 在 `RequestValidationError` 与未捕获异常上的默认输出**不是**
这个结构（校验错是 `{"detail": [...]}`，未捕获异常是纯文本）。若不动它，就会出现
「多数错误符合契约、少数错误不符合」，而客户端只能对前者做解析 —— 这类不一致
在联调时表现为"偶尔解析失败"，很难定位。

同时这也保证错误体**不含堆栈、异常类名、文件路径、密钥**（FR-019、constitution
原则 III）：默认的 HTML 错误页会带调试信息，我们的处理器只输出用户可见文案。
"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.chat import ChatError

from . import (
    CHAT_PATH_PREFIX,
    ERR_CHAT_INVALID_REQUEST,
    ERR_CHAT_SESSION_NOT_FOUND,
    ERR_CHAT_UNAVAILABLE,
    ERR_INTERNAL,
    ERR_INVALID_QUESTION,
    ERR_NOT_FOUND,
    MSG_CHAT_INVALID_REQUEST,
    MSG_CHAT_SESSION_NOT_FOUND,
    MSG_CHAT_UNAVAILABLE,
    MSG_INTERNAL,
    MSG_INVALID_QUESTION,
    MSG_NOT_FOUND,
    REQUEST_ID_HEADER,
)

logger = logging.getLogger(__name__)


class InvalidQuestionError(Exception):
    """问题不合法。由 `validate.py` 抛出，经处理器转为 422。

    刻意**不继承** `ValueError`：本项目已有一个 `IngestError` 因继承内建名而
    踩坑（见 `backend/index/__init__.py` 的注释）。用独立类型可以让
    `except ValueError` 不会意外捕获它。
    """

    def __init__(self, message: str = MSG_INVALID_QUESTION) -> None:
        super().__init__(message)
        self.message = message


def request_id_of(request: Request) -> str:
    """取请求标识：优先用客户端传入的 `X-Request-Id`，否则生成。

    客户端传入时**不做格式校验** —— 它的唯一用途是日志关联，客户端的
    request id 长什么样由客户端决定。但会被截断以防御超长头。
    """

    supplied = request.headers.get(REQUEST_ID_HEADER)
    if supplied:
        return supplied[:64]
    return str(uuid.uuid4())


def error_response(
    code: str, message: str, status_code: int, request_id: str
) -> JSONResponse:
    """构造统一错误体。"""

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """注册全部异常处理器。MUST 在挂载静态资源之前调用。"""

    @app.exception_handler(InvalidQuestionError)
    async def _invalid_question(
        request: Request, exc: InvalidQuestionError
    ) -> JSONResponse:
        request_id = request_id_of(request)
        logger.info(
            "问题被拒绝 request_id=%s code=%s", request_id, ERR_INVALID_QUESTION
        )
        return error_response(ERR_INVALID_QUESTION, exc.message, 422, request_id)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """请求体校验失败（字段缺失、类型不符）。**按路径分派错误码。**

        ---

        ⚠️ **这处分派是修一个既有的真实缺陷**（specs/010 的 R11）。

        原先的实现把所有校验失败统一映射为 `ERR_INVALID_QUESTION` /
        `"请输入你的问题"`，其 docstring 写着前提：「本服务只有 `POST /ask`
        一个 API 入口」。

        **那个前提在 `corpus_routes.py` 接入时就已不成立** —— 只是当时
        没有暴露。chat 接入后，一个 `{"session_id": 123}`（类型错）会返回
        "请输入你的问题"，用户完全无从判断该改哪里。曝光面从一个端点变成五个。

        ⚠️ **MUST NOT 泛化成 `INVALID_REQUEST`** —— 那会改动 `/ask` 的既有
        对外契约（SC-008 要求它逐字不变）。按路径分派是这个约束下唯一的做法：
        在路由内 catch 是不可行的，Pydantic 校验发生在进入函数体**之前**。

        **错误详情不外泄**：`exc.errors()` 含字段路径与内部类型名，
        只进日志，不进响应体。
        """

        request_id = request_id_of(request)
        logger.info("请求体校验失败 request_id=%s detail=%s", request_id, exc.errors())

        if request.url.path.startswith(CHAT_PATH_PREFIX):
            return error_response(
                ERR_CHAT_INVALID_REQUEST, MSG_CHAT_INVALID_REQUEST, 422, request_id
            )

        return error_response(
            ERR_INVALID_QUESTION, MSG_INVALID_QUESTION, 422, request_id
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """框架自身抛出的 HTTP 异常，统一成我们的错误体形状。

        ⚠️ 这个处理器是**实测才发现需要**的：

        请求体不是合法 JSON 时，FastAPI 的请求解析层抛的是 `HTTPException(400)`，
        **不是** `RequestValidationError` —— 它绕过了上面那个处理器，直接落到
        Starlette 默认输出 `{"detail": "There was an error parsing the body"}`。

        于是同一个接口出现了两种错误形状：多数符合契约、少数不符合。而客户端
        只能对前者做解析，后者会表现为"偶尔解析失败"，极难定位。

        映射规则：
          400 / 422  → 客户端发来的东西读不成一个问题 → INVALID_QUESTION
          404        → 路径不存在（多为静态资源），见 ERR_NOT_FOUND 的说明
          其它       → INTERNAL_ERROR
        """

        request_id = request_id_of(request)

        if exc.status_code in (400, 422):
            logger.info(
                "请求无法解析 request_id=%s status=%s", request_id, exc.status_code
            )
            # ⚠️ 与 `_validation_error` 同一条分派 —— 走 chat 路径的请求
            # （例如 `?limit=0`，它在路由里抛 `HTTPException(422)`）
            # 不该收到"请输入你的问题"。
            if request.url.path.startswith(CHAT_PATH_PREFIX):
                return error_response(
                    ERR_CHAT_INVALID_REQUEST, MSG_CHAT_INVALID_REQUEST, 422, request_id
                )
            return error_response(
                ERR_INVALID_QUESTION, MSG_INVALID_QUESTION, 422, request_id
            )

        if exc.status_code == 404:
            return error_response(ERR_NOT_FOUND, MSG_NOT_FOUND, 404, request_id)

        logger.warning(
            "HTTP 异常 request_id=%s status=%s", request_id, exc.status_code
        )
        return error_response(
            ERR_INTERNAL, MSG_INTERNAL, exc.status_code, request_id
        )

    @app.exception_handler(ChatError)
    async def _chat_error(request: Request, exc: ChatError) -> JSONResponse:
        """会话层失败。**按 `session_missing` 分派，而不是匹配错误文案。**

            会话不存在 / 已过期 → 404 CHAT_SESSION_NOT_FOUND
            其余（Redis 不可达、记录损坏） → 503 CHAT_UNAVAILABLE

        ⚠️ 这两者对客户端的含义完全不同：404 是"重新开一个会话"，
        503 是"稍后重试"。混在一起会让用户对着一个其实是故障的问题
        反复重开会话。

        ⚠️ **不回显异常原文** —— 它可能含 session_id 或内部细节，
        只进日志（constitution 原则 III：日志脱敏）。
        """

        request_id = request_id_of(request)
        if exc.session_missing:
            logger.info(
                "会话不存在 request_id=%s code=%s", request_id, ERR_CHAT_SESSION_NOT_FOUND
            )
            return error_response(
                ERR_CHAT_SESSION_NOT_FOUND, MSG_CHAT_SESSION_NOT_FOUND, 404, request_id
            )

        logger.error(
            "会话服务失败 request_id=%s reason=%s", request_id, exc.message.replace("\n", " ")
        )
        return error_response(
            ERR_CHAT_UNAVAILABLE, MSG_CHAT_UNAVAILABLE, 503, request_id
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """未预期异常。

        对外只说"服务暂时不可用"，**真实异常只进服务端日志** —— 供应商原始
        错误体、堆栈、文件路径都不得出现在响应里（docs/05 §2.3 硬性要求 1）。
        """

        request_id = request_id_of(request)
        logger.exception("未处理异常 request_id=%s", request_id)
        return error_response(ERR_INTERNAL, MSG_INTERNAL, 500, request_id)
