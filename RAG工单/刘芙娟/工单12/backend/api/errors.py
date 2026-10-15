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

from . import (
    ERR_INTERNAL,
    ERR_INVALID_QUESTION,
    ERR_NOT_FOUND,
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
        """请求体校验失败（字段缺失、类型不符）。

        统一映射为 `INVALID_QUESTION` —— 本服务只有 `POST /ask` 一个 API 入口，
        请求体校验失败必然意味着问题本身不合法。**错误详情不外泄**：`exc.errors()`
        含字段路径与内部类型名，只进日志，不进响应体。
        """

        request_id = request_id_of(request)
        logger.info("请求体校验失败 request_id=%s detail=%s", request_id, exc.errors())
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

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """未预期异常。

        对外只说"服务暂时不可用"，**真实异常只进服务端日志** —— 供应商原始
        错误体、堆栈、文件路径都不得出现在响应里（docs/05 §2.3 硬性要求 1）。
        """

        request_id = request_id_of(request)
        logger.exception("未处理异常 request_id=%s", request_id)
        return error_response(ERR_INTERNAL, MSG_INTERNAL, 500, request_id)
