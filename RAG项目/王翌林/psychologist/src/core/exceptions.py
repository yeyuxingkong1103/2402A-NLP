"""业务异常与统一响应封装。

设计要点：把"业务错误码"和"HTTP 状态码"拆成两个独立字段（code / http_status）。
- 业务码用于前端做逻辑分支（例如 401 触发刷新 token），HTTP 码用于网关/浏览器语义。
- 二者数值当前一致只是为了少踩坑；将来若业务码扩展（如 40001），HTTP 码仍可保持 400。
"""
from typing import Any, Optional


class AppError(Exception):
    """业务异常基类。

    为什么要有这个基类？上层（FastAPI 的全局异常处理器）只需 `except AppError`
    就能统一把异常翻译成 {code, message, data} 的 JSON 响应，不必为每种错误写分支。
    """

    # 类属性充当"默认值"，子类通过覆盖它来表达各自的语义（见下方各子类）。
    code = 400
    http_status = 400

    def __init__(self, message: str, code: Optional[int] = None, http_status: Optional[int] = None):
        # 先调用父类构造，保证 str(exc) 与日志打印行为正常。
        super().__init__(message)
        self.message = message
        # 注意这里是"实例属性覆盖类属性"：默认 None 表示沿用子类声明的类级默认值，
        # 只有显式传入时才允许在运行时临时改写（例如把某个 NotFound 改成 410）。
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status


class AuthError(AppError):
    """未认证：token 缺失/失效/被吊销。前端据此跳登录或走刷新流程。"""

    code = 401
    http_status = 401


class PermissionError_(AppError):
    """已登录但权限不足（角色/资源归属不匹配）。

    类名带下划线后缀是因为 PermissionError 是 Python 内置异常名，
    直接同名会让使用者混淆（到底 catch 哪个），故显式加 "_" 区分。
    """

    code = 403
    http_status = 403


class NotFoundError(AppError):
    """资源不存在。对心理陪伴场景，用它区分"没有该会话/文档"与"服务器出错"。"""

    code = 404
    http_status = 404


class ConflictError(AppError):
    """冲突：如用户名已注册、并发写同一资源。"""

    code = 409
    http_status = 409


class RateLimitError(AppError):
    """触发限流（按用户或 IP）。单独成类便于在限流中间件里精准抛出。"""

    code = 429
    http_status = 429


class ExternalServiceError(AppError):
    """下游依赖失败（LLM / Milvus / OCR 服务不可用）。

    固定 502 而非 500：告诉调用方"错不在本服务逻辑，而在上游依赖"，
    便于运维区分是业务 bug 还是第三方抖动。
    """

    code = 502
    http_status = 502


def ok(data: Any = None, message: str = "success") -> dict:
    """成功响应统一信封。

    为什么所有接口都套一层 {code, message, data}？前端只需一套解析逻辑，
    且成功/失败结构对称，HTTP 层偶发的 200 也能携带业务错误码。
    code=0 约定表示成功（不用 200，避免与 HTTP 状态码概念混淆）。
    """
    return {"code": 0, "message": message, "data": data}


def fail(code: int, message: str, data: Any = None) -> dict:
    """失败响应统一信封：与 ok() 结构保持一致，只是 code 非 0。"""
    return {"code": code, "message": message, "data": data}