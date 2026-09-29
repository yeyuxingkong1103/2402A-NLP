# 导入 FastAPI 的 HTTP 异常类，用于定义业务异常
from fastapi import HTTPException

# 导入 HTTP 状态码常量
from starlette import status


# 定义统一的业务错误码常量
class ErrorCode:
    """统一错误码定义，与接口文档保持一致。"""

    # 请求参数错误
    BAD_REQUEST = 40000
    # 资源不存在
    NOT_FOUND = 40001
    # 文件格式不支持
    UNSUPPORTED_FILE_FORMAT = 40002
    # 文件大小超限
    FILE_TOO_LARGE = 40003
    # 索引任务失败
    INDEX_FAILED = 40004
    # 未认证或 Token 无效
    UNAUTHORIZED = 40100
    # 无权限
    FORBIDDEN = 40300
    # 路径不存在
    PATH_NOT_FOUND = 40400
    # 资源冲突或重复操作
    CONFLICT = 40900
    # 请求频率超限
    RATE_LIMIT = 42900
    # 系统内部错误
    INTERNAL_ERROR = 50000
    # 模型服务不可用
    MODEL_UNAVAILABLE = 50001
    # 向量数据库不可用
    VECTOR_DB_UNAVAILABLE = 50002
    # 缓存或记忆服务不可用
    CACHE_UNAVAILABLE = 50003


# 定义业务异常基类，携带错误码和消息
class BusinessException(HTTPException):
    """业务异常基类，自动映射错误码到 HTTP 状态码。"""

    def __init__(self, code: int, message: str, http_status: int | None = None) -> None:
        # 如果未指定 HTTP 状态码，根据错误码自动推断
        if http_status is None:
            http_status = self._infer_http_status(code)
        # 调用父类初始化，设置 HTTP 状态码和详细信息
        super().__init__(status_code=http_status, detail={"code": code, "message": message})
        # 保存业务错误码
        self.code = code
        # 保存错误消息
        self.message = message

    @staticmethod
    def _infer_http_status(code: int) -> int:
        """根据业务错误码推断 HTTP 状态码。"""
        # 特判：40001"资源不存在"按 REST 语义映射 404；
        # 不能落入下方 40000-40099 区间（那里统一按参数错误返回 400）
        if code == ErrorCode.NOT_FOUND:
            return status.HTTP_404_NOT_FOUND
        # 40000-40099：客户端请求错误
        if 40000 <= code < 40100:
            return status.HTTP_400_BAD_REQUEST
        # 40100-40199：未认证
        if 40100 <= code < 40200:
            return status.HTTP_401_UNAUTHORIZED
        # 40300-40399：无权限
        if 40300 <= code < 40400:
            return status.HTTP_403_FORBIDDEN
        # 40400-40499：资源不存在
        if 40400 <= code < 40500:
            return status.HTTP_404_NOT_FOUND
        # 40900-40999：资源冲突
        if 40900 <= code < 41000:
            return status.HTTP_409_CONFLICT
        # 42900-42999：请求频率超限
        if 42900 <= code < 43000:
            return status.HTTP_429_TOO_MANY_REQUESTS
        # 50000 及以上：服务端错误
        if code >= 50000:
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        # 默认返回 400
        return status.HTTP_400_BAD_REQUEST


# 定义常见业务异常的快捷构造函数
def bad_request_error(message: str) -> BusinessException:
    """请求参数错误。"""
    return BusinessException(ErrorCode.BAD_REQUEST, message)


def not_found_error(message: str) -> BusinessException:
    """资源不存在。"""
    return BusinessException(ErrorCode.NOT_FOUND, message)


def unauthorized_error(message: str) -> BusinessException:
    """未认证或 Token 无效。"""
    return BusinessException(ErrorCode.UNAUTHORIZED, message)


def forbidden_error(message: str) -> BusinessException:
    """无权限（管理员接口拒绝普通用户等）。"""
    return BusinessException(ErrorCode.FORBIDDEN, message)


def internal_error(message: str) -> BusinessException:
    """系统内部错误。"""
    return BusinessException(ErrorCode.INTERNAL_ERROR, message)
