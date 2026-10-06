"""src/schemas/common.py —— 统一响应结构的通用模型。

在链路中的位置：
    为全项目的响应提供统一形状的模板；src/api/main.py 里的全局异常处理器
    手工构造的就是 APIResponse/ErrorResponse 的同款结构。

统一结构是 README 里承诺的对外契约：
    {"code": 业务码, "message": 说明, "data": 数据, "trace_id": 链路 id}
    调用方写一套解析逻辑就能处理所有接口的返回，
    trace_id 让"用户报错"能直接对应到后端日志。

说明：当前实现里只有异常响应（main.py 的 error_handler）严格使用了这个结构，
大部分业务路由直接返回了裸字典。这几个模型的价值在于
"把约定的形状写下来、可作为 response_model 使用"，
是把现有返回逐步收敛到统一格式的落点。
"""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """成功响应的统一结构（泛型，data 的类型由使用处指定）。

    字段：
        code:      业务码，0 表示成功（注意与 HTTP 状态码是两套体系）
        message:   说明文字，默认 "ok"
        data:      业务数据，可为空
        trace_id:  请求链路 id，默认 "-" 表示未设置

    用 Generic[T] 而不是把 data 定成 Any：
        这样 `APIResponse[TokenResponse]` 能表达"data 是令牌对象"，
        IDE 与文档都能给出准确类型，而不是一个什么都可能的 Any。
    """

    code: int = 0
    message: str = "ok"
    data: T | None = None
    trace_id: str = "-"


class ErrorResponse(BaseModel):
    """错误响应的统一结构。

    字段：
        code:     错误码（与 HTTP 状态码同值，见 main.py 里返回 500 的写法）
        message:  错误说明
        data:     恒为 None，保留字段是为了让成功与失败的响应结构完全一致
        trace_id: 链路 id

    保留 data 字段（尽管错误时永远是 None）：
        结构一致意味着前端可以无条件地按同一套键取值，不必先判断成功还是失败。
        多一个恒为空的字段，换来调用方少一堆分支判断。
    """

    code: int
    message: str
    data: Any = None
    trace_id: str = "-"


class HealthResponse(BaseModel):
    """健康检查响应的结构。

    字段：
        status:   总体状态（如 ok / degraded）
        trace_id: 链路 id
        services: 各依赖服务的明细，如 {"milvus": {"ok": True, "detail": "connected"}}
    """

    status: str
    trace_id: str = "-"
    services: dict[str, Any] = Field(default_factory=dict)
