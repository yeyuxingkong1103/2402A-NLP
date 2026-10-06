"""通用响应模型（统一响应体、分页结果、健康检查结果）。

这三个模型是全项目最底层的“响应外壳”，几乎每个接口都会用到：
- ApiResponse[T]：统一业务响应格式（code/message/data）；
- PageResult[T]：统一分页格式（total/page/page_size/items）；
- HealthResult：服务健康检查结果。

它们都是泛型（Generic[T]），T 表示 data/items 里实际承载的业务类型。
运行时不会做跨字段约束，但静态类型检查与 IDE 能据此给出准确补全。
"""
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

# 类型变量：占位符，使用方在实例化泛型时传入具体类型（如 ApiResponse[UserInfo]）
T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应外壳：所有接口都返回这个结构，前端只需判断一次 code。"""
    # 业务状态码：0 表示成功；非 0 表示业务失败（HTTP 状态码仍可能是 200）
    code: int = 0
    # 提示信息：成功时固定为 "success"，失败时为给用户看的原因
    message: str = "success"
    # 业务数据载荷：失败时通常为 None，所以类型是 Optional[T]
    data: Optional[T] = None


class PageResult(BaseModel, Generic[T]):
    """分页结果外壳：列表类接口统一用它返回。"""
    # 总记录数（不是本页条数），前端据此算总页数
    total: int = 0
    # 当前页码，从 1 开始计数（而非从 0），符合用户直觉
    page: int = 1
    # 每页条数，默认 20，是“一屏可浏览”的折中值；过大会拖慢响应与渲染
    page_size: int = 20
    # 当前页的数据列表。用 default_factory=list 而不是 []，
    # 是为了避免可变默认值被所有实例共享（Python 经典陷阱）
    items: List[T] = Field(default_factory=list)


class HealthResult(BaseModel):
    """健康检查结果：逐项报告依赖组件是否可用，供监控/探活接口使用。"""
    # 总体状态字符串，如 "ok" / "degraded"，便于运维直接读取
    status: str
    # 三个外部依赖的连通性布尔值，前端或监控可逐项定位故障组件
    mysql: bool
    redis: bool
    milvus: bool
    # 附加诊断信息（如各组件耗时、错误原因）。dict[str, Any] 表示键是字符串、值任意
    detail: Optional[dict[str, Any]] = None