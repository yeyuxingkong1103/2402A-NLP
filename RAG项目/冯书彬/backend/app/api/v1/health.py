from fastapi import APIRouter, Response, status

from backend.app.core.config import settings
from backend.app.core.metrics import record_health_statuses
from backend.app.core.startup_checks import is_ready, run_startup_checks

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live_health() -> dict[str, str]:
    # 存活检查只证明应用进程可响应，不触发任何外部依赖检查。
    return {"status": "alive"}


@router.get("/ready")
def ready_health(response: Response) -> dict[str, object]:
    # 每次 ready 请求即时运行轻量启动检查，便于编排系统判断是否可接流量。
    statuses = run_startup_checks(settings)
    record_health_statuses(statuses)
    # 只有所有依赖都正常时才返回 ready=true。
    ready = is_ready(statuses)
    # 任一依赖缺失时使用 503，明确告知上游当前不可接流量。
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    # dataclass 转成脱敏字典，detail 只包含固定错误原因。
    dependencies = [status_item.__dict__ for status_item in statuses]
    return {"ready": ready, "dependencies": dependencies}
