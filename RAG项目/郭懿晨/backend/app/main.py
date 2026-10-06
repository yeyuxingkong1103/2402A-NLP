import platform as _platform
from pathlib import Path
import sys

# Windows WMI 服务假死时，platform._wmi_query 会永久阻塞；torch 导入依赖
# platform.machine()（经 uname -> win32_ver -> _wmi_query），导致进程卡死。
# 此补丁让所有 platform WMI 查询立即抛 OSError，令其回退到
# sys.getwindowsversion / PROCESSOR_* 环境变量路径（见 platform.py 的 except OSError 分支）。
def _disable_wmi(*_args, **_kwargs):
    raise OSError("WMI disabled: Windows Management Instrumentation service is wedged")

_platform._wmi_query = _disable_wmi

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import AppSettings
from backend.app.routes_chat import router as chat_router
from backend.app.routes_documents import router as documents_router
from backend.app.routes_files import router as files_router
from backend.app.routes_tasks import router as tasks_router

app = FastAPI(title="RAG PDF 问答系统")
app.include_router(files_router)
app.include_router(tasks_router)
app.include_router(documents_router)
app.include_router(chat_router)

project_root = Path(__file__).resolve().parents[2]
frontend_dir = project_root / "frontend"
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir)), name="assets")


def main() -> None:
    """启动本地 API 服务。"""
    settings = AppSettings()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port, reload=False)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    main()
