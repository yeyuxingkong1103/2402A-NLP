import uvicorn

from app.core.config import get_settings


if __name__ == "__main__":
    # 读取 .env 后启动 Uvicorn；app.main:app 表示 app/main.py 中的 FastAPI 实例。
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
