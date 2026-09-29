"""FastAPI process lifecycle."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .auth import _init_auth_db
from .dependencies import SERVER_PORT, _get_lan_ip


async def lifespan(_: FastAPI):  # 服务启动/关闭时的钩子函数
    _init_auth_db()  # 启动时初始化用户/会话/历史数据库表
    print(f"MedRAG 本机访问： http://127.0.0.1:{SERVER_PORT}")  # 打印本机访问地址
    print(f"MedRAG 局域网访问： http://{_get_lan_ip()}:{SERVER_PORT}")  # 打印局域网访问地址
    yield  # 服务运行期间挂起在这里
    print("MedRAG 服务已停止。")  # 服务关闭时打印提示
