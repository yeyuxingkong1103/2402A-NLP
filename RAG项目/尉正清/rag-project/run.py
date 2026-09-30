# run.py
"""服务启动入口。

    python run.py              # 默认 0.0.0.0:8000
    python run.py --reload     # 开发模式，改代码自动重启
    PORT=9000 python run.py    # 指定端口
"""
import os
import sys

import uvicorn

if __name__ == "__main__":
    reload = "--reload" in sys.argv
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print("=" * 58)
    print("  多角色 RAG 角色扮演系统")
    print("  接口文档: http://127.0.0.1:%d/docs" % port)
    print("=" * 58)

    if reload:
        uvicorn.run("app.main:app", host=host, port=port, reload=True)
    else:
        from app.main import app
        uvicorn.run(app, host=host, port=port)
