# -*- coding: utf-8 -*-
"""RAG2 在线服务启动入口（监督循环：支持网页后台改端口/地址后一键自动重启）。

用法：
    python run.py
或（等价，但不支持自重启）：
    uvicorn rag2.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import uvicorn

from rag2 import server as server_mod
from rag2.config import load_config
from rag2.logging_config import get_logger, setup_logging


def main() -> None:
    logger = get_logger("run")
    # 监督循环：uvicorn 退出后检查是否要「带新配置重启」，
    # 实现管理员在网页后台改端口/地址后一键自动重启。
    while True:
        cfg = load_config()
        setup_logging(level=cfg.app.log_level, log_dir=cfg.app.log_dir)
        logger.info("启动 %s：http://%s:%d", cfg.app.name, cfg.app.host, cfg.app.port)

        server = uvicorn.Server(
            uvicorn.Config(
                server_mod.app,
                host=cfg.app.host,
                port=cfg.app.port,
                reload=False,
                log_level=cfg.app.log_level.lower(),
            )
        )
        server_mod.bind_server(server)
        server.run()

        if not server_mod.consume_restart():
            break
        logger.info("检测到重启请求，正在以新配置重启…")


if __name__ == "__main__":
    main()
