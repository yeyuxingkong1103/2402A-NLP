# -*- coding: utf-8 -*-
"""scripts/ 下独立脚本共用的 .env 加载器与 MySQL 连接参数（零明文机密）。

为什么单独成模块：
scripts/ 下的运维脚本（migrations、e2e）都要连 MySQL，此前各自把 host / user /
password 硬编码在源码里（口令是明文），既不可移植也无法轮换口令，
且违反项目"配置只从 .env 读"的约定——同一份口令散在 5 个文件里，改一处漏一处。
集中到本模块后：
- 口令只存在于项目根 .env（不进版本库），脚本源码里零机密；
- 换机器 / 换库 / 轮换口令只改 .env 一处；
- ``setdefault`` 语义：已存在的环境变量优先，便于临时用 shell 覆盖调试。

调用方约定（scripts/migrations/*、scripts/e2e/* 顶部）::

    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]   # rag/
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts._env import load_project_env, mysql_config

    load_project_env(PROJECT_ROOT)
    conn = pymysql.connect(**mysql_config())
"""

from __future__ import annotations

import os
from pathlib import Path

# 沙箱 / 系统代理会劫持对 127.0.0.1 的请求，本地脚本一律先剔除
PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


def load_project_env(project_root: Path, *, strip_proxy: bool = True) -> Path | None:
    """把项目根 .env 读进 os.environ，返回实际读取的文件路径（不存在则 None）。

    只做 ``os.environ.setdefault``：不覆盖调用环境里已有的同名变量，
    因此"临时在 shell 里 export MYSQL_HOST=其他库"依然生效。
    strip_proxy=True 时先剔除代理变量（本机脚本访问 127.0.0.1 会被劫持）。
    文本格式与 backend/app/core/config.py 的 .env 解析保持一致：
    跳过空行与 ``#`` 注释，只按第一个 ``=`` 切分（值里含 ``=`` 不受影响）。
    """
    if strip_proxy:
        for key in PROXY_KEYS:
            os.environ.pop(key, None)

    env_path = project_root / ".env"
    if not env_path.exists():
        # 不是错误：CI / 容器里通常直接用环境变量注入，此时只用已有变量
        return None

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
    return env_path


def mysql_config(charset: str | None = None) -> dict:
    """MySQL 连接参数字典（``pymysql.connect(**mysql_config())`` 直接用）。

    口令必填且显式报错：历史上硬编码时期"忘记配置"不可能发生，改为读 .env 后
    若静默使用空口令，连库失败只会报 MySQL 的 "Access denied"，掩盖真正的
    原因（MYSQL_PASSWORD 没读到），因此这里直接给出可操作的报错信息。

    不打印、不返回完整连接串——调用方如需打印请只打印 host:port/database。
    """
    password = os.getenv("MYSQL_PASSWORD", "")
    if not password:
        raise RuntimeError(
            "MYSQL_PASSWORD 未配置：请在项目根 .env 中设置，或导出同名环境变量。"
        )

    config = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "legal_rag"),
        "password": password,
        "database": os.getenv("MYSQL_DATABASE", "legal_rag"),
    }
    if charset:
        config["charset"] = charset
    return config
