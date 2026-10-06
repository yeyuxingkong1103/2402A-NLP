import sys
from pathlib import Path
import json
import socket
from typing import Callable
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings
from backend.app.system import create_services


def check(name: str, action: Callable[[], dict]) -> dict:
    try:
        return {"name": name, "ok": True, "detail": action()}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


def tcp_check(name: str, host: str, port: int, timeout: float = 2.0) -> dict:
    def action() -> dict:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return {"host": host, "port": int(port), "reachable": True}

    return check(name, action)


def summarize(rows: list[dict]) -> dict:
    return {
        "ok": all(row["ok"] for row in rows),
        "services": rows,
    }


def main() -> int:
    settings = get_settings()
    mysql = urlparse(settings.mysql_url.replace("mysql+pymysql://", "mysql://", 1)) if settings.mysql_url else None
    redis = urlparse(settings.redis_url)
    milvus = urlparse(settings.milvus_uri)
    dependency_rows = [
        tcp_check("mysql_tcp", mysql.hostname or "127.0.0.1", mysql.port or 3306) if mysql else {"name": "mysql_tcp", "ok": False, "error": "MYSQL_URL 未配置"},
        tcp_check("redis_tcp", redis.hostname or "127.0.0.1", redis.port or 6379),
        tcp_check("milvus_tcp", milvus.hostname or "127.0.0.1", milvus.port or 19530),
    ]
    try:
        services = create_services(settings)
    except Exception as exc:
        result = summarize([*dependency_rows, {"name": "app_services", "ok": False, "error": str(exc)}])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    rows = [
        *dependency_rows,
        check("mysql", services["mysql"].health),
        check("redis", services["redis"].health),
        check("milvus", services["milvus"].health),
        check("model", services["model"].health),
        check("workspace", services["workspace"].health),
    ]
    result = summarize(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
