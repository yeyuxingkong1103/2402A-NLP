"""测试夹具：把 src 加入 sys.path，并提供隔离的配置 / Redis 前缀。"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config():
    from role_rag.config import get_config

    return get_config()


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


@pytest.fixture(scope="session")
def redis_available(config) -> bool:
    return _port_open(str(config.get("redis.host")), int(config.get("redis.port")))


@pytest.fixture(scope="session")
def milvus_available(config) -> bool:
    uri = str(config.get("milvus.uri"))
    host = uri.split("//")[-1].split(":")[0]
    port = int(uri.rsplit(":", 1)[-1].strip("/"))
    return _port_open(host, port)


@pytest.fixture
def isolated_redis(config):
    """独立的 Redis 前缀 + db15，避免污染生产数据；用完即清理。"""

    from role_rag.store.redis_store import RedisStore

    store = RedisStore(config)
    store.db = 15
    store.prefix = "rolerag_test:"
    store._client = None
    try:
        store.client.ping()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Redis 不可用：{exc}")

    for key in store.client.scan_iter(match=store.prefix + "*", count=500):
        store.client.delete(key)
    yield store
    for key in store.client.scan_iter(match=store.prefix + "*", count=500):
        store.client.delete(key)
