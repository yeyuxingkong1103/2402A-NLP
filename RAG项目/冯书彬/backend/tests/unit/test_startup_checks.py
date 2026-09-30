from backend.app.core.config import AppSettings
from backend.app.core.startup_checks import (
    DependencyStatus,
    _check_mysql_readiness,
    _check_redis_readiness,
    is_ready,
)


def test_mysql_readiness_requires_database_url():
    result = _check_mysql_readiness(AppSettings(DATABASE_URL=""))

    assert result == DependencyStatus("mysql", False, "missing database url")


def test_mysql_readiness_executes_select_one(monkeypatch):
    executed = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement):
            executed.append(str(statement))

    class Engine:
        def connect(self):
            return Connection()

        def dispose(self):
            pass

    monkeypatch.setattr("backend.app.repositories.auth_repository.create_engine_from_url", lambda *_args, **_kwargs: Engine())
    result = _check_mysql_readiness(AppSettings(DATABASE_URL="sqlite://"))

    assert result == DependencyStatus("mysql", True)
    assert executed == ["SELECT 1"]


def test_redis_readiness_reports_connection_error(monkeypatch):
    class Client:
        def ping(self):
            raise TimeoutError("unreachable")

        def close(self):
            pass

    monkeypatch.setattr("redis.from_url", lambda *_args, **_kwargs: Client())
    result = _check_redis_readiness(AppSettings(REDIS_URL="redis://unreachable"))

    assert result == DependencyStatus("redis", False, "TimeoutError")


def test_ready_requires_all_dependencies_ok():
    statuses = [
        DependencyStatus(name="mysql", ok=True, detail=None),
        DependencyStatus(name="redis", ok=True, detail=None),
        DependencyStatus(name="milvus", ok=False, detail="connection failed"),
    ]

    assert is_ready(statuses) is False


def test_ready_passes_when_all_dependencies_ok():
    statuses = [
        DependencyStatus(name="mysql", ok=True, detail=None),
        DependencyStatus(name="gpu", ok=True, detail="cuda available"),
    ]

    assert is_ready(statuses) is True
