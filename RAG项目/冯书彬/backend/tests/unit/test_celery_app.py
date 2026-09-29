from backend.app.workers import celery_app as celery_module
from backend.app.workers.celery_app import check_celery_readiness


class FakeInspect:
    def __init__(self, result):
        self.result = result

    def ping(self):
        return self.result


class FakeControl:
    def __init__(self, result):
        self.result = result

    def inspect(self, timeout):
        return FakeInspect(self.result)


class FakeCeleryApp:
    def __init__(self, result):
        self.control = FakeControl(result)


def test_celery_readiness_reports_no_worker(monkeypatch):
    monkeypatch.setattr(celery_module, "celery_app", FakeCeleryApp(None))

    ok, detail = check_celery_readiness(timeout_seconds=0.01)

    assert ok is False
    assert detail == "no workers"


def test_celery_readiness_reports_worker(monkeypatch):
    monkeypatch.setattr(celery_module, "celery_app", FakeCeleryApp({"worker": {"ok": "pong"}}))

    ok, detail = check_celery_readiness(timeout_seconds=0.01)

    assert ok is True
    assert detail is None
