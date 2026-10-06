from backend.app import main as app_main


class FakeUvicorn:
    def __init__(self) -> None:
        self.calls = []

    def run(self, app, host, port, reload):
        self.calls.append({"app": app, "host": host, "port": port, "reload": reload})


class FakeSettings:
    app_host = "0.0.0.0"
    app_port = 9000


def test_main_function_runs_uvicorn_with_settings(monkeypatch):
    fake_uvicorn = FakeUvicorn()
    monkeypatch.setattr(app_main, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(app_main, "AppSettings", lambda: FakeSettings())

    app_main.main()

    assert fake_uvicorn.calls == [
        {"app": app_main.app, "host": "0.0.0.0", "port": 9000, "reload": False}
    ]
