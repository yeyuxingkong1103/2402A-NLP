import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from backend.app import main as app_main


def test_main_py_bootstraps_project_root_and_runs_uvicorn(monkeypatch):
    script_path = Path(app_main.__file__).resolve()
    project_root = str(script_path.parents[2])
    script_dir = str(script_path.parent)
    filtered_sys_path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(project_root)]
    monkeypatch.setattr(sys, "path", [script_dir, *filtered_sys_path])

    for module_name in list(sys.modules):
        if module_name == "backend" or module_name.startswith("backend."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: calls.append({"args": args, "kwargs": kwargs})),
    )

    runpy.run_path(script_path, run_name="__main__")

    assert calls
    assert calls[0]["kwargs"]["host"] == "127.0.0.1"
    assert calls[0]["kwargs"]["port"] == 8000
