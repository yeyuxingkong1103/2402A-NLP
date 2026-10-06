from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import BuildTask
from backend.app.storage import JsonStateStore
from backend.app import routes_files, routes_tasks


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_non_pdf():
    client = TestClient(app)

    response = client.post("/api/files/upload", files={"file": ("a.txt", b"x", "text/plain")})

    assert response.status_code == 400


def _use_tmp_upload_dependencies(monkeypatch, tmp_path):
    """把上传接口的落盘目录与 state.json 隔离到临时目录。"""
    data_dir = tmp_path / "data"
    store = JsonStateStore(tmp_path / "state.json")
    monkeypatch.setattr(routes_files, "get_file_dependencies", lambda: (data_dir, store))
    return store


def test_upload_rejects_duplicate_pdf_content(monkeypatch, tmp_path):
    store = _use_tmp_upload_dependencies(monkeypatch, tmp_path)
    client = TestClient(app)
    content = b"%PDF-1.7 fake body"
    files = {"file": ("a.pdf", content, "application/pdf")}

    first = client.post("/api/files/upload", files=files)
    second = client.post("/api/files/upload", files=files)

    assert first.status_code == 200
    assert second.status_code == 409
    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0].file_name == "a.pdf"
    assert documents[0].content_hash is not None


def test_upload_allows_same_name_with_different_content(monkeypatch, tmp_path):
    store = _use_tmp_upload_dependencies(monkeypatch, tmp_path)
    client = TestClient(app)

    first = client.post(
        "/api/files/upload", files={"file": ("a.pdf", b"%PDF-1.7 v1", "application/pdf")}
    )
    second = client.post(
        "/api/files/upload", files={"file": ("a.pdf", b"%PDF-1.7 v2", "application/pdf")}
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(store.list_documents()) == 2


def test_documents_endpoint_returns_list():
    client = TestClient(app)

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_start_task_endpoint_does_not_crash_when_building_pipeline(monkeypatch):
    client = TestClient(app)

    class FakeStore:
        def get_task(self, task_id):
            return BuildTask.new("doc-1")

    class FakePipeline:
        def run(self, task_id):
            return None

    monkeypatch.setattr(routes_tasks, "get_task_store", lambda: FakeStore())
    monkeypatch.setattr(routes_tasks, "build_pipeline", lambda: FakePipeline())

    response = client.post("/api/tasks/task-1/start")

    assert response.status_code == 200
    assert response.json()["status"] == "scheduled"
