from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

from backend.app import tasks
from backend.app.workspace.service import WorkspaceService
from run import start_celery_worker


class ImmediateExecutor:
    def submit(self, function, *args):
        return function(*args)


class FakeFileStore:
    def __init__(self, record):
        self.record = dict(record)
        self.parsed_contents = []
        self.ready_count = None
        self.stored_reason = ""

    def get(self, document_id):
        return dict(self.record) if document_id == self.record["document_id"] else None

    def set_extraction_result(self, document_id, **fields):
        self.record.update(fields)

    def save_parsed_content(self, document_id, user_id, session_id, text, parsed, extraction):
        self.parsed_contents.append(
            {
                "document_id": document_id,
                "user_id": user_id,
                "session_id": session_id,
                "text": text,
                "parsed": parsed,
                "extraction": extraction,
            }
        )

    def set_ready(self, document_id, chunk_count):
        self.ready_count = chunk_count

    def set_stored(self, document_id, reason=""):
        self.stored_reason = reason


class FakeVectorStore:
    def __init__(self):
        self.rows = []

    def insert(self, rows):
        self.rows.extend(rows)

    def delete(self, user_id, document_id):
        self.rows = [row for row in self.rows if row.get("document_id") != document_id]


class FakeEmbeddingModel:
    def embed(self, texts):
        return [[0.01] * 1024 for _ in texts]


class FailingVectorStore(FakeVectorStore):
    def __init__(self):
        super().__init__()
        self.deleted = []

    def insert(self, rows):
        self.rows.extend(rows)
        raise RuntimeError("Milvus insert failed")

    def delete(self, user_id, document_id):
        self.deleted.append((user_id, document_id))
        super().delete(user_id, document_id)


def test_project_startup_uses_the_same_celery_switch_as_upload_service(monkeypatch):
    """即使系统环境变量残留为 true，也应以已经加载的项目配置为准。"""
    popen_calls = []
    monkeypatch.setenv("START_CELERY_WORKER", "true")
    monkeypatch.setattr("run.subprocess.Popen", lambda *args, **kwargs: popen_calls.append((args, kwargs)))

    process = start_celery_worker(
        SimpleNamespace(workspace_use_celery=False, workspace_background_workers=4)
    )

    assert process is None
    assert popen_calls == []


def test_celery_disabled_processes_upload_with_local_executor(monkeypatch, tmp_path):
    """防止未启动 Celery worker 时，上传任务只进入 Redis 后永久停在 processing。"""
    celery_calls = []
    local_calls = []

    class CeleryTask:
        @staticmethod
        def delay(*args):
            celery_calls.append(args)

    service = object.__new__(WorkspaceService)
    service.settings = SimpleNamespace(workspace_use_celery=False)
    service._processing_executor = ImmediateExecutor()
    service._processing_count = 1
    service._processing_count_lock = Lock()
    service._process_saved_upload_background = lambda *args: local_calls.append(args)
    monkeypatch.setattr(tasks, "process_workspace_upload_task", CeleryTask())

    service._enqueue_saved_upload(
        {"user_id": "user-1"},
        {"document_id": "doc-1"},
        "doc-1",
        "借条.txt",
        str(tmp_path / "借条.txt"),
        "session-1",
    )

    assert len(local_calls) == 1
    assert celery_calls == []


def test_processed_upload_is_chunked_embedded_and_inserted_into_private_vectors(tmp_path):
    """防止解析结果只进 MySQL、却没有任何可供私有语义检索的向量。"""
    path = tmp_path / "借条.txt"
    path.write_text("占位文件", encoding="utf-8")
    record = {
        "document_id": "doc-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "file_name": path.name,
        "status": "processing",
    }
    files = FakeFileStore(record)
    vectors = FakeVectorStore()
    service = object.__new__(WorkspaceService)
    service.settings = SimpleNamespace(
        embedding_batch_size=8,
        embedding_batch_max_chars=30000,
        embedding_dim=1024,
    )
    service.files = files
    service.vectors = vectors
    service.model = FakeEmbeddingModel()
    service.user_state = None
    text = "借款人确认收到三万元借款，并承诺在约定期限内归还。" * 80
    extraction = {
        "extraction_method": "text_parser",
        "ocr_used": False,
        "ocr_status": "not_used",
        "multimodal_used": False,
        "multimodal_status": "not_used",
        "document_type": "text",
        "media_type": "text",
        "extracted_char_count": len(text),
        "extraction_preview": text[:1000],
    }
    service.extract_text = lambda _path, _name: (text, extraction)

    result = service._process_saved_upload(
        {"user_id": "user-1"},
        record,
        "doc-1",
        path.name,
        path,
        "session-1",
    )

    assert vectors.rows
    assert result["status"] == "ready"
    assert result["chunk_count"] == len(vectors.rows)
    assert result["vector_chunk_count"] == len(vectors.rows)
    assert result["vector_index_status"] == "ready"
    assert files.ready_count == len(vectors.rows)
    assert files.parsed_contents[0]["text"] == text
    assert all(len(row["embedding"]) == 1024 for row in vectors.rows)
    assert all(row["user_id"] == "user-1" for row in vectors.rows)
    assert all(row["session_id"] == "session-1" for row in vectors.rows)
    assert all(row["document_id"] == "doc-1" for row in vectors.rows)


def test_private_vector_failure_removes_partial_rows_and_keeps_mysql_text(tmp_path):
    """防止私有向量写入失败后留下半份索引，或丢失已经解析出的原文。"""
    path = tmp_path / "合同.txt"
    path.write_text("占位文件", encoding="utf-8")
    record = {
        "document_id": "doc-2",
        "user_id": "user-1",
        "session_id": "session-1",
        "file_name": path.name,
        "status": "processing",
    }
    files = FakeFileStore(record)
    vectors = FailingVectorStore()
    service = object.__new__(WorkspaceService)
    service.settings = SimpleNamespace(
        embedding_batch_size=8,
        embedding_batch_max_chars=30000,
        embedding_dim=1024,
    )
    service.files = files
    service.vectors = vectors
    service.model = FakeEmbeddingModel()
    service.user_state = None
    text = "双方约定价款三万元，并约定在收到货物后七日内支付。"
    extraction = {
        "extraction_method": "text_parser",
        "ocr_used": False,
        "ocr_status": "not_used",
        "multimodal_used": False,
        "multimodal_status": "not_used",
        "document_type": "text",
        "media_type": "text",
        "extracted_char_count": len(text),
        "extraction_preview": text,
    }
    service.extract_text = lambda _path, _name: (text, extraction)

    result = service._process_saved_upload(
        {"user_id": "user-1"},
        record,
        "doc-2",
        path.name,
        path,
        "session-1",
    )

    assert result["status"] == "stored"
    assert result["vector_chunk_count"] == 0
    assert files.ready_count is None
    assert files.stored_reason
    assert files.parsed_contents[-1]["text"] == text
    assert vectors.deleted == [("user-1", "doc-2")]
    assert vectors.rows == []


def test_private_search_uses_bge_vector_chunks_for_current_session():
    """防止在线问答忽略已经入库的私有向量，只返回整篇 MySQL 文本或等待提示。"""
    class SearchFiles:
        @staticmethod
        def list_for_user(user_id, session_id=None):
            assert user_id == "user-1"
            assert session_id == "session-1"
            return [
                {
                    "document_id": "doc-1",
                    "user_id": "user-1",
                    "session_id": "session-1",
                    "file_name": "借条.txt",
                    "status": "ready",
                }
            ]

        @staticmethod
        def search_parsed_contents(user_id, document_ids, session_id, terms, limit):
            return []

    class SearchVectors:
        @staticmethod
        def search(user_id, vector, document_ids, limit):
            assert user_id == "user-1"
            assert document_ids == ["doc-1"]
            assert len(vector) == 1024
            return [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "content": "借款人确认收到三万元，并约定于五月一日前归还。",
                    "file_name": "借条.txt",
                    "score": 0.91,
                    "source_type": "private",
                }
            ][:limit]

    service = object.__new__(WorkspaceService)
    service.settings = SimpleNamespace(retrieval_private_top_k=4, embedding_dim=1024)
    service.files = SearchFiles()
    service.vectors = SearchVectors()
    service.model = FakeEmbeddingModel()

    rows = service.search(
        {"user_id": "user-1"},
        "请根据我上传的借条说明借款金额",
        "session-1",
    )

    assert rows[0]["source_id"] == "chunk-1"
    assert rows[0]["retrieval_reason"] == "private_vector"
    assert rows[0]["content"] == "借款人确认收到三万元，并约定于五月一日前归还。"


def test_invalid_private_query_vector_falls_back_without_calling_milvus():
    """防止错误维度或异常数字进入 Milvus，并确保 MySQL 降级结果仍可使用。"""
    class SearchFiles:
        @staticmethod
        def list_for_user(user_id, session_id=None):
            return [{"document_id": "doc-1", "file_name": "借条.txt", "status": "ready"}]

        @staticmethod
        def search_parsed_contents(user_id, document_ids, session_id, terms, limit):
            return [
                {
                    "document_id": "doc-1",
                    "extracted_text": "借款金额三万元",
                    "parsed_json": {},
                    "extraction_json": {},
                }
            ]

    class WrongDimensionModel:
        @staticmethod
        def embed(texts):
            return [[0.1, 0.2, 0.3]]

    class TrackingVectors:
        called = False

        def search(self, user_id, vector, document_ids, limit):
            self.called = True
            return []

    vectors = TrackingVectors()
    service = object.__new__(WorkspaceService)
    service.settings = SimpleNamespace(retrieval_private_top_k=4, embedding_dim=1024)
    service.files = SearchFiles()
    service.vectors = vectors
    service.model = WrongDimensionModel()

    rows = service.search(
        {"user_id": "user-1"},
        "借款金额三万元",
        "session-1",
    )

    assert vectors.called is False
    assert rows[0]["source_id"] == "doc-1"
    assert rows[0]["retrieval_reason"] in {"local_exact", "local_keyword"}
