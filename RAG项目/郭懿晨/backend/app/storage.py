import json
from pathlib import Path

from backend.app.models import BuildTask, Chunk, Document


class JsonStateStore:
    """本地 JSON 状态存储。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save_document(self, document: Document) -> None:
        """保存文档记录。"""
        data = self._read()
        data["documents"][document.document_id] = document.model_dump(mode="json")
        self._write(data)

    def get_document(self, document_id: str) -> Document | None:
        """读取文档记录。"""
        item = self._read()["documents"].get(document_id)
        return Document.model_validate(item) if item else None

    def list_documents(self) -> list[Document]:
        """列出文档记录。"""
        return [Document.model_validate(item) for item in self._read()["documents"].values()]

    def save_task(self, task: BuildTask) -> None:
        """保存任务记录。"""
        data = self._read()
        data["tasks"][task.task_id] = task.model_dump(mode="json")
        self._write(data)

    def get_task(self, task_id: str) -> BuildTask | None:
        """读取任务记录。"""
        item = self._read()["tasks"].get(task_id)
        return BuildTask.model_validate(item) if item else None

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """保存文本块记录。"""
        data = self._read()
        for chunk in chunks:
            data["chunks"][chunk.chunk_id] = chunk.model_dump(mode="json")
        self._write(data)

    def list_chunks(self, document_id: str | None = None) -> list[Chunk]:
        """列出文本块记录。"""
        chunks = [Chunk.model_validate(item) for item in self._read()["chunks"].values()]
        if document_id is None:
            return chunks
        return [chunk for chunk in chunks if chunk.document_id == document_id]

    def _read(self) -> dict[str, dict[str, dict]]:
        if not self.path.exists():
            return {"documents": {}, "tasks": {}, "chunks": {}}
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, data: dict[str, dict[str, dict]]) -> None:
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
