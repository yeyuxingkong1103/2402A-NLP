from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import HTTPException, UploadFile

from data_pipeline import parse_document

from ..config import Settings
from ..models import ModelGateway
from ..storage.mysql import FileStore
from ..storage.vector import WorkspaceVectorStore
from ..system import set_user_context
from .file_utits import ALLOWED_UPLOAD_SUFFIXES, UPLOAD_BATCH_LIMIT, UPLOAD_BATCH_PASSWORD, UPLOAD_TYPE_TEXT, LocalStore, extraction_metadata, extract_text, media_type, safe_upload_name, validate_document_id
from .search import WorkspacePrivateSearchMixin

logger = logging.getLogger("law_rag.workspace.service")
class WorkspaceService(WorkspacePrivateSearchMixin):
    _INTERNAL_FIELDS = frozenset({"storage_path", "user_id"})

    def __init__(self, settings: Settings, files: FileStore, vectors: WorkspaceVectorStore, model: ModelGateway, user_state=None):
        self.settings = settings
        self.files = files
        self.local = LocalStore(settings)
        self.vectors = vectors
        self.model = model
        self.user_state = user_state
        workers = max(1, int(getattr(settings, "workspace_background_workers", 4) or 4))
        self._processing_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="workspace-process")
        self._background_processing_semaphore = threading.BoundedSemaphore(workers)
        self._processing_count = 0
        self._processing_count_lock = threading.Lock()

    @classmethod
    def _sanitize_return(cls, result: dict) -> dict:
        return {key: value for key, value in result.items() if key not in cls._INTERNAL_FIELDS}

    @staticmethod
    def extraction_preview(text: str, limit: int = 500) -> str:
        return " ".join(str(text or "").split())[:limit]

    @staticmethod
    def check_file_type(filename: str) -> None:
        name = str(filename or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        try:
            safe_upload_name(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        suffix = Path(name).suffix.lower()
        if suffix and suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix or '无扩展名'}，仅允许 {', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}")

    @staticmethod
    def validate_upload_batch(files: list[UploadFile], upload_password: str | None = None) -> None:
        if not files:
            raise HTTPException(status_code=400, detail="请至少选择一个文件")
        if len(files) > UPLOAD_BATCH_LIMIT and (not UPLOAD_BATCH_PASSWORD or (upload_password or "").strip() != UPLOAD_BATCH_PASSWORD):
            logger.warning("批量上传密码验证失败", extra={"event": "batch_upload_password_failed"})
            raise HTTPException(status_code=403, detail="批量上传超过5个文件需要管理员配置上传密码" if not UPLOAD_BATCH_PASSWORD else "批量上传密码错误")

    def extract_text(self, path: Path, safe_name: str) -> tuple[str, dict[str, Any]]:
        return extract_text(path, safe_name, self.settings, self.model)

    def save_upload_file(self, user_id: str, original_name: str, file: UploadFile) -> tuple[str, str, Path]:
        try:
            return self.local.save(user_id, original_name, file.file, self.settings.max_upload_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    def add_file_record(self, user_id: str, session_id: str | None, document_id: str, safe_name: str, path: Path) -> dict:
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            self.local.delete(path.relative_to(Path(self.settings.upload_dir).resolve()).as_posix())
            raise HTTPException(status_code=400, detail="文件不能为空")
        return self.files.add({"document_id": document_id, "user_id": user_id, "session_id": session_id, "file_name": safe_name, "storage_path": path.relative_to(Path(self.settings.upload_dir).resolve()).as_posix(), "size_bytes": size_bytes, "status": "processing"})

    def stored_without_index(self, user_id: str, document_id: str, safe_name: str, path: Path, record: dict, reason: str, text: str = "", extraction: dict[str, Any] | None = None) -> dict:
        extraction = extraction or extraction_metadata(path, text, "unparsed")
        processed_path = ""
        if str(text or "").strip():
            try:
                self.files.save_parsed_content(document_id, user_id, record.get("session_id"), text, parse_document(text), extraction)
            except Exception as exc:
                logger.warning("用户解析内容写入 MySQL 失败", extra={"event": "workspace_parsed_mysql_write_failed", "fields": {"document_id": document_id, "error": str(exc)}}, exc_info=True)
        try:
            self.files.set_extraction_result(document_id, text=text, method=extraction["extraction_method"], ocr_used=extraction["ocr_used"], multimodal_used=extraction["multimodal_used"], multimodal_status=extraction["multimodal_status"], document_type=extraction["document_type"], media_type=extraction["media_type"])
            self.files.set_stored(document_id, reason)
        except Exception as exc:
            logger.warning("用户文件待处理状态稍后同步", extra={"event": "workspace_stored_status_deferred", "fields": {"document_id": document_id, "error": str(exc)}}, exc_info=True)
        self._activity(user_id, "file_uploaded", "材料已保存", f"{safe_name} 已保存，但暂未完成解析入库：{reason}")
        return self._sanitize_return({**record, "status": "stored", "chunk_count": 0, "processed_path": processed_path, **extraction, "vector_chunk_count": 0, "vector_index_status": "unavailable", "processing_message": reason})

    def _activity(self, user_id: str, kind: str, title: str, detail: str) -> None:
        if self.user_state:
            try:
                self.user_state.add_activity(user_id, kind, title, detail)
            except Exception as exc:
                logger.warning("用户工作区动态记录失败，已忽略", extra={"event": "workspace_activity_ignored", "fields": {"kind": kind, "error": str(exc)}}, exc_info=True)

    def _save_upload_record(self, user: dict[str, Any], file: UploadFile, session_id: str | None = None) -> tuple[str, str, Path, dict]:
        original_name = file.filename or "upload.bin"
        set_user_context(user["user_id"], session_id)
        self.check_file_type(original_name)
        document_id, safe_name, path = self.save_upload_file(user["user_id"], original_name, file)
        logger.info("用户文件已保存", extra={"event": "workspace_upload_saved", "fields": {"document_id": document_id, "file_name": safe_name, "original_suffix": Path(original_name).suffix.lower()}})
        return document_id, safe_name, path, self.add_file_record(user["user_id"], session_id, document_id, safe_name, path)

    def _async_media_types(self) -> set[str]:
        return {item.strip().lower() for item in str(getattr(self.settings, "workspace_async_media_types", "audio,video") or "").split(",") if item.strip()}

    def should_defer_processing(self, path: Path) -> bool:
        return bool(getattr(self.settings, "workspace_async_media_processing", True) and media_type(path) in self._async_media_types())

    def _processing_result(self, document_id: str, safe_name: str, path: Path, record: dict, tasks: Any, user: dict[str, Any], session_id: str | None) -> dict:
        extraction = extraction_metadata(path, "", "pending")
        self.files.set_extraction_result(document_id, text="", method=extraction["extraction_method"], ocr_used=extraction["ocr_used"], multimodal_used=extraction["multimodal_used"], multimodal_status=extraction["multimodal_status"], document_type=extraction["document_type"], media_type=extraction["media_type"])
        limit = max(1, int(getattr(self.settings, "workspace_pending_limit", 1000) or 1000))
        with self._processing_count_lock:
            if self._processing_count >= limit:
                reason = "后台解析队列已满，文件已保存，请稍后重新处理"
                self.files.set_stored(document_id, reason)
                return {**record, "status": "stored", "chunk_count": 0, "processed_path": "", **extraction, "vector_chunk_count": 0, "vector_index_status": "queued", "processing_message": reason}
            self._processing_count += 1
        tasks.add_task(self._enqueue_saved_upload, dict(user), dict(record), document_id, safe_name, str(path), session_id)
        return {**record, "status": "processing", "chunk_count": 0, "processed_path": "", **extraction, "vector_chunk_count": 0, "vector_index_status": "processing", "processing_message": "文件已保存，正在后台解析并写入材料库"}

    def _enqueue_saved_upload(self, user: dict[str, Any], record: dict, document_id: str, safe_name: str, path_text: str, session_id: str | None) -> None:
        queued = False
        try:
            from ..tasks import process_workspace_upload_task

            process_workspace_upload_task.delay(path_text, user, record, document_id, safe_name, session_id)
            queued = True
        except Exception as exc:
            logger.warning("持久化任务队列不可用，回退本地线程处理", extra={"event": "workspace_task_queue_fallback", "fields": {"document_id": document_id, "error": str(exc)}})
            self._processing_executor.submit(self._process_saved_upload_background, user, record, document_id, safe_name, path_text, session_id)
        if queued:
            with self._processing_count_lock:
                self._processing_count = max(0, self._processing_count - 1)

    def pending_processing_count(self) -> int:
        with self._processing_count_lock:
            return self._processing_count

    def _process_saved_upload_background(self, user: dict[str, Any], record: dict, document_id: str, safe_name: str, path_text: str, session_id: str | None) -> None:
        set_user_context(user["user_id"], session_id)
        wait_started = perf_counter()
        self._background_processing_semaphore.acquire()
        started = perf_counter()
        try:
            logger.info("用户文件后台处理开始", extra={"event": "workspace_background_processing_started", "fields": {"wait_ms": round((started - wait_started) * 1000, 2), "document_id": document_id, "file_name": safe_name}})
            self._process_saved_upload(user, record, document_id, safe_name, Path(path_text), session_id)
            logger.info("用户文件后台处理完成", extra={"event": "workspace_background_processing_finished", "fields": {"elapsed_ms": round((perf_counter() - started) * 1000, 2), "document_id": document_id, "file_name": safe_name}})
        except Exception as exc:
            reason = str(getattr(exc, "detail", exc)) or "后台解析服务暂时不可用"
            logger.warning("用户文件后台处理失败，已保留原文件", extra={"event": "workspace_background_processing_failed", "fields": {"document_id": document_id, "error": reason}}, exc_info=True)
            try:
                self.files.set_failed(document_id, reason)
            except Exception:
                logger.exception("用户文件后台失败状态同步失败", extra={"event": "workspace_background_processing_status_failed", "fields": {"document_id": document_id}})
        finally:
            self._background_processing_semaphore.release()
            with self._processing_count_lock:
                self._processing_count = max(0, self._processing_count - 1)

    def _process_saved_upload(self, user: dict[str, Any], record: dict, document_id: str, safe_name: str, path: Path, session_id: str | None = None) -> dict:
        set_user_context(user["user_id"], session_id)
        current = self.files.get(document_id)
        if not current or str(current.get("user_id") or "") != str(user.get("user_id") or ""):
            return {"document_id": document_id, "status": "deleted", "deleted": True}
        try:
            text, extraction = self.extract_text(path, safe_name)
            if not text.strip():
                return self.stored_without_index(user["user_id"], document_id, safe_name, path, record, UPLOAD_TYPE_TEXT if extraction["media_type"] == "file" else "文件已保存，但未识别到可检索文本")
            self.files.set_extraction_result(document_id, text=text, method=extraction["extraction_method"], ocr_used=extraction["ocr_used"], multimodal_used=extraction["multimodal_used"], multimodal_status=extraction["multimodal_status"], document_type=extraction["document_type"], media_type=extraction["media_type"])
            if not self.files.get(document_id):
                return {"document_id": document_id, "status": "deleted", "deleted": True}
            parsed = parse_document(text)
            self.files.save_parsed_content(
                document_id,
                user["user_id"],
                session_id,
                text,
                parsed,
                extraction,
            )
            processed_path = "mysql"
        except Exception as exc:
            reason = str(getattr(exc, "detail", exc)) or "解析或索引服务暂时不可用"
            logger.warning("用户文件已保存，解析或索引稍后处理", extra={"event": "workspace_upload_degraded_to_stored", "fields": {"document_id": document_id, "error": reason}}, exc_info=True)
            return self.stored_without_index(user["user_id"], document_id, safe_name, path, record, f"文件已保存，解析或索引稍后处理：{reason}", text=locals().get("text", ""), extraction=locals().get("extraction"))
        try:
            if not self.files.get(document_id):
                return {"document_id": document_id, "status": "deleted", "deleted": True}
            self.files.set_ready(document_id, 1)
        except Exception as exc:
            logger.warning("用户文件索引状态稍后同步", extra={"event": "workspace_ready_status_deferred", "fields": {"document_id": document_id, "error": str(exc)}}, exc_info=True)
        self._activity(user["user_id"], "file_uploaded", "材料已上传", f"{safe_name} 已抽取 {extraction['extracted_char_count']} 字并保存完整解析内容到 MySQL")
        logger.info("用户文件解析内容保存完成", extra={"event": "workspace_upload_parsed_mysql", "fields": {"document_id": document_id, "extracted_char_count": extraction["extracted_char_count"], "extraction_method": extraction["extraction_method"], "ocr_used": extraction["ocr_used"], "ocr_status": extraction["ocr_status"], "multimodal_used": extraction["multimodal_used"], "multimodal_status": extraction["multimodal_status"], "media_type": extraction["media_type"]}})
        return self._sanitize_return({**record, "status": "ready", "chunk_count": 1, "processed_path": processed_path, **extraction, "content_record_count": 1, "vector_chunk_count": 0, "vector_index_status": "not_used"})

    def _upload_sync(self, user: dict[str, Any], file: UploadFile, session_id: str | None = None) -> dict:
        document_id, safe_name, path, record = self._save_upload_record(user, file, session_id)
        return self._process_saved_upload(user, record, document_id, safe_name, path, session_id)

    async def upload(self, user: dict[str, Any], file: UploadFile, session_id: str | None = None, tasks: Any | None = None) -> dict:
        if tasks is None:
            return await asyncio.to_thread(self._upload_sync, user, file, session_id)
        document_id, safe_name, path, record = await asyncio.to_thread(self._save_upload_record, user, file, session_id)
        return self._processing_result(document_id, safe_name, path, record, tasks, user, session_id)

    async def upload_batch(self, user: dict[str, Any], files: list[UploadFile], session_id: str | None = None, upload_password: str | None = None, tasks: Any | None = None) -> dict:
        self.validate_upload_batch(files, upload_password)
        semaphore = asyncio.Semaphore(max(1, int(getattr(self.settings, "upload_concurrency", 2) or 2)))
        async def upload_one(file: UploadFile) -> dict:
            try:
                async with semaphore:
                    return await self.upload(user, file, session_id, tasks=tasks)
            except HTTPException as exc:
                return {"file_name": file.filename or "upload.bin", "status": "failed", "error": str(exc.detail)}
            except Exception as exc:
                logger.exception("批量上传中的单个文件处理失败", extra={"event": "workspace_batch_upload_item_failed", "fields": {"file_name": file.filename or "upload.bin"}})
                return {"file_name": file.filename or "upload.bin", "status": "failed", "error": str(exc)}
        results = await asyncio.gather(*(upload_one(file) for file in files))
        uploaded = [item for item in results if item.get("status") in {"ready", "stored", "processing"}]
        failed = [item for item in results if item.get("status") == "failed"]
        return {"files": results, "uploaded_count": len(uploaded), "failed_count": len(failed), "processing_count": sum(item.get("status") == "processing" for item in uploaded), "stored_count": sum(item.get("status") == "stored" for item in uploaded), "ready_count": sum(item.get("status") == "ready" for item in uploaded)}

    def list_files(self, user: dict[str, Any], session_id: str | None = None) -> list[dict]:
        return self.files.list_for_user(user["user_id"], session_id)

    def delete_record(self, user: dict[str, Any], record: dict) -> list[str]:
        document_id = str(record.get("document_id") or "")
        user_id = str(user.get("user_id") or record.get("user_id") or "")
        warnings = []
        try:
            storage_path = str(record.get("storage_path") or "")
            if storage_path:
                self.local.delete(storage_path)
        except Exception as exc:
            warnings.append("file")
            logger.warning("用户原始文件清理失败", extra={"event": "workspace_file_storage_delete_failed", "fields": {"document_id": document_id, "error": str(exc)}}, exc_info=True)
        try:
            self.files.delete(document_id)
        except Exception as exc:
            warnings.append("database")
            logger.warning("用户文件数据库记录移除失败", extra={"event": "workspace_file_record_delete_ignored", "fields": {"document_id": document_id, "error": str(exc)}}, exc_info=True)
        return list(dict.fromkeys(warnings))

    def delete(self, user: dict[str, Any], document_id: str, session_id: str | None = None) -> dict:
        if not validate_document_id(document_id):
            raise HTTPException(status_code=400, detail="无效的文件ID")
        record = self.files.get(document_id)
        set_user_context(user["user_id"], session_id)
        if not record or record["user_id"] != user["user_id"]:
            return {"deleted": True, "document_id": document_id, "already_absent": True}
        if session_id and record.get("session_id") != session_id:
            return {"deleted": False, "document_id": document_id, "session_mismatch": True}
        warnings = self.delete_record(user, record)
        self._activity(user["user_id"], "file_deleted", "材料已移除", f"已删除 {record['file_name']} 的原文件、MySQL 解析内容和数据库记录")
        logger.info("用户文件删除已受理", extra={"event": "workspace_file_delete_accepted", "fields": {"document_id": document_id, "warnings": warnings}})
        return {"deleted": True, "document_id": document_id, "warnings": warnings}

    def delete_session_files(self, user: dict[str, Any], session_id: str) -> dict:
        if not validate_document_id(session_id):
            raise HTTPException(status_code=400, detail="无效的会话ID")
        set_user_context(user["user_id"], session_id)
        records = list(self.files.list_for_user(user["user_id"], session_id))
        warnings = [warning for record in records for warning in self.delete_record(user, record)]
        if records:
            self._activity(user["user_id"], "session_files_deleted", "对话材料已移除", f"已删除当前对话的 {len(records)} 份材料及其 MySQL 解析内容")
        logger.info("会话关联材料删除已受理", extra={"event": "workspace_session_files_delete_accepted", "fields": {"session_id": session_id, "file_count": len(records), "warnings": warnings}})
        return {"deleted": True, "session_id": session_id, "file_count": len(records), "warnings": warnings}

    def health(self) -> dict:
        return {"upload_dir": self.settings.upload_dir}


