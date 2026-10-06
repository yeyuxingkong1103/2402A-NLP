from __future__ import annotations

import logging
import re

from fastapi import HTTPException

from ..system import set_user_context
from .file_utits import OLD_MATERIAL_REFERENCE_HINTS, PRIVATE_FOLLOWUP_HINTS, PRIVATE_MATERIAL_HINTS, PRIVATE_QUERY_WEAK_TERMS, PRIVATE_TOKEN_PATTERN, PUBLIC_LAW_LOOKUP_HINTS, RESET_CONTEXT_HINTS

logger = logging.getLogger("law_rag.workspace.search")


def parsed_content_summary(parsed: object, extraction: object) -> str:
    if not isinstance(parsed, dict):
        return ""
    metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
    sections = metadata.get("sections") if isinstance(metadata.get("sections"), dict) else {}
    lines = []
    document_type = parsed.get("document_type") or metadata.get("document_type")
    char_count = metadata.get("char_count")
    if document_type or char_count:
        lines.append(f"文档类型：{document_type or '未知'}；解析字符数：{char_count or '未知'}")
    if isinstance(extraction, dict):
        method = extraction.get("extraction_method")
        ocr_status = extraction.get("ocr_status")
        multimodal_status = extraction.get("multimodal_status")
        if method or ocr_status or multimodal_status:
            lines.append(f"抽取方式：{method or '未知'}；OCR：{ocr_status or '未使用'}；多模态：{multimodal_status or '未使用'}")
    section_items = [f"{key}={value}" for key, value in sections.items() if value]
    if section_items:
        lines.append("识别到的结构：" + "；".join(section_items[:8]))
    return "\n".join(lines)

class WorkspacePrivateSearchMixin:
    @staticmethod
    def compact_match_text(value: object) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "")).casefold()

    @staticmethod
    def query_mentions_private_material(query: str) -> bool:
        return any(hint in str(query or "") for hint in PRIVATE_MATERIAL_HINTS)

    @staticmethod
    def query_is_public_law_lookup(query: str) -> bool:
        value = str(query or "")
        return bool(value and (re.search(r"第\s*[一二三四五六七八九十百千万零〇两\d]+\s*条", value) or any(hint in value for hint in PUBLIC_LAW_LOOKUP_HINTS)))

    @classmethod
    def query_resets_previous_context(cls, query: str) -> bool:
        value = cls.compact_match_text(query)
        return bool(value and any(cls.compact_match_text(hint) in value for hint in RESET_CONTEXT_HINTS))

    @classmethod
    def query_requests_old_materials(cls, query: str) -> bool:
        value = str(query or "").strip()
        if not value or cls.query_resets_previous_context(value):
            return False
        compact = cls.compact_match_text(value)
        return any(cls.compact_match_text(hint) in compact for hint in OLD_MATERIAL_REFERENCE_HINTS) and (cls.query_mentions_private_material(value) or any(hint in value for hint in PRIVATE_FOLLOWUP_HINTS))

    @classmethod
    def query_is_private_material_followup(cls, query: str) -> bool:
        value = str(query or "").strip()
        if not value or cls.query_resets_previous_context(value):
            return False
        candidates = [value, *(line.strip() for line in value.splitlines() if line.strip())]
        if any(cls.query_mentions_private_material(candidate) for candidate in candidates):
            return True
        return any(any(hint in candidate for hint in PRIVATE_FOLLOWUP_HINTS) and not cls.query_is_public_law_lookup(candidate) and len(cls.compact_match_text(candidate)) <= 40 for candidate in candidates)

    @classmethod
    def private_query_terms(cls, query: str) -> list[str]:
        terms: list[str] = []
        for token in PRIVATE_TOKEN_PATTERN.findall(str(query or "")):
            compact = cls.compact_match_text(token)
            if len(compact) >= 2 and compact not in PRIVATE_QUERY_WEAK_TERMS:
                terms.append(compact)
                terms.extend(match.group(0) for match in re.finditer(r"\d{2,}", compact))
        return list(dict.fromkeys(terms))[:10]

    def search_files_for_query(self, user: dict, session_id: str | None, query: str) -> tuple[list[dict], str]:
        if self.query_resets_previous_context(query):
            return [], "session_reset"
        files = list(self.list_files(user, session_id))
        if not session_id:
            return files, "user"
        if files:
            return files, "session"
        if self.query_requests_old_materials(query):
            fallback = list(self.list_files(user, None))
            return (fallback, "user_fallback") if fallback else (files, "session_empty_followup")
        return files, "session_empty"

    def score_private_snapshot(self, query: str, terms: list[str], file_name: str, content: str, material_reference: bool, file_rank: int) -> tuple[float, str]:
        haystack = self.compact_match_text(f"{file_name}\n{content}")
        query_norm = self.compact_match_text(query)
        if not haystack:
            return 0.0, ""
        if query_norm and len(query_norm) >= 4 and query_norm in haystack:
            return 0.98, "local_exact"
        file_norm = self.compact_match_text(file_name)
        if file_norm and query_norm and (file_norm in query_norm or query_norm in file_norm):
            return 0.94, "file_name_exact"
        matched = [term for term in terms if term in haystack]
        if matched:
            return min(0.95, 0.74 + min(0.16, len(matched) * 0.035) + min(0.05, max(map(len, matched)) * 0.008)), "local_keyword"
        return (max(0.62, 0.72 - file_rank * 0.03), "material_reference") if material_reference else (0.0, "")

    def pending_session_materials(self, files: list[dict], limit: int) -> list[dict]:
        rows = []
        for file_rank, record in enumerate(files):
            document_id = str(record.get("document_id") or "").strip()
            file_name = str(record.get("file_name") or "用户上传材料")
            if not document_id:
                continue
            status = str(record.get("status") or "processing")
            message = f"当前对话文件“{file_name}”已保存，但 MySQL 中暂未返回完整解析内容；不能把文件内容当作已识别事实使用。" if status == "ready" else (f"当前对话文件“{file_name}”已保存，但解析内容暂不可用；需要稍后重试或重新解析后才能依据文件内容判断。" if status == "stored" else f"当前对话文件“{file_name}”已保存，仍在后台解析并写入 MySQL；现在不能把文件内容当作已识别事实使用。")
            rows.append({"source_id": f"{document_id}_pending_material", "source_type": "private", "title": file_name, "file_name": file_name, "content": message, "score": round(max(0.5, 0.68 - file_rank * 0.03), 4), "document_id": document_id, "retrieval_channel": "private", "retrieval_reason": "material_pending"})
        return rows[:limit]

    @staticmethod
    def merge_private_rows(result_sets: list[list[dict]], limit: int) -> list[dict]:
        merged: dict[str, dict] = {}
        for rows in result_sets:
            for row in rows:
                key = f"{row.get('document_id', '')}_{row.get('_chunk_rank', row.get('chunk_index', 0))}" if row.get("document_id") else row.get("source_id") or f"hash_{hash(row.get('content', ''))}"
                if key not in merged or row.get("score", 0) > merged[key].get("score", 0):
                    merged[key] = row
        return sorted(merged.values(), key=lambda row: float(row.get("score", 0) or 0), reverse=True)[:limit]

    def search(self, user: dict, query: str, session_id: str | None = None, limit: int | None = None, query_vector: list[float] | None = None) -> list[dict]:
        set_user_context(user["user_id"], session_id)
        result_limit = limit or self.settings.retrieval_private_top_k
        files, scope = self.search_files_for_query(user, session_id, query)
        if scope == "session_empty":
            return []
        if scope in {"session_empty_followup", "session_reset"}:
            return [{"source_id": "current_session_empty_notice", "source_type": "private", "source_label": "当前会话", "title": "当前会话没有可用材料", "file_name": "当前会话没有可用材料", "content": "已按用户要求将本轮问题与之前对话和旧材料隔离；当前这条问题不会引用之前上传的材料。" if scope == "session_reset" else "当前会话没有上传材料；已按你的要求回看之前的旧材料，但没有找到可用材料。", "score": 0.01, "document_id": "", "retrieval_channel": "private", "retrieval_reason": scope}]
        document_ids = [str(file["document_id"]) for file in files if file.get("document_id")]
        mysql_results: list[dict] = []
        mysql_error = ""
        try:
            terms = self.private_query_terms(query)
            material_reference = scope in {"session", "user_fallback"} and self.query_is_private_material_followup(query)
            rows = self.files.search_parsed_contents(
                user["user_id"],
                document_ids,
                session_id if scope == "session" else None,
                [] if material_reference else terms,
                result_limit * 4,
            ) if document_ids else []
            for row in rows:
                content = str(row.get("extracted_text") or "").strip()
                parsed = row.get("parsed_json") or {}
                extraction = row.get("extraction_json") or {}
                summary = parsed_content_summary(parsed, extraction)
                evidence_content = f"【解析摘要】\n{summary}\n\n【完整识别文本】\n{content}" if summary else content
                file_record = next((item for item in files if str(item.get("document_id")) == str(row.get("document_id"))), {})
                file_name = str(file_record.get("file_name") or "用户上传材料")
                file_rank = next((index for index, item in enumerate(files) if str(item.get("document_id")) == str(row.get("document_id"))), 0)
                score, reason = self.score_private_snapshot(query, terms, file_name, content, material_reference, file_rank)
                if material_reference and not score:
                    score, reason = 0.62, "material_reference"
                if score:
                    mysql_results.append({"source_id": row.get("document_id"), "source_type": "private", "title": file_name, "file_name": file_name, "content": evidence_content, "score": round(score, 4), "document_id": row.get("document_id", ""), "retrieval_channel": "private", "retrieval_reason": reason, "parsed": parsed, "extraction": extraction})
            mysql_results.sort(key=lambda item: float(item.get("score", 0) or 0), reverse=True)
            mysql_results = mysql_results[:result_limit]
        except Exception as exc:
            mysql_error = str(exc)
            logger.warning("用户私有材料 MySQL 检索失败", extra={"event": "workspace_private_mysql_search_failed", "fields": {"document_count": len(files), "error": mysql_error}}, exc_info=True)
        pending_results = self.pending_session_materials(files, result_limit) if scope == "session" and document_ids and not mysql_results and not self.query_is_public_law_lookup(query) else []
        results = self.merge_private_rows([mysql_results, pending_results], result_limit)
        logger.info("用户私有材料检索完成", extra={"event": "workspace_private_retrieval_completed", "fields": {"document_count": len(files), "document_scope": scope, "mysql_result_count": len(mysql_results), "pending_result_count": len(pending_results), "mysql_error": mysql_error, "retrieval_sources": [{"source_id": row["source_id"], "file_name": row["title"], "score": row["score"], "retrieval_reason": row.get("retrieval_reason", "mysql")} for row in results]}})
        return results

