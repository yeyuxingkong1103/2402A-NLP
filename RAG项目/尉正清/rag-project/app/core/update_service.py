# app/core/update_service.py
"""知识库动态更新：按文件内容哈希判断是否需要重建，需要则「先删后增」。

关键点：入库时把文件的 SHA256 写进每条切片的 content_hash 字段，
这里查询旧哈希做比对，内容没变就直接跳过，避免无谓的重新向量化。
"""
import os
from typing import Dict, Optional

from app.config import settings
from app.core.ingest_service import get_ingest_service, file_hash
from app.db import redis_conn
from app.db.milvus_conn import expr_eq, get_milvus
import logging

logger = logging.getLogger(__name__)


class UpdateService:

    def __init__(self):
        self.milvus = get_milvus()
        self.collection = settings.MILVUS_COLLECTION

    def _old_hash(self, source: str, role_id: str) -> Optional[str]:
        """取该文件已入库的哈希。"""
        try:
            rows = self.milvus.query(
                self.collection, expr_eq(source=source, role_id=role_id),
                output_fields=["content_hash"], limit=1)
        except Exception as e:                              # pragma: no cover
            logger.warning("查询旧哈希失败: %s", e)
            return None
        for r in rows:
            if r.get("content_hash"):
                return r["content_hash"]
        return None

    def update_document(self, file_path: str, role_id: str,
                        force: bool = False) -> Dict:
        """更新单个文档。

        force=True 时跳过哈希比对，强制重建。
        """
        if not os.path.exists(file_path):
            return {"code": 400, "msg": "文件不存在: %s" % file_path}

        source = os.path.basename(file_path)
        new_hash = file_hash(file_path)

        if not force:
            old = self._old_hash(source, role_id)
            if old == new_hash:
                logger.info("[%s] %s 内容未变，跳过更新", role_id, source)
                return {"code": 200, "msg": "文件未变更，无需更新",
                        "source": source, "changed": False}

        # 先删旧切片
        self.milvus.delete_by_expr(
            self.collection, expr_eq(source=source, role_id=role_id))
        logger.info("[%s] 已删除 %s 的旧切片", role_id, source)

        # 再按类型重建
        ext = os.path.splitext(file_path)[1].lower()
        ingest = get_ingest_service()
        if ext == ".jsonl":
            result = ingest.ingest_jsonl(file_path, role_id, source=source)
        elif ext == ".pdf":
            result = ingest.ingest_pdf(file_path, role_id)
        elif ext in (".txt", ".md"):
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            n = ingest.ingest_text(text, role_id, source,
                                   title=os.path.splitext(source)[0])
            result = {"chunks": n}
        else:
            return {"code": 400, "msg": "不支持的文件类型: %s" % ext}

        chunks = result.get("chunks", 0)
        logger.info("[%s] %s 更新完成，写入 %s 个切片", role_id, source, chunks)
        return {"code": 200, "msg": "更新成功", "source": source,
                "changed": True, "chunks": chunks}

    def update_role_dataset(self, role_id: str, force: bool = False) -> Dict:
        """批量更新某个角色目录下的全部数据集。"""
        data_dir = os.path.join(settings.DATA_DIR, role_id)
        if not os.path.isdir(data_dir):
            return {"code": 400, "msg": "角色数据目录不存在: %s" % data_dir}

        results, changed = [], 0
        for fn in sorted(os.listdir(data_dir)):
            if not fn.lower().endswith((".jsonl", ".pdf", ".txt", ".md")):
                continue
            r = self.update_document(os.path.join(data_dir, fn), role_id, force=force)
            results.append(r)
            if r.get("changed"):
                changed += 1
        return {"code": 200, "role_id": role_id, "changed_files": changed,
                "total_files": len(results), "details": results}

    def delete_document(self, source: str, role_id: str) -> Dict:
        """按文件名删除某角色下的知识。"""
        self.milvus.delete_by_expr(
            self.collection, expr_eq(source=source, role_id=role_id))
        redis_conn.forget_ingested(role_id, source)
        logger.info("[%s] 已删除 %s", role_id, source)
        return {"code": 200, "msg": "删除成功", "source": source}


_update: Optional[UpdateService] = None


def get_update_service() -> UpdateService:
    global _update
    if _update is None:
        _update = UpdateService()
    return _update
