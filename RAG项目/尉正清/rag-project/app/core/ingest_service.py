# app/core/ingest_service.py
"""知识入库：把 JSONL 数据集与 PDF 文档切块、向量化、写入 Milvus。

切块策略（兼顾检索精度与上下文完整性）：
  - 短记录（<= MAX_CHUNK 字）：整条作为一条知识，用 embed_text 做向量。
    embed_text 是清洗过的「问题/标题」，比整段正文更适合被检索命中；
    display_text 原样存下来喂给大模型。
  - 长记录：走 chunk_service.semantic_split 语义分块，每块各自向量化。

写入的字段见 MilvusStore.ensure_knowledge_collection 的说明。
"""
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from app.config import settings
from app.config.components import get_embeddings
from app.core.chunk_service import semantic_split
from app.db import redis_conn
from app.db.milvus_conn import get_milvus
import logging

logger = logging.getLogger(__name__)

MAX_CHUNK = 1500            # display_text 超过该长度才切块
BATCH = 200                 # 向量化批大小
SUMMARY_CHARS = 200         # 摘要字段保留的字符数


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


class IngestService:

    def __init__(self):
        self.milvus = get_milvus()
        self.collection = settings.MILVUS_COLLECTION

    def _ensure(self):
        self.milvus.ensure_knowledge_collection(self.collection)

    # ---------------- 核心：记录 -> Milvus 行 ----------------
    def _rows_from_records(self, records: List[dict], role_id: str,
                           source: str, content_hash: str = "") -> List[dict]:
        """把数据集记录转成待写入的行（含向量）。

        分块策略：
            短记录（≤ MAX_CHUNK）  整条入库，不动——法条按「条」、问答按「一对」，
                                  本身就是不可再分的语义单元，占知识库 99.9%
            长记录                 走语义分块，在话题转折处切，而不是按固定字数
        """
        pending = []            # dict: embed / display / title / doc_type / meta
        embed_fn = get_embeddings().embed_documents

        for rec in records:
            display = (rec.get("display_text") or "").strip()
            embed = (rec.get("embed_text") or "").strip()
            if not display:
                continue
            meta = rec.get("meta") or {}
            law = meta.get("law") or ""
            article = meta.get("article") or ""
            # 法条记录没写 meta.title，用「《法律名》第X条」补上，
            # 否则检索结果里只能显示文件名，用户看不出命中了哪一条
            title = meta.get("title") or ""
            if not title and law and article:
                title = "《%s》%s" % (law, article)
            base = {
                "title": title,
                "doc_type": rec.get("doc_type") or "",
                "law": law,
                "article": article,
                # 数据集可自带 summary；没有才退化为检索目标的截断
                "summary": rec.get("summary") or "",
            }

            if len(display) <= MAX_CHUNK:
                pending.append(dict(base, embed=embed or display, display=display))
            else:
                # 长文按语义转折切块；每块自己当检索目标
                for piece in semantic_split(display, embed_fn):
                    piece = piece.strip()
                    if len(piece) < 20:
                        continue
                    pending.append(dict(base, embed=piece, display=piece))

        if not pending:
            return []

        emb = get_embeddings()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for i in range(0, len(pending), BATCH):
            batch = pending[i:i + BATCH]
            dense, sparse = emb.embed_both([p["embed"] for p in batch])
            for j, p in enumerate(batch):
                rows.append({
                    # id 由 Milvus 自增生成，不在这里指定
                    "vector": dense[j],
                    "sparse_vector": sparse[j],
                    "text": p["display"],
                    # 优先用数据集自带的摘要（信息量更足）；
                    # 没有则退回检索目标的前若干字
                    "summary": (p["summary"] or p["embed"])[:SUMMARY_CHARS],
                    "role_id": role_id,
                    "source": source,
                    "doc_type": p["doc_type"],
                    "title": p["title"],
                    "law": p["law"],
                    "article": p["article"],
                    "content_hash": content_hash,
                    "created_at": now,
                    "updated_at": now,
                })
        return rows

    # ---------------- JSONL 数据集入库 ----------------
    def ingest_jsonl(self, path: str, role_id: str,
                     source: Optional[str] = None) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError("文件不存在: %s" % path)
        source = source or os.path.basename(path)

        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        h = file_hash(path)
        self._ensure()
        rows = self._rows_from_records(records, role_id, source, content_hash=h)
        if not rows:
            return {"source": source, "records": len(records), "chunks": 0}
        self.milvus.insert(self.collection, rows)
        redis_conn.mark_ingested(role_id, [source])
        logger.info("[%s] %s -> %s 条记录 / %s 个切片", role_id, source,
                    len(records), len(rows))
        return {"source": source, "records": len(records), "chunks": len(rows)}

    def ingest_role_dir(self, role_id: str, data_dir: Optional[str] = None) -> Dict:
        """把一个角色目录下的所有 JSONL 灌进知识库。"""
        data_dir = data_dir or os.path.join(settings.DATA_DIR, role_id)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError("角色数据目录不存在: %s" % data_dir)

        files = sorted(f for f in os.listdir(data_dir) if f.endswith(".jsonl"))
        if not files:
            raise FileNotFoundError("目录下没有 .jsonl 文件: %s" % data_dir)

        details, chunks, records = [], 0, 0
        for fn in files:
            r = self.ingest_jsonl(os.path.join(data_dir, fn), role_id)
            details.append(r)
            chunks += r["chunks"]
            records += r["records"]
        return {"role_id": role_id, "files": len(files), "records": records,
                "chunks": chunks, "details": details}

    # ---------------- PDF 入库 ----------------
    def ingest_pdf(self, path: str, role_id: str,
                   remove_watermark: bool = True,
                   use_ocr: bool = True,
                   use_tables: bool = True,
                   doc_type: str = "pdf") -> Dict:
        from app.core.pdf_service import get_pdf_service

        pages, report = get_pdf_service().extract_pages(
            path, remove_watermark=remove_watermark, use_ocr=use_ocr,
            use_tables=use_tables)
        if not pages:
            return {"source": os.path.basename(path), "chunks": 0,
                    "watermark": report.to_dict(), "msg": "未提取到文本"}

        title = os.path.splitext(os.path.basename(path))[0]
        records = [{
            "embed_text": p["text"][:200],
            "display_text": "【%s · 第%d页】\n%s" % (title, p["page"], p["text"]),
            "doc_type": doc_type,
            "meta": {"title": "%s P%d" % (title, p["page"])},
        } for p in pages]

        h = file_hash(path)
        self._ensure()
        rows = self._rows_from_records(records, role_id,
                                       os.path.basename(path), content_hash=h)
        if rows:
            self.milvus.insert(self.collection, rows)
            redis_conn.mark_ingested(role_id, [os.path.basename(path)])
        logger.info("PDF 入库 [%s] %s -> %s 个切片 (%s)", role_id,
                    os.path.basename(path), len(rows), report)
        return {"source": os.path.basename(path), "pages": len(pages),
                "chunks": len(rows), "watermark": report.to_dict()}

    # ---------------- 纯文本入库 ----------------
    def ingest_text(self, text: str, role_id: str, source: str,
                    title: str = "", doc_type: str = "text") -> int:
        records = [{"embed_text": text[:200], "display_text": text,
                    "doc_type": doc_type, "meta": {"title": title}}]
        self._ensure()
        rows = self._rows_from_records(records, role_id, source)
        if rows:
            self.milvus.insert(self.collection, rows)
            redis_conn.mark_ingested(role_id, [source])
        return len(rows)


_ingest: Optional[IngestService] = None


def get_ingest_service() -> IngestService:
    global _ingest
    if _ingest is None:
        _ingest = IngestService()
    return _ingest
