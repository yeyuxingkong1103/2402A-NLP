# -*- coding: utf-8 -*-
"""知识库入库管线：解析 → 清洗 → 分块 → 编码 → 写入 Qdrant。

分块策略按语料类型分流：
    article   法条：按「第X条」切分（法条是天然的最小语义单元）
    paragraph 叙述性文本：按段落聚合到目标长度
    auto      pdf→paragraph；txt 含 3 个以上「第X条」→article，否则 paragraph
"""
import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF
from qdrant_client import models as qm
from sqlalchemy.orm import Session

from ..core import config
from ..core.logging import get_logger
from ..models import KbDocument
from . import embed
from .embed import encode
from .qdrant_store import get_client

log = get_logger("ingest")

# 「第一条 / 第123条 / 第二十条」——注意不要匹配到「第X章/节/款/项」
# 实测语料存在两种格式，正则需同时兼容：
#   A) 条文在行首       "第一条 为保护消费者的合法权益…"
#   B) 法名与条文同行   "《中华人民共和国民法典》第一条规定，…"
ARTICLE_RE = re.compile(
    r"(?m)^[ \t　]*(?:《[^》\n]{2,40}》)?[ \t　]*"
    r"(第[一二三四五六七八九十百千万零〇0-9]{1,8}条)"
)
NOISE_RE = re.compile(r"^[·•\-—–\s]*\d{1,4}[·•\-—–\s]*$")     # 纯页码行（含「· 977 ·」式刊眉）


# ================================================================ 解析
def extract_pdf(path: Path) -> list[tuple[int, str]]:
    """抽取 PDF 文本，并剔除页眉页脚/页码噪声。"""
    doc = fitz.open(str(path))
    pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    doc.close()
    return _clean_pages(pages)


def extract_txt(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """剔除在多数页面重复出现的行（页眉/页脚）与纯页码行。"""
    if not pages:
        return pages

    from collections import Counter
    counter: Counter = Counter()
    for _, text in pages:
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            counter[line] += 1

    threshold_long = max(3, int(len(pages) * 0.5))
    threshold_short = max(2, int(len(pages) * 0.2))
    # 长行（正文）需半数以上页面重复才算页眉页脚；
    # 短行（刊头「·指 南·」、页脚）20% 出现即视为噪声
    repeated = {
        line for line, n in counter.items()
        if n >= (threshold_short if len(line) <= 12 else threshold_long)
    }

    cleaned = []
    for pno, text in pages:
        kept = [
            ln for ln in text.splitlines()
            if ln.strip() and ln.strip() not in repeated and not NOISE_RE.match(ln)
        ]
        cleaned.append((pno, "\n".join(kept)))
    return cleaned


# ================================================================ 分块
def chunk_article(text: str, title: str = "") -> list[dict]:
    """按「第X条」切分。返回 [{article_no, text}]。"""
    matches = list(ARTICLE_RE.finditer(text))
    if len(matches) < 3:
        return [{"article_no": None, "text": c} for c in chunk_paragraph(text)]

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 10:
            continue

        # 入库文本带上法律名称：检索时「民法典 第577条」这类查询更容易命中。
        # 格式 B 的原文已自带《法名》前缀，避免重复叠加。
        if title and not body.startswith("《"):
            full = f"《{title}》{body}"
        else:
            full = body

        if len(full) <= config.CHUNK_MAX_CHARS:
            chunks.append({"article_no": m.group(1), "text": full})
        else:
            # 超长条文按句再切
            for piece in _split_long(full):
                chunks.append({"article_no": m.group(1), "text": piece})
    return chunks


def chunk_paragraph(text: str, size: int | None = None,
                    overlap: int | None = None) -> list[str]:
    """按段落聚合到目标长度。"""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # 没有空行分段（如部分 PDF 抽取结果）时，退化为按行聚合
    if len(paras) <= 1:
        paras = [ln.strip() for ln in text.splitlines() if ln.strip()]

    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf}\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[max(0, size - overlap):]
        buf = p
    if buf:
        chunks.append(buf)

    return [c.strip() for c in chunks if len(c.strip()) >= 20]


def _split_long(text: str, size: int = 600) -> list[str]:
    """长文本按句号切分。"""
    sentences = re.split(r"(?<=[。；;！？])", text)
    out, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) <= size:
            buf += s
        else:
            if buf:
                out.append(buf.strip())
            buf = s
    if buf.strip():
        out.append(buf.strip())
    return [c for c in out if len(c) >= 20] or [text[:size]]


def decide_strategy(path: Path, strategy: str = "auto") -> str:
    if strategy in ("article", "paragraph"):
        return strategy
    if path.suffix.lower() == ".txt":
        return "article" if len(ARTICLE_RE.findall(extract_txt(path))) >= 3 else "paragraph"
    return "paragraph"


# ================================================================ Qdrant
def _embed_device() -> str:
    """离线入库优先用 GPU（此时 Ollama 不需要显存），约比 CPU 快 28 倍。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_collection(name: str) -> None:
    client = get_client()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": qm.VectorParams(size=config.EMBED_DIM, distance=qm.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False))
            },
        )
        log.info("创建 collection: %s", name)


def _point_id(collection: str, source: str, article_no: str | None,
              text: str) -> int:
    """稳定 ID：同一条内容重复入库时为覆盖而非追加。"""
    raw = f"{collection}|{source}|{article_no or ''}|{text}"
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16], 16)


def _upsert(collection: str, points: list[qm.PointStruct], batch: int = 64) -> int:
    client = get_client()
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i:i + batch])
    return len(points)


def _write_secondary(collection: str, chunks: list[dict], dense_vecs: list,
                     sparse_vecs: list, path: Path) -> int:
    """按 config.VECTOR_STORE 把同一批数据同步写入 Milvus。

    ⚠️ 为什么必须有这个函数（2026-09-16 修）
    ----------------------------------------
    读路径（`retrieval.hybrid_search`）会按 `VECTOR_STORE` 走 Milvus，
    而写路径原先**无条件写 Qdrant**。于是 `VECTOR_STORE=milvus` 时：

        用户 POST /api/kb/ingest -> 返回 status=ready、chunk 若干
        但文档对之后**所有检索都不可见** —— 两边都不报错，静默分裂。

    现在写路径与读路径遵守同一个开关。**以 Qdrant 为权威源**：
    Qdrant 是主写入（上面已做），Milvus 是次写入；次写入失败不回滚主写入，
    但会记 ERROR 并抛出，避免"接口报成功、实际查不到"。
    """
    store = config.VECTOR_STORE
    if store == "qdrant":
        return 0

    from . import milvus_store

    rows = []
    for chunk, dv, sv in zip(chunks, dense_vecs, sparse_vecs):
        payload = {
            "text": chunk["text"],
            "source": path.name,
            "law_name": path.stem if collection == config.COLLECTION_LEGAL else None,
            "article_no": chunk.get("article_no"),
            "page": chunk.get("page"),
        }
        pid = _point_id(collection, path.name, chunk.get("article_no"), chunk["text"])
        rows.append(milvus_store.row_from_qdrant(payload, pid, dv, sv))

    try:
        n = milvus_store.upsert_rows(collection, rows)
        milvus_store.flush(collection)
        log.info("Milvus 次写入完成 %s | %d 条", collection, n)
        return n
    except Exception as e:
        # 不静默：让调用方知道「主写入成功但 Milvus 没同步」，
        # 否则就会出现「入库报成功、Milvus 模式检索不到」的分裂。
        log.exception("Milvus 次写入失败 %s", collection)
        raise RuntimeError(
            f"已写入 Qdrant，但 Milvus 同步失败：{e}。"
            f"可稍后用 scripts/migrate_to_milvus.py 补同步。"
        ) from e


# ================================================================ 入库主流程
def build_chunks(path: Path, strategy: str = "auto",
                 ocr_mode: str = "auto") -> tuple[list[dict], str]:
    """解析单个文件为 chunk 列表。返回 (chunks, 实际策略)。

    PDF 走**分流**：文本层 PDF 用 PyMuPDF；扫描件自动转 MinerU OCR
    （见 `app/services/ocr.py` 与 `docs/09-OCR分流说明.md`）。
    `ocr_mode`: auto / force / off。
    """
    suffix = path.suffix.lower()
    title = path.stem

    if suffix == ".pdf":
        # 延迟导入：ocr 模块内部会反过来用本模块的 extract_pdf，
        # 模块级互相导入会成环。
        from . import ocr as ocr_svc
        pages, _meta = ocr_svc.parse_pdf(path, ocr=ocr_mode)
        if not pages:
            return [], "paragraph"
        strategy = "paragraph" if strategy == "auto" else strategy
        if strategy == "article":
            chunks = chunk_article("\n".join(t for _, t in pages), title)
            return chunks, strategy
        # 段落模式：逐页处理，保留页码元数据
        chunks = []
        for pno, text in pages:
            for c in chunk_paragraph(text):
                chunks.append({"article_no": None, "text": c, "page": pno})
        return chunks, "paragraph"

    if suffix in (".txt", ".md"):
        text = extract_txt(path)
        strategy = decide_strategy(path, strategy)
        if strategy == "article":
            return chunk_article(text, title), strategy
        return [{"article_no": None, "text": c} for c in chunk_paragraph(text)], strategy

    log.warning("跳过不支持的文件类型: %s", path)
    return [], "auto"


def ingest_file(path: Path, collection: str, strategy: str = "auto",
                db: Session | None = None, ocr: str = "auto") -> dict:
    """入库单个文件并记录到 kb_documents。"""
    raw = path.read_bytes()
    file_hash = hashlib.sha1(raw).hexdigest()

    doc = None
    if db is not None:
        doc = (db.query(KbDocument)
               .filter_by(collection=collection, file_hash=file_hash).first())
        if doc is None:
            doc = KbDocument(
                collection=collection, source_path=str(path), file_hash=file_hash,
                doc_type=path.suffix.lower().lstrip("."), chunk_strategy=strategy,
                status="parsing",
            )
            db.add(doc)
        else:
            doc.status = "parsing"
            doc.source_path = str(path)
        db.commit()

    try:
        chunks, used_strategy = build_chunks(path, strategy, ocr_mode=ocr)
        if not chunks:
            raise ValueError("未解析出任何内容")

        ensure_collection(collection)
        texts = [c["text"] for c in chunks]
        dense_vecs, sparse_vecs = encode(texts, device=_embed_device(), progress=True)

        points = []
        for i, (chunk, dv, sv) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
            points.append(qm.PointStruct(
                id=_point_id(collection, path.name, chunk.get("article_no"), chunk["text"]),
                vector={"dense": dv, "sparse": qm.SparseVector(**sv)},
                payload={
                    "text": chunk["text"],
                    "source": path.name,
                    "collection": collection,
                    "law_name": path.stem if collection == config.COLLECTION_LEGAL else None,
                    "article_no": chunk.get("article_no"),
                    "page": chunk.get("page"),
                    "chunk_index": i,
                },
            ))
        _upsert(collection, points)
        _write_secondary(collection, chunks, dense_vecs, sparse_vecs, path)

        if doc is not None:
            doc.status = "ready"
            doc.chunk_count = len(points)
            doc.chunk_strategy = used_strategy
            doc.error_msg = None
            db.commit()

        log.info("入库完成 %s -> %s | %d chunks (%s)",
                 path.name, collection, len(points), used_strategy)
        return {"file": path.name, "collection": collection,
                "chunks": len(points), "strategy": used_strategy, "status": "ready"}

    except Exception as e:
        log.error("入库失败 %s: %s", path.name, e)
        if doc is not None:
            doc.status = "failed"
            doc.error_msg = f"{type(e).__name__}: {str(e)[:400]}"
            db.commit()
        return {"file": path.name, "collection": collection,
                "chunks": 0, "status": "failed", "error": str(e)[:200]}


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for pat in ("*.pdf", "*.txt", "*.md"):
        files.extend(sorted(path.glob(pat)))
    return files


def ingest_path(path: str, collection: str, strategy: str = "auto",
                ocr: str = "auto") -> dict:
    """入库文件或目录（后台任务入口，自带会话）。"""
    from ..core.db import SessionLocal

    target = Path(path)
    if not target.exists():
        return {"ok": False, "error": f"路径不存在: {path}"}

    files = collect_files(target)
    if not files:
        return {"ok": False, "error": f"目录下没有可入库的文件: {path}"}

    db = SessionLocal()
    results = []
    try:
        ensure_collection(collection)
        for f in files:
            results.append(ingest_file(f, collection, strategy, db, ocr=ocr))
    finally:
        db.close()
        embed.release_gpu()      # 批量入库结束，把显存还给 Ollama

    ok = sum(1 for r in results if r["status"] == "ready")
    total_chunks = sum(r["chunks"] for r in results)
    log.info("批量入库结束 | %d/%d 文件成功 | 共 %d chunks", ok, len(files), total_chunks)
    return {"ok": True, "files": len(files), "succeeded": ok,
            "total_chunks": total_chunks, "results": results}


def delete_document(doc_id: int) -> dict:
    """删除文档：连带删除其在 Qdrant 中的向量。"""
    from ..core.db import SessionLocal

    db = SessionLocal()
    try:
        doc = db.get(KbDocument, doc_id)
        if not doc:
            return {"ok": False, "error": "文档不存在"}

        source_name = Path(doc.source_path).name if doc.source_path else ""

        client = get_client()
        if client.collection_exists(doc.collection):
            client.delete(
                collection_name=doc.collection,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                    qm.FieldCondition(key="source", match=qm.MatchValue(value=source_name))
                ])),
            )

        # 同步删除 Milvus 侧 —— 否则 `VECTOR_STORE=milvus` 时向量删不掉，
        # 已删除的文档仍会被检索到（与写入路径同一类分裂问题）。
        if config.VECTOR_STORE in ("milvus", "both") and source_name:
            from . import milvus_store
            try:
                n = milvus_store.delete_by_source(doc.collection, source_name)
                log.info("Milvus 同步删除 %s | %s | %d 条",
                         doc.collection, source_name, n)
            except Exception:
                log.exception("Milvus 同步删除失败 %s | %s", doc.collection, source_name)

        db.delete(doc)
        db.commit()
        return {"ok": True, "deleted": doc_id}
    finally:
        db.close()


def collection_stats() -> list[dict]:
    client = get_client()
    stats = []
    for c in client.get_collections().collections:
        try:
            count = client.count(c.name).count
        except Exception:
            count = -1
        stats.append({"name": c.name, "points": count})
    return stats
