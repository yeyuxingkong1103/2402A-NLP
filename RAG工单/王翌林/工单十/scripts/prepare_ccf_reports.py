# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
scripts/prepare_ccf_reports.py —— 工单七 ccf_competition 年报语料准备（新增）

职责：
  1. 读取附件 ccf_competition.zip（zip 内文件名为 GBK 编码，自动还原中文名）
  2. 9 份 A 股年报（2019~2021，银行/保险/券商）解压到 data/ccf_reports/ 并规范命名
  3. PyMuPDF 逐页抽取文本（年报为文本型 PDF），复用 01-06 工单的
     src.chunker.chunk_parsed_document 分块
  4. 复用 src.embedding / src.vector_store.VectorStore 将 chunk 嵌入并增量写入
     Milvus rag_chunks（doc_id = "{简称}{年份}年报"，幂等：先按 doc_id 删除旧数据）

用法：
  python scripts/prepare_ccf_reports.py                    # 解压+分块+入库
  python scripts/prepare_ccf_reports.py --skip-extract     # 跳过解压
  python scripts/prepare_ccf_reports.py --skip-ingest      # 仅解压+分块，不入 Milvus
"""
import argparse
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

# 工单七：CPU 嵌入防 OOM（沿用工单五经验）
os.environ.setdefault("RAG_EMBED_DEVICE", "cpu")
os.environ.setdefault("RAG_EMBED_BATCH_SIZE", "8")
os.environ.setdefault("RAG_EMBED_MAX_SEQ", "512")

from loguru import logger  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
ZIP_PATH = Path(os.getenv(
    "CCF_ZIP_PATH",
    "/mnt/c/Users/Lenovo/Desktop/zg6工单/RAG 工单/附件/ccf_competition.zip"))
REPORT_DIR = PROJECT_ROOT / "data" / "ccf_reports"
CHUNK_DIR = REPORT_DIR / "chunks"
REGISTRY_PATH = REPORT_DIR / "ccf_corpus.json"

# 工单七：zip 条目名形如
# ccf_competition/pdf/2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf
_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})__(.+?)__(\d{6})__(.+?)__(\d{4})年")


def decode_zip_name(raw: str) -> str:
    """工单七：zip 中文文件名还原（Python 按 cp437 解码了 GBK 字节）"""
    try:
        return raw.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def extract_pdfs(zip_path: Path, out_dir: Path) -> list:
    """工单七：解压 9 份年报 PDF 并规范命名，返回语料注册表"""
    if not zip_path.exists():
        raise FileNotFoundError(f"ccf_competition.zip 不存在: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            real = decode_zip_name(info.filename)
            if not real.lower().endswith(".pdf"):
                continue
            m = _NAME_RE.search(real)
            if not m:
                logger.warning(f"[ccf] 文件名无法解析，跳过: {real}")
                continue
            date, full_name, code, short_name, year = m.groups()
            doc_id = f"{short_name}{year}年报"
            safe_name = f"{code}_{short_name}_{year}年报.pdf"
            pdf_path = out_dir / safe_name
            if not pdf_path.exists():
                pdf_path.write_bytes(zf.read(info))
            corpus.append({
                "doc_id": doc_id, "code": code,
                "company": short_name, "full_name": full_name,
                "year": int(year), "publish_date": date,
                "filename": safe_name,
                "pdf_path": str(pdf_path.relative_to(PROJECT_ROOT)),
                "zip_entry": real,
            })
    corpus.sort(key=lambda x: (x["year"], x["code"]))
    REGISTRY_PATH.write_text(
        json.dumps({"work_order": WORK_ORDER,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "source_zip": str(zip_path), "docs": corpus},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"[ccf] 解压完成：{len(corpus)} 份年报 → {out_dir}")
    return corpus


def chunk_report(meta: dict) -> dict:
    """工单七：PyMuPDF 抽文本 → 复用工单一起 chunker 分块"""
    import fitz
    from src.chunker import chunk_parsed_document

    pdf_path = PROJECT_ROOT / meta["pdf_path"]
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, 1):
        pages.append({"page": i, "text": page.get_text()})
    doc.close()
    parsed = {"doc_id": meta["doc_id"], "filename": meta["filename"],
              "pages": pages}
    result = chunk_parsed_document(parsed)
    # 工单七：补全入库所需字段
    for ch in result["chunks"]:
        ch["doc_id"] = meta["doc_id"]
        ch.setdefault("metadata", {})
        ch["metadata"].update({"company": meta["company"],
                               "year": meta["year"], "code": meta["code"],
                               "work_order": WORK_ORDER})
    return result


def ingest_corpus(corpus: list) -> dict:
    """工单七：分块 + bge-m3 嵌入 + 写入 Milvus rag_chunks（按 doc_id 幂等）"""
    from src.embedding import get_embedder
    from src.vector_store import VectorStore

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    embedder = get_embedder()
    vs = VectorStore()
    vs.ensure_collection()
    stats = []
    for meta in corpus:
        t0 = time.time()
        result = chunk_report(meta)
        chunks = result["chunks"]
        chunk_json = CHUNK_DIR / f"{meta['doc_id']}_chunks.json"
        chunk_json.write_text(json.dumps(
            {"doc_id": meta["doc_id"], "chunks": chunks},
            ensure_ascii=False), encoding="utf-8")

        # 工单七：幂等——先删该 doc_id 旧向量再插入
        vs.delete_by_doc_id(meta["doc_id"])
        texts = [c["text"] for c in chunks]
        vectors = embedder.encode(texts, batch_size=8,
                                  show_progress_bar=False)
        inserted = vs.insert_chunks(chunks, vectors)
        cost = time.time() - t0
        logger.info(f"[ccf] {meta['doc_id']}: {inserted} chunks 入库 "
                    f"({cost:.0f}s)")
        stats.append({"doc_id": meta["doc_id"], "chunks": inserted,
                      "seconds": round(cost, 1)})
    try:
        total = vs.client.num_rows(collection_name=vs.collection) \
            if hasattr(vs.client, "num_rows") else vs.get_stats().get("num_entities")
    except Exception:
        total = "?"
    vs.close()
    return {"per_doc": stats, "rag_chunks_total": total}


def main() -> int:
    parser = argparse.ArgumentParser(description=WORK_ORDER)
    parser.add_argument("--skip-extract", action="store_true",
                        help="跳过解压，读取已有注册表")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="仅解压，不做嵌入入库")
    args = parser.parse_args()

    if args.skip_extract and REGISTRY_PATH.exists():
        corpus = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["docs"]
    else:
        corpus = extract_pdfs(ZIP_PATH, REPORT_DIR)
    print(json.dumps([{k: d[k] for k in ("doc_id", "code", "year")}
                      for d in corpus], ensure_ascii=False, indent=1))
    if args.skip_ingest:
        logger.info("[ccf] --skip-ingest，仅解压完成")
        return 0
    summary = ingest_corpus(corpus)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
