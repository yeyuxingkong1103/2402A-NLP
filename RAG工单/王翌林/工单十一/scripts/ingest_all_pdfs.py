# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
scripts/ingest_all_pdfs.py —— 工单三多文档 PDF 入库脚本

职责（见 docs/03_多文档知识库方案.md §四）：
  1. 扫描附件目录下所有 PDF，按自然顺序排序
  2. 文件名自动识别：
     - 已知文件名（招股说明书1.pdf / 招股说明书2.pdf）→ 用对应公司名
     - 其他文件按"招股说明书{i}"顺序命名（i 从 1 开始）
  3. 对每个 PDF 调用 src.pdf_parser_v3.parse_pdf_v3 完成解析
  4. 写出：
     - data/parsed_v3/{doc_name}_text.json   （文本 + 文本 chunk + 表格 chunk 摘要）
     - data/tables/{doc_name}_tables.json     （结构化表格 + table-text）
  5. 汇总写入 data/parsed_v3/ingest_summary.json（多文档注册表 + 索引摘要）

用法：
  python scripts/ingest_all_pdfs.py
  python scripts/ingest_all_pdfs.py --attach-dir /home/dabaie/code/工单/附件 --skip-existing
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# 让脚本不依赖 PYTHONPATH 也能跑
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_parser_v3 import (  # noqa: E402
    parse_pdf_v3, write_text_json, write_tables_json, make_doc_id,
)

# 工单三：已知 PDF 文件名 → 公司名映射（缺失则按顺序命名 + 尝试从首页文本推断）
KNOWN_PDFS = {
    "招股说明书1.pdf": "武汉兴图新科电子股份有限公司",
    "招股说明书1-无水印.pdf": "武汉兴图新科电子股份有限公司",
    "招股说明书2.pdf": "武汉力源信息技术股份有限公司",
}

# 工单三：招股说明书文件名匹配模式（招股说明书1.pdf / 招股说明书1-无水印.pdf / 招股说明书10.pdf）
_ZHAOSHU_RE = __import__("re").compile(r"招股说明书\s*(\d+)")


def _natural_sort_key(path: Path) -> tuple:
    """工单三：自然排序键（招股说明书2.pdf 排在 招股说明书10.pdf 之前）"""
    name = path.name
    parts = []
    for tok in __import__("re").finditer(r"\d+|\D+", name):
        s = tok.group()
        parts.append((int(s), "") if s.isdigit() else (0, s))
    return tuple(parts)


def _normalize_doc_name(stem: str) -> str:
    """工单三：把 stem 归一化为 招股说明书N

    招股说明书1 → 招股说明书1
    招股说明书1-无水印 → 招股说明书1
    招股说明书2 → 招股说明书2
    其他 → stem 原样
    """
    m = _ZHAOSHU_RE.search(stem)
    if m:
        return f"招股说明书{m.group(1)}"
    return stem


def scan_pdfs(attach_dir: Path) -> list[Path]:
    """工单三：扫描附件目录下所有 PDF，按自然顺序排序

    优先返回 stem 含"招股说明书"的 PDF；若无，则返回全部 PDF（按顺序命名兜底）。
    """
    all_pdfs = sorted(
        [p for p in attach_dir.glob("*.pdf") if p.is_file()],
        key=_natural_sort_key,
    )
    zhaoshu = [p for p in all_pdfs if "招股说明书" in p.stem]
    return zhaoshu if zhaoshu else all_pdfs


def _dedupe_by_doc_name(pdf_paths: list[Path]) -> list[Path]:
    """工单三：同一 doc_name 多个 PDF 时去重

    优先级：无水印 > 有水印 > 文件小 > 排序靠前
    （招股说明书1.pdf 与 招股说明书1-无水印.pdf 同 doc_name，保留 -无水印 版本）
    """
    seen: dict[str, Path] = {}
    for p in pdf_paths:
        dname = _normalize_doc_name(p.stem)
        if dname not in seen:
            seen[dname] = p
            continue
        cur = seen[dname]
        # 优先保留无水印版本
        cur_nowm = "无水印" in cur.stem
        new_nowm = "无水印" in p.stem
        if new_nowm and not cur_nowm:
            seen[dname] = p
        elif cur_nowm and not new_nowm:
            pass  # 保留当前
        else:
            # 都无水印或都有水印 → 保留更小的（更快解析）
            if p.stat().st_size < cur.stat().st_size:
                seen[dname] = p
    return list(seen.values())


def guess_company(pdf_path: Path, idx: int) -> str:
    """工单三：从已知映射或顺序命名猜测公司名

    未知文件按"招股说明书{i}"命名，公司名留 None（下游会用 doc_name 兜底）。
    """
    if pdf_path.name in KNOWN_PDFS:
        return KNOWN_PDFS[pdf_path.name]
    return ""


def doc_name_for(pdf_path: Path, idx: int) -> str:
    """工单三：文档名生成

    - stem 含"招股说明书N"（含 -无水印 变体）→ 归一化为 "招股说明书N"
    - 未知文件 → "招股说明书{i}"（i 从 1 开始，按扫描顺序）
    """
    stem = pdf_path.stem
    if "招股说明书" in stem:
        return _normalize_doc_name(stem)
    return f"招股说明书{idx}"


def ingest_all(
    attach_dir: Path,
    out_dir: Path,
    tables_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """工单三：扫描并入库所有 PDF

    Returns:
        summary dict（写回 data/parsed_v3/ingest_summary.json）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _dedupe_by_doc_name(scan_pdfs(attach_dir))
    # 重新按 doc_name 排序，保证入库顺序稳定（招股说明书1 → 招股说明书2）
    pdfs = sorted(pdfs, key=lambda p: _natural_sort_key(p))
    if not pdfs:
        logger.error(f"[ingest] 附件目录无 PDF: {attach_dir}")
        return {"ingested": [], "skipped": [], "failed": [], "total": 0}

    logger.info(f"[ingest] 去重后入库 {len(pdfs)} 个 PDF：{[p.name for p in pdfs]}")

    ingested, skipped, failed = [], [], []
    for idx, pdf in enumerate(pdfs, 1):
        doc_name = doc_name_for(pdf, idx)
        company = guess_company(pdf, idx)
        text_out = out_dir / f"{doc_name}_text.json"
        tables_out = tables_dir / f"{doc_name}_tables.json"
        doc_id = make_doc_id(str(pdf))

        # 去重：text+tables 都已存在则跳过（除非 --no-skip-existing）
        if skip_existing and text_out.exists() and tables_out.exists():
            logger.info(f"[ingest] 跳过（已存在）: {doc_name} ({pdf.name})")
            skipped.append({"doc_name": doc_name, "doc_id": doc_id,
                             "file": pdf.name, "reason": "exists"})
            continue

        logger.info(f"[ingest] 入库 #{idx}: {doc_name} ({pdf.name})")
        try:
            parsed = parse_pdf_v3(
                str(pdf), doc_name=doc_name, company=company,
                doc_type="招股说明书", use_camelot_fallback=False,
            )
            write_text_json(parsed, str(text_out))
            write_tables_json(parsed, str(tables_out))
            ingested.append({
                "doc_id": parsed["doc_id"],
                "doc_name": doc_name,
                "company": company,
                "filename": pdf.name,
                "file_hash": parsed["file_hash"],
                "total_pages": parsed["total_pages"],
                "total_chars": parsed["total_chars"],
                "text_chunk_count": parsed["text_chunk_count"],
                "table_count": parsed["table_count"],
                "raw_table_count": parsed["raw_table_count"],
                "parse_seconds": parsed["parse_seconds"],
                "text_json": str(text_out.relative_to(PROJECT_ROOT)),
                "tables_json": str(tables_out.relative_to(PROJECT_ROOT)),
                "errors": parsed["errors"],
            })
        except Exception as e:
            logger.exception(f"[ingest] 入库失败 {pdf.name}: {e}")
            failed.append({"doc_name": doc_name, "doc_id": doc_id,
                           "file": pdf.name, "error": str(e)})

    summary = {
        "ingest_started_at": datetime.now().isoformat(timespec="seconds"),
        "attach_dir": str(attach_dir),
        "total_pdfs_scanned": len(pdfs),
        "ingested_count": len(ingested),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "ingested": ingested,
        "skipped": skipped,
        "failed": failed,
        # 全量 chunk 索引摘要（供后续 Step 4 检索模块直接读用）
        "all_text_chunks_files": [
            str((out_dir / f"{d['doc_name']}_text.json").relative_to(PROJECT_ROOT))
            for d in (ingested + skipped)
        ],
        "all_table_chunks_files": [
            str((tables_dir / f"{d['doc_name']}_tables.json").relative_to(PROJECT_ROOT))
            for d in (ingested + skipped)
        ],
    }
    summary_path = out_dir / "ingest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    logger.info(
        f"[ingest] 完成：ingested={len(ingested)} skipped={len(skipped)} "
        f"failed={len(failed)} → {summary_path}"
    )
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(description="工单三：多文档 PDF 入库 CLI")
    parser.add_argument("--attach-dir", default=str(PROJECT_ROOT / "附件"),
                        help="附件目录（默认 项目根/附件）")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "parsed_v3"),
                        help="文本 JSON 输出目录")
    parser.add_argument("--tables-dir", default=str(PROJECT_ROOT / "data" / "tables"),
                        help="表格 JSON 输出目录")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="已存在则跳过（默认开启）")
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false",
                        help="强制重新解析（覆盖已有产出）")
    args = parser.parse_args()

    summary = ingest_all(
        Path(args.attach_dir), Path(args.out_dir),
        Path(args.tables_dir), skip_existing=args.skip_existing,
    )
    # 控制台打印摘要
    print(json.dumps({
        "total_scanned": summary["total_pdfs_scanned"],
        "ingested": summary["ingested_count"],
        "skipped": summary["skipped_count"],
        "failed": summary["failed_count"],
        "ingested_detail": [
            {k: v for k, v in d.items()
             if k in ("doc_name", "company", "total_pages",
                      "text_chunk_count", "table_count", "parse_seconds")}
            for d in summary["ingested"]
        ],
        "skipped_detail": summary["skipped"],
        "failed_detail": summary["failed"],
    }, ensure_ascii=False, indent=2))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
