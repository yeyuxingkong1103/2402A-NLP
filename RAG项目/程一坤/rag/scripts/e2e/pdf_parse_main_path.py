# -*- coding: utf-8 -*-
"""PDF 解析主路径验收：真实 PDF → MinerU 主分支 → 清洗 → 切块。

与 `tests/test_pdf_parser.py` 的区别：这里**真的调用远程 MinerU**（消耗额度），
用真实 PDF 跑完整条"解析 → 清洗 → 切块"链路，并把过程证据（batch_id、轮询状态、
页数、条号序列、清洗删了哪些行）全部打出来。

用法：
    python scripts/e2e/pdf_parse_main_path.py                       # 默认样本
    python scripts/e2e/pdf_parse_main_path.py --pdf jm_labor_law.pdf
    python scripts/e2e/pdf_parse_main_path.py --out-dir .pdf_runs   # 指定产物目录

退出码：0 = 主路径跑通；1 = MinerU 未返回完整结果（主路径未验证通过）；2 = 前置不满足。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# 项目根 = scripts/e2e/ 向上两层
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_project_env  # noqa: E402

# 除 _env 剔除的 4 个外，all_proxy 同样会劫持外网调用
for _name in ("all_proxy", "ALL_PROXY"):
    os.environ.pop(_name, None)

load_project_env(PROJECT_ROOT)

from app.core.config import settings  # noqa: E402
from app.ingest.chunker import chunk_document  # noqa: E402
from app.ingest.parser import parse_document  # noqa: E402
from app.models.mineru import build_mineru_client_from_settings  # noqa: E402
from app.models.qwen_vl import build_qwen_vl_client_from_settings  # noqa: E402

PDF_DIR = PROJECT_ROOT / "data" / "pdf_samples"

DEFAULT_PDF = "jm_labor_contract_law.pdf"
DEFAULT_TITLE = "中华人民共和国劳动合同法"


def line(char: str = "=", width: int = 78) -> str:
    return char * width


def resolve_pdf(name: str) -> Path:
    """接受"样本名"或"任意路径"，都统一返回绝对路径。"""
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()
    return PDF_DIR / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF 解析主路径（真实 MinerU）验收")
    parser.add_argument(
        "pdf_positional",
        nargs="?",
        help="兼容旧用法：PDF 样本名（等价于 --pdf）",
    )
    parser.add_argument("--pdf", default=None, help=f"PDF 样本名或路径（默认 {DEFAULT_PDF}）")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="文档标题（切块用）")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="产物目录（默认写在系统临时目录，跑完可整目录删除）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    name = args.pdf or args.pdf_positional or DEFAULT_PDF
    pdf_path = resolve_pdf(name)
    if not pdf_path.is_file():
        print(f"❌ PDF 不存在：{pdf_path}")
        print(f"   可用样本：{sorted(p.name for p in PDF_DIR.glob('*.pdf'))}")
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="pdf_parse_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    size = pdf_path.stat().st_size
    print(line())
    print(f"目标 PDF：{pdf_path.name}  大小={size:,} B")
    print(
        f"MinerU base={settings.mineru_api_base_url}  key长度={len(settings.mineru_api_key)}"
        f"  language={settings.mineru_language}  is_ocr={settings.mineru_is_ocr}"
    )
    print(f"轮询间隔={settings.mineru_poll_interval_seconds}s  上限={settings.mineru_max_poll_seconds}s")
    print(f"产物目录：{out_dir}")
    print(line())

    mineru = build_mineru_client_from_settings()
    qwen = build_qwen_vl_client_from_settings()

    # ---------- 1) MinerU 原始解析 ----------
    print("\n[1] MinerU 原始解析（未清洗）")
    started = time.monotonic()
    raw = mineru.parse(pdf_path)
    mineru_elapsed = time.monotonic() - started

    trace = mineru.last_trace
    print(f"    batch_id      = {trace.get('batch_id')}")
    print(f"    轮询次数      = {trace.get('polls')}  最终状态 = {trace.get('final_state')}")
    states = [e.get("state") for e in trace.get("events", []) if e.get("step") == "poll"]
    print(f"    状态流转      = {' -> '.join(str(s) for s in states) if states else '(无)'}")
    for event in trace.get("events", []):
        if event.get("step") in ("upload", "download_zip"):
            print(f"    {event['step']:<12}  = {event['bytes']:,} B")
    print(f"    complete      = {raw['complete']}   error={raw['error'] or '(空)'}")
    print(f"    page_count    = {raw['page_count']}")
    print(f"    原始正文字符数 = {len(raw['content']):,}")
    print(f"    解析耗时      = {mineru_elapsed:.1f}s")

    if not raw["complete"]:
        print("\n❌ MinerU 未返回完整结果，主路径验证失败")
        return 1

    (out_dir / f"{pdf_path.stem}.mineru_raw.md").write_text(raw["content"], encoding="utf-8")

    # ---------- 2) parse_document（含清洗） ----------
    print("\n[2] parse_document（MinerU 优先 → 清洗）")
    t0 = time.monotonic()
    parsed = parse_document(pdf_path, mineru, qwen)
    parse_elapsed = time.monotonic() - t0
    print(f"    走上分支      = {parsed.parser_name}")
    print(f"    content_format= {parsed.content_format}")
    print(f"    page_count    = {parsed.page_count}")
    print(f"    清洗后字符数   = {len(parsed.content):,}")
    print(f"    Qwen-VL 是否被调用 = {qwen.last_trace.get('page_count') is not None}")
    print(f"    耗时          = {parse_elapsed:.1f}s")

    cleaned_path = out_dir / f"{pdf_path.stem}.cleaned.txt"
    cleaned_path.write_text(parsed.content, encoding="utf-8")

    print("\n    正文前 300 字：")
    print("    " + parsed.content[:300].replace("\n", "\n    "))

    # ---------- 3) 切块 ----------
    print("\n[3] chunk_document 切块")
    document_id = "pdf-" + pdf_path.stem
    chunks = chunk_document(parsed, document_title=args.title, document_id=document_id)
    parents = [c for c in chunks if c.chunk_level == "parent"]
    children = [c for c in chunks if c.chunk_level == "child"]
    items = [c for c in children if c.item_no]
    paragraphs = [c for c in children if not c.item_no]
    print(f"    总块数        = {len(chunks)}")
    print(f"    条（父块）     = {len(parents)}")
    print(f"    款/项（子块）  = {len(children)}  （其中识别到项 = {len(items)}，纯款 = {len(paragraphs)}）")

    article_nos = [c.article_no for c in parents]
    print(f"    条号序列（前 12）= {article_nos[:12]}")
    print(f"    条号序列（末 6） = {article_nos[-6:]}")
    print(f"    有 paragraph_no 的子块 = {sum(1 for c in children if c.paragraph_no)}")
    print(f"    有 item_no 的子块      = {sum(1 for c in children if c.item_no)}")

    # ---------- 4) 章节标题存活检查 ----------
    print("\n[4] 章节标题短行存活检查（cleaner 专查）")
    heading_probe = [
        "第一章", "第二章", "第三章", "第四章", "第五章", "第六章", "第七章",
        "第八章", "第九章", "总则", "附则",
    ]
    raw_content = raw["content"]
    cleaned_content = parsed.content
    for heading in heading_probe:
        in_raw = raw_content.count(heading)
        in_cleaned = cleaned_content.count(heading)
        if in_raw:
            mark = "✔" if in_cleaned >= in_raw else "✗ 被删"
            print(f"    {heading:<6} 原始={in_raw:>2} 清洗后={in_cleaned:>2}  {mark}")

    # 清洗前后行数对比（看清洗到底删了多少行）
    raw_lines = [l.strip() for l in raw_content.split("\n") if l.strip()]
    clean_lines = [l.strip() for l in cleaned_content.split("\n") if l.strip()]
    print(
        f"\n    非空行：原始 {len(raw_lines)} → 清洗后 {len(clean_lines)}"
        f"（删了 {len(raw_lines) - len(clean_lines)} 行）"
    )
    removed = [l for l in raw_lines if l not in set(clean_lines)]
    print("    被删行样例（最多 15 条）：")
    for text in removed[:15]:
        print(f"      - {text[:70]}")

    # ---------- 5) 落盘 ----------
    summary = {
        "pdf": pdf_path.name,
        "size_bytes": size,
        "parser_name": parsed.parser_name,
        "page_count": parsed.page_count,
        "raw_chars": len(raw_content),
        "cleaned_chars": len(cleaned_content),
        "mineru_elapsed_seconds": round(mineru_elapsed, 2),
        "batch_id": trace.get("batch_id"),
        "poll_states": states,
        "chunks": {
            "total": len(chunks),
            "parents": len(parents),
            "children": len(children),
            "children_with_item": len(items),
            "children_plain_paragraph": len(paragraphs),
        },
        "article_numbers": article_nos,
    }
    (out_dir / f"{pdf_path.stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n落盘：{cleaned_path}")
    print(f"       {out_dir / f'{pdf_path.stem}.summary.json'}")
    print(line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
