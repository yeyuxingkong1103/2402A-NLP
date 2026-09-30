# -*- coding: utf-8 -*-
"""PDF 解析兜底分支验收：假 MinerU 触发 + **真实 Qwen-VL** OCR。

R2 口径（用户裁决）：MinerU 侧是**人为注入**的假客户端（不联网），只负责把链路推进
兜底分支；Qwen-VL 侧是**真实调用** —— 对真实 PDF 的真实渲染页做 OCR，输出真实正文。
这样验的是"兜底分支本身能不能出可用正文"，而不是"MinerU 会不会失败"。

用法：
    python scripts/e2e/pdf_parse_qwen_fallback.py                          # 默认样本
    python scripts/e2e/pdf_parse_qwen_fallback.py --pdf jm_labor_law.pdf
    python scripts/e2e/pdf_parse_qwen_fallback.py --out-dir .pdf_runs      # 指定产物目录

退出码：0 = 兜底分支跑通；1 = 未走上兜底分支或 Qwen-VL 未出正文；2 = 前置不满足。
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
from app.models.qwen_vl import build_qwen_vl_client_from_settings  # noqa: E402

PDF_DIR = PROJECT_ROOT / "data" / "pdf_samples"

DEFAULT_PDF = "jm_lc_implement_regulation.pdf"
DEFAULT_TITLE = "中华人民共和国劳动合同法实施条例"

STUB_ERROR = "R2 人为注入：模拟 MinerU 返回不完整"


class IncompleteMineruStub:
    """假 MinerU：只为触发兜底，不做任何网络调用。"""

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, source_path: Path) -> dict:
        self.calls += 1
        return {
            "content": "",
            "page_count": 0,
            "complete": False,
            "error": STUB_ERROR,
        }


def line(char: str = "=", width: int = 78) -> str:
    return char * width


def resolve_pdf(name: str) -> Path:
    """接受"样本名"或"任意路径"，都统一返回绝对路径。"""
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()
    return PDF_DIR / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF 兜底分支（真实 Qwen-VL OCR）验收")
    parser.add_argument("pdf_positional", nargs="?", help="兼容旧用法：PDF 样本名（等价于 --pdf）")
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

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="pdf_qwen_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(line())
    print(f"目标 PDF：{pdf_path.name}  大小={pdf_path.stat().st_size:,} B")
    print(f"Qwen-VL base={settings.qwen_vl_api_base_url}")
    print(
        f"           model={settings.qwen_vl_model}  超时={settings.qwen_vl_timeout_seconds}s"
        f"  每请求页数上限={settings.qwen_vl_max_pages_per_request}"
        f"  请求体上限={settings.qwen_vl_max_request_bytes} B"
    )
    print(f"           key长度={len(settings.qwen_vl_api_key)}")
    print(f"产物目录：{out_dir}")
    print(line())

    mineru = IncompleteMineruStub()
    qwen = build_qwen_vl_client_from_settings()

    # 先单独跑一次 Qwen-VL，拿到页面渲染与识别的真实统计
    print("\n[1] Qwen-VL 渲染 + OCR（真实调用）")
    started = time.monotonic()
    text = qwen.parse(pdf_path)
    elapsed = time.monotonic() - started
    trace = qwen.last_trace

    print(f"    渲染页数      = {trace.get('page_count')}")
    print(f"    请求批次数    = {len(trace.get('batches') or [])}")
    for index, batch in enumerate(trace.get("batches") or [], start=1):
        print(f"      批 {index}: 图 {batch['images']} 张 → 识别 {batch['chars']} 字")
    print(f"    识别总字符数  = {len(text):,}")
    print(f"    耗时          = {elapsed:.1f}s")
    if not text:
        print(f"    ❌ 失败原因：{qwen.last_error}")
        return 1

    print("\n    识别正文前 300 字：")
    print("    " + text[:300].replace("\n", "\n    "))

    (out_dir / f"{pdf_path.stem}.qwen_raw.txt").write_text(text, encoding="utf-8")

    # ---------- 2) 走 parse_document 完整链路 ----------
    print("\n[2] parse_document（假 MinerU 不完整 → Qwen-VL 兜底）")
    qwen2 = build_qwen_vl_client_from_settings()
    parsed = parse_document(pdf_path, mineru, qwen2)
    print(f"    MinerU 被调用次数 = {mineru.calls}")
    print(f"    走上分支          = {parsed.parser_name}")
    print(f"    content_format    = {parsed.content_format}")
    print(f"    page_count        = {parsed.page_count}")
    print(f"    清洗后字符数      = {len(parsed.content):,}")

    if parsed.parser_name != "qwen-vl":
        print("\n❌ 没有走上兜底分支，本次验收无效")
        return 1

    (out_dir / f"{pdf_path.stem}.qwen_cleaned.txt").write_text(parsed.content, encoding="utf-8")

    # ---------- 3) 切块 ----------
    print("\n[3] chunk_document 切块")
    document_id = "pdf-qwen-" + pdf_path.stem
    chunks = chunk_document(parsed, document_title=args.title, document_id=document_id)
    parents = [c for c in chunks if c.chunk_level == "parent"]
    children = [c for c in chunks if c.chunk_level == "child"]
    items = [c for c in children if c.item_no]
    print(f"    总块数        = {len(chunks)}")
    print(f"    条（父块）     = {len(parents)}")
    print(
        f"    款/项（子块）  = {len(children)}"
        f"  （其中识别到项 = {len(items)}，纯款 = {len(children) - len(items)}）"
    )
    print(f"    条号序列      = {[c.article_no for c in parents][:15]}")

    # ---------- 4) 章节标题存活 ----------
    print("\n[4] 章节标题短行存活检查（cleaner 专查）")
    for heading in ("第一章", "第二章", "总则", "附则"):
        raw_count = text.count(heading)
        clean_count = parsed.content.count(heading)
        if raw_count or clean_count:
            mark = "✔" if clean_count >= raw_count else "✗ 被删"
            print(f"    {heading:<6} Qwen原始={raw_count:>2} 清洗后={clean_count:>2}  {mark}")

    summary = {
        "pdf": pdf_path.name,
        "branch": parsed.parser_name,
        "mineru_stub_calls": mineru.calls,
        "mineru_stub_error": STUB_ERROR,
        "qwen_model": settings.qwen_vl_model,
        "qwen_pages": trace.get("page_count"),
        "qwen_batches": len(trace.get("batches") or []),
        "qwen_elapsed_seconds": round(elapsed, 2),
        "qwen_raw_chars": len(text),
        "cleaned_chars": len(parsed.content),
        "chunks": {
            "total": len(chunks),
            "parents": len(parents),
            "children": len(children),
            "children_with_item": len(items),
        },
        "article_numbers": [c.article_no for c in parents],
    }
    (out_dir / f"{pdf_path.stem}.qwen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n落盘：{out_dir / f'{pdf_path.stem}.qwen_cleaned.txt'}")
    print(f"       {out_dir / f'{pdf_path.stem}.qwen_summary.json'}")
    print(line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
