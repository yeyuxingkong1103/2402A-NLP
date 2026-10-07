# 工单编号：人工智能NLP-RAG-功能测试及评估
# 工单07 - 功能测试及评估
"""
工单07 · ccf 语料取证工具：按短语定位到「文档 + 印刷页码 + 原文」。

  python scripts/lookup_ccf.py "逾越者联盟"
  python scripts/lookup_ccf.py "拨备覆盖率" --doc payh --context 120
  python scripts/lookup_ccf.py "绿色金融" --limit 8

【为什么需要它】工单07 的产出物里，每道题都要写**人工核实的参考答案**与
**证据页**。页码一旦写错，"检索结果"与"评估结果"就全不可信 —— 而这份语料里
页码是最脏的部分（招商银行/邮储银行/中国太保三份都在「财务报告」节重新编号）。
所以出题时不靠记忆、不靠模型，一律用本工具回到 PDF 上**逐条取证**。

【为什么用 search_for 而不是全文解析】pymupdf 的 `page.search_for()` 是底层
文本检索，265 页只要零点几秒；而走 parse_pdf 每份要 30 秒以上（还要建表）。
取证这件事只关心"这句话在哪一页"，用不着分块与表格解析。

【页码口径与入库完全一致】页码取自 `DocProfile.printed_label`（页脚实读优先），
所以这里看到的页码就是检索结果里会显示的页码 —— 两边同一套规则。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                          # noqa: E402
from app.core.doc_profiles import doc_profiles_for_ccf                   # noqa: E402


def block_text(b: dict) -> str:
    return "".join(s["text"] for line in b.get("lines", [])
                   for s in line.get("spans", [])).strip()


def snippet(page, needle: str, width: int) -> str:
    """把命中所在的那一**行**截出来，前后各留 width//2 个字。"""
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            t = "".join(s["text"] for s in line.get("spans", [])).strip()
            if needle in t:
                i = t.index(needle)
                lo = max(0, i - width // 2)
                return ("…" if lo else "") + t[lo:lo + width] + \
                       ("…" if lo + width < len(t) else "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="工单07 · ccf 语料取证")
    ap.add_argument("needle", help="要定位的短语/数字")
    ap.add_argument("--doc", default=None, help="只在某一份里找（按 key，如 payh）")
    ap.add_argument("--limit", type=int, default=20, help="每份最多打印多少处")
    ap.add_argument("--context", type=int, default=90, help="snippet 宽度（字）")
    args = ap.parse_args()

    import pymupdf

    profs = doc_profiles_for_ccf()
    keys = [args.doc] if args.doc else list(profs)
    total = 0
    for key in keys:
        prof = profs[key]
        pdf = settings.data_path / "raw" / prof.doc_name
        if not pdf.exists():
            print(f"[跳过] {prof.doc_name}：文件不在 {pdf}")
            continue
        doc = pymupdf.open(pdf)
        shown = 0
        try:
            for i in range(doc.page_count):
                page = doc[i]
                if not page.search_for(args.needle):
                    continue
                label = prof.printed_label(
                    page, page.get_text("dict")["blocks"], prof.label(i))
                if shown == 0:
                    print(f"\n=== {prof.doc_name}（key={key}）===")
                print(f"  p.{label:<5} (idx {i:>3})  {snippet(page, args.needle, args.context)}")
                shown += 1
                total += 1
                if shown >= args.limit:
                    print(f"  …（还有更多，用 --limit 放开）")
                    break
        finally:
            doc.close()
    print(f"\n共 {total} 处命中")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
