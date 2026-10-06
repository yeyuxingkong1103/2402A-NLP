"""把 PDF 全文导出为带页码标记的纯文本，供开发期检索核对使用。

用法:
    python _tools/dump_pdf.py <pdf> <out.txt>
输出格式:
    \f===PAGE n===\n<该页文本>
"""
from __future__ import annotations

import sys


def main() -> int:
    pdf, out = sys.argv[1], sys.argv[2]
    import pymupdf  # PyMuPDF >= 1.24

    with pymupdf.open(pdf) as doc:
        total = doc.page_count
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# source: {pdf}\n# pages: {total}\n")
            for index in range(total):
                text = doc.load_page(index).get_text("text")
                fh.write(f"\f===PAGE {index + 1}===\n{text}\n")
                if (index + 1) % 50 == 0:
                    print(f"... {index + 1}/{total}", flush=True)
    print(f"done: {total} pages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
