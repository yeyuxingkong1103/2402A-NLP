"""在导出的 PDF 文本中按关键词检索，打印命中上下文（开发期核对答案用）。

用法:
    python _tools/search_pdf.py <dump.txt> <关键词1> [关键词2 ...]
    python _tools/search_pdf.py <dump.txt> @terms.txt      # 从文件读取关键词（每行一个）
    python _tools/search_pdf.py <dump.txt> @terms.txt -C 3 # 上下文各 3 行
"""
from __future__ import annotations

import re
import sys

PAGE_RE = re.compile(r"\f===PAGE (\d+)===")


def load_pages(path: str) -> list[tuple[int, list[str]]]:
    pages: list[tuple[int, list[str]]] = []
    current: list[str] | None = None
    number = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            match = PAGE_RE.match(line)
            if match:
                if current is not None:
                    pages.append((number, current))
                number = int(match.group(1))
                current = []
            elif current is not None:
                current.append(line.rstrip("\n"))
    if current is not None:
        pages.append((number, current))
    return pages


def main() -> int:
    path = sys.argv[1]
    args = sys.argv[2:]
    context = 3
    if "-C" in args:
        index = args.index("-C")
        context = int(args[index + 1])
        args = args[:index] + args[index + 2 :]

    terms: list[str] = []
    for arg in args:
        if arg.startswith("@"):
            with open(arg[1:], encoding="utf-8") as fh:
                terms.extend(t.strip() for t in fh if t.strip())
        else:
            terms.append(arg)

    pages = load_pages(path)
    print(f"# loaded {len(pages)} pages from {path}")
    for term in terms:
        hits = 0
        print(f"\n{'=' * 70}\n### TERM: {term}\n{'=' * 70}")
        for number, lines in pages:
            for i, line in enumerate(lines):
                if term in line:
                    hits += 1
                    if hits > 12:
                        break
                    lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                    print(f"--- page {number} (line {i}) ---")
                    for j in range(lo, hi):
                        mark = ">>" if j == i else "  "
                        print(f"{mark} {lines[j]}")
            if hits > 12:
                break
        if hits == 0:
            print("(no hit)")
        elif hits > 12:
            print(f"... (more than 12 hits, truncated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
