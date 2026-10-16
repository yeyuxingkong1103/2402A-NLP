# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

补齐源码头部的工单编号注释：``研发/scripts/补齐工单头注释.py``。

共享约定 §4.2 要求「所有源码文件头部注释必须含本工单编号」。
本工单大量复用 工单7/工单10 的既有文件，其中一部分原本**没有**任何 `工单编号` 字样，
需要在文件头补一行（不能动逻辑，也不能破坏 `# -*- coding: utf-8 -*-` 必须在头两行的约定）。

    pwsh -NoProfile -File run_py.ps1 研发/scripts/补齐工单头注释.py --dry-run
    pwsh -NoProfile -File run_py.ps1 研发/scripts/补齐工单头注释.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"
_HEADER = f"# 工单编号：{WORK_ORDER}\n"
_NOTE = "# （本文件复用自 工单7/工单10 基座；工单17 只补头部编号，未改动其逻辑）\n"

_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 补齐工单头注释")
    ap.add_argument("--root", default=str(_ROOT / "研发"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root)
    fixed, skipped = [], []
    for path in sorted(root.rglob("*.py")):
        if "基座快照" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skipped.append(str(path.relative_to(_ROOT)))
            continue
        if "工单编号" in text:
            continue
        lines = text.splitlines(keepends=True)
        insert_at = 0
        # 保留 `# -*- coding: utf-8 -*-` 必须在第 1 或第 2 行的约定
        if lines and "coding" in lines[0] and "coding:" in lines[0].replace("coding=", "coding:"):
            insert_at = 1
        if not args.dry_run:
            lines.insert(insert_at, _HEADER)
            lines.insert(insert_at + 1, _NOTE)
            path.write_text("".join(lines), encoding="utf-8")
        fixed.append(str(path.relative_to(_ROOT)))

    print(f"{'（DRY RUN）' if args.dry_run else ''}已补齐 {len(fixed)} 个文件：")
    for name in fixed:
        print(f"  - {name}")
    if skipped:
        print(f"跳过 {len(skipped)} 个（读取失败）：{skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
