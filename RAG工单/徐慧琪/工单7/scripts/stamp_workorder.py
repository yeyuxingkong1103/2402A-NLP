# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
工单07 · 为工程内全部 Python 文件补齐工单编号注释（需求硬性要求6）

要求原文：「每个 Python 文件头部必须有  # 工单编号：人工智能NLP-RAG-功能测试及评估」

本工程大量文件复用自上游工单 01~06，其头部标注的是上游工单编号。为**同时满足
"头部含本工单编号"与"保留上游出处可追溯"**，本脚本在每个文件头插入两行：

    # 工单编号：人工智能NLP-RAG-功能测试及评估
    # （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）

插入位置：紧跟在 `# -*- coding: ... -*-` 或 shebang 之后，**位于任何 docstring 与
`from __future__` 之前**——注释不改变 Python 语义，`from __future__` 仍居首个语句位。

安全性：
  · 幂等——已含本工单编号的文件跳过；
  · 逐文件 `ast.parse` 校验，任一文件语法不过则**回滚该文件**并报错；
  · 不删改任何既有内容，只插入注释行。

用法
----
    python scripts/stamp_workorder.py --dry-run    # 只列出将要修改的文件
    python scripts/stamp_workorder.py              # 实际写入
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TICKET = "人工智能NLP-RAG-功能测试及评估"
LINE1 = f"# 工单编号：{TICKET}\n"
LINE2 = "# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）\n"

SKIP_DIRS = {"__pycache__", ".git", "_stale", "_scratch", ".pytest_cache", ".ruff_cache"}
# 这些文件由上游工单原样带入且与本工单交付无关，同样补注释（保持工程一致性），
# 但若其头部已有**其他工单**编号，仍按本脚本插入而不删除原编号（保留可追溯性）。


def _insert_pos(lines: list[str]) -> int:
    """返回插入行号：跳过 shebang 与 coding 声明。"""
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    if i < len(lines) and "coding" in lines[i] and lines[i].lstrip().startswith("#"):
        i += 1
    return i


def _needs_stamp(text: str) -> bool:
    return TICKET not in "".join(text.splitlines(keepends=True)[:10])


def main() -> int:
    ap = argparse.ArgumentParser(description="为全部 .py 补齐工单编号注释")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = skipped = failed = 0
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, ROOT)
            try:
                text = io.open(path, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError) as exc:
                print(f"  [跳过] {rel}：{exc}")
                skipped += 1
                continue
            if not _needs_stamp(text):
                skipped += 1
                continue

            lines = text.splitlines(keepends=True)
            pos = _insert_pos(lines)
            new_text = "".join(lines[:pos]) + LINE1 + LINE2 + "".join(lines[pos:])

            try:
                ast.parse(new_text)
            except SyntaxError as exc:
                print(f"  [失败] {rel}：插入后语法错误 {exc}")
                failed += 1
                continue

            if args.dry_run:
                print(f"  [将改] {rel}")
            else:
                io.open(path, "w", encoding="utf-8", newline="").write(new_text)
                print(f"  [已改] {rel}")
            changed += 1

    verb = "将修改" if args.dry_run else "已修改"
    print(f"\n[汇总] {verb} {changed} 个，跳过 {skipped} 个，失败 {failed} 个")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
