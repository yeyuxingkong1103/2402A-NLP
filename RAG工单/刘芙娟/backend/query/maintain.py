"""维护命令：`purge` 与 `show` 的实现。

一个破坏性（清理）、一个只读（查看），放在一起是因为它们都服务于
"事后处理已有留存"这件事，且都不涉及编码。

⚠️ `purge` 是**破坏性操作**，其安全约束写在函数文档里而不是注释里 ——
   默认预演、需显式 `--yes`、缺目标时什么都不做。这三条都是实现在代码里的
   行为，不是建议。
"""

import argparse
import math
import sys

from . import (
    EXIT_ARGS,
    EXIT_DATA,
    EXIT_OK,
    F_ANSWER_ID,
    F_ASKED_AT,
    F_ERROR,
    F_QUESTION,
    F_STATUS,
    F_VECTOR,
)
from . import store

__all__ = ["cmd_purge", "cmd_show"]


def cmd_purge(args: argparse.Namespace) -> int:
    """按日期清理留存。**默认预演，需 --yes 才真删。**"""

    if args.date and args.before:
        print("错误：--date 与 --before 互斥，只能给一个。", file=sys.stderr)
        return EXIT_ARGS
    if not args.date and not args.before:
        # 不传日期就什么都不删（store 层也是这个行为）—— 宁可什么都不做，
        # 也不要因为参数漏传把整个留存目录清空。
        print("错误：必须给出 --date 或 --before。", file=sys.stderr)
        print("（本命令不会在缺少目标的情况下删除任何东西）", file=sys.stderr)
        return EXIT_ARGS

    dry_run = not args.yes
    targets = store.purge_files(day=args.date, before=args.before, dry_run=dry_run)

    if not targets:
        print("没有匹配的留存文件。")
        return EXIT_OK

    total_lines = sum(t[1] for t in targets)
    total_bytes = sum(t[2] for t in targets)

    print("预演（未删除任何文件）：" if dry_run else "已删除：")
    for path, lines, size in targets:
        print("  %s  %d 条  %.1f KB" % (path.name, lines, size / 1024))
    print("  合计 %d 个文件、%d 条、%.1f KB"
          % (len(targets), total_lines, total_bytes / 1024))

    if dry_run:
        scope = ("--date %s" % args.date.strftime("%Y%m%d") if args.date
                 else "--before %s" % args.before.strftime("%Y%m%d"))
        print("\n这是预演。确认无误后加 --yes 真正删除：")
        print("    python -m backend.query_embed purge %s --yes" % scope)

    return EXIT_OK


def cmd_show(args: argparse.Namespace) -> int:
    """打印某条记录的可读摘要。**只读。**

    不假设文件名与内容日期一致，因此扫描全部文件而非按日期直取 ——
    本契约只保证同一文件内的顺序，不保证文件名与内部时间戳永远对应
    （手工搬动过的文件不受此保证，contracts/store.md §6）。
    """

    target = args.answer_id.strip()
    found = None
    for path, lineno, record in store.iter_records(store.list_files()):
        if record.get(F_ANSWER_ID) == target:
            found = (path, lineno, record)
            break

    if found is None:
        print("未找到 answer_id = %s 的记录。" % target, file=sys.stderr)
        return EXIT_DATA

    path, lineno, record = found
    vector = record.get(F_VECTOR)

    print("answer_id : %s" % record.get(F_ANSWER_ID))
    print("question  : %s" % record.get(F_QUESTION))
    print("asked_at  : %s" % record.get(F_ASKED_AT))
    print("status    : %s" % record.get(F_STATUS))
    if record.get(F_ERROR):
        print("error     : %s" % record.get(F_ERROR))
    print("位置      : %s 第 %d 行" % (path.name, lineno))

    fingerprint_block = record.get("embedding") or {}
    print("口径      : %s" % fingerprint_block.get("rule_version", "<空>"))
    print("dim       : %s" % fingerprint_block.get("dim", "<空>"))

    if isinstance(vector, list) and vector:
        # 只给摘要：1024 个数字对排障没有用，反而把关键信息挤出屏幕。
        head = ", ".join("%.6f" % float(x) for x in vector[:5])
        norm = math.sqrt(sum(float(x) * float(x) for x in vector))
        print("vector    : %d 维，前 5 维 [%s]，L2 范数 %.8f"
              % (len(vector), head, norm))
    else:
        print("vector    : 无（尚未向量化）")

    return EXIT_OK
