"""批量重算：`embed` 子命令的实现。

只处理 `status != "ok"` 的行，因此天然幂等（FR-013 / SC-009）。
"""

import argparse
import time
from datetime import datetime

from . import (
    EXIT_OK,
    F_ANSWER_ID,
    F_ASKED_AT,
    F_QUESTION,
    F_STATUS,
    STATUS_OK,
)
from . import service, store

__all__ = ["cmd_embed"]


def cmd_embed(args: argparse.Namespace) -> int:
    """批量向量化尚未完成的记录。**幂等**：只处理 `status != "ok"` 的行。"""

    files = store.list_files(args.date)
    if not files:
        scope = args.date.strftime("%Y%m%d") if args.date else "全部"
        print("没有留存文件可处理（范围：%s）。" % scope)
        return EXIT_OK

    pending = 0
    for _path, _lineno, record in store.iter_records(files):
        if record.get(F_STATUS) != STATUS_OK:
            pending += 1

    print("待处理 %d 条（共 %d 个文件）" % (pending, len(files)))

    if pending == 0:
        print("没有需要处理的记录 —— 幂等，重复运行不会重算。")
        return EXIT_OK

    if args.dry_run:
        limit = min(pending, args.limit) if args.limit else pending
        print("--dry-run：将处理 %d 条，不写任何文件。" % limit)
        return EXIT_OK

    if args.limit:
        pending = min(pending, args.limit)

    # ⚠️ 首次调用会加载权重（7–11 秒）。先说清楚，否则这 7 秒像是卡死。
    print("正在加载 BGE-M3 权重（约 2.3 GB，首次约 10 秒）…", flush=True)
    started = time.time()
    done = 0

    for path in files:
        state = {"seen": 0, "limit": pending}

        def transform(lineno: int, record: dict) -> dict:
            """未完成的行重算；已完成的行原样返回。

            返回**原对象**或新对象都可以 —— `rewrite_records` 靠序列化结果
            判断这一行到底变没变（见 store.py 的说明）。
            """
            if record.get(F_STATUS) == STATUS_OK:
                return record
            if state["seen"] >= state["limit"]:
                return record
            state["seen"] += 1
            return _reembed(record)

        changed = store.rewrite_records(path, transform)
        done += state["seen"]
        if changed:
            print("  %s：更新 %d 行" % (path.name, changed))
        if done >= pending:
            break

    print("完成：处理 %d 条，耗时 %.1f s。" % (done, time.time() - started))
    return EXIT_OK


def _reembed(record: dict) -> dict:
    """重算一条记录的向量，**保留原 answer_id 与 asked_at**。

    保留这两项是关键：重新向量化改变的是"这个问题的向量是多少"，
    不是"这是哪一次提问、什么时候问的"。若让 `capture_and_embed` 自己生成
    新的时间戳，一次批量重跑就会把全部历史记录的提问时间抹成"今天" ——
    而那是**无法从产物中恢复**的信息。
    """

    question = record.get(F_QUESTION) or ""
    answer_id = record.get(F_ANSWER_ID) or ""
    raw = record.get(F_ASKED_AT)

    try:
        asked_at = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        # 时间戳坏了，无法安全重算 —— 返回原记录（它仍然是 failed，
        # 会继续留在队列里，而不是被静默"修好"）。
        return record

    return service.capture_and_embed(question, answer_id, asked_at)
