# -*- coding: utf-8 -*-
"""T7 优化：清理「改动前」遗留的答案级缓存（防止旧算法结果被复用）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

为什么需要它：答案级缓存落在 ``研发/data/index/qa_runtime.sqlite3``，键里带
``PIPELINE_VERSION``（位于 ``研发/app/core/qa_engine.py``，**不在 T7 的改动范围内**）。
T7 改了检索融合/生成护栏后，若不清旧条目，命中缓存会返回**旧算法**的答案
（实测现象：题 207 命中旧缓存后引用页从 479 变成 32，citation_ok 由 True 变 False）。

本脚本按「写入时间」删除旧条目；``--keep-seconds`` 可只删 N 秒前的条目。
只操作问答侧 ``qa_runtime.sqlite3``，**绝不触碰 T4 的知识库 rag.sqlite3**。

用法：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/purge_cache.py --db 研发/data/index/qa_runtime.sqlite3
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def purge(db: Path, *, keep_seconds: float = 0.0, dry_run: bool = False) -> dict:
    """删除答案缓存中写入时间早于 ``now - keep_seconds`` 的条目。"""
    if not db.is_file():
        return {"db": str(db), "existed": False, "deleted": 0}
    connection = sqlite3.connect(str(db))
    try:
        total = connection.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
        cutoff = time.time() - max(0.0, float(keep_seconds))
        rows = connection.execute(
            "SELECT COUNT(*) FROM answer_cache WHERE created_at <= ?", (cutoff,)).fetchone()[0]
        if not dry_run and rows:
            connection.execute("DELETE FROM answer_cache WHERE created_at <= ?", (cutoff,))
            connection.execute("DELETE FROM qa_history")
            connection.commit()
        return {"db": str(db), "existed": True, "entries_before": int(total),
                "deleted": int(rows), "remaining": int(total - rows) if not dry_run else int(total),
                "dry_run": bool(dry_run)}
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T7：清理改动前的答案级缓存（问答侧库）")
    parser.add_argument("--db", default=str(ROOT / "研发" / "data" / "index" / "qa_runtime.sqlite3"))
    parser.add_argument("--keep-seconds", type=float, default=0.0, help="只删 N 秒前的条目（0=全清）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = purge(Path(args.db), keep_seconds=args.keep_seconds, dry_run=args.dry_run)
    print(f"[purge_cache] {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
