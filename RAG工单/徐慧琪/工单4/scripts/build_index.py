# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""一键建库。用法：python scripts/build_index.py [baseline_03|full_04] [--reset]

``--reset``（RC-2 重建安全）：先把三个 collection 清空再建。分块改动会改变
chunk_id，不清空直接 upsert 会让新旧块共存并污染检索。执行前请自行备份
``data/qdrant``（如 ``cp -r data/qdrant data/qdrant_bak_rc2``）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag04.config import get_settings, ensure_dirs        # noqa: E402
from rag04.obs.logging import setup_logging                # noqa: E402
from rag04.pipeline import build_all                        # noqa: E402


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reset = "--reset" in sys.argv
    mode = args[0] if args else "full_04"
    s = get_settings(mode)
    ensure_dirs(s)
    log = setup_logging(s, f"rag04.build.{mode}")
    # setup_logging 只给 "rag04.build.<mode>" 装了处理器，而 rag04.pipeline /
    # rag04.ingest.* 的 logger 属另一分支：默认无处理器 → INFO 全程丢弃、
    # WARNING 只进控制台不落盘（实测全量建库日志里只剩首尾两行）。把父 logger
    # "rag04" 指向同一组处理器，进度与逐页/逐图告警才会写入同一日志文件。
    # 建库 logger 自身 propagate=False（setup_logging 设置），不会重复输出。
    parent = logging.getLogger("rag04")
    for h in log.handlers:
        if h not in parent.handlers:
            parent.addHandler(h)
    parent.setLevel(logging.INFO)
    parent.propagate = False

    log.info("=" * 60)
    log.info("开始建库 | 模式=%s | 语料=%s | reset=%s", mode, s.corpus, reset)
    log.info("=" * 60)

    stats = build_all(s, reset=reset)
    total = 0
    for st in stats:
        log.info(
            "[%s] 文本%d（入库侧剔除样板%d）表格%d 图区%d 分块%d 用时%.1fs 告警%d",
            st.doc_id, st.n_text, st.n_text_boilerplate, st.n_table, st.n_figure,
            st.n_chunk, st.seconds, len(st.warnings),
        )
        total += st.n_chunk
    log.info("建库完成，总块数 %d", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
