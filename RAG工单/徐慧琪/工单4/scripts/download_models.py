# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""一键下载 CLIP 与 reranker 权重。失败不阻断——reranker 可降级。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag04.config import get_settings, ensure_dirs          # noqa: E402
from rag04.ingest.clip_dl import ensure_clip, ensure_reranker, ModelDownloadError  # noqa: E402
from rag04.obs.logging import setup_logging                  # noqa: E402


def main() -> int:
    s = get_settings()
    ensure_dirs(s)
    log = setup_logging(s, "rag04.download")

    try:
        p = ensure_clip(s)
        log.info("CLIP 就绪：%s", p)
    except ModelDownloadError as e:
        log.error("CLIP 下载失败，图像跨模态检索将不可用：%s", e)
        return 1

    try:
        p = ensure_reranker(s)
        log.info("Reranker 就绪：%s", p)
    except ModelDownloadError as e:
        log.warning("Reranker 下载失败，将降级为启发式重排：%s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
