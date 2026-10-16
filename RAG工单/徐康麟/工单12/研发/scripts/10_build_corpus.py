# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""步骤1：从 data/raw 的 PDF 构建统一块表（文本/表格/图像）。

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/10_build_corpus.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag import corpus as corpus_mod          # noqa: E402
from lightrag.config import dump_effective_config, load_config  # noqa: E402
from lightrag.logging_util import get_logger, write_json         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--files", nargs="*", default=None, help="只解析指定文件名")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("build_corpus", cfg, console=True)
    log.info("开始构建语料：raw_dir=%s", cfg.path_str("raw_dir"))

    dump_effective_config(cfg, os.path.join(cfg.path_str("log_dir"), "effective_config.json"))

    t0 = time.perf_counter()
    chunks, stats = corpus_mod.build_documents(cfg.path("raw_dir"), cfg=cfg, files=args.files, logger=log)
    if not chunks:
        log.error("未产出任何块，请检查 %s 下是否有 PDF", cfg.path_str("raw_dir"))
        return 2
    stats["build_seconds"] = round(time.perf_counter() - t0, 3)
    saved = corpus_mod.save_corpus(chunks, stats, cfg=cfg)
    write_json(os.path.join(cfg.path_str("log_dir"), "corpus_stats.json"), stats)

    print(json.dumps({k: stats[k] for k in ("chunk_total", "chunk_by_type", "char_total", "build_seconds")},
                     ensure_ascii=False, indent=2))
    for k, v in saved.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
