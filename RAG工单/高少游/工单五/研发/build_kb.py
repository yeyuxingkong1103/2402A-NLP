# -*- coding: utf-8 -*-
"""知识库构建入口。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

用法：
    python build_kb.py            # 增量/全量构建（解析 PDF → 切片 → 向量化 → 落盘）
    python build_kb.py --rebuild  # 强制重建
"""
from __future__ import annotations

import argparse
import sys
import time

from src import config
from src.knowledge_base import build_from_scratch


def main() -> int:
    ap = argparse.ArgumentParser(description="构建招股说明书知识库（多轮问答版）")
    ap.add_argument("--rebuild", action="store_true", help="强制重建（覆盖已有索引）")
    ap.add_argument("--max-pages", type=int, default=None, help="每份 PDF 最多解析页数（调试用）")
    args = ap.parse_args()

    if config.META_PATH.exists() and not args.rebuild:
        import json
        meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
        print(f"知识库已存在（{meta['chunks']} 块，{meta['built_at']}）。"
              f"如需重建请加 --rebuild。")
        return 0

    t0 = time.time()
    print("=" * 68)
    print("【构建知识库】招股说明书1（兴图新科） + 招股说明书2（力源信息）")
    print("=" * 68)
    meta = build_from_scratch(max_pages=args.max_pages, verbose=True)
    print("-" * 68)
    print(f"构建完成，用时 {time.time() - t0:.1f}s")
    print(f"  块数: {meta['chunks']}  维度: {meta['dim']}  模型: {meta['model']}")
    print(f"  类型分布: {meta['kinds']}")
    print(f"  落盘位置: {config.DB_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())