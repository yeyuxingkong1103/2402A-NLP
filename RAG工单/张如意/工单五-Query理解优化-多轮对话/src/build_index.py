# -*- coding: utf-8 -*-
"""
工单05 建索引脚本：把《招股说明书1》《招股说明书2》同时入库
工单编号：人工智能NLP-RAG-Query理解优化任务

为什么必须两份文档一起入库：
  工单05 的 5 轮对话是跨文档的——前 3 轮问武汉兴图新科（招股说明书1），
  第 4 轮切换到武汉力源信息（招股说明书2），第 5 轮继续在文档 2 里查组织结构图。
  如果只索引一份文档，第 4/5 轮的跨文档检索必然失败。

建索引内容：
  · 向量库（BGE 嵌入，collection=prospectus，与 rag_core.api 服务共用）
  · BM25 倒排索引（config.INDEX_DIR/bm25.pkl）
  · 结构分块（structure 策略）+ 表格块（工单03 能力）+ 图像语义块（工单04 能力）

运行：
    python build_index.py                # 增量/覆盖建索引（默认 rebuild）
    python build_index.py --no-rebuild   # 保留已有向量库（追加）
    python build_index.py --no-images    # 跳过图像解析（更快，但第 5 轮图表题会受影响）
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# 项目根目录加入 sys.path（src 的上级的上级 = 工单作业/）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core.pipeline import PRESETS, Pipeline   # noqa: E402

from wo05_common import (                         # noqa: E402
    COLLECTION, DOC_NAMES, PDF_PATHS, RESULT_DIR, check_data_files,
    print_llm_usage, write_json,
)


def build(rebuild: bool = True, with_images: bool = True,
          collection: str = COLLECTION, verbose: bool = True) -> dict:
    """执行建索引，返回统计信息。"""
    check_data_files()                                   # 先检查 PDF 素材

    cfg = PRESETS["wo05_multiturn"]
    if not with_images:
        cfg = replace(cfg, with_images=False)

    print("=" * 78)
    print("工单05 知识库构建（多轮对话需要跨文档检索）")
    print("=" * 78)
    for p, n in zip(PDF_PATHS, DOC_NAMES):
        print(f"  · 《{n}》← {p}")
    print(f"  · 集合名：{collection}    分块策略：{cfg.chunk_strategy}")
    print(f"  · 表格解析：{'开' if cfg.with_tables else '关'}    "
          f"图像解析：{'开' if cfg.with_images else '关'}")
    print("-" * 78)

    pipeline = Pipeline(cfg, collection=collection)
    stats = pipeline.build_index(
        pdf_paths=[Path(p) for p in PDF_PATHS],
        doc_names=DOC_NAMES,
        rebuild=rebuild,
        verbose=verbose,
    )

    print("-" * 78)
    print(f"索引完成：{stats['n_chunks']} 个块 / 向量 {stats['n_vectors']} 条 / "
          f"BM25 文档 {stats['n_bm25_docs']} 条，耗时 {stats['build_seconds']}s")
    print(f"块类型分布：{stats['chunk_types']}")
    print("提示：第 4/5 轮跨文档问题依赖《招股说明书2》已入库。")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="工单05 建索引（招股说明书1 + 2）")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="不重建向量库（追加写入，调试用）")
    ap.add_argument("--no-images", action="store_true",
                    help="跳过图像多模态解析（更快，但组织结构图类问题会受影响）")
    ap.add_argument("--collection", default=COLLECTION, help="向量库集合名")
    args = ap.parse_args()

    try:
        stats = build(rebuild=not args.no_rebuild, with_images=not args.no_images,
                      collection=args.collection)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        return 2
    except Exception as e:                                # 容错：打印后以非零码退出
        print(f"\n[错误] 建索引失败：{type(e).__name__}: {e}")
        return 1

    out = write_json(RESULT_DIR / "index_stats.json", stats)
    print(f"统计信息已保存：{out}")
    print_llm_usage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
