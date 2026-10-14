# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
CLI：知识图谱构建

用法：
  python build_kg.py --layer1              # 仅第1层（词典+正则，秒级）
  python build_kg.py --layer1 --limit 500  # 抽样调试
  python build_kg.py --all                 # 第1层 + 第2层（LLM，耗时较长）
  python build_kg.py --all --resume        # 断点续抽
  python build_kg.py --stats               # 只查看现有图谱统计
  python build_kg.py --report              # 生成 docs/03-知识图谱构建.md
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

from src import bootstrap  # noqa: F401
from src import config, kg_build
from src.kg_store import KnowledgeGraph


def _print_stats(st: dict) -> None:
    print("=" * 60)
    print(f"节点 {st['n_nodes']}    边 {st['n_edges']}    "
          f"覆盖文档 {st['n_docs']}    带provenance的边 {st['edges_with_provenance']}")
    print("-" * 60)
    print("实体类型分布：")
    for t, n in sorted(st["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {t:<14} {n}")
    print("关系类型分布：")
    for t, n in sorted(st["by_relation"].items(), key=lambda x: -x[1]):
        print(f"  {t:<14} {n}")
    print("=" * 60)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单08 知识图谱构建")
    ap.add_argument("--layer1", action="store_true", help="只跑第1层")
    ap.add_argument("--all", action="store_true", help="第1层 + 第2层")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 块")
    ap.add_argument("--resume", action="store_true", help="第2层断点续抽")
    ap.add_argument("--stats", action="store_true", help="只打印现有图谱统计")
    ap.add_argument("--report", action="store_true",
                    help="生成 docs/03-知识图谱构建.md 的数据章节")
    args = ap.parse_args()

    if args.report:
        from src.kg_report import write_report
        path = write_report()
        print(f"报告已写入 {path}")
        return 0

    if args.stats:
        # 优先读轻量的 stats 缓存；缺失时回退为加载图谱再统计。
        if os.path.isfile(config.KG_STATS):
            with io.open(config.KG_STATS, encoding="utf-8") as f:
                st = json.load(f)
        else:
            st = KnowledgeGraph.load().stats()
        _print_stats(st)
        return 0

    if not (args.layer1 or args.all):
        ap.print_help()
        return 2

    chunks = kg_build.load_chunks()
    print(f"语料 {len(chunks)} 块")

    kg = kg_build.build_layer1(chunks, limit=args.limit)
    if args.all:
        kg_build.build_layer2(kg, chunks, resume=args.resume, limit=args.limit)
    result = kg_build.save_all(kg)
    _print_stats(result["stats"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
