# -*- coding: utf-8 -*-
"""
知识图谱构建：从年报 chunk 抽取实体关系 → 实体归一化 → 社区检测 → 社区摘要
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

流程（对应 Graph RAG 索引阶段的 ①~⑤ 步）：
    ① 读取 prepare_corpus.py 产出的文本块
    ② 逐块调用 LLM 抽取实体与关系（profile="optimized"：金融领域类型体系 +
       few-shot + 二次补漏 gleaning，见 rag_core.graph_rag 的 PromptProfile）
    ③ 实体归一化：别名消解、指代还原、同向关系合并加权（由 KnowledgeGraph 完成）
    ④ 社区检测：Louvain 把图谱切成若干主题簇（全局检索的基础）
    ⑤ 社区摘要：为每个簇生成 150 字摘要，构成「全局检索」的语料

产物：
    data/graph/kg.json                图谱本体（实体/关系/社区/摘要/统计）
    results/graph_stats.json          统计报告（实体数/关系数/类型分布/社区数）
    data/graph/extractions_optimized.jsonl  抽取缓存（断点续跑用）

关于「贵」的处理 —— 抽取是整条链路最贵的一步（9 份年报 ≈ 上万块），因此：
    · rag_core.build_knowledge_graph 内置 JSONL 增量缓存，每抽完一块立即 flush；
    · 中断后重新运行本脚本会自动跳过已完成的块（resume=True）；
    · 社区摘要同样缓存到 kg.json，重跑不会重复调用 LLM。

运行：
    python 工单08-GraphRAG金融问答/src/build_graph.py
    python .../build_graph.py --max-chunks 20 --no-summaries   # 小样本试跑
    python .../build_graph.py --no-resume                     # 忽略缓存全量重抽
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import config                                     # noqa: E402
from rag_core.graph_rag import build_knowledge_graph            # noqa: E402

from prepare_corpus import (                                    # noqa: E402
    CORPUS_CHUNKS, RESULTS_DIR, load_chunks, load_graph,
)

# 图谱产物（与 rag_core.api 的 GET /api/graph、Neo4j 导入脚本共用同一份文件）
GRAPH_PATH = config.GRAPH_DIR / "kg.json"
# 抽取缓存（JSONL，一行一个 chunk 的抽取结果，断点续跑的关键）
EXTRACT_CACHE = config.GRAPH_DIR / "extractions_optimized.jsonl"
STATS_PATH = RESULTS_DIR / "graph_stats.json"


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _doc_of_chunk(chunk_id: str) -> str:
    """chunk_id 形如「平安银行股份有限公司2019年年度报告-tex-00042」，取文档名部分。"""
    parts = chunk_id.rsplit("-", 2)
    return parts[0] if len(parts) == 3 else chunk_id


def extraction_stats(cache_path: Path = EXTRACT_CACHE) -> dict:
    """
    直接读抽取缓存 JSONL 统计「每份文档抽出了多少实体/关系」。

    之所以从缓存统计而不是从图谱统计：图谱做了实体归一化与关系合并，
    已经无法回溯到单篇文档；而缓存保留了每一次抽取的原始结果。
    """
    per_doc: dict[str, dict] = defaultdict(
        lambda: {"chunks": 0, "entities_raw": 0, "relations_raw": 0})
    total_lines = 0
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except Exception:
                continue
            total_lines += 1
            d = per_doc[_doc_of_chunk(o.get("chunk_id", ""))]
            data = o.get("data") or {}
            d["chunks"] += 1
            d["entities_raw"] += len(data.get("entities") or [])
            d["relations_raw"] += len(data.get("relations") or [])
    return {
        "n_extracted_chunks": total_lines,
        "per_doc": {k: dict(v) for k, v in sorted(per_doc.items())},
    }


def build_stats(kg, extra: dict | None = None) -> dict:
    """汇总图谱统计：规模、类型分布、社区、核心实体。"""
    stats = dict(kg.stats())                      # n_entities / n_relations / 类型分布
    ent_types = Counter(e.type for e in kg.entities.values())

    # 节点度数分布（度 = 该实体在合并后的关系图中的连通数）
    degrees = sorted((e.degree for e in kg.entities.values()), reverse=True)
    top_entities = sorted(kg.entities.values(), key=lambda e: -e.degree)[:30]

    stats.update({
        "work_order": "人工智能NLP-RAG-基于Graph RAG 实现金融问答",
        "graph_file": str(GRAPH_PATH),
        "entity_type_distribution": dict(ent_types.most_common()),
        "relation_type_distribution": dict(
            Counter(r.type for r in getattr(kg, "relations_merged", [])).most_common()),
        "n_entities": len(kg.entities),
        "n_relations": len(getattr(kg, "relations_merged", [])),
        "n_communities": len(kg.communities),
        "degree_max": degrees[0] if degrees else 0,
        "degree_avg": round(sum(degrees) / max(len(degrees), 1), 2),
        "isolated_entities": sum(1 for d in degrees if d == 0),
        "top_entities": [
            {"name": e.name, "type": e.type, "degree": e.degree,
             "description": e.description[:120]}
            for e in top_entities
        ],
        "communities": [
            {"id": cid, "size": len(members), "members": members[:20],
             "summary": (getattr(kg, "community_summaries", {}) or {}).get(cid, "")[:300]}
            for cid, members in sorted(kg.communities.items(),
                                       key=lambda x: -len(x[1]))
        ],
    })
    if extra:
        stats.update(extra)
    return stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="工单08 知识图谱构建（实体关系抽取 + 社区检测 + 社区摘要）")
    ap.add_argument("--profile", default="optimized",
                    choices=["baseline", "optimized"],
                    help="抽取档位：optimized=金融类型体系+few-shot+二次补漏（默认）")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="最多抽取多少块（调试用；已缓存的块不计入）")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略抽取缓存，全量重抽（慎用，很贵）")
    ap.add_argument("--no-summaries", action="store_true",
                    help="跳过 LLM 社区摘要（只做社区检测，省调用）")
    ap.add_argument("--communities-top", type=int, default=20,
                    help="最多为多少个社区生成摘要，默认 20")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Louvain 分辨率，越大社区越细，默认 1.0")
    args = ap.parse_args()

    # ---- ① 语料 ----
    chunks = load_chunks(CORPUS_CHUNKS)
    print(f"载入语料：{len(chunks)} 个 chunk，来自 "
          f"{len({c.doc for c in chunks})} 份文档")

    # ---- ② 抽取（带增量缓存 / 断点续跑）----
    t0 = time.perf_counter()
    kg = build_knowledge_graph(
        chunks, profile=args.profile,
        cache_path=EXTRACT_CACHE,
        max_chunks=args.max_chunks,
        resume=not args.no_resume,
        verbose=True,
    )
    extract_seconds = round(time.perf_counter() - t0, 2)

    # ---- ③ 社区检测 ----
    print(f"\n社区检测（Louvain, resolution={args.resolution}）…")
    communities = kg.detect_communities(resolution=args.resolution)
    print(f"  得到 {len(communities)} 个社区，"
          f"最大社区 {max((len(v) for v in communities.values()), default=0)} 个实体")

    # ---- ④ 社区摘要（全局检索的语料；很贵，因此默认只做前 N 个）----
    if args.no_summaries:
        kg.community_summaries = {}
        print("  已跳过社区摘要（--no-summaries）")
    else:
        print(f"生成社区摘要（前 {args.communities_top} 个社区）…")
        kg.summarize_communities(top_n=args.communities_top)
        print(f"  完成 {len(kg.community_summaries)} 条社区摘要")

    # ---- ⑤ 落盘 ----
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    kg.save(GRAPH_PATH)
    stats = build_stats(kg, extra={
        "profile": args.profile,
        "extraction_seconds": extract_seconds,
        "extraction_stats": extraction_stats(EXTRACT_CACHE),
        "extract_cache": str(EXTRACT_CACHE),
    })
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    # 再存一份到工单08 自有目录，便于单工单打包交付
    mirror = Path(__file__).resolve().parents[1] / "data" / "graph" / "kg.json"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(GRAPH_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"图谱构建完成：{stats['n_entities']} 实体 / {stats['n_relations']} 关系 / "
          f"{stats['n_communities']} 社区")
    print(f"  实体类型分布：{stats['entity_type_distribution']}")
    print(f"  关系类型分布：{stats['relation_type_distribution']}")
    print(f"  图谱文件：{GRAPH_PATH}")
    print(f"  统计报告：{STATS_PATH}")
    print("  下一步：python 工单08-GraphRAG金融问答/src/visualize_graph.py")


if __name__ == "__main__":
    main()
