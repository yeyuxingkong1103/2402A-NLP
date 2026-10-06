"""重排环节诊断（批次 11 任务 3，只诊断不改代码）。

对 confuse-046 / confuse-049 / asof-031 三题打印候选池全明细：
chunk_key / 法规+条号 / 向量分 / BM25 分 / RRF 分 / 重排分 / 是否 golden / 文本长度。

四个检查点：
  1) 传给重排模型的文本是什么（service.py:252 → candidate.content，纯正文）
  2) 最终排序按重排分还是混入 RRF 分（fusion.py:_rank_score 代码 + 实测）
  3) golden 条在候选池里的重排分排名与文本长度（是否长条文被截断）
  4) 同父块多子块是否占重排名额（父块归并在重排之后，service.py:182）

用法：python evaluation/diagnose_rerank.py [--questions confuse-046 confuse-049 asof-031]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from run_eval import prepare_env, canonical_title, title_matches, article_matches  # noqa: E402

prepare_env()

EVAL_SET = PROJECT_ROOT / "data" / "evaluation" / "eval_set_v1.jsonl"


def load_eval_items() -> dict[str, dict]:
    return {
        obj["id"]: obj
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip() and (obj := json.loads(line))
    }


def is_golden(item, golden: list[dict]) -> bool:
    for gold in golden:
        if title_matches(canonical_title(gold["law"]), item.document_title) and article_matches(
            gold["article"], item.article_number
        ):
            return True
    return False


def gold_label(item, golden: list[dict]) -> str:
    for gold in golden:
        if title_matches(canonical_title(gold["law"]), item.document_title) and article_matches(
            gold["article"], item.article_number
        ):
            return f"{gold['law']}第{gold['article']}条"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", nargs="*", default=["confuse-046", "confuse-049", "asof-031"])
    args = parser.parse_args()

    eval_items = load_eval_items()
    # 父块归并两函数在批次 18 已拆到 app.retrieval.parent_collapse（原在 fusion）
    from app.retrieval.fusion import fuse_results
    from app.retrieval.parent_collapse import load_keyword_parent_items
    from app.retrieval.filters import build_filter_expression
    from app.retrieval.assembly import build_default_retrieval_service

    service = build_default_retrieval_service()

    for qid in args.questions:
        item = eval_items[qid]
        question = item["question"]
        golden = item.get("golden") or []
        as_of = item.get("as_of_date")
        print("=" * 110)
        print(f"【{qid}】{question}")
        if as_of:
            print(f"  as_of = {as_of}")
        print(f"  golden = {[g['law'] + '第' + g['article'] + '条' for g in golden]}")

        filter_expr = build_filter_expression(
            as_of_date=as_of, jurisdiction="中国大陆", document_types=None, only_current=False
        )

        # 向量召回（父块）
        vector_articles = service.vector_retriever.retrieve(
            question, recall_limit=20, rerank_top_n=0, filter_expr=filter_expr
        )
        from app.retrieval.fusion import RankedItem

        vector_results = [
            RankedItem(
                chunk_key=art.chunk_key,
                score=art.recall_score,
                source="vector",
                sources=("vector",),
                parent_chunk_key=getattr(art, "parent_chunk_key", None) or art.chunk_key,
                content=art.content,
                document_title=art.document_title,
                article_number=art.article_number,
                vector_score=art.recall_score,
                recall_score=art.recall_score,
            )
            for art in vector_articles
        ]

        # 关键词召回
        keyword_results = []
        if service.keyword_searcher and service.session_factory:
            hits = service.keyword_searcher.search(
                question, top_k=20, jurisdiction="中国大陆", as_of_date=as_of, only_current=False
            )
            keyword_results = load_keyword_parent_items(service.session_factory, hits)

        # golden 是否在两路召回池里
        for name, pool in (("向量", vector_results), ("关键词", keyword_results)):
            found = [x.chunk_key for x in pool if is_golden(x, golden)]
            print(f"  [{name}路] 召回 {len(pool)} 条，golden 命中: {found or '无'}")

        # RRF 融合
        fused = fuse_results([vector_results, keyword_results])
        print(f"  [融合] 共 {len(fused)} 条（去重前 向量{len(vector_results)} + 关键词{len(keyword_results)}）")

        # 同 content 重复占位统计（同父块多子块）
        content_seen: dict[str, int] = {}
        for x in fused:
            key = (x.content or "")[:50]
            content_seen[key] = content_seen.get(key, 0) + 1
        dup = {k: v for k, v in content_seen.items() if v > 1}
        print(f"  [占位检查] 融合列表中同正文重复: {sum(v - 1 for v in dup.values())} 个多余名额"
              f"（{len(dup)} 组）")

        # 批次 12-A 新链路：融合 → 父块归并 → 截断送重排
        from app.retrieval.parent_collapse import collapse_parent_items

        premerged = collapse_parent_items(fused, len(fused))
        rerank_candidates = premerged[:20]
        print(f"  [归并] 父块级候选 {len(premerged)} 条 → 取前 {len(rerank_candidates)} 送重排；"
              f"重排输入 = 纯 content"
              f"（长度 {min(len(x.content or '') for x in rerank_candidates)}~"
              f"{max(len(x.content or '') for x in rerank_candidates)} 字）")

        # 重排
        ranked = service.reranker.rerank(
            question, [c.content or "" for c in rerank_candidates], top_n=len(rerank_candidates)
        )

        # 表头
        print(f"  {'融合#':>4} {'重排#':>4} {'chunk_key':<20} {'法规+条号':<30} "
              f"{'向量分':>7} {'BM25':>7} {'RRF':>7} {'重排分':>7} {'字数':>5} golden")
        fused_order = {c.chunk_key: i for i, c in enumerate(rerank_candidates)}
        rows = []
        for r_idx, cand in enumerate(ranked):
            c = rerank_candidates[cand.index]
            rows.append((fused_order[c.chunk_key], r_idx, c, cand.score))
        rows.sort()
        for f_idx, r_idx, c, r_score in rows:
            gl = gold_label(c, golden)
            mark = f" <-- GOLDEN {gl}" if gl else ""
            law_art = f"{(c.document_title or '')[:12]}·{c.article_number or '?'}"
            print(f"  {f_idx + 1:>4} {r_idx + 1:>4} {c.chunk_key[:18]:<20} {law_art:<26} "
                  f"{c.vector_score if c.vector_score is not None else -1:>7.4f} "
                  f"{c.keyword_score if c.keyword_score is not None else -1:>7.2f} "
                  f"{c.fusion_score:>7.4f} {r_score:>7.4f} {len(c.content or ''):>5}{mark}")

        # golden 重排分排名
        for f_idx, r_idx, c, r_score in rows:
            if gold_label(c, golden):
                print(f"  >> golden「{gold_label(c, golden)}」融合第{f_idx + 1}名 → 重排第{r_idx + 1}名"
                      f"（重排分 {r_score:.4f}，正文 {len(c.content or '')} 字）")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
