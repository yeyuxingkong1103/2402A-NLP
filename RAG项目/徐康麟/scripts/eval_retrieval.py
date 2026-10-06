#!/usr/bin/env python3
"""检索侧评测：**不调大模型**，只比"对的条文有没有进 top-k / 排第几"。

为什么要这个工具（P8 的直接对策）：整轮评测（107 题 × 生成）要 20+ 分钟 GPU，而换重排器是
**高频小改动**。这里只做召回 + 重排打分，把同一批候选交给**多个重排器**逐一对齐比较
（apples-to-apples），几十秒就能看出谁更好；vLLM 都可以不开（省显存）。

指标（全是检索侧，和 `eval/qa_set.jsonl` 的 `sources` 对齐）：
* ``Recall@k``：期望来源出现在前 k 条里的题占比；
* ``MRR``：第一个命中期望来源的排名倒数均值（1.0 = 每次都排第一）；
* ``zero``：前 k 条里**一条期望来源都没有**的题（最该看的坏例子）。

用法（云端，无需 vLLM）：
    python scripts/eval_retrieval.py --qa-file eval/qa_set.jsonl --role lawyer \
        --ranker cosine= --ranker bge=/root/.../bge-reranker-v2-m3/snapshots/master \
        --ranker bce=/root/.../bce-reranker-base_v1 \
        --limit 5 --out eval/results/retrieval-rerank.json

注意：``cosine=`` 表示"不换重排器、沿用融合分顺序"（基线）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.schemas import SearchHit  # noqa: E402


def source_of(hit: SearchHit) -> str:
    return str(getattr(getattr(hit, "chunk", None), "source", "") or "")


def text_of(hit: SearchHit) -> str:
    return str(getattr(getattr(hit, "chunk", None), "text", "") or "")


def hit_matches(hit: SearchHit, sources: tuple[str, ...],
                articles: tuple[str, ...] = ()) -> bool:
    """这条命中算不算"对的"：来源对得上；**给了条号就还要条号对得上**。

    为什么要条号（P8 度量缺陷）：只比文件时，一部法几十个条文块随便中一个就算过，
    文件级 Recall@5 会虚高到 0.88，把"那一条没进前 k"的真问题盖住。
    """
    if not sources:
        return False
    if not any(expected in source_of(hit) for expected in sources):
        return False
    if not articles:
        return True                       # 没标条号 ⇒ 退回文件级
    body = text_of(hit)
    return any(article in body for article in articles)


def rank_of_expected(hits: list[SearchHit], sources: tuple[str, ...],
                     articles: tuple[str, ...] = ()) -> int | None:
    """期望来源（有 ``articles`` 时要求条号也中）**最早**出现的那条是第几名；没有则 ``None``。"""
    if not sources:
        return None
    for index, hit in enumerate(hits, start=1):
        if hit_matches(hit, sources, articles):
            return index
    return None


def score_ranking(rank: int | None, *, k: int) -> dict:
    """把"排名"折成三个可聚合的量。"""
    if rank is None:
        return {"hit_at_k": False, "mrr": 0.0, "zero": True}
    return {"hit_at_k": rank <= k, "mrr": 1.0 / rank, "zero": rank > k}


def summarize_rankings(rows: list[dict], *, k: int) -> dict:
    """``rows`` 是每题的 ``{"id", "rank"}``；返回聚合指标 + 坏例子清单。"""
    scored = [(row, score_ranking(row.get("rank"), k=k)) for row in rows]
    total = len(scored) or 1
    return {
        "questions": len(scored),
        "k": k,
        "recall_at_k": round(sum(1 for _, s in scored if s["hit_at_k"]) / total, 4),
        "mrr": round(sum(s["mrr"] for _, s in scored) / total, 4),
        "zero_hit": [row["id"] for row, s in scored if s["zero"]],
        "ranks": {row["id"]: row.get("rank") for row, _ in scored},
        # 选择器专项：挑中**正确条**的题数；其中"原本没进 top-k"（=0）才是净救援。
        "picked_correct": sum(1 for row, _ in scored if row.get("pick_expected") is not None),
        "picked_rescued": sum(1 for row, _ in scored if row.get("pick_expected") == 0),
        "picked_any": sum(1 for row, _ in scored if row.get("picks")),
        "picked_ranks": {row["id"]: row.get("pick_expected") for row, _ in scored
                         if row.get("pick_expected") is not None},
    }


def parse_ranker(spec: str) -> tuple[str, str]:
    name, _, model = spec.partition("=")
    return name.strip(), model.strip()


def selector_kwargs(config: dict) -> dict:
    """从"缓存配置"里取出**能传给 ``ListwiseSelector`` 的构造参数**。

    缓存键比构造参数**多一项**（``prompt`` = 提示词版本，只用于判断缓存是否作废）——
    真机踩到：直接把整个字典 `**` 展开当参数，会 `TypeError: unexpected keyword argument
    'prompt'`，整个评测在 5 秒内崩掉且不产出报告。
    """
    allowed = ("top_k", "max_candidates", "snippet_chars")
    return {key: config[key] for key in allowed if key in config}


def load_pick_cache(path: str | Path, config: dict) -> dict[str, list[str]]:
    """读挑条缓存；**只认同一套 prompt 配置**，否则返回空（宁可重问，也不混用口径）。

    缓存与 prompt 配置绑定（真机教训）：候选数/片段长度/挑条数一变，模型的**输入**就变了，
    旧的选择不再是"对同一个问题的选择"；旧格式（无配置记录）同样不认。
    任何读取异常都只当作"没有缓存"（评测照常，重问即可）。
    """
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"!! 选择器缓存读不了（当作没有缓存）：{type(exc).__name__}: {exc}", file=sys.stderr)
        return {}
    if not (isinstance(payload, dict) and "picks" in payload):
        print("!! 选择器缓存是旧格式（无配置记录）=> 忽略缓存、全部重问", file=sys.stderr)
        return {}
    if payload.get("config") != config:
        print(f"!! 选择器缓存配置不符（缓存={payload.get('config')} 本次={config}）"
              f"=> 忽略缓存、全部重问", file=sys.stderr)
        return {}
    return {str(k): [str(x) for x in v] for k, v in (payload.get("picks") or {}).items()}


def isolated_pool(pool: list[SearchHit]) -> list[SearchHit]:
    """给某个重排器用的**候选副本**。

    为什么必须有（真机踩到）：CrossEncoder 的 `rerank()` 把分数写回 hit 对象
    （`rerank_score` / `score`），候选池却是跨重排器复用的 ⇒ 排在后面的 `cosine`
    会拿上一个重排器的分排序，两臂结果一模一样（实测指纹：m3 在前时 cosine 报出的
    Recall/MRR/零命中清单与 m3 逐字相同）。副本让每个重排器只看原始候选。
    """
    return [replace(hit) for hit in pool]


def build_ranker(name: str, model: str, config):
    """``cosine`` = 不重排（用融合分）；其余走 CrossEncoder（sentence-transformers）。"""
    from legal_rag.retrieve.rerank import build_reranker

    if name in ("cosine", "none", ""):
        return build_reranker("cosine", "", config)
    return build_reranker("bge_rerank", model, config)


def main() -> int:
    parser = argparse.ArgumentParser(description="检索侧重排对比（不调大模型）")
    parser.add_argument("--qa-file", default="eval/qa_set.jsonl")
    parser.add_argument("--role", default="lawyer")
    parser.add_argument("--ranker", action="append", default=[],
                        metavar="NAME=MODEL", help="可多次；cosine= 表示基线")
    parser.add_argument("--limit", type=int, default=5, help="看前 k 条（默认 5）")
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--candidate-pool", type=int, default=0,
                        help="先截断候选池（0 = 不截断；比较重排器时建议 0）")
    parser.add_argument("--selector", action="store_true",
                        help="叠加 LLM 列表式选择器（在**法内宽候选**里挑条并置顶；需大模型可用）")
    parser.add_argument("--selector-provider", default="", help="默认取 config.llm_provider")
    parser.add_argument("--selector-model", default="", help="默认取 config.llm_model")
    parser.add_argument("--selector-top-k", type=int, default=0,
                        help="选择器最多挑几条（0 = 取 config.law_selector_top_k）")
    parser.add_argument("--selector-max-candidates", type=int, default=0,
                        help="送进 prompt 的候选上限（0 = 取 config；直接决定 TTFT）")
    parser.add_argument("--selector-snippet-chars", type=int, default=0,
                        help="每条候选截断字数（0 = 取 config）")
    parser.add_argument("--selector-trigger", default="",
                        help="always = 每题都调；low_confidence = 只在没把握时调（默认取 config）")
    parser.add_argument("--selector-cache", default="",
                        help="选择器挑条结果落盘/复用（JSON）。挑选只取决于问题+法内候选，"
                             "与被测标注无关 ⇒ 改标注后复测不必再花一次大模型调用。")
    parser.add_argument("--allow-degraded-keyword", action="store_true",
                        help="允许关键词通道未就绪也继续（默认**拒绝**，见下面的预热闸门）")
    args = parser.parse_args()

    from legal_rag.config import RagConfig
    from legal_rag.embedding.base import build_embedder
    from legal_rag.eval_set import load_qa_set
    from legal_rag.retrieve.hybrid import HybridRetriever
    from legal_rag.retrieve.law_scope import LawScope
    from legal_rag.retrieve.version_filter import VersionFilter
    from legal_rag.store.base import build_store

    config = RagConfig.from_env()
    items = [item for item in load_qa_set(args.qa_file) if item.sources]
    if args.limit_questions:
        items = items[: args.limit_questions]
    print(f"题集 {args.qa_file}：{len(items)} 道**有期望来源**的题（role={args.role}）")

    # ⚠️ 空库闸门（真机踩过）：忘了 `source env_cloud.sh` 时 VECTOR_STORE 默认 memory（空库）、
    # 嵌入退回哈希，结果是"跑完了但每题 rank 都是 None"，看着像"检索全崩"，其实是没接上环境。
    print(f"向量库={config.vector_store} collection={config.collection} "
          f"嵌入={config.embedding_provider}/{config.embedding_model} "
          f"重排基线={config.rerank_provider}", flush=True)

    embedder = build_embedder(config.embedding_provider, config.embedding_model,
                              config.embedding_dim, config=config)
    store = build_store(config.vector_store, persist_dir=config.index_dir,
                        collection=config.collection, config=config, embedder=embedder)
    if config.vector_store == "memory":
        print("!! 向量库是 memory（没设 VECTOR_STORE / LEGAL_RAG_MILVUS_URI？）—— 空库上做"
              "检索评测毫无意义。请先 `source /root/env_cloud.sh` 再跑。", file=sys.stderr)
        return 2
    try:
        row_count = store.count()
    except Exception as exc:  # noqa: BLE001
        print(f"!! 取库内条数失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if row_count <= 0:
        print(f"!! 库里 0 条（collection={config.collection}）—— 先确认环境与 collection 名",
              file=sys.stderr)
        return 2
    print(f"库里 {row_count} 条", flush=True)
    scope = {"is_parent": False, "role_id": args.role}
    law_scope = LawScope.load_or_build(config.index_dir, config.knowledge_dir)
    version_filter = VersionFilter.load_or_none(config.index_dir)

    ranker_specs = args.ranker or ["cosine="]
    report: dict = {"role": args.role, "k": args.limit, "questions": len(items),
                    "law_scope_laws": len(law_scope), "version_filter_stale": len(version_filter or []),
                    "rankers": {}}
    pools: dict[str, list[SearchHit]] = {}
    internals: dict[str, list[SearchHit]] = {}
    pick_cache: dict[str, list[SearchHit]] = {}

    # LLM 列表式选择器（P8 第五条杠杆）：候选用**法内宽候选**（不是最终 top-k，也不是召回池）——
    # 根因就是"正确的那条没进榜"，从错的池子里挑等于白挑。
    selector = None
    picked_labels_of = None
    pick_store: dict[str, list[str]] = {}
    selector_config: dict = {}
    if args.selector:
        from legal_rag.generate.llm_base import build_llm_client
        from legal_rag.retrieve.selector import ListwiseSelector, article_of, prompt_version

        picked_labels_of = lambda hits: [f"{h.chunk.source}#{article_of(h.chunk.text)}"  # noqa: E731
                                         for h in hits]
        # 先把"喂给模型的东西"定下来：缓存只认**同一套 prompt 配置**
        # （提示词版本 + 候选数 + 片段字数 + 挑条数）。提示词版本取决于是否加版本标记。
        version_labels = bool(getattr(config.retrieval, "law_selector_version_labels", False))
        selector_config = {
            "prompt": prompt_version({"x": "现行有效"} if version_labels else None),
            "max_candidates": (args.selector_max_candidates
                               or config.retrieval.law_selector_max_candidates),
            "snippet_chars": (args.selector_snippet_chars
                              or config.retrieval.law_selector_snippet_chars),
            "top_k": (args.selector_top_k or config.retrieval.law_selector_top_k),
        }
        if args.selector_cache:
            pick_store = load_pick_cache(args.selector_cache, selector_config)
            if pick_store:
                print(f"选择器挑条缓存已载入：{len(pick_store)} 题（配置一致）", flush=True)
        provider = args.selector_provider or config.llm_provider
        model = args.selector_model or config.llm_model
        client = build_llm_client(provider, model=model, temperature=0.0,
                                  max_tokens=256,
                                  timeout=(float(getattr(config.retrieval,
                                                         "law_selector_timeout", 0.0) or 0.0)
                                           or config.llm_timeout),
                                  deepseek_base_url=config.deepseek_base_url,
                                  openai_base_url=config.openai_compat_base_url)
        selector = ListwiseSelector(client, **selector_kwargs(selector_config))
        if args.selector_trigger:
            config.retrieval.law_selector_trigger = args.selector_trigger
        print(f"列表式选择器已开启：provider={provider} model={model} "
              f"top_k={selector.top_k} 候选上限={selector.max_candidates} "
              f"每条字数={selector.snippet_chars} 触发={config.retrieval.law_selector_trigger}"
              f"（min_law_hits={config.retrieval.law_selector_min_law_hits} "
              f"min_hits={config.retrieval.law_selector_min_hits}）", flush=True)
        report["selector"] = {"provider": provider, "model": model, "top_k": selector.top_k,
                              "max_candidates": selector.max_candidates,
                              "snippet_chars": selector.snippet_chars,
                              "trigger": config.retrieval.law_selector_trigger,
                              "min_law_hits": config.retrieval.law_selector_min_law_hits,
                              "min_hits": config.retrieval.law_selector_min_hits}

    for spec in ranker_specs:
        name, model = parse_ranker(spec)
        print(f"\n=== 重排器 {name}（{model or '融合分'}）")
        retriever = HybridRetriever(embedder, store, config.retrieval, None,
                                    law_scope=law_scope, cache_dir=config.index_dir,
                                    version_filter=version_filter, selector=selector)
        # ⚠️ 预热闸门（真机踩过，代价是一整天的错误基线）：评测**必须**先预热，否则 BM25 是
        # 懒触发的后台线程建的 —— 前几十题只有向量通道、后几十题两通道齐全，同一轮里口径不一，
        # 而且"这一轮到底有没有关键词通道"取决于机器快慢。实测证据：同题集同库，一次跑出
        # cosine Recall@5=0.3590（池均 16.3），另一次 0.2821（池均 19.1），m3 却完全一致
        # —— 差异就来自关键词通道在与不在。所以这里**不给过**（除非显式 --allow-degraded-keyword）。
        warm = retriever.warm_up(scope)
        print(f"  预热：embed={warm.get('embed')} BM25={warm.get('bm25_chunks')} "
              f"错误={warm.get('errors')}", flush=True)
        report.setdefault("warmup", {})[name] = {
            "embed": warm.get("embed"), "bm25_chunks": warm.get("bm25_chunks"),
            "errors": list(warm.get("errors") or []),
        }
        if not warm.get("bm25_chunks"):
            message = (f"!! 关键词通道未就绪（预热报告 BM25={warm.get('bm25_chunks')}，"
                       f"错误={warm.get('errors')}）—— 此时召回只有向量通道，指标与"
                       f"「两通道齐全」的轮次**不可比**。请先让缓存/索引就绪，"
                       f"或显式 --allow-degraded-keyword 接受降级口径。")
            print(message, file=sys.stderr)
            if not args.allow_degraded_keyword:
                return 2
        ranker = build_ranker(name, model, config.retrieval)
        rows: list[dict] = []
        calls_at_start = selector.calls if selector is not None else 0
        seconds_at_start = selector.seconds if selector is not None else 0.0
        started = time.perf_counter()
        for index, item in enumerate(items, 1):
            # 多轮题用**全部轮次拼成查询**：真实链路带历史，只拿最后一轮会冤枉它
            # （"那补偿是按什么标准算的"单看确实找不到《劳动合同法》）。
            query = " ".join(item.messages)
            pool = pools.get(item.id)
            if pool is None:
                pool = retriever.candidates(query, where=scope)
                pools[item.id] = pool
            if args.candidate_pool > 0:
                pool = pool[: args.candidate_pool]
            # ⚠️ 每个重排器打**独立的 hit 副本**（真机踩到）：CrossEncoder 的 `rerank()` 会把
            # 分数**写回 hit 对象**（`hit.rerank_score` / `hit.score`），而候选池是跨重排器复用的
            # ⇒ 排在 CrossEncoder 后面的 `cosine`（按 `hit.score` 排序）实际用的是**上一个重排器
            # 的分**，两臂数字会诡异地完全相同。实测反例：把 m3 放前面时，cosine 臂报出的
            # Recall/MRR 与 m3 一字不差，且零命中清单也相同 —— 这就是污染的指纹。
            own_pool = isolated_pool(pool) if pool else []
            ranked = ranker.rerank(query, own_pool, top_k=args.limit) if own_pool else []
            pick_expected = None       # 选择器挑中的**正确条**在原重排榜里的名次；0 = 原本没进榜
            picks = 0                  # 本轮选择器实际挑了几条（0 = 没识别到法名 / 没挑出来）
            if selector is not None:
                wide = internals.get(item.id)
                if wide is None:
                    wide = retriever.law_internal_candidates(query, scope)
                    internals[item.id] = wide
                if wide:
                    # 选择器结果只取决于（问题, 法内候选），与重排器无关 ⇒ 跨重排器复用同一次调用
                    picked = pick_cache.get(item.id)
                    if picked is None and item.id in pick_store:
                        by_id = {hit.chunk.id: hit for hit in wide}
                        ids = pick_store[item.id]
                        picked = [by_id[cid] for cid in ids if cid in by_id]
                        if len(picked) != len(ids):
                            # 缓存的 chunk id 在当前候选里找不到 ⇒ 候选池本身变了
                            # （语料/版本策略/过滤条件改过），**缓存的判断对新池子无效**：
                            # 必须重问一次模型，不能拿残缺的旧结果当"模型的选择"。
                            print(f"  !! {item.id} 选择器缓存有 {len(ids) - len(picked)} 条不在"
                                  f"当前法内候选（候选池已变）⇒ 重新调用模型", file=sys.stderr)
                            picked = None
                        else:
                            pick_cache[item.id] = picked
                    if picked is None:
                        outcome = selector.select(query, wide)
                        picked = list(outcome.get("picked") or [])
                        pick_cache[item.id] = picked
                    pick_store[item.id] = [hit.chunk.id for hit in picked]
                    picks = len(picked)
                    before = {hit.chunk.id: pos for pos, hit in enumerate(ranked, 1)}
                    hits_expected = [hit for hit in picked
                                     if hit_matches(hit, item.sources, item.articles)]
                    if hits_expected:
                        # 0 表示"原本压根不在 top-k 里"—— 这才是选择器的**净救援**，
                        # 而 1..k 只说明它在重排榜里、选择器只是把它提到前面。
                        pick_expected = min(before.get(hit.chunk.id, 0) for hit in hits_expected)
                    if picked:
                        chosen = {hit.chunk.id for hit in picked}
                        ranked = (picked + [hit for hit in ranked if hit.chunk.id not in chosen])
                        ranked = ranked[: args.limit]
            rank = rank_of_expected(ranked, item.sources, item.articles)
            rows.append({"id": item.id, "rank": rank, "pool": len(pool),
                         "picks": picks, "pick_expected": pick_expected,
                         # 挑中的具体条文（含来源与条号）：既方便排障，也是**标注核查**的依据
                         "picked_labels": (picked_labels_of(pick_cache.get(item.id, []))
                                           if picked_labels_of else []),
                         "articles": list(item.articles)})
            if index % 20 == 0:
                print(f"  {index}/{len(items)} …")
        summary = summarize_rankings(rows, k=args.limit)
        summary["seconds"] = round(time.perf_counter() - started, 1)
        summary["avg_pool"] = round(sum(r["pool"] for r in rows) / max(len(rows), 1), 1)
        summary["detail"] = rows          # 逐题留档：事后能算"哪几题翻转"，不必重跑云端
        report["rankers"][name] = summary
        print(f"  Recall@{args.limit}={summary['recall_at_k']}  MRR={summary['mrr']}  "
              f"零命中 {len(summary['zero_hit'])} 题  ({summary['seconds']}s)")
        if selector is not None:
            print(f"  选择器：{summary['picked_any']}/{len(items)} 题挑出条，"
                  f"其中挑中正确条 {summary['picked_correct']} 题"
                  f"（{summary['picked_rescued']} 题原本不在 top-{args.limit}）")
            calls = selector.calls - calls_at_start
            print(f"  选择器开销：本轮调用 {calls} 次，均 "
                  f"{(selector.seconds - seconds_at_start) / max(calls, 1):.2f}s/次，"
                  f"prompt 均 {selector.prompt_chars // max(selector.calls, 1)} 字")
            summary["selector_calls"] = calls
            summary["selector_seconds"] = round(selector.seconds - seconds_at_start, 1)
        report["selector_cost"] = {"calls": selector.calls if selector else 0,
                                   "seconds": round(selector.seconds, 1) if selector else 0.0,
                                   "prompt_chars": selector.prompt_chars if selector else 0}

    print("\n=== 对比 ===")
    base = report["rankers"].get("cosine") or next(iter(report["rankers"].values()))
    for name, summary in report["rankers"].items():
        delta = summary["recall_at_k"] - base["recall_at_k"]
        print(f"  {name:<10} Recall@{args.limit}={summary['recall_at_k']:<7} "
              f"MRR={summary['mrr']:<7} ΔRecall={delta:+.4f} 零命中={summary['zero_hit']}")

    if args.selector_cache and pick_store:
        Path(args.selector_cache).parent.mkdir(parents=True, exist_ok=True)
        Path(args.selector_cache).write_text(
            json.dumps({"version": 2, "config": selector_config, "picks": pick_store},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"选择器挑条已落盘：{len(pick_store)} 题 -> {args.selector_cache}")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
