# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
"""
工单02 · 检索参数扫描（两阶段）。

  python scripts/sweep.py                      # 阶段一网格 → 阶段二只跑 Pareto 前沿
  python scripts/sweep.py --phase1-only        # 只跑便宜的网格部分
  python scripts/sweep.py --dump-candidates 95 # 调试：看某题的候选池构成

【为什么要分两阶段】
朴素做法是"每个格子都跑完整生成 + LLM judge"：3×3×2×2×2 = 72 格 × 10 题
≈ 720 次 qwen3 生成，约 20 分钟起，而且绝大多数格子连确定性指标都过不了，
纯属烧电。

所以：
  阶段一 —— **不调 LLM**，只算确定性指标（页精确率/页召回率/CKC/上下文字数）。
           查询向量与任何被扫的参数都无关，所以**每题只嵌入一次**、
           跨所有格子复用，靠 CachedEmbedder 做到，整轮网格约 1 分钟。
  阶段二 —— 只对 Pareto 前沿（指标好、上下文短的那些格子）跑真实生成，
           量 TTFT 与规则命中。LLM judge 仍然只跑最终选中的那一个格子
           （见 eval.py），这里不跑。

【时间口径】阶段一的 retrieval_ms 是**检索装配耗时**，不含嵌入 ——
嵌入被缓存了，而它本来也不随这些参数变化。真实端到端的 TTFT 看阶段二。
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                          # noqa: E402
from app.core.evaluator import Evaluator, retrieval_metrics              # noqa: E402
from app.core.generator import Generator                                 # noqa: E402
from app.core.profiles import PROFILES                                   # noqa: E402
from app.core.retriever import Retriever, build_context                  # noqa: E402

# ----------------------------------------------------------------------
# 扫描网格
# ----------------------------------------------------------------------
# context_chunk_chars 起步就是 600，**绝不向下扫** ——
# 300 字会静默截断答案（工单01 的 id=793 就是这么坏的），见 profiles.MIN_CONTEXT_CHARS。
GRID = {
    "top_k":                [3, 4, 5],
    "pool_size":            [20, 30, 50],
    "context_chunk_chars":  [600, 700],
    "expand_neighbors":     [False, True],
    "neighbor_total_chars": [450, 900],
}


class CachedEmbedder:
    """给 Embedder 套一层记忆化。

    查询视角（原问句 + 抽象版）在一整轮扫描里是固定的，而 top_k / pool /
    窗口都不影响嵌入结果 —— 所以每题只需真嵌入一次，其余 71 个格子直接命中缓存。
    这把阶段一从"720 次嵌入"降到"20 次嵌入"。
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.cache: dict[str, list[float]] = {}
        self.n_calls = 0
        self.n_hits = 0

    async def embed_one(self, text: str) -> list[float]:
        if text in self.cache:
            self.n_hits += 1
            return self.cache[text]
        self.n_calls += 1
        vec = await self.inner.embed_one(text)
        self.cache[text] = vec
        return vec

    async def embed_many(self, texts):        # 透传，供其它调用方使用
        return await self.inner.embed_many(texts)


# ----------------------------------------------------------------------
def cell_profile(base, **over):
    """按格子参数派生剖面。名字保持 optimized，避免触发窗口下限断言。"""
    return base.derived(**over)


async def eval_cell(base, params: dict, questions, embedder) -> dict:
    """阶段一：一个格子跑 10 题的确定性指标。"""
    p = cell_profile(base, **params)
    r = Retriever(profile=p, embedder=embedder)

    prec, rec, ckc, nchars, times = [], [], [], [], []
    for q in questions:
        res = await r.retrieve(q["question"])
        m = retrieval_metrics(res.hits, q.get("rule_keywords", []),
                              q.get("evidence_page", ""))
        if m["page_precision"] is not None:
            prec.append(m["page_precision"])
        if m["page_recall"] is not None:
            rec.append(m["page_recall"])
        if m["context_keyword_coverage"] is not None:
            ckc.append(m["context_keyword_coverage"])
        ctx = build_context(res.hits, query=q["question"], profile=p)
        nchars.append(len(ctx))
        times.append(res.seconds * 1000)

    mean = lambda xs: round(statistics.mean(xs), 4) if xs else None  # noqa: E731
    return {
        **params,
        "page_precision": mean(prec),
        "page_recall": mean(rec),
        "ckc": mean(ckc),
        "avg_context_chars": int(statistics.mean(nchars)) if nchars else 0,
        "retrieval_ms": int(statistics.mean(times)) if times else 0,
    }


def pareto_front(rows: list[dict]) -> list[dict]:
    """非支配集：CKC 与页召回越大越好、上下文字数越小越好。

    只留"没有任何一格在三个维度上都不劣于它"的格子 —— 这是阶段二该跑的最小集合。
    """
    def key(r):
        return (r["ckc"] or 0, r["page_recall"] or 0, -r["avg_context_chars"])

    front = []
    for r in rows:
        kr = key(r)
        dominated = False
        for o in rows:
            if o is r:
                continue
            ko = key(o)
            if (ko[0] >= kr[0] and ko[1] >= kr[1] and ko[2] >= kr[2]
                    and ko != kr):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front


def behaviour_key(r: dict) -> tuple:
    """把"参数不同但行为必然相同"的格子折叠成同一个键。

    `expand_neighbors=False` 时，`neighbor_total_chars` 根本不会被读到 ——
    前沿里因此成对出现四个只差这个死参数的格子。不折叠的话阶段二会白跑一倍。
    """
    return (r["top_k"], r["pool_size"], r["context_chunk_chars"],
            r["expand_neighbors"],
            r["neighbor_total_chars"] if r["expand_neighbors"] else 0)


def distinct_cells(rows: list[dict], extra: list[dict] | None = None) -> list[dict]:
    """前沿去重 + 可选的对照格。

    对照格的作用是回答"多给上下文到底有没有用"——只跑前沿是回答不了的，
    因为前沿按定义就只留了上下文最短的那些。加一格 top_k 最大的做对照，
    结论才有说服力（哪怕结论是"没用"）。
    """
    seen, out = set(), []
    for r in rows + (extra or []):
        k = behaviour_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


async def phase2(rows: list[dict], base, questions, embedder) -> list[dict]:
    """阶段二：对前沿格子跑真实生成，量 TTFT 与规则命中。"""
    from app.core.evaluator import rule_hit
    gen = Generator()
    out = []
    for r in rows:
        params = {k: r[k] for k in GRID}
        p = cell_profile(base, **params)
        ret = Retriever(profile=p, embedder=embedder)
        hits_n, ttfts, ctxs = 0, [], []
        for q in questions:
            res = await ret.retrieve(q["question"])
            ctx = build_context(res.hits, query=q["question"], profile=p)
            ctxs.append(len(ctx))
            t = time.perf_counter()
            ans = await gen.generate(q["question"], res.hits, context=ctx, profile=p)
            ttfts.append(int((time.perf_counter() - t) * 1000))
            ok, _ = rule_hit(ans, q.get("rule_keywords", []), q.get("rule_type", "any"))
            hits_n += 1 if ok else 0
        out.append({
            **params,
            "ckc": r["ckc"], "page_recall": r["page_recall"],
            "avg_context_chars": int(statistics.mean(ctxs)) if ctxs else 0,
            "rule_hits": hits_n, "n": len(questions),
            "ttft_p95_ms": sorted(ttfts)[min(len(ttfts) - 1,
                                             int(round(0.95 * (len(ttfts) - 1))))],
        })
    return out


# ----------------------------------------------------------------------
async def dump_candidates(qid: int, profile_name: str) -> None:
    """调试：看某题的候选池是怎么构成的（谁进了组、谁是代表、邻块补了谁）。

    这是回答"权威页优先是不是真在候选池里选到了 1-1-26"这类问题的唯一诚实方式 ——
    不许靠猜。
    """
    from app.core.retriever import _tokens, abstract_query
    questions = Evaluator.load_questions()
    q = next((x for x in questions if x["id"] == qid), None)
    if q is None:
        print(f"没有 id={qid} 的题", file=sys.stderr)
        return

    p = PROFILES.get(profile_name, PROFILES["optimized"])
    r = Retriever(profile=p)
    res = await r.retrieve(q["question"])

    print(f"\n题 id={qid}  剖面={res.profile}")
    print(f"问：{q['question']}")
    print(f"证据页：{q['evidence_page']}")
    print(f"候选 {res.n_candidates} 个 → 聚合合并 {res.n_merged} → "
          f"冗余过滤 {res.n_filtered} → 命中 {len(res.evidence_hits)} → "
          f"邻块 {res.n_added_neighbors}")
    print(f"查询视角：{res.query_views}")
    print(f"查询词：{sorted(_tokens(abstract_query(q['question'])))}")
    print(f"\n{'序号':<5}{'页码':<10}{'类型':<8}{'分数':>8}  {'内容':<40}")
    for i, h in enumerate(res.hits, 1):
        tag = "邻块" if h.is_neighbor else "命中"
        print(f"{i:<5}{h.page_label:<10}{tag:<8}{h.score:>8.3f}  "
              f"{h.content[:40].replace(chr(10), ' ')}")
        if h.section_path:
            print(f"       └ {h.section_path[:70]}")


async def main() -> int:
    ap = argparse.ArgumentParser(description="工单02 · 检索参数扫描")
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    ap.add_argument("--profile", default="optimized", help="以哪个剖面为基准派生")
    ap.add_argument("--dump-candidates", type=int, default=None,
                    help="只打印某题的候选池，不做扫描")
    args = ap.parse_args()

    if args.dump_candidates is not None:
        await dump_candidates(args.dump_candidates, args.profile)
        return 0

    base = PROFILES.get(args.profile, PROFILES["optimized"])
    questions = Evaluator.load_questions()
    if args.limit:
        questions = questions[:args.limit]

    from app.core.embedder import Embedder
    embedder = CachedEmbedder(Embedder())

    keys = list(GRID)
    cells = [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]
    print(f"阶段一：{len(cells)} 个格子 × {len(questions)} 题（不调 LLM）…")

    rows = []
    t0 = time.perf_counter()
    for i, params in enumerate(cells, 1):
        rows.append(await eval_cell(base, params, questions, embedder))
        if i % 12 == 0 or i == len(cells):
            print(f"  {i}/{len(cells)} …", flush=True)
    print(f"  完成，用时 {time.perf_counter() - t0:.1f}s；"
          f"嵌入调用 {embedder.n_calls} 次、命中缓存 {embedder.n_hits} 次")

    rows.sort(key=lambda r: (-(r["ckc"] or 0), -(r["page_recall"] or 0),
                             r["avg_context_chars"]))
    head = (f"{'top_k':>6}{'pool':>6}{'窗口':>6}{'邻块':>6}{'邻字':>6}"
            f"{'CKC':>8}{'页召回':>8}{'页精确':>8}{'上下文':>8}{'检索ms':>8}")
    print("\n" + "=" * len(head))
    print("  阶段一 · 确定性指标（按 CKC → 页召回 → 上下文升序）")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for r in rows[:24]:
        print(f"{r['top_k']:>6}{r['pool_size']:>6}{r['context_chunk_chars']:>6}"
              f"{str(r['expand_neighbors']):>6}{r['neighbor_total_chars']:>6}"
              f"{(r['ckc'] or 0):>8.3f}{(r['page_recall'] or 0):>8.3f}"
              f"{(r['page_precision'] or 0):>8.3f}{r['avg_context_chars']:>8}"
              f"{r['retrieval_ms']:>8}")
    if len(rows) > 24:
        print(f"  …（共 {len(rows)} 格，仅显示前 24）")

    front = pareto_front(rows)

    # 对照格：top_k 最大 + 窗口最大 + 邻块开着的那个格子。
    # 它一定在前沿之外（上下文最长），但"多给上下文到底有没有用"只有跑它才知道。
    control = max(rows, key=lambda r: (r["top_k"], r["context_chunk_chars"],
                                       r["expand_neighbors"]))
    cells = distinct_cells(front, [control])
    print(f"\n  Pareto 前沿 {len(front)} 格 → 去掉行为等价的重复后 {len(cells)} 格"
          f"（含 1 个 top_k/窗口最大的对照格）")

    result = {"phase1": rows, "pareto": front, "phase2_cells": cells}
    if not args.phase1_only:
        print(f"\n阶段二：{len(cells)} 格跑真实生成（qwen3，会慢）…")
        p2 = await phase2(cells, base, questions, embedder)
        p2.sort(key=lambda r: (-r["rule_hits"], r["ttft_p95_ms"]))
        head2 = (f"{'top_k':>6}{'pool':>6}{'窗口':>6}{'邻块':>6}{'邻字':>6}"
                 f"{'规则命中':>10}{'上下文':>8}{'TTFT-P95':>10}{'≤3s':>6}")
        print("\n" + "=" * len(head2))
        print("  阶段二 · 真实生成（这张表就是「召回率 vs TTFT」的权衡表）")
        print("=" * len(head2))
        print(head2)
        print("-" * len(head2))
        for r in p2:
            print(f"{r['top_k']:>6}{r['pool_size']:>6}{r['context_chunk_chars']:>6}"
                  f"{str(r['expand_neighbors']):>6}{r['neighbor_total_chars']:>6}"
                  f"{str(r['rule_hits']) + '/' + str(r['n']):>10}"
                  f"{r['avg_context_chars']:>8}{str(r['ttft_p95_ms']) + 'ms':>10}"
                  f"{'✅' if r['ttft_p95_ms'] < 3000 else '❌':>6}")
        result["phase2"] = p2

    out = settings.data_path / "eval"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告：{p}")
    print("  说明：阶段一的 retrieval_ms 不含嵌入（已缓存），"
          "真实端到端耗时看阶段二的 TTFT。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
