"""
混合检索策略评测（工单6 核心交付物之一）
工单编号：人工智能NLP-RAG-混合检索任务

对应工单验收标准：
    准确率：优化后的检索策略应显著提升检索结果的准确率，目标准确率提升至 90% 以上。
    召回率：确保重要信息的召回率，目标召回率提升至 95% 以上。
    响应时间：从用户提问到返回答案的时间应不超过 3 秒。

—— 指标怎么定义的（这三条决定了报告能不能被复核）——

工单说的「准确率」和「召回率」，在**检索层**必须落到可观测量上。本脚本的定义：

  准确率 accuracy   = 「关键事实全部被召回」的题目占比
                      （must_have 里的每一条都出现在 top-k 文本里）
  召回率 recall     = 关键事实的**平均覆盖率**
                      （must_have 中被召回的条目数 / 总条目数，逐题求平均）
  命中率 hit@k      = 权威页（gold.pages）出现在 top-k 里的题目占比
  MRR              = 第一个权威页的倒数平均排名（排序质量的细粒度指标）
  响应时间          = 纯检索耗时（不含 LLM 生成），给 mean / p95 两个数

为什么准确率用「全部事实命中」而不是"单条命中"：
    单条命中太宽松 —— 一个块里只要出现"C4ISR"就算对，而工单要的是
    「检索到的答案能支撑作答」。全部命中才等价于"这一问的答案确实被召回了"。

为什么同时报 hit@k 与 fact 指标：
    两者失效的方式不同。hit@k 只看页码，不关心内容是否完整
    （召回了正确页但切块切碎了也算过）；fact 指标只看文本，不关心页码
    （从别处碰巧捞到同样的数字也算过）。两个都报，才能发现"是不是靠运气过的"。

用法：
    python scripts/eval_strategies.py                     # 跑默认网格（快，不调 LLM）
    python scripts/eval_strategies.py --with-llm          # 额外跑 LLM 重排器（慢、消耗 token）
    python scripts/eval_strategies.py --grid weights      # 只跑权重扫描
    python scripts/eval_strategies.py --limit 5           # 先跑前 5 题试水
    python scripts/eval_strategies.py --no-clip           # 关掉 CLIP 跨模态路

产出：
    data/eval/工单6_混合检索评测_<时间戳>.json   机器可读（每题每条链路的明细）
    data/eval/工单6_混合检索评测_<时间戳>.md     人可读（可交付给验收方）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, hybrid  # noqa: E402

_NUM_RE = re.compile(r"^[\d,]+(?:\.\d+)?%?$")


def fact_hit(text: str, fact: str) -> bool:
    """
    判断一条关键事实是否出现在召回文本里。

    数字类事实必须做**边界匹配**：`must_have_counts` 里是 "4"、"6" 这种裸数字，
    直接子串匹配的话，"4" 会命中 "1,670.4"、"第 44 页" 里的一堆 4，
    准确率会被虚高抬起来。用前后非数字的断言来卡住。
    """
    if not fact:
        return False
    if _NUM_RE.match(fact.strip()):
        return re.search(r"(?<![\d.])" + re.escape(fact) + r"(?![\d])", text) is not None
    return fact in text


def gold_hit(pages: list[int], gold: list[dict]) -> bool:
    for g in gold or []:
        if set(g.get("pages") or []) & set(pages):
            return True
    return False


def gold_rank(pages: list[int], gold: list[dict]) -> int:
    """第一个权威页在召回里的名次（1-based）；没命中返回 0。"""
    want: set[int] = set()
    for g in gold or []:
        want |= set(g.get("pages") or [])
    for i, p in enumerate(pages, 1):
        if p in want:
            return i
    return 0


def _observe(q: dict, items, ms: float, gated: bool, trace: dict) -> dict:
    """
    把一次检索的结果折算成一行观测。

    单独抽出来是为了让**优化前基线**（工单1~5 的 `retriever.retrieve`）与
    新的混合检索能走**同一个折算函数** —— 否则两边的指标口径一旦差一点点，
    「提升了多少」这个结论就不成立了。
    """
    pages = [it.page for it in items]
    text = "\n".join(it.text for it in items)

    must = list(q.get("must_have") or []) + list(q.get("must_have_counts") or [])
    hits = [f for f in must if fact_hit(text, f)]
    rank = gold_rank(pages, q.get("gold"))

    return {
        "id": q["id"],
        "question": q["question"],
        "n_items": len(items),
        "pages": pages,
        "must_total": len(must),
        "must_hit": len(hits),
        "must_missing": [f for f in must if f not in hits],
        "fact_all_hit": bool(must) and len(hits) == len(must),
        "gold_rank": rank,
        "gold_hit": rank > 0,
        "sources": sorted({s for it in items for s in (getattr(it, "sources", []) or [])}),
        "latency_ms": round(ms, 2),
        "rerank_ms": (trace.get("rerank") or {}).get("ms", 0.0),
        "gated": gated,
        "trace_steps": trace.get("steps") or [],
    }


def run_one(q: dict, cfg: hybrid.HybridConfig, kb) -> dict:
    """跑一题（混合检索），返回该题在一条链路上的全部原始观测。"""
    t0 = time.perf_counter()
    r = hybrid.search(q["question"], cfg, kb=kb)
    ms = (time.perf_counter() - t0) * 1000
    return _observe(q, r.items, ms, r.gated, r.trace or {})


def run_legacy(q: dict, kb, top_k: int, use_clip: bool) -> dict:
    """
    跑一题（**工单1~5 的检索实现**）。

    它是本工单的「优化前」：单路融合（余弦 + β·BM25归一 + δ·共识 + 图像先验），
    有阈值闸门，但**没有**可切换的策略、没有可调权重、没有重排器。
    评测报告里必须带上它，否则「混合检索提升了多少」这句话没有参照物。
    """
    from src.retriever import retrieve as legacy_retrieve

    t0 = time.perf_counter()
    r = legacy_retrieve(q["question"], top_k=top_k, use_clip=use_clip)
    ms = (time.perf_counter() - t0) * 1000
    return _observe(q, r.items, ms, r.gated, r.trace or {})


def summarize(rows: list[dict]) -> dict:
    """把逐题观测汇成一份指标。"""
    n = len(rows)
    if not n:
        return {}
    acc = sum(1 for r in rows if r["fact_all_hit"]) / n
    tot_must = sum(r["must_total"] for r in rows)
    rec = (sum(r["must_hit"] for r in rows) / tot_must) if tot_must else 0.0
    lat = sorted(r["latency_ms"] for r in rows)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "recall": round(rec, 4),
        "hit@1": round(sum(1 for r in rows if r["gold_rank"] == 1) / n, 4),
        "hit@3": round(sum(1 for r in rows if 0 < r["gold_rank"] <= 3) / n, 4),
        "hit@5": round(sum(1 for r in rows if 0 < r["gold_rank"] <= 5) / n, 4),
        "mrr": round(sum((1.0 / r["gold_rank"]) if r["gold_rank"] else 0.0 for r in rows) / n, 4),
        "latency_mean_ms": round(sum(lat) / n, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_max_ms": round(max(lat), 2),
        "gate_rate": round(sum(1 for r in rows if r["gated"]) / n, 4),
    }


# ============================================================ 评测网格

def build_grid(which: str, args) -> list[tuple[str, hybrid.HybridConfig]]:
    """
    构造 (标签, 配置) 列表。

    每个网格都遵循「单变量」原则：一次只改一个旋钮，其余全部固定，
    否则报告里的差异无法归因（这是工单1~5 反复强调的纪律）。
    """
    # 基准配置 == 线上默认（重排器、权重、返回条数都读 config，
    # 不写死 tfidf）——否则报告会出现「汇总用 feedback、主表用 tfidf」的口径打架。
    # 只有「重排器对比」那一组会单变量地换掉 reranker。
    base = hybrid.HybridConfig(
        strategy=config.RETRIEVAL_STRATEGY, fusion=config.HYBRID_FUSION,
        vector_weight=config.HYBRID_VECTOR_WEIGHT, reranker=config.RERANKER,
        use_clip=not args.no_clip, top_k=args.top_k,
    )
    out: list[tuple[str, hybrid.HybridConfig]] = []
    which = which or "all"

    def add(label: str, cfg: hybrid.HybridConfig) -> None:
        out.append((label, cfg))

    if which in ("all", "strategies"):
        add("向量检索（纯）", replace(base, strategy="vector"))
        add("全文检索（纯）", replace(base, strategy="fulltext"))
        add("混合检索", replace(base, strategy="hybrid"))

    if which in ("all", "fusions"):
        for f in ("weighted", "rrf", "borda", "vote", "evidence"):
            add(f"混合·{hybrid.FUSIONS[f]['name']}", replace(base, fusion=f))

    if which in ("all", "weights"):
        for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            add(f"混合·向量权重 {w:.1f}", replace(base, vector_weight=w))

    if which in ("all", "rerankers"):
        rk = ["none", "tfidf", "feedback"] + (["llm"] if args.with_llm else [])
        for r in rk:
            add(f"混合·重排器 {r}", replace(base, reranker=r))

    if which in ("all", "syntax"):
        add("全文·默认 OR", replace(base, strategy="fulltext"))
        add("全文·严格 AND", replace(base, strategy="fulltext", ft_mode="and"))
        add("全文·关闭查询语法", replace(base, strategy="fulltext", ft_syntax=False))

    return out


def to_md(rep: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# 工单6 · 混合检索策略评测报告")
    A("")
    A(f"- 工单编号：{rep['work_order_no']}")
    A(f"- 生成时间：{rep['generated_at']}")
    A(f"- 评测集：`{rep['dataset']}`（共 {rep['num_questions']} 题："
      f"验收 {rep['num_acceptance']} 题 + 探针 {rep['num_probe']} 题）")
    A(f"- 索引：{rep['index']['num_chunks']} 块 · 嵌入模型 `{rep['index']['embedding_model']}`"
      f"（{rep['index']['embedding_dim']} 维）")
    A(f"- 倒排索引：{rep['fulltext']['terms']} 词 / {rep['fulltext']['docs']} 块"
      f"（构建 {rep['fulltext']['build_ms']} ms，"
      f"字段权重 {rep['fulltext']['field_weights']}）")
    A(f"- 总耗时：{rep['elapsed_seconds']} s")
    A("")
    A("> 指标口径：**准确率** = 关键事实全部被召回的题目占比；"
      "**召回率** = 关键事实的平均覆盖率；"
      "**hit@k** = 权威页出现在 top-k 的题目占比；**MRR** = 首个权威页的倒数平均排名。")
    A("> 探针题（错别字等）不计入汇总，单列于文末。")
    A("")

    for section in rep["sections"]:
        A(f"## {section['title']}")
        A("")
        A(section["desc"])
        A("")
        A("| 链路 | 准确率 | 召回率 | hit@1 | hit@3 | hit@5 | MRR | 平均耗时 | p95 | 闸门拦空 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for row in section["rows"]:
            m = row["metrics"]
            if not m:
                A(f"| {row['label']} | — | — | — | — | — | — | — | — | — |")
                continue
            A(f"| {row['label']} | {m['accuracy']:.2%} | {m['recall']:.2%} | "
              f"{m['hit@1']:.2%} | {m['hit@3']:.2%} | {m['hit@5']:.2%} | {m['mrr']:.3f} | "
              f"{m['latency_mean_ms']:.0f} ms | {m['latency_p95_ms']:.0f} ms | {m['gate_rate']:.0%} |")
        A("")
        best = max((r for r in section["rows"] if r["metrics"]),
                   key=lambda r: (r["metrics"]["accuracy"], r["metrics"]["recall"]),
                   default=None)
        if best:
            A(f"**该组最优**：`{best['label']}` —— 准确率 {best['metrics']['accuracy']:.2%}、"
              f"召回率 {best['metrics']['recall']:.2%}、MRR {best['metrics']['mrr']:.3f}、"
              f"平均 {best['metrics']['latency_mean_ms']:.0f} ms。")
        A("")

    if rep.get("probe"):
        A("## 探针题（不计入汇总）")
        A("")
        A("| 链路 | 题目 | 权威页名次 | 召回页（前 5） | 关键事实 |")
        A("|---|---|---|---|---|")
        for row in rep["probe"]:
            r = row["result"]
            A(f"| {row['label']} | {r['question']} | {r['gold_rank'] or '未命中'} | "
              f"{r['pages'][:5]} | {r['must_hit']}/{r['must_total']} |")
        A("")

    A(f"## 逐题明细（线上默认配置：{rep['default_desc']}）")
    A("")
    A("| id | 问题 | 权威页名次 | 召回页（前 5） | 关键事实命中 | 召回路径 | 耗时 |")
    A("|---|---|---|---|---|---|---|")
    for r in rep["detail"]:
        A(f"| {r['id']} | {r['question'][:44]} | {r['gold_rank'] or '—'} | {r['pages'][:5]} | "
          f"{r['must_hit']}/{r['must_total']} | {'/'.join(r['sources']) or '—'} | "
          f"{r['latency_ms']:.0f} ms |")
    A("")
    A("---")
    A("")
    A("报告由 `scripts/eval_strategies.py` 生成，可复现："
      "`python scripts/eval_strategies.py`。")
    return "\n".join(L)


# ============================================================ 入口

def main() -> int:
    ap = argparse.ArgumentParser(description="工单6 混合检索策略评测")
    ap.add_argument("--dataset", default="questions_wot6.json")
    ap.add_argument("--grid", default="all",
                    choices=["all", "strategies", "fusions", "weights", "rerankers", "syntax"])
    ap.add_argument("--rerankers", default="", help="（保留参数，重排器网格由 --grid rerankers 控制）")
    ap.add_argument("--with-llm", action="store_true", help="额外评测 LLM 重排器（慢、消耗 token）")
    ap.add_argument("--no-clip", action="store_true", help="关掉 CLIP 跨模态召回路")
    # 默认跟随线上配置（RETRIEVAL_FINAL_TOP_K），避免评测用的是 6、线上用的是 8
    ap.add_argument("--top-k", type=int, default=config.RETRIEVAL_FINAL_TOP_K)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（试水用）")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    from src.fulltext import get_index
    from src.hybrid import stop_terms_for
    from src.index_store import KnowledgeBase

    t_start = time.time()
    kb = KnowledgeBase.get()
    idx = get_index(kb)
    stop_terms_for(kb)

    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = config.EVAL_DIR / args.dataset
    qs = json.loads(ds_path.read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]
    acceptance = [q for q in qs if q.get("group", "acceptance") == "acceptance"]
    probes = [q for q in qs if q.get("group") == "probe"]
    print(f"[eval] 评测集 {ds_path.name}：{len(qs)} 题"
          f"（验收 {len(acceptance)}，探针 {len(probes)}）")

    grid = build_grid(args.grid, args)
    sections: list[dict] = []
    detail: list[dict] = []
    probe_rows: list[dict] = []

    def group_of(label: str) -> tuple[str, str]:
        """把一条配置归到报告的某一节（顺序敏感：先精确、后前缀）。"""
        if label in ("向量检索（纯）", "全文检索（纯）", "混合检索"):
            return ("一、检索策略对比（向量 / 全文 / 混合）",
                    "三种策略在同一套配置下只改召回路径：向量路 = 嵌入+余弦相似度，"
                    "全文路 = 倒排索引 + BM25F，混合 = 两路同时执行后融合再重排。")
        if label.startswith("混合·向量权重"):
            return ("三、权重扫描（向量权重 0 → 1）",
                    "0 = 纯全文，1 = 纯向量。用来回答「两路各给多少权重最合适」"
                    "——这是工单「支持权重调整」的量化依据。")
        if label.startswith("混合·重排器"):
            return ("四、重排器对比",
                    "召回完全相同，只换精排方式。`不重排（对照）` 用来量化重排的净收益。")
        if label.startswith("混合·"):
            return ("二、融合算法对比（同一候选池，只换合并算法）",
                    "加权平均用归一化后的分数；RRF / Borda / 多数投票用名次；"
                    "依据分是工单1~5 的原始公式。")
        if label.startswith("全文·"):
            return ("五、全文查询语义对比（OR / 严格 AND / 关闭语法）",
                    "同一批问题，只改全文路的查询模式，看召回与精度的此消彼长。")
        return ("其他", "")

    # ---- 零、优化前基线：工单1~5 的检索实现（本工单的参照物）----
    if args.grid == "all":
        t0 = time.time()
        base_rows = [run_legacy(q, kb, args.top_k, not args.no_clip) for q in acceptance]
        bm = summarize(base_rows)
        sections.append({
            "title": "零、优化前基线（工单1~5 的检索实现）",
            "desc": "工单1~5 的检索：单路融合（余弦 + β·BM25归一 + δ·文档共识 + 图像先验）、"
                    "带阈值闸门，但**没有**可切换的检索策略、可调权重与重排器。"
                    "本工单的所有改进都以这一行为参照。",
            "rows": [{"label": "工单1~5 基线", "config": {}, "metrics": bm, "cases": base_rows}],
        })
        print(f"[{'工单1~5 基线（优化前）':<22}] 准确率 {bm.get('accuracy', 0):.2%}  "
              f"召回率 {bm.get('recall', 0):.2%}  hit@3 {bm.get('hit@3', 0):.2%}  "
              f"MRR {bm.get('mrr', 0):.3f}  {bm.get('latency_mean_ms', 0):.0f} ms  "
              f"（{time.time() - t0:.1f}s）")

    cur_title = sections[-1]["title"] if sections else None
    for label, cfg in grid:
        title, desc = group_of(label)
        if title != cur_title:
            sections.append({"title": title, "desc": desc, "rows": []})
            cur_title = title

        rows = []
        t0 = time.time()
        for q in acceptance:
            try:
                rows.append(run_one(q, cfg, kb))
            except Exception as exc:  # noqa: BLE001
                print(f"    ! 题目 {q['id']} 失败：{type(exc).__name__}: {exc}")
        m = summarize(rows)
        sections[-1]["rows"].append({"label": label, "config": cfg.to_dict(),
                                     "metrics": m, "cases": rows})
        print(f"[{label:<26}] 准确率 {m.get('accuracy', 0):.2%}  "
              f"召回率 {m.get('recall', 0):.2%}  hit@3 {m.get('hit@3', 0):.2%}  "
              f"MRR {m.get('mrr', 0):.3f}  {m.get('latency_mean_ms', 0):.0f} ms  "
              f"（{time.time() - t0:.1f}s）")

        # 默认配置（混合+加权平均+TF-IDF）跑一遍探针题
        if label == "混合检索" or (args.grid == "syntax" and label.endswith("默认 OR")):
            for q in probes:
                probe_rows.append({"label": label, "result": run_one(q, cfg, kb)})

    # 逐题明细：用「线上默认配置」再跑一遍（保证与报告开头声明一致）
    # 工单编号：人工智能NLP-RAG-混合检索任务
    # 说明：这里必须跟随 config 的默认重排器/权重/返回条数，不能写死 tfidf，
    #       否则会出现「汇总用 feedback、明细用 tfidf」的自相矛盾报告。
    default_cfg = hybrid.HybridConfig(strategy=config.RETRIEVAL_STRATEGY,
                                      fusion=config.HYBRID_FUSION,
                                      vector_weight=config.HYBRID_VECTOR_WEIGHT,
                                      reranker=config.RERANKER,
                                      use_clip=not args.no_clip, top_k=args.top_k)
    for q in acceptance:
        detail.append(run_one(q, default_cfg, kb))

    rep = {
        "work_order_no": config.WORK_ORDER_NO_HYBRID,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": ds_path.name,
        "num_questions": len(qs),
        "num_acceptance": len(acceptance),
        "num_probe": len(probes),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "index": kb.meta,
        "fulltext": idx.stats(),
        "grid": args.grid,
        "with_llm": bool(args.with_llm),
        "use_clip": not args.no_clip,
        "default_desc": "{strategy} · {fusion} · 向量权重 {vw:g} · 重排器 {rr} · 返回 {tk} 条".format(
            strategy=hybrid.STRATEGIES.get(default_cfg.strategy, {}).get("name", default_cfg.strategy),
            fusion=hybrid.FUSIONS.get(default_cfg.fusion, {}).get("name", default_cfg.fusion),
            vw=default_cfg.vector_weight, rr=default_cfg.reranker, tk=default_cfg.top_k),
        "sections": sections,
        "probe": probe_rows,
        "detail": detail,
    }

    out_dir = Path(args.out_dir) if args.out_dir else config.EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"工单6_混合检索评测_{ts}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(to_md(rep), encoding="utf-8")
    print(f"\n[eval] 报告已生成：\n  {out_dir / (stem + '.md')}\n  {out_dir / (stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
