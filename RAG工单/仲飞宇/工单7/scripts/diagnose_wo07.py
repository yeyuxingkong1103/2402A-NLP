# 工单编号：人工智能NLP-RAG-功能测试及评估
# 工单07 - 功能测试及评估
"""
工单07 · 检索问题归因：把「答不出来」拆成可归因的三类。

  python scripts/diagnose_wo07.py                       # 默认读 eval/wo07_questions.json
  python scripts/diagnose_wo07.py --profile hybrid
  python scripts/diagnose_wo07.py --limit 3

【为什么必须做这一步】工单07 的产出物一是「分析检索结果存在的问题」。
"哪几题没答对"是现象，不是分析。要给出**可归因**的结论，必须把每题拆成三段：

  库里有没有？ ──▶ 检索有没有召回到？ ──▶ 生成有没有用上？
     │                    │                     │
     │                    │                     └─ 生成侧漏答：要点在送进 prompt 的
     │                    │                        上下文里，答案却没写出来
     │                    └─ 检索侧漏召：要点在库里，但没进 top-k
     └─ 语料缺口：要点根本不在库里（出题错了 / 解析丢了内容）

三类问题的**修法完全不同**（改出题 / 改检索 / 改生成），所以必须先分开。
判据用的是题目自带的 rule_keywords（判"答案对不对"）与 recall_keywords
（判"要点有没有被召回"），与评估器**同一套匹配规则**（evaluator.keyword_hits），
避免出现"评估说召回了、归因说没召回"两套口径打架。

【输出】每题的归因结论 + 一张汇总表（三类各几题、涉及哪些 id）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                          # noqa: E402
from app.core.evaluator import keyword_hits                              # noqa: E402
from app.core.profiles import get_profile                                # noqa: E402


async def diagnose(questions: list[dict], profile_name: str, limit=None) -> list[dict]:
    from app.core.generator import Generator
    from app.core.retriever import Retriever, build_context
    from app.core.vectorstore import VectorStore

    prof = get_profile(profile_name)
    retriever = Retriever(profile=prof)
    store = VectorStore()
    gen = Generator()

    out: list[dict] = []
    for q in (questions[:limit] if limit else questions):
        kws = list(q.get("rule_keywords") or [])
        rkws = list(q.get("recall_keywords") or kws)
        rres = await retriever.retrieve(q["question"])
        ctx = build_context(rres.hits, query=q["question"], profile=prof)

        # ---- 段 3：生成侧有没有用上 ----
        answer = await gen.generate(q["question"], rres.hits, context=ctx,
                                    doc=None)
        ctx_all = "".join(h.content for h in rres.hits)
        ans_hit = keyword_hits(answer, kws)
        # ---- 段 2：召回上下文（含邻块）覆盖了哪些要点 / 哪些判据 ----
        ctx_hit = keyword_hits(ctx_all, rkws)
        rule_in_ctx = keyword_hits(ctx_all, kws)
        # ---- 段 1：库里到底有没有这些东西 ----
        routed = rres.routed_doc
        flt = f'doc_name == "{routed}"' if routed else "id >= 0"
        rows = store.client.query(collection_name=store.collection, filter=flt,
                                  output_fields=["chunk_index", "page_label",
                                                 "content"], limit=16384)
        in_store = {k: [] for k in rkws}
        for r in rows:
            for k in rkws:
                if keyword_hits(r["content"], [k]):
                    in_store[k].append(f'{r["page_label"]}#{r["chunk_index"]}')

        # 【①「语料缺口」只看**答案判据**（rule_keywords）】要点词是松散的描述性
        # 短语（如「生态系统」），原文可能只写「生态」—— 拿它去判"库里没有"，
        # 会把检索问题误报成语料问题（实测 id=702/710 被这样误报过）。
        # 真正的"库里没有"应当是**答案判据**在库里找不到。
        missing_in_store = [k for k in kws
                            if not any(keyword_hits(r["content"], [k]) for r in rows)]
        missed_recall = [k for k in rkws if in_store[k] and k not in ctx_hit]

        # 【归因口径：三段各用哪套判据】这是本脚本最容易写错的地方，实测踩过两次：
        #   · 段3「生成侧漏答」必须用**答案判据**（rule_keywords）来判，不能用
        #     要点词（recall_keywords）。要点词是**概念词**（「不良贷款率」），
        #     一个简洁而完全正确的答案（「1.65%、3.01%、183.12%」）本来就不会
        #     重复它。第一版拿要点词去判，把 703–706 四道**答对的题**全归成了
        #     "生成侧漏答"，整节问题分析都会跟着错。
        #   · 段2「检索侧漏召」要看**答案判据有没有进上下文**，而不是看要点词 ——
        #     要点词常常在（讲风险的那一段当然会出现"拨备覆盖率"字样），
        #     但**具体数值**没进，这题照样答不出来（实测 id=708）。
        missed_rule = [k for k in kws if k not in rule_in_ctx]
        missed_gen = [k for k in kws if k in rule_in_ctx and k not in ans_hit]

        if missing_in_store:
            verdict = "① 语料缺口（要点不在库里 → 多半是出题/解析问题）"
        elif missed_rule:
            verdict = "② 检索侧漏召（答案判据在库里，但没进召回上下文）"
        elif missed_gen:
            verdict = "③ 生成侧漏答（判据已进上下文，答案却没写出来）"
        else:
            verdict = "✅ 三段全通"

        out.append({
            "id": q["id"], "question": q["question"], "verdict": verdict,
            "rule_hit": bool(keyword_hits(answer, kws)) and
                        len(keyword_hits(answer, kws)) == len(kws) if kws else None,
            "answer_keywords_hit": f"{len(ans_hit)}/{len(kws)}",
            "recall_hit": f"{len(ctx_hit)}/{len(rkws)}",
            "missing_in_store": missing_in_store,
            "missed_recall": missed_recall,
            "missed_rule": missed_rule,
            "missed_gen": missed_gen,
            "rule_in_ctx": f"{len(rule_in_ctx)}/{len(kws)}",
            "retrieved": [f"{h.page_label}{'*' if h.is_neighbor else ''}"
                          for h in rres.hits],
            "routed_doc": rres.routed_doc or "(全库)",
            "in_store_pages": {k: v[:4] for k, v in in_store.items() if v},
            "answer": answer,
        })
    return out


def report(items: list[dict], profile_name: str) -> None:
    print("=" * 78)
    print(f"  工单07 · 检索问题归因（剖面 {profile_name}）")
    print("=" * 78)
    for it in items:
        print(f"\n--- id={it['id']}  {it['verdict']}")
        print(f"    题面：{it['question'][:60]}")
        print(f"    路由：{it['routed_doc']}")
        print(f"    召回页：{it['retrieved']}")
        print(f"    答案命中 {it['answer_keywords_hit']} ｜ 判据进上下文 "
              f"{it['rule_in_ctx']} ｜ 要点召回 {it['recall_hit']}")
        if it["missing_in_store"]:
            print(f"    ⚠️ 库里查不到：{it['missing_in_store']}")
        if it["missed_rule"]:
            print(f"    ⚠️ 答案判据没进召回上下文：{it['missed_rule']}")
        if it["missed_recall"]:
            print(f"    ⚠️ 要点未召回：{it['missed_recall']}")
            for k in it["missed_recall"][:3]:
                print(f"         「{k}」其实在：{it['in_store_pages'].get(k)}")
        if it["missed_gen"]:
            print(f"    ⚠️ 上下文里有、答案没写：{it['missed_gen']}")
        print(f"    答案：{it['answer'][:150]}")

    print("\n" + "=" * 78)
    print("  汇总：")
    tally: dict[str, list[int]] = {}
    for it in items:
        tally.setdefault(it["verdict"][:2] + it["verdict"][2:6], []).append(it["id"])
    for v, ids in tally.items():
        print(f"    {v}：{len(ids)} 题 {ids}")


def main() -> int:
    ap = argparse.ArgumentParser(description="工单07 · 检索问题归因")
    ap.add_argument("--questions", default="eval/wo07_questions.json")
    ap.add_argument("--profile", default="optimized")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", default=None, help="把归因结果另存一份 JSON")
    args = ap.parse_args()

    path = Path(args.questions)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    qs = json.loads(path.read_text(encoding="utf-8"))["questions"]
    items = asyncio.run(diagnose(qs, args.profile, args.limit))
    report(items, args.profile)
    if args.json:
        Path(args.json).write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  归因明细：{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
