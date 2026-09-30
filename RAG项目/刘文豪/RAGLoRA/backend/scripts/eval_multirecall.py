# -*- coding: utf-8 -*-
"""多路召回 vs 单路向量检索 —— 实测增益对比。

为什么必须做这个对照
--------------------
「三路都跑通了」不等于「检索变好了」。多路召回引入的
**文档限定加分**与**图谱精确命中**都可能同时带来帮助和干扰：

    - 有帮助：问「民法典第五百七十七条」时，向量路返回的是
      第二百四十七条 / 第一千零五十七条（数字形近、语义相近但**不是**那一条），
      而图谱路能精确命中。
    - 有干扰：文档限定加分若权重不当，会把语义更相关的其它条文压下去。

因此这里在**检索层**做 A/B 对照（不涉及生成，结果可直接归因）：
    A：单路向量检索（`multi_recall(..., multi=False)`）
    B：三路召回 + RRF 融合

指标：**来源命中率** —— `qa_set` 里标注的 `expect_sources` 是否出现在召回结果中。
这是本项目自研评测用的同一口径，可直接对照。

用法
----
    D:\\anaconda3\\envs\\rag_env\\python.exe scripts\\eval_multirecall.py
    D:\\anaconda3\\envs\\rag_env\\python.exe scripts\\eval_multirecall.py --topk 5
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config                      # noqa: E402
from app.services import retrieval               # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
ROLE_MAP = {"doctor": "medical", "lawyer": "legal"}
COLL_OF = {"medical": config.COLLECTION_MEDICAL, "legal": config.COLLECTION_LEGAL}


def load_qa() -> dict:
    return json.loads((EVAL_DIR / "qa_set.json").read_text(encoding="utf-8"))


def source_hit(hits: list[dict], expect: list[str]) -> bool:
    """召回结果里是否出现期望来源（与 run_eval.py 同口径）。"""
    if not expect:
        return False
    blob = " ".join(
        f"{h.get('source') or ''} {h.get('law_name') or ''}" for h in hits
    )
    return any(e.lower() in blob.lower() for e in expect)


def run(topk: int) -> int:
    qa = load_qa()
    print("=" * 76)
    print(f"多路召回 vs 单路向量检索　|　top-{topk}")
    print(f"向量库：{config.VECTOR_STORE}　多路开关：{config.MULTI_RECALL_ENABLED}")
    print("=" * 76)

    # ⚠️ 先预热：首次检索会加载 bge-m3（约数百毫秒）。
    #    不做预热的话，先跑的那一路会把模型加载耗时算进自己账上 ——
    #    初版实测因此得出「多路比单路快 70%」的**错误结论**。
    t0 = time.time()
    retrieval.search("预热", COLL_OF["medical"], 3, multi=False)
    retrieval.multi_recall("预热", COLL_OF["medical"], 3)
    print(f"预热完成（{time.time()-t0:.1f}s），开始计时对照\n")

    rows = []
    for role, coll in COLL_OF.items():
        items = qa[role]
        print(f"\n▶ {role}（{len(items)} 题）")
        print("-" * 76)
        print(f"{'#':>3} {'单路':>6} {'多路':>6}  问题")

        for i, item in enumerate(items, 1):
            q, expect = item["q"], item.get("expect_sources", [])

            # A：单路向量
            t0 = time.time()
            h_a, _ = retrieval.search(q, coll, topk, multi=False)
            ms_a = (time.time() - t0) * 1000

            # B：三路融合
            t0 = time.time()
            h_b, tr = retrieval.multi_recall(q, coll, topk)
            ms_b = (time.time() - t0) * 1000

            a, b = source_hit(h_a, expect), source_hit(h_b, expect)
            rows.append({
                "role": role, "q": q, "single": a, "multi": b,
                "ms_single": ms_a, "ms_multi": ms_b,
                "delta": (b - a),
                "graph_n": tr["paths"].get("graph", {}).get("n", 0),
                "doc_in": (tr.get("doc_meta") or {}).get("in_library"),
                "paths": {k: v.get("n") for k, v in tr["paths"].items()},
            })
            mark = lambda v: "✓" if v else "✗"
            flag = "  ← 多路改善" if (b and not a) else ("  ← 多路变差" if (a and not b) else "")
            print(f"{i:>3} {mark(a):>6} {mark(b):>6}  {q[:34]}{flag}")

    print()
    print("=" * 76)
    print("结果汇总")
    print("=" * 76)
    print(f"{'角色':<10} {'题数':>5} {'单路命中':>10} {'多路命中':>10} {'变化':>8}")
    print("-" * 76)

    total = {"n": 0, "single": 0, "multi": 0}
    for role in COLL_OF:
        sel = [r for r in rows if r["role"] == role]
        if not sel:
            continue
        s = sum(r["single"] for r in sel)
        m = sum(r["multi"] for r in sel)
        total["n"] += len(sel); total["single"] += s; total["multi"] += m
        print(f"{role:<10} {len(sel):>5} {s/len(sel)*100:>9.1f}% {m/len(sel)*100:>9.1f}% "
              f"{(m-s)/len(sel)*100:>+7.1f}pp")

    print("-" * 76)
    n = total["n"]
    print(f"{'合计':<10} {n:>5} {total['single']/n*100:>9.1f}% {total['multi']/n*100:>9.1f}% "
          f"{(total['multi']-total['single'])/n*100:>+7.1f}pp")

    imp = [r for r in rows if r["multi"] and not r["single"]]
    wor = [r for r in rows if r["single"] and not r["multi"]]
    print()
    print(f"多路改善 {len(imp)} 题，多路变差 {len(wor)} 题")
    for r in imp[:5]:
        print(f"  ↑ {r['q'][:40]}  (图谱 {r['graph_n']} 条, 库内={r['doc_in']})")
    for r in wor[:5]:
        print(f"  ↓ {r['q'][:40]}  (路径 {r['paths']})")

    print()
    ms_a = statistics.mean(r["ms_single"] for r in rows)
    ms_b = statistics.mean(r["ms_multi"] for r in rows)
    print(f"平均耗时：单路 {ms_a:.0f}ms　多路 {ms_b:.0f}ms　（+{ms_b-ms_a:.0f}ms, "
          f"{(ms_b/ms_a-1)*100:+.0f}%）")
    print()
    print("⚠️ 本对照只评估**检索层**的来源命中，不含生成质量与精排。")
    print("   多路召回的最终价值还要看端到端评测（eval/run_eval.py）。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5, help="取前 K 条判命中（0=全部召回）")
    args = ap.parse_args()
    return run(args.topk or config.RECALL_TOP_K)


if __name__ == "__main__":
    sys.exit(main())
