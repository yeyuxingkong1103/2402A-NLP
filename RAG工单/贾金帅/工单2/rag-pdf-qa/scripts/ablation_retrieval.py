"""
检索消融实验：逐层开启优化，量化每一层的贡献
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

工单2 要求「从 pdf 解析的处理、分块优化、检索优化等层面，进行检索准确率的优化」，
并要求「对比优化前后检索精确度的变化」。本脚本就是「检索精确度变化」的量化依据。

--------------------------------------------------------------------------
为什么不用 LLM 打分
--------------------------------------------------------------------------
LLM-as-judge 有随机性、成本高、不可复现，适合评**答案**，不适合评**检索**。
检索质量用**客观规则指标**衡量才站得住：

    fact_recall        关键事实命中率 —— 每题人工标注的 must_have 关键事实
                       （数字/名称，如 "6,464.51"、"程家明"）在召回上下文中的覆盖率。
                       这是最贴近业务的定义：检索有没有把「能回答问题的原文」捞回来。
    all_facts_rate     全部关键事实都命中的题目比例（最严格口径，一题错一个就是 0）
    gold_page_hit@K    正确页码（gold_pages，由「同时包含全部关键事实的块」反查得到，
                       与任何模型输出无关）是否出现在 top-K 召回的题目比例
    mrr                第一个命中正确页码的块的平均倒数排名（1.0 = 每次都在第一）

--------------------------------------------------------------------------
配置矩阵：不止「逐层叠加」，还做了**受控对照**
--------------------------------------------------------------------------
只用「逐层叠加」有个方法论陷阱：分块层的效果会被后面的检索层掩盖，
无法判断到底是分块变了、还是检索变了。所以额外加了 A2b —— 把检索层**固定**在
混合检索，只切换分块方式，让「分块层到底有没有用」这个问题有确定答案。

    A0_naive          朴素解析（不断行/不抽表）+ 定长 500 字滑窗 + 纯向量      ← 优化前
    A1_parse          A0 + 行内标题断行
    A2_chunk          A1 + 标题感知分块 + 表格独立成块  （检索仍是纯向量）
    A2b_fixed_hybrid  定长滑窗分块 + 混合检索            （检索固定，只换分块）
    A3_hybrid         标题感知分块 + 混合检索            ← 与 A2b 对照即「分块层」的净效果
    A4_df_filter      A3 + 无区分度词（DF）过滤
    A5_gate           A4 + 依据分阈值闸门                ← 优化后（当前主线）

用法：
    python scripts/ablation_retrieval.py
    python scripts/ablation_retrieval.py --top-k 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.baseline import NaiveIndex, chunk_naive  # noqa: E402
from src.embedder import embed_texts  # noqa: E402
from src.index_store import KnowledgeBase  # noqa: E402
from src.query_norm import normalize_query  # noqa: E402
from src.retriever import retrieve  # noqa: E402


# ------------------------------------------------------------------ 配置矩阵

CONFIGS = [
    {"key": "A0_naive", "name": "朴素 RAG（优化前）", "layer": "基线",
     "desc": "朴素解析 + 定长 500 字滑窗 + 纯向量余弦 top-k"},
    {"key": "A1_parse", "name": "+ 解析层：行内标题断行", "layer": "PDF 解析",
     "desc": "在 A0 基础上增加「行内标题断行」"},
    {"key": "A2_chunk", "name": "+ 分块层（检索仍为纯向量）", "layer": "分块",
     "desc": "改为标题感知分块 + 表格独立成块，检索保持纯向量"},
    {"key": "A2b_fixed_hybrid", "name": "受控：定长分块 + BM25 混合", "layer": "对照",
     "desc": "检索加 BM25，分块仍是定长滑窗 —— 与 A3 配对，隔离分块层（尚无 DF 过滤）"},
    {"key": "A2c_fixed_full", "name": "受控：定长分块 + 完整检索", "layer": "对照",
     "desc": "检索层与最终版完全一致（BM25+DF），分块仍是定长 —— 与 A4 配对，"
             "这是判断「分块层到底有没有用」最严格的一格"},
    {"key": "A3_hybrid", "name": "+ 检索层：BM25 混合", "layer": "检索",
     "desc": "标题感知分块 + 余弦与 BM25 加权融合"},
    {"key": "A4_df_filter", "name": "+ 检索层：DF 过滤（词级+短语级）", "layer": "检索",
     "desc": "在 A3 基础上剔除语料级无区分度成分（公司全称等）"},
    {"key": "A5_gate", "name": "完整优化系统（优化后）", "layer": "检索",
     "desc": "在 A4 基础上增加依据分阈值闸门"},
]

# 每一层「净效果」由哪两个配置相减得到（受控对照，只变一个变量）
LAYER_DELTAS = [
    ("解析层", "行内标题断行", "A0_naive", "A1_parse"),
    ("分块层", "标题感知分块（纯向量下）", "A1_parse", "A2_chunk"),
    ("检索层", "BM25 混合", "A2_chunk", "A3_hybrid"),
    ("检索层", "DF 过滤（词级 + 短语级）", "A3_hybrid", "A4_df_filter"),
    ("分块层", "标题感知分块（完整检索下·严格受控）", "A2c_fixed_full", "A4_df_filter"),
    ("检索层", "阈值闸门", "A4_df_filter", "A5_gate"),
]


# ------------------------------------------------------------------ 准备各配置

def make_retrievers(top_k: int, progress=print):
    """返回 {key: callable(question) -> list[dict]}；所有配置共用同一套检索算法。"""

    def main_kb(**kw):
        return lambda q: retrieve(q, top_k=top_k, **kw)

    out: dict[str, callable] = {}

    # ---- A0：落盘的朴素基线索引（定长 + 纯向量，无 BM25）
    naive_idx = NaiveIndex.load()
    kb0 = KnowledgeBase([c.to_dict() for c in naive_idx.chunks], naive_idx.embeddings, {})
    out["A0_naive"] = main_kb(kb=kb0, use_sparse=False, use_df_filter=False, apply_gate=False)
    progress(f"[准备] A0 朴素索引 {len(kb0.chunks)} 块")

    # ---- A1 / A2b：定长分块，但解析时做行内标题断行（只在内存里构建，不落盘）
    from src.pdf_parser import parse_pdf

    t = time.perf_counter()
    parsed1 = parse_pdf(config.PDF_PATH, with_tables=False, split_inline=True)
    chunks1 = chunk_naive(parsed1)
    vecs1 = embed_texts([c.text for c in chunks1])
    kb1 = KnowledgeBase([c.to_dict() for c in chunks1], vecs1, {})
    progress(f"[准备] A1 索引 {len(kb1.chunks)} 块，耗时 {time.perf_counter()-t:.1f}s")

    out["A1_parse"] = main_kb(kb=kb1, use_sparse=False, use_df_filter=False, apply_gate=False)
    out["A2b_fixed_hybrid"] = main_kb(kb=kb1, use_sparse=True, use_df_filter=False, apply_gate=False)
    out["A2c_fixed_full"] = main_kb(kb=kb1, use_sparse=True, use_df_filter=True, apply_gate=False)

    # ---- A2~A5：主线索引，逐层打开检索开关
    kb = KnowledgeBase.get()
    _ = kb.bm25
    progress(f"[准备] 主线索引 {len(kb.chunks)} 块（A2~A5 共用）")

    out["A2_chunk"] = main_kb(use_sparse=False, use_df_filter=False, apply_gate=False)
    out["A3_hybrid"] = main_kb(use_sparse=True, use_df_filter=False, apply_gate=False)
    out["A4_df_filter"] = main_kb(use_sparse=True, use_df_filter=True, apply_gate=False)
    out["A5_gate"] = main_kb(use_sparse=True, use_df_filter=True, apply_gate=True)
    return out


# ------------------------------------------------------------------ 指标

def score_case(items: list[dict], must_have: list[str], gold_pages: list[int]) -> dict:
    joined = "\n".join(it["text"] for it in items)
    hit = [k for k in must_have if k in joined]
    fact_recall = len(hit) / len(must_have) if must_have else None
    all_facts = len(hit) == len(must_have) if must_have else None

    rank = None
    for i, it in enumerate(items, 1):
        span = set(range(it["page"], it["page_end"] + 1)) if it["page"] else set()
        if span & set(gold_pages):
            rank = i
            break
    return {
        "fact_recall": fact_recall,
        "all_facts": all_facts,
        "missed": [k for k in must_have if k not in joined],
        "gold_page_hit": rank is not None,
        "mrr": (1.0 / rank) if rank else 0.0,
        "gold_rank": rank,
        "top_evidence": it_top(items),
        "n_items": len(items),
    }


def it_top(items: list[dict]) -> float:
    return items[0]["evidence"] if items else 0.0


def _avg(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=config.RETRIEVAL_FINAL_TOP_K)
    ap.add_argument("--dataset", default="ticket_questions.json")
    args = ap.parse_args()

    config.ensure_dirs()
    questions = json.loads((config.EVAL_DIR / args.dataset).read_text(encoding="utf-8"))

    print(f"[工单] {config.WORK_ORDER_NO_OPT}（{config.WORK_ORDER_SHORT}）")
    print(f"[实验] 检索消融，{len(questions)} 题 × {len(CONFIGS)} 配置，top_k={args.top_k}")
    print("[说明] 所有配置统一使用规则归一化后的原问题作为检索式，以隔离「结构优化」的贡献")
    print("[说明] 所有配置共用同一套融合算法（retrieve），只切换分块来源与三个检索开关\n")

    retrievers = make_retrievers(args.top_k)

    results: dict[str, dict] = {}
    for cfg in CONFIGS:
        fn = retrievers[cfg["key"]]
        per_case, lat = [], []
        for q in questions:
            query = normalize_query(q["question"])
            t0 = time.perf_counter()
            r = fn(query)
            lat.append((time.perf_counter() - t0) * 1000)
            items = [
                {"text": x.text, "page": x.page, "page_end": x.page_end, "evidence": x.evidence}
                for x in r.items
            ]
            s = score_case(items, q.get("must_have") or [], q.get("gold_pages") or [])
            s["id"] = q["id"]
            per_case.append(s)

        summary = {
            "fact_recall": _avg([c["fact_recall"] for c in per_case]),
            "all_facts_rate": _avg([1.0 if c["all_facts"] else 0.0 for c in per_case]),
            "gold_page_hit_rate": _avg([1.0 if c["gold_page_hit"] else 0.0 for c in per_case]),
            "mrr": _avg([c["mrr"] for c in per_case]),
            "avg_top_evidence": _avg([c["top_evidence"] for c in per_case]),
            "avg_latency_ms": _avg(lat),
        }
        results[cfg["key"]] = {**cfg, "summary": summary, "cases": per_case}
        print(f"  [{cfg['key']:<17}] 关键事实 {summary['fact_recall']:.3f} | "
              f"全命中 {summary['all_facts_rate']*len(questions):.0f}/{len(questions)} | "
              f"正确页 {summary['gold_page_hit_rate']:.3f} | MRR {summary['mrr']:.3f}")

    # ---- 层归因：每一层的净效果
    layers = []
    for layer, item, a, b in LAYER_DELTAS:
        sa, sb = results[a]["summary"], results[b]["summary"]
        layers.append({
            "layer": layer,
            "item": item,
            "from": a, "to": b,
            "fact_recall_delta": round((sb["fact_recall"] or 0) - (sa["fact_recall"] or 0), 4),
            "mrr_delta": round((sb["mrr"] or 0) - (sa["mrr"] or 0), 4),
        })

    a0, a5 = results["A0_naive"]["summary"], results["A5_gate"]["summary"]
    improvement = {
        "fact_recall_delta": round((a5["fact_recall"] or 0) - (a0["fact_recall"] or 0), 4),
        "gold_page_hit_delta": round((a5["gold_page_hit_rate"] or 0) - (a0["gold_page_hit_rate"] or 0), 4),
        "mrr_delta": round((a5["mrr"] or 0) - (a0["mrr"] or 0), 4),
        "all_facts_before": round((a0["all_facts_rate"] or 0) * len(questions)),
        "all_facts_after": round((a5["all_facts_rate"] or 0) * len(questions)),
    }

    report = {
        "work_order_no": config.WORK_ORDER_NO_OPT,
        "work_order_short": config.WORK_ORDER_SHORT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_questions": len(questions),
        "top_k": args.top_k,
        "metric_note": "关键事实命中率 = 人工标注 must_have 在召回上下文中的覆盖率；"
                       "gold_pages 由「同时包含全部关键事实的块」反查得到，与任何模型输出无关",
        "improvement": improvement,
        "layers": layers,
        "configs": results,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_检索消融实验_{stamp}.json"
    md_path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_检索消融实验_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("\n=== 各层净效果 ===")
    for x in layers:
        print(f"  [{x['layer']}] {x['item']:<28} 关键事实 {x['fact_recall_delta']:+.3f}  MRR {x['mrr_delta']:+.3f}")
    print("\n=== 总提升（A0 → A5）===")
    print(f"  关键事实命中率：{a0['fact_recall']:.3f} → {a5['fact_recall']:.3f}")
    print(f"  全命中题目数：{improvement['all_facts_before']}/10 → {improvement['all_facts_after']}/10")
    print(f"  正确页命中率：{a0['gold_page_hit_rate']:.3f} → {a5['gold_page_hit_rate']:.3f}")
    print(f"  MRR：{a0['mrr']:.3f} → {a5['mrr']:.3f}")
    print(f"\n[OK] {json_path}")
    print(f"[OK] {md_path}")
    return 0


def to_markdown(report: dict) -> str:
    cfgs = list(report["configs"].values())
    n = report["num_questions"]
    L = [
        "# 检索消融实验报告（工单2）",
        "",
        f"- 工单编号：{report['work_order_no']}",
        f"- 生成时间：{report['generated_at']}",
        f"- 题目数：{n}，top_k={report['top_k']}",
        "",
        "## 一、逐层开启优化的效果",
        "",
        "| 配置 | 优化层面 | 关键事实命中率 | 全部命中题数 | 正确页命中率 | MRR | 平均耗时 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cfgs:
        s = c["summary"]
        L.append(
            f"| **{c['key']}** {c['name']} | {c['layer']} | {s['fact_recall']:.3f} | "
            f"{s['all_facts_rate']*n:.0f}/{n} | {s['gold_page_hit_rate']:.3f} | "
            f"{s['mrr']:.3f} | {s['avg_latency_ms']:.1f} ms |"
        )

    L += [
        "",
        "## 二、各优化层的净效果（受控对照）",
        "",
        "> 每行只改变**一个变量**，其余保持不变。",
        "",
        "| 层面 | 优化项 | 对照 | 关键事实命中率变化 | MRR 变化 |",
        "|---|---|---|---|---|",
    ]
    for x in report["layers"]:
        L.append(
            f"| {x['layer']} | {x['item']} | {x['from']} → {x['to']} | "
            f"{x['fact_recall_delta']:+.3f} | {x['mrr_delta']:+.3f} |"
        )

    a0 = report["configs"]["A0_naive"]["summary"]
    a5 = report["configs"]["A5_gate"]["summary"]
    d = report["improvement"]
    L += [
        "",
        "## 三、优化前后总对比",
        "",
        "| 指标 | 优化前 (A0) | 优化后 (A5) | 提升 |",
        "|---|---|---|---|",
        f"| 关键事实命中率 | {a0['fact_recall']:.3f} | {a5['fact_recall']:.3f} | +{d['fact_recall_delta']:.3f} |",
        f"| 全部关键事实命中的题目数 | {d['all_facts_before']}/{n} | {d['all_facts_after']}/{n} | "
        f"+{d['all_facts_after']-d['all_facts_before']} |",
        f"| 正确页命中率 | {a0['gold_page_hit_rate']:.3f} | {a5['gold_page_hit_rate']:.3f} | +{d['gold_page_hit_delta']:.3f} |",
        f"| MRR | {a0['mrr']:.3f} | {a5['mrr']:.3f} | +{d['mrr_delta']:.3f} |",
        "",
        "## 四、逐题明细（优化前 vs 优化后）",
        "",
        "| id | 关键事实命中率 A0 → A5 | 最早命中排名 A0 → A5 | 优化前漏掉的事实 |",
        "|---|---|---|---|",
    ]
    a0c = {c["id"]: c for c in report["configs"]["A0_naive"]["cases"]}
    a5c = {c["id"]: c for c in report["configs"]["A5_gate"]["cases"]}
    for qid in a0c:
        x, y = a0c[qid], a5c[qid]
        L.append(
            f"| {qid} | {x['fact_recall']:.2f} → {y['fact_recall']:.2f} | "
            f"{x['gold_rank'] or '未命中'} → {y['gold_rank'] or '未命中'} | "
            f"{'、'.join(x['missed']) or '—'} |"
        )
    L += ["", "## 五、指标口径说明", "", f"- {report['metric_note']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
