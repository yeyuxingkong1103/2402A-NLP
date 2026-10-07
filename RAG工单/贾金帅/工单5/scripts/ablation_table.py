"""
表格解析消融实验：量化「表格结构化」对检索精确度的贡献
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单3 的产出物要求「针对如下问题进行检索，**显示检索到的答案及检索精确度**」，
备注又要求「分析待检索的问题答案在 pdf 中的定位，**使用表格解析技术实现**」。
因此这里的对比必须回答一个可证伪的问题：

    把表格结构化，到底有没有让检索更准？准了多少？准在哪几题上？

--------------------------------------------------------------------------
为什么用「严格受控对照」而不是「优化前后两套系统」
--------------------------------------------------------------------------
工单2 已经做过 A0→A5 的逐层消融。工单3 **不再重复那一套**，只切一个变量：

    T0_no_table   表格不做结构化：表格内容作为普通文本行混进段落参与分块
    T1_table      表格结构化：表格区域从正文剔除，另以「列名：值」的形态独立成块

**其余全部一致**（这是结论能成立的前提）：
  * 同一个解析器（都做行内标题断行、都去页眉页码），只有 with_tables 一个开关不同
  * 同一个分块器 chunk_document、同一套 CHUNK_SIZE/OVERLAP 参数
  * 同一个向量模型、同一份 BM25 词表、同一套融合公式与 top_k
  * 同一个检索式（规则归一化后的原问题），不调 LLM 改写 —— 隔离「结构」的贡献

--------------------------------------------------------------------------
指标口径
--------------------------------------------------------------------------
    fact_recall        关键事实命中率：人工标注的 must_have 在召回上下文里的覆盖率
    all_facts_rate     全部关键事实都命中的题目比例（最严口径）
    gold_page_hit@K    正确页（**文档 + 页码**，两文档语料下页码会撞车，必须带文档键）
                       是否出现在 top-K 里
    mrr                第一个命中正确页的块的倒数排名

用法：
    python scripts/ablation_table.py
    python scripts/ablation_table.py --top-k 8
    python scripts/ablation_table.py --with-gate       # 顺带跑一遍带阈值闸门的口径
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chunker import chunk_document  # noqa: E402
from src.index_store import KnowledgeBase  # noqa: E402
from src.pdf_parser import parse_pdf  # noqa: E402
from src.query_norm import normalize_query  # noqa: E402
from src.retriever import retrieve  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO_TABLE

CONFIGS = [
    {"key": "T0_no_table", "name": "表格不结构化（优化前）",
     "desc": "表格内容展平成普通文本行，与上下文段落一起参与分块与检索"},
    {"key": "T1_table", "name": "表格结构化（本工单实现）",
     "desc": "表格区域从正文剔除；每张表输出「表名 + 来源 + 列名清单 + 行级『列名：值』」独立成块"},
]


# ------------------------------------------------------------------ 建对照索引


def build_no_table_index(progress=print) -> KnowledgeBase:
    """
    取「表格不结构化」的对照索引。

    优先用落盘的那一份（`data/index_notable/`，由
    `python scripts/build_index.py --no-tables --out notable` 生成）——
    向量化 1835 块在 CPU 上要 3 分钟，重建一次就够慢了，不该每跑一次消融就重来。
    落盘文件不存在才现场构建（只在内存里）。

    ⚠️ 现场构建必须用 `parse_pdf(with_tables=False)` 而不是「把已抽好的表格再拍平成文本」：
    前者才是真正的「优化前」实现 —— 它连表格区域都没有识别出来，
    正文里表格的行仍然按原文的字符顺序排列（一行里的多列之间只有空格），
    这正是工单1/2 的做法。
    """
    import re

    from src.embedder import embed_texts

    try:
        kb = KnowledgeBase.get(config.INDEX_NOTABLE_DIR)
        n_tbl = sum(1 for c in kb.chunks if c["type"] == "table")
        progress(f"[准备] 对照索引（落盘）{len(kb.chunks)} 块，其中 table 型 {n_tbl}")
        return kb
    except FileNotFoundError:
        progress("[准备] 对照索引未落盘，现场构建（约 3 分钟）…")

    chunks_all = []
    seq = 0
    t0 = time.perf_counter()
    for d in config.DOCS:
        parsed = parse_pdf(
            config.RAW_DIR / d["file"],
            with_tables=False,                       # ← 唯一变量
            header_res=[re.compile(d["header_re"])],
            doc_name=d["name"],
        )
        chunks = chunk_document(parsed, doc_key=d["key"], seq_start=seq)
        seq += len(chunks)
        chunks_all.extend(chunks)
        progress(f"[准备] {d['key']} 分块 {len(chunks)} 条")
    vecs = embed_texts([c.text for c in chunks_all])
    progress(f"[准备] 对照索引 {len(chunks_all)} 块，向量 {vecs.shape}，"
             f"耗时 {time.perf_counter()-t0:.1f}s")
    return KnowledgeBase([c.to_dict() for c in chunks_all], vecs, {})


# ------------------------------------------------------------------ 指标


def score_case(items: list[dict], must_have: list[str], gold: list[dict]) -> dict:
    """
    计算单题指标。

    gold 形如 ``[{"doc": "liyuan", "pages": [22, 24]}]``。
    **两文档语料下必须带文档键**：兴图第 22 页和力源第 22 页都存在，
    只比页码会把「命中另一家公司的第 22 页」误判成命中。
    """
    joined = "\n".join(it["text"] for it in items)
    hit = [k for k in must_have if k in joined]
    fact_recall = len(hit) / len(must_have) if must_have else None

    rank = None
    for i, it in enumerate(items, 1):
        span = set(range(it["page"], it["page_end"] + 1)) if it["page"] else set()
        for g in gold:
            doc = g.get("doc")
            pages = set(g.get("pages") or [])
            if doc and doc != it.get("doc_key"):
                continue
            if span & pages:
                rank = i
                break
        if rank:
            break

    return {
        "fact_recall": fact_recall,
        "all_facts": (len(hit) == len(must_have)) if must_have else None,
        "missed": [k for k in must_have if k not in joined],
        "gold_page_hit": rank is not None,
        "mrr": (1.0 / rank) if rank else 0.0,
        "gold_rank": rank,
        "n_items": len(items),
        "top_evidence": items[0]["evidence"] if items else 0.0,
        "top_doc": items[0]["doc_key"] if items else "",
        "top_page": items[0]["page"] if items else 0,
        "table_chunks_in_topk": sum(1 for it in items if it["type"] == "table"),
    }


def _avg(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def summarize(cases: list[dict]) -> dict:
    n = len(cases) or 1
    return {
        "fact_recall": _avg([c["fact_recall"] for c in cases]),
        "all_facts_rate": _avg([1.0 if c["all_facts"] else 0.0 for c in cases]),
        "all_facts_count": sum(1 for c in cases if c["all_facts"]),
        "gold_page_hit_rate": _avg([1.0 if c["gold_page_hit"] else 0.0 for c in cases]),
        "mrr": _avg([c["mrr"] for c in cases]),
        "avg_top_evidence": _avg([c["top_evidence"] for c in cases]),
        "avg_table_chunks": _avg([c["table_chunks_in_topk"] for c in cases]),
        "gated_cases": sum(1 for c in cases if c["n_items"] == 0),
        "num_cases": n,
    }


# ------------------------------------------------------------------ 主流程


def run(build_no_table=build_no_table_index, progress=print, with_gate: bool = False,
        top_k: int | None = None):
    questions = json.loads((config.EVAL_DIR / "questions_wot3.json").read_text(encoding="utf-8"))
    top_k = top_k or config.RETRIEVAL_FINAL_TOP_K

    print(f"[工单] {WORK_ORDER_NO}")
    print(f"[实验] 表格解析消融，{len(questions)} 题 × {len(CONFIGS)} 配置，top_k={top_k}")
    print("[说明] 两份索引只有「表格是否结构化」一个变量不同；检索式一律用规则归一化的原问题\n")

    kb_table = KnowledgeBase.get()
    _ = kb_table.bm25                       # 预热：BM25 词表 + 无区分度短语表
    progress(f"[准备] 结构化索引 {len(kb_table.chunks)} 块（含表格块 "
             f"{sum(1 for c in kb_table.chunks if c['type']=='table')}）")
    kb_no = build_no_table(progress)

    kbs = {"T1_table": kb_table, "T0_no_table": kb_no}

    results: dict[str, dict] = {}
    for cfg in CONFIGS:
        kb = kbs[cfg["key"]]
        per_case = []
        for q in questions:
            query = normalize_query(q["question"])
            r = retrieve(query, kb=kb, top_k=top_k, use_sparse=True, use_df_filter=True,
                         apply_gate=with_gate)
            items = [
                {"text": x.text, "page": x.page, "page_end": x.page_end,
                 "evidence": x.evidence, "type": x.type, "doc_key": x.doc_key}
                for x in r.items
            ]
            s = score_case(items, q.get("must_have") or [], q.get("gold") or [])
            s["id"] = q["id"]
            s["source"] = q.get("source", "")
            s["question"] = q["question"]
            per_case.append(s)
        results[cfg["key"]] = {**cfg, "summary": summarize(per_case), "cases": per_case}
        s = results[cfg["key"]]["summary"]
        print(f"  [{cfg['key']:<12}] 关键事实 {s['fact_recall']:.3f} | "
              f"全命中 {s['all_facts_count']}/{len(questions)} | "
              f"正确页 {s['gold_page_hit_rate']:.3f} | MRR {s['mrr']:.3f}")

    # ---- 分层：新题（答案在表格里） vs 旧题（答案在正文里）
    def subset(key: str, src: str) -> dict:
        return summarize([c for c in results[key]["cases"] if c["source"] == src])

    groups = {}
    for src, label in (("工单3新增", "新题（力源·答案在表格中）"),
                       ("工单1/2", "旧题（兴图·答案在正文中）")):
        if not any(c["source"] == src for c in results["T1_table"]["cases"]):
            continue
        groups[label] = {
            "T0_no_table": subset("T0_no_table", src),
            "T1_table": subset("T1_table", src),
        }

    s0 = results["T0_no_table"]["summary"]
    s1 = results["T1_table"]["summary"]
    improvement = {
        "fact_recall_delta": round((s1["fact_recall"] or 0) - (s0["fact_recall"] or 0), 4),
        "all_facts_before": s0["all_facts_count"],
        "all_facts_after": s1["all_facts_count"],
        "gold_page_hit_delta": round((s1["gold_page_hit_rate"] or 0) - (s0["gold_page_hit_rate"] or 0), 4),
        "mrr_delta": round((s1["mrr"] or 0) - (s0["mrr"] or 0), 4),
    }

    return {
        "work_order_no": WORK_ORDER_NO,
        "work_order_short": config.WORK_ORDER_SHORT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_questions": len(questions),
        "top_k": top_k,
        "with_gate": with_gate,
        "metric_note": "关键事实命中率 = 人工标注 must_have 在召回上下文中的覆盖率；"
                       "正确页命中 = 「文档键 + 页码」同时对上（两文档语料的页码会撞车，必须带文档键）",
        "corpus": {
            k: {"chunks": len(kb.chunks),
                "table_chunks": sum(1 for c in kb.chunks if c["type"] == "table")}
            for k, kb in kbs.items()
        },
        "improvement": improvement,
        "groups": groups,
        "configs": results,
    }


# ------------------------------------------------------------------ 报告


def to_markdown(report: dict) -> str:
    n = report["num_questions"]
    cfgs = list(report["configs"].values())
    L = [
        "# 表格解析消融实验报告（工单3）",
        "",
        f"- 工单编号：{report['work_order_no']}",
        f"- 生成时间：{report['generated_at']}",
        f"- 题目数：{n}（4 道新题 + 10 道旧题），top_k={report['top_k']}",
        f"- 阈值闸门：{'开' if report['with_gate'] else '关（隔离检索本身的精确度）'}",
        "",
        "## 一、语料规模",
        "",
        "| 配置 | 分块总数 | 其中表格块 |",
        "|---|---|---|",
    ]
    for k, v in report["corpus"].items():
        L.append(f"| {k} | {v['chunks']} | {v['table_chunks']} |")

    L += [
        "",
        "## 二、总体检索精确度",
        "",
        "| 配置 | 关键事实命中率 | 全部命中题数 | 正确页命中率 | MRR | top1 平均依据分 |",
        "|---|---|---|---|---|---|",
    ]
    for c in cfgs:
        s = c["summary"]
        L.append(f"| **{c['key']}** {c['name']} | {s['fact_recall']:.3f} | "
                 f"{s['all_facts_count']}/{n} | {s['gold_page_hit_rate']:.3f} | "
                 f"{s['mrr']:.3f} | {s['avg_top_evidence']:.4f} |")

    d = report["improvement"]
    L += [
        "",
        "### 净效果（T0 → T1，单变量受控）",
        "",
        "| 指标 | 优化前 T0 | 优化后 T1 | 变化 |",
        "|---|---|---|---|",
        f"| 关键事实命中率 | {report['configs']['T0_no_table']['summary']['fact_recall']:.3f} | "
        f"{report['configs']['T1_table']['summary']['fact_recall']:.3f} | +{d['fact_recall_delta']:.3f} |",
        f"| 全部关键事实命中的题目数 | {d['all_facts_before']}/{n} | {d['all_facts_after']}/{n} | "
        f"+{d['all_facts_after']-d['all_facts_before']} |",
        f"| 正确页命中率 | {report['configs']['T0_no_table']['summary']['gold_page_hit_rate']:.3f} | "
        f"{report['configs']['T1_table']['summary']['gold_page_hit_rate']:.3f} | +{d['gold_page_hit_delta']:.3f} |",
        f"| MRR | {report['configs']['T0_no_table']['summary']['mrr']:.3f} | "
        f"{report['configs']['T1_table']['summary']['mrr']:.3f} | +{d['mrr_delta']:.3f} |",
    ]

    if report.get("groups"):
        L += ["", "## 三、分组看收益落在哪里", "",
              "| 分组 | 配置 | 关键事实命中率 | 全部命中 | 正确页命中率 | MRR |",
              "|---|---|---|---|---|---|"]
        for label, g in report["groups"].items():
            for key, s in g.items():
                cnt = s["num_cases"]
                L.append(f"| {label} | {key} | {s['fact_recall']:.3f} | "
                         f"{s['all_facts_count']}/{cnt} | {s['gold_page_hit_rate']:.3f} | {s['mrr']:.3f} |")

    L += ["", "## 四、逐题明细", "",
          "| id | 来源 | 关键事实命中率 T0 → T1 | 正确页排名 T0 → T1 | T0 漏掉的事实 |",
          "|---|---|---|---|---|"]
    t0c = {c["id"]: c for c in report["configs"]["T0_no_table"]["cases"]}
    t1c = {c["id"]: c for c in report["configs"]["T1_table"]["cases"]}
    for qid in t0c:
        x, y = t0c[qid], t1c[qid]
        L.append(f"| {qid} | {x['source']} | {x['fact_recall']:.2f} → {y['fact_recall']:.2f} | "
                 f"{x['gold_rank'] or '未命中'} → {y['gold_rank'] or '未命中'} | "
                 f"{'、'.join(x['missed']) or '—'} |")

    L += ["", "## 五、口径说明", "", f"- {report['metric_note']}", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=config.RETRIEVAL_FINAL_TOP_K)
    ap.add_argument("--with-gate", action="store_true", help="顺带跑带阈值闸门的口径")
    args = ap.parse_args()

    config.ensure_dirs()

    report = run(with_gate=args.with_gate, top_k=args.top_k)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "带闸门" if args.with_gate else "纯检索"
    json_path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_表格解析消融_{tag}_{stamp}.json"
    md_path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_表格解析消融_{tag}_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("\n=== 各层净效果 ===")
    d = report["improvement"]
    print(f"  关键事实命中率 {d['fact_recall_delta']:+.4f} | "
          f"全命中 {d['all_facts_before']} → {d['all_facts_after']} | "
          f"正确页 {d['gold_page_hit_delta']:+.4f} | MRR {d['mrr_delta']:+.4f}")
    for label, g in report.get("groups", {}).items():
        print(f"  [{label}] {g['T0_no_table']['fact_recall']:.3f} → {g['T1_table']['fact_recall']:.3f}"
              f"（正确页 {g['T0_no_table']['gold_page_hit_rate']:.3f} → "
              f"{g['T1_table']['gold_page_hit_rate']:.3f}）")
    print(f"\n[OK] {json_path}")
    print(f"[OK] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
