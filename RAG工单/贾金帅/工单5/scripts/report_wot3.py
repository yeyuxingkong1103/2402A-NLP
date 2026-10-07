"""
工单3 交付报告：针对验收问题进行检索，输出「检索到的答案 + 检索精确度」
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单产出物原文要求：
    「针对如下问题进行检索，**显示检索到的答案及检索精确度**」

本脚本把这句话落成一份可核对的文件：14 道题 × 2 条链路（表格不结构化 / 结构化），
每题都给出
    ① 召回片段的（文档 / 页码 / 类型 / 依据分）清单 —— 「从 pdf 哪里检索到的」
    ② 关键事实命中清单 —— 「检索精确度」的客观口径（人工标注的 must_have 覆盖情况）
    ③ 两条链路各自生成的答案 —— 「检索到的答案」
    ④ 正确答案在 PDF 中的定位（表格名 + 页码）—— 工单备注「分析答案在 pdf 中的定位」

用法：
    python scripts/report_wot3.py                # 14 题全跑（约 1~2 分钟）
    python scripts/report_wot3.py --id 2 --id 3  # 只跑指定题号
    python scripts/report_wot3.py --no-answer    # 不调大模型，只出检索与命中（零成本）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm, rag  # noqa: E402
from src.index_store import KnowledgeBase  # noqa: E402
from src.query_norm import normalize_query  # noqa: E402
from src.retriever import build_context, retrieve  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO_TABLE


def run_one(question: str, top_k: int, kb: KnowledgeBase, with_answer: bool) -> dict:
    """跑一条链路：检索 → （可选）生成。返回结构化结果。"""
    t0 = time.perf_counter()
    r = retrieve(question, kb=kb, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    items = list(r.items)
    contexts = [it.text for it in items]
    out = {
        "items": [
            {"rank": i, "doc_key": it.doc_key, "doc": it.doc, "page": it.page,
             "page_end": it.page_end, "type": it.type, "section": it.section,
             "evidence": round(it.evidence, 4), "score": round(it.score, 4),
             "preview": it.text[:160].replace("\n", " ")}
            for i, it in enumerate(items, 1)
        ],
        "contexts": contexts,
        "n_items": len(items),
        "n_table_chunks": sum(1 for it in items if it.type == "table"),
        "gated": r.gated,
        "retrieval_ms": round(retrieval_ms, 2),
        "doc_hint": r.trace.get("doc_hint") or [],
    }
    if with_answer and items:
        ctx = build_context(items, max_chars=4800)
        t = time.perf_counter()
        text = llm.chat([
            {"role": "system", "content": rag._RAG_SYSTEM},
            {"role": "user",
             "content": f"【资料片段】\n{ctx}\n\n【问题】\n{question}\n\n请依据上述资料片段回答。"},
        ])
        out["answer"] = text.strip()
        out["generation_ms"] = round((time.perf_counter() - t) * 1000, 2)
        out["total_ms"] = round(retrieval_ms + out["generation_ms"], 2)
    elif not items:
        out["answer"] = rag.NO_EVIDENCE_TEXT
        out["total_ms"] = out["retrieval_ms"]
    return out


def fact_hits(contexts: list[str], must_have: list[str]) -> dict:
    joined = "\n".join(contexts)
    hit = [k for k in must_have if k in joined]
    return {
        "must_have": must_have, "hit": hit,
        "missed": [k for k in must_have if k not in hit],
        "rate": round(len(hit) / len(must_have), 4) if must_have else None,
    }


def gold_rank(items: list[dict], gold: list[dict]) -> int | None:
    for it in items:
        span = set(range(it["page"], (it["page_end"] or it["page"]) + 1)) if it["page"] else set()
        for g in gold:
            if g.get("doc") and g["doc"] != it["doc_key"]:
                continue
            if span & set(g.get("pages") or []):
                return it["rank"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, action="append", default=[])
    ap.add_argument("--top-k", type=int, default=config.RETRIEVAL_FINAL_TOP_K)
    ap.add_argument("--no-answer", action="store_true", help="不调大模型，只出检索与命中")
    args = ap.parse_args()

    config.ensure_dirs()
    questions = json.loads((config.EVAL_DIR / config.EVAL_DATASET).read_text(encoding="utf-8"))
    if args.id:
        want = set(args.id)
        questions = [q for q in questions if q["id"] in want]
    if not questions:
        print("[FAIL] 没有匹配的题目")
        return 2

    print(f"[工单] {WORK_ORDER_NO}")
    print(f"[题量] {len(questions)} 题，top_k={args.top_k}"
          f"，生成：{'关（--no-answer）' if args.no_answer else '开'}\n")

    kb_tbl = KnowledgeBase.get()
    _ = kb_tbl.bm25
    _ = kb_tbl.ubiquitous_phrases
    kb_not = KnowledgeBase.get_notable()
    _ = kb_not.bm25
    print(f"[索引] 结构化 {len(kb_tbl.chunks)} 块 / 不结构化 {len(kb_not.chunks)} 块\n")

    cases = []
    t_all = time.perf_counter()
    for i, q in enumerate(questions, 1):
        qid, question = q["id"], q["question"]
        must = q.get("must_have") or []
        gold = q.get("gold") or []
        print(f"[{i}/{len(questions)}] id={qid} {question[:38]}…")

        rec = {
            "id": qid, "source": q.get("source", ""), "question": question,
            "reference": q.get("reference", ""),
            "gold": gold, "gold_note": q.get("gold_note", ""),
        }
        for label, kb in (("no_table", kb_not), ("table", kb_tbl)):
            r = run_one(normalize_query(question), args.top_k, kb, not args.no_answer)
            r["fact_hits"] = fact_hits(r["contexts"], must)
            r["gold_rank"] = gold_rank(r["items"], gold)
            r.pop("contexts", None)          # 上下文太长，报告里只留预览
            rec[label] = r
            print(f"      {label:<9} 召回 {r['n_items']} 条（表格块 {r['n_table_chunks']}）"
                  f" | 关键事实 {(r['fact_hits']['hit'] and len(r['fact_hits']['hit'])) or 0}"
                  f"/{len(must)} | 正确页排名 {r['gold_rank'] or '未命中'}")
        cases.append(rec)

    report = {
        "work_order_no": WORK_ORDER_NO,
        "work_order_short": config.WORK_ORDER_SHORT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_questions": len(cases),
        "top_k": args.top_k,
        "elapsed_seconds": round(time.perf_counter() - t_all, 1),
        "documents": [{"key": d["key"], "name": d["name"], "file": d["file"]} for d in config.DOCS],
        "cases": cases,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_检索结果对照_{stamp}.json"
    md_path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_检索结果对照_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print(f"\n[OK] {json_path}")
    print(f"[OK] {md_path}")
    return 0


def to_markdown(report: dict) -> str:
    n = report["num_questions"]
    L = [
        "# 工单3 检索结果对照报告",
        "",
        f"- 工单编号：{report['work_order_no']}",
        f"- 生成时间：{report['generated_at']}",
        f"- 题目数：{n}，top_k={report['top_k']}，总耗时 {report['elapsed_seconds']}s",
        "",
        "## 语料",
        "",
        "| 文档键 | 文档 | 文件 |",
        "|---|---|---|",
    ]
    for d in report["documents"]:
        L.append(f"| {d['key']} | {d['name']} | {d['file']} |")

    L += ["", "## 汇总", "",
          "| id | 来源 | 正确答案定位（表格 / 页码） | 不结构化：关键事实 | 结构化：关键事实 | 不结构化：正确页排名 | 结构化：正确页排名 |",
          "|---|---|---|---|---|---|---|"]

    def rng(gold: list[dict]) -> str:
        return "；".join(f"{g.get('doc')} p{g.get('pages')}" for g in gold) or "—"

    for c in report["cases"]:
        nt, tb = c["no_table"], c["table"]
        L.append(
            f"| {c['id']} | {c['source']} | {rng(c['gold'])} | "
            f"{(nt['fact_hits']['hit'] and len(nt['fact_hits']['hit'])) or 0}/{len(nt['fact_hits']['must_have'])} | "
            f"{(tb['fact_hits']['hit'] and len(tb['fact_hits']['hit'])) or 0}/{len(tb['fact_hits']['must_have'])} | "
            f"{nt['gold_rank'] or '未命中'} | {tb['gold_rank'] or '未命中'} |"
        )

    L += ["", "## 逐题明细", ""]
    for c in report["cases"]:
        L += [
            f"### id={c['id']}（{c['source']}）",
            "",
            f"**问题**：{c['question']}",
            "",
            f"**参考答案**：{c['reference']}",
            "",
        ]
        if c.get("gold_note"):
            L += [f"**答案定位**：{c['gold_note']}", ""]
        for key, label in (("no_table", "优化前 · 表格不结构化"), ("table", "优化后 · 表格结构化")):
            r = c[key]
            fh = r["fact_hits"]
            L += [
                f"**{label}** —— 召回 {r['n_items']} 条（表格块 {r['n_table_chunks']}），"
                f"检索 {r['retrieval_ms']} ms，正确页排名 {r['gold_rank'] or '未命中'}",
                "",
                f"- 关键事实命中：{len(fh['hit'])}/{len(fh['must_have'])}"
                + (f"（漏：{'、'.join(fh['missed'])}）" if fh["missed"] else "（全中）"),
                "",
            ]
            for it in r["items"]:
                L.append(f"  - {it['rank']}. `{it['doc_key']}` 第{it['page']}页 "
                         f"[{it['type']}] 依据分 {it['evidence']} ｜ {it['preview'][:90]}")
            L.append("")
            if r.get("answer"):
                L += [f"> {r['answer'].replace(chr(10), chr(10) + '> ')}", ""]
        L += ["---", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
