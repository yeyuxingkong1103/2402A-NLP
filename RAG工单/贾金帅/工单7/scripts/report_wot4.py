"""
工单4 交付报告：针对验收问题进行检索，输出「检索到的答案 + 检索精确度」
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单产出物原文要求：
    「针对如下问题进行检索，**显示检索到的答案及检索精确度**」

本脚本把这句话落成一份可核对的文件：**16 道题 × 3 条链路**

    ① 对照（PDF 图像不解析）  —— 工单4 要证明的「优化前」
    ② 主线（图像多模态解析）  —— 工单4 要交付的「优化后」（含低信息图排序惩罚）
    ③ 主线 + CLIP 跨模态召回  —— 图像解析的两条路线里另一条（以文搜图）
    （②③ 用的都是交付配置：retrieve 默认 use_image_prior=True）

每题都给出
    ① 召回片段的（文档 / 页码 / 类型 / 依据分 / CLIP 相似度）清单 —— 「从 pdf 哪里检索到的」
    ② **检索精确度**三项客观口径：
         - 关键事实覆盖率 must_have 命中率（人工标注）
         - 正确页召回（gold 页是否被召回）
         - 精确率 P@k / MRR（召回块里有多少真的落在标注答案页上）
    ③ 各链路生成的答案 —— 「检索到的答案」
    ④ 正确答案在 PDF 中的定位（表格名 / 图名 + 页码）—— 工单备注「分析答案在 pdf 中的定位」

用法：
    python scripts/report_wot4.py                    # 16 题全跑（调大模型，约 3~5 分钟）
    python scripts/report_wot4.py --id 5 --id 6      # 只跑新增的两道图像题
    python scripts/report_wot4.py --no-answer        # 不调大模型，只出检索与命中（零成本）
    python scripts/report_wot4.py --no-clip          # 不跑 CLIP 链路（省一次模型加载）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm, rag  # noqa: E402
from src.index_store import KnowledgeBase  # noqa: E402
from src.query_norm import normalize_query  # noqa: E402
from src.retriever import build_context, retrieve  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO_IMAGE


def _norm(s: str) -> str:
    """归一化：NFKC（全角→半角、「−」→「-」）+ 去掉所有空白。

    为什么必须去空白：PDF 文字层与多模态描述的空格口径不一致 ——
    正文层出「2008年中国 IC 市场」，图描述出「2008 年中国IC 市场」；
    「IC卡」与「IC 卡」也是同一个东西。按字符比对前先把空白抹平，
    否则会有一批「明明答对了却被判漏」的假阴性。
    """
    return "".join(unicodedata.normalize("NFKC", s or "").split())


def _count_span(text: str, token: str) -> int:
    return text.count(token)


def fact_hits(contexts: list[str], must_have: list[str]) -> dict:
    joined = _norm("\n".join(contexts))
    hit = [k for k in must_have if _norm(k) in joined]
    return {
        "must_have": must_have,
        "hit": hit,
        "missed": [k for k in must_have if k not in hit],
        "rate": round(len(hit) / len(must_have), 4) if must_have else None,
    }


def counts_hits(contexts: list[str], counts: list[str]) -> dict:
    """「层级数量」这类关键事实：判断字符是否**出现**（数字 4 / 6 是否读到）。

    这种判定天然弱（正文里到处是 4 和 6），因此单独成一项统计，
    **不并进 must_have 里**，避免它把覆盖率指标抬得虚高。
    """
    joined = _norm("\n".join(contexts))
    hit = [c for c in (counts or []) if _norm(c) in joined]
    return {"expected": counts or [], "hit": hit}


def relation_hits(contexts: list[str], relations: list[str]) -> dict:
    """「归属关系」命中情况（工单4 图像题的关键事实形态）。

    id=5 的答案不是"有哪些部门"这种**名词清单**，而是"谁挂在谁下面"这种**关系**：
    名词清单在图内散字里也能凑出来，关系却只存在于连接线的几何拓扑中。
    因此单独列一路，判据用**带箭头的完整关系串**——箭头是图像解析独有的产物。
    """
    joined = _norm("\n".join(contexts))
    hit = [r for r in (relations or []) if _norm(r) in joined]
    return {
        "expected": relations or [],
        "hit": hit,
        "missed": [r for r in (relations or []) if r not in hit],
        "rate": round(len(hit) / len(relations), 4) if relations else None,
    }


def _gold_pages(case: dict) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for g in case.get("gold") or []:
        for p in g.get("pages") or []:
            out.add((g.get("doc") or "", int(p)))
    return out


def gold_rank(items: list[dict], gold: list[dict]) -> int | None:
    for it in items:
        span = set(range(it["page"], (it["page_end"] or it["page"]) + 1)) if it["page"] else set()
        for g in gold:
            if g.get("doc") and g["doc"] != it["doc_key"]:
                continue
            if span & set(g.get("pages") or []):
                return it["rank"]
    return None


def precision_at_k(items: list[dict], gold: list[dict], k: int) -> float:
    """P@k：召回的前 k 块里，有多少块落在人工标注的答案页上。

    这是「检索精确度」最直白的口径 —— 依据分高不高是一回事，
    **捞上来的东西到底是不是答案所在的那一页**是另一回事。
    """
    if not items or k <= 0:
        return 0.0
    rel = 0
    for it in items[:k]:
        span = set(range(it["page"], (it["page_end"] or it["page"]) + 1)) if it["page"] else set()
        if any((g.get("doc") in (None, "", it["doc_key"])) and (span & set(g.get("pages") or []))
               for g in gold):
            rel += 1
    return round(rel / min(k, len(items)), 4)


def run_one(question: str, top_k: int, kb: KnowledgeBase, with_answer: bool,
            use_clip: bool) -> dict:
    """跑一条链路：检索 → （可选）生成。返回结构化结果。"""
    t0 = time.perf_counter()
    r = retrieve(question, kb=kb, top_k=top_k, use_clip=use_clip)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    items = list(r.items)
    out = {
        "items": [
            {"rank": i, "doc_key": it.doc_key, "doc": it.doc, "page": it.page,
             "page_end": it.page_end, "type": it.type, "section": it.section,
             "evidence": round(it.evidence, 4), "score": round(it.score, 4),
             "clip_sim": round(it.clip_sim, 4),
             "image": it.image, "fig_type": it.fig_type, "caption": it.caption,
             "preview": it.text[:200].replace("\n", " ")}
            for i, it in enumerate(items, 1)
        ],
        "contexts": [it.text for it in items],
        "n_items": len(items),
        "n_table_chunks": sum(1 for it in items if it.type == "table"),
        "n_image_chunks": sum(1 for it in items if it.type == "image"),
        "gated": r.gated,
        "retrieval_ms": round(retrieval_ms, 2),
        "doc_hint": r.trace.get("doc_hint") or [],
        "trace_steps": r.trace.get("steps") or [],
    }
    if with_answer and items:
        ctx = build_context(items, max_chars=5200)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, action="append", default=[])
    ap.add_argument("--top-k", type=int, default=config.RETRIEVAL_FINAL_TOP_K)
    ap.add_argument("--no-answer", action="store_true", help="不调大模型，只出检索与命中")
    ap.add_argument("--no-clip", action="store_true", help="不跑 CLIP 链路")
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

    kb_img = KnowledgeBase.get()
    _ = kb_img.bm25
    _ = kb_img.ubiquitous_phrases
    kb_no = KnowledgeBase.get_noimage()
    _ = kb_no.bm25
    print(f"[索引] 图像解析 {len(kb_img.chunks)} 块 / 不解析 {len(kb_no.chunks)} 块\n")

    links: list[tuple[str, KnowledgeBase, bool]] = [
        ("no_image", kb_no, False),        # 对照：图像不解析，也不走 CLIP
        ("image", kb_img, False),          # 主线：图像多模态解析（含低信息图排序惩罚）
        ("image_clip", kb_img, True),      # 主线 + CLIP 跨模态召回
    ]
    if args.no_clip:
        links = links[:2]

    cases = []
    t_all = time.perf_counter()
    for n, q in enumerate(questions, 1):
        qid, question = q["id"], q["question"]
        must = q.get("must_have") or []
        counts = q.get("must_have_counts") or []
        gold = q.get("gold") or []
        print(f"[{n}/{len(questions)}] id={qid} {question[:40]}…")

        rec = {
            "id": qid, "source": q.get("source", ""), "question": question,
            "reference": q.get("reference", ""), "gold": gold,
            "gold_note": q.get("gold_note", ""),
            "image_only": bool(q.get("image_only")),
        }
        for label, kb, use_clip in links:
            r = run_one(normalize_query(question), args.top_k, kb, not args.no_answer, use_clip)
            r["fact_hits"] = fact_hits(r["contexts"], must)
            r["counts_hits"] = counts_hits(r["contexts"], counts)
            r["relation_hits"] = relation_hits(r["contexts"],
                                               q.get("must_have_relations") or [])
            r["gold_rank"] = gold_rank(r["items"], gold)
            r["precision_at_k"] = precision_at_k(r["items"], gold, args.top_k)
            r.pop("contexts", None)          # 上下文太长，报告里只留预览
            rec[label] = r
            rel = r["relation_hits"]
            rel_txt = (f" | 归属关系 {len(rel['hit'])}/{len(rel['expected'])}"
                       if rel["expected"] else "")
            print(f"      {label:<11} 召回 {r['n_items']}（表 {r['n_table_chunks']} / 图 {r['n_image_chunks']}）"
                  f" | 关键事实 {len(r['fact_hits']['hit'])}/{len(must)}"
                  f"{rel_txt}"
                  f" | P@k {r['precision_at_k']:.2f} | 正确页排名 {r['gold_rank'] or '未命中'}")
        cases.append(rec)

    report = {
        "work_order_no": WORK_ORDER_NO,
        "work_order_short": config.WORK_ORDER_SHORT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_questions": len(cases),
        "top_k": args.top_k,
        "elapsed_seconds": round(time.perf_counter() - t_all, 1),
        "links": [lab for lab, _, _ in links],
        "documents": [{"key": d["key"], "name": d["name"], "file": d["file"]} for d in config.DOCS],
        "summary": summarize(cases, [lab for lab, _, _ in links]),
        "cases": cases,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_检索结果与精确度_{stamp}.json"
    md_path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_检索结果与精确度_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("\n" + summary_table(report))
    print(f"\n[OK] {json_path}")
    print(f"[OK] {md_path}")
    return 0


def summarize(cases: list[dict], links: list[str]) -> dict:
    """按链路 × 题型分层汇总。

    分层是必要的：图像题（id 5/6）只有 2 道，混进 14 道表格/正文题里算平均，
    优化效果会被稀释到看不出来；反过来只报图像题又会显得像挑数据。
    两个口径都给出，读者自己判断。
    """
    def block(sub: list[dict]) -> dict:
        out: dict = {}
        for lab in links:
            rs = [c[lab] for c in sub]
            if not rs:
                continue
            n_must = sum(len(r["fact_hits"]["must_have"]) for r in rs)
            n_hit = sum(len(r["fact_hits"]["hit"]) for r in rs)
            n_rel = sum(len(r["relation_hits"]["expected"]) for r in rs)
            n_rel_hit = sum(len(r["relation_hits"]["hit"]) for r in rs)
            out[lab] = {
                "n": len(rs),
                "relation_coverage": round(n_rel_hit / n_rel, 4) if n_rel else None,
                "fact_coverage": round(n_hit / n_must, 4) if n_must else None,
                "gold_recall": round(sum(1 for r in rs if r["gold_rank"]) / len(rs), 4),
                "precision_at_k": round(sum(r["precision_at_k"] for r in rs) / len(rs), 4),
                "avg_retrieval_ms": round(sum(r["retrieval_ms"] for r in rs) / len(rs), 1),
                "avg_items": round(sum(r["n_items"] for r in rs) / len(rs), 2),
            }
        return out

    return {
        "all": block(cases),
        "image_questions": block([c for c in cases if c["image_only"]]),
        "legacy_questions": block([c for c in cases if not c["image_only"]]),
    }


def _rng(gold: list[dict]) -> str:
    return "；".join(f"{g.get('doc')} p{'/'.join(str(p) for p in g.get('pages') or [])}"
                     for g in gold) or "—"


def summary_table(report: dict) -> str:
    links = report["links"]
    L = ["", "== 汇总（检索精确度）==", ""]
    for scope, key in (("全部题目", "all"), ("图像题（id 5/6）", "image_questions"),
                       ("原有题目（14 道）", "legacy_questions")):
        blk = report["summary"].get(key) or {}
        L.append(f"-- {scope} --")
        L.append(f"{'链路':<24}{'关键事实覆盖':<14}{'正确页召回':<12}{'P@k':<10}{'检索ms':<10}")
        for lab in links:
            m = blk.get(lab)
            if not m:
                continue
            L.append(f"{lab:<24}{m['fact_coverage']:<14}{m['gold_recall']:<12}"
                     f"{m['precision_at_k']:<10}{m['avg_retrieval_ms']:<10}")
        L.append("")
    return "\n".join(L)


def to_markdown(report: dict) -> str:
    links = report["links"]
    n = report["num_questions"]
    L = [
        "# 工单4 检索结果与检索精确度报告",
        "",
        f"- 工单编号：{report['work_order_no']}",
        f"- 生成时间：{report['generated_at']}",
        f"- 题目数：{n}，top_k={report['top_k']}，总耗时 {report['elapsed_seconds']}s",
        f"- 对比链路：{' / '.join(links)}",
        "",
        "> 口径说明：`关键事实覆盖` = 人工标注的关键事实（must_have）在召回片段中的覆盖率；",
        "> `正确页召回` = 人工标注的答案页（gold）被召回的比例；",
        "> `P@k` = 前 k 条召回中落在答案页上的比例。三者都是**只看检索、不看生成**的客观指标。",
        "",
        "## 语料",
        "",
        "| 文档键 | 文档 | 文件 |",
        "|---|---|---|",
    ]
    for d in report["documents"]:
        L.append(f"| {d['key']} | {d['name']} | {d['file']} |")

    L += ["", "## 汇总", ""]
    for scope, key in (("全部题目", "all"), ("图像题（工单4 新增，id 5/6）", "image_questions"),
                       ("原有题目（工单1/2/3，14 道）", "legacy_questions")):
        blk = report["summary"].get(key) or {}
        L += [f"### {scope}", "",
              "| 链路 | 关键事实覆盖 | 正确页召回 | P@k | 平均检索耗时 | 平均召回条数 |",
              "|---|---|---|---|---|---|"]
        for lab in links:
            m = blk.get(lab)
            if not m:
                continue
            L.append(f"| {lab} | {m['fact_coverage']} | {m['gold_recall']} | "
                     f"{m['precision_at_k']} | {m['avg_retrieval_ms']} ms | {m['avg_items']} |")
        L.append("")

    L += ["## 逐题汇总", "",
          "| id | 来源 | 正确答案定位 | 关键事实（无图 / 有图 / +CLIP） | 正确页排名（无图 / 有图 / +CLIP） |",
          "|---|---|---|---|---|"]

    def _cov(r: dict) -> str:
        return f"{len(r['fact_hits']['hit'])}/{len(r['fact_hits']['must_have'])}"

    for c in report["cases"]:
        L.append(
            f"| {c['id']} | {c['source']} | {_rng(c['gold'])} | "
            + " / ".join(_cov(c[lab]) for lab in links)
            + " | " + " / ".join(str(c[lab]["gold_rank"] or "未命中") for lab in links) + " |"
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
            L += [f"**答案在 PDF 中的定位**：{c['gold_note']}", ""]
        for lab in links:
            r = c[lab]
            fh = r["fact_hits"]
            L += [
                f"#### {lab} —— 召回 {r['n_items']} 条"
                f"（表格块 {r['n_table_chunks']} / 图像块 {r['n_image_chunks']}），"
                f"检索 {r['retrieval_ms']} ms",
                "",
                f"- 关键事实命中：{len(fh['hit'])}/{len(fh['must_have'])}"
                + (f"（漏：{'、'.join(fh['missed'])}）" if fh["missed"] else "（全中）"),
                f"- 检索精确度：P@k = {r['precision_at_k']}，正确页排名 = {r['gold_rank'] or '未命中'}",
            ]
            rel = r["relation_hits"]
            if rel["expected"]:
                L += [f"- 归属关系命中：{len(rel['hit'])}/{len(rel['expected'])}"
                      + (f"（漏：{'；'.join(rel['missed'])}）" if rel["missed"] else "（全中）")]
            L.append("")
            for it in r["items"]:
                tag = {"table": "表格", "image": "图像", "text": "正文"}.get(it["type"], it["type"])
                extra = f" CLIP {it['clip_sim']}" if it["clip_sim"] else ""
                L.append(f"  - {it['rank']}. `{it['doc_key']}` 第{it['page']}页 [{tag}] "
                         f"依据分 {it['evidence']}{extra} ｜ {it['preview'][:110]}")
                if it["type"] == "image" and it["image"]:
                    L.append(f"    - 图：`{it['image']}`（{it['fig_type']}；图题：{it['caption'] or '无'}）")
            L.append("")
            if r.get("answer"):
                L += [f"> {r['answer'].replace(chr(10), chr(10) + '> ')}", ""]
        L += ["---", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
