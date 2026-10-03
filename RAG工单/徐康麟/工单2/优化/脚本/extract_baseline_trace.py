# -*- coding: utf-8 -*-
"""从工单1 真实运行日志（logs/rag_trace.jsonl）还原基线逐题「实际送入生成器的上下文」。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集（真实日志取证）

为什么需要：`环境事实.md` §4.1.0 定的判据是「golden evidence 原文是否落在**本次返回**的
chunk 中」，而不是「答案引用了哪一页」。日志里 `Retriever.retrieve` / `retrieve_multi`
的返回值深度字段被截断成 '...'，但**分数字段保留完整**，因此可以：
    1. 用 `BM25Index.search` 的完整 chunk_id+分数列表；
    2. 与 `Retriever.retrieve` 返回项里的 `bm25_score` 精确对齐（浮点等值），
反推某个证据 chunk 是否真的进入了返回集合。这是可复核的一手证据链。

只读工单1，绝不写入。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

W1 = Path(r"E:\gao6gongdan\工单1")
W2 = Path(r"E:\gao6gongdan\工单2")
TRACE = W1 / "logs" / "rag_trace.jsonl"
OUT = W2 / "优化" / "基线" / "baseline_trace_retrieval.json"

# 基线评估（eval_records.json）的真实运行窗口
WINDOW_START = "2026-10-02T19:37"
WINDOW_END = "2026-10-02T19:42"

TS_RE = re.compile(r'"ts"\s*:\s*"([^"]+)"')
WS_RE = re.compile(r"\s+")
PUNCT_TO_STRIP = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"


def norm_punct(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in PUNCT_TO_STRIP)


def main() -> int:
    golden = [
        json.loads(line)
        for line in (W1 / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    golden.sort(key=lambda g: int(g["id"]))
    ev_norm = {int(g["id"]): norm_punct(g["evidence"]) for g in golden}
    chunks = {
        json.loads(l)["chunk_id"]: json.loads(l)
        for l in (W1 / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    # 证据 chunk 候选集：证据原文（严格口径 B）落在哪些块里
    evidence_chunks: dict[int, set[str]] = {}
    for g in golden:
        needle = ev_norm[int(g["id"])]
        evidence_chunks[int(g["id"])] = {
            cid for cid, c in chunks.items() if needle and needle in norm_punct(c.get("content", ""))
        }

    records: dict[int, dict] = {}
    current_q: int | None = None
    scanned = 0

    with TRACE.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TS_RE.search(line[:120])
            if not m:
                continue
            ts = m.group(1)
            if ts[:16] < WINDOW_START or ts[:16] > WINDOW_END:
                continue
            scanned += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            module = obj.get("module", "")
            function = obj.get("function", "")
            event = obj.get("event", "")

            # 问题定位：QAEngine.ask enter 的 args[0] 即原问题
            if module == "app.core.qa_engine" and function == "QAEngine.ask" and event == "enter":
                args = obj.get("args") or []
                qtext = args[0] if args else ""
                current_q = None
                for g in golden:
                    if g["question"] == qtext:
                        current_q = int(g["id"])
                        records.setdefault(
                            current_q,
                            {
                                "question": g["question"],
                                "bm25_searches": [],
                                "retrieve_exits": [],
                                "retrieve_multi_exits": [],
                                "generate_contexts_meta": [],
                                "answer": None,
                                "citations": [],
                            },
                        )
                        break
                continue

            if current_q is None:
                continue
            rec = records[current_q]

            if module == "app.core.bm25_index" and function == "BM25Index.search" and event == "exit":
                res = obj.get("result") or []
                pairs = [(r[0], r[1]) for r in res if isinstance(r, list) and len(r) == 2 and isinstance(r[0], str)]
                rec["bm25_searches"].append({"query": (obj.get("args") or [""])[0] if obj.get("args") else "", "hits": pairs})

            elif module == "app.core.retriever" and function == "Retriever.retrieve" and event == "exit":
                res = obj.get("result")
                items = []
                if isinstance(res, list):
                    for it in res:
                        if isinstance(it, dict):
                            items.append(
                                {
                                    "score": it.get("score"),
                                    "vector_score": it.get("vector_score"),
                                    "bm25_score": it.get("bm25_score"),
                                    "rerank_score": it.get("rerank_score"),
                                    "source": it.get("source"),
                                    "page": (it.get("chunk") or {}).get("page") if isinstance(it.get("chunk"), dict) else None,
                                }
                            )
                rec["retrieve_exits"].append({"query": (obj.get("args") or [""])[0] if obj.get("args") else "", "items": items})

            elif module == "app.core.retriever" and function == "Retriever.retrieve_multi" and event == "exit":
                res = obj.get("result")
                items = []
                if isinstance(res, list):
                    for it in res:
                        if isinstance(it, dict):
                            items.append(
                                {
                                    "score": it.get("score"),
                                    "vector_score": it.get("vector_score"),
                                    "bm25_score": it.get("bm25_score"),
                                    "source": it.get("source"),
                                }
                            )
                rec["retrieve_multi_exits"].append({"items": items})

            elif module == "app.core.generator" and function == "Generator.generate" and event == "enter":
                kwargs = obj.get("kwargs") or {}
                ctxs = kwargs.get("contexts")
                rec["generate_contexts_meta"].append(
                    {"question": kwargs.get("question"), "context_count": len(ctxs) if isinstance(ctxs, list) else None}
                )

            elif module == "app.core.qa_engine" and function == "QAEngine.ask" and event == "exit":
                res = obj.get("result") or {}
                if isinstance(res, dict):
                    rec["answer"] = res.get("answer")
                    cites = res.get("citations") or []
                    rec["citations"] = [
                        {"page": c.get("page"), "chunk_id": c.get("chunk_id")} for c in cites if isinstance(c, dict)
                    ]

    # ---- 对齐分析 ----
    rows = []
    for qid, rec in sorted(records.items()):
        ev_ids = evidence_chunks.get(qid, set())
        bm25_pairs: list[tuple[str, float]] = []
        for s in rec["bm25_searches"]:
            bm25_pairs.extend(s["hits"])

        # 返回项里的 bm25_score 集合（用于判断证据 chunk 是否进入返回集合）
        returned_bm25_scores = set()
        for ex in rec["retrieve_exits"]:
            for it in ex["items"]:
                if isinstance(it.get("bm25_score"), (int, float)):
                    returned_bm25_scores.add(round(float(it["bm25_score"]), 9))
        for ex in rec["retrieve_multi_exits"]:
            for it in ex["items"]:
                if isinstance(it.get("bm25_score"), (int, float)):
                    returned_bm25_scores.add(round(float(it["bm25_score"]), 9))

        bm25_hits_for_evidence = [
            {"chunk_id": cid, "bm25_score": round(sc, 9), "bm25_rank": i + 1}
            for i, (cid, sc) in enumerate(bm25_pairs)
            if cid in ev_ids
        ]
        evidence_in_returned_context = any(
            round(h["bm25_score"], 9) in returned_bm25_scores for h in bm25_hits_for_evidence
        )

        # 答案引用页的 chunk 是否含证据原文
        cited_pages = sorted({c["page"] for c in rec["citations"] if isinstance(c.get("page"), int)})
        evidence_pages = sorted({int(chunks[cid]["page"]) for cid in ev_ids if cid in chunks})
        cited_page_has_evidence = bool(set(cited_pages) & set(evidence_pages))

        rows.append(
            {
                "question_id": qid,
                "question": rec["question"],
                "bm25_searches": rec["bm25_searches"],
                "retrieve_exit_sizes": [len(ex["items"]) for ex in rec["retrieve_exits"]],
                "retrieve_multi_sizes": [len(ex["items"]) for ex in rec["retrieve_multi_exits"]],
                "generate_context_counts": [g["context_count"] for g in rec["generate_contexts_meta"]],
                "evidence_chunk_ids": sorted(ev_ids),
                "evidence_pages": evidence_pages,
                "bm25_hits_for_evidence": bm25_hits_for_evidence,
                "evidence_in_returned_context_by_bm25_score": evidence_in_returned_context,
                "answer": rec["answer"],
                "citation_pages": cited_pages,
                "citation_page_contains_evidence": cited_page_has_evidence,
            }
        )

    OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "工单": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
                    "source": str(TRACE),
                    "window": [WINDOW_START, WINDOW_END],
                    "scanned_lines": scanned,
                    "证据 chunk 口径": "严格口径（整段 evidence，归一化 B）落在 chunk 正文中",
                    "context判定方法": "BM25 检索日志给出完整 chunk_id+分数；retrieve 返回项的 chunk_id 被日志截断，"
                    "但 bm25_score 完整保留，故用浮点等值对齐判断证据 chunk 是否进入返回集合",
                },
                "per_question": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"窗口内扫描行数: {scanned}；命中题目数: {len(rows)}")
    print(f"\n{'题号':<6}{'证据页':<14}{'BM25中证据chunk':<40}{'证据进入返回集合':<18}{'引用页':<14}{'引用页含证据':<14}")
    for r in rows:
        ev = ",".join(f"{h['chunk_id']}#{h['bm25_rank']}" for h in r["bm25_hits_for_evidence"]) or "—"
        print(
            f"{r['question_id']:<6}{str(r['evidence_pages'])[:12]:<14}{ev[:38]:<40}"
            f"{'✅ 是' if r['evidence_in_returned_context_by_bm25_score'] else '❌ 否/无记录':<16}"
            f"{str(r['citation_pages'])[:12]:<14}{'✅' if r['citation_page_contains_evidence'] else '❌':<14}"
        )
    print(f"\n输出: {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[FATAL] 日志取证失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
