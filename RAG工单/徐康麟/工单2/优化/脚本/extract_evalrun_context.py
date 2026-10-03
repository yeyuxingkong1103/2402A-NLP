# -*- coding: utf-8 -*-
"""从工单1 真实日志还原**基线评估运行**（eval_records.json 对应的那一次）逐题送入生成器的上下文页码。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集（真实日志取证）

判据：`环境事实.md` §4.1.0 ——「golden evidence 原文是否落在**本次返回**的 chunk 中」。
日志里 `Generator.generate` 的返回值带 `pages`（本次上下文页码，完整整数列表被截断为前 5 个 + 计数）
与 `primary_chunk_id`、`mode`，据此判定证据页是否真的进入了生成器上下文。

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
OUT = W2 / "优化" / "基线" / "baseline_evalrun_context.json"

WINDOW = ("2026-10-02T19:37", "2026-10-02T19:42")
TS_RE = re.compile(r'"ts"\s*:\s*"([^"]+)"')


def main() -> int:
    golden = [
        json.loads(l) for l in (W1 / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    golden.sort(key=lambda g: int(g["id"]))
    qtext2id = {g["question"]: int(g["id"]) for g in golden}

    ev = json.loads((W1 / "优化" / "评估结果" / "eval_results" / "eval_records.json").read_text(encoding="utf-8"))
    eval_answers = {int(r["question_id"]): r["answer"] for r in ev["records"] if r["mode"] == "rag"}
    eval_first_token = {int(r["question_id"]): r["first_token_ms"] for r in ev["records"] if r["mode"] == "rag"}
    eval_created = {int(r["question_id"]): r["created_at"] for r in ev["records"] if r["mode"] == "rag"}

    chunks = {
        json.loads(l)["chunk_id"]: json.loads(l)
        for l in (W1 / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    norms = lambda t: "".join(ch for ch in (t or "") if ch not in "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000")  # noqa: E731
    ev_pages = {}
    ev_chunk_ids = {}
    for g in golden:
        qid = int(g["id"])
        ids = {cid for cid, c in chunks.items() if norms(g["evidence"]) in norms(c["content"])}
        ev_chunk_ids[qid] = ids
        ev_pages[qid] = sorted({int(chunks[cid]["page"]) for cid in ids})

    current: int | None = None
    gen_calls: dict[int, list[dict]] = {}

    with TRACE.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TS_RE.search(line[:120])
            if not m:
                continue
            ts = m.group(1)
            if not (WINDOW[0] <= ts[:16] <= WINDOW[1]):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            mod, fn, event = obj.get("module", ""), obj.get("function", ""), obj.get("event", "")
            if mod == "app.core.qa_engine" and fn == "QAEngine.ask" and event == "enter":
                args = obj.get("args") or []
                current = qtext2id.get(args[0]) if args else None
                continue
            if current is None:
                continue
            if mod == "app.core.generator" and fn == "Generator.generate" and event == "exit":
                res = obj.get("result") or {}
                if not isinstance(res, dict):
                    continue
                gen_calls.setdefault(current, []).append(
                    {
                        "ts": ts,
                        "mode": res.get("mode"),
                        "answer": res.get("answer"),
                        "pages": res.get("pages"),
                        "primary_chunk_id": res.get("primary_chunk_id"),
                        "retrieved_count": res.get("retrieved_count"),
                        "first_token_ms": res.get("first_token_ms"),
                        "is_matched_to_eval_record": res.get("answer") == eval_answers.get(current),
                    }
                )

    rows = []
    for qid in sorted(gen_calls):
        calls = gen_calls[qid]
        matched = [c for c in calls if c["is_matched_to_eval_record"]]
        chosen = matched[-1] if matched else calls[-1]
        pages = chosen.get("pages") or []
        page_ints = [p for p in pages if isinstance(p, int)]
        rows.append(
            {
                "question_id": qid,
                "eval_record_created_at": eval_created.get(qid),
                "eval_record_first_token_ms": eval_first_token.get(qid),
                "eval_record_answer": eval_answers.get(qid),
                "证据页": ev_pages.get(qid),
                "证据chunk": sorted(ev_chunk_ids.get(qid, set()))[:4],
                "matched_to_eval_record": bool(matched),
                "运行模式": chosen.get("mode"),
                "上下文页(前5+计数)": chosen.get("pages"),
                "上下文条数": chosen.get("retrieved_count"),
                "primary_chunk_id": chosen.get("primary_chunk_id"),
                "证据页是否进入上下文": bool(set(page_ints) & set(ev_pages.get(qid) or [])),
                "生成器调用次数": len(calls),
                "全部调用": [
                    {"ts": c["ts"], "mode": c["mode"], "matched": c["is_matched_to_eval_record"], "pages": c["pages"]}
                    for c in calls
                ],
            }
        )

    OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "工单": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
                    "source": str(TRACE),
                    "eval_records": str(W1 / "优化" / "评估结果" / "eval_results" / "eval_records.json"),
                    "window": list(WINDOW),
                    "说明": "Chosen = 答案与 eval_records 逐字匹配的那次 Generator.generate（即基线评估的真实那次）；"
                    "上下文页仅前 5 个整数被完整记录，其余被日志截断为计数。",
                },
                "per_question": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"{'题号':<6}{'模式':<12}{'证据页':<16}{'基线上下文页':<34}{'证据页进入上下文':<18}{'基线回答':<46}")
    for r in rows:
        print(
            f"{r['question_id']:<6}{str(r['运行模式'])[:10]:<12}{str(r['证据页'])[:14]:<16}"
            f"{str(r['上下文页(前5+计数)'])[:32]:<34}"
            f"{('✅ 是' if r['证据页是否进入上下文'] else '❌ 否'):<16}"
            f"{(r['eval_record_answer'] or '')[:44]}"
        )
    print(f"\n输出: {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[FATAL] 取证失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
