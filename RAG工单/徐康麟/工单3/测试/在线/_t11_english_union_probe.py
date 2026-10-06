# -*- coding: utf-8 -*-
"""t11 独立复核用：英文问题集**并集**探针（engineer 组 + tester 组）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用途：captain 要求 t11 复核英文侧时使用「两组并集」并分别标注来源。
- engineer 组题面取自 `优化/评估结果/过程日志/_t22_after7.py` 的 ``EN``/``NEG`` 列表（逐字复制）；
- tester 组题面取自 `测试/在线/test_t7_online_english.py`。

输出：`测试/留痕/t11_english_union.json`（含语言、拒答、引用可回溯、english_ratio、降级事件）。
本文件以下划线开头，pytest 不会收集（它是复核工具，不是用例）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent
for extra in (str(TESTS_DIR), str(REPO_ROOT / "研发")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from common import assertions, paths  # noqa: E402
from common.reports import trace_events  # noqa: E402

# ---- engineer 组（逐字取自 _t22_after7.py）----
ENGINEER_EN = [
    ("EN1", "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    ("EN2", "Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    ("EN3", "In which field has Wuhan Xingtu Xinke Electronics Co., Ltd. become an important supplier?", None),
    ("EN4", "What is the registered capital of Wuhan Liyuan Information Technology Co., Ltd.?", ["招股说明书2.pdf"]),
    ("EN5", "How many shares will Wuhan Liyuan Information Technology Co., Ltd. issue?", ["招股说明书2.pdf"]),
    ("EN6", "What is the registered address of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    ("EN7", "What is the main business of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
]
ENGINEER_NEG = [
    ("N-Tesla", "What is the registered capital of Tesla, Inc.?", None),
    ("N-2099", "What are the plans of Wuhan Xingtu Xinke Electronics Co., Ltd. for the 2099 lunar base project?", None),
]
# ---- tester 组（与 测试/在线/test_t7_online_english.py 一致）----
TESTER_EN = [
    ("T-EN1", "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    ("T-EN2", "Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    ("T-EN3", "What were the company's revenues from the military domain during the reporting period?", None),
    ("T-EN4", "According to the prospectus, what are the upstream industries of the electronic information industry?", None),
    ("T-EN5", "In which domain has Wuhan Xingtu Xinke Electronics Co., Ltd. become an important supplier?", None),
    ("T-EN6", "What is the number of shares issued by Wuhan P&S Information Technology Co., Ltd.?", None),
    ("T-EN7", "Which related-party enterprises are not controlled by Wuhan P&S Information Technology Co., Ltd.?", None),
]


def main() -> int:
    """跑两组并集并落留痕。"""
    from app.core.language import english_ratio  # noqa: PLC0415
    from app.core.qa_engine import build_engine  # noqa: PLC0415

    from common import artifacts  # noqa: PLC0415

    chunk_map = {row["chunk_id"]: row for row in artifacts.load_chunks()}
    counts = paths.page_counts() if hasattr(paths, "page_counts") else {
        p.name: __import__("common.pdf_probe", fromlist=["x"]).page_count(str(p))
        for p in paths.discover_pdf_files()}
    from common import pdf_probe  # noqa: PLC0415

    lookup = pdf_probe.lookup_factory()
    engine = build_engine(warmup=True)

    rows: list[dict] = []
    for source, items in (("engineer", ENGINEER_EN + ENGINEER_NEG), ("tester", TESTER_EN)):
        for tag, question, file_names in items:
            before = {n: len(trace_events(n)) for n in ("app.log", "rag_trace.jsonl")}
            answer = engine.ask(question, session_id=f"t11-{source}-{tag}",
                                file_names=file_names, stream=False)
            after = {n: trace_events(n)[before[n]:] for n in before}
            events = [str(r.get("event", "")) for rows_ in after.values() for r in rows_]
            cits = list(getattr(answer, "citations", []) or [])
            cits_ok = []
            for cite in cits:
                check = assertions.check_citation_traceable(
                    cite, page_lookup=lookup, page_counts=counts,
                    discovered_files=list(counts), chunk_lookup=chunk_map)
                cits_ok.append({"file_name": getattr(cite, "file_name", ""),
                                "page": getattr(cite, "page", None),
                                "chunk_id": getattr(cite, "chunk_id", ""), "ok": check.ok})
            rows.append({
                "source": source, "tag": tag, "question": question, "file_names": file_names,
                "language": str(getattr(answer, "language", "")),
                "english_ratio": round(float(english_ratio(str(answer.text))), 4),
                "is_unknown": bool(getattr(answer, "is_unknown", False)),
                "unknown_reason": getattr(answer, "unknown_reason", None),
                "answer_body": str(answer.text)[:200],
                "first_token_ms": float(getattr(answer, "first_token_ms", 0.0) or 0.0),
                "citations": cits_ok,
                "citation_ok": bool(cits_ok) and all(c["ok"] for c in cits_ok),
                "degrade_events": sorted({e for e in events if "degrade" in e or "keep_chinese" in e}),
            })

    out = paths.ensure_trace_dir() / "t11_english_union.json"
    out.write_text(json.dumps({
        "work_order": assertions.WORK_ORDER,
        "ragas": assertions.RAGAS_BANNER,
        "note": "t11 复核：英文问题集并集（engineer 组逐字取 _t22_after7.py；tester 组取本仓用例）",
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"留痕：{out}")
    for row in rows:
        print(f"  [{row['source']}-{row['tag']}] lang={row['language']} ratio={row['english_ratio']} "
              f"unknown={row['is_unknown']}({row['unknown_reason']}) cite_ok={row['citation_ok']} "
              f"degrade={row['degrade_events']}")
        print(f"        A: {row['answer_body'][:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
