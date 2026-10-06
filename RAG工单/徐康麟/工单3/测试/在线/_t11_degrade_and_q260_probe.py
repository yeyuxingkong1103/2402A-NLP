# -*- coding: utf-8 -*-
"""t11 补充实测：① 英文降级事件是否真的存在、在什么条件下触发；② 题 260 首字 2909.6 ms 的成因。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

背景：
    * 代码里确有降级机制（`generator.py` 的 ``ENGLISH_RATIO_MIN = 0.5``，低于阈值写
      ``generation.language_degraded`` WARNING，回退路径含 ``keep_chinese``）；
    * 但我在英文并集探针中**未观测到**该事件，且当时把 ``english_ratio`` 算在**整段渲染文本**
      （含末尾 ``References：[…]`` 中文引用行）上 → 可能把比例算低了，属**测量口径疑似失真**。
    * 本题用 engineer 的 EN3/EN4/EN5/EN6（**限定 ``file_names``，与 engineer 同条件**）复测，
      并把 ``english_ratio`` **分别**算在「答案正文」与「整段文本」上，同时按 ``trace_id`` 扫
      ``rag_trace.jsonl`` / ``app.log``，给出降级事件是否存在及其 ``ratio``/``threshold``/``path_used``。

输出：`测试/留痕/t11_english_degrade_probe.json`
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

from common import paths  # noqa: E402

# engineer 的条件（逐字取自 _t22_after7.py）
CASES = [
    ("engineer-EN3", "In which field has Wuhan Xingtu Xinke Electronics Co., Ltd. become an important supplier?", None),
    ("engineer-EN4", "What is the registered capital of Wuhan Liyuan Information Technology Co., Ltd.?", ["招股说明书2.pdf"]),
    ("engineer-EN5", "How many shares will Wuhan Liyuan Information Technology Co., Ltd. issue?", ["招股说明书2.pdf"]),
    ("engineer-EN6", "What is the registered address of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
]
Q260 = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"


def _scan_logs(trace_id: str) -> list[dict]:
    """按 trace_id 扫两份日志，返回与之相关的记录（含降级事件）。"""
    hits: list[dict] = []
    for name in ("rag_trace.jsonl", "app.log"):
        path = paths.trace_file(name)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-60000:]:
            if trace_id not in line or not line.strip().startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            hits.append({"log": name, "event": row.get("event"), "func": row.get("func"),
                         "level": row.get("level"), "elapsed_ms": row.get("elapsed_ms"),
                         "ratio": row.get("ratio"), "ratio_before": row.get("ratio_before"),
                         "ratio_after": row.get("ratio_after"), "threshold": row.get("threshold"),
                         "path_used": row.get("path_used")})
    return hits


def main() -> int:
    """跑降级探针 + 题 260 首字成因探针。"""
    from app.core.citation import answer_body  # noqa: PLC0415
    from app.core.language import english_ratio  # noqa: PLC0415
    from app.core.qa_engine import build_engine  # noqa: PLC0415

    engine = build_engine(warmup=True)
    rows: list[dict] = []

    for tag, question, file_names in CASES:
        answer = engine.ask(question, session_id=f"t11-degrade-{tag}",
                            file_names=file_names, stream=False)
        trace_id = str(getattr(answer, "trace_id", "") or "")
        body = answer_body(str(answer.text))
        events = _scan_logs(trace_id) if trace_id else []
        degraded = [e for e in events if "language_degraded" in str(e.get("event", ""))]
        rows.append({
            "kind": "degrade", "tag": tag, "question": question, "file_names": file_names,
            "trace_id": trace_id, "language": str(getattr(answer, "language", "")),
            "is_unknown": bool(getattr(answer, "is_unknown", False)),
            "english_ratio_body": round(float(english_ratio(body)), 4),
            "english_ratio_full_text": round(float(english_ratio(str(answer.text))), 4),
            "degrade_events": degraded,
            "degrade_event_count": len(degraded),
            "events_for_trace": events[:40],
            "answer_body": body[:200],
        })

    # 题 260 首字成因：连续两次提问，各取 trace_id 与事件时间线
    for attempt in (1, 2):
        answer = engine.ask(Q260, session_id=f"t11-q260-{attempt}", stream=False)
        trace_id = str(getattr(answer, "trace_id", "") or "")
        events = _scan_logs(trace_id) if trace_id else []
        rows.append({
            "kind": "q260", "attempt": attempt, "trace_id": trace_id,
            "first_token_ms": round(float(getattr(answer, "first_token_ms", 0.0) or 0.0), 2),
            "total_ms": round(float(getattr(answer, "total_ms", 0.0) or 0.0), 2),
            "llm_requests": [e for e in events if str(e.get("event", "")).startswith("llm.")],
            "generation_events": [e for e in events if str(e.get("event", "")).startswith("generation.")],
            "answer_body": answer_body(str(answer.text))[:120],
        })

    out = paths.ensure_trace_dir() / "t11_english_degrade_probe.json"
    out.write_text(json.dumps({
        "work_order": "人工智能NLP-RAG-PDF文档的表格解析及检索优化",
        "ragas": "RAGAS 未运行（依赖不可用，本机断网）",
        "note": "t11 补充实测：英文降级事件触发条件 + 题 260 首字成因；english_ratio 分别按正文/整段计算",
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"留痕：{out}")
    for row in rows:
        if row["kind"] == "degrade":
            print(f"  [{row['tag']}] trace={row['trace_id']} lang={row['language']} "
                  f"unknown={row['is_unknown']} ratio(body)={row['english_ratio_body']} "
                  f"ratio(full)={row['english_ratio_full_text']} degrade_ev={row['degrade_event_count']}")
            for e in row["degrade_events"][:2]:
                print(f"        DEGRADE: ratio_after={e.get('ratio_after')} threshold={e.get('threshold')} path={e.get('path_used')}")
            print(f"        A: {row['answer_body'][:80]!r}")
        else:
            print(f"  [题260 #{row['attempt']}] trace={row['trace_id']} first={row['first_token_ms']}ms "
                  f"total={row['total_ms']}ms llm_events={len(row['llm_requests'])}")
            for e in row["llm_requests"][:6]:
                print(f"        {e.get('event')} func={e.get('func')} elapsed={e.get('elapsed_ms')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
