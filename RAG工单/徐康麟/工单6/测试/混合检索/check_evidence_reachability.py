# -*- coding: utf-8 -*-
"""工单6 证据可达性核查 —— 证明「召回率缺的那 1 题」是**结构不可达**，不是检索没做好。

工单编号：人工智能NLP-RAG-混合检索任务

为什么需要这个脚本
------------------
评测报告里「召回率（权威层）90%」看起来像没达标。要判断这是**检索系统的缺陷**还是
**判定口径的结构性限制**，必须回答一个可证伪的问题：

    golden 要求的「完整证据原文」，在这 2676 个已索引块里**到底存不存在**？

本脚本做**穷举扫描**（不做任何检索、不调模型）：
    对每道题的每条 ``retrieval_evidence``，用与 ``common.golden.check_retrieval_hit_layers``
    **逐字相同**的判定谓词，逐块检查该证据能否在**某一单块**内被完整满足。
    * 找得到 ⇒ 该证据「可达」，检索没进 Top-5 才是检索的问题；
    * 找不到 ⇒ 该证据「不可达」，**任何**检索策略、任何 Top-K 都不可能命中权威层。

判定谓词（与 golden 严格同口径，勿自行简化）
    * ``kind=verbatim``：``normalize(text)`` 是某一块的归一化正文的**连续子串**；
    * ``kind=chart_pairs|pairs|labels``：``pairs`` 的**每个 pair 的每个分量**都出现在**同一块**里
      ——注意是「分量各自出现」，**不是**把 pair 拼成一整串去匹配。这一条曾把本脚本写出过假阳性
      （把 13 字的图例串当整串去匹配 ⇒ 误判题 6 不可达）。

结论用法：报告里必须**同时**给出「权威层召回率」与「可达题上的召回率」，
后者才是检索系统真正该背的指标。

用法
----
.. code-block:: powershell

    pwsh -NoProfile -File run_py.ps1 测试/混合检索/check_evidence_reachability.py
    # 输出：测试/测试数据/混合检索评测/证据可达性.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _p in (_REPO / "研发", _REPO / "测试"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.golden import load_golden_index, normalize  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"
#: 工单6 的十道题（与 run_strategy_eval.py 保持一致）
T6_QUESTION_IDS: tuple[int, ...] = (5, 6, 1, 2, 3, 4, 260, 95, 33, 34)
CHUNKS_FILE = "chunks_indexed.jsonl"
DEFAULT_INDEX_DIR = _REPO / "研发" / "data" / "index" / "bge-m3_1024"
#: 配对类证据的 kind（与 golden._layer_hit 一致）
PAIR_KINDS = ("chart_pairs", "pairs", "labels")


def load_chunks(index_dir: Path) -> list[dict[str, Any]]:
    """读块表（行序 == 向量行序；本脚本只读不改）。"""
    path = index_dir / CHUNKS_FILE
    if not path.is_file():
        raise SystemExit(f"[FATAL] 块表不存在：{path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def describe_evidence(ev: dict[str, Any]) -> dict[str, Any]:
    """把一条证据归一成「可判定的形式」。

    * ``mode="substring"``：``needle`` 需连续命中；
    * ``mode="pairs"``：``pairs`` 内每个分量各自命中（同块）。
    """
    kind = str(ev.get("kind") or "verbatim")
    if kind in PAIR_KINDS:
        raw_pairs = ev.get("pairs") or [[lbl] for lbl in ev.get("labels", [])]
        pairs = [[normalize(str(p)) for p in pair] for pair in raw_pairs]
        return {"kind": kind, "page": ev.get("page"), "mode": "pairs", "pairs": pairs,
                "text_head": " / ".join("+".join(p) for p in pairs)[:70],
                "text_len": sum(len(p) for pair in pairs for p in pair)}
    needle = normalize(str(ev.get("text") or ""))
    return {"kind": kind, "page": ev.get("page"), "mode": "substring", "needle": needle,
            "text_head": needle[:70], "text_len": len(needle)}


def matches(spec: dict[str, Any], chunk_text: str) -> bool:
    """与 ``golden._layer_hit`` 权威层逐字同口径的判定谓词。"""
    if spec["mode"] == "pairs":
        pairs = spec["pairs"]
        if not pairs:
            return False
        return all(all(part in chunk_text for part in pair) for pair in pairs)
    needle = spec["needle"]
    return bool(needle) and needle in chunk_text


def check(index_dir: Path, ids: tuple[int, ...], *, verbose: bool = True) -> dict[str, Any]:
    golden = load_golden_index()
    chunks = load_chunks(index_dir)
    normalized = [(str(r.get("chunk_id")), normalize(str(r.get("content") or ""))) for r in chunks]
    report: dict[str, Any] = {
        "work_order": WORK_ORDER,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "index_dir": str(index_dir), "chunk_count": len(chunks),
        "question_ids": list(ids), "per_question": {}, "summary": {},
    }
    reachable_evidence = 0
    evidence_total = 0
    unreachable_questions: list[int] = []
    for qid in ids:
        record = golden.get(qid)
        if record is None:
            report["per_question"][str(qid)] = {"error": "golden 缺该 id"}
            continue
        items: list[dict[str, Any]] = []
        question_reachable = False
        for ev in record.get("retrieval_evidence") or []:
            spec = describe_evidence(ev)
            evidence_total += 1
            hit_chunk = next((cid for cid, text in normalized if matches(spec, text)), None)
            reachable = hit_chunk is not None
            question_reachable = question_reachable or reachable
            reachable_evidence += 1 if reachable else 0
            items.append({"kind": spec["kind"], "page": spec["page"], "mode": spec["mode"],
                          "text_len": spec["text_len"], "reachable": reachable,
                          "chunk_id": hit_chunk, "text_head": spec["text_head"]})
        if not question_reachable:
            unreachable_questions.append(qid)
        report["per_question"][str(qid)] = {
            "question": record.get("question"),
            "evidence_pages": record.get("evidence_pages"),
            "reachable": question_reachable,
            "evidence": items,
        }
        if verbose:
            flag = "可达  " if question_reachable else "不可达"
            n_ok = sum(1 for i in items if i["reachable"])
            print(f"  题 {qid:>4}  [{flag}]  证据 {n_ok}/{len(items)} 条可达  页={record.get('evidence_pages')}")
            for item in items:
                if not item["reachable"]:
                    print(f"          · 不可达 kind={item['kind']} page={item['page']} "
                          f"len={item['text_len']}：{item['text_head']}…")
    reachable_questions = len(ids) - len(unreachable_questions)
    report["summary"] = {
        "questions": len(ids),
        "questions_reachable": reachable_questions,
        "questions_unreachable": len(unreachable_questions),
        "unreachable_question_ids": unreachable_questions,
        "evidence_items": evidence_total,
        "evidence_items_reachable": reachable_evidence,
        "reachability_rate": round(100.0 * reachable_questions / max(len(ids), 1), 2),
        "definition": "题级可达 = 该题**至少一条** retrieval_evidence 能在某个已索引块里被完整满足；"
                      "不可达的题，任何检索策略/任何 Top-K 都不可能命中权威层 hit_complete。",
        "predicate": "与 common.golden._layer_hit 权威层同口径：verbatim 用连续子串；"
                     "chart_pairs/pairs/labels 用「每个 pair 的每个分量同块出现」。",
        "normalize_note": "使用 golden 的 normalize（去空白/全角转半角/统一引号），与命中判定同口径。",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"[{WORK_ORDER}] 证据可达性穷举核查（不检索、不调模型）")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--ids", default=None, help="逗号分隔题号（默认工单十题）")
    parser.add_argument("--out", default=None, help="输出 JSON 路径")
    args = parser.parse_args(argv)
    ids = tuple(int(x) for x in str(args.ids).split(",") if x.strip()) if args.ids else T6_QUESTION_IDS
    index_dir = Path(args.index_dir)
    print(f"[{WORK_ORDER}] 证据可达性核查：index_dir={index_dir}")
    report = check(index_dir, ids)
    out = Path(args.out) if args.out else (
        _REPO / "测试" / "测试数据" / "混合检索评测" / "证据可达性.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = report["summary"]
    print()
    print(f"[汇总] 块数 {report['chunk_count']}；题级可达 {s['questions_reachable']}/{s['questions']}"
          f"（{s['reachability_rate']}%）；不可达题号 {s['unreachable_question_ids']}")
    print(f"       证据条可达 {s['evidence_items_reachable']}/{s['evidence_items']}")
    print(f"[落盘] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
