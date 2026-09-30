"""用**当前判据**重新评一份已存的答案（``eval/results/<tag>.jsonl``）。

为什么需要（真机教训）：判据（`score_item`）本身会被修（2026-09-24 修了"引用保真度"把
"抄自证据的名字"当成编造），而**旧报告是用旧判据算的**。这时不该重跑大模型（贵、且答案会变），
而应**拿存下来的答案重新判分** —— 同一批答案、同一把新尺子，A/B 才可比。

用法::

    python scripts/rescore_answers.py --in eval/results/v5.jsonl --out eval/results/v5.rescored.json
    python scripts/rescore_answers.py --in eval/results/v5.jsonl --in eval/results/v6.jsonl --compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.eval_set import QAItem, corpus_law_names, score_item, summarize  # noqa: E402
from legal_rag.logging_setup import setup_utf8_stdout  # noqa: E402


def item_from_row(row: dict) -> QAItem:
    """从答案记录里还原题集条目（只取判分需要的字段）。"""
    return QAItem.from_dict({
        "id": row.get("id"), "category": row.get("category") or "legal",
        "expect": row.get("expect") or "answerable",
        "question": row.get("question") or "",
        "turns": row.get("turns") or [],
        "sources": row.get("sources") or [], "claims": row.get("claims") or [],
        "notes": row.get("notes") or "",
    })


def rescore_rows(rows: list[dict], known_laws: set[str]) -> tuple[list[dict], dict]:
    """重新判分，返回 ``(逐题判分, 汇总)``。纯函数，便于单测。"""
    scores: list[dict] = []
    for row in rows:
        if row.get("error"):
            scores.append({"id": row.get("id"), "category": row.get("category"),
                           "expect": row.get("expect"), "ok": False,
                           "source_hit": None, "claims_ok": None,
                           "missing_claims": list(row.get("claims") or []),
                           "citations": [], "unknown_citations": [],
                           "citations_from_evidence": [], "citations_not_a_law": [],
                           "citations_known": [], "refused": False, "warned": False,
                           "cites_law": False, "answer_chars": 0,
                           "error": row.get("error")})
            continue
        scores.append(score_item(item_from_row(row), str(row.get("answer") or ""),
                                 row.get("citations_raw") or [], known_laws))
    return scores, summarize(scores)


def main() -> int:
    setup_utf8_stdout()   # 控制台中文/表格按 UTF-8 输出（否则 GBK 终端上会乱码、机器也读不了）
    parser = argparse.ArgumentParser(description="用当前判据重新评已存答案")
    parser.add_argument("--in", dest="inputs", action="append", required=True,
                        help="可多次；每次一个 eval/results/<tag>.jsonl")
    parser.add_argument("--corpus", default="knowledge/lawyer")
    parser.add_argument("--out", default="", help="把逐题判分写到这个 JSON（单份输入时）")
    parser.add_argument("--compare", action="store_true", help="多份输入时打印对照表")
    args = parser.parse_args()

    known = corpus_law_names(args.corpus)
    reports: dict[str, dict] = {}
    for path_text in args.inputs:
        path = Path(path_text)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        scores, summary = rescore_rows(rows, known)
        tag = path.stem
        reports[tag] = {"summary": summary, "scores": scores}
        print(f"== {tag}（{len(rows)} 题）：通过率 {summary['ok_rate']}  "
              f"来源命中 {summary['source_hit_rate']}  要点覆盖 {summary['claim_coverage']}  "
              f"引用保真 {summary['citation_fidelity']}  分流 {summary['route_accuracy']}")
        if summary.get("unknown_citation_items"):
            print(f"   编造嫌疑：{summary['unknown_citation_items']}")
        if summary.get("citations_from_evidence_items"):
            print(f"   抄自证据（不算编造）：{summary['citations_from_evidence_items']}")
        if summary.get("citations_not_a_law_items"):
            print(f"   非法律名（抽取器误报）：{summary['citations_not_a_law_items']}")

    comparison: dict | None = None
    if args.compare and len(reports) > 1:
        keys = ("ok_rate", "source_hit_rate", "claim_coverage", "citation_fidelity",
                "route_accuracy")
        tags = list(reports)
        print("\n=== 对照 ===")
        print(f"{'指标':18s} " + "  ".join(f"{tag:>10s}" for tag in tags))
        for key in keys:
            values = [reports[tag]["summary"].get(key) for tag in tags]
            print(f"{key:18s} " + "  ".join(f"{str(v):>10s}" for v in values))
        comparison = {"tags": tags, "metrics": {},
                      "summaries": {tag: reports[tag]["summary"] for tag in tags}}
        for key in keys:
            comparison["metrics"][key] = [reports[tag]["summary"].get(key) for tag in tags]
        if len(tags) == 2:
            before, after = (reports[tag]["summary"] for tag in tags)
            flip = [score["id"] for score in reports[tags[1]]["scores"]
                    if (next((s["ok"] for s in reports[tags[0]]["scores"]
                              if s["id"] == score["id"]), None) != score["ok"])]
            print(f"\n逐题通过翻转：{flip or '无'}")
            deltas: dict[str, float] = {}
            for key in keys:
                try:
                    delta = float(after.get(key) or 0) - float(before.get(key) or 0)
                    deltas[key] = round(delta, 4)
                    print(f"  Δ{key:18s} {delta:+.4f}")
                except (TypeError, ValueError):
                    pass
            comparison["flipped_items"] = flip
            comparison["deltas"] = deltas

    # ⚠️ 以前 `--out` 只在**单份输入**时生效，`--compare` 时被**静默忽略**：
    #    A/B 驱动写了 `--out ...-compare.json`，跑完却什么都没有，而日志里看不出异常
    #    （2026-09-27 真机发现：只有日志里的表格，没有可归档的证据文件）。
    #    现在对照表也落盘；另外单份输入之外的 `--out` 一定会有明确回话，不再闷掉。
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(reports) == 1:
            tag = next(iter(reports))
            target.write_text(json.dumps(reports[tag], ensure_ascii=False, indent=2),
                              encoding="utf-8")
        elif comparison is not None:
            target.write_text(json.dumps(comparison, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        else:
            print(f"\n[注意] --out 没写：给了 {len(reports)} 份输入但没加 --compare，"
                  f"逐题判分该写哪一份说不清；加 --compare 会把对照表写到 {target}",
                  file=sys.stderr)
            return 0
        print(f"\n已写：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
