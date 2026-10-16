# -*- coding: utf-8 -*-
"""T7 优化前后对比（verify 命令）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

用法：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/baseline_compare.py --before 优化/基线/ --after 优化/评估结果/

做什么：
    1. 读 ``--before`` / ``--after`` 目录下的 ``accuracy_report.json``（**同一评测协议**：
       ``测试/在线/run_16q_eval.py``，16 题 + 负例 + 中英同题）；
    2. 打印逐项对比表（准确率 / 权威层命中（总·BM25·向量）/ 延迟 P50·P95·最大（wall 与 server）/
       引用可回溯 / 拒答通过数 / 英文路径通过数），并给出**增益**；
    3. 落盘 ``优化/评估结果/优化前后对比.json`` 与 ``优化/评估结果/优化前后对比.md``；
    4. 退出码：``0`` = 全部达标；``1`` = 有指标未达标（**不隐藏退化**）。

达标线（来自 T7 任务书 acceptance）：
    * 准确率优化后 ≥ 优化前，且 = 16/16；
    * 延迟 P95（wall）≤ 3000 ms；
    * 负例（拒答）4/4（靶点 A 修复）；
    * 英文路径 3/3（靶点 B 修复）；
    * 引用可回溯 ≥ 15/16；
    * 权威层命中：向量通道**不低于**优化前（靶点 C：不得变差），且总数不低于优化前。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: 达标线
PASS_ACCURACY = 16
PASS_P95_WALL_MS = 3000.0
PASS_NEGATIVES = 4
PASS_BILINGUAL_EN = 3
PASS_CITATION = 15


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def resolve_report(directory: Path) -> Path:
    """定位评测报告：先看目录本身，再看常见子目录（run1_full / 优化后 / after）。"""
    candidates = [directory / "accuracy_report.json",
                  directory / "run1_full" / "accuracy_report.json",
                  directory / "优化后" / "accuracy_report.json",
                  directory / "after" / "accuracy_report.json"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"在 {directory} 下找不到 accuracy_report.json（候选：{[str(c) for c in candidates]}）；"
        f"请先跑 测试/在线/run_16q_eval.py --out-dir <目录>")


def load_report(directory: Path) -> dict:
    """读一个评测输出目录的 accuracy_report.json。"""
    path = resolve_report(directory)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(report: dict) -> dict:
    """把评测报告压成对比所需的一行指标（缺字段一律 None，**不补数**）。"""
    summary = report.get("summary") or {}
    negatives = report.get("negatives") or {}
    bilingual = report.get("bilingual") or {}
    neg_items = negatives.get("items") or []
    bi_items = bilingual.get("items") or []
    en_rows = [r for r in bi_items if r.get("language") == "en"]
    zh_rows = [r for r in bi_items if r.get("language") == "zh"]
    en_pass = sum(1 for r in en_rows if r.get("en_criterion_passed") is True)
    zh_pass = sum(1 for r in zh_rows if r.get("correct") is True)
    return {
        "n_questions": summary.get("n_questions"),
        "accuracy": summary.get("accuracy"),
        "n_correct": summary.get("n_correct"),
        "hit_complete": summary.get("n_hit_complete"),
        "hit_bm25": summary.get("n_hit_complete_bm25"),
        "hit_vector": summary.get("n_hit_complete_vector"),
        "citation_ok": summary.get("n_citation_ok"),
        "latency_wall_p50_ms": summary.get("latency_wall_p50_ms"),
        "latency_wall_p95_ms": summary.get("latency_wall_p95_ms"),
        "latency_wall_max_ms": summary.get("latency_wall_max_ms"),
        "latency_server_p50_ms": summary.get("latency_server_p50_ms"),
        "latency_server_p95_ms": summary.get("latency_server_p95_ms"),
        "latency_server_max_ms": summary.get("latency_server_max_ms"),
        "n_over_3000ms": summary.get("n_latency_over_3000ms"),
        "first_token_min_ms": summary.get("first_token_min_ms"),
        "first_token_max_ms": summary.get("first_token_max_ms"),
        "negatives_passed": negatives.get("n_passed"),
        "negatives_total": negatives.get("n"),
        "negatives_failed_ids": [r.get("id") for r in neg_items if r.get("passed") is not True],
        "bilingual_zh_passed": zh_pass,
        "bilingual_en_passed": en_pass,
        "bilingual_en_total": len(en_rows),
        "run_tag": (report.get("run_meta") or {}).get("run_tag"),
        "use_cache": (report.get("run_meta") or {}).get("use_cache"),
        "index_chunk_total": (report.get("run_meta") or {}).get("index_chunk_total"),
    }


def _delta(before: float | None, after: float | None, *, lower_is_better: bool = False) -> str:
    if before is None or after is None:
        return "-"
    diff = after - before
    if lower_is_better:
        diff = -diff
    return f"{diff:+.1f}" if abs(diff) > 0.05 else "0"


ROWS: tuple[tuple[str, str, bool], ...] = (
    ("准确率（判对题数）", "n_correct", False),
    ("准确率", "accuracy", False),
    ("权威层命中（总数）", "hit_complete", False),
    ("权威层命中（BM25 通道）", "hit_bm25", False),
    ("权威层命中（向量通道）", "hit_vector", False),
    ("引用可回溯", "citation_ok", False),
    ("拒答负例通过（/4）", "negatives_passed", False),
    ("英文路径通过（/3）", "bilingual_en_passed", False),
    ("中文路径通过（/3）", "bilingual_zh_passed", False),
    ("延迟 wall P50 (ms)", "latency_wall_p50_ms", True),
    ("延迟 wall P95 (ms)", "latency_wall_p95_ms", True),
    ("延迟 wall 最大 (ms)", "latency_wall_max_ms", True),
    ("延迟 server P50 (ms)", "latency_server_p50_ms", True),
    ("延迟 server P95 (ms)", "latency_server_p95_ms", True),
    ("延迟 server 最大 (ms)", "latency_server_max_ms", True),
    ("超 3000 ms 题数", "n_over_3000ms", True),
    ("首字最小 (ms)", "first_token_min_ms", True),
    ("首字最大 (ms)", "first_token_max_ms", True),
)


def compare(before: dict, after: dict, *, t6_baseline: dict | None = None) -> tuple[dict, list[str]]:
    """生成对比结果与未达标清单。"""
    checks: list[str] = []
    ok = True
    if (after.get("n_correct") or 0) < PASS_ACCURACY:
        ok = False
        checks.append(f"准确率未达 {PASS_ACCURACY}/16（实际 {after.get('n_correct')}）")
    if (before.get("accuracy") is not None and after.get("accuracy") is not None
            and after["accuracy"] < before["accuracy"] - 1e-9):
        ok = False
        checks.append(f"准确率较优化前退化：{_pct(before['accuracy'])} → {_pct(after['accuracy'])}")
    if after.get("latency_wall_p95_ms") is None or after["latency_wall_p95_ms"] > PASS_P95_WALL_MS:
        ok = False
        checks.append(f"P95 逾线：{after.get('latency_wall_p95_ms')} ms > {PASS_P95_WALL_MS:.0f} ms")
    if (after.get("n_over_3000ms") or 0) > 0:
        ok = False
        checks.append(f"有 {after['n_over_3000ms']} 题超过 3000 ms")
    if (after.get("negatives_passed") or 0) < PASS_NEGATIVES:
        ok = False
        checks.append(f"拒答负例未全通过：{after.get('negatives_passed')}/4（失败 {after.get('negatives_failed_ids')}）")
    if (after.get("bilingual_en_passed") or 0) < PASS_BILINGUAL_EN:
        ok = False
        checks.append(f"英文路径未全通过：{after.get('bilingual_en_passed')}/{PASS_BILINGUAL_EN}")
    if (after.get("citation_ok") or 0) < PASS_CITATION:
        ok = False
        checks.append(f"引用可回溯不足：{after.get('citation_ok')}/16")
    if before.get("hit_vector") is not None and after.get("hit_vector") is not None \
            and after["hit_vector"] < before["hit_vector"]:
        ok = False
        checks.append(f"向量通道权威层命中退化：{before['hit_vector']} → {after['hit_vector']}")
    if before.get("hit_complete") is not None and after.get("hit_complete") is not None \
            and after["hit_complete"] < before["hit_complete"]:
        ok = False
        checks.append(f"权威层命中总数退化：{before['hit_complete']} → {after['hit_complete']}")

    payload = {
        "work_order": "人工智能NLP-RAG-图像内容解析及检索优化",
        "baseline_source": "T6 实测（测试/测试报告.md §5/§7.1/§8.1）与本脚本复现的优化前跑（同一评测协议）",
        "protocol": "测试/在线/run_16q_eval.py（16 题 + 负例 + 中英同题；--no-cache 全量）",
        "before": before,
        "after": after,
        "t6_baseline": t6_baseline,
        "rows": [{"metric": label, "key": key, "before": before.get(key), "after": after.get(key),
                  "gain": _delta(before.get(key), after.get(key), lower_is_better=lower)}
                 for label, key, lower in ROWS],
        "all_passed": ok,
        "unmet": checks,
    }
    return payload, checks


def render_markdown(payload: dict) -> str:
    lines = ["| 指标 | 优化前（T6 基线） | 优化前（本脚本复现） | 优化后（run1） | 优化后（run2） | 增益（复现→run1） |",
             "| --- | --- | --- | --- | --- | --- |"]
    t6 = payload.get("t6_baseline") or {}
    before, after = payload["before"], payload["after"]
    after2 = payload.get("after_run2") or {}
    for row in payload["rows"]:
        key = row["key"]
        gain = row["gain"]
        lines.append(f"| {row['metric']} | {t6.get(key, '-' if key not in t6 else t6.get(key))} | "
                     f"{before.get(key)} | {after.get(key)} | {after2.get(key, '-')} | {gain} |")
    lines.append("")
    lines.append(f"**达标结论**：{'全部达标' if payload['all_passed'] else '未达标：' + '；'.join(payload['unmet'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T7 优化前后对比（同一评测协议）")
    parser.add_argument("--before", default=str(ROOT / "优化" / "基线"), help="优化前评测目录")
    parser.add_argument("--after", default=str(ROOT / "优化" / "评估结果"), help="优化后评测目录")
    parser.add_argument("--after-run2", default=None, help="优化后第二次跑（英文路径稳定性对照）")
    parser.add_argument("--out", default=None, help="对比产物输出目录（默认=--after）")
    parser.add_argument("--markdown", action="store_true", help="同时打印 Markdown 表格")
    args = parser.parse_args(argv)

    before_dir, after_dir = Path(args.before), Path(args.after)
    out_dir = Path(args.out) if args.out else after_dir
    if not before_dir.is_dir() or not after_dir.is_dir():
        print(f"[错误] 目录不存在：{before_dir} / {after_dir}", file=sys.stderr)
        return 1
    before = summarize(load_report(before_dir))
    after = summarize(load_report(after_dir))
    after2_dir = Path(args.after_run2) if args.after_run2 else after_dir / "run2_full"
    after2 = summarize(load_report(after2_dir)) if (after2_dir / "accuracy_report.json").is_file() else {}
    t6_path = after_dir / "T6_评测基线.json"
    t6 = {}
    if t6_path.is_file():
        raw = json.loads(t6_path.read_text(encoding="utf-8"))
        t6 = summarize({"summary": raw.get("summary") or {},
                        "negatives": {"n": (raw.get("negative") or {}).get("n"),
                                      "n_passed": (raw.get("negative") or {}).get("n_passed"),
                                      "items": []},
                        "bilingual": {"items": [{"language": "en",
                                                 "en_criterion_passed": i < (raw.get("bilingual") or {}).get("en_atom_passed", 0)}
                                                for i in range((raw.get("bilingual") or {}).get("en_total", 0))]}})
        t6["bilingual_en_passed"] = (raw.get("bilingual") or {}).get("en_atom_passed")
        t6["negatives_passed"] = (raw.get("negative") or {}).get("n_passed")

    payload, checks = compare(before, after, t6_baseline=t6)
    payload["after_run2"] = after2
    payload["stability"] = {
        "run1": {k: after.get(k) for k in ("n_correct", "bilingual_en_passed", "negatives_passed",
                                           "hit_vector", "citation_ok")},
        "run2": {k: after2.get(k) for k in ("n_correct", "bilingual_en_passed", "negatives_passed",
                                            "hit_vector", "citation_ok")} if after2 else None,
        "consistent": bool(after2) and all(
            after.get(k) == after2.get(k) for k in ("n_correct", "bilingual_en_passed",
                                                    "negatives_passed", "hit_vector")),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "优化前后对比.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    markdown = render_markdown(payload)
    (out_dir / "优化前后对比.md").write_text(markdown + "\n", encoding="utf-8")

    print(f"[T7 优化前后对比] 优化前 = {before_dir}（复现）/ T6 基线 / 优化后 = {after_dir}")
    print(f"{'指标':<26}{'优化前(T6)':>12}{'优化前(复现)':>14}{'优化后(run1)':>14}{'优化后(run2)':>14}{'增益':>10}")
    for row in payload["rows"]:
        key = row["key"]
        print(f"{row['metric']:<26}{str(t6.get(key, '-')):>12}{str(before.get(key)):>14}"
              f"{str(after.get(key)):>14}{str(after2.get(key, '-')):>14}{row['gain']:>10}")
    print(f"\n英文路径连续两次一致：{payload['stability']['consistent']}（run1={payload['stability']['run1']}）")
    print(f"达标：{payload['all_passed']}" + ("" if payload["all_passed"] else "；未达标：" + "；".join(checks)))
    print(f"落盘 → {out_dir / '优化前后对比.json'} / 优化前后对比.md")
    if args.markdown:
        print("\n" + markdown)
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
