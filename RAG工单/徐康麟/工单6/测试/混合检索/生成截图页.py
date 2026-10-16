# -*- coding: utf-8 -*-
"""工单6 测试截图页生成 —— 把评测 JSON 渲染成自包含 HTML，供截图留痕。

工单编号：人工智能NLP-RAG-混合检索任务

为什么不直接截终端：评测产物是 JSON/CSV，终端截图在长表格上会换行错位、
且中文字体在不同控制台下宽窄不一；渲染成固定宽度的 HTML 再截图，**数字与中文都可读**、
且页面本身就是可复现的产物（任何人重跑评测后再跑本脚本，能得到同样的页面）。

用法
----
.. code-block:: powershell

    pwsh -NoProfile -File run_py.ps1 测试/混合检索/生成截图页.py --run-tag t6-full
    # 输出：测试/测试数据/混合检索评测/<tag>/截图页/*.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

_CSS = """
:root { --fg:#1a1d24; --muted:#5b6472; --line:#dfe3ec; --bg:#fff; --accent:#1f6feb; }
* { box-sizing:border-box; }
body { font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif; margin:0; padding:22px 26px;
  background:#f4f6fb; color:var(--fg); font-size:14px; line-height:1.6; }
.card { background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:18px 20px;
  max-width:1500px; margin:0 auto 16px; box-shadow:0 1px 3px rgba(20,30,60,.06); }
h1 { font-size:20px; margin:0 0 4px; }
h2 { font-size:16px; margin:18px 0 8px; padding-bottom:5px; border-bottom:2px solid var(--accent); }
.sub { color:var(--muted); font-size:12.5px; margin-bottom:12px; }
table { width:100%; border-collapse:collapse; font-size:12.5px; margin:8px 0 4px; }
th, td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { background:#eef2fa; font-weight:600; white-space:nowrap; }
tbody tr:nth-child(even) { background:#fafbfe; }
.ok { color:#0a7d34; font-weight:600; }
.bad { color:#c0392b; font-weight:600; }
.warn { color:#b7791f; font-weight:600; }
.mono { font-family:Consolas,"Courier New",monospace; font-size:12px; }
.note { background:#f7f9fd; border-left:3px solid var(--accent); padding:9px 12px; margin:10px 0;
  font-size:12.5px; color:var(--muted); border-radius:0 6px 6px 0; }
.kpi { display:flex; gap:12px; flex-wrap:wrap; margin:10px 0; }
.kpi div { background:#f7f9fd; border:1px solid var(--line); border-radius:8px; padding:9px 14px; min-width:150px; }
.kpi b { display:block; font-size:19px; color:var(--accent); }
.kpi span { font-size:11.5px; color:var(--muted); }
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _page(title: str, body: str, subtitle: str = "") -> str:
    return (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<div class=\"card\"><h1>{_esc(title)}</h1>"
        f"<div class=\"sub\">任务编号：<b>{WORK_ORDER}</b>　｜　{subtitle}</div>"
        f"{body}</div></body></html>"
    )


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _yn(flag: Any) -> str:
    return f'<span class="{"ok" if flag else "bad"}">{"✓" if flag else "✗"}</span>'


def build_accuracy_page(summary: dict[str, Any]) -> str:
    ps = summary.get("per_strategy") or {}
    reach = summary.get("reachability") or {}
    ov = summary.get("overall") or {}
    kpi = (
        '<div class="kpi">'
        f'<div><b>{_esc(ov.get("n_runs"))}</b><span>总运行次数</span></div>'
        f'<div><b>{_esc(ov.get("accuracy"))}%</b><span>总准确率</span></div>'
        f'<div><b>{_esc(ov.get("recall"))}%</b><span>总召回率（权威层）</span></div>'
        f'<div><b>{_esc((ov.get("wall_ms") or {}).get("p95"))} ms</b><span>端到端 P95</span></div>'
        f'<div><b>{_esc((ov.get("wall_ms") or {}).get("over_3000"))}</b><span>超 3000 ms 轮次</span></div>'
        "</div>"
    )
    rows = []
    for name, cell in ps.items():
        rows.append([
            _esc(cell.get("label") or name), _esc(cell.get("n")),
            f'<span class="{"ok" if cell.get("accuracy", 0) >= 90 else "bad"}">{_esc(cell.get("accuracy"))}%</span>'
            f' ({_esc(cell.get("correct"))}/{_esc(cell.get("n"))})',
            f'<span class="{"ok" if cell.get("recall", 0) >= 95 else "warn"}">{_esc(cell.get("recall"))}%</span>'
            f' ({_esc(cell.get("hit_complete"))}/{_esc(cell.get("n"))})',
            f'<span class="{"ok" if (cell.get("recall_on_reachable") or 0) >= 95 else "warn"}">'
            f'{_esc(cell.get("recall_on_reachable"))}%</span>'
            f' ({_esc(cell.get("hit_on_reachable"))}/{_esc(cell.get("n_reachable"))})',
            f'{_esc(cell.get("recall_any_evidence"))}%',
            _esc(cell.get("refused")),
            f'{_esc((cell.get("retrieval_ms") or {}).get("p50"))} / '
            f'{_esc((cell.get("retrieval_ms") or {}).get("p95"))} / '
            f'{_esc((cell.get("retrieval_ms") or {}).get("max"))}',
        ])
    body = kpi + _table(
        ["检索策略", "题数", "准确率（目标 ≥90%）", "召回率·权威层（目标 ≥95%）",
         "召回率·仅可达题", "召回率·诊断层A", "拒答", "检索耗时 P50/P95/max (ms)"], rows)
    gains = summary.get("hybrid_gains") or {}
    if gains:
        grows = []
        for key, gain in gains.items():
            grows.append([
                _esc(key.replace("hybrid_vs_", "混合 vs ")),
                f'<span class="{"ok" if gain.get("accuracy_delta_pp", 0) > 0 else ""}">'
                f'{gain.get("accuracy_delta_pp"):+.2f} pp</span>',
                f'{gain.get("recall_delta_pp"):+.2f} pp',
                _esc(gain.get("fixed_ids") or "无"), _esc(gain.get("broken_ids") or "无"),
            ])
        body += "<h2>混合检索相比单一检索的提升</h2>" + _table(
            ["对比", "准确率变化", "召回率变化", "混合修正的题号", "混合退化的题号"], grows)
    if reach:
        body += (
            "<h2>证据可达性（穷举扫描 2676 块，不检索、不调模型）</h2>"
            '<div class="note">'
            f'题级可达 <b>{_esc(reach.get("questions_reachable"))}/{_esc(reach.get("questions"))}'
            f'（{_esc(reach.get("reachability_rate"))}%）</b>；'
            f'不可达题号 <b>{_esc(reach.get("unreachable_question_ids"))}</b>；'
            f'证据条可达 {_esc(reach.get("evidence_items_reachable"))}/{_esc(reach.get("evidence_items"))}。<br>'
            f'判定谓词：{_esc(reach.get("predicate"))}<br>'
            '不可达的题，<b>任何检索策略、任何 Top-K</b> 都不可能命中权威层 '
            '&mdash;&mdash; 这是分块结构决定的判定口径上限，不是检索缺陷。'
            "</div>")
    return _page("准确率 / 召回率汇总", body, "判据：common/golden.py::judge_answer（确定性关键字+正则，非 LLM 自评）")


def build_timing_page(summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    ps = summary.get("per_strategy") or {}
    trows = []
    for name, cell in ps.items():
        r, w = cell.get("retrieval_ms") or {}, cell.get("wall_ms") or {}
        trows.append([
            _esc(cell.get("label") or name), _esc(r.get("n")),
            _esc(r.get("p50")), _esc(r.get("p95")), _esc(r.get("max")),
            f'<span class="{"bad" if (r.get("over_3000") or 0) else "ok"}">{_esc(r.get("over_3000"))}</span>',
            _esc(w.get("p50")), _esc(w.get("p95")), _esc(w.get("max")),
            f'<span class="{"bad" if (w.get("over_3000") or 0) else "ok"}">{_esc(w.get("over_3000"))}</span>',
        ])
    body = (
        '<div class="note">工单指标：<b>响应时间 ≤ 3000 ms</b>。本表把「检索层」与「端到端（含 LLM 生成）」'
        '分开统计；全部数据由 <span class="mono">use_cache=False</span> 强制真实执行取得'
        '（工单6 修正：内层检索/生成缓存原本不受外层开关管辖，会量到缓存命中的约 1 ms）。</div>'
        + _table(["检索策略", "n", "检索 P50", "检索 P95", "检索 max", "检索超线",
                  "端到端 P50", "端到端 P95", "端到端 max", "端到端超线"], trows))
    drow = []
    for r in rows:
        if not r.get("ok"):
            continue
        drow.append([
            _esc(r.get("id")), _esc(r.get("strategy")),
            f'<span class="mono">{_esc(r.get("retrieval_ms"))}</span>',
            f'<span class="mono">{_esc(r.get("wall_ms"))}</span>',
            f'<span class="mono">{_esc((r.get("latency_ms") or {}).get("generate"))}</span>',
            _yn(r.get("correct")), _yn(r.get("hit_complete")),
        ])
    body += "<h2>逐题逐策略耗时明细（毫秒）</h2>" + _table(
        ["题号", "策略", "检索层", "端到端", "其中生成", "判对", "召回"], drow)
    return _page("响应时间测试", body, "目标：端到端 ≤3000 ms；数据源 summary.json / results.jsonl")


def build_abnormal_page(abnormal: dict[str, Any]) -> str:
    body = (
        '<div class="kpi">'
        f'<div><b>{_esc(abnormal.get("pass_rate"))}%</b><span>通过率</span></div>'
        f'<div><b>{_esc(abnormal.get("passed"))}/{_esc(abnormal.get("total"))}</b><span>通过/总数</span></div>'
        "</div>"
        f'<div class="note">{_esc(abnormal.get("rule"))}</div>'
        + _table(["策略", "用例", "输入长度", "通过", "是否拒答", "错误", "回应摘要"],
                 [[_esc(c.get("strategy")), _esc(c.get("tag")), _esc(c.get("input_len")),
                   _yn(c.get("ok")), _esc(c.get("refused")),
                   f'<span class="bad">{_esc(c.get("error"))}</span>' if c.get("error") else "—",
                   f'<span class="mono">{_esc(c.get("answer_preview"))}</span>']
                  for c in (abnormal.get("cases") or [])]))
    return _page("异常输入测试", body, "通过判据：不崩溃 + 有非空回应（不要求答对，拒答也算通过）")


def build_multiturn_page(multiturn: dict[str, Any]) -> str:
    body = ""
    for sname, cell in (multiturn.get("by_strategy") or {}).items():
        lat = cell.get("latency") or {}
        body += (f'<h2>{_esc(sname)}　答案正确 '
                 f'<span class="ok">{_esc(cell.get("answer_ok"))}/{_esc(cell.get("total"))}</span>　'
                 f'改写策略正确 <span class="ok">{_esc(cell.get("rewrite_ok"))}/{_esc(cell.get("total"))}</span>　'
                 f'P95 {_esc(lat.get("p95"))} ms</h2>')
        body += _table(
            ["轮", "用例", "改写策略", "期望策略", "改写正确", "改写成句", "答案正确", "引用页", "耗时 ms"],
            [[_esc(t.get("turn")), _esc(t.get("tag")),
              f'<span class="mono">{_esc(t.get("rewrite_strategy"))}</span>',
              _esc(t.get("expect_strategy") or "—"), _yn(t.get("rewrite_ok")),
              f'<span class="mono">{_esc(t.get("rewritten_query"))}</span>',
              _yn(t.get("answer_ok")), _esc(t.get("cited_pages")), _esc(t.get("wall_ms"))]
             for t in (cell.get("turns") or [])])
    return _page("多轮对话回归（工单5 五轮序列）", body,
                 "检查：指代消解/省略继承是否仍生效，且三种检索策略下答案一致正确")


def build_detail_page(rows: Sequence[dict[str, Any]]) -> str:
    body = _table(
        ["题号", "策略", "判对", "召回·权威层", "召回·诊断A", "拒答", "检索 ms", "端到端 ms",
         "Top-1 证据", "策略摘要", "答案"],
        [[_esc(r.get("id")), _esc(r.get("strategy")),
          _yn(r.get("correct")) if r.get("ok") else '<span class="bad">异常</span>',
          _yn(r.get("hit_complete")) if r.get("ok") else "—",
          _yn(r.get("hit_any_evidence")) if r.get("ok") else "—",
          _esc(r.get("refused")), _esc(r.get("retrieval_ms")), _esc(r.get("wall_ms")),
          f'<span class="mono">{_esc(((r.get("top_chunks") or [{}])[0]).get("chunk_id"))}</span>',
          f'<span class="mono">{_esc(r.get("strategy_label"))}</span>',
          f'<span class="mono">{_esc(str(r.get("answer") or r.get("error") or "")[:160])}</span>']
         for r in rows])
    return _page("逐题逐策略检索明细", body, "数据源 results.jsonl（30 行 = 10 题 × 3 策略）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"[{WORK_ORDER}] 测试截图页生成")
    parser.add_argument("--run-tag", default="t6-full")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else (
        _REPO / "测试" / "测试数据" / "混合检索评测" / args.run_tag)
    if not run_dir.is_dir():
        print(f"[FATAL] 评测目录不存在：{run_dir}")
        return 2
    out_dir = run_dir / "截图页"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(name: str, default: Any) -> Any:
        path = run_dir / name
        if not path.is_file():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    summary = _load("summary.json", {})
    abnormal = _load("abnormal.json", {})
    multiturn = _load("multiturn.json", {})
    rows = []
    rp = run_dir / "results.jsonl"
    if rp.is_file():
        for line in rp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    pages = {
        "准确率汇总.html": build_accuracy_page(summary),
        "响应时间测试.html": build_timing_page(summary, rows),
        "异常输入测试.html": build_abnormal_page(abnormal),
        "多轮对话回归.html": build_multiturn_page(multiturn),
        "逐题逐策略明细.html": build_detail_page(rows),
    }
    for name, content in pages.items():
        (out_dir / name).write_text(content, encoding="utf-8")
        print(f"  [写出] {out_dir / name}")
    print(f"[完成] {len(pages)} 个截图页 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
