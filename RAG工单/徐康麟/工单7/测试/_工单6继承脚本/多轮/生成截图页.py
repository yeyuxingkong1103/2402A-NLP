# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

把多轮评测的**实跑产物**渲染成可直接截图的 HTML 证据页。

用途：`测试/测试截图/` 里的专项截图（响应时间 / 准确率汇总 / 单轮回归 / 异常输入）
需要「真实数据页」而不是手绘表格，本脚本从
`测试/测试数据/多轮评测结果/<run-tag>/` 读取 JSON，生成自包含的 HTML 页
（内联 CSS、无外部依赖、无网络请求），再用浏览器截图留痕。

**数字纪律**：本脚本**只呈现**产物里已有的数字，不做任何计算外的推断，也不预填；
产物缺失时如实显示「未运行」。

用法：
    pwsh -NoProfile -File run_py.ps1 测试/多轮/生成截图页.py --run-tag mt-local
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_TAG = "mt-local"

_CSS = """
* { box-sizing: border-box; }
body { margin:0; padding:18px 22px; background:#f5f6f8;
       font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif; color:#1f2329; }
h1 { font-size:19px; margin:0 0 2px; }
.sub { color:#6b7280; font-size:12px; margin-bottom:14px; }
.card { background:#fff; border:1px solid #e3e6ea; border-radius:10px; padding:14px 16px; margin-bottom:14px; }
.card h2 { font-size:15px; margin:0 0 10px; padding-bottom:7px; border-bottom:2px solid #eef1f4; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th,td { border:1px solid #e3e6ea; padding:5px 8px; text-align:left; vertical-align:top; }
th { background:#f0f3f7; font-weight:600; }
.pass { color:#0a7a35; font-weight:700; }
.fail { color:#c02626; font-weight:700; }
.kv { display:flex; flex-wrap:wrap; gap:8px; }
.kpi { flex:0 0 auto; min-width:150px; background:#f7f9fc; border:1px solid #e3e6ea;
       border-radius:8px; padding:9px 12px; }
.kpi .n { font-size:21px; font-weight:700; color:#14428a; }
.kpi .l { font-size:11.5px; color:#6b7280; margin-top:3px; }
.kpi.ok .n { color:#0a7a35; } .kpi.bad .n { color:#c02626; }
.note { font-size:11.5px; color:#6b7280; margin-top:9px; line-height:1.6; }
.mono { font-family:Consolas,monospace; font-size:12px; }
"""


def _esc(v: Any) -> str:
    return html.escape(str(v if v is not None else "—"))


def _page(title: str, subtitle: str, body: str) -> str:
    return (f"<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
            f"<h1>{_esc(title)}</h1>"
            f"<div class=\"sub\">工单5：{_esc(WORK_ORDER)}　|　{subtitle}</div>"
            f"{body}</body></html>")


def _load(p: Path) -> Any:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 产物损坏时如实显示「未运行」，不伪造
        return None


def _missing(what: str) -> str:
    return (f"<div class=\"card\"><h2>{_esc(what)}</h2>"
            f"<div class=\"note\">产物缺失或不可解析 —— 记为「未运行」，不预填任何数字。</div></div>")


def _kpi(n: Any, label: str, cls: str = "") -> str:
    return f"<div class=\"kpi {cls}\"><div class=\"n\">{_esc(n)}</div><div class=\"l\">{_esc(label)}</div></div>"


# --------------------------------------------------------------------- 页 1
def page_summary(out_dir: Path, summary: dict, rows: list[dict]) -> str:
    s = (summary or {}).get("summary") or {}
    meta = (summary or {}).get("run_meta") or {}
    lat = s.get("latency_wall") or {}
    acc = s.get("accuracy") if isinstance(s.get("accuracy"), dict) else {}
    n_pass = s.get("n_passed")
    n_tot = s.get("n_turns")
    rate = s.get("accuracy_turn") if s.get("accuracy_turn") is not None else acc.get("turn")

    body = ["<div class=\"card\"><h2>总览</h2><div class=\"kv\">"]
    body.append(_kpi(f"{n_pass}/{n_tot}" if n_tot else "未运行", "多轮逐轮判过（主口径）",
                     "ok" if (n_tot and n_pass == n_tot) else ("bad" if n_tot else "")))
    body.append(_kpi(s.get("sessions_passed_str") or
                     (f"{s.get('n_sessions_passed')}/{s.get('n_sessions')}"
                      if s.get("n_sessions") else "—"), "会话整体口径"))
    body.append(_kpi(lat.get("max", "—"), "最大逐轮延迟 (ms)"))
    body.append(_kpi(meta.get("cold_start_ms", "—"), "冷启动 (ms，不计入)"))
    body.append(_kpi(meta.get("use_cache") if meta.get("use_cache") is not None else "—", "use_cache"))
    body.append(_kpi(meta.get("engine", "—"), "评测模式"))
    body.append("</div>")
    if rate is not None:
        body.append(f"<div class=\"note\">逐轮准确率 = {_esc(rate)}（判分："
                    f"<span class=\"mono\">测试/测试数据/多轮判分.py::judge_case</span>，确定性判定，**非 LLM 自评**）</div>")
    body.append("</div>")

    # 逐轮表
    body.append("<div class=\"card\"><h2>多轮序列逐轮结果</h2><table>"
                "<tr><th>id</th><th>轮</th><th>用户问句</th><th>改写后的独立问句</th>"
                "<th>策略</th><th>焦点实体</th><th>答案（摘要）</th><th>判定</th>"
                "<th>wall(ms)</th><th>改写(ms)</th></tr>")
    for r in rows:
        ck = r.get("checks") or {}
        bad = [k for k, v in ck.items() if v is False]
        verdict = "<span class=\"pass\">通过</span>" if r.get("passed") else \
                  f"<span class=\"fail\">未通过</span><div class=\"note\">{_esc('、'.join(bad))}</div>"
        body.append(
            f"<tr><td>{_esc(r.get('id'))}</td><td>{_esc(r.get('turn'))}</td>"
            f"<td>{_esc((r.get('question') or '')[:40])}</td>"
            f"<td>{_esc((r.get('rewritten_query') or '')[:46])}</td>"
            f"<td class=\"mono\">{_esc(r.get('strategy'))}</td>"
            f"<td>{_esc(r.get('focus_entity'))}</td>"
            f"<td>{_esc((r.get('answer') or '').replace(chr(10), ' ')[:64])}</td>"
            f"<td>{verdict}</td><td>{_esc(r.get('latency_wall_ms', r.get('wall_ms')))}</td>"
            f"<td>{_esc(r.get('latency_rewrite_ms', r.get('rewrite_ms')))}</td></tr>")
    body.append("</table></div>")
    return _page("工单5 多轮对话准确率汇总", f"产物目录 {out_dir}", "".join(body))


# --------------------------------------------------------------------- 页 2
def page_latency(out_dir: Path, summary: dict, rows: list[dict]) -> str:
    s = (summary or {}).get("summary") or {}
    meta = (summary or {}).get("run_meta") or {}
    lat = s.get("latency_wall") or {}
    rw = s.get("latency_rewrite") or {}
    body = ["<div class=\"card\"><h2>端到端延迟（客户端 wall，逐轮）</h2><div class=\"kv\">"]
    for label, key in (("P50", "p50"), ("P95", "p95"), ("max", "max"), ("min", "min"), ("mean", "mean")):
        body.append(_kpi(lat.get(key, "—"), f"{label} (ms)"))
    body.append(_kpi(lat.get("n_over_3000ms", "—"), "超 3000ms 轮数",
                     "ok" if lat.get("n_over_3000ms") == 0 else "bad"))
    body.append(_kpi(meta.get("cold_start_ms", "—"), "冷启动 (ms，不计入)"))
    body.append("</div>")
    body.append("<div class=\"card\"><h2>改写阶段（工单5 新增开销）</h2><div class=\"kv\">")
    for label, key in (("P50", "p50"), ("P95", "p95"), ("max", "max")):
        body.append(_kpi(rw.get(key, "—"), f"改写 {label} (ms)"))
    body.append(_kpi("≤3000", "验收阈值 (ms)", "ok"))
    body.append("</div>")
    body.append("<div class=\"note\">口径：客户端 wall = 请求发出 → 完整答案返回；"
                "冷启动轮（模型/索引首轮加载）单独记录且**不计入**统计。"
                "改写阶段是纯规则+状态实现，不调用生成模型。</div></div>")

    body.append("<div class=\"card\"><h2>逐轮延迟</h2><table>"
                "<tr><th>id</th><th>轮</th><th>策略</th><th>wall(ms)</th><th>改写(ms)</th>"
                "<th>检索(ms)</th><th>生成(ms)</th><th>≤3000ms</th></tr>")
    for r in rows:
        wall = r.get("latency_wall_ms", r.get("wall_ms"))
        try:
            ok = float(wall) <= 3000.0
        except (TypeError, ValueError):
            ok = None
        mark = "<span class=\"pass\">是</span>" if ok else ("<span class=\"fail\">否</span>" if ok is False else "—")
        body.append(
            f"<tr><td>{_esc(r.get('id'))}</td><td>{_esc(r.get('turn'))}</td>"
            f"<td class=\"mono\">{_esc(r.get('strategy'))}</td><td>{_esc(wall)}</td>"
            f"<td>{_esc(r.get('latency_rewrite_ms', r.get('rewrite_ms')))}</td>"
            f"<td>{_esc(r.get('latency_retrieve_ms', (r.get('latency_ms') or {}).get('retrieve')))}</td>"
            f"<td>{_esc(r.get('latency_generate_ms', (r.get('latency_ms') or {}).get('generate')))}</td>"
            f"<td>{mark}</td></tr>")
    body.append("</table></div>")
    return _page("工单5 多轮对话响应时间", f"产物目录 {out_dir}", "".join(body))


# --------------------------------------------------------------------- 页 3
def page_regression(out_dir: Path, reg: dict, abn: dict) -> str:
    body = []
    if not reg:
        body.append(_missing("单轮回归（工单4 的 16 题）"))
    else:
        body.append("<div class=\"card\"><h2>单轮回归（工单4 的 16 题，验证工单5 未退化）</h2><div class=\"kv\">")
        body.append(_kpi(f"{reg.get('n_correct')}/{reg.get('n_judged')}", "判对题数",
                         "ok" if reg.get("n_correct") == reg.get("n_judged") else "bad"))
        body.append(_kpi(reg.get("accuracy"), "准确率"))
        body.append(_kpi(reg.get("regression_status", "—"), "回归结论"))
        body.append(_kpi((reg.get("latency_wall") or {}).get("p50", "—"), "延迟 P50 (ms)"))
        body.append(_kpi((reg.get("latency_wall") or {}).get("max", "—"), "延迟 max (ms)"))
        body.append("</div>")
        body.append("<div class=\"note\">判分复用 <span class=\"mono\">测试/common/golden.py::judge_answer</span>"
                    "（工单4 的确定性判分器，非 LLM 自评）。</div></div>")
        items = reg.get("items") or reg.get("details") or reg.get("rows") or []
        if items:
            body.append("<div class=\"card\"><h2>逐题明细</h2><table>"
                        "<tr><th>golden id</th><th>类别</th><th>判对</th><th>wall(ms)</th>"
                        "<th>系统答案（摘要）</th></tr>")
            for it in items:
                mark = ('<span class="pass">是</span>' if it.get("correct")
                        else '<span class="fail">否</span>')
                txt = (it.get("system_answer") or it.get("answer") or "").replace(chr(10), " ")
                body.append(f"<tr><td>{_esc(it.get('id'))}</td><td>{_esc(it.get('category'))}</td>"
                            f"<td>{mark}</td>"
                            f"<td>{_esc(it.get('latency_wall_ms', it.get('wall_ms')))}</td>"
                            f"<td>{_esc(txt[:80])}</td></tr>")
            body.append("</table></div>")

    if not abn:
        body.append(_missing("异常输入"))
    else:
        body.append("<div class=\"card\"><h2>异常输入（空 / 纯符号 / 乱码 / 注入串）</h2><div class=\"kv\">")
        body.append(_kpi(f"{abn.get('n_passed')}/{abn.get('n_cases') or abn.get('n')}", "通过条数",
                         "ok" if abn.get("n_passed") == (abn.get("n_cases") or abn.get("n")) else "bad"))
        body.append(_kpi(abn.get("pass_rate", abn.get("accuracy")), "通过率"))
        body.append(_kpi("不崩溃 / 非空 / 不 5xx", "判据"))
        body.append("</div>")
        body.append(f"<div class=\"note\">{_esc(abn.get('criterion', ''))}</div></div>")
        rows = abn.get("items") or abn.get("details") or abn.get("rows") or []
        if rows:
            body.append("<div class=\"card\"><h2>逐条明细</h2><table>"
                        "<tr><th>id</th><th>输入</th><th>期望</th><th>判定</th>"
                        "<th>用时(ms)</th><th>回答（摘要）</th></tr>")
            for it in rows:
                mark = ('<span class="pass">通过</span>' if it.get("passed")
                        else '<span class="fail">未通过</span>')
                qtext = it.get("question_repr") or it.get("question") or ""
                body.append(f"<tr><td>{_esc(it.get('id'))}</td>"
                            f"<td class=\"mono\">{_esc(str(qtext)[:34])}</td>"
                            f"<td>{_esc(it.get('expect'))}</td><td>{mark}</td>"
                            f"<td>{_esc(it.get('elapsed_ms', it.get('wall_ms')))}</td>"
                            f"<td>{_esc((it.get('answer') or '').replace(chr(10), ' ')[:60])}</td></tr>")
            body.append("</table></div>")
    return _page("工单5 单轮回归与异常输入", f"产物目录 {out_dir}", "".join(body))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"多轮评测证据页生成（工单：{WORK_ORDER}）")
    ap.add_argument("--run-tag", default=DEFAULT_RUN_TAG, help="评测产物子目录名")
    ap.add_argument("--out-dir", default=None, help="评测产物根目录")
    args = ap.parse_args(argv)

    root = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "测试" / "测试数据" / "多轮评测结果")
    run_dir = root / args.run_tag
    page_dir = REPO_ROOT / "测试" / "测试数据" / "截图页"
    page_dir.mkdir(parents=True, exist_ok=True)

    summary = _load(run_dir / "summary.json") or {}
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()] if (run_dir / "results.jsonl").is_file() else []
    reg = _load(run_dir / "regression16.json") or {}
    abn = _load(run_dir / "abnormal.json") or {}

    pages = {
        "准确率汇总.html": page_summary(run_dir, summary, rows),
        "响应时间_多轮.html": page_latency(run_dir, summary, rows),
        "单轮回归与异常输入.html": page_regression(run_dir, reg, abn),
    }
    for name, content in pages.items():
        (page_dir / name).write_text(content, encoding="utf-8")
        print(f"  写出 测试/测试数据/截图页/{name}  ({len(content)} 字符)")
    print(f"[完成] 证据页目录 {page_dir}")
    if not summary:
        print("[警告] summary.json 缺失 —— 页面会显示「未运行」占位，不会伪造数字。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
