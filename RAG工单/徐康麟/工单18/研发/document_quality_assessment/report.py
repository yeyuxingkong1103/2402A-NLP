# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单

模块：document_quality_assessment.report
职责：报告输出。
  * ``write_json_report`` ：结构化全量 JSON 报告
  * ``write_per_file_jsonl``：逐文件明细（jsonl，供路由/审计）
  * ``write_html_report``  ：**自包含** HTML 简报（内联 CSS + 纯 SVG 图表，零外部资源，
    可离线双击打开），重点突出三类列表：待确认的版本冲突 / 待审核的敏感信息 / 待确认的 PDF 分流。
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

# 图表配色（与标签语义无关，仅视觉）
PALETTE = [
    "#2f6fd0", "#d9534f", "#f0ad4e", "#5cb85c", "#8e6fd0",
    "#3aa8c1", "#c1703a", "#7a8699", "#b0486f", "#4f9e6b",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_report(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    _ensure_parent(p)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def write_per_file_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    _ensure_parent(p)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            row = dict(rec)
            row.pop("sensitive_detail", None)  # 明细进主报告，jsonl 只留计数，避免重复放大
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


# ---------------------------------------------------------------------------
# 纯 SVG 图表（无 JS / 无外部依赖，保证离线可看）
# ---------------------------------------------------------------------------
def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def svg_pie(items: Sequence[tuple[str, float]], *, size: int = 300, title: str = "") -> str:
    """饼图（含图例）。``items``：[(标签, 数值)]。"""
    total = sum(max(0.0, float(v)) for _, v in items)
    if total <= 0:
        return f'<div class="chart-empty">（{_esc(title)}：无数据）</div>'
    cx = cy = size / 2
    r = size / 2 - 12
    angle = -90.0
    paths: list[str] = []
    legend: list[str] = []
    for i, (label, value) in enumerate(items):
        value = max(0.0, float(value))
        if value <= 0:
            continue
        sweep = 360.0 * value / total
        color = PALETTE[i % len(PALETTE)]
        a0, a1 = angle, angle + sweep
        large = 1 if sweep > 180 else 0
        import math

        x0 = cx + r * math.cos(math.radians(a0))
        y0 = cy + r * math.sin(math.radians(a0))
        x1 = cx + r * math.cos(math.radians(a1))
        y1 = cy + r * math.sin(math.radians(a1))
        if sweep >= 359.999:
            paths.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{color}"/>')
        else:
            paths.append(
                f'<path d="M {cx:.1f} {cy:.1f} L {x0:.1f} {y0:.1f} '
                f'A {r:.1f} {r:.1f} 0 {large} 1 {x1:.1f} {y1:.1f} Z" fill="{color}"/>'
            )
        pct = 100.0 * value / total
        legend.append(
            f'<li><span class="swatch" style="background:{color}"></span>'
            f'{_esc(label)} <b>{int(value)}</b> <span class="muted">({pct:.1f}%)</span></li>'
        )
        angle = a1
    return (
        f'<div class="chart-row"><svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
        f'role="img" aria-label="{_esc(title)}">{"".join(paths)}</svg>'
        f'<ul class="legend">{"".join(legend)}</ul></div>'
    )


def svg_bars(items: Sequence[tuple[str, float]], *, width: int = 720, bar_h: int = 26, title: str = "") -> str:
    """横向条形图。"""
    if not items:
        return f'<div class="chart-empty">（{_esc(title)}：无数据）</div>'
    top = max(max(float(v) for _, v in items), 1.0)
    label_w = 150
    value_w = 90
    bar_max = max(40, width - label_w - value_w - 20)
    height = bar_h * len(items) + 10
    parts: list[str] = []
    for i, (label, value) in enumerate(items):
        y = i * bar_h + 6
        w = bar_max * max(0.0, float(value)) / top
        color = PALETTE[i % len(PALETTE)]
        parts.append(
            f'<text x="0" y="{y + bar_h * 0.62:.0f}" class="bar-label">{_esc(label)}</text>'
            f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{bar_h - 10}" fill="{color}" rx="3"/>'
            f'<text x="{label_w + w + 8:.0f}" y="{y + bar_h * 0.62:.0f}" class="bar-value">{_esc(_fmt(value))}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
        f'aria-label="{_esc(title)}">{"".join(parts)}</svg>'
    )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *, cls: str = "", max_rows: int = 0) -> str:
    rows = list(rows)
    truncated = 0
    if max_rows and len(rows) > max_rows:
        truncated = len(rows) - max_rows
        rows = rows[:max_rows]
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for r in rows:
        cells = "".join(f"<td>{_esc(c) if not isinstance(c, str) or '<' not in c else c}</td>" for c in r)
        body.append(f"<tr>{cells}</tr>")
    note = f'<div class="muted">（表格已截断，另有 {truncated} 行见 JSON 报告）</div>' if truncated else ""
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>{note}'


def _clip(text: Any, limit: int = 96) -> str:
    s = str(text).replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# HTML 简报
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#1b2027;--muted:#6b7280;--line:#dfe4ea;--band:#f6f8fb;--accent:#2f6fd0;
--warn:#b26a00;--bad:#c0392b;--ok:#1e8e4e;}
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 "Microsoft YaHei","Segoe UI",Arial,sans-serif;color:var(--ink);background:#eef1f6}
.wrap{max-width:1180px;margin:0 auto;padding:24px 18px 60px}
header{background:linear-gradient(135deg,#1d3a6b,#2f6fd0);color:#fff;border-radius:12px;padding:22px 26px}
header h1{margin:0 0 6px;font-size:24px}
header .sub{opacity:.9;font-size:13px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0;
box-shadow:0 1px 2px rgba(16,24,40,.04)}
h2{font-size:19px;margin:4px 0 12px;padding-left:10px;border-left:5px solid var(--accent)}
h3{font-size:16px;margin:16px 0 8px;color:#33405a}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 4px}
.kpi{flex:1 1 168px;background:var(--band);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .k{font-size:12px;color:var(--muted)}
.kpi .v{font-size:22px;font-weight:700;margin-top:2px}
.kpi.warn .v{color:var(--warn)} .kpi.bad .v{color:var(--bad)} .kpi.ok .v{color:var(--ok)}
table{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#e9eef7;font-weight:600;position:sticky;top:0}
tr:nth-child(even) td{background:var(--band)}
.chart-row{display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap}
.legend{list-style:none;margin:0;padding:0;font-size:13px;min-width:260px}
.legend li{padding:3px 0;border-bottom:1px dashed var(--line)}
.swatch{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:7px}
.bar-label{font-size:12px;fill:#33405a} .bar-value{font-size:12px;fill:#1b2027}
.muted{color:var(--muted);font-size:12px}
.chart-empty{color:var(--muted);font-size:13px;padding:10px 0}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;background:#e6eefc;color:#1d4ea8;font-size:12px;margin:1px 3px 1px 0}
.tag.bad{background:#fdecea;color:#a5281b}.tag.warn{background:#fdf4e3;color:#8a5a00}
code{background:#f2f4f8;padding:1px 5px;border-radius:4px;font-size:12px}
.hl{background:#fff3cd;padding:0 2px;border-radius:3px}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:22px}
"""


def render_html(report: dict[str, Any], *, max_rows: int = 60) -> str:
    r = report
    s = r.get("summary", {})
    skill = r.get("skill", {})
    run = r.get("run", {})
    fmt = r.get("format_distribution", {})
    pdf = r.get("pdf_type_analysis", {})
    length = r.get("length_distribution", {}).get("all", {})
    dup = r.get("duplicate_detection", {})
    sens = r.get("sensitive_detection", {})
    tags = r.get("tag_system", {}).get("distribution", {})
    pending = r.get("pending_lists", {})

    kpis = [
        ("文档总数", _fmt(s.get("total_files", 0)), ""),
        ("总体积 (MB)", _fmt(s.get("total_mb", 0)), ""),
        ("扫描型 PDF 占比", f"{100 * float(s.get('scan_pdf_ratio', 0) or 0):.1f}%", "warn"),
        ("MD5 完全重复组", _fmt(s.get("md5_duplicate_groups", 0)), "warn"),
        ("待确认版本冲突", _fmt(s.get("version_conflicts", 0)), "bad"),
        ("敏感信息待审文件", _fmt(s.get("sensitive_files", 0)), "bad"),
        ("损坏/加密文件", _fmt(s.get("corrupted_files", 0)), "bad" if s.get("corrupted_files") else "ok"),
        ("单份平均耗时 (s)", f"{float(run.get('avg_sec_per_file', 0) or 0):.3f}", ""),
    ]
    kpi_html = "".join(
        f'<div class="kpi {cls}"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v, cls in kpis
    )

    # 格式分布
    ext_items = [(row["name"], row["count"]) for row in fmt.get("by_extension", [])]
    group_items = [(row["name"], row["count"]) for row in fmt.get("by_group", [])]
    fmt_rows = [
        (row["name"], row["count"], f"{100 * row['ratio']:.2f}%", f"{row['bytes'] / 1048576:.2f} MB")
        for row in fmt.get("by_extension", [])
    ]

    # PDF 分流
    pdf_rows = [
        (k, v, f"{100 * float(pdf.get('ratios', {}).get(k, 0)):.2f}%")
        for k, v in (pdf.get("counts") or {}).items()
    ]

    # 长度
    qnames = ["P25", "P50", "P75", "P90", "P99"]
    qvals = length.get("quantiles", {})
    hist_items = [(h["label"], h["count"]) for h in length.get("histogram", [])]

    # 标签
    tag_items = [(row["name"], row["count"]) for row in tags.get("by_primary", [])]

    # 三个重点列表
    conflicts = pending.get("version_conflicts", [])
    conflict_rows = []
    for c in conflicts[:max_rows]:
        snippet = ""
        if c.get("similar_snippets"):
            sp = c["similar_snippets"][0]
            snippet = f"A: {_clip(sp.get('a', ''), 60)} ／ B: {_clip(sp.get('b', ''), 60)}"
        conflict_rows.append(
            (
                _clip(c.get("file_a"), 60),
                _clip(c.get("file_b"), 60),
                c.get("hamming_distance"),
                f"{float(c.get('similarity', 0)):.4f}",
                _esc(snippet) if snippet else "<span class='muted'>（无共同分片，见 JSON）</span>",
            )
        )

    sens_rows = []
    for item in sens.get("pending_review", [])[:max_rows]:
        for h in item.get("hits", [])[:3]:
            ctx = (h.get("contexts") or [""])[0]
            sens_rows.append(
                (
                    _clip(item.get("file"), 55),
                    h.get("rule"),
                    h.get("value"),
                    h.get("severity"),
                    html.escape(ctx).replace("【", '<span class="hl">【').replace("】", "】</span>"),
                )
            )

    pdf_confirm_rows = [
        (
            _clip(x.get("file"), 62),
            x.get("pdf_type"),
            f"{float(x.get('scan_ratio', 0)):.4f}",
            f"{x.get('scanned_pages')}/{x.get('sampled_pages')}",
            _clip(x.get("reason"), 70),
        )
        for x in pending.get("pdf_type_pending_confirm", [])[:max_rows]
    ]
    corrupted_rows = [
        (_clip(x.get("file"), 62), _fmt(x.get("size_bytes")), _clip(x.get("error"), 80), x.get("action"))
        for x in pending.get("corrupted_files", [])[:max_rows]
    ]

    thresholds = skill.get("thresholds", {})
    thr_rows = [(_esc(k), _esc(v)) for k, v in thresholds.items()]

    def _sec(title: str, inner: str) -> str:
        return f'<section class="card"><h2>{_esc(title)}</h2>{inner}</section>'

    parts: list[str] = []
    parts.append(
        f'<header><h1>{_esc(skill.get("name"))} v{_esc(skill.get("version"))} 质检简报</h1>'
        f'<div class="sub">生成时间 {_esc(skill.get("generated_at"))}　|　分析文件 {_fmt(run.get("analyzed_files"))} 份'
        f'（发现 {_fmt(run.get("discovered_files"))} 份，断点续跑跳过 {_fmt(run.get("skipped_from_state"))} 份）'
        f'　|　总耗时 {_fmt(run.get("wall_clock_sec"))} s　|　并行 {_esc(run.get("workers"))} 进程'
        f'　|　配置 <code>{_esc(skill.get("config_path"))}</code></div></header>'
    )
    parts.append(f'<section class="card"><h2>总览</h2><div class="kpis">{kpi_html}</div>'
                 f'<div class="muted">全部阈值来自配置文件，未硬编码；启发式分流结论须结合"待确认列表"人工复核。</div></section>')

    parts.append(_sec("一、格式分布统计",
                      svg_pie(group_items, title="格式大类分布") +
                      '<h3>按扩展名</h3>' + _table(["扩展名", "数量", "占比", "体积"], fmt_rows) +
                      '<h3>按格式大类（条形图）</h3>' + svg_bars(group_items, title="格式大类")))
    parts.append(_sec("二、PDF 文字型 / 扫描型 / 混合型分流",
                      svg_pie([(k, v) for k, v in (pdf.get("counts") or {}).items() if v], title="PDF 类型") +
                      '<h3>类型统计</h3>' + _table(["类型", "数量", "占比"], pdf_rows) +
                      '<h3>判定阈值（来自配置）</h3>' + _table(["阈值项", "取值"], [(k, v) for k, v in (pdf.get("thresholds") or {}).items()]) +
                      f'<div class="muted">抽检页合计 {_fmt(pdf.get("sampled_pages_total"))} 页，判为扫描页 {_fmt(pdf.get("scanned_pages_total"))} 页'
                      f'（页级扫描率 {100 * float(pdf.get("scanned_page_ratio", 0) or 0):.2f}%）。</div>'))
    parts.append(_sec("三、文档长度分布",
                      '<h3>分位数（字符数）</h3>' + _table(["分位", "字符数"], [(n, _fmt(qvals.get(n, 0))) for n in qnames]) +
                      f'<div class="muted">有效样本 {_fmt(length.get("count"))} 份；均值 {_fmt(length.get("mean"))}，'
                      f'最小 {_fmt(length.get("min"))}，最大 {_fmt(length.get("max"))}；'
                      f'过短 {_fmt(length.get("too_short"))} 份，过长 {_fmt(length.get("too_long"))} 份。</div>' +
                      '<h3>长度区间分布（直方图）</h3>' + svg_bars(hist_items, title="长度区间分布")))
    parts.append(_sec("四、重复检测（MD5 精确 + SimHash 近似）",
                      f'<div class="muted">MD5 完全重复：{_fmt(dup.get("md5_exact", {}).get("total_groups"))} 组 / '
                      f'{_fmt(dup.get("md5_exact", {}).get("duplicate_file_count"))} 个冗余文件 / 浪费 '
                      f'{float(dup.get("md5_exact", {}).get("wasted_bytes", 0)) / 1048576:.2f} MB；'
                      f'SimHash 候选比对 {_fmt(dup.get("simhash_near", {}).get("candidates_compared"))} 对'
                      f'（暴力全比需 {_fmt(dup.get("simhash_near", {}).get("comparisons_if_brute_force"))} 对），'
                      f'海明距离阈值 ≤ {_esc(dup.get("simhash_near", {}).get("hamming_threshold"))}。</div>' +
                      '<h3>MD5 完全重复组</h3>' +
                      _table(["MD5", "份数", "保留", "其余重复文件"],
                             [(g["md5"][:16] + "…", g["count"], _clip(g["keep"], 46),
                               _clip("；".join(g["files"][1:]), 90)) for g in dup.get("md5_exact", {}).get("duplicate_groups", [])],
                             max_rows=max_rows) +
                      '<h3>⚠ 待确认的版本冲突列表（SimHash 高相似，需人工确认）</h3>' +
                      _table(["文件 A", "文件 B", "海明距离", "相似度", "相似片段对照"],
                             conflict_rows, cls="warn-table", max_rows=max_rows)))
    parts.append(_sec("五、敏感信息检测（待审核列表）",
                      f'<div class="muted">启用规则：{_esc("、".join(sens.get("enabled_rules", [])))}；'
                      f'默认关闭：{_esc("、".join(sens.get("disabled_rules", [])))}（工单原文要求银行卡默认关闭）。'
                      f'命中合计 {_fmt(sens.get("total_hits"))} 处，涉及 {_fmt(sens.get("files_with_hits"))} 个文件。</div>' +
                      '<h3>⚠ 待审核列表（每条附上下文片段）</h3>' +
                      _table(["文件", "规则", "命中值", "级别", "上下文（【】内为命中值，前后各 30 字符）"],
                             sens_rows, cls="bad-table", max_rows=max_rows)))
    parts.append(_sec("六、分类标签体系与分布",
                      '<h3>主标签分布</h3>' + svg_bars(tag_items, title="主标签分布") +
                      "<h3>全部标签命中次数</h3>" +
                      _table(["标签", "文件数", "占比"],
                             [(row["name"], row["count"], f"{100 * row['ratio']:.2f}%") for row in tags.get("by_all_tags", [])]) +
                      '<h3>标签规则（来自配置，顺序即优先级）</h3>' +
                      _table(["标签", "含义"],
                             [(k, (v or {}).get("description", "")) for k, v in (r.get("tag_system", {}).get("definitions") or {}).items()])))
    parts.append(_sec("七、待确认 / 异常清单",
                      '<h3>PDF 分流待确认</h3>' +
                      _table(["文件", "判定类型", "扫描页占比", "扫描页/抽检页", "待确认原因"], pdf_confirm_rows, max_rows=max_rows) +
                      '<h3>损坏 / 加密 / 0 字节文件</h3>' +
                      _table(["文件", "字节", "错误", "建议动作"], corrupted_rows, max_rows=max_rows)))
    parts.append(_sec("八、复现信息与阈值快照",
                      _table(["项", "值"], thr_rows + [
                          ("配置文件", _esc(skill.get("config_path"))),
                          ("配置版本", _esc(skill.get("config_version"))),
                          ("Python", _esc(skill.get("python"))),
                          ("平台", _esc(skill.get("platform"))),
                          ("逐文件明细", "<code>研发/data/reports/per_file.jsonl</code>"),
                          ("结构化全量报告", "<code>研发/data/reports/quality_report.json</code>"),
                      ])))
    parts.append(
        '<footer>本简报由 DocumentQualityAssessmentSkill 自动生成，内联 CSS + SVG，无外部依赖，可离线打开。'
        '所有数字均可回溯到 <code>研发/data/reports/quality_report.json</code> 与 <code>per_file.jsonl</code>。</footer>'
    )
    body = "".join(parts)
    return (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(skill.get('name'))} 质检简报</title><style>{CSS}</style></head>"
        f'<body><div class="wrap">{body}</div></body></html>'
    )


def write_html_report(report: dict[str, Any], path: str | Path, *, max_rows: int | None = None) -> Path:
    p = Path(path)
    _ensure_parent(p)
    if max_rows is None:
        max_rows = 60
    content = render_html(report, max_rows=max_rows)
    p.write_text(content, encoding="utf-8")
    return p
