# -*- coding: utf-8 -*-
"""DQA · 报告生成（JSON 结构化 + HTML 简报 + Markdown）"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path


def build_report(root: str | Path, format_stats: dict, pdf_types: dict,
                 lengths: dict, dedup: dict, sensitive: dict,
                 labels: dict, elapsed: float, config: dict | None = None) -> dict:
    """把各模块结果组装成统一的结构化报告（去掉内部 _detail/_files 大字段）"""
    def strip(d):
        return {k: v for k, v in (d or {}).items() if not k.startswith("_")}

    return {
        "报告类型": "文档质量评估报告",
        "工单": "人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流",
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "输入目录": str(root),
        "耗时秒": round(elapsed, 2),
        "配置": config or {},
        "总览": {
            "文件总数": format_stats.get("文件总数", 0),
            "总大小MB": format_stats.get("总大小MB", 0),
            "PDF扫描型": pdf_types.get("类型分布", {}).get("scanned", {}).get("数量", 0),
            "PDF混合型": pdf_types.get("类型分布", {}).get("mixed", {}).get("数量", 0),
            "PDF文字型": pdf_types.get("类型分布", {}).get("text", {}).get("数量", 0),
            "MD5精确重复可清理": dedup.get("MD5可清理文件数", 0),
            "版本冲突待确认": dedup.get("待确认数量", 0),
            "敏感信息待审核": sensitive.get("待审核总数", 0),
            "需要OCR路由": labels.get("需要OCR路由的文件数", 0),
        },
        "一_格式分布统计": strip(format_stats),
        "二_PDF页面类型识别": strip(pdf_types),
        "三_文档长度分布": strip(lengths),
        "四_重复检测": strip(dedup),
        "五_敏感信息检测": strip(sensitive),
        "六_文档分类标签": strip(labels),
    }


# ------------------------------------------------------------------ HTML
_CSS = """
body{font-family:"Microsoft YaHei",Arial,sans-serif;margin:0;background:#f6f8fa;color:#1f2933}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 6px}
h2{font-size:16px;margin:26px 0 10px;padding-left:8px;border-left:4px solid #2b7de9}
.sub{color:#7b8794;font-size:12px;margin-bottom:18px}
.cards{display:flex;flex-wrap:wrap;gap:10px}
.card{flex:1 1 150px;background:#fff;border:1px solid #e3e8ee;border-radius:10px;padding:12px}
.card .k{font-size:12px;color:#7b8794}
.card .v{font-size:22px;font-weight:600;margin-top:4px}
.card.warn .v{color:#d9534f}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;
      border:1px solid #e3e8ee;border-radius:8px;overflow:hidden}
th,td{border-bottom:1px solid #eef2f6;padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f0f4f9;font-weight:600}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;
       background:#eef4ff;color:#2b6cb0;margin-right:4px}
.badge.risk{background:#fdecea;color:#b3261e}
.empty{color:#9aa5b1;font-style:italic}
details{background:#fff;border:1px solid #e3e8ee;border-radius:8px;padding:8px 12px;margin-top:8px}
summary{cursor:pointer;font-weight:600}
"""


def _table(rows, headers):
    if not rows:
        return '<div class="empty">（无）</div>'
    out = ["<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(str(c))[:300]}</td>" for c in r) + "</tr>")
    return "".join(out) + "</table>"


def to_html(rep: dict, max_rows: int = 60) -> str:
    ov = rep["总览"]
    cards = [("文件总数", ov["文件总数"], ""),
             ("总大小(MB)", ov["总大小MB"], ""),
             ("扫描型PDF", ov["PDF扫描型"], "warn"),
             ("混合型PDF", ov["PDF混合型"], ""),
             ("MD5重复可清理", ov["MD5精确重复可清理"], "warn"),
             ("版本冲突待确认", ov["版本冲突待确认"], "warn"),
             ("敏感信息待审核", ov["敏感信息待审核"], "warn"),
             ("需OCR路由", ov["需要OCR路由"], "")]
    card_html = "".join(
        f'<div class="card {cls}"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v, cls in cards)

    fs = rep["一_格式分布统计"]
    ext_rows = [[k, v["数量"], f"{v['占比']:.1%}"] for k, v in
                list(fs.get("按扩展名", {}).items())[:max_rows]]
    grp_rows = [[k, v["数量"], f"{v['占比']:.1%}"] for k, v in fs.get("按类别", {}).items()]

    pt = rep["二_PDF页面类型识别"]
    type_rows = [[k, v["数量"], f"{v['占比']:.1%}"] for k, v in pt.get("类型分布", {}).items()]
    review_rows = [[r["file"], r["type"], r["scanned_ratio"], r["avg_chars"], r["reason"]]
                   for r in pt.get("待确认列表", [])[:max_rows]]

    ln = rep["三_文档长度分布"]
    len_rows = [[k, ln.get(k)] for k in
                ["统计文件数", "总字符数", "平均字符数", "P25", "P50", "P75", "P90", "P99",
                 "最小值", "最大值"]]
    bin_rows = [[b["区间"], b["数量"], f"{b['占比']:.1%}"] for b in ln.get("长度区间分布", [])]

    dd = rep["四_重复检测"]
    dup_rows = [[g["md5"][:10], g["数量"], "、".join(g["文件"][:3])]
                for g in dd.get("MD5重复组", [])[:max_rows]]
    vc_rows = [[c["文件A"], c["文件B"], c["相似度"], c["建议"]]
               for c in dd.get("待确认的版本冲突列表", [])[:max_rows]]

    se = rep["五_敏感信息检测"]
    se_rows = [[r["类型"], r["命中内容"], r["文件"], r["上下文"]]
               for r in se.get("待审核敏感信息列表", [])[:max_rows]]

    lb = rep["六_文档分类标签"]
    lb_rows = [[r["file"], " ".join(r["tags"]), r["domain"], r["route"]]
               for r in lb.get("标签清单", [])[:max_rows]]

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>文档质量评估报告</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>知识库文档质量评估报告</h1>
<div class="sub">目录：{html.escape(rep['输入目录'])}　|　生成时间：{rep['生成时间']}　|　耗时：{rep['耗时秒']}s</div>
<div class="cards">{card_html}</div>

<h2>一、格式分布统计</h2>
{_table(grp_rows, ["类别", "数量", "占比"])}
<details><summary>按扩展名明细</summary>{_table(ext_rows, ["扩展名", "数量", "占比"])}</details>

<h2>二、PDF 页面类型识别（文字型 / 扫描型 / 混合型）</h2>
{_table(type_rows, ["类型", "数量", "占比"])}
<details open><summary>⚠️ 待确认列表（启发式边界，需人工复核）</summary>
{_table(review_rows, ["文件", "判定", "扫描页占比", "平均字符数", "原因"])}</details>

<h2>三、文档长度分布</h2>
{_table(len_rows, ["指标", "值"])}
{_table(bin_rows, ["长度区间", "数量", "占比"])}

<h2>四、重复检测</h2>
<p>MD5 精确重复组（可直接清理）：{dd.get('MD5精确重复组数', 0)} 组</p>
{_table(dup_rows, ["MD5", "数量", "文件"])}
<details open><summary>⚠️ 待确认的版本冲突列表（SimHash 高相似，需人工确认）</summary>
{_table(vc_rows, ["文件A", "文件B", "相似度", "建议"])}</details>

<h2>五、敏感信息检测</h2>
{_table(se_rows, ["类型", "命中内容", "文件", "上下文"])}

<h2>六、文档分类标签</h2>
{_table(lb_rows, ["文件", "标签", "领域", "建议路由"])}
</div></body></html>"""


def to_markdown(rep: dict, max_rows: int = 40) -> str:
    ov = rep["总览"]
    L = ["# 知识库文档质量评估报告", "",
         f"- 输入目录：`{rep['输入目录']}`",
         f"- 生成时间：{rep['生成时间']}　耗时：{rep['耗时秒']}s", "",
         "## 总览", "", "| 指标 | 值 |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in ov.items()]

    fs = rep["一_格式分布统计"]
    L += ["", "## 一、格式分布统计", "", "| 类别 | 数量 | 占比 |", "|---|---|---|"]
    L += [f"| {k} | {v['数量']} | {v['占比']:.1%} |" for k, v in fs.get("按类别", {}).items()]

    pt = rep["二_PDF页面类型识别"]
    L += ["", "## 二、PDF 页面类型", "", "| 类型 | 数量 | 占比 |", "|---|---|---|"]
    L += [f"| {k} | {v['数量']} | {v['占比']:.1%} |" for k, v in pt.get("类型分布", {}).items()]
    L += ["", "### ⚠️ 待确认列表", "", "| 文件 | 判定 | 扫描占比 | 平均字符 | 原因 |", "|---|---|---|---|---|"]
    L += [f"| {r['file']} | {r['type']} | {r['scanned_ratio']} | {r['avg_chars']} | {r['reason']} |"
          for r in pt.get("待确认列表", [])[:max_rows]]

    ln = rep["三_文档长度分布"]
    L += ["", "## 三、文档长度分布", "",
          f"P25={ln.get('P25')}　P50={ln.get('P50')}　P75={ln.get('P75')}　"
          f"P90={ln.get('P90')}　P99={ln.get('P99')}", "",
          "| 区间 | 数量 | 占比 |", "|---|---|---|"]
    L += [f"| {b['区间']} | {b['数量']} | {b['占比']:.1%} |" for b in ln.get("长度区间分布", [])]

    dd = rep["四_重复检测"]
    L += ["", "## 四、重复检测", "",
          f"- MD5 精确重复组：{dd.get('MD5精确重复组数')}，可清理 {dd.get('MD5可清理文件数')} 个文件",
          "", "### ⚠️ 待确认的版本冲突列表", "",
          "| 文件A | 文件B | 相似度 | 建议 |", "|---|---|---|---|"]
    L += [f"| {c['文件A']} | {c['文件B']} | {c['相似度']} | {c['建议']} |"
          for c in dd.get("待确认的版本冲突列表", [])[:max_rows]]

    se = rep["五_敏感信息检测"]
    L += ["", "## 五、敏感信息检测（待审核列表）", "",
          f"各类型命中：{se.get('各类型命中数')}", "",
          "| 类型 | 命中内容 | 文件 | 上下文 |", "|---|---|---|---|"]
    L += [f"| {r['类型']} | {r['命中内容']} | {r['文件']} | {r['上下文'][:60]} |"
          for r in se.get("待审核敏感信息列表", [])[:max_rows]]

    lb = rep["六_文档分类标签"]
    L += ["", "## 六、文档分类标签", "", f"标签分布：{lb.get('标签分布')}", "",
          "| 文件 | 标签 | 领域 | 建议路由 |", "|---|---|---|---|"]
    L += [f"| {r['file']} | {' '.join(r['tags'])} | {r['domain']} | {r['route']} |"
          for r in lb.get("标签清单", [])[:max_rows]]
    return "\n".join(L)


def save_all(rep: dict, outdir: str | Path, stem: str = "quality_report") -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p_json = outdir / f"{stem}.json"
    p_html = outdir / f"{stem}.html"
    p_md = outdir / f"{stem}.md"
    p_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    p_html.write_text(to_html(rep), encoding="utf-8")
    p_md.write_text(to_markdown(rep), encoding="utf-8")
    return {"json": str(p_json), "html": str(p_html), "md": str(p_md)}
