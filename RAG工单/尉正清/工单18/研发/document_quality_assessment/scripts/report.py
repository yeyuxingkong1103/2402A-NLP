# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""出报告：结构化 JSON（给程序）+ HTML 简报（给人看）

工单要求 API 端点「以清晰、可读的格式（**JSON 及 HTML 简报**）返回评估报告，
特别是各类『待确认』、『待审核』列表」。

## 为什么两种都要

- **JSON**：给程序用 —— 工作流的决策节点要读 `tags.by_route` 来路由，
  验收标准里也点名要「返回结构化的 JSON 报告」
- **HTML**：给人用 —— 质检完总得有人扫一眼那几个列表

## HTML 简报的设计原则

工单验收标准有一条「**列表可操作性**」：每一项都必须包含足以让操作员判断的
关键信息（文件路径、相似片段、敏感信息上下文）。

所以简报里**不出现只有数字的列表** —— 凡是列出来的，都带路径和上下文。
只有统计意义的表格（格式分布、长度分位）例外，那些本来就是看数字的。
"""
import html
import json
from pathlib import Path

# details 各段的表头，和 assess.build_report 里造的行一一对应
COLS = {
    "md5_groups": ["MD5 前 8 位", "文件数", "文件（同一 MD5 = 完全相同的文件）"],
    "sim_pairs": ["文件 A", "文件 B", "相似度", "汉明距离"],
    "uncertain": ["文件", "初判", "扫描页占比", "判定依据"],
    "hits": ["文件", "类型", "命中内容", "上下文"],
    "errors": ["文件", "错误原因"],
    "docs": ["文件", "标签（按规则优先级）", "主路由（工作流据此选解析器）"],
}
KIND_CN = {"scanned": "扫描型", "text": "文字型", "mixed": "混合型",
           "unreadable": "打不开"}
TYPE_CN = {"phone": "手机号", "email": "邮箱", "id_card": "身份证", "bank_card": "银行卡"}


def build(summary, details, cfg):
    """把各部分汇总成一份报告 dict。"""
    return {
        "report_version": "1.0",
        "skill": "DocumentQualityAssessmentSkill",
        "summary": summary,
        "details": details,
    }


def write_json(report, path):
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return path


# ---------------------------------------------------------------- HTML

def _esc(s):
    return html.escape(str(s), quote=True)


def _table(title, key, rows, cap, note="", empty="✓ 无", warn=False):
    """一个「列表」小节：标题 + 说明 + 表格。

    统一走这里，就不会出现「有的列表带路径、有的只有数字」这种不一致。
    """
    cols = COLS[key]
    P = [f"<h2>{title}</h2>"]
    if note:
        P.append(f"<p class='{'warn' if warn else 'ctx'}'>{note}</p>")
    if not rows:
        P.append(f"<p class='ok'>{empty}</p>")
        return P
    P.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
    for r in rows[:cap]:
        cells = []
        for i, c in enumerate(r):
            if key == "uncertain" and c in KIND_CN:
                c = KIND_CN[c]
            if key == "hits" and c in TYPE_CN:
                c = TYPE_CN[c]
            cls = " class='ctx'" if key == "hits" and i == len(r) - 1 else ""
            cells.append(f"<td{cls}>{_esc(c)}</td>")
        P.append("<tr>" + "".join(cells) + "</tr>")
    P.append("</table>")
    if len(rows) > cap:
        P.append(f"<p class='ctx'>（共 {len(rows)} 条，此处只列前 {cap} 条，"
                 f"完整列表见同目录 JSON 报告）</p>")
    return P


def write_html(report, path, cfg):
    """生成 HTML 简报。每个列表最多列 html_max_rows 条，超出提示总数。"""
    s, d = report["summary"], report["details"]
    cap = cfg["output"]["html_max_rows"]
    P = ["<html><head><meta charset='utf-8'><title>文档质量评估报告</title>",
         """<style>
    body{font-family:"Microsoft YaHei",sans-serif;margin:24px;color:#222;line-height:1.6}
    h1{font-size:20px;border-bottom:2px solid #333;padding-bottom:6px}
    h2{font-size:16px;margin-top:28px;border-left:4px solid #0b5cad;padding-left:8px}
    table{border-collapse:collapse;margin:8px 0;font-size:13px}
    th,td{border:1px solid #ddd;padding:5px 9px;text-align:left;vertical-align:top}
    th{background:#f4f6f8}
    .k{font-weight:bold}
    .warn{color:#b42318;font-weight:bold}
    .ok{color:#1a7f37}
    .ctx{color:#555;font-size:12px}
    code{background:#f4f4f4;padding:1px 4px;border-radius:3px}
    </style></head><body>""",
         "<h1>文档质量评估报告</h1>",
         f"<p>扫描根目录：<code>{_esc(s['root'])}</code>　"
         f"文件总数：<span class='k'>{s['total_files']}</span>　"
         f"总体积：{s['total_bytes'] / 1048576:.1f}MB　"
         f"解析用时：{s.get('elapsed_sec', 0)}s</p>"]

    # ---- 1. 格式分布 ----
    P.append("<h2>1. 格式分布</h2><table><tr><th>扩展名</th><th>数量</th><th>占比</th></tr>")
    for ext, n in d["format"]["by_ext"].items():
        P.append(f"<tr><td>{_esc(ext)}</td><td>{n}</td>"
                 f"<td>{d['format']['ratio'][ext]:.1%}</td></tr>")
    P.append("</table>")

    # ---- 2. PDF 分流 ----
    pt = d["pdf"]
    P.append("<h2>2. PDF 文字型 / 扫描型分流</h2>")
    P.append("<table><tr><th>类型</th><th>数量</th><th>占比</th><th>下游解析器</th></tr>")
    for k, route in [("scanned", "OCR 解析器"), ("text", "直接抽文字"),
                     ("mixed", "逐页判断"), ("unreadable", "人工处理")]:
        n = pt["counts"].get(k, 0)
        P.append(f"<tr><td>{KIND_CN[k]}</td><td>{n}</td>"
                 f"<td>{pt['ratios'].get(k, 0):.1%}</td><td>{route}</td></tr>")
    P.append("</table>")
    if pt["uncertain"]:
        P.append(f"<p class='warn'>⚠ 有 {len(pt['uncertain'])} 份判定存疑"
                 f"（扫描页占比落在中间地带，或只抽样了前若干页），见第 5 节。</p>")

    # ---- 3. 长度分布 ----
    L = d["length"]
    P.append("<h2>3. 文档长度分布</h2>")
    P.append("<table><tr><th>分位</th>"
             + "".join(f"<th>{k}</th>" for k in L["percentiles"]) + "</tr><tr><td>字符数</td>"
             + "".join(f"<td>{v:,}</td>" for v in L["percentiles"].values()) + "</tr></table>")
    P.append("<table><tr><th>区间（字符）</th><th>数量</th><th>占比</th></tr>")
    for b in L["distribution"]:
        P.append(f"<tr><td>{_esc(b['range'])}</td><td>{b['count']}</td>"
                 f"<td>{b['ratio']:.1%}</td></tr>")
    P.append("</table>")
    P.append(f"<p class='ctx'>过短（&lt;{cfg['length']['too_short']} 字符）{L['too_short']} 份，"
             f"空文档 {L['empty']} 份，超长（&gt;{cfg['length']['too_long']} 字符）"
             f"{L['too_long']} 份，解析失败 {L['failed_count']} 份。</p>")

    # ---- 4~8. 各列表 ----
    dup = d["dup"]
    P += _table("4. 完全重复的文件（MD5 精确匹配）", "md5_groups", dup["md5_groups"], cap,
                note="同一 MD5 就是同一个文件，可直接去重，无需人工确认。")
    P += _table("5. 待确认的 PDF 类型判定", "uncertain", pt["uncertain"], cap,
                note="启发式判定拿不准的（扫描页占比在中间地带，或只抽样了部分页），"
                     "列出来人工定夺。", warn=bool(pt["uncertain"]))
    P += _table("6. 待确认的版本冲突列表（SimHash 近似）", "sim_pairs", dup["sim_pairs"], cap,
                note=(dup["note"] or "SimHash 是近似匹配，**每条都需要人工确认**"
                                     "是不是同一份文档的不同版本。"),
                warn=bool(dup["note"]))
    P += _table("7. 待审核的敏感信息列表", "hits", d["sensitive"]["hits"], cap,
                note="每条都带前后文，方便判断是不是误报（专利号、申请号、统计表格里的"
                     "长数字最容易误判）。",
                warn=bool(d["sensitive"]["hits"]))

    # ---- 8. 标签 ----
    T = d["tags"]
    P.append("<h2>8. 文档分类标签与路由</h2>"
             "<table><tr><th>标签</th><th>数量</th><th>下游解析器</th></tr>")
    for name, n in sorted(T["by_tag"].items(), key=lambda x: -x[1]):
        P.append(f"<tr><td>{_esc(name)}</td><td>{n}</td>"
                 f"<td>{_esc('、'.join(T['tag_routes'].get(name, [])))}</td></tr>")
    P.append("</table>")
    P.append("<p class='k'>主路由汇总（工作流决策节点读这个）："
             + "；".join(f"<code>{_esc(k)}</code> {v} 份"
                         for k, v in sorted(T["by_primary"].items(), key=lambda x: -x[1]))
             + "</p>")
    P.append("<p class='ctx'>上表「下游解析器」列是该标签单独对应的去向；"
             "一份文档可能带多个标签（如「文字型 + 超长」），"
             "所以各行相加不等于总数。<b>主路由</b>取每份文档第一条标签（规则按"
             "最确定→最兜底排序），一份文档只有一个主路由，工作流按它选解析器，"
             "其余标签当参数用。</p>")
    P += _table("9. 逐份文档标签与主路由", "docs", T["docs"], cap,
                note="每份文档打得什么标签、主路由送去哪个解析器。")

    # ---- 10. 失败 ----
    P += _table("10. 解析失败 / 打不开的文件", "errors", d["errors"], cap,
                note="这些文件没进统计口径，需要人工处理或重下。")

    P.append("</body></html>")
    Path(path).write_text("\n".join(P), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 终端简报

def print_brief(rep):
    """终端简报 —— 演示视频和日常巡检看这个就够。"""
    s = rep["summary"]
    sim = s["similar_pairs"]
    if sim is None:
        simtxt = "跳过（文件数超出 simhash_max_files）"
    elif s.get("simhash_skipped_thin"):
        simtxt = f"{sim} 对（另有 {s['simhash_skipped_thin']} 份正文过短未参与比较）"
    else:
        simtxt = f"{sim} 对"
    print("\n========== 文档质量评估简报 ==========")
    print(f"文件总数 {s['total_files']} 份，{s['total_bytes'] / 1048576:.0f}MB"
          f"，解析失败 {s['parse_failed']} 份")
    print("格式分布  " + "   ".join(f"{k} {v}" for k, v in list(s["ext"].items())[:8]))
    print("PDF 分流  " + "   ".join(f"{k} {v}" for k, v in s["pdf"].items())
          + f"   （扫描型占 {s['pdf_scanned_ratio']:.0%}）")
    print(f"长度分布  P50 {s['chars'].get('P50', 0):,} 字符"
          f"   P90 {s['chars'].get('P90', 0):,}   总字符 {s['chars']['total']:,}")
    print(f"完全重复  {s['exact_dup_groups']} 组（冗余副本 {s['redundant_copies']} 份）")
    print(f"待确认版本冲突  {simtxt}")
    print(f"敏感信息命中  {s['sensitive_hits']} 条")
    print(f"存疑项  PDF 判定待确认 {s['uncertain_pdf']} 份，打不开 {s['unreadable_pdf']} 份")
    print("主路由    " + "   ".join(
        f"{k} ← {v} 份" for k, v in sorted(s["by_primary_route"].items(),
                                           key=lambda x: -x[1])))
    print("======================================")
