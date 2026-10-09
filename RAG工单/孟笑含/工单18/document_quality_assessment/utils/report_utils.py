def to_html_brief(report):
    s = report["summary"]
    tc = report["to_confirm"]
    rows = []
    rows.append("<h2>文档质量评估简报</h2>")
    rows.append("<p>目录：%s</p>" % s["folder"])
    rows.append("<table border='1' cellpadding='6'>")
    rows.append("<tr><th>指标</th><th>数值</th></tr>")
    rows.append("<tr><td>总文件数</td><td>%d</td></tr>" % s["total_files"])
    rows.append("<tr><td>PDF 文件数</td><td>%d</td></tr>" % s["pdf_files"])
    rows.append("<tr><td>扫描型 PDF</td><td>%d</td></tr>" % s["scan_pdf"])
    rows.append("<tr><td>文字型 PDF</td><td>%d</td></tr>" % s["text_pdf"])
    rows.append("<tr><td>混合型 PDF</td><td>%d</td></tr>" % s["mixed_pdf"])
    rows.append("</table>")

    rows.append("<h3>格式分布</h3><ul>")
    for ext, v in report["format_distribution"]["distribution"].items():
        rows.append("<li>%s: %d (%.2f%%)</li>" % (ext, v["count"], v["ratio"] * 100))
    rows.append("</ul>")

    rows.append("<h3>待确认列表</h3>")
    rows.append("<p>混合型 PDF：%d</p>" % len(tc["pdf_mixed"]))
    rows.append("<p>版本冲突：%d</p>" % len(tc["version_conflicts"]))
    rows.append("<p>敏感信息待审：%d</p>" % len(tc["sensitive_items"]))

    rows.append("<h3>敏感信息待审核（前 20 条）</h3>")
    rows.append("<table border='1' cellpadding='6'>")
    rows.append("<tr><th>文件</th><th>类型</th><th>匹配</th><th>上下文</th></tr>")
    for it in tc["sensitive_items"][:20]:
        rows.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            it["file"], it["type"], it["match"], it["context"].replace("<", "&lt;")))
    rows.append("</table>")

    rows.append("<h3>分类标签统计</h3><ul>")
    from collections import Counter
    c = Counter(report["classification_labels"].values())
    for label, n in c.most_common():
        rows.append("<li>%s: %d</li>" % (label, n))
    rows.append("</ul>")

    return "<html><meta charset='utf-8'><body>%s</body></html>" % "".join(rows)
