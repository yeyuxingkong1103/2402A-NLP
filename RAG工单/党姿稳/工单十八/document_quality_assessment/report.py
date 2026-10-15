# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：把质检结果组装成结构化报告字典，并导出 JSON 与可直接阅读的 HTML 简报；
#           HTML 重点呈现“待确认/待审核”列表，供操作员快速定位问题文档。

import os                                                          # 路径
import json                                                        # JSON 输出
import html                                                        # HTML 转义
import time                                                        # 生成时间

TITLE = "文档质量评估报告"                                          # 报告标题


def build(config_note, cfg, target, format_stats, pdf_triage, length,
          dedup, classify, records, errors, elapsed):
    """组装完整报告字典（这是 Skill 的标准输出结构）。"""
    sensitive_items = [{"path": r["path"], "name": r.get("name"),   # 敏感信息待审核列表
                        "type": h["type"], "value": h["value"],
                        "char_offset": h.get("char_offset"),
                        "context": h.get("context")}
                       for r in records                            # 逐文档
                       for h in (r.get("sensitive") or [])]
    return {"meta": {"title": TITLE,                               # 标题
                     "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "target": target,                             # 评估目标
                     "config_source": cfg.source,                  # 配置来源
                     "config_note": config_note,                   # 配置加载提示
                     "config_snapshot": cfg.as_dict(),             # 本次运行的口径
                     "elapsed_seconds": round(elapsed, 2)},        # 总耗时
            "format_stats": format_stats,                          # 一、格式分布
            "pdf_triage": pdf_triage,                              # 二、PDF 页面类型分流
            "length_stats": length,                                # 三、长度分布
            "dedup": dedup,                                        # 四、重复检测
            "classification": classify,                            # 五、分类标签体系
            "errors": errors,                                      # 解析失败清单
            "to_review": {"pdf_type_to_confirm": pdf_triage.get("to_confirm", []),
                          "version_conflicts": dedup.get("near_pairs", []),
                          "sensitive_items": sensitive_items}}     # 三类“待确认/待审核”


def summary_lines(report):
    """生成控制台可读的摘要行。"""
    fs = report["format_stats"]                                    # 格式统计
    tri = report["pdf_triage"]                                     # PDF 分诊
    lg = report["length_stats"]["overall"]                         # 长度
    du = report["dedup"]                                           # 重复
    tr = report["to_review"]                                       # 待确认
    return ["总文件数：%d（%.1f MB）" % (fs["total"], fs["total_size_mb"]),
            "PDF 分诊：文字型 %d / 扫描型 %d / 混合型 %d"
            % (tri["text"], tri["scan"], tri["mixed"]),
            "长度分位数：" + "，".join("%s=%.0f" % (k, v)
                                       for k, v in sorted(lg["percentiles"].items())),
            "精确重复组：%d 组；版本冲突候选：%d 对"
            % (len(du["exact_groups"]), len(du["near_pairs"])),
            "待确认 PDF 类型：%d；待审核敏感信息：%d"
            % (len(tr["pdf_type_to_confirm"]), len(tr["sensitive_items"])),
            "解析失败：%d" % len(report["errors"])]


def write_json(report, path, indent=2):
    """把报告写成 JSON 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)              # 确保目录
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=indent)   # 中文不转义
    return path


def _e(value):
    """HTML 转义，None 转空串。"""
    return html.escape("" if value is None else str(value), quote=True)


def _table(headers, rows):
    """生成 HTML 表格。"""
    out = ["<table><thead><tr>"
           + "".join("<th>%s</th>" % _e(h) for h in headers) + "</tr></thead><tbody>"]
    for row in rows:                                               # 逐行
        out.append("<tr>" + "".join("<td>%s</td>" % _e(c) for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def to_html(report, path):
    """导出 HTML 简报：概览 + 五类统计 + 三类待确认/待审核列表。"""
    m, fs = report["meta"], report["format_stats"]                 # 元信息与格式统计
    tri, lg = report["pdf_triage"], report["length_stats"]["overall"]
    du, cls = report["dedup"], report["classification"]            # 重复与分类
    tr = report["to_review"]                                       # 待确认列表
    p = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         "<title>%s</title><style>" % _e(m["title"]),
         "body{font-family:'Microsoft YaHei',sans-serif;margin:24px;color:#222}",
         "h1{font-size:22px}h2{font-size:17px;margin-top:26px;border-left:4px solid #2b6cb0;padding-left:8px}",
         "table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}",
         "th,td{border:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:top}",
         "th{background:#f4f6f8}.warn{color:#b42318;font-weight:600}",
         ".ctx{font-family:Consolas,monospace;font-size:12px;color:#444}",
         "</style></head><body>",
         "<h1>%s</h1>" % _e(m["title"]),
         "<p>目标：%s<br>生成时间：%s　耗时：%s 秒　配置文件：%s</p>"
         % (_e(m["target"]), _e(m["generated_at"]), _e(m["elapsed_seconds"]),
            _e(os.path.basename(m["config_source"] or "")))]
    if m.get("config_note"):                                       # 配置提示
        p.append("<p class='warn'>%s</p>" % _e(m["config_note"]))

    p.append("<h2>一、格式分布</h2>")
    p.append(_table(["后缀", "文件数", "占比", "体积(MB)"],
                    [(r["ext"], r["count"], "%.2f%%" % (r["ratio"] * 100), r["size_mb"])
                     for r in fs["by_ext"]]))

    p.append("<h2>二、PDF 页面类型分流</h2>")
    p.append("<p>文字型 %d 个，扫描型 %d 个，混合型 %d 个，其中 <span class='warn'>%d 个待确认</span>。</p>"
             % (tri["text"], tri["scan"], tri["mixed"], len(tr["pdf_type_to_confirm"])))
    p.append(_table(["文件", "类型", "页数", "扫描页", "扫描占比", "图像覆盖率", "待确认原因"],
                    [(r["name"], r["kind"], r["page_count"], r["scan_pages"],
                      "%.2f" % r["scan_ratio"], "%.2f" % r["image_page_ratio"], r["confirm_reason"])
                     for r in tri["to_confirm"][:100]]))

    p.append("<h2>三、文档长度分布</h2>")
    p.append("<p>P25=%.0f　P50=%.0f　P75=%.0f　P90=%.0f　P99=%.0f（单位：字符）</p>"
             % (lg["percentiles"]["P25"], lg["percentiles"]["P50"], lg["percentiles"]["P75"],
                lg["percentiles"]["P90"], lg["percentiles"]["P99"]))
    p.append(_table(["长度区间", "文档数", "占比"],
                    [(r["range"], r["count"], "%.2f%%" % (r["ratio"] * 100))
                     for r in lg["histogram"]]))

    p.append("<h2>四、重复检测</h2>")
    p.append("<p>MD5 精确重复 %d 组；SimHash 版本冲突候选 %d 对（需人工确认）。</p>"
             % (len(du["exact_groups"]), len(du["near_pairs"])))
    p.append(_table(["MD5", "文件数", "重复文件"],
                    [(g["md5"][:12], g["count"], "<br>".join(_e(f["path"]) for f in g["files"]))
                     for g in du["exact_groups"][:50]]))
    p.append(_table(["相似度", "汉明距离", "文件 A", "文件 B", "片段 A", "片段 B"],
                    [(p2["similarity"], p2["hamming_distance"], p2["file_a"]["path"],
                      p2["file_b"]["path"],
                      "<div class='ctx'>%s</div>" % _e(p2["file_a"]["excerpt"][:80]),
                      "<div class='ctx'>%s</div>" % _e(p2["file_b"]["excerpt"][:80]))
                     for p2 in du["near_pairs"][:50]]))

    p.append("<h2>五、分类标签体系</h2>")
    p.append(_table(["主标签", "数量", "占比"],
                    [(r["label"], r["count"], "%.2f%%" % (r["ratio"] * 100))
                     for r in cls["label_distribution"]]))
    p.append("<h3>质量标签</h3>")
    p.append(_table(["质量标签", "数量", "占比"],
                    [(r["tag"], r["count"], "%.2f%%" % (r["ratio"] * 100))
                     for r in cls["tag_distribution"]]))

    p.append("<h2>待审核：敏感信息（%d 条）</h2>" % len(tr["sensitive_items"]))
    p.append(_table(["类型", "命中值", "字符偏移", "上下文", "文件"],
                    [(i["type"], i["value"], i["char_offset"],
                      "<div class='ctx'>%s</div>" % _e(i["context"]), i["path"])
                     for i in tr["sensitive_items"][:200]]))

    if report["errors"]:                                           # 解析失败
        p.append("<h2>解析失败清单（%d）</h2>" % len(report["errors"]))
        p.append(_table(["文件", "错误"], [(e["path"], e["error"]) for e in report["errors"]]))
    p.append("</body></html>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:                  # 写盘
        fh.write("\n".join(p))
    return path
