# -*- coding: utf-8 -*-
"""演示：document_ingestion_workflow 按质检标签自动路由。

重点演示：一份 Scan_PDF 文档如何自动进入 OCR 解析流程（验收演示项）。
输出文本日志 + 渲染 HTML 供截图。
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "研发" / "src"))

from document_quality_assessment.workflow import DocumentIngestionWorkflow

rep = json.loads((BASE / "研发" / "src" / "results" / "quality_report.json").read_text(encoding="utf-8"))
wf = DocumentIngestionWorkflow(rep)

# 选取 4 类代表性文档
records = rep["records"]


def pick(tag):
    for r in records:
        if tag in r["tags"]:
            return r["path"]
    return None


demo_paths = [
    pick("Scan_PDF"),
    pick("Text_PDF"),
    pick("Sensitive_Info"),
    pick("Near_Duplicate"),
]
demo_paths = [p for p in demo_paths if p]

lines = ["$ 执行 document_ingestion_workflow（质检驱动的自动路由演示）", ""]
for p in demo_paths:
    out = wf.run(p)
    lines.append(f"文件: {Path(p).name}")
    lines.append(f"  标签: {', '.join(out['tags'])}")
    for t in out["trace"]:
        lines.append(f"  [{t['node']}] {t['action']}")
    lines.append(f"  => 最终状态: {out['result'].get('status')}")
    lines.append("")

text = "\n".join(lines)
(BASE / "测试" / "workflow_demo.log").write_text(text, encoding="utf-8")
print(text)

# 渲染成终端风格 HTML
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>工作流路由演示</title>
<style>
body{{background:#0f1419;padding:30px;font-family:Consolas,monospace}}
pre{{color:#7ec699;font-size:13px;line-height:1.7;white-space:pre-wrap}}
h1{{color:#4a90d9;font-family:Microsoft YaHei;font-size:18px}}
</style></head><body>
<h1>document_ingestion_workflow 智能体自动路由演示</h1>
<pre>{text.replace('<','&lt;')}</pre></body></html>"""
(BASE / "测试" / "workflow_demo.html").write_text(html, encoding="utf-8")
print("\nHTML 已生成")
