"""工单07可复现评估入口：生成十题检索结果和问题分析报告。"""
import json
from pathlib import Path
import server

server.load_index()
report = server.evaluate()
Path("evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"cases": len(report["cases"]), "before_rate": report["before_rate"], "after_rate": report["after_rate"], "after_avg_ms": report["after_avg_ms"]}, ensure_ascii=False))
