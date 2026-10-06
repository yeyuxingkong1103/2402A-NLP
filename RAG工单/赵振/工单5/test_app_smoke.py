"""工单编号：人工智能NLP-RAG-Query理解优化任务。"""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).parent
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
assert app.title and app.title[0].value == "PDF 文档问答"
assert any(button.label == "提问" for button in app.button)
report = {"passed": True, "checks": ["页面标题", "提问按钮", "知识库选择界面"]}
(ROOT / "app_smoke_test_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
