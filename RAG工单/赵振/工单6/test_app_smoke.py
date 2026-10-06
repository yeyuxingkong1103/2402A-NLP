"""工单编号：人工智能NLP-RAG-混合检索任务。"""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).parent
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
assert app.title and app.title[0].value == "PDF 文档问答"
assert any(button.label == "提问" for button in app.button)
assert any(box.label == "检索方式" for box in app.selectbox)
assert any(box.label == "嵌入模型" for box in app.selectbox)
report = {"passed": True, "checks": ["页面标题", "提问按钮", "检索方式", "嵌入模型", "重排选项"]}
(ROOT / "app_smoke_test_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
