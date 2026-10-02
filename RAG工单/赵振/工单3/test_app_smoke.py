"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化。"""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).parent


def main():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("PDF 文档问答" in item.value for item in app.title)
    assert any("选择当前 PDF" in item.label for item in app.selectbox)
    assert any("提问" in item.label for item in app.button)
    result = {"pass": True, "title": "PDF 文档问答", "upload_control": True, "question_button": True}
    (ROOT / "app_smoke_test_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
