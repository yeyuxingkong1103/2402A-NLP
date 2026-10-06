"""无浏览器依赖的 Streamlit 页面启动冒烟测试。"""

import os
import json

from streamlit.testing.v1 import AppTest

from rag import ROOT


def main():
    os.chdir(ROOT)
    app = AppTest.from_file("app.py", default_timeout=60).run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.title, "页面标题未显示"
    assert app.text_area, "问题输入框未显示"
    assert any(button.label == "提问" for button in app.button), "提问按钮未显示"
    result = {"page_startup": "passed", "question_input": "passed", "ask_button": "passed", "exceptions": 0}
    (ROOT / "app_smoke_test_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Streamlit AppTest: 页面启动、输入框和提问按钮均通过。")


if __name__ == "__main__":
    main()
