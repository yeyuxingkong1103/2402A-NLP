# -*- coding: utf-8 -*-
"""应用自检脚本（工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化）

用 Streamlit 官方 AppTest 以无头方式运行 app.py，输出：
    - 是否抛出异常（及异常内容）
    - 标题 / 按钮 / 下拉框数量
用于排查「页面空白」类问题，并验证界面可正常渲染。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest


def main() -> None:
    t0 = time.time()
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=300)
    at.run()
    print("run() 用时: %.2fs" % (time.time() - t0))
    print("异常数量:", len(at.exception))
    for e in at.exception:
        print("  EXC:", e.value)
    print("标题:", [t.value for t in at.title])
    print("按钮数量:", len(at.button))
    print("下拉框数量:", len(at.selectbox))
    print("文本域数量:", len(at.text_area))
    print("指标数量:", len(at.metric))


if __name__ == "__main__":
    main()