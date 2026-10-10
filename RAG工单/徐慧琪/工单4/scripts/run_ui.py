# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""启动 Streamlit 界面：python scripts/run_ui.py

界面默认为 headless（不开浏览器）监听 8501；中文/英文可在侧栏切换。
首次提问会触发模型加载，建议先跑 ``scripts/build_index.py`` 建库。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    app = ROOT / "src" / "rag04" / "ui" / "app.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app),
         "--server.port", "8501", "--server.headless", "true"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
