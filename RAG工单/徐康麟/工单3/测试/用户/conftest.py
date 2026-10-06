# -*- coding: utf-8 -*-
"""用户级 conftest（模拟用户视角跑端到端场景）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用户级同样需要真实服务（问答/引用/拒答都要走完整链路），因此沿用同一门控与日志初始化；
与在线级的区别是**视角**：这里断言的是「用户会怎么用、会不会被误导」，
例如同页互污染、编造金额、跨语料串题、界面可用性。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import online_gate


@pytest.fixture(scope="session", autouse=True)
def online_gate_ready(ollama_status: dict[str, Any]) -> None:
    """用户级门控：服务不可用即失败（禁止静默跳过）。"""
    online_gate.enforce(ollama_status)


@pytest.fixture(scope="session", autouse=True)
def logging_ready(app_config: Any) -> Any:
    """初始化产品结构化日志。"""
    from app.core.logging_conf import setup_logging  # noqa: PLC0415

    setup_logging(app_config, force=True)
    return app_config


@pytest.fixture(scope="session")
def ui_files(app_config: Any) -> dict[str, Any]:
    """两个界面的源码文本（用于「界面只复用 app/core」的静态断言）。"""
    from common import paths  # noqa: PLC0415

    streamlit_app = paths.DEV_DIR / "app" / "ui" / "streamlit_app.py"
    fallback = paths.DEV_DIR / "app" / "ui" / "serve_fallback.py"
    return {
        "streamlit": streamlit_app,
        "fallback": fallback,
        "streamlit_text": streamlit_app.read_text(encoding="utf-8") if streamlit_app.exists() else "",
        "fallback_text": fallback.read_text(encoding="utf-8") if fallback.exists() else "",
    }
