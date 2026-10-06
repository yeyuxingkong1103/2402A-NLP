"""在线测试：接口级验证（"服务已启动 + 模型可用"的路径）。

测试目标（工单 9.2）：
1. 以**引擎直连**方式（等价于服务内部调用路径）发送工单 10 个问题，
   检查答案非空、引用完整、首字时间 < 3000ms；
2. 本机**没有 LLM 服务**（vLLM/SGLang 未部署），因此验证系统能自动降级为抽取式回答，
   且降级后仍然给出带引用的答案；
3. 通过 HTTP 访问 Streamlit 页面属于**可选分支**：只有服务真的在运行时才执行，
   否则用 ``pytest.skip`` 明确说明原因，避免把"没启动服务"伪装成"测试失败"。

索引缺失时整个模块跳过：
``索引未就绪：请先运行 python scripts/build_index.py``。

运行方式::

    $env:PYTHONPATH="E:\\gao6gongdan\\工单1"
    cd E:\\gao6gongdan\\工单1
    & ".gao6gongdan-src\\python.exe" -m pytest tests/online -v
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# 目录结构：<root>/测试/tests/<子目录>/xxx.py
#   parents[0]=子目录 parents[1]=tests parents[2]=测试 parents[3]=项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "研发"
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden_qa.jsonl"

# 首字返回时间预算（工单验收标准 5）
FIRST_TOKEN_BUDGET_MS = 3000.0
# Streamlit 默认端口，可用环境变量覆盖
UI_HOST = os.environ.get("RAG_UI_HOST", "127.0.0.1")
UI_PORT = int(os.environ.get("RAG_UI_PORT", "8501"))


def _load_golden_items():
    """读取标准答案文件（缺失时返回空列表，由固件负责跳过）。"""
    from app.models.schemas import GoldenQA

    if not GOLDEN_PATH.exists():
        return []
    return [
        GoldenQA(**json.loads(line))
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pytest_generate_tests(metafunc) -> None:
    """把工单 10 题参数化到在线用例上。"""
    if "golden_item" not in metafunc.fixturenames:
        return
    metafunc.parametrize(
        "golden_item",
        [pytest.param(item, id=f"q{item.id}") for item in _load_golden_items()],
    )


def _port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """探测端口是否可连接（用于判断服务是否已启动）。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ==========================================================================
# 1. 引擎直连：10 个问题
# ==========================================================================
def test_engine_direct_returns_answer_with_citation(engine, golden_item, requires_index) -> None:
    """直连模式下每题都必须给出非空答案、有效引用，且首字耗时小于 3 秒。"""
    answer = engine.ask(golden_item.question, save=False)

    assert answer.answer.strip(), f"题号 {golden_item.id} 返回空答案"
    assert answer.is_unknown is False, f"题号 {golden_item.id} 被判为不清楚：{answer.unknown_reason!r}"
    assert answer.citations, f"题号 {golden_item.id} 的答案没有引用来源"
    assert all(1 <= citation.page <= 548 for citation in answer.citations), (
        f"题号 {golden_item.id} 存在越界引用：{[c.page for c in answer.citations]}"
    )
    assert answer.first_token_ms < FIRST_TOKEN_BUDGET_MS, (
        f"题号 {golden_item.id} 首字耗时 {answer.first_token_ms:.1f}ms 超过 {FIRST_TOKEN_BUDGET_MS:.0f}ms 预算"
    )


def test_engine_direct_latency_summary(engine, golden_qa, requires_index) -> None:
    """10 题的延迟统计：最大首字耗时必须满足 3 秒预算。"""
    timings = []
    for item in golden_qa:
        answer = engine.ask(item.question, save=False)
        timings.append((item.id, answer.first_token_ms, answer.total_ms))

    worst_id, worst_first, _ = max(timings, key=lambda row: row[1])
    average_first = sum(row[1] for row in timings) / len(timings)
    assert worst_first < FIRST_TOKEN_BUDGET_MS, (
        f"最差首字延迟出现在题号 {worst_id}：{worst_first:.1f}ms（平均 {average_first:.1f}ms），"
        f"超过 {FIRST_TOKEN_BUDGET_MS:.0f}ms 预算"
    )


# ==========================================================================
# 2. 无 LLM 服务时的降级路径
# ==========================================================================
def test_llm_service_is_absent_and_system_degrades(engine, requires_index) -> None:
    """本机没有 LLM 服务时必须自动降级为抽取式回答，且仍然给答案。"""
    available = engine.generator.check_llm_available(timeout=1.0)
    assert available is False, (
        f"检测到 LLM 服务可用（{engine.settings.llm.base_url}）；"
        "本用例面向『无 LLM 服务』的离线环境，请改跑带 LLM 的验收流程"
    )

    answer = engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", save=False)
    assert answer.answer.strip(), "LLM 不可用时仍必须给出答案（抽取式降级）"
    assert answer.mode in {"extractive", "fallback"}, f"降级后的生成模式异常：{answer.mode}"


def test_engine_health_exposes_llm_config(engine, requires_index) -> None:
    """健康信息必须暴露 LLM 地址与模型名，便于部署排障。"""
    stats = engine.stats()
    assert stats["llm_base_url"], "健康信息缺少 LLM 地址"
    assert stats["llm_model"], "健康信息缺少 LLM 模型名"
    assert stats["vector_backend"], "健康信息缺少向量库后端"
    assert stats["embedder"], "健康信息缺少嵌入器名称"


# ==========================================================================
# 3. 网页界面：无头渲染（真实执行）+ 可选 HTTP 分支
# ==========================================================================
def test_streamlit_app_renders_without_exception() -> None:
    """用 Streamlit 官方 AppTest 无头执行界面脚本：必须能渲染且不抛异常。

    这是"网页界面可用"的自动化等价验证：真正执行 ``app/ui/streamlit_app.py``，
    检查页面组件（会话输入框、按钮、引用说明）是否渲染出来。
    """
    app_test_module = pytest.importorskip(
        "streamlit.testing.v1", reason="未安装 streamlit，跳过网页界面的无头渲染验证"
    )
    ui_path = SOURCE_ROOT / "app" / "ui" / "streamlit_app.py"
    assert ui_path.exists(), f"界面文件缺失：{ui_path}"

    app = app_test_module.AppTest.from_file(str(ui_path), default_timeout=180)
    app.run()

    assert not app.exception, (
        "界面脚本渲染时抛出异常：" + "；".join(str(item.value)[:200] for item in app.exception)
    )
    assert len(app.title) >= 1, "界面没有渲染出标题"
    assert any("RAG" in str(item.value) for item in app.title), "标题中未包含 RAG 字样"

    has_input = len(app.chat_input) > 0 or len(app.text_input) > 0
    assert has_input, "界面没有渲染出任何提问输入框（chat_input / text_input）"
    assert len(app.button) > 0, "界面没有渲染出任何按钮（发送/清空/反馈等）"

    page_text = " ".join(str(item.value) for item in app.markdown)
    assert "引用" in page_text, "界面没有说明引用来源的使用方式"
    assert "不清楚" in page_text, "界面没有说明『不清楚』兜底策略"


def test_streamlit_http_optional() -> None:
    """可选分支：网页服务真的在运行时，才通过 HTTP 做一次连通性检查。

    默认不启动网页服务（``streamlit run`` 属于长驻进程，会让自动化测试显著变慢），
    因此本用例默认跳过——界面本身已由上面的 AppTest 用例真实执行验证。
    需要执行 HTTP 分支时请先启动：

    ``streamlit run app/ui/streamlit_app.py --server.port 8501``
    """
    if not _port_is_open(UI_HOST, UI_PORT):
        pytest.skip(
            f"未启动网页服务（{UI_HOST}:{UI_PORT}）：为避免自动化测试过慢，"
            "HTTP 分支默认不执行（界面渲染已由 test_streamlit_app_renders_without_exception 覆盖）；"
            "如需验证请先运行 `streamlit run app/ui/streamlit_app.py --server.port 8501`"
        )

    url = f"http://{UI_HOST}:{UI_PORT}/"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - 仅在服务异常时触发
        pytest.fail(f"Streamlit 端口已开放但 HTTP 访问失败：{type(exc).__name__}: {exc}")

    assert status == 200, f"网页应返回 200，实际 {status}"
    assert "streamlit" in body.lower(), "返回内容不像 Streamlit 页面，请检查页面配置"


def test_ui_module_is_importable() -> None:
    """界面模块必须存在且已接入问答引擎（避免部署时才发现入口缺失）。"""
    path = SOURCE_ROOT / "app" / "ui" / "streamlit_app.py"
    assert path.exists(), f"界面文件缺失：{path}"
    source = path.read_text(encoding="utf-8")
    assert "QAEngine" in source or "get_qa_engine" in source, "界面未接入问答引擎"
    assert "引用" in source, "界面未展示引用来源"
