# -*- coding: utf-8 -*-
"""T8 三级测试的 pytest 公共配置（根 conftest）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

职责：
    1. 把 ``工单3/测试`` 与 ``工单3/研发`` 挂到 ``sys.path``（产品包名 ``app.core.*``）；
    2. 提供跨三级共用的 fixture（golden / 语料发现 / 页文本查询 / 留痕写入）；
    3. 每次运行都在终端摘要里打印「RAGAS 未运行（依赖不可用，本机断网）」，
       避免任何报告漏标（验收纪律）。

注意：本文件**不在导入期导入产品代码**（``app.core``），仅在 fixture 被使用时延迟导入，
这样 engineer 正在改动核心模块时，离线用例的收集不会因为半成品语法错误而整体崩掉。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
for extra in (str(TESTS_DIR), str(REPO_ROOT / "研发")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from common import assertions, golden as golden_mod, negatives as negatives_mod, paths, pdf_probe  # noqa: E402

#: 允许在线用例在服务不可用时「显式跳过」而不是失败（默认关闭：失败更诚实）
ALLOW_ONLINE_SKIP = os.environ.get("RAG_TEST_ALLOW_ONLINE_SKIP", "0") == "1"


def pytest_configure(config: pytest.Config) -> None:
    """注册自定义 mark，便于按验收项挑选用例。"""
    config.addinivalue_line("markers", "offline: 不依赖网络/LLM 的确定性用例")
    config.addinivalue_line("markers", "online: 依赖 Ollama / HTTP 服务的用例")
    config.addinivalue_line("markers", "user: 模拟用户视角的场景用例")
    config.addinivalue_line("markers", "linkage: 断言与设计文档条款的对应关系")
    config.addinivalue_line("markers", "slow: 耗时用例（端到端问答）")


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """终端摘要强制打印 RAGAS 标注（任何测试报告都不得漏标）。"""
    terminalreporter.write_sep("-", "RAGAS 状态")
    terminalreporter.write_line(assertions.RAGAS_BANNER)
    terminalreporter.write_line("替代口径：确定性指标（14 题命中/准确率/引用可回溯/首字/拒答率）")


# ---------------------------------------------------------------------------
# 通用 fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """工单3 根目录。"""
    return REPO_ROOT


@pytest.fixture(scope="session")
def discovered_pdfs() -> list[Path]:
    """自动发现的 PDF 列表（禁硬编码文件名）。"""
    found = paths.discover_pdf_files()
    assert found, f"未发现任何 PDF：{paths.display(paths.RAW_DIR)}（语料自动发现失败，无法继续）"
    return found


@pytest.fixture(scope="session")
def pdf_names(discovered_pdfs: list[Path]) -> list[str]:
    """已发现 PDF 的文件名列表。"""
    return [p.name for p in discovered_pdfs]


@pytest.fixture(scope="session")
def page_lookup() -> Any:
    """``page_text_lookup(file_name, page) -> str``（独立于产品链路的真实页文本）。"""
    return pdf_probe.lookup_factory()


@pytest.fixture(scope="session")
def page_counts(discovered_pdfs: list[Path]) -> dict[str, int]:
    """``{文件名: 物理页数}``。"""
    return {p.name: pdf_probe.page_count(str(p)) for p in discovered_pdfs}


@pytest.fixture(scope="session")
def golden_items() -> list[golden_mod.GoldenItem]:
    """14 题固定输入（含 evidence_verbatim）。"""
    return golden_mod.load_golden()


@pytest.fixture(scope="session")
def golden_index(golden_items: list[golden_mod.GoldenItem]) -> dict[int, golden_mod.GoldenItem]:
    """14 题按 id 索引。"""
    return golden_mod.by_id(golden_items)


@pytest.fixture(scope="session")
def runnable_items(golden_items: list[golden_mod.GoldenItem]) -> list[golden_mod.GoldenItem]:
    """当前语料下可跑的题（PDF2 缺席时自动剔除依赖它的 4 题，并留痕）。"""
    return golden_mod.unlocked_items(golden_items)


@pytest.fixture(scope="session")
def unknown_cases() -> list[negatives_mod.UnknownCase]:
    """不可答负例集。"""
    return negatives_mod.load_unknown_cases()


@pytest.fixture(scope="session")
def leakage_cases() -> list[negatives_mod.LeakageCase]:
    """互污染/要素齐备用例集。"""
    return negatives_mod.load_leakage_cases()


@pytest.fixture(scope="session")
def app_config() -> Any:
    """产品配置对象（``app.core.config.get_config``，session 级单例）。"""
    paths.ensure_dev_on_path()
    from app.core.config import get_config  # noqa: PLC0415

    return get_config()


@pytest.fixture(scope="session")
def cfg(app_config: Any) -> Any:
    """``app_config`` 的别名（兼容研发侧用例惯用的 ``cfg`` 命名，避免 fixture not found）。"""
    return app_config


# ---------------------------------------------------------------------------
# 需要服务的 fixture（**惰性**：只有在线/用户级用例请求时才创建，
# 离线级不触碰网络，也不需要 Ollama）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ollama_status() -> dict[str, Any]:
    """Ollama 可用性探测结果（0.5 s 超时、不重试，只探一次）。"""
    from common import online_gate  # noqa: PLC0415

    return online_gate.probe_ollama()


@pytest.fixture(scope="session")
def engine(app_config: Any) -> Any:
    """端到端问答引擎 ``QAEngine``（含启动预热，§3.22）。"""
    from app.core.qa_engine import build_engine  # noqa: PLC0415

    return build_engine(cfg=app_config, warmup=True)


@pytest.fixture(scope="session")
def retriever(app_config: Any) -> Any:
    """混合检索器（向量 + BM25 + RRF + 加权 + 重排）。"""
    from app.core.retriever import build_retriever  # noqa: PLC0415

    return build_retriever(cfg=app_config)


@pytest.fixture(scope="session")
def generator(app_config: Any) -> Any:
    """答案生成器（含引用与可答性判定）。"""
    from app.core.generator import build_generator  # noqa: PLC0415

    return build_generator(cfg=app_config)


@pytest.fixture
def trace() -> Iterator[Any]:
    """留痕写入器：把断言结果落到 ``测试/留痕/<名称>.json``（唯一允许写的目录）。"""

    class _Trace:
        """收集并落盘一次用例的断言结果。"""

        def __init__(self) -> None:
            self.reports: list[assertions.Report] = []

        def add(self, report: assertions.Report) -> assertions.Report:
            """登记一个报告。"""
            self.reports.append(report)
            return report

        def flush(self, name: str, extra: dict[str, Any] | None = None) -> Path:
            """写盘：``测试/留痕/<name>.json``（含工单编号与 RAGAS 标注）。"""
            paths.ensure_trace_dir()
            payload = {
                "work_order": assertions.WORK_ORDER,
                "name": name,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "ragas": assertions.RAGAS_BANNER,
                "extra": extra or {},
                "reports": [r.to_dict() for r in self.reports],
            }
            target = paths.TRACE_DIR / f"{name}.json"
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return target

    writer = _Trace()
    yield writer
    if writer.reports:
        writer.flush(f"trace_{time.strftime('%Y%m%d_%H%M%S')}")
