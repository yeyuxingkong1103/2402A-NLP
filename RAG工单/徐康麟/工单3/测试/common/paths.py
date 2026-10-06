# -*- coding: utf-8 -*-
"""T8 测试路径解析（唯一实现）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

纪律：
    * 语料一律从 ``研发/data/raw/`` **自动发现**（``*.pdf``，大小写扩展名去重），
      严禁在测试里硬编码 PDF 文件名（``设计/需求分析.md`` §4.1 / FR-1）；
    * 环境变量覆盖优先（``RAG_DATA__RAW_DIR`` / ``RAG_DATA__INDEX_DIR``），
      与 ``研发/app/core/config.py`` 保持一致，保证测试与产品读同一份文件；
    * 所有函数只读，任何测试都不得写 ``研发/`` 与 ``优化/`` 目录。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 目录常量（从本文件位置反推，保证任何工作目录下都成立）
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parents[1]          # 工单3/测试
REPO_ROOT = TESTS_DIR.parent                             # 工单3
DEV_DIR = REPO_ROOT / "研发"
DEPLOY_DIR = REPO_ROOT / "部署"
OPTIM_DIR = REPO_ROOT / "优化"
DESIGN_DIR = REPO_ROOT / "设计"

RAW_DIR = Path(os.environ.get("RAG_DATA__RAW_DIR") or (DEV_DIR / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("RAG_DATA__PROCESSED_DIR") or (DEV_DIR / "data" / "processed"))
INDEX_DIR = Path(os.environ.get("RAG_DATA__INDEX_DIR") or (DEV_DIR / "data" / "index"))
EVAL_DIR = DEV_DIR / "data" / "eval"
LOG_DIR = Path(os.environ.get("RAG_LOG__DIR") or (DEPLOY_DIR / "日志"))

TEST_DATA_DIR = TESTS_DIR / "测试数据"                    # 固定输入与期望输出
TRACE_DIR = TESTS_DIR / "留痕"                            # 测试执行留痕（本目录由 tester 独占写）

GOLDEN_FIXTURE = TEST_DATA_DIR / "golden_qa_14.jsonl"
UNKNOWN_FIXTURE = TEST_DATA_DIR / "unknown_questions.jsonl"
LEAKAGE_FIXTURE = TEST_DATA_DIR / "leakage_cases.jsonl"

# 只读参考目录（工单1/工单2）——严禁写入，连 __pycache__ 都不允许
REFERENCE_DIRS = (Path(r"E:\gao6gongdan\工单1"), Path(r"E:\gao6gongdan\工单2"))

# 唯一可用解释器（环境事实 §1；与 run_py.ps1 的默认值一致）
PYTHON_EXE = Path(os.environ.get("RAG_SCHEDULER_PYTHON")
                  or r"E:\gao6gongdan\工单1\.venv\Scripts\python.exe")


def ensure_dev_on_path() -> list[str]:
    """把 ``研发/`` 加入 ``sys.path``（幂等），返回实际加入的路径字符串列表。

    产品代码包名是 ``app.core.*``（``设计/接口设计.md`` §1.1），因此必须挂 ``研发/`` 而不是 ``研发/app``。
    """
    target = str(DEV_DIR)
    if target not in sys.path:
        sys.path.insert(0, target)
    return [target]


def discover_pdf_files() -> list[Path]:
    """自动发现 ``RAW_DIR`` 下的 PDF（按文件名排序，大小写扩展名去重）。

    返回真实存在的文件路径；**不做任何文件名特判**。若产品侧有
    ``app.core.config.discover_pdfs``，测试与产品应得到同一集合（由用例交叉断言）。
    """
    if not RAW_DIR.is_dir():
        return []
    seen: dict[str, Path] = {}
    for item in sorted(RAW_DIR.iterdir(), key=lambda p: p.name):
        if not item.is_file():
            continue
        if item.name.startswith("~$"):
            continue                                     # Office 临时文件
        if item.suffix.lower() != ".pdf":
            continue
        seen.setdefault(item.name.lower(), item)
    return [seen[key] for key in sorted(seen)]


def resolve_corpus(hint: str) -> Path | None:
    """把「语料提示」解析为真实 PDF 路径（自动发现结果里按文件名匹配）。

    ``hint`` 支持：
        * 完整文件名（如 ``招股说明书2.pdf``）—— 命中即返回；
        * 单个数字（如 ``"2"``）—— 取去扩展名后**以该数字结尾**的文件，
          再退化到「按文件名排序的第 N 个」（N = 该数字）；
        * 其它字符串 —— 视为文件名片段做包含匹配。

    找不到返回 ``None``（由调用方断言失败并给出可诊断信息，不静默跳过）。
    """
    files = discover_pdf_files()
    if not files:
        return None
    text = str(hint or "").strip()
    if not text:
        return None
    for path in files:
        if path.name == text:
            return path
    if text.isdigit():
        index = int(text) - 1
        for path in files:
            if path.stem.endswith(text):
                return path
        if 0 <= index < len(files):
            return files[index]
        return None
    for path in files:
        if text in path.name:
            return path
    return None


def file_names() -> list[str]:
    """自动发现到的 PDF 文件名列表（顺序 = 文件名排序）。"""
    return [p.name for p in discover_pdf_files()]


def index_model_dir(model_slug: str) -> Path:
    """索引目录（``研发/data/index/<嵌入模型 slug>``），与 T4 产物布局一致。"""
    return INDEX_DIR / model_slug


def sqlite_path() -> Path:
    """会话/块持久化数据库路径（``设计/接口设计.md`` §5.3）。"""
    return INDEX_DIR / "rag.sqlite3"


def processed_jsonl(stem: str, kind: str) -> Path:
    """解析产物路径：``kind ∈ {pages, tables, text_blocks}``。"""
    return PROCESSED_DIR / f"{stem}.{kind}.jsonl"


def chunks_jsonl() -> Path:
    """全量分块产物（T3 落盘）。"""
    return PROCESSED_DIR / "chunks.jsonl"


def trace_file(name: str) -> Path:
    """``部署/日志`` 下的留痕文件（app.log / error.log / rag_trace.jsonl …）。"""
    return LOG_DIR / name


def ensure_trace_dir() -> Path:
    """确保 ``测试/留痕`` 存在（唯一允许 tester 创建的目录）。"""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return TRACE_DIR


def display(path: Path | str) -> str:
    """把绝对路径显示成相对工单3 的形式（日志与报告更易读）。"""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)
