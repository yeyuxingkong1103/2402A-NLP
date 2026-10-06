"""pytest 公共固件：路径、配置、标准问答、索引就绪判定与隔离数据库。

设计原则（对应工单第 9 节与验收标准）：

1. **不污染 ``data/``**：所有会写库的用例都在 ``tmp_path`` 下的数据库副本上运行，
   真实索引库只以只读方式访问（URI ``mode=ro``）。
2. **索引未就绪一律跳过**：索引由 ``python scripts/build_index.py`` 生成，
   缺失时统一抛出 ``pytest.skip("索引未就绪：请先运行 python scripts/build_index.py")``，
   绝不让用例因环境缺件而"假失败"。
3. **重资源只加载一次**：装载 3000+ 分块并恢复向量索引需要数秒到十几秒，
   因此引擎、样例解析结果等固件使用 session 作用域。

注意：本文件只用中文书写注释与断言消息（工单硬性要求）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
import time
from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# 让测试在没有设置 PYTHONPATH 的情况下也能 import app.*
#
# 目录结构：<root>/测试/tests/conftest.py，源码在 <root>/研发/app/，因此：
#   parents[0] = tests    parents[1] = 测试
#   parents[2] = 项目根    -> 源码根 = 项目根 / "研发"
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
SOURCE_ROOT = PROJECT_ROOT / "研发"
for candidate in (SOURCE_ROOT, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


# --------------------------------------------------------------------------
# 临时目录：指向 **tests/ 下、按进程号唯一**的目录。
#
# 1. 受限沙箱可能拒绝访问系统临时目录（%TEMP%），导致所有 tmp_path 用例在
#    setup 阶段就报 PermissionError；
# 2. 若所有 pytest 会话共用同一个 basetemp，两个并发运行的会话会互相清空
#    对方的临时目录（实测会导致"用例跑到一半数据库文件消失"）；
# 3. 放在 tests/ 下而不是项目级临时目录，是因为会话级的临时文件不应出现在
#    交付目录里，也不该被其它清理脚本顺手删掉。
# 通过 PYTEST_DEBUG_TEMPROOT 生效；会话结束时自动删除。
# --------------------------------------------------------------------------
_TEMPROOT = _HERE.parent / f".pytest_run_{os.getpid()}"
_TEMPROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TEMPROOT))


# --------------------------------------------------------------------------
# 受限沙箱兼容补丁：``Path.mkdir(mode=0o700)`` 会创建出"连创建者自己都无法访问"
# 的目录（在 DSH 的文件沙箱下复现为 WinError 5 拒绝访问），而 pytest 的
# ``tmp_path`` 插件固定用 ``mode=0o700`` 创建 basetemp，会导致所有使用
# ``tmp_path`` 的用例在 setup 阶段直接报 PermissionError。
# 这里把 0o700 放宽为 0o777（仅影响测试进程内的目录创建，不改变应用行为）。
# --------------------------------------------------------------------------
def _patch_pathlib_mkdir_for_sandbox() -> None:
    """让测试进程内的目录创建不因 0o700 权限位而在沙箱中不可访问。"""
    if getattr(pathlib.Path, "_rag_test_mkdir_patched", False):
        return
    original_mkdir = pathlib.Path.mkdir

    def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False):  # type: ignore[no-untyped-def]
        if mode == 0o700:
            mode = 0o777
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    pathlib.Path.mkdir = mkdir  # type: ignore[method-assign]
    pathlib.Path._rag_test_mkdir_patched = True  # type: ignore[attr-defined]

    # 标准库 tempfile 还会用 os.mkdir(x, 0o700) 与 os.chmod(x, 0o700) 创建/复位临时目录，
    # 在沙箱下同样会变成"不可访问"，导致 TemporaryDirectory 清理时报 PermissionError。
    # 这里一并对 0o700 做放宽处理（只影响测试进程）。
    import os

    original_os_mkdir = os.mkdir
    original_os_chmod = os.chmod

    def os_mkdir(path, mode=0o777, *args, **kwargs):  # type: ignore[no-untyped-def]
        if mode == 0o700:
            mode = 0o777
        return original_os_mkdir(path, mode, *args, **kwargs)

    def os_chmod(path, mode, *args, **kwargs):  # type: ignore[no-untyped-def]
        if mode == 0o700:
            mode = 0o777
        return original_os_chmod(path, mode, *args, **kwargs)

    os.mkdir = os_mkdir  # type: ignore[assignment]
    os.chmod = os_chmod  # type: ignore[assignment]


def _patch_sqlite_connect_retry() -> None:
    """给 ``sqlite3.connect`` 加少量重试，抵御"索引库正被重建"造成的瞬时打不开。

    背景：``scripts/build_index.py`` 重建索引时会重写 ``data/index/rag.sqlite3``，
    同时在受限沙箱下偶发出现 ``unable to open database file``（SQLITE_CANTOPEN）。
    测试进程内对连接做最多 3 次重试：数据库真损坏时依旧会失败，不会掩盖真实缺陷。
    """
    if getattr(sqlite3, "_rag_test_connect_patched", False):
        return
    original_connect = sqlite3.connect

    def connect_with_retry(database, *args, **kwargs):  # type: ignore[no-untyped-def]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return original_connect(database, *args, **kwargs)
            except sqlite3.OperationalError as exc:  # pragma: no cover - 环境抖动时才触发
                last_error = exc
                if "unable to open database file" not in str(exc):
                    raise
                time.sleep(0.3 * (attempt + 1))
        raise last_error  # type: ignore[misc]

    sqlite3.connect = connect_with_retry  # type: ignore[assignment]
    sqlite3._rag_test_connect_patched = True  # type: ignore[attr-defined]


_patch_pathlib_mkdir_for_sandbox()
_patch_sqlite_connect_retry()


@pytest.fixture(scope="session", autouse=True)
def redirect_tempdir(tmp_path_factory):
    """把标准库的临时目录重定向到项目内的 basetemp。

    原因：受限沙箱可能拒绝对系统临时目录（``%TEMP%``）的访问，
    导致 streamlit / tempfile 在创建临时文件或退出清理时抛 PermissionError。
    该固件让整个测试会话的临时文件都落在项目内，最后随 basetemp 一起清理。
    """
    import os
    import tempfile

    root = tmp_path_factory.mktemp("tmp")
    previous = tempfile.tempdir
    tempfile.tempdir = str(root)
    os.environ["TMP"] = os.environ["TEMP"] = str(root)
    try:
        yield root
    finally:
        tempfile.tempdir = previous


@pytest.fixture(scope="session", autouse=True)
def cleanup_temproot():
    """会话结束后删除本次进程专用临时根目录（尽力而为，失败不影响结果）。"""
    import shutil

    try:
        yield
    finally:
        shutil.rmtree(_TEMPROOT, ignore_errors=True)


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    """会话收尾时再删一次自己的临时根目录。

    覆盖 ``--collect-only`` 等不执行 session 固件的场景；
    只删本进程号对应的目录，避免误删并发运行的其它 pytest 会话。
    """
    import shutil

    shutil.rmtree(_TEMPROOT, ignore_errors=True)

from app.core.config import get_settings  # noqa: E402
from app.models.schemas import GoldenQA  # noqa: E402
from app.storage.sqlite_manager import SQLiteManager  # noqa: E402

# 索引缺件时统一的跳过原因（工单指定文案）
INDEX_NOT_READY_MSG = "索引未就绪：请先运行 python scripts/build_index.py"

# 工单第 5.10 节要求的表清单
EXPECTED_TABLES = [
    "chunks",
    "conversations",
    "documents",
    "eval_results",
    "feedback",
    "golden_qa",
    "logs",
    "messages",
]


# ==========================================================================
# 工具函数
# ==========================================================================
def copy_sqlite(src: Path, dst: Path, attempts: int = 3) -> None:
    """用 SQLite 备份 API 复制数据库（自动包含 WAL 中尚未落盘的数据）。

    索引可能正在被 ``scripts/build_index.py`` 重建（文件被替换/写入中），
    因此复制后立即做一次可读性校验；失败则重试，避免偶发的"复制到一半"
    让用例变成环境性失败。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            source = sqlite3.connect(str(src), timeout=30.0)
            try:
                target = sqlite3.connect(str(dst), timeout=30.0)
                try:
                    source.backup(target)
                finally:
                    target.close()
            finally:
                source.close()

            # 校验副本可用（能打开、能查询）
            check = sqlite3.connect(str(dst), timeout=30.0)
            try:
                check.execute("SELECT COUNT(*) FROM chunks").fetchone()
            finally:
                check.close()
            return
        except sqlite3.Error as exc:  # pragma: no cover - 仅在索引被占用/写入中时触发
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"复制索引数据库失败：{src} -> {dst}（{last_error}）")


def _index_state(settings) -> tuple[bool, str]:
    """只读方式判断索引是否就绪，返回 ``(是否就绪, 说明)``。

    使用 URI ``mode=ro`` 打开数据库，避免测试反过来修改 ``data/index``。
    """
    db_path: Path = settings.paths.sqlite_path
    if not db_path.exists():
        return False, f"索引数据库不存在：{db_path}"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15.0)
        try:
            indexed = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status='indexed' AND chunk_count > 0"
            ).fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            doc_id = conn.execute(
                "SELECT doc_id FROM documents WHERE status='indexed' ORDER BY is_default DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return False, f"读取索引数据库失败：{type(exc).__name__}: {exc}"
    if not indexed or not chunks:
        return False, "索引库中还没有已建立索引的文档分块"
    return True, f"doc_id={doc_id[0] if doc_id else '?'}，分块数={chunks}"


# ==========================================================================
# 基础固件
# ==========================================================================
@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录（所有相对路径的基准）。"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def settings():
    """全局配置对象（与运行时使用同一份缓存实例）。"""
    return get_settings()


@pytest.fixture(scope="session")
def index_state(settings) -> tuple[bool, str]:
    """索引就绪状态：``(是否就绪, 说明)``。"""
    return _index_state(settings)


@pytest.fixture(scope="session")
def requires_index(index_state) -> str:
    """需要索引的用例声明此固件；索引缺失时直接跳过并给出建索引命令。"""
    ready, detail = index_state
    if not ready:
        pytest.skip(INDEX_NOT_READY_MSG)
    return detail


@pytest.fixture(scope="session")
def golden_qa(settings) -> list[GoldenQA]:
    """工单固定 10 题的标准答案（``data/eval/golden_qa.jsonl``）。"""
    path: Path = settings.paths.data_eval / "golden_qa.jsonl"
    if not path.exists():
        pytest.skip(f"标准答案文件不存在：{path}")
    items = [
        GoldenQA(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        pytest.skip(f"标准答案文件为空：{path}")
    return items


@pytest.fixture(scope="session")
def default_pdf(settings) -> Path:
    """语料 PDF 路径（缺失即跳过，避免用例因缺数据失败）。"""
    path: Path = settings.paths.default_pdf
    if not path.exists():
        pytest.skip(f"语料 PDF 不存在：{path}")
    return path


@pytest.fixture(scope="session")
def pdf_page_count(default_pdf) -> int:
    """用 PyMuPDF 直接读取 PDF 物理页数（不解析正文，毫秒级）。"""
    pymupdf = pytest.importorskip("pymupdf", reason="未安装 PyMuPDF，无法校验页码范围")
    with pymupdf.open(str(default_pdf)) as document:
        return int(document.page_count)


# ==========================================================================
# 隔离数据库
# ==========================================================================
@pytest.fixture
def isolated_store(settings, tmp_path) -> SQLiteManager:
    """真实索引库的**副本**（位于 ``tmp_path``），可安全写入。"""
    target = tmp_path / "rag_isolated.sqlite3"
    if settings.paths.sqlite_path.exists():
        copy_sqlite(settings.paths.sqlite_path, target)
    else:  # 索引尚未生成时也给出一个可用的空库
        SQLiteManager(target)
    return SQLiteManager(target)


@pytest.fixture
def empty_store(tmp_path) -> SQLiteManager:
    """全新的空数据库（用于验证"无索引 -> 不清楚"的兜底路径）。"""
    return SQLiteManager(tmp_path / "rag_empty.sqlite3")


@pytest.fixture
def real_message_count(settings) -> int:
    """真实索引库当前的 messages 行数（用于断言测试没有污染真实库）。"""

    def _count() -> int:
        if not settings.paths.sqlite_path.exists():
            return 0
        conn = sqlite3.connect(f"file:{settings.paths.sqlite_path}?mode=ro", uri=True, timeout=15.0)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        except sqlite3.Error:
            return -1
        finally:
            conn.close()

    return _count()


# ==========================================================================
# 引擎固件（需要索引）
# ==========================================================================
@pytest.fixture(scope="session")
def engine(settings, index_state, tmp_path_factory):
    """全链路问答引擎：抽取式模式 + **索引库副本**，不依赖任何 LLM 服务。"""
    ready, detail = index_state
    if not ready:
        pytest.skip(INDEX_NOT_READY_MSG)

    from app.core.qa_engine import QAEngine

    workdir = tmp_path_factory.mktemp("engine_index")
    db_path = workdir / "rag.sqlite3"
    copy_sqlite(settings.paths.sqlite_path, db_path)
    store = SQLiteManager(db_path)
    qa = QAEngine(store=store, force_extractive=True)
    if not qa.stats().get("index_ready"):
        pytest.skip(INDEX_NOT_READY_MSG)
    return qa


@pytest.fixture
def empty_engine(empty_store):
    """空索引引擎：用于验证"检索不到内容 -> 回复不清楚"的兜底契约。"""
    from app.core.qa_engine import QAEngine

    return QAEngine(store=empty_store, force_extractive=True, auto_load_index=False)


# ==========================================================================
# 样例解析结果（供分块/表格用例复用，避免重复解析）
# ==========================================================================
@pytest.fixture(scope="session")
def sample_document(default_pdf):
    """前 30 页（含表格）的解析结果：兼顾覆盖度与运行速度。"""
    from app.core.pdf_parser import PDFParser

    return PDFParser().parse(default_pdf, doc_id="doc_test_sample", max_pages=30, extract_tables=True)
