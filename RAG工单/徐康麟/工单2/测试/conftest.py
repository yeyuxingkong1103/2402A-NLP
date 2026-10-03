"""T3 离线测试套件的公共夹具与口径工具。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：测试 / 离线（七件套公共层）

本文件提供：
1. ``sys.path`` 注入：让 ``app.core.*`` 可被导入（无需设置环境变量）；
2. **证据块映射**：一律在运行时由「证据原文严格匹配 chunks.jsonl」求得，
   **绝不硬编码 chunk_id**——因为 chunk_id 跨索引不稳定（新索引 ``c000265`` = p84，
   旧索引 ``c000265`` = p52，见 环境事实.md §4.1.4）；
3. 两种归一化（A 仅去空白 / B 去空白+标点）与两种命中口径（宽松前 80 字 / 严格整段包含），
   供 ``test_retriever_v2.py`` 输出校准矩阵；
4. 判分口径常量断言所需的真实 ``Evaluator``。

口径纪律（来源：设计/验收标准.md §2、环境事实.md §4.2）：
- ``FUZZY_THRESHOLD`` 必须保持 ``0.62``，测试只断言、不修改；
- 引用正确性**不得**以 ``golden_qa.jsonl.evidence_pages`` 为唯一真值（存在错标）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# 路径注入
# --------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent          # 工单2/测试
PROJECT_ROOT = TESTS_DIR.parent                      # 工单2
SOURCE_ROOT = PROJECT_ROOT / "研发"                   # 工单2/研发
TEST_DATA = TESTS_DIR / "测试数据"
EVAL_RESULTS = PROJECT_ROOT / "优化" / "评估结果"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:  # 控制台中文乱码（环境事实 6.7）
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

# --------------------------------------------------------------------------
# 归一化与口径常量
# --------------------------------------------------------------------------
#: 与 ``app.core.evaluator.PUNCT_TO_STRIP`` 同款（判分同源的标点集合）
PUNCT_TO_STRIP = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"
_WS_RE = re.compile(r"\s+")


def norm_a(text: str) -> str:
    """归一化 A：仅去空白（保留标点）。"""
    return _WS_RE.sub("", text or "")


def norm_b(text: str) -> str:
    """归一化 B：去空白 + 中英文标点（与判分口径同源，推荐口径）。

    为什么必须同时报 A 与 B：Q795/Q957 的严格命中差异**只来自一个标点**
    （证据末尾「。」vs 块内「，」），只报 B 会掩盖该敏感性。
    """
    return "".join(ch for ch in norm_a(text) if ch not in PUNCT_TO_STRIP)


def contains(haystack: str, needle: str) -> bool:
    """空串不算命中（避免把无证据的题误判为命中）。"""
    return bool(needle) and needle in (haystack or "")


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def settings():
    """全局配置（真实读取，不使用测试替身）。"""
    from app.core.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def golden() -> list:
    """10 个工单问题（标准答案 + 证据原文）。"""
    from app.models.schemas import GoldenQA

    path = TEST_DATA / "golden_qa.jsonl"
    assert path.exists(), f"缺少测试数据: {path}"
    items = [GoldenQA(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(items) == 10, f"golden_qa.jsonl 应为 10 条，实际 {len(items)}"
    return items


@pytest.fixture(scope="session")
def chunks():
    """1669 个索引块（真实从 chunks.jsonl 加载）。"""
    from app.core.chunker import Chunker

    path = SOURCE_ROOT / "data" / "processed" / "chunks.jsonl"
    assert path.exists(), f"缺少分块文件: {path}"
    items = Chunker.load(path)
    assert items, "分块文件为空"
    return items


@pytest.fixture(scope="session")
def parsed_document():
    """解析结果（招股说明书1.parsed.json）。"""
    from app.core.pdf_parser import PDFParser

    path = SOURCE_ROOT / "data" / "processed" / "招股说明书1.parsed.json"
    assert path.exists(), f"缺少解析产物: {path}"
    return PDFParser.load_parsed(path)


@pytest.fixture(scope="session")
def evidence_map(chunks, golden) -> dict[int, dict]:
    """逐题求「证据原文落在哪些块」——运行时求得，不硬编码 chunk_id。

    Returns:
        ``{question_id: {"strict": {chunk_id: page}, "loose": {chunk_id: page}}}``
        其中 strict = 整段 evidence 完整包含；loose = 前 80 字前缀包含（归一化 B）。
    """
    texts = {c.chunk_id: norm_b(c.content) for c in chunks}
    page_of = {c.chunk_id: c.page for c in chunks}
    result: dict[int, dict] = {}
    for item in golden:
        ev = norm_b(item.evidence)
        prefix = ev[:80]
        strict = {cid: page_of[cid] for cid, text in texts.items() if contains(text, ev)}
        loose = {cid: page_of[cid] for cid, text in texts.items() if contains(text, prefix)}
        result[item.id] = {"strict": strict, "loose": loose}
    return result


@pytest.fixture(scope="session")
def retriever(chunks):
    """已装载 1669 块索引的真实检索器（向量 bge-m3 1024 + BM25）。"""
    from app.core.retriever import get_retriever

    r = get_retriever()
    loaded = r.load_index(chunks)
    assert loaded, "索引加载失败（先运行 研发/scripts/build_index.py）"
    assert r.ready, "检索器未就绪"
    return r


@pytest.fixture(scope="session")
def reranker():
    """真实重排器（本机无重排模型权重，mode 为 rule）。"""
    from app.core.reranker import get_reranker

    return get_reranker()


@pytest.fixture(scope="session")
def evaluator():
    """真实判分器（确定性，不依赖 LLM）。"""
    from app.core.evaluator import get_evaluator

    return get_evaluator()


@pytest.fixture(scope="session")
def query_understanding():
    """真实 Query 理解器（use_llm=False，确定性）。"""
    from app.core.query_understanding import QueryUnderstanding

    return QueryUnderstanding(use_llm=False)


@pytest.fixture(scope="session")
def generator_extractive():
    """强制抽取式生成器（不调用 LLM，保证可回归）。"""
    from app.core.generator import Generator

    return Generator(force_extractive=True)


@pytest.fixture(scope="session")
def engine():
    """端到端编排引擎（强制抽取式，确定性可回归）。

    为什么需要它：**引用（``Answer.citations``）由编排层附着**，不是生成层——
    ``app/core/qa_engine.py`` 通过 ``CitationManager.build`` 写入 citations，
    而 ``app/core/generator.py`` 全文不引用 citations（实测）。
    因此任何「回答带引用」的断言必须走本引擎，否则测的是错误的分层。
    """
    from app.core.qa_engine import QAEngine

    eng = QAEngine(force_extractive=True)
    assert eng.load_index(), "引擎加载索引失败（先运行 研发/scripts/build_index.py）"
    assert eng.ready, "引擎未就绪"
    return eng


@pytest.fixture(scope="session")
def evidence_hit_chunks(chunks, evidence_map, golden) -> dict[int, set[str]]:
    """逐题「可判定证据块集合」（运行时求得，不写死块号）。

    - 常规题：取严格命中块（整段 evidence 完整包含，归一化 B）；
    - **Q95**：其 ``evidence`` 含省略号「……」，任何归一化下都**无严格命中**，
      属数据缺陷（环境事实 §5.1 / §4.1.4）。改用运行时替代判据：同时含
      「军队视频指挥领域」与「技术标准」的块；
    - **Q207**：``evidence`` 是**合成引用文本**（含括注），0 块命中，改用运行时
      替代判据：页 ∈ {479, 490} 且含 15,000 万元金额的块。

    调用方统计命中率时须把这两题的替代判据单独标注，不得混入严格口径。
    """
    page_of = {c.chunk_id: c.page for c in chunks}
    texts = {c.chunk_id: norm_b(c.content) for c in chunks}
    hits: dict[int, set[str]] = {}
    for item in golden:
        if item.id == 95:
            hits[item.id] = {cid for cid, t in texts.items()
                             if "军队视频指挥领域" in t and "技术标准" in t}
        elif item.id == 207:
            hits[item.id] = {cid for cid, t in texts.items()
                             if page_of[cid] in (479, 490) and "15000" in t}
        else:
            hits[item.id] = set(evidence_map[item.id]["strict"])
    return hits


_CACHE: dict[str, object] = {}


def _chunks_ref():
    """供 ``evidence_hit_chunks`` 使用的惰性分块引用（避免夹具间循环依赖）。"""
    if "chunks" not in _CACHE:
        from app.core.chunker import Chunker

        _CACHE["chunks"] = Chunker.load(SOURCE_ROOT / "data" / "processed" / "chunks.jsonl")
    return _CACHE["chunks"]


def _page_of_ref() -> dict[str, int]:
    """chunk_id -> page 的惰性映射。"""
    if "page_of" not in _CACHE:
        _CACHE["page_of"] = {c.chunk_id: c.page for c in _chunks_ref()}
    return _CACHE["page_of"]      # type: ignore[return-value]
