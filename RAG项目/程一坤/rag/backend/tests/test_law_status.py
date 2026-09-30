"""法规效力状态词表单一来源：锁定测试（防"改状态名漏改某处"）。

背景：law_versions.status 原先存在**两套互不认识的词表**——
列注释写英文（effective / amended / repealed / superseded），
抽取器产出中文（现行有效 / 已废止 / 已失效），
三个消费点（retrieval/context_builder.resolve_current_status、
db/vector_index_service 写 Milvus 的 is_current、retrieval/keyword_search._resolve_current）
只按中文逐字比较。后果是静默的：写成英文的取值不会被任何判断命中，
已废止的法规会被判"现行有效"并照常返回（批次 26 真的往库里写过 5 行英文取值）。

现已收敛到 app/db/law_status.py（唯一定义处）。本测试把词表钉在五个层面：
1) 字面量值本身（值域 = 库里既有取值，改了就是不兼容变更）
2) 模块间是同**一个对象**（不是各处复制了一份同名常量）
3) 抽取器只产出词表取值（不适用由文书类型判定，不从页面文字认领）
4) 判定行为（已废止/已失效判失效；不适用与空值不判失效）
5) 源码扫描（app/ 下不再出现裸字面量；注释与文档字符串不算"取值"）
"""
import ast
from datetime import date
from pathlib import Path

from app.db import law_status
from app.db.law_status import (
    EFFECTIVE,
    EXPIRED_STATUSES,
    LAPSED,
    LAW_STATUSES,
    NOT_APPLICABLE,
    PAGE_STATUS_KEYWORDS,
    REPEALED,
)
from app.db.sql_models import Document, DocumentChunk, DocumentVersion, Law, LawVersion
from app.db.vector_index_service import _fetch_chunk_metadata
from app.ingest.law_metadata_extractor import extract_status
from app.retrieval.context_builder import resolve_current_status
from app.retrieval.keyword_search import _resolve_current

from conftest import create_test_session

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def test_status_literals_match_database_vocabulary() -> None:
    """① 值域锁定：四个取值就是库里使用的中文词，改名即破坏兼容。"""
    assert EFFECTIVE == "现行有效"
    assert REPEALED == "已废止"
    assert LAPSED == "已失效"
    assert NOT_APPLICABLE == "不适用"
    assert LAW_STATUSES == ("现行有效", "已废止", "已失效", "不适用")
    # 失效判定只认已废止 / 已失效；不适用不在其中（案例材料不参与时效判断）
    assert EXPIRED_STATUSES == ("已废止", "已失效")


def test_consumers_share_the_same_constant_objects() -> None:
    """② 单一来源：三个消费点拿到的是同一个对象，没有本地副本。"""
    from app.db import vector_index_service
    from app.retrieval import context_builder, keyword_search

    assert context_builder.EXPIRED_STATUSES is law_status.EXPIRED_STATUSES
    assert vector_index_service.EXPIRED_STATUSES is law_status.EXPIRED_STATUSES
    assert keyword_search.EXPIRED_STATUSES is law_status.EXPIRED_STATUSES
    # 抽取器取同一份页面字样表
    from app.ingest import law_metadata_extractor

    assert law_metadata_extractor.PAGE_STATUS_KEYWORDS is law_status.PAGE_STATUS_KEYWORDS


def test_extractor_returns_vocabulary_values_only() -> None:
    """③ 抽取器只产出词表取值；不适用不从页面文字里认领（页面不会这么写）。"""
    assert extract_status("该法规页面标注：现行有效") == EFFECTIVE
    assert extract_status("该法规页面标注：已废止") == REPEALED
    assert extract_status("该法规页面标注：已失效") == LAPSED
    # 正文里出现"不适用"三个字属于法律用语，不是效力状态
    assert extract_status("本情形不适用于该条款。") is None
    # 抽不到就留空，绝不猜
    assert extract_status("页面没有任何状态字样") is None
    assert set(PAGE_STATUS_KEYWORDS) == {EFFECTIVE, REPEALED, LAPSED}


def test_expired_vocabulary_is_judged_not_current_on_both_sides() -> None:
    """④ 已废止/已失效在展示侧与关键词侧都判为失效（无失效日期也照样判失效）。"""
    for expired in EXPIRED_STATUSES:
        assert resolve_current_status(expired, None) is False, f"展示侧漏判 {expired}"
        assert _resolve_current(expired, None) is False, f"关键词侧漏判 {expired}"


def test_not_applicable_is_not_judged_expired() -> None:
    """④ 不适用（案例材料）不判失效——它既不是现行有效也不是已失效，只是不参与时效判断。"""
    assert resolve_current_status(NOT_APPLICABLE, None) is True
    assert _resolve_current(NOT_APPLICABLE, None) is True
    # 空值（待人工补录）同样不判失效：缺数据不能成为静默隐藏条文的理由
    assert resolve_current_status(None, None) is None


def _seed_version_with_status(session, status: str):
    """造一条"法规 + 版本 + 文档版本 + 分块"的最小图，返回分块行。

    与生产同构：Milvus 的 is_current 由 _fetch_chunk_metadata 从
    document_chunks ⋈ law_versions ⋈ laws 联表算出，所以索引侧的真实口径
    必须用这张图验证，而不是只测纯函数。
    """
    law = Law(
        law_key="law_status_probe",
        name="状态判定探针法规",
        short_name="探针法规",
        document_type="法律",
        authority_level=2,
        issuing_authority="全国人民代表大会常务委员会",
        jurisdiction="中国大陆",
        source_url="https://example.com/status-probe",
    )
    session.add(law)
    session.flush()

    document = Document(
        document_key="doc-status-probe",
        source_url="https://example.com/status-probe",
        title="状态判定探针法规",
    )
    session.add(document)
    session.flush()

    version = DocumentVersion(
        version_key="ver-status-probe",
        document_id=document.id,
        content_hash="hash-status-probe",
        raw_file_path="raw/status-probe.html",
        media_type="text/html",
        cleaned_content="第一条 探针正文。",
        version_status="approved",
        processing_status="indexed",
    )
    session.add(version)
    session.flush()

    chunk = DocumentChunk(
        chunk_key="chunk-status-probe",
        document_version_id=version.id,
        chunk_type="parent",
        article_number="第一条",
        sequence=1,
        content="第一条 探针正文。",
        retrieval_text="探针法规 第一条 探针正文。",
    )
    session.add(chunk)
    session.flush()

    session.add(
        LawVersion(
            version_key="ver-status-probe",
            law_id=law.id,
            version_number="v1",
            effective_date=date(2008, 1, 1),
            expiration_date=None,
            status=status,
            document_version_id=version.id,
        )
    )
    session.flush()
    return chunk


def test_index_side_marks_expired_vocabulary_version_as_not_current() -> None:
    """④ 索引侧（写 Milvus is_current 的那段）对已废止版本必须算出 False。"""
    session = create_test_session()
    try:
        chunk = _seed_version_with_status(session, REPEALED)
        metadata = _fetch_chunk_metadata(session, [chunk])
        assert metadata["chunk-status-probe"]["is_current"] is False
    finally:
        session.close()


def test_index_side_keeps_not_applicable_version_current() -> None:
    """④ 索引侧对不适用（案例材料）不得判失效：Milvus 是非空 BOOL，写 True。"""
    session = create_test_session()
    try:
        chunk = _seed_version_with_status(session, NOT_APPLICABLE)
        metadata = _fetch_chunk_metadata(session, [chunk])
        assert metadata["chunk-status-probe"]["is_current"] is True
    finally:
        session.close()


def _string_literals_outside_docstrings(path: Path) -> list[tuple[int, str]]:
    """取出文件里"作为取值使用"的字符串常量（行号, 值）。

    只走 AST：注释根本不是语法节点，天然被排除；文档字符串（模块/类/函数的第一条
    字符串语句）也排除——它们是在**描述**词表，而本规则管的是**代码里的取值**。
    f-string 里嵌入的字面量属于 JoinedStr 的常量片段，不在文档字符串位置上，照样会被查到。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    ]


def test_no_bare_status_literals_in_app_source() -> None:
    """⑤ 源码扫描：四个取值在 app/ 下只允许出现在词表模块里。

    这条守住的正是本批的事故：只要有人在别处再写一个中文（或英文）取值字面量，
    两套词表就会重新分叉，而分叉的表现是"静默漏判"而不是报错。
    """
    allowed = APP_DIR / "db" / "law_status.py"
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path == allowed:
            continue
        for lineno, literal in _string_literals_outside_docstrings(path):
            hits = [value for value in LAW_STATUSES if value in literal]
            if hits:
                offenders.append(f"{path.relative_to(APP_DIR)}:{lineno}: {literal!r} 含 {hits}")
    assert offenders == []


def test_column_comment_documents_the_chinese_vocabulary() -> None:
    """⑤ 列注释必须写中文词表：英文词表曾与消费点分叉，是本次事故的直接来源。

    只禁"把英文当取值写"（带引号、或"英文-中文"对照表那种取值列表形态）；
    注释里回顾历史时提到英文词名是允许的——不然等于禁止记录事故原因。
    """
    from app.db import law_models

    source = Path(law_models.__file__).read_text(encoding="utf-8")
    # 英文取值形态不得复活（effective_date 这类字段名不算）
    assert '"effective"' not in source
    assert "effective-现行有效" not in source
    assert "repealed-已废止" not in source
    # 中文词表与"不适用不参与时效判断"的说明必须在列注释里
    assert "现行有效 / 已废止 / 已失效 / 不适用" in source
    assert "不参与时效判断" in source
