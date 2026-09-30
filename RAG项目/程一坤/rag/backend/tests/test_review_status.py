"""审核状态词表单一来源：锁定测试（防"改状态名漏改某处"）。

背景：状态字面量原先分散在 8 处——review_service 定义一份、review_detail 为避免
循环导入又本地持有一份、api/review 的列表接口默认值、db/import_service 的新版本
初始化，以及 db/vector_index_service、retrieval/keyword_search、retrieval/vector_search
的 SQL 过滤条件里各有一处裸字面量。改一处漏一处只会表现为"数据查不出来"，极难定位。

现已全部收敛到 app/db/version_status.py（document_versions.version_status 取值的
唯一定义处）。本测试把词表钉死在五个层面：
1) 字面量值本身（值域 = 数据库既有行的实际取值，改了就是不兼容变更）
2) 模块间是同**一个对象**（不是各处复制了一份同名常量）
3) 落库往返（新建版本写进去的原始字符串就是词表值）
4) 分层方向（db / retrieval 不得反向 import review —— 这正是下沉要解决的问题）
5) 源码扫描（app/ 下不再出现裸字面量）
"""
import re
from inspect import signature
from pathlib import Path

from sqlalchemy import select

from app.api.review import list_documents_for_review
from app.db import version_status
from app.db.import_service import import_package
from app.db.sql_models import Document, DocumentVersion
from app.db.version_status import APPROVED, PENDING_REVIEW, REJECTED, REVIEW_STATUSES
from app.review import review_detail, review_service

from conftest import build_validated_package, create_test_session


def test_status_literals_match_database_vocabulary() -> None:
    """① 值域锁定：常量值就是库里既有的三个字符串，改名即破坏兼容。"""
    assert PENDING_REVIEW == "pending_review"
    assert APPROVED == "approved"
    assert REJECTED == "rejected"
    assert REVIEW_STATUSES == ("pending_review", "approved", "rejected")


def test_consumers_share_the_same_constant_objects() -> None:
    """② 单一来源：各模块拿到的是同一个对象，且没有留下本地副本/兼容别名。"""
    assert review_service.PENDING_REVIEW is version_status.PENDING_REVIEW
    assert review_service.APPROVED is version_status.APPROVED
    assert review_service.REJECTED is version_status.REJECTED
    assert review_detail.PENDING_REVIEW is version_status.PENDING_REVIEW
    # 旧名不得复活（否则等于又出现了一份定义）
    assert not hasattr(review_service, "STATUS_PENDING")
    assert not hasattr(review_detail, "STATUS_PENDING")


def test_list_api_default_status_comes_from_vocabulary() -> None:
    """③ 6.3 列表接口默认值取自词表：接口默认口径与库里待审核口径永远一致。"""
    status_param = signature(list_documents_for_review).parameters["status"]
    # FastAPI 的 Query(...) 包装：真实默认值在其 .default 上
    assert status_param.default.default == PENDING_REVIEW


def test_new_imported_version_persists_pending_review_literal() -> None:
    """④ 落库往返：新导入版本写进 document_versions.version_status 的就是词表值。"""
    session = create_test_session()
    try:
        package = build_validated_package()
        result = import_package(session, package)
        assert result.version_status == "new"  # 导入审计语义（new/updated/unchanged）
        version = session.scalar(
            select(DocumentVersion).join(
                Document, Document.id == DocumentVersion.document_id
            )
        )
        # 列里存的是审核状态词表值，与常量模块逐字一致
        assert version.version_status == PENDING_REVIEW == "pending_review"
    finally:
        session.close()


def test_data_and_retrieval_layers_do_not_import_review_layer() -> None:
    """⑤ 分层方向：词表下沉到 db 后，db / retrieval 不得再反向 import review。

    这是本次下沉的核心收益，必须由测试守住——否则将来有人在 keyword_search 里
    `from app.review.review_status import APPROVED`，循环依赖会悄悄回来。
    """
    app_dir = Path(__file__).resolve().parents[1] / "app"
    offenders = [
        f"{path.relative_to(app_dir)}:{lineno}: {line.strip()}"
        for sub in ("db", "retrieval")
        for path in sorted((app_dir / sub).rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.match(r"\s*(from|import)\s+app\.review\b", line)
    ]
    assert offenders == []


def test_no_bare_status_literals_in_app_source() -> None:
    """⑥ 源码扫描：三个状态字面量在 app/ 下只允许出现在词表模块里。"""
    app_dir = Path(__file__).resolve().parents[1] / "app"
    allowed = app_dir / "db" / "version_status.py"
    pattern = re.compile(r"""["'](?:pending_review|approved|rejected)["']""")
    offenders = [
        f"{path.relative_to(app_dir)}:{lineno}: {line.strip()}"
        for path in sorted(app_dir.rglob("*.py"))
        if path != allowed
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []
