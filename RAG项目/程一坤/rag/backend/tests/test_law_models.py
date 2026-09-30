# 导入日期类型
from datetime import date

# 导入 pytest
import pytest

# 导入 SQLAlchemy 异常
from sqlalchemy.exc import IntegrityError

# 导入测试工具
from tests.conftest import create_test_session

# 导入新的法规表模型
from app.db.sql_models import Law, LawVersion, Article


def test_create_law_with_required_fields() -> None:
    """测试创建法规记录。"""
    session = create_test_session()

    # 创建法规记录
    law = Law(
        law_key="law_labor_contract",
        name="中华人民共和国劳动合同法",
        short_name="劳动合同法",
        document_type="法律",
        authority_level=2,
        issuing_authority="全国人民代表大会常务委员会",
        jurisdiction="national",
        legal_domain="劳动法",
        source_url="https://example.com/law/labor-contract",
    )

    session.add(law)
    session.commit()

    # 验证记录已保存
    assert law.id is not None
    assert law.name == "中华人民共和国劳动合同法"
    assert law.authority_level == 2


def test_law_key_must_be_unique() -> None:
    """测试法规标识必须唯一。"""
    session = create_test_session()

    # 创建第一个法规
    law1 = Law(
        law_key="law_test",
        name="测试法规A",
        document_type="法律",
        authority_level=2,
        issuing_authority="测试机关",
        jurisdiction="national",
        source_url="https://example.com/law/test-a",
    )
    session.add(law1)
    session.commit()

    # 尝试创建相同 law_key 的法规应该失败
    law2 = Law(
        law_key="law_test",
        name="测试法规B",
        document_type="法律",
        authority_level=2,
        issuing_authority="测试机关",
        jurisdiction="national",
        source_url="https://example.com/law/test-b",
    )
    session.add(law2)

    with pytest.raises(IntegrityError):
        session.commit()


def test_create_law_version_with_effective_dates() -> None:
    """测试创建法规版本记录并设置生效失效日期。"""
    session = create_test_session()

    # 先创建法规
    law = Law(
        law_key="law_labor_contract",
        name="中华人民共和国劳动合同法",
        document_type="法律",
        authority_level=2,
        issuing_authority="全国人民代表大会常务委员会",
        jurisdiction="national",
        source_url="https://example.com/law/labor-contract",
    )
    session.add(law)
    session.commit()

    # 创建法规版本（2008版）
    version = LawVersion(
        version_key="law_labor_contract_v2008",
        law_id=law.id,
        version_number="2008版",
        promulgation_date=date(2007, 6, 29),
        effective_date=date(2008, 1, 1),
        expiration_date=date(2012, 12, 27),  # 被修正版替代
        status="superseded",
        revision_note="原始版本",
    )
    session.add(version)
    session.commit()

    # 验证版本记录
    assert version.id is not None
    assert version.law_id == law.id
    assert version.effective_date == date(2008, 1, 1)
    assert version.expiration_date == date(2012, 12, 27)
    assert version.status == "superseded"


def test_law_version_number_must_be_unique_per_law() -> None:
    """测试同一法规的版本号必须唯一。"""
    session = create_test_session()

    # 创建法规
    law = Law(
        law_key="law_test",
        name="测试法规",
        document_type="法律",
        authority_level=2,
        issuing_authority="测试机关",
        jurisdiction="national",
        source_url="https://example.com/law/test",
    )
    session.add(law)
    session.commit()

    # 创建第一个版本
    version1 = LawVersion(
        version_key="law_test_v2008",
        law_id=law.id,
        version_number="2008版",
        promulgation_date=date(2008, 1, 1),
        effective_date=date(2008, 1, 1),
        status="effective",
    )
    session.add(version1)
    session.commit()

    # 尝试创建相同版本号应该失败
    version2 = LawVersion(
        version_key="law_test_v2008_dup",
        law_id=law.id,
        version_number="2008版",
        promulgation_date=date(2008, 1, 1),
        effective_date=date(2008, 1, 1),
        status="effective",
    )
    session.add(version2)

    with pytest.raises(IntegrityError):
        session.commit()


def test_create_articles_with_clause_and_item_numbers() -> None:
    """测试创建条文记录，包含条/款/项三级结构。"""
    session = create_test_session()

    # 创建法规和版本
    law = Law(
        law_key="law_labor_contract",
        name="中华人民共和国劳动合同法",
        document_type="法律",
        authority_level=2,
        issuing_authority="全国人民代表大会常务委员会",
        jurisdiction="national",
        source_url="https://example.com/law/labor-contract",
    )
    session.add(law)
    session.commit()

    version = LawVersion(
        version_key="law_labor_contract_v2013",
        law_id=law.id,
        version_number="2013修正",
        promulgation_date=date(2012, 12, 28),
        effective_date=date(2013, 7, 1),
        status="effective",
    )
    session.add(version)
    session.commit()

    # 创建条文：第四十七条（无款项）
    article_47 = Article(
        article_key="law_labor_contract_v2013_art47",
        law_version_id=version.id,
        article_number="47",
        paragraph_number=None,
        item_number=None,
        article_path="47",
        chapter_path="第四章 劳动合同的解除和终止",
        content="经济补偿按劳动者在本单位工作的年限...",
        sequence=47,
    )

    # 创建条文：第四十七条第二款
    article_47_2 = Article(
        article_key="law_labor_contract_v2013_art47-2",
        law_version_id=version.id,
        article_number="47",
        paragraph_number="2",
        item_number=None,
        article_path="47-2",
        chapter_path="第四章 劳动合同的解除和终止",
        content="劳动者月工资高于用人单位所在直辖市、设区的市级人民政府公布的本地区上年度职工月平均工资三倍的...",
        sequence=472,
    )

    # 创建条文：第四十七条第二款第一项
    article_47_2_1 = Article(
        article_key="law_labor_contract_v2013_art47-2-1",
        law_version_id=version.id,
        article_number="47",
        paragraph_number="2",
        item_number="1",
        article_path="47-2-1",
        chapter_path="第四章 劳动合同的解除和终止",
        content="向其支付经济补偿的标准按职工月平均工资三倍的数额支付",
        sequence=4721,
    )

    session.add_all([article_47, article_47_2, article_47_2_1])
    session.commit()

    # 验证条文记录
    assert article_47.id is not None
    assert article_47.article_number == "47"
    assert article_47.paragraph_number is None
    assert article_47.article_path == "47"

    assert article_47_2.article_number == "47"
    assert article_47_2.paragraph_number == "2"
    assert article_47_2.item_number is None
    assert article_47_2.article_path == "47-2"

    assert article_47_2_1.article_number == "47"
    assert article_47_2_1.paragraph_number == "2"
    assert article_47_2_1.item_number == "1"
    assert article_47_2_1.article_path == "47-2-1"


def test_article_must_be_unique_per_version() -> None:
    """测试同一版本的 article_path 必须唯一。"""
    session = create_test_session()

    # 创建法规和版本
    law = Law(
        law_key="law_test",
        name="测试法规",
        document_type="法律",
        authority_level=2,
        issuing_authority="测试机关",
        jurisdiction="national",
        source_url="https://example.com/law/test",
    )
    session.add(law)
    session.commit()

    version = LawVersion(
        version_key="law_test_v2020",
        law_id=law.id,
        version_number="2020版",
        promulgation_date=date(2020, 1, 1),
        effective_date=date(2020, 1, 1),
        status="effective",
    )
    session.add(version)
    session.commit()

    # 创建第一条第一款
    article1 = Article(
        article_key="law_test_v2020_art1-1",
        law_version_id=version.id,
        article_number="1",
        paragraph_number="1",
        item_number=None,
        article_path="1-1",
        content="测试条文A",
        sequence=1,
    )
    session.add(article1)
    session.commit()

    # 尝试创建相同 article_path 的条文应该失败
    article2 = Article(
        article_key="law_test_v2020_art1-1_dup",
        law_version_id=version.id,
        article_number="1",
        paragraph_number="1",
        item_number=None,
        article_path="1-1",
        content="测试条文B",
        sequence=2,
    )
    session.add(article2)

    with pytest.raises(IntegrityError):
        session.commit()
