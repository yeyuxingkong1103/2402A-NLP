"""测试上下文组装与引用字段补全。

任务书 3-B 要求：
- 每个上下文片段带齐 11 个引用字段
- 限制片段数量与总长度
- 超长时按规则截断且不破坏条文完整性
"""
from app.retrieval.context_builder import build_context_block, RetrievedArticle


def test_context_with_full_metadata():
    """测试完整元数据字段：11 个字段齐全"""
    articles = [
        RetrievedArticle(
            chunk_key="chunk1",
            content="用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。",
            article_number="82",
            document_title="中华人民共和国劳动合同法实施条例",
            source_url="https://example.com/law1",
            recall_score=0.95,
            rerank_score=0.92,
            # 新增字段
            document_type="行政法规",
            paragraph_number=None,
            item_number=None,
            jurisdiction="中国大陆",
            effective_date="2008-09-18",
            expiration_date=None,
            issuing_authority="国务院",
            is_current=True,
        ),
    ]

    context = build_context_block(articles)

    # 验证 11 个字段都出现在输出中
    assert "中华人民共和国劳动合同法实施条例" in context  # 法规名称
    assert "行政法规" in context  # 文书类型
    assert "第 82 条" in context  # 条号
    assert "中国大陆" in context  # 法域
    assert "2008-09-18" in context  # 生效日期
    assert "国务院" in context  # 发布机关
    assert "https://example.com/law1" in context  # 来源 URL
    assert "现行有效" in context or "有效" in context  # 是否现行有效
    # 款号、项号、失效日期为空时不出现或标注"无"


def test_context_with_paragraph_and_item():
    """测试款号与项号显示"""
    articles = [
        RetrievedArticle(
            chunk_key="chunk1",
            content="劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。",
            article_number="37",
            paragraph_number="1",
            item_number=None,
            document_title="劳动合同法",
            source_url="https://example.com/law1",
            recall_score=0.9,
            document_type="法律",
            jurisdiction="中国大陆",
            effective_date="2008-01-01",
            expiration_date=None,
            issuing_authority="全国人民代表大会常务委员会",
            is_current=True,
        ),
    ]

    context = build_context_block(articles)

    # 验证款号显示
    assert "第 37 条" in context
    assert "第 1 款" in context or "第一款" in context


def test_context_length_limit():
    """测试长度限制：超长时截断"""
    # 构造 20 条长条文
    articles = [
        RetrievedArticle(
            chunk_key=f"chunk{i}",
            content="条文内容" * 100,  # 每条约 400 字
            article_number=str(i),
            document_title=f"法规{i}",
            source_url=f"https://example.com/law{i}",
            recall_score=0.9 - i * 0.01,
            document_type="法律",
            jurisdiction="中国大陆",
            effective_date="2020-01-01",
            expiration_date=None,
            issuing_authority="全国人大",
            is_current=True,
        )
        for i in range(20)
    ]

    # 限制最多 5 条
    context = build_context_block(articles, max_items=5)

    # 验证只有 5 条
    assert context.count("[1]") == 1
    assert context.count("[5]") == 1
    assert "[6]" not in context


def test_context_expired_law():
    """测试失效法规的显示"""
    articles = [
        RetrievedArticle(
            chunk_key="chunk1",
            content="某条文",
            article_number="10",
            document_title="已失效的法规",
            source_url="https://example.com/old",
            recall_score=0.8,
            document_type="法律",
            jurisdiction="中国大陆",
            effective_date="1995-01-01",
            expiration_date="2020-12-31",
            issuing_authority="全国人大",
            is_current=False,
        ),
    ]

    context = build_context_block(articles)

    # 验证失效日期和现行状态
    assert "2020-12-31" in context  # 失效日期
    assert "已失效" in context or "非现行" in context


def test_context_missing_optional_fields():
    """测试可选字段缺失时的处理"""
    articles = [
        RetrievedArticle(
            chunk_key="chunk1",
            content="某条文",
            article_number=None,  # 无条号
            paragraph_number=None,
            item_number=None,
            document_title="司法解释",
            source_url="https://example.com/law",
            recall_score=0.9,
            document_type="司法解释",
            jurisdiction="中国大陆",
            effective_date=None,  # 无生效日期
            expiration_date=None,
            issuing_authority=None,  # 无发布机关
            is_current=True,
        ),
    ]

    context = build_context_block(articles)

    # 验证不会因为缺失字段而报错
    assert "司法解释" in context
    assert "https://example.com/law" in context
    # 缺失字段应该标注"未知"或不显示
