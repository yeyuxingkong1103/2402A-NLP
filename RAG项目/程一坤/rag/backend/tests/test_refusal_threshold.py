"""拒答阈值（阶段 8.4 校准产物）测试。

口径：`should_refuse` 在"零候选"之外，新增"候选最高向量相似度低于阈值"这一拒答条件；
阈值取 0（默认）时行为与历史完全一致——不新增拒答。
阈值为什么用向量分而不是重排分：见 reports/refusal_calibration_*.md（重排分两簇重叠会误拒）。
"""
from types import SimpleNamespace

from app.chat.guard import should_refuse
from app.chat.service import ChatService


def _article(vector_score: float | None, rerank_score: float | None = None) -> SimpleNamespace:
    """最小候选替身：只带护栏与法源组装会用到的字段。"""
    return SimpleNamespace(
        rerank_score=rerank_score,
        vector_score=vector_score,
        recall_score=vector_score or 0.5,
        chunk_key="chunk-1",
        content="第四十七条 经济补偿按劳动者在本单位工作的年限计算。",
        document_title="中华人民共和国劳动合同法",
        article_number="第四十七条",
        paragraph_number=None,
        item_number=None,
    )


def test_threshold_zero_keeps_legacy_behaviour():
    """阈值 0 = 不启用阈值：有候选就作答（历史行为不变）。"""
    assert should_refuse([_article(0.50)], min_vector_score=0.0) is False
    assert should_refuse([]) is True


def test_threshold_refuses_when_best_vector_score_below_cutoff():
    """候选最高向量分低于阈值 → 判为查不到，拒答。"""
    candidates = [_article(0.61), _article(0.55)]
    assert should_refuse(candidates, min_vector_score=0.6305) is True


def test_threshold_answers_when_best_vector_score_above_cutoff():
    """候选中最高向量分达到阈值 → 作答。"""
    candidates = [_article(0.6328), _article(0.51)]
    assert should_refuse(candidates, min_vector_score=0.6305) is False


def test_threshold_ignores_candidates_without_vector_score():
    """候选全部来自关键词召回（无向量分）时不凭阈值拒答。"""
    candidates = [_article(None), _article(None)]
    assert should_refuse(candidates, min_vector_score=0.9) is False


def test_chat_service_refuses_on_low_vector_score():
    """服务层接入阈值：低分候选直接走拒答分支，不调用大模型。"""
    calls: list[str] = []

    class FakeRetrieval:
        def retrieve(self, question, **kwargs):
            return SimpleNamespace(
                articles=[_article(0.55)],
                context_block="上下文",
                stats={"vector_recall_count": 1},
                query_rewrite=None,
            )

    class FakeLlm:
        def chat(self, system, user):  # pragma: no cover - 不应被调用
            calls.append("llm")
            return "不该发生的回答"

    service = ChatService(
        retrieval_service=FakeRetrieval(),
        llm_client=FakeLlm(),
        refusal_min_vector_score=0.6305,
    )
    result = service.chat("与知识库无关的问题")

    assert result.refused is True
    assert calls == []
    assert "refuse_low_score" in result.guardrail_applied


def test_chat_service_threshold_default_is_off():
    """默认阈值 0：同一低分候选仍会正常作答（保证默认行为不变）。"""

    class FakeRetrieval:
        def retrieve(self, question, **kwargs):
            return SimpleNamespace(
                articles=[_article(0.55)],
                context_block="上下文",
                stats={},
                query_rewrite=None,
            )

    class FakeLlm:
        def chat(self, system, user):
            return "根据《劳动合同法》第四十七条，[1] 经济补偿按年限计算。"

    service = ChatService(retrieval_service=FakeRetrieval(), llm_client=FakeLlm())
    result = service.chat("经济补偿怎么算")

    assert result.refused is False
    assert "refuse_low_score" not in result.guardrail_applied
