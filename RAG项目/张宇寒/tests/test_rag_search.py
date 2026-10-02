"""基础检索的行为测试：数据库和云端接口用小样例代替，不写真实法律库。"""

import json
from types import SimpleNamespace

import pytest

from backend.app.rag.plan import CollectionConfig, SearchRouter
from backend.app.rag.search import EvidenceBuilder, ProcessingPipeline, Reranker, VectorRetriever, rrf
from backend.app.storage.vector import MilvusStore


def sample(source_id, content, score=0.8, channel="vector", collection="civil_cases"):
    return {"source_id": source_id, "source_type": "public", "collection": collection,
            "title": "测试资料", "content": content, "score": score, "retrieval_channel": channel}


def settings(**values):
    values.setdefault("processed_dir", "__missing_test_processed__")
    return SimpleNamespace(embedding_dim=1024, retrieval_candidate_pool_max=20,
                           retrieval_priority_top_k=4, retrieval_non_priority_top_k=1,
                           retrieval_max_query_variants=4, retrieval_max_collections=6, **values)


class ExampleModel:
    def __init__(self, scores=(0.9, 0.1)):
        self.scores = scores
        self.embedded = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    def rerank(self, query, docs):
        return sorted(enumerate(self.scores[:len(docs)]), key=lambda pair: pair[1], reverse=True)


class ExampleMilvus:
    def __init__(self, exact=(), keyword=(), vector=()):
        self.exact = exact
        self.keyword = keyword
        self.vector = vector
        self.vector_collections = []

    def query_public(self, collection, expression, limit, score=0.9):
        assert expression == 'id == "civil_code_articles_article_577"'
        return [dict(row) for row in self.exact][:limit]

    def search_public_keyword(self, collection, keywords, limit):
        return [dict(row) for row in self.keyword][:limit]

    def search_public(self, collection, vector, limit):
        assert len(vector) == 1024
        self.vector_collections.append(collection)
        return [dict(row) for row in self.vector][:limit]


def test_public_search_uses_database_not_unpublished_processed_files(tmp_path):
    local = tmp_path / "civil_code_articles.json"
    local.write_text(json.dumps([{"id": "civil_code_articles_article_577", "article_number": "577",
                                  "article_content": "尚未入库的本地版本"}]), encoding="utf-8")
    database_row = sample("civil_code_articles:577", "数据库正式版本", 1.0, "exact", "civil_code_articles")
    model = ExampleModel()
    result = VectorRetriever(model, ExampleMilvus(exact=[database_row]),
                             settings(processed_dir=tmp_path)).search_public(
        "民法典第577条", ["civil_code_articles"], article_numbers=["577"])
    assert result[0]["content"] == "数据库正式版本"
    assert model.embedded == []  # 明确查条号时不必请求嵌入接口。


def test_daily_language_can_retrieve_civil_code_by_vector():
    model = ExampleModel()
    milvus = ExampleMilvus(vector=[sample("law:1", "相关条文", collection="civil_code_articles")])
    plan = SearchRouter(settings()).route({"query": "签了合同对方不履行怎么办", "route": "law_only"})
    result = VectorRetriever(model, milvus, settings()).search_public(
        plan.query, ["civil_code_articles"], collection_configs=plan.collection_configs)
    assert result[0]["content"] == "相关条文"


def test_local_retrieval_mode_never_uses_web_for_current_law_questions():
    plan = SearchRouter(settings()).route(
        {"query": "latest civil law rule", "route": "current_law_web"},
        retrieval_mode="local",
    )

    assert plan.retrieval_mode == "local"
    assert plan.include_web is False


def test_auto_retrieval_mode_uses_web_when_current_information_is_needed():
    plan = SearchRouter(settings()).route(
        {"query": "latest civil law rule", "route": "current_law_web"},
        retrieval_mode="auto",
    )

    assert plan.retrieval_mode == "auto"
    assert plan.include_web is True


def test_force_retrieval_mode_uses_web_for_an_ordinary_legal_question():
    plan = SearchRouter(settings()).route(
        {"query": "loan dispute", "route": "law_only"},
        retrieval_mode="force",
    )

    assert plan.retrieval_mode == "force"
    assert plan.include_web is True


def test_public_search_limits_variants_and_merges_duplicate_sources():
    model = ExampleModel()
    milvus = ExampleMilvus(keyword=[sample("case:1", "同一案例", 0.75)],
                           vector=[sample("case:1", "同一案例", 0.8)])
    result = VectorRetriever(model, milvus, settings()).search_public(
        "原始问题", ["civil_cases"], keywords=["违约"], priority_collections=["civil_cases"],
        query_variants=["原始问题", "改写一", "改写二", "改写三"])
    assert [row["source_id"] for row in result] == ["case:1"]
    assert model.embedded == ["原始问题", "改写一"]
    assert set(result[0]["matched_channels"]) == {"keyword", "vector"}


def test_rrf_scores_do_not_prevent_model_reranking():
    rows = [sample("case:1", "无关案例"), sample("case:2", "真正相关的案例")]
    fused = rrf([rows])
    result = Reranker(ExampleModel(scores=(0.1, 0.9))).rerank("用户问题", fused)
    assert result[0]["source_id"] == "case:2"
    assert result[0]["rerank_score"] == 0.9


def test_exact_article_is_kept_while_other_candidates_are_reranked():
    exact = sample("law:577", "指定条文", 1.0, "exact", "civil_code_articles")
    rows = [exact, sample("case:1", "无关案例"), sample("case:2", "相关案例")]
    result = Reranker(ExampleModel(scores=(0.1, 0.9))).rerank("用户问题", rrf([rows]))
    assert [row["source_id"] for row in result] == ["law:577", "case:2", "case:1"]
    assert result[1]["rerank_score"] == 0.9


def test_low_rerank_score_cannot_be_overridden_by_high_vector_score():
    row = {**sample("case:1", "模型判定不相关", 0.95), "rerank_score": 0.1}
    assert EvidenceBuilder().build([row]) == []


def test_retrieval_preserves_article_and_case_metadata():
    store = object.__new__(MilvusStore)  # 这里只测试结果整理，不初始化数据库连接。
    entity = {"case_id": "case_1", "summary": "案例正文", "case_number": "测试案号",
              "cause_of_action": "合同纠纷", "embedding": [0.1] * 1024}
    row = store.format_public_entity("civil_cases", entity, 0.8)
    assert row["case_number"] == "测试案号"
    assert row["cause_of_action"] == "合同纠纷"
    assert "embedding" not in row
    assert "summary" not in row  # 正文统一放content，不重复携带大段文本。


def test_complete_article_reaches_answer_prompt_without_silent_truncation():
    from backend.app.rag.answer import AnswerGenerator
    from backend.app.rag.context import ContextBuilder

    text = "条文正文。" * 300 + "条文末尾的重要例外条件。"
    row = sample("law:1", text, 1.0, "exact", "civil_code_articles")
    evidence = EvidenceBuilder(max_content_chars=200).build([row])
    context = ContextBuilder(evidence_max_chars=200).build_context("问题", evidence=evidence)
    prompt = AnswerGenerator._evidence_context(context["evidence"], max_chars=200)
    assert text in prompt


def test_truncated_case_is_explicitly_marked_as_excerpt():
    row = sample("case:1", "案例内容。" * 100)
    result = EvidenceBuilder(max_content_chars=200).build([row])
    assert result[0]["content_truncated"] is True
    assert result[0]["content"].endswith("【节选，非完整原文】")


def test_keyword_failure_does_not_disable_vector_retrieval():
    class KeywordFailure(ExampleMilvus):
        def search_public_keyword(self, *args):
            raise RuntimeError("模拟关键词故障")

    milvus = KeywordFailure(vector=[sample("case:1", "仍可检索到的案例")])
    result = VectorRetriever(ExampleModel(), milvus, settings()).search_public(
        "合同问题", ["civil_cases"], keywords=["违约"])
    assert result[0]["content"] == "仍可检索到的案例"


def test_total_database_failure_is_not_reported_as_no_matching_law():
    class DatabaseFailure(ExampleMilvus):
        def search_public(self, *args):
            raise RuntimeError("数据库不可用")

    with pytest.raises(RuntimeError, match="公共法律库检索失败"):
        VectorRetriever(ExampleModel(), DatabaseFailure(), settings()).search_public("合同问题", ["civil_cases"])


@pytest.mark.parametrize("vectors", [[[0.1] * 512], [[float("nan")] * 1024], []])
def test_invalid_query_vectors_are_not_sent_to_milvus(vectors):
    model = SimpleNamespace(embed=lambda texts: vectors)
    milvus = ExampleMilvus(vector=[sample("case:1", "不应返回的结果")])
    with pytest.raises(ValueError, match="向量"):
        VectorRetriever(model, milvus, settings()).search_public("合同问题", ["civil_cases"])
    assert milvus.vector_collections == []


def test_rerank_failure_keeps_candidates_without_fabricating_model_scores():
    model = SimpleNamespace(rerank=lambda query, docs: [(0, 0.5)], last_rerank_error="接口故障")
    result = Reranker(model).rerank("问题", rrf([[sample("case:1", "案例")]]))
    assert result[0]["rerank_degraded"] is True
    assert "rerank_score" not in result[0]


def test_pipeline_keeps_only_relevant_evidence():
    pipeline = ProcessingPipeline(Reranker(ExampleModel(scores=(0.1, 0.9))), EvidenceBuilder())
    result = pipeline.process("问题", [[sample("case:1", "无关"), sample("case:2", "相关")]])
    assert [row["source_id"] for row in result["evidence"]] == ["case:2"]


def test_unsafe_article_number_is_not_inserted_into_filter():
    assert VectorRetriever.article_filter("civil_interpretations", '577") or id != ""') == ""


def test_collection_load_failure_is_reported_as_retrieval_failure():
    store = object.__new__(MilvusStore)
    store.public = SimpleNamespace(has_collection=lambda name: True)
    store.load_public_collection = lambda name: False
    with pytest.raises(RuntimeError, match="加载失败"):
        store.search_public("civil_cases", [0.1] * 1024)


def test_rerank_degradation_counts_as_low_confidence_even_with_high_vector_score():
    from backend.app.rag.workflow import RagWorkflow

    workflow = object.__new__(RagWorkflow)
    workflow.settings = SimpleNamespace(log_low_confidence_score=0.7)
    row = {**sample("case:1", "案例", 0.99), "rerank_degraded": True}
    assert workflow.warn_low_confidence([row]) == 1


def test_single_candidate_is_scored_by_model_not_automatically_given_one(monkeypatch):
    from backend.app.models.rerank import RerankClient

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.1}]}

    monkeypatch.setattr("backend.app.models.rerank.httpx.post", lambda *args, **kwargs: Response())
    config = SimpleNamespace(siliconflow_api_key="test-only", siliconflow_reranker_base_url="https://example.invalid",
                             reranker_model="example-model", reranker_top_n=6)
    assert RerankClient(config).rerank("问题", ["不相关的案例"]) == [(0, 0.1)]


def test_article_too_large_for_context_budget_is_not_silently_cut():
    from backend.app.rag.context import ContextBuilder

    row = sample("law:1", "条文。" * 1000, 1.0, "exact", "civil_code_articles")
    assert ContextBuilder().select_evidence([row], token_budget=100) == []


def test_citation_relation_requested_by_user_is_kept_as_evidence():
    class CitationMilvus(ExampleMilvus):
        def query_public(self, collection, expression, limit, score=0.9):
            return [sample("citation:1", "引用关系", 0.95, "exact", "civil_citations")]

    rows = VectorRetriever(ExampleModel(), CitationMilvus(), settings()).search_public(
        "民法典第577条的引用关系", ["civil_citations"], article_numbers=["577"], wants_citation_relations=True)
    assert EvidenceBuilder().build(rows)[0]["source_id"] == "citation:1"


def test_named_interpretation_article_does_not_return_same_number_civil_code():
    class NamedLawMilvus(ExampleMilvus):
        def query_public(self, collection, expression, limit, score=0.9):
            if collection == "civil_code_articles":
                return [sample("law:1", "民法典第一条", 1.0, "exact", collection)]
            assert 'id like "civil_interpretations_%_1_%"' in expression
            assert 'law_name like "%民法典总则编司法解释%"' in expression
            return [{**sample("interpretation:1", "请求的解释第一条", 0.95, "exact", collection),
                     "id": "civil_interpretations_2_1_1", "law_name": "民法典总则编司法解释"}]

    rows = VectorRetriever(ExampleModel(), NamedLawMilvus(), settings()).search_public(
        "《民法典总则编司法解释》第一条", ["civil_code_articles", "civil_interpretations"], article_numbers=["1"])
    assert [row["content"] for row in rows] == ["请求的解释第一条"]


def test_target_law_load_failure_is_not_hidden_by_empty_auxiliary_collection():
    class TargetFailure(ExampleMilvus):
        def query_public(self, collection, *args, **kwargs):
            if collection == "civil_code_articles":
                raise RuntimeError("指定法律库不可用")
            return []

    with pytest.raises(RuntimeError, match="公共法律库检索失败"):
        VectorRetriever(ExampleModel(), TargetFailure(), settings()).search_public(
            "民法典第577条", ["civil_code_articles", "civil_interpretations"], article_numbers=["577"])


def test_workflow_reports_only_sources_that_reach_answer_context():
    from backend.app.rag.context import ContextBuilder
    from backend.app.rag.plan import SearchPlan
    from backend.app.rag.workflow import RagWorkflow

    class ExampleGenerator:
        def stream_answer_text(self, question, evidence, history, **kwargs):
            assert evidence, "没有上下文依据时不能进入有依据的流式回答分支"
            yield "模拟回答"

        def generate(self, question, evidence, history, **kwargs):
            return {"answer": "当前没有足够可用依据，需进一步核对。", "citations": []}

    row = sample("law:1", "条文正文。" * 1000, 1.0, "exact", "civil_code_articles")
    workflow = object.__new__(RagWorkflow)
    workflow.settings = SimpleNamespace(log_low_confidence_score=0.7)
    workflow.memory = None
    workflow.user_state = None
    workflow.understanding = SimpleNamespace(understand=lambda query, memory_context=None: {"query": query})
    workflow.router = SimpleNamespace(route=lambda info, include_web: SearchPlan(
        query=info["query"], collections=["civil_code_articles"], search_private=False, include_web=False))
    workflow.retriever = SimpleNamespace(search_public_knowledge=lambda plan: [row])
    workflow.ranker = Reranker(ExampleModel())
    workflow.evidence = EvidenceBuilder()
    workflow.processing = ProcessingPipeline(workflow.ranker, workflow.evidence)
    workflow.context_builder = ContextBuilder(budget={"evidence": 100})
    workflow.generator = ExampleGenerator()
    result = workflow.run("问题", thinking_enabled=False)
    assert result["sources"] == []
    assert result["meta"]["evidence_count"] == 0
    assert result["meta"]["context"]["dropped_evidence_count"] == 1


@pytest.mark.parametrize("query", ["民事诉讼法第一条", "反家庭暴力法第二十三条",
                                   "民法典总则编解释第一条", "民法典总则编解释的第一条",
                                   "民法典总则编解释中的第一条"])
def test_unquoted_other_law_is_not_guessed_to_be_civil_code(query):
    class WrongLawMilvus(ExampleMilvus):
        def query_public(self, collection, *args, **kwargs):
            return [sample("law:1", "不该返回的民法典同号条文", 1.0, "exact", "civil_code_articles")]

    result = VectorRetriever(ExampleModel(), WrongLawMilvus(), settings()).search_public(
        query, ["civil_code_articles", "civil_interpretations"], article_numbers=["1"])
    assert result == []


def test_interpretation_about_a_statute_is_not_the_statute_itself():
    class AboutLawMilvus(ExampleMilvus):
        def query_public(self, collection, *args, **kwargs):
            return [{**sample("interpretation:1", "解释自身第一条", 0.95, "exact", collection),
                     "id": "civil_interpretations_2_1_1",
                     "law_name": "最高人民法院关于适用中华人民共和国民事诉讼法的解释"}]

    result = VectorRetriever(ExampleModel(), AboutLawMilvus(), settings()).search_public(
        "《民事诉讼法》第一条", ["civil_code_articles", "civil_interpretations"], article_numbers=["1"])
    assert result == []
