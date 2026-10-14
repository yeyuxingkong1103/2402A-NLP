import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from pydantic import ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer

from feedback import FeedbackStore
from rag import Chunk, Hit, LocalIndex, SearchConfig, load_faiss_index, rerank_hits, save_faiss_index
from app import AskRequest, FeedbackRequest, _score_retrieval, _summarize_strategy, ask, feedback as feedback_route


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")


def make_index():
    chunks = [
        Chunk("vector", "普通公司介绍", "公司概况", 1),
        Chunk("keyword", "公司的注册资本为一亿元", "发行人基本情况", 2),
        Chunk("other", "注册资本为二亿元", "另一份文件", 3, document="prospectus_2"),
    ]
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1, 0], [0, 1], [0, 1]], dtype="float32"))
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    return LocalIndex(index, chunks, FakeEncoder(), (vectorizer, matrix))


def run_immediate(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("测试协程不应等待真实网络或定时器")


class RetrievalModeTests(unittest.TestCase):
    def test_modes_and_weight_boundaries_change_first_result(self):
        local = make_index()
        vector = local.search_candidates(["注册资本"], SearchConfig("vector", top_k=3), "prospectus_1")
        fulltext = local.search_candidates(["注册资本"], SearchConfig("fulltext", top_k=3), "prospectus_1")
        zero = local.search_candidates(["注册资本"], SearchConfig("hybrid", vector_weight=0, top_k=3), "prospectus_1")
        one = local.search_candidates(["注册资本"], SearchConfig("hybrid", vector_weight=1, top_k=3), "prospectus_1")
        self.assertEqual((vector[0].chunk.chunk_id, fulltext[0].chunk.chunk_id), ("vector", "keyword"))
        self.assertEqual((zero[0].chunk.chunk_id, one[0].chunk.chunk_id), ("keyword", "vector"))

    def test_document_filter_excludes_other_prospectus(self):
        hits = make_index().search_candidates(["注册资本"], SearchConfig("fulltext", top_k=3), "prospectus_1")
        self.assertTrue(hits and all(hit.chunk.document == "prospectus_1" for hit in hits))

    def test_single_modes_skip_the_unused_retriever(self):
        local = make_index()
        class RaisingEncoder:
            def encode(self, *_args, **_kwargs): raise AssertionError("全文模式不应编码向量")
        local.encoder = RaisingEncoder()
        self.assertTrue(local.search_candidates(["注册资本"], SearchConfig("fulltext", top_k=3)))

        local = make_index()
        class RaisingVectorizer:
            def transform(self, *_args): raise AssertionError("向量模式不应计算全文分数")
        local.lexical = (RaisingVectorizer(), None)
        self.assertTrue(local.search_candidates(["注册资本"], SearchConfig("vector", top_k=3)))

    def test_invalid_search_configuration_is_rejected(self):
        for kwargs in ({"mode":"bad"},{"reranker":"bad"},{"vector_weight":-0.1},{"vector_weight":1.1},{"top_k":2},{"top_k":11}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SearchConfig(**kwargs)

    def test_faiss_unicode_path_roundtrip(self):
        index = faiss.IndexFlatIP(2); index.add(np.asarray([[1, 0]], dtype="float32"))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "中文" / "索引.faiss"; path.parent.mkdir()
            save_faiss_index(index, path); loaded = load_faiss_index(path)
        self.assertEqual(loaded.ntotal, 1)


class FeedbackStoreTests(unittest.TestCase):
    def test_votes_adjust_and_cap_score(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            store = FeedbackStore(Path(directory) / "feedback.json")
            for _ in range(20): store.record(["a"], True)
            store.record(["b"], False)
            self.assertEqual(store.adjustment("a"), .15)
            self.assertEqual(store.adjustment("b"), -.03)

    def test_corrupt_or_unwritable_store_is_safe(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            bad = Path(directory) / "bad.json"; bad.write_text("bad", encoding="utf-8")
            self.assertEqual(FeedbackStore(bad).adjustment("a"), 0)
            before = set(Path(__file__).parent.glob("*.tmp"))
            store = FeedbackStore(Path(directory)); store.record(["a"], True)
            self.assertEqual(store.adjustment("a"), 0)
            self.assertEqual(set(Path(__file__).parent.glob("*.tmp")), before)


class FakeLLM:
    def __init__(self, order=None, error=None): self.order, self.error = order, error
    async def rerank(self, _question, _hits):
        if self.error: raise self.error
        return self.order


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.hits = [Hit(Chunk("a", "公司日常经营情况", "概况", 1), .60), Hit(Chunk("b", "注册资本为一亿元", "基本情况", 2), .55)]

    def test_tfidf_and_adaptive_reranking(self):
        ranked, meta = run_immediate(rerank_hits("注册资本是多少", self.hits, SearchConfig(reranker="tfidf"), None, None))
        self.assertEqual((ranked[0].chunk.chunk_id, meta["used"]), ("b", "tfidf"))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            store = FeedbackStore(Path(directory) / "f.json")
            for _ in range(5): store.record(["a"], True); store.record(["b"], False)
            ranked, meta = run_immediate(rerank_hits("公司情况", self.hits, SearchConfig(reranker="adaptive"), store, None))
        self.assertEqual((ranked[0].chunk.chunk_id, meta["used"]), ("a", "adaptive"))

    def test_llm_order_and_invalid_fallback(self):
        ranked, meta = run_immediate(rerank_hits("问题", self.hits, SearchConfig(reranker="llm"), None, FakeLLM([1,0])))
        self.assertEqual(([x.chunk.chunk_id for x in ranked], meta["used"]), (["b","a"], "llm"))
        for llm in (FakeLLM([0,0]), FakeLLM(error=RuntimeError("timeout"))):
            ranked, meta = run_immediate(rerank_hits("注册资本是多少", self.hits, SearchConfig(reranker="llm"), None, llm))
            self.assertEqual(meta["used"], "tfidf"); self.assertTrue(meta["fallback_reason"])


class ApiAndEvaluationTests(unittest.TestCase):
    def test_request_validates_strategy_fields(self):
        request = AskRequest(question="问题", retrieval_mode="fulltext", reranker="adaptive", vector_weight=.25, top_k=6)
        self.assertEqual((request.retrieval_mode, request.top_k), ("fulltext", 6))
        for kwargs in ({"retrieval_mode":"bad"},{"reranker":"bad"},{"vector_weight":1.1},{"top_k":2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError): AskRequest(question="问题", **kwargs)

    def test_ask_route_passes_strategy(self):
        seen=[]
        class FakeRAG:
            async def ask(self, question, config): seen.append(config); return {"answer":"一亿元","citations":[],"total_ms":1}
        req=AskRequest(question="武汉兴图新科电子股份有限公司注册资本是多少？",retrieval_mode="vector",reranker="none",vector_weight=.4,top_k=5)
        with patch("app.get_rag", return_value=FakeRAG()): result=run_immediate(ask(req))
        self.assertEqual(seen[0], SearchConfig("vector","none",.4,5)); self.assertEqual(result["search_config"]["mode"],"vector")

    def test_feedback_route_records_chunks(self):
        calls=[]
        class Store:
            def record(self, ids, helpful): calls.append((ids,helpful))
        with patch("app._feedback_store", Store()): result=feedback_route(FeedbackRequest(answer_id="1",helpful=True,chunk_ids=["a","b"]))
        self.assertTrue(result["saved"]); self.assertEqual(calls,[(["a","b"],True)])

    def test_metric_helpers_and_question_count(self):
        expected=[["707.04万元","707.04"],["2019年"]]
        self.assertTrue(_score_retrieval("募集资金707.04万元",expected)); self.assertFalse(_score_retrieval("无",expected))
        summary=_summarize_strategy([{"answer_score":100,"recalled":True,"elapsed_ms":100},{"answer_score":50,"recalled":False,"elapsed_ms":300}])
        self.assertEqual(summary,{"accuracy":75.0,"recall":50.0,"average_ms":200.0,"question_count":2})
        questions=json.loads((Path(__file__).parent/"questions.json").read_text(encoding="utf-8")); self.assertEqual(len(questions),17)


class HtmlIdCollector(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=set()
    def handle_starttag(self, _tag, attrs):
        values=dict(attrs)
        if values.get("id"): self.ids.add(values["id"])


class UserInterfaceTests(unittest.TestCase):
    def test_page_exposes_strategy_and_evaluation_controls(self):
        parser=HtmlIdCollector(); parser.feed((Path(__file__).parent/"index.html").read_text(encoding="utf-8"))
        expected={"retrievalMode","reranker","vectorWeight","weightValue","topK","runRetrievalEval","retrievalSummary"}
        self.assertTrue(expected.issubset(parser.ids), expected-parser.ids)


if __name__ == "__main__": unittest.main()
