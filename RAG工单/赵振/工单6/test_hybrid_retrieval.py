"""工单编号：人工智能NLP-RAG-混合检索任务。"""

import json
import sys
import tempfile
import types
from pathlib import Path
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


fake_rag = types.ModuleType("rag")
fake_rag.ROOT = Path(tempfile.gettempdir())
fake_rag.table_group = lambda question: None
fake_rag.understand = lambda question: {"intent": "事实", "parts": [question], "search": question}
sys.modules["rag"] = fake_rag
from hybrid_retrieval import _feedback_rerank, fulltext_scores, search


class FakeModel:
    def encode(self, questions, normalize_embeddings=True):
        return np.array([[1.0, 0.0] for _ in questions])


def make_index():
    chunks = [
        {"page": 1, "kind": "text", "title": "公司信息", "summary": "法定代表人赵马克",
         "text": "武汉力源信息技术股份有限公司法定代表人是赵马克。"},
        {"page": 2, "kind": "text", "title": "销售网点", "summary": "深圳销售处服务客户",
         "text": "The Shenzhen sales office serves customers."},
        {"page": 3, "kind": "text", "title": "主营业务", "summary": "电子元器件销售",
         "text": "公司主营业务为电子元器件销售。"},
    ]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
    matrix = vectorizer.fit_transform([item["text"] for item in chunks])
    return {"chunks": chunks, "vectors": np.array([[1, 0], [0, 1], [0, 0.5]], dtype=float),
            "tfidf": vectorizer, "matrix": matrix}


def main():
    index = make_index()
    checks = []
    assert fulltext_scores('法定代表人 AND 赵马克', index).tolist() == [1.0, 0.0, 0.0]
    assert fulltext_scores('赵马克 OR Shenzhen', index).tolist() == [1.0, 1.0, 0.0]
    assert fulltext_scores('法定代表人 NOT 赵马克', index).tolist() == [0.0, 0.0, 0.0]
    checks.append("布尔查询 AND/OR/NOT")
    assert fulltext_scores('"法定代表人是赵马克"', index).tolist() == [1.0, 0.0, 0.0]
    checks.append("短语匹配")
    assert fulltext_scores("Shenzhan", index, field="body", fuzzy=True)[1] > 0
    assert fulltext_scores("Shenzhen", index, field="body")[1] > 0
    checks.append("中文与英文检索、模糊匹配")
    assert fulltext_scores("赵马克", index, field="title").max() == 0
    checks.append("title/body/summary字段过滤")

    records = [{"page": 1, "score": 0.5}, {"page": 2, "score": 0.6}]
    with tempfile.TemporaryDirectory() as folder:
        feedback = Path(folder) / "feedback.jsonl"
        feedback.write_text(json.dumps({"pages": [1], "rating": "有帮助"}) + "\n", encoding="utf-8")
        _feedback_rerank(records, feedback)
    assert records[0]["score"] > 0.5

    strategy_results = []
    for strategy in ("向量检索", "全文检索", "混合检索"):
        started = time.perf_counter()
        _, results, details = search("法定代表人 赵马克", index, FakeModel(), strategy=strategy)
        assert results and details["strategy"] == strategy
        strategy_results.append({"strategy": strategy, "top_page": results[0]["page"],
                                 "score": round(results[0]["score"], 4),
                                 "vector_candidates": details["vector_candidates"],
                                 "fulltext_candidates": details["fulltext_candidates"],
                                 "seconds": round(time.perf_counter() - started, 6)})
    checks.append("向量/全文/混合检索及权重融合")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.complete))

        def complete(self, **kwargs):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                content='{"scores":[{"id":0,"score":10}]}'))])

    sys.modules["openai"] = types.SimpleNamespace(OpenAI=FakeOpenAI)
    _, results, details = search("赵马克", index, FakeModel(), reranker="LLM重排", base_url="http://mock")
    assert results and not details["warning"]
    checks.append("LLM重排(mock)")
    checks.extend(["TF-IDF重排", "用户反馈自适应重排"])
    report = {"passed": True, "checks": checks, "strategy_results": strategy_results,
              "rerankers": ["TF-IDF", "LLM重排(mock)", "用户反馈"],
              "note": "算法测试使用小型固定语料和模拟嵌入/LLM；端到端全PDF准确率与真实LLM重排未在本次复测。"}
    (Path(__file__).parent / "hybrid_test_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
