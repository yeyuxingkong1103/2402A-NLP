import pytest

from backend.app.embeddings.embedding_factory import get_embedding_client
from backend.app.rerank.rerank_factory import get_rerank_client


@pytest.mark.gpu
def test_bge_models_load_on_cuda():
    embedding = get_embedding_client()
    reranker = get_rerank_client()

    vector = embedding.embed_texts(["离婚时孩子抚养权如何判断？"])[0]
    scores = reranker.score("抚养权", ["不满两周岁的子女，以由母亲直接抚养为原则。"])

    assert len(vector) > 100
    assert len(scores) == 1
