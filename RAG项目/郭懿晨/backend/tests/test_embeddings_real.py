from pathlib import Path

from backend.app.embeddings import BgeM3Embedder


class FakeFlagEmbeddingModel:
    def __init__(self, *args, **kwargs) -> None:
        raise OSError("missing config.json")


class FakeSentenceTransformer:
    def __init__(self, model_path, *args, **kwargs) -> None:
        self.model_path = model_path

    def encode(self, texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_bge_m3_embedder_falls_back_to_sentence_transformers(tmp_path, monkeypatch):
    model_path = tmp_path / "bge-m3"
    model_path.mkdir()
    monkeypatch.setattr("backend.app.embeddings.BGEM3FlagModel", FakeFlagEmbeddingModel, raising=False)
    monkeypatch.setattr("backend.app.embeddings.SentenceTransformer", FakeSentenceTransformer, raising=False)

    embedder = BgeM3Embedder(model_path)
    results = embedder.embed_texts(["第一段文本"])

    assert len(results) == 1
    assert results[0].dense == [0.1, 0.2, 0.3]
    assert results[0].sparse_indices
    assert results[0].sparse_values
    assert Path(embedder.model_path) == model_path
