from backend.app.embeddings import EmbeddingResult, FakeBgeM3Embedder


def test_torch_out_of_memory_alias_is_available():
    import backend.app.embeddings as embeddings_module

    if embeddings_module.torch is None:
        return

    assert hasattr(embeddings_module.torch.cuda, "OutOfMemoryError")
    assert hasattr(embeddings_module.torch, "OutofMemoryError")
    assert embeddings_module.torch.OutofMemoryError is embeddings_module.torch.cuda.OutOfMemoryError


def test_fake_embedder_returns_dense_and_sparse():
    embedder = FakeBgeM3Embedder(size=3)
    result = embedder.embed_texts(["第一段", "第二段"])

    assert len(result) == 2
    assert result[0].dense == [1.0, 0.0, 0.0]
    assert result[0].sparse_indices == [0]
    assert result[0].sparse_values == [1.0]


def test_embedding_result_has_named_vectors():
    item = EmbeddingResult(dense=[0.1, 0.2], sparse_indices=[1], sparse_values=[0.5])

    assert item.to_qdrant_vectors() == {
        "dense": [0.1, 0.2],
        "sparse": {"indices": [1], "values": [0.5]},
    }


def test_embed_texts_returns_empty_for_empty_input():
    embedder = FakeBgeM3Embedder(size=3)

    assert embedder.embed_texts([]) == []


def test_fake_embedder_sparse_indices_are_unique_within_a_text():
    embedder = FakeBgeM3Embedder(size=3)

    result = embedder.embed_texts(["重复重复"])[0]

    assert len(result.sparse_indices) == len(set(result.sparse_indices))
