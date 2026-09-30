from unittest.mock import Mock

from src.edu_rag_ingest.retrieval.milvus_store import MilvusChunkStore


def test_existing_chunk_ids_reads_ids_in_batches():
    store = object.__new__(MilvusChunkStore)
    store.config = Mock()
    store.client = Mock()
    store.client.get.side_effect = [
        [{"id": "a"}, {"id": "b"}],
        [{"id": "c"}],
    ]

    ids = [str(index) for index in range(101)]
    result = store.existing_chunk_ids(ids)

    assert result == {"a", "b", "c"}
    assert store.client.get.call_count == 2
