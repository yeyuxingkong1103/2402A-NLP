from datetime import datetime

import pytest
from pymilvus import MilvusClient

from backend.app.core.config import AppSettings
from backend.app.database.milvus import MilvusVectorStore, check_milvus_readiness
from backend.app.embeddings.embedding_factory import get_embedding_client
from backend.app.ingestion.index_writer import index_materials
from backend.app.models.knowledge_base import KnowledgeMaterial
from backend.app.rag.hybrid_retriever import retrieve_candidates
from backend.app.rag.pipeline import rerank_and_select
from backend.app.rag.result_merger import RetrievalFilters
from backend.app.rerank.rerank_factory import get_rerank_client
from backend.app.services.chat_service import ChatService


class FakeLlm:
    def __init__(self):
        self.requests = []

    async def stream_chat(self, request):
        self.requests.append(request)
        yield "根据已发布司法解释，离婚案件中子女抚养应以未成年子女利益为原则，具体结合年龄、照护情况和双方条件判断。"


@pytest.mark.gpu
def test_indexes_materials_into_milvus_and_answers_with_real_rag():
    settings = AppSettings(MILVUS_URI="http://localhost:19530", MILVUS_COLLECTION="myrag_rag_e2e_20260918", MILVUS_VECTOR_DIMENSION=1024)
    ok, detail = check_milvus_readiness(settings)
    if not ok:
        pytest.skip(f"Milvus 不可用：{detail}")
    client = MilvusClient(uri=settings.MILVUS_URI, timeout=5)
    if client.has_collection(settings.MILVUS_COLLECTION):
        client.drop_collection(settings.MILVUS_COLLECTION)
    try:
        material = KnowledgeMaterial(
            id="mat-custody-1",
            snapshot_id="snap-custody-1",
            source_url="https://example.test/marriage-family/custody",
            publisher="最高人民法院",
            material_type="judicial_interpretation",
            raw_text="离婚案件中，涉及未成年子女抚养问题时，应当以未成年子女利益最大化为原则，结合子女年龄、实际照护情况、父母双方抚养能力和生活环境综合判断。",
            attachments=[],
            status="published",
            searchable=True,
            created_at=datetime(2026, 9, 18),
            updated_at=datetime(2026, 9, 18),
            effective_from="2021-01-01",
        )
        setattr(material, "version_id", "v-custody-1")
        setattr(material, "relationship_types", ["general"])
        setattr(material, "article", "子女抚养规则")
        index_materials([material], embedding_client=get_embedding_client(), vector_store=MilvusVectorStore(settings.MILVUS_URI, settings.MILVUS_COLLECTION, timeout_seconds=5), app_settings=settings)
        client.flush(settings.MILVUS_COLLECTION)
        client.load_collection(settings.MILVUS_COLLECTION)

        async def rag_decider(text: str):
            candidates = retrieve_candidates(
                text,
                RetrievalFilters(materials=[material]),
                embedding_client=get_embedding_client(),
                vector_store=MilvusVectorStore(settings.MILVUS_URI, settings.MILVUS_COLLECTION, timeout_seconds=5),
            )
            return rerank_and_select(text, candidates, get_rerank_client(), filters=RetrievalFilters(materials=[material]), threshold=-100.0)

        llm = FakeLlm()
        service = ChatService(rag_decider=rag_decider, llm_client=llm)

        import asyncio

        result = asyncio.run(service.send_message("u-rag-e2e", "c-rag-e2e", "我在上海，已婚，有孩子，想离婚并争取抚养权，无危险，抚养权如何判断？"))

        assert result.status == "answered"
        assert result.citations
        assert result.citations[0]["material_id"] == "mat-custody-1"
        assert llm.requests
        assert "未成年子女利益" in llm.requests[0].messages[1]["content"]
    finally:
        if client.has_collection(settings.MILVUS_COLLECTION):
            client.drop_collection(settings.MILVUS_COLLECTION)
