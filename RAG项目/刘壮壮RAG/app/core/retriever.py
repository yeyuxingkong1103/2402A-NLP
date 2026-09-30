import asyncio

from app.config import get_settings
from app.core.embeddings import EmbeddingClient
from app.core.vector_store import VectorStore


def _build_embedder() -> EmbeddingClient:
    settings = get_settings()
    return EmbeddingClient(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )


def build_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(
        uri=settings.milvus_uri,
        text_dim=settings.text_embedding_dim,
        image_dim=settings.image_embedding_dim,
    )


async def search_knowledge(character_id: int, query: str, top_k: int = 4) -> list[str]:
    embedder = _build_embedder()
    store = build_vector_store()
    vec = (await embedder.embed([query]))[0]
    hits = await asyncio.to_thread(store.search_text, character_id, vec, top_k)
    return [h["text"] for h in hits]


async def search_memory(character_id: int, query: str, top_k: int = 4) -> list[str]:
    embedder = _build_embedder()
    store = build_vector_store()
    vec = (await embedder.embed([query]))[0]
    hits = await asyncio.to_thread(store.search_memory, character_id, vec, top_k)
    return [h["text"] for h in hits]


async def retrieve_context(
    character_id: int, query: str, top_k: int = 4
) -> tuple[list[str], list[str]]:
    knowledge, memory = await asyncio.gather(
        search_knowledge(character_id, query, top_k),
        search_memory(character_id, query, top_k),
    )
    return knowledge, memory
