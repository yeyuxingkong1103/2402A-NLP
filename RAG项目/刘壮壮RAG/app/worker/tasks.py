import asyncio

from app.config import get_settings
from app.core.embeddings import EmbeddingClient
from app.core.file_processing import chunk_text, extract_text, ocr_image
from app.core.llm import LLMClient
from app.core.memory import ShortTermMemory
from app.core.redis_client import get_redis
from app.core.vector_store import VectorStore
from app.core.vision import VisionEmbeddingClient
from app.database import SessionLocal
from app.models.knowledge_file import KnowledgeFile

MEMORY_EXTRACT_PROMPT = (
    "从以下对话中提取值得长期记住的重要事实（例如用户身份、偏好、重要结论），"
    "最多 5 条，每条一行。若无值得记忆的内容，输出空字符串。\n\n{conversation}"
)


def build_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(
        uri=settings.milvus_uri,
        text_dim=settings.text_embedding_dim,
        image_dim=settings.image_embedding_dim,
    )


async def process_file(ctx: dict, file_id: int) -> None:
    settings = get_settings()
    async with SessionLocal() as db:
        f = await db.get(KnowledgeFile, file_id)
        if f is None:
            return
        f.status = "processing"
        await db.commit()

        store = build_vector_store()
        try:
            if f.file_type == "image":
                ocr_text = await asyncio.to_thread(ocr_image, f.file_path)
                if ocr_text:
                    chunks = chunk_text(ocr_text)
                    embedder = EmbeddingClient(
                        model=settings.embedding_model,
                        base_url=settings.embedding_base_url,
                        api_key=settings.embedding_api_key,
                    )
                    vectors = await embedder.embed(chunks)
                    await asyncio.to_thread(
                        store.insert_text_chunks,
                        f.user_id,
                        f.character_id,
                        f.id,
                        chunks,
                        vectors,
                    )
                vision = VisionEmbeddingClient(
                    model=settings.vision_embedding_model,
                    base_url=settings.vision_embedding_base_url,
                    api_key=settings.vision_embedding_api_key,
                )
                vec = await vision.embed_image(f.file_path)
                await asyncio.to_thread(
                    store.insert_image_vector,
                    f.user_id,
                    f.character_id,
                    f.id,
                    f.file_path,
                    vec,
                )
            else:
                text = await asyncio.to_thread(extract_text, f.file_path, f.file_type)
                chunks = chunk_text(text)
                embedder = EmbeddingClient(
                    model=settings.embedding_model,
                    base_url=settings.embedding_base_url,
                    api_key=settings.embedding_api_key,
                )
                vectors = await embedder.embed(chunks)
                await asyncio.to_thread(
                    store.insert_text_chunks,
                    f.user_id,
                    f.character_id,
                    f.id,
                    chunks,
                    vectors,
                )

            f.status = "done"
            await db.commit()
        except Exception:
            f.status = "failed"
            await db.commit()
            raise


async def extract_memory(
    ctx: dict, conversation_id: int, character_id: int, user_id: int
) -> None:
    settings = get_settings()
    memory = ShortTermMemory(get_redis())
    history = await memory.get(conversation_id)
    if not history:
        return

    conversation_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-12:]
    )
    prompt = MEMORY_EXTRACT_PROMPT.format(conversation=conversation_text)

    llm = LLMClient(
        model="gpt-4o-mini",
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0.2,
    )
    reply = ""
    async for piece in llm.chat_stream([{"role": "user", "content": prompt}]):
        reply += piece

    facts = [line.strip("- ").strip() for line in reply.splitlines() if line.strip()]
    facts = [f for f in facts if f][:5]
    if not facts:
        return

    embedder = EmbeddingClient(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )
    store = VectorStore(
        uri=settings.milvus_uri,
        text_dim=settings.text_embedding_dim,
        image_dim=settings.image_embedding_dim,
    )
    vectors = await embedder.embed(facts)
    for fact, vec in zip(facts, vectors):
        await asyncio.to_thread(store.insert_memory, user_id, character_id, fact, vec)
