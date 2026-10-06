from pathlib import Path

from sqlalchemy import select

from ..deps import get_embedding, get_llm, get_milvus
from ..db.base import async_session_factory
from ..db.models import Document, MemoryTask, Message
from ..services.document_ingestor import DocumentIngestor
from ..services.memory_extractor import MemoryExtractor


async def extract_memory_task(ctx, session_id: int, user_id: int, character_id: int):
    extractor = MemoryExtractor(get_llm(), get_embedding(), get_milvus())
    async with async_session_factory() as db:
        task = await db.scalar(select(MemoryTask).where(MemoryTask.session_id == session_id))
        if task is None:
            task = MemoryTask(session_id=session_id)
            db.add(task)
            await db.flush()
        task.status = "running"
        await db.commit()

        rows = await db.scalars(
            select(Message).where(
                Message.session_id == session_id, Message.id > task.last_processed_msg_id
            ).order_by(Message.id)
        )
        conversation = [{"role": m.role, "content": m.content} for m in rows]
        if not conversation:
            task.status = "done"
            await db.commit()
            return 0

        written = await extractor.run(session_id, user_id, character_id, conversation)
        task.last_processed_msg_id = max(m.id for m in rows)
        task.status = "done"
        await db.commit()
        return written


async def ingest_document_task(ctx, document_id: int):
    milvus = get_milvus()
    await milvus.init_collections()
    ingestor = DocumentIngestor(get_embedding(), milvus)
    async with async_session_factory() as db:
        doc = await db.get(Document, document_id)
        if doc is None:
            return 0
        doc.status = "running"
        await db.commit()
        try:
            data = Path(doc.path).read_bytes()
            n = await ingestor.ingest(doc.id, doc.filename, data)
            doc.status = "done"
            doc.chunk_count = n
        except Exception as e:  # noqa: BLE001 解析失败落库供前端展示
            doc.status = "failed"
            doc.error = str(e)
        await db.commit()
        return doc.chunk_count
