from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.core.database import SessionLocal
from backend.app.models.entities import Document, DocumentChunk, IngestTask
from vectorstores.milvus_store import MilvusStore
from embedding.bge_m3_embedder import get_embedding_service
from parsing.document_parser import DocumentParser, save_json
from cleaning.document_cleaner import DocumentCleaner
from chunking.legal_chunker import LegalChunker


class IngestService:
    """负责完整的文档解析、清洗、分块、向量化和入库流水线。"""

    def __init__(self) -> None:
        # 创建文档解析器，优先远程 MinerU，失败可降级本地文本解析。
        self.parser = DocumentParser()
        # 创建清洗器，用于去除页码、重复页眉页脚和明显噪声。
        self.cleaner = DocumentCleaner()
        # 创建法律结构化分块器，用于按章节和条款切分。
        self.chunker = LegalChunker()
        # 创建本地 BGE-M3 向量服务。
        self.embedding_service = get_embedding_service()
        # 创建 Milvus 写入服务。
        self.milvus = MilvusStore()

    async def run(self, task_id: int) -> None:
        # 后台任务不能复用请求里的 db，所以这里创建新的数据库会话。
        db = SessionLocal()
        try:
            # 查询任务记录。
            task = db.get(IngestTask, task_id)
            # 如果任务不存在，直接结束。
            if not task:
                return
            # 查询任务关联的文档。
            document = db.get(Document, task.document_id)
            # 如果文档不存在，标记任务失败。
            if not document:
                self._mark_failed(db, task, None, "文档不存在")
                return
            # 标记开始解析。
            self._update_step(db, task, document, "processing", "正在解析文档")
            # 执行文档解析。
            parsed_blocks = await self.parser.parse(Path(document.file_path), document.id)
            # 保存解析结果到 data/parsed。
            save_json(Path("data/parsed") / f"document_{document.id}.json", parsed_blocks)
            # 标记开始清洗。
            self._update_step(db, task, document, "processing", "正在清洗文档")
            # 执行清洗。
            cleaned_blocks = self.cleaner.clean(parsed_blocks)
            # 保存清洗结果到 data/cleaned。
            save_json(Path("data/cleaned") / f"document_{document.id}.json", cleaned_blocks)
            # 标记开始分块。
            self._update_step(db, task, document, "processing", "正在结构化分块")
            # 执行法律结构化分块。
            chunks = self.chunker.chunk(cleaned_blocks, document.id, document.user_id, document.knowledge_base_id)
            # 保存分块结果到 data/chunks。
            save_json(Path("data/chunks") / f"document_{document.id}.json", chunks)
            # 标记开始向量化。
            self._update_step(db, task, document, "processing", "正在生成向量")
            # 提取每个分块的正文。
            texts = [chunk["content"] for chunk in chunks]
            # 使用 BGE-M3 批量生成向量。
            vectors = self.embedding_service.encode(texts)
            # 保存向量化记录到 data/embeddings，只存编号和维度，避免文件过大。
            save_json(
                Path("data/embeddings") / f"document_{document.id}.json",
                [{"chunk_uid": chunk["chunk_uid"], "dim": len(vector)} for chunk, vector in zip(chunks, vectors)],
            )
            # 标记开始写入 MySQL。
            self._update_step(db, task, document, "processing", "正在写入 MySQL")
            # 把分块元数据和全文写入 MySQL。
            self._save_chunks_to_mysql(db, chunks)
            # 标记开始写入 Milvus。
            self._update_step(db, task, document, "processing", "正在写入 Milvus")
            # 把分块文本、分块向量和必要元数据一起写入 Milvus。
            self.milvus.insert_chunks(chunks, vectors, document.filename)
            # 更新文档分块数量。
            document.chunk_count = len(chunks)
            # 标记全部完成。
            self._update_step(db, task, document, "done", "处理完成")
        except Exception as exc:
            # 捕获所有异常，保证任务状态能展示给前端。
            task = db.get(IngestTask, task_id)
            # 查询文档用于同步状态。
            document = db.get(Document, task.document_id) if task else None
            # 写入失败状态和错误消息。
            self._mark_failed(db, task, document, str(exc))
        finally:
            # 关闭数据库会话。
            db.close()

    def _save_chunks_to_mysql(self, db: Session, chunks: list[dict]) -> None:
        # 遍历所有分块，逐条写入 MySQL。
        for chunk in chunks:
            # 创建 ORM 对象。
            db.add(DocumentChunk(**chunk))
        # 提交事务，让分块记录真正保存。
        db.commit()

    def _update_step(self, db: Session, task: IngestTask, document: Document, status: str, step: str) -> None:
        # 更新任务状态。
        task.status = status
        # 更新当前步骤说明，前端会展示给用户。
        task.current_step = step
        # 同步文档状态。
        document.status = status
        # 提交数据库事务。
        db.commit()

    def _mark_failed(self, db: Session, task: IngestTask | None, document: Document | None, error: str) -> None:
        # 如果任务存在，写入失败状态。
        if task:
            task.status = "failed"
            task.current_step = "处理失败"
            task.error_message = error
        # 如果文档存在，也同步失败状态。
        if document:
            document.status = "failed"
            document.error_message = error
        # 提交失败信息，方便前端查看原因。
        db.commit()
