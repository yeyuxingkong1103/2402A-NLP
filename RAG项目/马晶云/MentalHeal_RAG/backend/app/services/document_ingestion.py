from datetime import datetime
import json
from pathlib import Path

from app.core.config import Settings
from app.db.session import get_session_factory
from app.ingestion.chunking import DocumentChunker
from app.ingestion.cli import embed_chunks
from app.ingestion.cleaning import DocumentCleaner
from app.ingestion.config import IngestionConfig
from app.ingestion.pipeline import PdfIngestionPipeline
from app.ingestion.text_processing import document_id
from app.memory.redis_memory import get_chat_memory
from app.models.chat import Document, IngestJob, KnowledgeChunk


# 读取入库配置，并提前创建需要用到的目录。
def get_ingestion_config(*, validate: bool = True) -> IngestionConfig:
    config = IngestionConfig.from_env()
    if validate:
        config.validate()
    for directory in (
        config.raw_dir,
        config.processed_dir,
        config.cleaned_dir,
        config.chunks_dir,
        config.vectorized_dir,
        config.failed_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return config


# 把上传的原始文件登记到关系型数据库中。
# 这里登记的是文档信息和各阶段文件路径，还没有真正把向量写入 Milvus。
def register_document(path: Path, settings: Settings) -> Document:
    now = datetime.utcnow()
    # 根据文件内容生成稳定的文档 id，方便判断是不是同一份文档。
    document_id_value = document_id(path)
    db = get_session_factory()()
    try:
        # 如果这份文档之前登记过，就直接复用记录，避免重复创建。
        document = db.query(Document).filter(Document.document_id == document_id_value).first()
        if document:
            return document
        document = Document(
            document_id=document_id_value,
            source_file=path.name,
            raw_path=str(path),
            processed_path=str(settings.processed_data_dir / f"{document_id_value}.json"),
            markdown_path=str(settings.cleaned_data_dir / f"{document_id_value}.json"),
            content_type_counts={},
            cleaning_flag_counts={},
            created_at=now,
            updated_at=now,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        return document
    finally:
        db.close()


# 创建一条入库任务记录，用来跟踪任务是排队、运行中、成功还是失败。
def create_ingest_job(document_id_value: str, operation: str, config: IngestionConfig) -> int:
    now = datetime.utcnow()
    db = get_session_factory()()
    try:
        job = IngestJob(
            job_name=f"{operation}-{document_id_value}",
            status="queued",
            raw_dir=str(config.raw_dir),
            processed_dir=str(config.processed_dir),
            chunks_dir=str(config.chunks_dir),
            embeddings_dir=str(config.vectorized_dir),
            started_at=now,
            finished_at=now,
            created_at=now,
            summary_json={"document_id": document_id_value, "operation": operation},
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return int(job.id)
    finally:
        db.close()


# 更新任务状态，并同步通知 Redis 里的文档状态。
def _update_job(job_id: int, **values: object) -> None:
    db = get_session_factory()()
    try:
        job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
        if job:
            # values 里传什么字段，就更新什么字段。
            for key, value in values.items():
                setattr(job, key, value)
            db.commit()
            if "status" in values:
                document_id_value = (job.summary_json or {}).get("document_id")
                if document_id_value:
                    get_chat_memory().set_document_status(
                        str(document_id_value),
                        str(values["status"]),
                    )
    finally:
        db.close()


# 判断同一篇文档是否已经有相同类型的任务在排队或运行，防止重复处理。
def has_active_job(document_id_value: str, operation: str) -> bool:
    db = get_session_factory()()
    try:
        return (
            db.query(IngestJob)
            .filter(
                IngestJob.job_name == f"{operation}-{document_id_value}",
                IngestJob.status.in_(["queued", "running"]),
            )
            .first()
            is not None
        )
    finally:
        db.close()


# 处理文档前半段：PDF 解析、文本清洗、切块。
# 这个函数生成 chunks/*.json，后面的 embed_document 会读取它进行向量化。
def process_document(document_id_value: str, job_id: int | None = None) -> None:
    config: IngestionConfig | None = None
    # 如果没有传任务 id，就现场创建一条“解析任务”。
    if job_id is None:
        config = get_ingestion_config()
        job_id = create_ingest_job(document_id_value, "parse", config)
    try:
        config = config or get_ingestion_config()
        _update_job(job_id, status="running", started_at=datetime.utcnow())
        db = get_session_factory()()
        try:
            document = db.query(Document).filter(Document.document_id == document_id_value).first()
            if not document:
                raise FileNotFoundError("文档不存在")
            source = Path(document.raw_path)
            # 第一步：调用 MinerU 解析 PDF，得到按页组织的文本。
            parsed = PdfIngestionPipeline(config).process_file(source)
            # 第二步：清洗解析结果，去掉无效字符、重复内容等。
            DocumentCleaner(config.chunk_size, config.chunk_overlap).clean_file(
                Path(document.processed_path), Path(document.markdown_path)
            )
            # 第三步：把清洗后的文本切成多个 chunk，保存到 chunks 目录。
            chunk_result = DocumentChunker(config.chunk_size, config.chunk_overlap).chunk_file(
                Path(document.markdown_path), config.chunks_dir / f"{document_id_value}.json"
            )
            document.page_count = int(parsed.page_count)
            document.total_text_chars = sum(len(page.mineru_text) for page in parsed.pages)
            document.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()
        _update_job(
            job_id,
            status="success",
            document_count=1,
            page_count=int(parsed.page_count),
            chunk_count=len(chunk_result.get("chunks", [])),
            note="文档解析、清洗和分块完成",
            summary_json={
                "document_id": document_id_value,
                "chunk_file": str(config.chunks_dir / f"{document_id_value}.json"),
            },
            finished_at=datetime.utcnow(),
        )
    except Exception as exc:
        # 失败时记录错误信息，并把异常继续抛出去交给上层处理。
        _update_job(
            job_id,
            status="failed",
            note=str(exc)[:2000],
            summary_json={"document_id": document_id_value, "error_type": type(exc).__name__},
            finished_at=datetime.utcnow(),
        )
        raise


# 处理文档后半段：读取 chunk 文件、生成向量、写入 Milvus，并同步写入本地数据库记录。
def embed_document(document_id_value: str, job_id: int | None = None) -> None:
    config: IngestionConfig | None = None
    # 如果没有传任务 id，就创建一条“向量化任务”。
    if job_id is None:
        config = get_ingestion_config()
        job_id = create_ingest_job(document_id_value, "embed", config)
    try:
        # 向量化阶段不一定要求 MinerU API key 有效，所以这里关闭完整配置校验。
        config = config or get_ingestion_config(validate=False)
        # 向量化的输入文件就是前面切块生成的 JSON 文件。
        chunk_file = config.chunks_dir / f"{document_id_value}.json"
        if not chunk_file.exists():
            raise FileNotFoundError("请先完成文档解析和分块")

        _update_job(job_id, status="running", started_at=datetime.utcnow())
        # 这里会进入 ingestion/cli.py：读取 chunk、生成 embedding、写入 Milvus。
        embed_chunks(config, document_id=document_id_value)
        # 同一个 chunk 文件再读取一次，用来给关系型数据库写入元数据记录。
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))

        db = get_session_factory()()
        try:
            # 先删除这篇文档旧的 KnowledgeChunk 元数据，避免重复记录。
            # 注意：这里操作的是关系型数据库，不是 Milvus 向量本身。
            db.query(KnowledgeChunk).filter(
                KnowledgeChunk.document_id == document_id_value
            ).delete()
            now = datetime.utcnow()
            # 遍历本篇文档的每个 chunk，写入数据库元数据。
            for index, chunk in enumerate(payload.get("chunks", []), start=1):
                text = str(chunk.get("text", ""))
                metadata = chunk.get("metadata", {})
                db.add(
                    KnowledgeChunk(
                        chunk_id=str(chunk.get("chunk_id", "")),
                        document_id=document_id_value,
                        source_file=Path(str(payload.get("source_path", ""))).name,
                        page_start=int(chunk.get("page_start", 0)),
                        page_end=int(chunk.get("page_end", 0)),
                        chunk_index=index,
                        chunk_type="text",
                        char_count=len(text),
                        # 这里是一个粗略 token 数估计，不是 embedding 向量本身。
                        token_estimate=max(1, len(text) // 2),
                        has_table=False,
                        quality_score=1.0,
                        milvus_collection=config.milvus_collection_knowledge,
                        knowledge_version="current",
                        text_preview=text[:512],
                        metadata_json=metadata,
                        created_at=now,
                    )
                )
            db.commit()
        finally:
            db.close()

        # Milvus 和关系型数据库都成功后，才把这个向量化任务标记为 success。
        _update_job(
            job_id,
            status="success",
            document_count=1,
            chunk_count=len(payload.get("chunks", [])),
            note="文档向量化并写入知识库完成",
            summary_json={"document_id": document_id_value},
            finished_at=datetime.utcnow(),
        )
    except Exception as exc:
        # 任意一步失败都记录任务失败原因，方便前端显示和排查问题。
        _update_job(
            job_id,
            status="failed",
            note=str(exc)[:2000],
            summary_json={"document_id": document_id_value, "error_type": type(exc).__name__},
            finished_at=datetime.utcnow(),
        )
        raise
