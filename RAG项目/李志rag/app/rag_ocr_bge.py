"""RAG 离线主流程：文件解析、文本切块、BGE-M3 向量化和 Milvus 入库。

答辩时从 ``build_index`` 自上而下阅读即可看到完整离线数据流。本文件只负责编排，
具体 OCR、PDF 解析、文本处理和向量库细节分别放在同目录的小模块中，便于理解。
"""

import json
import logging
import re
import uuid
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Document, DocumentCatalog
from app import rag_ocr_models, rag_pdf_parser, rag_text_processing, rag_vector_store


settings = get_settings()
logger = logging.getLogger(__name__)
# BGE-M3 与在线 BGE-reranker 共用加载锁，避免两个大模型同时首次初始化。
_model_lock = rag_vector_store._model_lock

# 对外统一导出底层能力，使旧代码仍可从 rag_ocr_bge 或 rag_main 使用原函数名。
TextChunk = rag_text_processing.TextChunk
clean_pdf_text = rag_text_processing.clean_pdf_text
clean_text = rag_text_processing.clean_text
remove_repeated_watermarks = rag_text_processing.remove_repeated_watermarks
chunk_text = rag_text_processing.chunk_text

ocr_engine = rag_ocr_models.ocr_engine
vl_components = rag_ocr_models.vl_components
_ocr_file = rag_ocr_models._ocr_file
_vl_image = rag_ocr_models._vl_image
_clean_vl_output = rag_ocr_models._clean_vl_output

_render_region = rag_pdf_parser._render_region
_find_tables = rag_pdf_parser._find_tables
_looks_like_chart = rag_pdf_parser._looks_like_chart
_visual_task = rag_pdf_parser._visual_task
# 代理运行时可能临时替换模块属性，因此先保存永远指向底层实现的函数对象。
_parser_table_text = rag_pdf_parser._table_text
_parser_extract_pdf_text = rag_pdf_parser.extract_pdf_text

embedding_model = rag_vector_store.embedding_model
encode_documents = rag_vector_store.encode_documents
milvus_client = rag_vector_store.milvus_client
ensure_collection = rag_vector_store.ensure_collection
delete_document = rag_vector_store.delete_document


def _safe_vl(path: Path, task: str) -> str:
    """视觉识别兼容入口，允许旧测试替换本模块的 ``_vl_image``。"""
    original = rag_ocr_models._vl_image
    rag_ocr_models._vl_image = _vl_image
    try:
        return rag_ocr_models._safe_vl(path, task)
    finally:
        rag_ocr_models._vl_image = original


def _table_text(page, folder: Path, tables: list, use_visual: bool) -> str:
    """表格解析兼容入口，实际实现位于 ``rag_pdf_parser``。"""
    original = rag_pdf_parser._safe_vl
    rag_pdf_parser._safe_vl = _safe_vl
    try:
        return _parser_table_text(page, folder, tables, use_visual)
    finally:
        rag_pdf_parser._safe_vl = original


def extract_pdf_text(path: Path, use_visual: bool = True) -> str:
    """第 1 步：提取 PDF 原文、表格、图片文字，清洗并去除重复水印。"""
    original_safe = rag_pdf_parser._safe_vl
    original_table = rag_pdf_parser._table_text
    rag_pdf_parser._safe_vl = _safe_vl
    rag_pdf_parser._table_text = _table_text
    try:
        return _parser_extract_pdf_text(path, use_visual)
    finally:
        rag_pdf_parser._safe_vl = original_safe
        rag_pdf_parser._table_text = original_table


def build_index(
    document: Document,
    database: Session,
    use_visual: bool = False,
    mark_processing: bool = True,
) -> Document:
    """执行离线流程并把可检索结果写入 Milvus。

    1. PyMuPDF、PP-OCRv6、PaddleOCR-VL 抽取和清洗 PDF。
    2. 按段落、句子以及滑动窗口切成文本块。
    3. BGE-M3 生成稠密向量，Milvus 同时自动生成 BM25 稀疏向量。
    4. 首次索引用 insert；视觉增强用稳定主键 upsert 覆盖基础文本块。
    """
    if mark_processing:
        document.status, document.error_message = "processing", "正在解析 PDF"
        database.commit()
    try:
        # 第 1 步：抽取文字、表格和图片语义，并执行清洗、文字去水印。
        logger.info("文档 %s 开始解析：%s", document.id, document.filename)
        extracted_text = extract_pdf_text(Path(document.stored_path), use_visual)
        logger.info("文档 %s 解析完成，共 %s 个字符", document.id, len(extracted_text))

        # 第 2 步：分块长度和重叠量来自 .env 对应的统一配置。
        if mark_processing:
            document.error_message = "正在清洗和分块"
            database.commit()
        chunks = chunk_text(extracted_text, settings.chunk_size, settings.chunk_overlap)

        # 第 3 步：同一批文本块按顺序得到一一对应的 BGE-M3 向量。
        if mark_processing:
            document.error_message = f"正在使用 BGE-M3 向量化（{len(chunks)} 个分块）"
            database.commit()
        vectors = encode_documents([chunk.text for chunk in chunks])
        logger.info("文档 %s 向量化完成，共 %s 个分块", document.id, len(chunks))

        # 第 4 步：组装 Milvus 字段；sparse_vector 由 Milvus BM25 Function 自动生成。
        if mark_processing:
            document.error_message = "正在写入 Milvus 向量数据库"
            database.commit()
            delete_document(document.id)
        created_at = int(document.created_at.timestamp())
        updated_at = int(document.updated_at.timestamp())
        rows = [
            {
                # UUID5 由“文档 ID + 块序号”稳定生成，视觉增强时可以覆盖同一块。
                "id": uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}:{chunk.index}").hex,
                "document_id": document.id,
                "owner_id": document.owner_id,
                "role_id": document.role_id,
                "is_public": document.is_public,
                "source": document.source_url or document.filename,
                "summary": chunk.summary,
                "created_at": created_at,
                "updated_at": updated_at,
                "text": chunk.text,
                "dense_vector": vector,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if not rows:
            raise ValueError("文档分块结果为空")
        if mark_processing:
            milvus_client().insert(settings.milvus_collection, rows)
        else:
            milvus_client().upsert(settings.milvus_collection, rows)
        milvus_client().flush(settings.milvus_collection)

        # Milvus 成功后更新 MySQL 管理状态，文档此时已经可以被在线流程检索。
        document.status = "ready"
        document.chunk_count = len(rows)
        document.summary = rows[0]["summary"]
        document.error_message = ""
        database.commit()
        logger.info("文档 %s 索引完成，可检索分块数：%s", document.id, len(rows))

        # 首页分类只是辅助功能；失败不能撤销已经成功写入的知识库向量。
        try:
            update_document_catalog(document, "\n\n".join(chunk.text for chunk in chunks), database)
        except Exception:
            database.rollback()
            logger.exception("文档 %s 目录生成失败，但基础索引仍可检索", document.id)
    except Exception as error:
        if mark_processing:
            document.status, document.chunk_count = "failed", 0
            document.error_message = str(error)[:1000]
            database.commit()
        raise
    database.commit()
    database.refresh(document)
    return document


def _fallback_catalog(filename: str, text: str) -> dict:
    """DeepSeek 不可用时，根据正文开头生成首页标题和快捷问题。"""
    clean_name = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 4]
    title = (lines[0] if lines else clean_name or "未命名文档")[:200]
    return {
        "category": "其他资料",
        "title": title,
        "summary": " ".join(lines[:3])[:500],
        "questions": [f"{title}主要讲了什么？", f"{title}有哪些重要内容？", f"请总结{title}。"],
    }


def _parse_json(content: str) -> dict:
    """清洗并限制 DeepSeek 返回的目录 JSON 字段。"""
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        raise ValueError("模型未返回 JSON")
    data = json.loads(match.group())
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    return {
        "category": str(data.get("category", "其他资料"))[:100],
        "title": str(data.get("title", "未命名文档"))[:200],
        "summary": str(data.get("summary", ""))[:1000],
        "questions": [str(item)[:200] for item in questions[:5] if str(item).strip()],
    }


def classify_document(filename: str, text: str) -> dict:
    """为知识库首页生成分类、标题、摘要和快捷问题。"""
    fallback = _fallback_catalog(filename, text)
    if not settings.llm_enabled or not settings.deepseek_api_key:
        return fallback
    prompt = (
        "分析下面的文档内容，为知识库首页生成导航信息。文档可以属于任何领域。"
        "只返回 JSON，不要使用 Markdown。格式："
        "{'category':'简短分类','title':'准确标题','summary':'一句话摘要',"
        "'questions':['可由文档回答的问题1','问题2','问题3']}。"
        "分类不超过10个汉字，问题必须能从文档中找到答案，不得编造。\n\n"
        f"文件名：{filename}\n文档内容：\n{text[:12000]}"
    )
    try:
        with httpx.Client(timeout=90) as client:
            response = client.post(
                f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                json={
                    "model": settings.deepseek_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
        result = _parse_json(response.json()["choices"][0]["message"]["content"])
        return result if result["questions"] else fallback
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


def update_document_catalog(document: Document, text: str, database: Session) -> None:
    """新增或更新 MySQL 中与文档一一对应的首页导航记录。"""
    data = classify_document(document.filename, text)
    catalog = database.scalar(
        select(DocumentCatalog).where(DocumentCatalog.document_id == document.id)
    )
    if not catalog:
        catalog = DocumentCatalog(document_id=document.id, title=data["title"])
        database.add(catalog)
    catalog.category = data["category"]
    catalog.title = data["title"]
    catalog.summary = data["summary"]
    catalog.questions_json = json.dumps(data["questions"], ensure_ascii=False)
