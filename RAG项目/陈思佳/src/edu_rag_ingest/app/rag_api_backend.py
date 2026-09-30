from __future__ import annotations

"""RAG 后端 HTTP API 入口：提供问答、文档、会话、教案和试题接口。"""

import json
from functools import lru_cache
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config.config import load_config
from ..storage.content_repository import (
    Database,
    GeneratedQuestionSet,
    LessonPlan,
    serialize_lesson_plan,
    serialize_question_set,
)
from ..ingestion.document_service import DocumentService
from ..ingestion.document_tasks import DocumentTaskRunner
from ..generation.education_generation import EducationGenerationService
from ..storage.memory_store import MemoryStore
from ..generation.rag_service import TeacherRAGService

app = FastAPI(title="九年级语文教师 RAG API", version="0.1.0")
document_tasks = DocumentTaskRunner(max_workers=1)
database_init_lock = Lock()


class ChatRequest(BaseModel):
    """普通或流式聊天接口的请求参数。"""
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    subject: str | None = Field(default=None, max_length=64)
    grade: str | None = Field(default=None, max_length=64)
    chapter: str | None = Field(default=None, max_length=128)
    document_type: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    """普通聊天接口返回的答案和引用。"""
    session_id: str
    answer: str
    citations: list[dict]


class SessionCreateRequest(BaseModel):
    """创建 Redis 会话时使用的请求参数。"""
    title: str = "新对话"


@lru_cache(maxsize=1)
def get_service() -> TeacherRAGService:
    """按进程缓存教师 RAG 服务，避免每次请求重复加载模型和 Milvus。"""
    config = load_config("configs/crawler.yaml")
    return TeacherRAGService(config)


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    """按进程缓存 Redis 记忆存储客户端。"""
    config = load_config("configs/crawler.yaml")
    return MemoryStore(config.memory)


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    """按进程缓存文档上传和入库服务。"""
    config = load_config("configs/crawler.yaml")
    return DocumentService(config)


class LessonPlanRequest(BaseModel):
    subject: str = Field(default="语文", min_length=1, max_length=64)
    grade: str = Field(default="九年级", min_length=1, max_length=64)
    chapter: str = Field(min_length=1, max_length=128)
    lesson_hours: int = Field(default=2, ge=1, le=8)
    requirements: str = Field(default="", max_length=2000)


class QuestionGenerationRequest(BaseModel):
    subject: str = Field(default="语文", min_length=1, max_length=64)
    grade: str = Field(default="九年级", min_length=1, max_length=64)
    chapter: str = Field(min_length=1, max_length=128)
    knowledge_point: str = Field(min_length=1, max_length=128)
    question_type: str = Field(default="选择题", min_length=1, max_length=64)
    question_count: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="中等", min_length=1, max_length=32)


class LessonPlanSaveRequest(LessonPlanRequest):
    title: str = Field(default="未命名教案", max_length=255)
    content: str = Field(min_length=1)
    citations: list[dict] = Field(default_factory=list)


class QuestionSetSaveRequest(QuestionGenerationRequest):
    content: str = Field(min_length=1)
    citations: list[dict] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_database() -> Database:
    """创建数据库连接并确保教案、试题表已经存在。"""
    config = load_config("configs/crawler.yaml")
    database = Database(config.database.url, config.database.echo)
    with database_init_lock:
        database.create_tables()
    return database


@lru_cache(maxsize=1)
def get_generation_service() -> EducationGenerationService:
    """按进程缓存教案和试题生成服务。"""
    config = load_config("configs/crawler.yaml")
    return EducationGenerationService(config)


@app.post("/api/lesson-plans/generate")
def generate_lesson_plan(request: LessonPlanRequest) -> dict:
    try:
        result = get_generation_service().generate_lesson_plan(
            request.subject,
            request.grade,
            request.chapter,
            request.lesson_hours,
            request.requirements,
        )
        return {"content": result.content, "citations": result.citations}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"教案生成失败：{error}") from error


@app.post("/api/questions/generate")
def generate_questions(request: QuestionGenerationRequest) -> dict:
    try:
        result = get_generation_service().generate_questions(
            request.subject,
            request.grade,
            request.chapter,
            request.knowledge_point,
            request.question_type,
            request.question_count,
            request.difficulty,
        )
        return {"content": result.content, "citations": result.citations}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"试题生成失败：{error}") from error


@app.post("/api/questions/generate")
def generate_questions(request: QuestionGenerationRequest) -> dict:
    try:
        result = get_generation_service().generate_questions(
            request.subject,
            request.grade,
            request.chapter,
            request.knowledge_point,
            request.question_type,
            request.question_count,
            request.difficulty,
        )
        return {"content": result.content, "citations": result.citations}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"试题生成失败：{error}") from error


@app.post("/api/lesson-plans")
def save_lesson_plan(request: LessonPlanSaveRequest) -> dict:
    database = get_database()
    with database.session() as session:
        item = LessonPlan(
            subject=request.subject,
            grade=request.grade,
            chapter=request.chapter,
            title=request.title,
            content=request.content,
            citations_json=json.dumps(request.citations, ensure_ascii=False),
        )
        session.add(item)
        session.commit()
        return serialize_lesson_plan(item, request.citations)


@app.get("/api/lesson-plans")
def list_lesson_plans() -> dict[str, list[dict]]:
    database = get_database()
    with database.session() as session:
        items = session.query(LessonPlan).order_by(LessonPlan.updated_at.desc()).all()
        return {"lesson_plans": [serialize_lesson_plan(item, json.loads(item.citations_json)) for item in items]}


@app.get("/api/lesson-plans/{item_id}")
def get_lesson_plan(item_id: int) -> dict:
    database = get_database()
    with database.session() as session:
        item = session.get(LessonPlan, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="教案不存在")
        return serialize_lesson_plan(item, json.loads(item.citations_json))


@app.delete("/api/lesson-plans/{item_id}")
def delete_lesson_plan(item_id: int) -> dict[str, str]:
    database = get_database()
    with database.session() as session:
        item = session.get(LessonPlan, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="教案不存在")
        session.delete(item)
        session.commit()
        return {"status": "deleted"}


@app.post("/api/questions")
def save_question_set(request: QuestionSetSaveRequest) -> dict:
    database = get_database()
    with database.session() as session:
        item = GeneratedQuestionSet(
            subject=request.subject,
            grade=request.grade,
            chapter=request.chapter,
            knowledge_point=request.knowledge_point,
            question_type=request.question_type,
            difficulty=request.difficulty,
            content=request.content,
            citations_json=json.dumps(request.citations, ensure_ascii=False),
        )
        session.add(item)
        session.commit()
        return serialize_question_set(item, request.citations)


@app.get("/api/questions")
def list_question_sets() -> dict[str, list[dict]]:
    database = get_database()
    with database.session() as session:
        items = session.query(GeneratedQuestionSet).order_by(GeneratedQuestionSet.updated_at.desc()).all()
        return {"questions": [serialize_question_set(item, json.loads(item.citations_json)) for item in items]}


@app.get("/api/questions/{item_id}")
def get_question_set(item_id: int) -> dict:
    database = get_database()
    with database.session() as session:
        item = session.get(GeneratedQuestionSet, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="试题不存在")
        return serialize_question_set(item, json.loads(item.citations_json))


@app.delete("/api/questions/{item_id}")
def delete_question_set(item_id: int) -> dict[str, str]:
    database = get_database()
    with database.session() as session:
        item = session.get(GeneratedQuestionSet, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="试题不存在")
        session.delete(item)
        session.commit()
        return {"status": "deleted"}

@app.get("/api/documents")
def list_documents() -> dict[str, list[dict]]:
    return {"documents": get_document_service().store.list()}


@app.get("/api/documents/{document_id}")
def get_document(document_id: str) -> dict:
    document = get_document_service().store.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


@app.post("/api/documents/upload", status_code=202)
def upload_document(file: UploadFile = File(...)) -> dict:
    """保存上传文件，并提交后台任务完成解析和向量入库。"""
    try:
        content = file.file.read()
        if not content:
            raise ValueError("上传文件不能为空")
        service = get_document_service()
        record = service.save_upload(file.filename or "", content)
        document_tasks.submit(service.process, record["id"])
        return record
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    try:
        deleted = get_document_service().delete(document_id)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"文档删除失败：{error}") from error
    if not deleted:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"status": "deleted"}


@app.get("/health")
def health() -> dict[str, str]:
    memory_status = "ok" if get_memory_store().is_available() else "unavailable"
    return {"status": "ok", "memory": memory_status}


@app.get("/api/sessions")
def list_sessions() -> dict[str, list[dict]]:
    return {"sessions": get_memory_store().list_sessions()}


@app.post("/api/sessions")
def create_session(request: SessionCreateRequest) -> dict[str, str]:
    session_id = uuid4().hex
    session = get_memory_store().create_session(session_id, request.title)
    return session


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str) -> dict[str, list[dict]]:
    messages = list(reversed(get_memory_store().get_session_messages(session_id)))
    return {"messages": messages}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    get_memory_store().delete_session(session_id)
    return {"status": "deleted"}


@app.delete("/api/sessions/{session_id}/messages")
def clear_messages(session_id: str) -> dict[str, str]:
    get_memory_store().clear_messages(session_id)
    return {"status": "cleared"}


@app.post("/api/chat/stream")
def stream_chat(request: ChatRequest) -> StreamingResponse:
    """以 Server-Sent Events 方式逐段返回模型回答。"""
    session_id = request.session_id or uuid4().hex
    filters = {
        key: value
        for key, value in {
            "subject": request.subject,
            "grade": request.grade,
            "chapter": request.chapter,
            "document_type": request.document_type,
        }.items()
        if value
    }

    def events():
        try:
            for event in get_service().stream_ask(request.question, session_id, request.top_k, filters):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as error:
            yield f"data: {json.dumps({'event': 'error', 'content': str(error)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """执行一次完整的同步 RAG 问答。"""
    try:
        session_id = request.session_id or uuid4().hex
        filters = {
            key: value
            for key, value in {
                "subject": request.subject,
                "grade": request.grade,
                "chapter": request.chapter,
                "document_type": request.document_type,
            }.items()
            if value
        }
        result = get_service().ask(
            request.question,
            session_id=session_id,
            top_k=request.top_k,
            filters=filters,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return ChatResponse(session_id=session_id, answer=result.answer, citations=result.citations)
