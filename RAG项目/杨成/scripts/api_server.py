from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import MilvusClient

from memory_store import MemoryStore
import rag_answer


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
MEMORY_DATABASE = ROOT / "data" / "runtime" / "memory.sqlite3"

app = FastAPI(title="Hypertension RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
_runtime = None
_runtime_lock = Lock()
_memory_store = None
_memory_store_lock = Lock()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="用户问题")
    session_id: str | None = Field(default=None, min_length=1, max_length=128, description="短期记忆会话 ID")
    user_id: str = Field(default="local-user", min_length=1, max_length=128, description="长期记忆用户 ID")


class Source(BaseModel):
    section_path: str
    chunk_type: str
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int
    answer_mode: str | None = None
    notice: str | None = None
    session_id: str | None = None
    memory_count: int | None = None


class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class Message(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class MemoryCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1, max_length=2000)
    category: str = Field(default="其他", min_length=1, max_length=50)


class Memory(BaseModel):
    id: int
    user_id: str
    content: str
    category: str
    created_at: str


@dataclass
class RagRuntime:
    client: MilvusClient
    collection: str
    embedding_model: BGEM3FlagModel
    reranker: FlagReranker
    model: str
    timeout: int


def get_rag_runtime():
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                embedding_model = BGEM3FlagModel(rag_answer.DEFAULT_EMBEDDING_MODEL_PATH, use_fp16=False)
                reranker = FlagReranker(rag_answer.DEFAULT_RERANKER_MODEL_PATH, use_fp16=False)
                client = MilvusClient(uri=rag_answer.DEFAULT_URI)
                client.load_collection(collection_name=rag_answer.DEFAULT_COLLECTION)
                _runtime = RagRuntime(
                    client=client,
                    collection=rag_answer.DEFAULT_COLLECTION,
                    embedding_model=embedding_model,
                    reranker=reranker,
                    model=rag_answer.DEFAULT_MODEL,
                    timeout=rag_answer.DEFAULT_TIMEOUT,
                )
    return _runtime


def get_memory_store():
    global _memory_store
    if _memory_store is None:
        with _memory_store_lock:
            if _memory_store is None:
                _memory_store = MemoryStore(MEMORY_DATABASE)
    return _memory_store


def configure_memory_store(database_path):
    global _memory_store
    with _memory_store_lock:
        if _memory_store is not None:
            _memory_store.close()
        _memory_store = MemoryStore(database_path)


def close_memory_store():
    global _memory_store
    with _memory_store_lock:
        if _memory_store is not None:
            _memory_store.close()
            _memory_store = None


def format_conversation(messages):
    if not messages:
        return "无"
    role_names = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{role_names.get(item['role'], item['role'])}：{item['content']}" for item in messages)


def format_long_term_memory(memories):
    if not memories:
        return "无"
    return "\n".join(f"{item['category']}：{item['content']}" for item in memories)


def answer_with_runtime(runtime, question, conversation_context="", long_term_memory=""):
    return rag_answer.answer_question(
        client=runtime.client,
        collection=runtime.collection,
        embedding_model=runtime.embedding_model,
        reranker=runtime.reranker,
        question=question,
        model=runtime.model,
        timeout=runtime.timeout,
        conversation_context=conversation_context,
        long_term_memory=long_term_memory,
    )


def format_sources(retrieved):
    return [
        {
            "section_path": item.get("section_path", ""),
            "chunk_type": item.get("chunk_type", ""),
            "rerank_score": item.get("rerank_score"),
        }
        for item in retrieved
    ]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/design-preview", include_in_schema=False)
def design_preview():
    return FileResponse(WEB_DIR / "design-preview.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, response_model_exclude_unset=True)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid4())
    store = get_memory_store()
    store.ensure_session(request.user_id, session_id, request.question[:40])
    recent_messages = store.get_recent_messages(session_id, max_rounds=6)
    memories = store.list_memories(request.user_id)
    try:
        runtime = get_rag_runtime()
        result = answer_with_runtime(
            runtime,
            request.question,
            conversation_context=format_conversation(recent_messages),
            long_term_memory=format_long_term_memory(memories),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    store.add_message(session_id, "user", request.question)
    store.add_message(session_id, "assistant", result.get("answer", ""))
    response = {
        "answer": result.get("answer", ""),
        "sources": format_sources(result.get("retrieved", [])),
        "latency_ms": result.get("latency", {}).get("total_ms", 0),
        "answer_mode": result.get("answer_mode"),
        "notice": result.get("notice"),
    }
    if request.session_id is not None:
        response["session_id"] = session_id
        response["memory_count"] = len(recent_messages) // 2 + 1
    return response


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions(user_id: str = Query(..., min_length=1, max_length=128)):
    return get_memory_store().list_sessions(user_id)


@app.get("/sessions/{session_id}/messages", response_model=list[Message])
def get_session_messages(
    session_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
):
    messages = get_memory_store().get_messages(session_id, user_id=user_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="session not found")
    return messages


@app.post("/memories", response_model=Memory, status_code=status.HTTP_201_CREATED)
def create_memory(request: MemoryCreateRequest):
    try:
        return get_memory_store().create_memory(request.user_id, request.content, request.category)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/memories", response_model=list[Memory])
def list_memories(user_id: str = Query(..., min_length=1, max_length=128)):
    return get_memory_store().list_memories(user_id)


@app.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(memory_id: int, user_id: str = Query(..., min_length=1, max_length=128)):
    if not get_memory_store().delete_memory(user_id, memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
