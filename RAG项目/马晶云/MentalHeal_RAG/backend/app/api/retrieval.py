from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.memory.redis_memory import get_chat_memory
from app.models.chat import User
from app.rag.retriever import get_retriever
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedChunk
from app.security.auth import get_current_user
from app.services.chat_history import ChatHistoryRepository

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


def build_retrieval_summary(results: list[dict]) -> str:
    if not results:
        return "当前知识库没有找到足够相关的内容。"
    lines = ["已从心理健康知识库检索到相关资料："]
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item['title']} 第{item['page_start']}-{item['page_end']}页")
    return "\n".join(lines)


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve(
    request: RetrievalRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> RetrievalResponse:
    try:
        query = request.query.strip()
        results = get_retriever().retrieve(query, request.top_k)
        history = ChatHistoryRepository(db)
        chat_session = history.get_or_create_session(
            request.session_id,
            query,
            current_user.user_id,
            "mental-health",
        )
        summary = build_retrieval_summary(results)
        history.add_message(chat_session, "user", query)
        history.add_message(chat_session, "assistant", summary, sources=results)
        history.commit()
        get_chat_memory().save_turn(
            chat_session.session_id,
            query,
            summary,
            current_user.user_id,
            chat_session.role_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="知识库检索暂时不可用") from exc
    return RetrievalResponse(
        session_id=chat_session.session_id,
        query=request.query,
        results=[RetrievedChunk(**item) for item in results],
    )
