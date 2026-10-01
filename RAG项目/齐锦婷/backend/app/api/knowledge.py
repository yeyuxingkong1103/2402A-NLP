from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user
from backend.app.models.entities import KnowledgeBase, User
from backend.app.models.schemas import KnowledgeBaseCreate, KnowledgeBaseOut


router = APIRouter(prefix="/knowledge-bases", tags=["知识库"])


@router.get("", response_model=list[KnowledgeBaseOut])
def list_knowledge_bases(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[KnowledgeBase]:
    return db.query(KnowledgeBase).filter(KnowledgeBase.user_id == current_user.id).order_by(KnowledgeBase.id.desc()).all()


@router.post("", response_model=KnowledgeBaseOut)
def create_knowledge_base(payload: KnowledgeBaseCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> KnowledgeBase:
    kb = KnowledgeBase(user_id=current_user.id, name=payload.name, description=payload.description)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


def require_kb(db: Session, user_id: int, knowledge_base_id: int) -> KnowledgeBase:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id, KnowledgeBase.user_id == user_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在或无权访问")
    return kb
