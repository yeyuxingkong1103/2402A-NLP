"""知识库接口：上传、入库、检索、动态更新（管理员）。

知识库是 RAG（检索增强生成）的"语料来源"：文档先被分块、向量化后写入向量库，
聊天时再按语义检索相关片段作为上下文。写操作（上传/删除/重建）均需管理员权限，
检索测试接口也限定管理员，因为这些都属于后台运维能力。
"""
import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from src.api.deps import get_current_admin
from src.core.config import settings
from src.core.exceptions import AppError, ok
from src.db.mysql import get_db
from src.models import User
from src.schemas import KnowledgeRebuildRequest, KnowledgeSearchRequest
from src.services import knowledge_service, persona_service
from src.services.persona_seed import KNOWLEDGE_DIRS

router = APIRouter(prefix="/knowledge", tags=["知识库"])

# 白名单：只允许这些可解析的文档格式入库，防止上传可执行文件等危险内容。
ALLOWED_EXT = {"pdf", "txt", "md", "docx"}


@router.post("/upload", summary="上传知识文档并入库（K-01 ~ K-09）")
def upload_knowledge(
    persona_id: int = Form(..., description="归属心理医生角色 ID"),
    strategy: str = Form("paragraph", description="分块策略：fixed/sentence/paragraph/heading/semantic"),
    file: UploadFile = File(...),
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    # 该接口使用 multipart/form-data：persona_id/strategy 走 Form 字段，文件走 File 字段。
    # FastAPI 的 UploadFile 支持流式读取，但这里文档通常不大，直接整读即可。
    persona_service.get_persona(db, persona_id)
    # 从文件名后缀提取类型并做白名单校验，是上传安全的第一道防线。
    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if ext not in ALLOWED_EXT:
        raise AppError(f"不支持的文件类型：{ext}，仅支持 {sorted(ALLOWED_EXT)}")

    # 先落盘到本地上传目录，再做解析入库；文件名前缀加时间戳避免重名覆盖。
    upload_dir = os.path.join(settings.data_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, f"{int(time.time())}_{file.filename}")
    with open(saved_path, "wb") as f:
        f.write(file.file.read())

    # 解析、分块、向量化、写入向量库等重活全部交给服务层。
    result = knowledge_service.ingest_file(db, saved_path, persona_id, strategy=strategy)
    return ok(result, "文档入库完成")


@router.get("/docs", summary="知识库文档列表（K-10）")
def list_docs(persona_id: Optional[int] = Query(None), admin: User = Depends(get_current_admin),
              db: Session = Depends(get_db)):
    docs = knowledge_service.list_docs(db, persona_id)
    # 控制器负责把 ORM 对象整理成对外的字典结构（序列化），
    # 这里对 created_at 做字符串格式化，避免直接暴露 datetime 对象给 JSON 序列化。
    items = [
        {
            "id": d.id, "persona_id": d.persona_id, "title": d.title, "source": d.source,
            "file_type": d.file_type, "status": d.status, "chunk_count": d.chunk_count,
            "error_msg": d.error_msg,
            "created_at": d.created_at.strftime("%Y-%m-%d %H:%M:%S") if d.created_at else None,
        }
        for d in docs
    ]
    return ok({"items": items, "total": len(items)})


@router.delete("/docs/{doc_id}", summary="删除文档及其向量（K-11）")
def delete_doc(doc_id: int, admin: User = Depends(get_current_admin),
               db: Session = Depends(get_db)):
    # 删除文档时还必须同步清理其在向量库中的向量，否则会出现"检索到已删内容"的脏数据。
    return ok(knowledge_service.delete_doc(db, doc_id), "文档已删除")


@router.post("/rebuild", summary="按角色重建索引（K-11 / K-12）")
def rebuild(payload: KnowledgeRebuildRequest, admin: User = Depends(get_current_admin),
            db: Session = Depends(get_db)):
    # 重建索引：把指定角色（或全部角色）的知识目录重新解析入库。
    # persona_id 为空则重建所有角色，否则只重建指定角色。
    personas = persona_service.list_personas(db, only_active=False)
    targets = [p for p in personas if payload.persona_id in (None, p.id)]
    results = []
    for persona in targets:
        # 通过角色代码找到其对应的知识目录；没有配置目录的角色直接跳过。
        dirs = KNOWLEDGE_DIRS.get(persona.persona_code)
        if not dirs:
            continue
        results.append(knowledge_service.rebuild_persona_index(
            db, persona.id, dirs, drop_existing=payload.drop_existing
        ))
    return ok({"results": results}, "索引重建完成")


@router.post("/search", summary="知识库检索测试（R-01 ~ R-06）")
def search(payload: KnowledgeSearchRequest, user: User = Depends(get_current_admin),
           db: Session = Depends(get_db)):
    # 检索接口返回原始命中的 chunk 及重排后得分，用于调试/验证 RAG 检索质量。
    hits, rewritten = knowledge_service.search(db, payload.persona_id, payload.query, payload.top_k)
    return ok({
        "query": payload.query,
        "rewritten_query": rewritten,
        "hits": [
            {
                "doc_id": h.get("doc_id"), "chunk_id": h.get("chunk_id"),
                "source": h.get("source"), "text": h.get("text"),
                # 优先展示重排分（rerank_score），没有则回退到原始检索分（score），保留 4 位小数。
                "score": round(float(h.get("rerank_score", h.get("score", 0))), 4),
            }
            for h in hits
        ],
    })


@router.get("/stats", summary="知识库统计")
def stats(admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return ok(knowledge_service.knowledge_stats(db))