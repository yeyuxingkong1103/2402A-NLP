# -*- coding: utf-8 -*-
'''
对话接口：
接口	流程
POST /api/chat	RAG 检索 → 记忆检索 → LLM 生成 → 后处理校验 → 落库 → 沉淀记忆
POST /api/chat/stream	同上，但用 SSE 逐块推送：delta* → sources → warnings → [DONE]
GET /api/chat/history	按 user+role 查 MySQL 历史
RAG 检索：
双层开关：全局 rag_enabled + 请求级 use_rag。
recall_k=30 粗召回 → 重排 → top_k=4 注入。
长期记忆：
默认开启，失败静默降级（不影响对话主流程）。
检索相关历史注入 prompt，对话结束后沉淀本轮。
知识库接口：
接口	说明
POST /upload	PDF/图片 → 解析/OCR → 分块 → 向量化 → 入库
GET /docs	按来源聚合的文档列表
DELETE /doc	删除某文档全部块（立即生效）
SSE 时序要点：
用户消息在流开始前落库。
助手消息在流结束后拼完整再落库。
sources/warnings 在末尾推送，会有一小段延迟。
'''

"""对话接口（非流式/SSE/历史）与知识库接口（上传/列表/删除）。"""
import json
# 解析：JSON 模块（SSE 事件序列化）
import logging
# 解析：日志模块

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
# 解析：路由、依赖、文件上传组件
from fastapi.responses import StreamingResponse
# 解析：流式响应（SSE）
from sqlalchemy.orm import Session
# 解析：数据库会话类型

from app.api.deps import (
    # 解析：依赖导入
    get_current_user,
    # 解析：鉴权
    get_db,
    # 解析：会话
    get_knowledge_service,
    # 解析：知识库服务
    get_llm,
    # 解析：大模型
    get_memory,
    # 解析：短期记忆
    get_memory_service,
    # 解析：长期记忆
)
from app.config import settings
# 解析：全局配置
from app.models.tables import Message
# 解析：消息表
from app.models.tables import Role
# 解析：角色表
from app.rag.verification import verify_answer
# 解析：后处理校验
from app.schemas import ChatIn, ChatOut, MessageOut
# 解析：请求/响应模型
from app.services.chat_service import ChatService
# 解析：对话编排服务

# ================= 对话 =================

chat_router = APIRouter(prefix="/api/chat", tags=["对话"])
# 解析：对话路由组（前缀 /api/chat）


async def _retrieve_memories(memory_service, user_id: int, role_id: int, query: str) -> list[str]:
    """长期记忆检索：默认开启，失败/关闭时返回空（不影响主流程）。"""
    if not settings.longterm_memory:
        # 解析：功能关闭
        return []
        # 解析：返回空
    try:
        # 解析：尝试检索
        return await _async_memory_retrieve(memory_service, user_id, role_id, query)
        # 解析：检索相关记忆
    except Exception:
        # 解析：检索失败（如 Milvus 掉线）
        return []
        # 解析：静默回退空——记忆失败不影响对话主流程
'''
全局开关：longterm_memory=false 直接返回空。
try/except 全捕获：Milvus 掉线、模型加载失败等都不影响对话主流程，静默降级。
async 但内部 retrieve 是同步的——见下。
'''



async def _async_memory_retrieve(memory_service, user_id, role_id, query):
    # 解析：调用记忆检索
    return memory_service.retrieve(
        # 解析：服务检索
        user_id, role_id, query,
        # 解析：双方 ID 与问题
        top_k=settings.memory_top_k,
        # 解析：注入条数
        recall_k=settings.memory_top_k * 4,
        # 解析：召回条数（4 倍注入数，留过滤空间）
    )
'''
注意：这里其实是同步调用包在 async 函数里，会阻塞事件循环。
正确做法应是 await asyncio.to_thread(memory_service.retrieve, ...) 或让 retrieve 本身异步。
recall_k = top_k * 4：先粗召回 4 倍，重排后再取 top_k，留过滤空间。
'''

async def _remember_turn(memory_service, user_id: int, role: Role, user_input: str, reply: str) -> None:
    """把本轮对话沉淀为长期记忆；失败不影响响应。"""
    if not settings.longterm_memory:
        # 解析：功能关闭
        return
        # 解析：直接返回
    try:
        # 解析：尝试沉淀
        memory_service.remember(user_id, role.id, role.name, user_input, reply)
        # 解析：本轮对话原文入库记忆库
    except Exception:
        # 解析：沉淀失败
        pass
        # 解析：静默忽略——不影响响应
'''
把本轮 (user_input, reply) 沉淀进长期记忆。
同样失败静默忽略。
传入 role.name 便于记忆里标注角色身份。
'''


def _get_role(db: Session, role_id: int) -> Role:
    # 解析：查角色（对话与知识库接口共用）
    role = db.get(Role, role_id)
    # 解析：按主键查
    if role is None:
        # 解析：不存在
        raise HTTPException(status_code=404, detail="角色不存在")
        # 解析：404
    return role
    # 解析：返回角色
'''
对话和知识库接口共用。
db.get 按主键查，比 query().filter().first() 更快（走 identity map）。
'''


async def _retrieve(
    # 解析：RAG 检索
    knowledge_service, role_id: int, query: str, use_rag: bool
    # 解析：服务、角色、问题、请求级开关
) -> tuple[str, list[str]]:
    """RAG 检索：返回 (拼接后的知识文本, 引用来源)。"""
    if not (settings.rag_enabled and use_rag):
        # 解析：全局或请求级关闭
        return "", []
        # 解析：跳过检索
    chunks = await knowledge_service.retrieve(
        # 解析：知识库检索
        role_id, query, top_k=settings.rag_top_k, recall_k=settings.rag_recall_k
        # 解析：角色、问题、Top4、召回30
    )
    return "\n\n".join(chunks), chunks
    # 解析：返回拼接知识文本与来源列表
'''
双层开关：全局 rag_enabled 且 请求级 use_rag，任一为假都跳过。
top_k=4：最终注入 4 块。
recall_k=30：先召回 30 块，重排后取 4。
返回 (拼接文本, 来源列表)：
文本给 LLM 当上下文。
来源给前端展示引用、给 verify_answer 做校验。
'''

@chat_router.post("", response_model=ChatOut)
# 解析：非流式对话接口
# 非流式对话接口：RAG检索 → 生成 → 后处理校验 → 双重持久化
async def chat(
    # 解析：对话处理
    body: ChatIn,
    # 解析：请求体
    db: Session = Depends(get_db),
    # 解析：会话
    user=Depends(get_current_user),
    # 解析：鉴权
    llm=Depends(get_llm),
    # 解析：大模型
    memory=Depends(get_memory),
    # 解析：短期记忆
    knowledge_service=Depends(get_knowledge_service),
    # 解析：知识库服务
    memory_service=Depends(get_memory_service),
    # 解析：长期记忆服务
):
    role = _get_role(db, body.role_id)
    # 解析：查角色
    knowledge, sources = await _retrieve(knowledge_service, role.id, body.content, body.use_rag)
    # 解析：RAG 检索
    memories = await _retrieve_memories(memory_service, user.id, role.id, body.content)
    # 解析：长期记忆检索
    reply = await ChatService(llm, memory).chat(
        # 解析：对话编排
        role, body.content, user.id, role.id, knowledge, memories
        # 解析：角色、输入、双方ID、知识、记忆
    )
    warnings = verify_answer(reply, sources)  # 后处理校验
    # 解析：回答数值与知识块对照

    db.add(Message(user_id=user.id, role_id=role.id, sender="user", content=body.content))
    # 解析：用户消息落 MySQL
    db.add(Message(user_id=user.id, role_id=role.id, sender="assistant", content=reply))
    # 解析：角色回复落 MySQL
    db.commit()
    # 解析：提交
    await _remember_turn(memory_service, user.id, role, body.content, reply)
    # 解析：本轮对话沉淀长期记忆
    return ChatOut(role_id=role.id, reply=reply, sources=sources, warnings=warnings)
    # 解析：返回完整响应


@chat_router.post("/stream")
# 解析：SSE 流式对话接口
# SSE 流式对话接口：delta 分块 → sources → warnings → DONE
async def chat_stream(
    # 解析：流式对话处理
    body: ChatIn,
    # 解析：请求体
    db: Session = Depends(get_db),
    # 解析：会话
    user=Depends(get_current_user),
    # 解析：鉴权
    llm=Depends(get_llm),
    # 解析：大模型
    memory=Depends(get_memory),
    # 解析：短期记忆
    knowledge_service=Depends(get_knowledge_service),
    # 解析：知识库服务
    memory_service=Depends(get_memory_service),
    # 解析：长期记忆服务
):
    role = _get_role(db, body.role_id)
    # 解析：查角色
    knowledge, sources = await _retrieve(knowledge_service, role.id, body.content, body.use_rag)
    # 解析：RAG 检索
    memories = await _retrieve_memories(memory_service, user.id, role.id, body.content)
    # 解析：长期记忆检索
    db.add(Message(user_id=user.id, role_id=role.id, sender="user", content=body.content))
    # 解析：用户消息落 MySQL
    db.commit()
    # 解析：提交

    async def generate():
        # 解析：SSE 生成器（响应发送时执行）
        parts = []
        # 解析：收集分块拼完整回复
        async for chunk in ChatService(llm, memory).chat_stream(
            # 解析：流式对话
            role, body.content, user.id, role.id, knowledge, memories
            # 解析：全部上下文
        ):
            parts.append(chunk)
            # 解析：收集分块
            yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            # 解析：SSE 事件——每个 delta 一个 JSON 事件（前端逐字渲染）
        full_reply = "".join(parts)
        # 解析：拼完整回复
        if full_reply:  # 只有完整回复才持久化
            # 解析：有内容
            db.add(
                # 解析：回复落 MySQL
                Message(
                    # 解析：消息对象
                    user_id=user.id,
                    # 解析：用户
                    role_id=role.id,
                    # 解析：角色
                    sender="assistant",
                    # 解析：发送方
                    content=full_reply,
                    # 解析：完整回复
                )
            )
            db.commit()
            # 解析：提交
        await _remember_turn(memory_service, user.id, role, body.content, full_reply)
        # 解析：沉淀长期记忆
        if sources:
            # 解析：有引用来源
            yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
            # 解析：SSE 事件——引用来源
        warnings = verify_answer(full_reply, sources)
        # 解析：后处理校验
        if warnings:
            # 解析：有提醒
            yield f"data: {json.dumps({'warnings': warnings}, ensure_ascii=False)}\n\n"
            # 解析：SSE 事件——校验提醒
        yield "data: [DONE]\n\n"
        # 解析：SSE 结束事件

    return StreamingResponse(generate(), media_type="text/event-stream")
    # 解析：返回流式响应（SSE 媒体类型）


@chat_router.get("/history", response_model=list[MessageOut])
# 解析：历史记录接口
# 会话历史查询（MySQL 全量消息，按时间升序）
def history(
    # 解析：查历史
    role_id: int,
    # 解析：角色 ID
    db: Session = Depends(get_db),
    # 解析：会话
    user=Depends(get_current_user),
    # 解析：鉴权
):
    return (
        # 解析：查询
        db.query(Message)
        # 解析：消息表
        .filter(Message.user_id == user.id, Message.role_id == role_id)
        # 解析：只查当前用户与该角色的消息
        .order_by(Message.created_at, Message.id)
        # 解析：按时间与 ID 升序
        .all()
        # 解析：取全部
    )

# ================= 知识库 =================

logger = logging.getLogger("rag-roleplay.knowledge")
# 解析：知识库模块 logger
kb_router = APIRouter(prefix="/api/knowledge", tags=["知识库"])
# 解析：知识库路由组（前缀 /api/knowledge）


def _check_role(db: Session, role_id: int) -> None:
    # 解析：校验角色存在（知识库接口共用）
    if db.get(Role, role_id) is None:
        # 解析：不存在
        raise HTTPException(status_code=404, detail="角色不存在")
        # 解析：404


@kb_router.post("/upload")
# 解析：上传文档接口
async def upload(
    # 解析：上传处理
    role_id: int,
    # 解析：角色 ID（查询参数）
    file: UploadFile = File(...),
    # 解析：上传文件（必填）
    db: Session = Depends(get_db),
    # 解析：会话
    user=Depends(get_current_user),
    # 解析：鉴权
    knowledge_service=Depends(get_knowledge_service),
    # 解析：知识库服务
):
    """上传文档：PDF（解析+表格提取）或图片（OCR 文字识别）→ 分块 → 向量化 → 入库。"""
    _check_role(db, role_id)
    # 解析：校验角色
    filename = (file.filename or "").lower()
    # 解析：文件名小写（扩展名判断）
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # 解析：图片
        is_image = True
        # 解析：标记图片
    elif filename.endswith(".pdf"):
        # 解析：PDF
        is_image = False
        # 解析：标记 PDF
    else:
        # 解析：其他格式
        raise HTTPException(status_code=400, detail="仅支持 PDF / JPG / PNG 文件")
        # 解析：400
    content = await file.read()
    # 解析：读文件字节
    if not content:
        # 解析：空文件
        raise HTTPException(status_code=400, detail="文件为空")
        # 解析：400
    try:
        # 解析：入库
        if is_image:
            # 解析：图片路径
            result = knowledge_service.ingest_image(
                # 解析：OCR 入库
                role_id=role_id, image_bytes=content, source=file.filename
                # 解析：角色、字节、文件名
            )
        else:
            # 解析：PDF 路径
            result = knowledge_service.ingest_pdf(
                # 解析：PDF 入库
                role_id=role_id, pdf_bytes=content, source=file.filename
                # 解析：角色、字节、文件名
            )
    except ValueError as exc:
        # 解析：入库失败（空文本等）
        raise HTTPException(status_code=400, detail=str(exc))
        # 解析：转 400
    logger.info("角色 %s 已入库文档 %s（%s 块）", role_id, file.filename, result["chunks"])
    # 解析：记录日志
    return result
    # 解析：返回入库结果


@kb_router.get("/docs")
# 解析：文档列表接口
def list_docs(
    # 解析：查文档
    role_id: int,
    # 解析：角色 ID
    db: Session = Depends(get_db),
    # 解析：会话
    user=Depends(get_current_user),
    # 解析：鉴权
    knowledge_service=Depends(get_knowledge_service),
    # 解析：知识库服务
):
    """知识库内文档列表（按来源聚合，含块数）。"""
    _check_role(db, role_id)
    # 解析：校验角色
    return knowledge_service.list_sources(role_id)
    # 解析：返回聚合文档列表（含摘要）


@kb_router.delete("/doc")
# 解析：删除文档接口
def delete_doc(
    # 解析：删文档
    role_id: int,
    # 解析：角色 ID
    source: str,
    # 解析：文档名（查询参数）
    db: Session = Depends(get_db),
    # 解析：会话
    user=Depends(get_current_user),
    # 解析：鉴权
    knowledge_service=Depends(get_knowledge_service),
    # 解析：知识库服务
):
    """删除某文档的全部块（知识库动态更新）。"""
    _check_role(db, role_id)
    # 解析：校验角色
    removed = knowledge_service.delete_source(role_id, source)
    # 解析：删除该文档全部块（立即生效）
    return {"source": source, "removed": removed}
    # 解析：返回删除结果
