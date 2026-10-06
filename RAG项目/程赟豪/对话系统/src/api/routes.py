"""API路由模块"""
import json as _json
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime

from src.utils.logger import logger
from src.rag import Retriever, AnswerGenerator
from src.rag.knowledge import knowledge_service
from src.rag.query_rewriter import QueryRewriter
from src.memory import short_term_memory, long_term_memory
from src.llm import get_llm_client
from src.role import role_manager, get_available_roles


# API路由
router = APIRouter()


# ============ 数据模型 ============

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: str
    role_id: Optional[str] = None
    use_history: bool = True


class ChatResponse(BaseModel):
    """聊天响应"""
    message: str
    session_id: str
    sources: List[Dict[str, Any]] = []
    role: Optional[Dict[str, Any]] = None


class RoleCreateRequest(BaseModel):
    """创建角色请求"""
    role_type: str
    role_id: str
    name: Optional[str] = None
    custom_prompt: Optional[str] = None
    knowledge_sources: Optional[List[str]] = None


class RoleResponse(BaseModel):
    """角色响应"""
    role_id: str
    name: str
    description: str


class KnowledgeAddRequest(BaseModel):
    """添加知识请求"""
    file_path: str
    source_name: Optional[str] = None
    chunking_method: str = "semantic"


class KnowledgeBuildRequest(BaseModel):
    """批量构建知识库请求"""
    raw_dir: Optional[str] = None
    drop_existing: bool = False


# ============ 初始化 ============

# 全局组件
_retriever: Optional[Retriever] = None
_generator: Optional[AnswerGenerator] = None
_llm_client: Optional[Any] = None


def _ensure_llm() -> Optional[Any]:
    """惰性获取大模型客户端；失败时返回 None，链路降级为无 LLM 生成。"""
    global _llm_client
    if _llm_client is None:
        try:
            _llm_client = get_llm_client()
        except Exception as e:
            logger.warning(f"大模型初始化失败，将使用降级生成: {e}")
            _llm_client = None
    return _llm_client


def get_retriever() -> Retriever:
    """获取检索器（含 Query 改写器）"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(query_rewriter=QueryRewriter(llm_client=_ensure_llm()))
    return _retriever


def get_generator() -> AnswerGenerator:
    """获取生成器"""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator(llm_client=_ensure_llm())
    return _generator


# ============ 聊天接口 ============

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口
    
    支持多轮对话，使用短期记忆（Redis）存储对话历史
    使用长期记忆（Milvus）存储用户画像
    """
    try:
        logger.info(f"收到聊天请求: session={request.session_id}, message={request.message[:50]}...")
        
        # 获取对话历史
        history = []
        if request.use_history:
            history = short_term_memory.get_history(request.session_id)
        
        # 获取角色
        role = None
        if request.role_id:
            role = role_manager.get_role(request.role_id)
        
        # 检索相关文档
        retriever = get_retriever()
        docs = retriever.retrieve_with_context(
            request.message,
            conversation_history=history
        )
        
        # 生成回答
        generator = get_generator()
        result = generator.generate(
            query=request.message,
            context_docs=docs,
            system_prompt=role.system_prompt if role else None,
            conversation_history=history
        )
        
        # 保存对话到短期记忆
        short_term_memory.save_message(
            request.session_id,
            role="user",
            content=request.message
        )
        short_term_memory.save_message(
            request.session_id,
            role="assistant",
            content=result.text
        )
        
        return ChatResponse(
            message=result.text,
            session_id=request.session_id,
            sources=[
                {
                    "text": doc.get("text", "")[:200],
                    "source": doc.get("source", ""),
                    "page": doc.get("page_number", "")
                }
                for doc in result.source_docs[:3]
            ],
            role={"role_id": role.role_id, "name": role.name} if role else None
        )
        
    except Exception as e:
        logger.error(f"聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口（SSE），对应文档.txt 的「流式输出」。"""
    try:
        history = (
            short_term_memory.get_history(request.session_id)
            if request.use_history else []
        )
        role = role_manager.get_role(request.role_id) if request.role_id else None
        docs = get_retriever().retrieve_with_context(
            request.message, conversation_history=history
        )
        generator = get_generator()

        async def event_stream():
            collected = []
            try:
                for chunk in generator.generate_stream(
                    query=request.message,
                    context_docs=docs,
                    system_prompt=role.system_prompt if role else None,
                    conversation_history=history,
                ):
                    collected.append(chunk)
                    yield f"data: {_json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"

                # 流式结束后把本轮对话写入短期记忆
                short_term_memory.save_message(request.session_id, "user", request.message)
                short_term_memory.save_message(request.session_id, "assistant", "".join(collected))
                yield f"data: {_json.dumps({'done': True}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"流式生成失败: {e}")
                yield f"data: {_json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"流式聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 角色管理接口 ============

@router.post("/api/role/create", response_model=RoleResponse)
async def create_role(request: RoleCreateRequest):
    """创建角色"""
    try:
        role = role_manager.create_role(
            role_type=request.role_type,
            role_id=request.role_id,
            name=request.name,
            custom_prompt=request.custom_prompt,
            knowledge_sources=request.knowledge_sources
        )
        
        return RoleResponse(
            role_id=role.role_id,
            name=role.name,
            description=role.description
        )
        
    except Exception as e:
        logger.error(f"创建角色失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/role/list")
async def list_roles():
    """获取角色列表"""
    return {
        "roles": role_manager.list_roles(),
        "available_types": get_available_roles()
    }


@router.get("/api/role/{role_id}")
async def get_role(role_id: str):
    """获取角色详情"""
    role = role_manager.get_role(role_id)
    
    if not role:
        raise HTTPException(status_code=404, detail="角色不存在")
    
    return {
        "role_id": role.role_id,
        "name": role.name,
        "description": role.description,
        "system_prompt": role.system_prompt,
        "personality": role.personality,
        "speaking_style": role.speaking_style,
        "knowledge_sources": role.knowledge_sources
    }


# ============ 知识库接口 ============

@router.post("/api/knowledge/add")
async def add_knowledge(request: KnowledgeAddRequest):
    """添加单个文档到知识库（解析+分块+向量化+入库）"""
    try:
        result = knowledge_service.ingest_file(
            file_path=request.file_path,
            source_name=request.source_name,
            chunking_method=request.chunking_method,
        )
        if result["chunks"] == 0:
            return {"status": "ok", "message": "文档未提取到有效文本", "chunks": 0}
        return {
            "status": "ok",
            "message": f"已添加 {result['source']}，共 {result['chunks']} 个分块",
            "source": result["source"],
            "chunks": result["chunks"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"添加知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/knowledge/build")
async def build_knowledge(request: KnowledgeBuildRequest):
    """批量索引 raw 目录下的所有文档（PDF/txt/md/json）"""
    from pathlib import Path

    raw_dir = request.raw_dir or str(Path(__file__).resolve().parent.parent.parent / "data" / "raw")
    try:
        result = knowledge_service.ingest_directory(raw_dir, drop_existing=request.drop_existing)
        return {
            "status": "ok",
            "message": f"构建完成：处理 {result['files']} 个文件，共 {result['chunks']} 个分块",
            **result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"构建知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/knowledge/stats")
async def knowledge_stats():
    """知识库统计"""
    return knowledge_service.get_stats()


@router.get("/api/knowledge/list")
async def knowledge_list():
    """列出知识库中的所有文档来源（知识库动态更新）。"""
    return {"sources": knowledge_service.list_sources()}


@router.delete("/api/knowledge/{source:path}")
async def delete_knowledge(source: str):
    """按来源删除文档全部分块（知识库动态更新）。"""
    try:
        result = knowledge_service.delete_source(source)
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"删除知识库来源失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 会话管理接口 ============

@router.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """清除会话"""
    short_term_memory.clear_history(session_id)
    return {"status": "ok", "message": f"会话 {session_id} 已清除"}


# ============ 健康检查 ============

@router.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }
