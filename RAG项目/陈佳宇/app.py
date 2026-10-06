from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from db import (list_characters, get_character_by_id, verify_user, register_user,
                add_character, update_character, delete_character)
from pdf_parser import parse_pdf, split_text
from logger import get_logger
from auth import create_token, verify_token
from rag_core import rag_chat, rag_chat_stream, get_vs, clear_history, export_history
import os
import json

logger = get_logger()
app = FastAPI(title="基于RAG的角色扮演系统API", version="1.9.0")

@app.on_event("startup")
async def startup_event():
    logger.info("===== 服务启动，预加载模型 =====")
    try:
        chars = list_characters()
        for c in chars:
            try:
                logger.info(f"预加载角色模型：{c[1]}")
                get_vs(c[1])
            except Exception as e:
                logger.warning(f"预加载角色{c[1]}失败：{e}")
        logger.info("===== 所有模型预加载完成 =====")
    except Exception as e:
        logger.error(f"预加载失败：{e}")

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    character_name: str
    query: str

class ImportRequest(BaseModel):
    character_name: str
    file_path: str

class AddKnowledgeRequest(BaseModel):
    character_name: str
    text: str

class ClearHistoryRequest(BaseModel):
    character_name: str

class DeleteKnowledgeRequest(BaseModel):
    character_name: str
    doc_id: Optional[int] = None
    text: Optional[str] = None

class AddCharacterRequest(BaseModel):
    name: str
    description: str
    system_prompt: str
    collection_name: str

class UpdateCharacterRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    collection_name: Optional[str] = None

def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供Token")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token格式错误")
    token = authorization[7:]
    username = verify_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Token无效或已过期")
    return username

@app.post("/api/register", summary="用户注册")
def register(req: RegisterRequest):
    try:
        success, msg = register_user(req.username, req.password)
        if success:
            logger.info(f"用户注册成功：{req.username}")
            return {"code": 0, "msg": msg}
        raise HTTPException(status_code=400, detail=msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"注册失败：{str(e)}")

@app.post("/api/login", summary="用户登录，返回Token")
def login(req: LoginRequest):
    try:
        user = verify_user(req.username, req.password)
        if user:
            token = create_token(req.username)
            logger.info(f"API登录成功：{req.username}")
            return {"code": 0, "msg": "登录成功", "data": {"token": token, "username": req.username}}
        logger.warning(f"API登录失败：{req.username}")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登录异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"登录失败：{str(e)}")

@app.get("/api/characters", summary="获取角色列表（无需鉴权）")
def list_characters_api():
    try:
        characters = list_characters()
        data = [{"id": c[0], "name": c[1], "description": c[2]} for c in characters]
        return {"code": 0, "msg": "success", "data": data}
    except Exception as e:
        logger.error(f"获取角色列表异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取角色列表失败：{str(e)}")

@app.get("/api/characters/{char_id}", summary="获取角色详情（需鉴权）")
def get_character_detail(char_id: int, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    try:
        char = get_character_by_id(char_id)
        if not char:
            raise HTTPException(status_code=404, detail="角色不存在")
        data = {"id": char[0], "name": char[1], "description": char[2], "system_prompt": char[3], "collection_name": char[4]}
        return {"code": 0, "msg": "success", "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取角色详情异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取角色详情失败：{str(e)}")

@app.post("/api/characters", summary="新增角色（需鉴权）")
def create_character(req: AddCharacterRequest, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    try:
        success, msg = add_character(req.name, req.description, req.system_prompt, req.collection_name)
        if success:
            logger.info(f"新增角色：{req.name}")
            return {"code": 0, "msg": msg}
        raise HTTPException(status_code=400, detail=msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"新增角色异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"新增角色失败：{str(e)}")

@app.put("/api/characters/{char_id}", summary="更新角色（需鉴权）")
def update_character_api(char_id: int, req: UpdateCharacterRequest, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    try:
        success, msg = update_character(char_id, req.name, req.description, req.system_prompt, req.collection_name)
        if success:
            logger.info(f"更新角色ID={char_id}")
            return {"code": 0, "msg": msg}
        raise HTTPException(status_code=400, detail=msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新角色异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"更新角色失败：{str(e)}")

@app.delete("/api/characters/{char_id}", summary="删除角色（需鉴权）")
def delete_character_api(char_id: int, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    try:
        success, msg = delete_character(char_id)
        if success:
            logger.info(f"删除角色ID={char_id}")
            return {"code": 0, "msg": msg}
        raise HTTPException(status_code=404, detail=msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除角色异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"删除角色失败：{str(e)}")

@app.post("/api/chat", summary="对话接口（需鉴权）")
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    username = get_current_user(authorization)
    try:
        logger.info(f"API对话：用户={username}, 角色={req.character_name}, 问题={req.query}")
        character, answer, retrieved_docs, elapsed = rag_chat(username, req.character_name, req.query)
        if not character:
            raise HTTPException(status_code=404, detail=answer)
        logger.info(f"API回答完成，耗时{elapsed:.2f}s")
        return {"code": 0, "msg": "success", "data": {"answer": answer, "character": character["name"], "retrieved_docs": retrieved_docs, "elapsed": round(elapsed, 2)}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"对话失败：{str(e)}")

@app.post("/api/chat/stream", summary="流式对话接口（SSE，需鉴权）")
def chat_stream(req: ChatRequest, authorization: Optional[str] = Header(None)):
    username = get_current_user(authorization)
    logger.info(f"API流式对话：用户={username}, 角色={req.character_name}, 问题={req.query}")
    def event_generator():
        for event_type, data in rag_chat_stream(username, req.character_name, req.query):
            yield f"event: {event_type}\ndata: {data}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/knowledge/add", summary="添加知识库文本（需鉴权）")
def add_knowledge(req: AddKnowledgeRequest, authorization: Optional[str] = Header(None)):
    username = get_current_user(authorization)
    try:
        vs = get_vs(req.character_name)
        if not vs:
            raise HTTPException(status_code=404, detail="角色不存在")
        vs.insert_texts([req.text])
        logger.info(f"用户{username}添加知识：角色={req.character_name}")
        return {"code": 0, "msg": "添加成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加知识异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"添加知识失败：{str(e)}")

@app.get("/api/knowledge/list", summary="查看知识库文档列表（需鉴权）")
def list_knowledge(character_name: str, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    try:
        vs = get_vs(character_name)
        if not vs:
            raise HTTPException(status_code=404, detail="角色不存在")
        docs = vs.list_texts(limit=100)
        return {"code": 0, "msg": "success", "data": {"count": len(docs), "docs": docs}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查看知识库异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"查看知识库失败：{str(e)}")

@app.post("/api/knowledge/delete", summary="删除知识库文档（需鉴权）")
def delete_knowledge(req: DeleteKnowledgeRequest, authorization: Optional[str] = Header(None)):
    username = get_current_user(authorization)
    try:
        vs = get_vs(req.character_name)
        if not vs:
            raise HTTPException(status_code=404, detail="角色不存在")
        if req.doc_id is not None:
            vs.delete_by_id(req.doc_id)
            logger.info(f"用户{username}删除知识：角色={req.character_name}, ID={req.doc_id}")
            return {"code": 0, "msg": f"已删除ID={req.doc_id}的文档"}
        elif req.text is not None:
            count = vs.delete_by_text(req.text)
            logger.info(f"用户{username}删除知识：角色={req.character_name}, 匹配{count}条")
            return {"code": 0, "msg": f"已删除{count}条匹配文档"}
        else:
            raise HTTPException(status_code=400, detail="请提供doc_id或text参数")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除知识异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"删除知识失败：{str(e)}")

@app.post("/api/knowledge/import", summary="导入文件到知识库（需鉴权）")
def import_file(req: ImportRequest, authorization: Optional[str] = Header(None)):
    username = get_current_user(authorization)
    try:
        vs = get_vs(req.character_name)
        if not vs:
            raise HTTPException(status_code=404, detail="角色不存在")
        if not os.path.exists(req.file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        if req.file_path.lower().endswith(".pdf"):
            blocks = parse_pdf(req.file_path)
        elif req.file_path.lower().endswith(".txt"):
            with open(req.file_path, "r", encoding="utf-8") as f:
                blocks = [f.read()]
        else:
            raise HTTPException(status_code=400, detail="仅支持pdf和txt")
        all_chunks = []
        for block in blocks:
            all_chunks.extend(split_text(block, method="sentence", chunk_size=300))
        vs.insert_texts(all_chunks)
        logger.info(f"用户{username}导入文件：角色={req.character_name}, 块数={len(all_chunks)}")
        return {"code": 0, "msg": "导入成功", "data": {"chunk_count": len(all_chunks)}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导入文件异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"导入文件失败：{str(e)}")

@app.post("/api/history/clear", summary="清空对话历史（需鉴权）")
def clear_history_api(req: ClearHistoryRequest, authorization: Optional[str] = Header(None)):
    username = get_current_user(authorization)
    try:
        session_id = f"{username}_{req.character_name}"
        clear_history(session_id)
        return {"code": 0, "msg": "清空成功"}
    except Exception as e:
        logger.error(f"清空历史异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"清空历史失败：{str(e)}")

@app.get("/api/history/export", summary="导出对话历史（需鉴权）")
def export_history_api(character_name: str, format: str = "markdown", authorization: Optional[str] = Header(None)):
    username = get_current_user(authorization)
    try:
        session_id = f"{username}_{character_name}"
        content = export_history(session_id, character_name, format)
        logger.info(f"用户{username}导出对话历史：角色={character_name}, 格式={format}")
        return {"code": 0, "msg": "导出成功", "data": {"content": content, "format": format}}
    except Exception as e:
        logger.error(f"导出历史异常：{str(e)}")
        raise HTTPException(status_code=500, detail=f"导出失败：{str(e)}")

@app.get("/api/health", summary="健康检查（无需鉴权）")
def health():
    return {"code": 0, "msg": "服务正常", "status": "running"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
