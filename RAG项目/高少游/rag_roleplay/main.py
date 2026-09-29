# -*- coding: utf-8 -*-
"""
RAG 角色扮演系统 - FastAPI 主应用

完整链路：
    文档上传 → 解析 → 切分 → 向量化 → 写入 Milvus
    用户提问 → 多轮记忆(Redis) → 混合检索 → 重排 → 上下文拼装 → LLM → 后处理 → 返回

接口一览：
    GET  /                      健康检查
    GET  /roles                 列出所有可用角色
    POST /ingest/files          上传文件入库（txt/md/docx/wps/doc/pdf）
    POST /ingest/text           粘贴文本入库
    POST /chat                  多轮对话（带角色 + 记忆 + RAG）
    POST /chat/stream           多轮对话流式输出
    POST /cases                 查判例/查原文（不调 LLM）
    POST /memory/save           保存长期记忆
    POST /memory/clear          清空短期记忆
    GET  /kb/sources            列出知识库所有来源及条数
    GET  /kb/sources/{source}   查看某来源的 chunk 详情
    DELETE /kb/sources/{source} 按来源删除（删某个文件的所有 chunk）
    PUT  /kb/update             按来源更新（删旧 → 重新切分 → 写入）
    POST /kb/rebuild            全量重建（删集合 → 重新创建 → 写入）
"""

from contextlib import asynccontextmanager  # 生命周期钩子
from typing import List, Optional  # 类型标注
import os  # 路径操作
import json  # JSON 序列化

import uvicorn  # ASGI 服务器
from fastapi import FastAPI, UploadFile, File, HTTPException, Request  # Web 框架
from fastapi.middleware.cors import CORSMiddleware  # 跨域支持（前端访问需要）
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse  # 流式响应 + 前端页面托管
from fastapi.staticfiles import StaticFiles  # 静态文件服务
from pydantic import BaseModel, Field  # 请求体模型

from config import API_HOST, API_PORT, TOP_K_RERANK  # 配置
from logger import get_logger  # 日志

# 导入各层模块
from db_milvus import (  # Milvus 数据层
    get_embedding, get_vectorstore, get_milvus_client,
    ingest_documents,
    save_long_term_memory, search_long_term_memory,
    fetch_all_text,
)
from db_redis import (  # Redis 短期记忆层
    add_message, get_history_for_llm, clear_session, set_user_active, get_online_users,
)
from db_mysql import (  # MySQL 业务层
    init_db, list_roles, get_role, save_conversation,
    SessionLocal, Role,
)
from doc_parser import extract_text, extract_tables_from_pdf  # 文档解析
from text_splitter import split_text  # 文本切分
from retriever import (  # 检索
    hybrid_search, multi_route_recall,
    refresh_bm25, refresh_all_bm25,
)
from prompt_templates import get_role_template, build_context, ROLE_TEMPLATES  # 角色提示词
from llm_client import chat, stream_chat  # LLM 调用
from post_processor import post_process  # 后处理
from kb_update import (  # 知识库动态更新
    delete_by_source, update_by_source, list_sources,
    get_source_detail, rebuild_collection,
)
from config import get_collection_for_role, MILVUS_COLLECTIONS  # 集合路由

logger = get_logger(__name__)  # 本模块 logger


# ==================== 生命周期：启动时加载模型 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时一次性加载全部模型；关闭时什么都不用做"""
    logger.info("[启动] 开始初始化 RAG 角色扮演系统 ...")

    # 1. MySQL 建表
    init_db()  # 自动 CREATE TABLE IF NOT EXISTS

    # 2. 向量模型（Ollama HTTP 调用，连接即加载）
    get_embedding()
    logger.info("[启动] Ollama bge-m3 向量模型就绪")

    # 3. Milvus：为每个角色初始化集合
    get_milvus_client()
    for role_key, coll_name in MILVUS_COLLECTIONS.items():
        get_vectorstore(coll_name)
    logger.info(f"[启动] Milvus {len(MILVUS_COLLECTIONS)} 个集合连接就绪")

    # 4. 重排模型（首次加载约十几秒）
    from retriever import get_reranker
    get_reranker()
    logger.info("[启动] bge-reranker 重排模型就绪")

    # 5. BM25 索引重建（所有集合）
    refresh_all_bm25()
    logger.info("[启动] BM25 索引就绪")

    # 6. 预置角色写入 MySQL（如果还没有）
    _ensure_default_roles()

    logger.info(f"[启动] 完成！打开 http://127.0.0.1:{API_PORT}/docs 测试")
    yield  # 服务运行期间停在这里
    logger.info("[关闭] 服务退出")


def _ensure_default_roles():
    """
    把预置角色同步到 MySQL：不存在则插入，已存在则更新。

    必须更新已存在的行：对话接口读的是 MySQL 里的 role.system_prompt，
    如果只在"不存在时插入"，改了 prompt_templates.py 里的提示词也不会生效。
    """
    session = SessionLocal()
    try:
        for key, tmpl in ROLE_TEMPLATES.items():
            exists = session.query(Role).filter(Role.name == tmpl["name"]).first()
            if exists:
                # 已存在：同步代码里的最新提示词、描述和温度
                exists.description = tmpl.get("description", "")
                exists.system_prompt = tmpl["system_prompt"]
                exists.temperature = tmpl.get("temperature", 0.5)
            else:
                session.add(Role(
                    name=tmpl["name"],
                    description=tmpl.get("description", ""),
                    system_prompt=tmpl["system_prompt"],
                    temperature=tmpl.get("temperature", 0.5),
                ))
        session.commit()
        logger.info("预置角色已同步到 MySQL（新增 + 更新）")
    finally:
        session.close()


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="RAG 角色扮演系统",
    description="多角色 + 多用户 + 多轮记忆 + 混合检索 + 重排 + 后处理的完整 RAG 系统",
    version="1.0",
    lifespan=lifespan,
)

# CORS：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端静态文件托管：把"前端版"文件夹挂在 /app 路径下
_frontend_dir = os.path.join(os.path.dirname(__file__), "前端版")  # 前端文件目录
app.mount("/app", StaticFiles(directory=_frontend_dir), name="frontend")  # /app/* 访问前端静态资源


# ==================== 请求体模型 ====================
class TextIngestRequest(BaseModel):
    source: str = Field(default="手动输入", description="来源名称")
    text: str = Field(..., description="要入库的原始文本")
    strategy: str = Field(default="paragraph", description="切分策略：fixed/sentence/paragraph/markdown/semantic")
    role_key: str = Field(default="lawyer", description="角色 key：lawyer/psychologist/virtual_friend（决定写入哪个集合）")


class ChatRequest(BaseModel):
    user_id: int = Field(..., description="用户 ID")
    role_id: int = Field(..., description="角色 ID（从 /roles 获取）")
    query: str = Field(..., description="用户消息")
    top_k: int = Field(default=TOP_K_RERANK, ge=1, le=20, description="引用资料条数")
    use_local_llm: bool = Field(default=False, description="是否用本地部署模型")
    role_key: str = Field(default="", description="角色 key：lawyer/psychologist/virtual_friend（决定检索哪个集合，空则按 role_id 查 MySQL")


class CaseRequest(BaseModel):
    query: str = Field(..., description="查询关键词")
    top_k: int = Field(default=TOP_K_RERANK, ge=1, le=20, description="返回条数")
    role_key: str = Field(default="lawyer", description="角色 key：决定检索哪个集合")


class MemorySaveRequest(BaseModel):
    user_id: int = Field(...)
    role_id: int = Field(...)
    content: str = Field(..., description="要记住的内容")
    summary: str = Field(default="", description="摘要")


class MemoryClearRequest(BaseModel):
    user_id: int = Field(...)
    role_id: int = Field(...)


class UpdateDocRequest(BaseModel):
    source: str = Field(..., description="要更新的来源文件名（必须与入库时一致）")
    text: str = Field(..., description="新的完整文本内容")
    strategy: str = Field(default="paragraph", description="切分策略")


class RebuildRequest(BaseModel):
    confirm: bool = Field(default=False, description="必须传 true 才执行全量重建（防误操作）")


# ==================== 接口 ====================

@app.get("/", tags=["系统"])
def root():
    """根路径重定向到前端"""
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/app/index.html">')

@app.get("/health", tags=["系统"])
def health():
    """健康检查"""
    return {"status": "ok", "online_users": get_online_users()}


@app.get("/roles", tags=["角色"])
def get_roles():
    """列出所有可用角色（含 role_key 和建议问题）"""
    roles = list_roles()
    result = []
    for r in roles:
        # 从 ROLE_TEMPLATES 找到对应的 key 和建议
        role_key = None
        suggestions = []
        for k, tmpl in ROLE_TEMPLATES.items():
            if tmpl["name"] == r.name:
                role_key = k
                suggestions = tmpl.get("suggestions", [])
                break
        result.append({
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "temperature": r.temperature,
            "role_key": role_key,
            "suggestions": suggestions,
        })
    return {"roles": result}


@app.post("/ingest/text", tags=["入库"])
def ingest_text(req: TextIngestRequest):
    """粘贴文本入库：按指定角色路由到对应集合"""
    collection_name = get_collection_for_role(req.role_key)
    is_md = req.source.lower().endswith(".md")
    docs = split_text(req.text, source=req.source, strategy=req.strategy, is_markdown=is_md)
    if not docs:
        raise HTTPException(400, "切分结果为空，检查文本内容")
    total = ingest_documents(docs, collection_name=collection_name)
    refresh_bm25(collection_name)  # 刷新该集合的 BM25 索引
    return {"message": "入库成功", "chunks": len(docs), "corpus_total": total,
            "collection": collection_name, "role_key": req.role_key}


@app.post("/ingest/files", tags=["入库"])
def ingest_files(files: List[UploadFile] = File(...), role_key: str = "lawyer"):
    """上传文件入库：支持 txt/md/docx/wps/doc/pdf，按 role_key 路由集合"""
    collection_name = get_collection_for_role(role_key)
    supported = (".txt", ".md", ".docx", ".wps", ".doc", ".pdf")
    all_docs = []
    names = []
    for upload in files:
        filename = upload.filename or "未命名.txt"
        if not filename.lower().endswith(supported):
            raise HTTPException(400, f"暂不支持 {filename}，请上传 txt/md/docx/wps/doc/pdf")
        raw_bytes = upload.file.read()
        raw_text, is_md = extract_text(raw_bytes, filename)
        docs = split_text(raw_text, source=filename, is_markdown=is_md)
        all_docs.extend(docs)
        names.append(filename)
    if not all_docs:
        raise HTTPException(400, "没有可入库的内容")
    total = ingest_documents(all_docs, collection_name=collection_name)
    refresh_bm25(collection_name)
    return {"message": "入库成功", "files": names, "chunks": len(all_docs),
            "corpus_total": total, "collection": collection_name, "role_key": role_key}


@app.post("/cases", tags=["检索"])
def query_cases(req: CaseRequest):
    """查判例/查原文：不调 LLM，直接返回检索重排后的文档片段"""
    collection_name = get_collection_for_role(req.role_key)
    try:
        docs = hybrid_search(req.query, top_k=req.top_k, collection_name=collection_name)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"query": req.query, "count": len(docs), "documents": [
        {"index": i + 1, "content": d.page_content,
         "source": d.metadata.get("source", ""), "section": d.metadata.get("section", "")}
        for i, d in enumerate(docs)
    ]}


@app.post("/chat", tags=["对话"])
def chat_with_role(req: ChatRequest):
    """
    多轮对话（核心接口）：
    1. 从 MySQL 取角色信息（系统提示词 + 温度）
    2. 从 Redis 取短期记忆（最近 N 条对话）
    3. 混合检索知识库 + 长期记忆
    4. 拼装上下文 → 调 LLM → 后处理 → 返回
    5. 把本轮问答存入 Redis 短期记忆 + MySQL 日志
    """
    role = get_role(req.role_id)  # 从 MySQL 取角色
    if not role:
        raise HTTPException(404, f"角色 ID {req.role_id} 不存在")

    # 1. 取短期记忆
    history = get_history_for_llm(req.user_id, req.role_id)

    # 2. 确定角色 key（从前端传或从 MySQL 查）
    role_key = req.role_key
    if not role_key:
        # 从 ROLE_TEMPLATES 按 role.name 反查
        for k, tmpl in ROLE_TEMPLATES.items():
            if tmpl["name"] == role.name:
                role_key = k
                break

    # 3. 多路召回（知识库按角色路由 + 长期记忆）
    try:
        docs = multi_route_recall(req.query, user_id=req.user_id, top_k=req.top_k, role_key=role_key)
    except RuntimeError as e:
        docs = []  # 知识库为空时不阻塞对话
        logger.warning(f"检索失败：{e}")

    # 3. 拼装上下文（人设类角色走"你自己的记忆"框架，其他角色走"参考资料"框架）
    if docs:
        user_message = build_context(docs, req.query, role_key=role_key)  # 资料 + 问题
    else:
        user_message = req.query  # 无资料直接传问题

    # 4. 调 LLM
    try:
        raw_answer = chat(
            system_prompt=role.system_prompt,
            history=history,
            user_message=user_message,
            temperature=role.temperature,
            use_local=req.use_local_llm,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    # 5. 后处理
    source_names = [d.metadata.get("source", "") for d in docs]
    result = post_process(raw_answer, sources=source_names)
    answer = result["answer"]

    # 6. 存短期记忆（Redis）
    add_message(req.user_id, req.role_id, "user", req.query)
    add_message(req.user_id, req.role_id, "assistant", answer)

    # 7. 存对话日志（MySQL）
    save_conversation(req.user_id, req.role_id, role.name, req.query, answer,
                      json.dumps(source_names, ensure_ascii=False))

    # 8. 标记用户在线
    set_user_active(req.user_id)

    return {
        "answer": answer,
        "role_name": role.name,
        "sources": [{"index": i + 1, "content": d.page_content[:200],
                     "source": d.metadata.get("source", "")}
                    for i, d in enumerate(docs)],
        "citations": result["citations"],
    }


@app.post("/chat/stream", tags=["对话"])
def chat_stream(req: ChatRequest):
    """
    多轮对话流式输出（SSE）

    响应协议：第一条消息是引用来源标记行，格式为
        ###SOURCES###[{"index":1,"content":"...","source":"..."}]\n
    之后才是逐个 token 的正文。前端剥离首行渲染引用卡片，正文原样显示。

    这样一次请求就能同时拿到回答和来源。旧版前端会再调一次 /chat 取来源，
    导致 LLM 被调两次、Redis 记忆存两遍、当前问题在历史里重复出现。
    """
    role = get_role(req.role_id)
    if not role:
        raise HTTPException(404, f"角色 ID {req.role_id} 不存在")

    history = get_history_for_llm(req.user_id, req.role_id)

    # 确定角色 key
    role_key = req.role_key
    if not role_key:
        for k, tmpl in ROLE_TEMPLATES.items():
            if tmpl["name"] == role.name:
                role_key = k
                break

    try:
        docs = multi_route_recall(req.query, user_id=req.user_id, top_k=req.top_k, role_key=role_key)
    except RuntimeError:
        docs = []

    # 人设类角色走"你自己的记忆"框架，其他角色走"参考资料"框架
    user_message = build_context(docs, req.query, role_key=role_key) if docs else req.query

    # 检索在流开始前就完成了，引用来源可以提前组装好
    sources_payload = [
        {"index": i + 1, "content": d.page_content[:200], "source": d.metadata.get("source", "")}
        for i, d in enumerate(docs)
    ]
    source_names = [d.metadata.get("source", "") for d in docs]

    def generate():
        # 第一条消息：引用来源行。json.dumps 会把内容里的真实换行转义成 \n 两个字符，
        # 所以这一行内部不会出现裸换行，前端按 \n 切分不会截断 JSON
        yield "###SOURCES###" + json.dumps(sources_payload, ensure_ascii=False) + "\n"

        full_answer = ""
        try:
            for chunk in stream_chat(
                system_prompt=role.system_prompt,
                history=history,
                user_message=user_message,
                temperature=role.temperature,
                use_local=req.use_local_llm,
            ):
                full_answer += chunk
                yield chunk  # 逐 chunk 返回给前端
        except RuntimeError as e:
            yield f"\n[错误] {e}"
            return

        # 流完后存短期记忆（Redis）
        add_message(req.user_id, req.role_id, "user", req.query)
        add_message(req.user_id, req.role_id, "assistant", full_answer)

        # 存对话日志（MySQL）：原先只有 /chat 写日志，改成单请求后必须在这里补上，
        # 否则前端只调 /chat/stream，MySQL 里就再也看不到对话记录了
        save_conversation(req.user_id, req.role_id, role.name, req.query, full_answer,
                          json.dumps(source_names, ensure_ascii=False))

        # 标记用户在线（原先只有 /chat 做，同样需要补上）
        set_user_active(req.user_id)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/memory/save", tags=["记忆"])
def save_memory(req: MemorySaveRequest):
    """保存长期记忆到 Milvus"""
    save_long_term_memory(req.user_id, req.role_id, req.content, req.summary)
    return {"message": "长期记忆已保存"}


@app.post("/memory/clear", tags=["记忆"])
def clear_memory(req: MemoryClearRequest):
    """清空短期记忆（Redis）"""
    clear_session(req.user_id, req.role_id)
    return {"message": "短期记忆已清空"}


# ==================== 知识库动态更新 ====================

@app.get("/kb/sources", tags=["知识库管理"])
def kb_list_sources(role_key: str = "lawyer"):
    """列出指定角色知识库中所有文档来源及其 chunk 数量"""
    collection_name = get_collection_for_role(role_key)
    sources = list_sources(collection_name)
    return {"total_sources": len(sources), "sources": sources, "collection": collection_name}


@app.get("/kb/sources/{source}", tags=["知识库管理"])
def kb_get_source(source: str, role_key: str = "lawyer"):
    """查看某个来源的 chunk 列表（前 100 条，截取前 200 字预览）"""
    collection_name = get_collection_for_role(role_key)
    detail = get_source_detail(source, collection_name)
    return detail


@app.delete("/kb/sources/{source}", tags=["知识库管理"])
def kb_delete_source(source: str, role_key: str = "lawyer"):
    """按来源删除：删掉某个文件的所有 chunk（文档废弃时用）"""
    collection_name = get_collection_for_role(role_key)
    deleted = delete_by_source(source, collection_name)
    if deleted == 0:
        raise HTTPException(404, f"来源 '{source}' 不在知识库中")
    refresh_bm25(collection_name)
    return {"message": f"已删除来源 '{source}' 的 {deleted} 条 chunk", "deleted": deleted,
            "collection": collection_name}


@app.put("/kb/update", tags=["知识库管理"])
def kb_update_doc(req: UpdateDocRequest, role_key: str = "lawyer"):
    """按来源更新：删旧 chunk -> 重新切分 -> 重新写入"""
    collection_name = get_collection_for_role(role_key)
    is_md = req.source.lower().endswith(".md")
    result = update_by_source(
        source=req.source,
        new_text=req.text,
        is_markdown=is_md,
        strategy=req.strategy,
        collection_name=collection_name,
    )
    refresh_bm25(collection_name)
    return result


@app.post("/kb/rebuild", tags=["知识库管理"])
def kb_rebuild(req: RebuildRequest, role_key: str = "lawyer"):
    """全量重建：删集合 -> 建新集合 -> 全量写入"""
    if not req.confirm:
        raise HTTPException(400, "请确认操作：传 confirm=true 才执行全量重建")
    collection_name = get_collection_for_role(role_key)
    existing_docs = fetch_all_text(collection_name)
    result = rebuild_collection(existing_docs, collection_name)
    refresh_bm25(collection_name)
    return result


# ==================== 启动入口 ====================
if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
