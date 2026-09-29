# -*- coding: utf-8 -*-
"""
PersonaRAG · 多角色智能顾问系统 - FastAPI 主应用

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

本模块在系统中的位置：
    上游：前端页面（挂在 /app 下的「web」目录，纯原生 html/css/js，含对话/检索台/知识库三个视图）
          通过 HTTP 调这些接口；文件末尾的 __main__ 用 uvicorn 把 app 起在 0.0.0.0:8000。
    下游：本模块只做「编排」，真正的算法都在各层模块里——
          db_mysql（用户/角色/对话日志）、db_redis（短期记忆窗口）、
          db_milvus + retriever（向量召回 + BM25 融合 + 重排）、llm_client（DeepSeek 调用）、
          doc_parser / text_splitter（入库解析与切分）、kb_update（知识库增删改）、
          post_processor（回答后处理，抽取引用）。

关键设计取舍：
    1. 模型只加载一次：bge-m3 与 bge-reranker 体积大、加载慢，统一放在 lifespan 启动阶段加载，
       之后靠各模块的模块级单例复用，绝不可以在请求处理里重新加载。
    2. 接口都写成同步 def：FastAPI 会把同步函数丢到线程池执行，不会阻塞事件循环；
       而 Milvus/Redis/MySQL 客户端本身都是同步库，写同步代码最直接。
    3. /cases 与 /chat 分开：前者只检索不调 LLM，也不写任何记忆，供前端「检索台」看引用原文；
       后者才走完整对话链路（记忆 → 检索 → 生成 → 落库）。
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
    """服务启动时一次性加载全部模型；关闭时什么都不用做

    这是 FastAPI 的 lifespan 生命周期钩子：yield 之前的代码在服务启动时执行，
    yield 之后的代码在进程收到关闭信号时执行。

    启动顺序不可随意调换，每一步都被后面的步骤依赖：
        建表 → 向量模型 → Milvus 连接/建集合 → 重排模型 → BM25 索引 → 预置角色。
        比如先把向量模型加载好，后面初始化 Milvus 集合时才能把 embedding 函数绑上去；
        BM25 依赖「库里现在有哪些文本」，必须在集合就绪之后重建。

    参数：
        app：FastAPI 应用对象，由框架自动传入，本函数内不使用。

    异常与降级：
        任何一步失败（模型路径写错、Milvus 连不上）都会让服务启动直接失败——这是有意为之：
        带病启动只会让之后每个请求都报错，不如启动时就暴露问题。
    """
    logger.info("[启动] 开始初始化 PersonaRAG · 多角色智能顾问系统 ...")

    # 1. MySQL 建表
    init_db()  # 自动 CREATE TABLE IF NOT EXISTS：已存在的表不会被覆盖，重复启动是安全的

    # 2. 向量模型（本地 bge-m3 权重，加载即用）
    get_embedding()  # 首次调用才真正读盘加载权重（约十几秒），之后全局复用同一实例
    logger.info("[启动] 本地 bge-m3 向量模型就绪")

    # 3. Milvus：为每个角色初始化集合
    get_milvus_client()  # 建立到 Milvus 的连接（模块级单例，后续所有查询复用）
    for role_key, coll_name in MILVUS_COLLECTIONS.items():
        get_vectorstore(coll_name)  # 逐个集合检查/创建并绑定向量模型：启动阶段就暴露连接与建表问题
    logger.info(f"[启动] Milvus {len(MILVUS_COLLECTIONS)} 个集合连接就绪")

    # 4. 重排模型（首次加载约十几秒）
    from retriever import get_reranker  # 延迟导入：retriever 也会 import 本模块的配置，放函数内避免循环导入
    get_reranker()  # CPU 上跑 HuggingFaceCrossEncoder，同样只加载一次，请求时直接复用
    logger.info("[启动] bge-reranker 重排模型就绪")

    # 5. BM25 索引重建（所有集合）
    refresh_all_bm25()  # 必须放在集合就绪之后：BM25 要先把库里已有文本读出来建jieba倒排索引
    logger.info("[启动] BM25 索引就绪")

    # 6. 预置角色写入 MySQL（如果还没有）
    _ensure_default_roles()  # 幂等操作：角色已存在就跳过，不会重复插入

    logger.info(f"[启动] 完成！打开 http://127.0.0.1:{API_PORT}/docs 测试")
    yield  # 服务运行期间停在这里：此后的代码只在进程关闭时才会执行
    logger.info("[关闭] 服务退出")  # 模型、Milvus/Redis/MySQL 客户端都由进程退出统一回收，无需手动释放


def _ensure_default_roles():
    """把预置角色写入 MySQL（角色不存在才插入，已存在跳过）

    调用方：lifespan 启动阶段的第 6 步。
    数据来源：prompt_templates.ROLE_TEMPLATES（角色 key -> 角色模板字典）。
    幂等策略：按角色名先查一次，存在就跳过，所以反复重启服务不会插入重复角色。
    异常与降级：无论中途是否抛异常，finally 都会关闭会话，不会泄漏数据库连接；
                即使插入失败，服务仍能启动，只是 /roles 里会少预置角色。
    """
    session = SessionLocal()  # 每次操作单独开一个会话，用完即关（SQLAlchemy 的标准用法）
    try:
        for key, tmpl in ROLE_TEMPLATES.items():
            exists = session.query(Role).filter(Role.name == tmpl["name"]).first()  # 按角色名查重
            if not exists:
                session.add(Role(  # 只有不存在才插入
                    name=tmpl["name"],
                    description=tmpl.get("description", ""),  # 用 get 取可选字段：模板里没写就落空串
                    system_prompt=tmpl["system_prompt"],
                    temperature=tmpl.get("temperature", 0.5),  # 没配温度就取 0.5，偏稳重的默认值
                ))
        session.commit()  # 循环外统一提交，只走一次事务
        logger.info("预置角色已写入 MySQL")
    finally:
        session.close()  # 成败都关闭，避免连接池被占满


# ==================== FastAPI 应用 ====================
# lifespan 把上面写的启动/关闭逻辑注册进来；title/description/version 会显示在 /docs 页面上
app = FastAPI(
    title="PersonaRAG · 多角色智能顾问系统",
    description="多角色 + 多用户 + 多轮记忆 + 混合检索 + 重排 + 后处理的完整 RAG 系统",
    version="1.0",
    lifespan=lifespan,
)

# CORS：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 教学项目图省事放开所有来源；生产环境应改成明确的白名单
    allow_methods=["*"],  # 放开全部方法（GET/POST/PUT/DELETE 前端都会用到）
    allow_headers=["*"],
)

# 前端静态文件托管：把 "web" 文件夹挂在 /app 路径下
_frontend_dir = os.path.join(os.path.dirname(__file__), "web")  # 前端文件目录：用 __file__ 拼绝对路径，任意工作目录启动都能找到
app.mount("/app", StaticFiles(directory=_frontend_dir), name="frontend")  # /app/* 访问前端静态资源（index.html、css、js）


# ==================== 请求体模型 ====================
# 下面这些类都是 pydantic 的 BaseModel：FastAPI 会自动把请求 JSON 按字段类型解析并校验
# （类型不符直接返回 422），请求体结构也会自动出现在 /docs 里，所以接口函数里不用再手工校验类型。
# Field(...) 里的 ... 表示该字段必填；Field(default=xxx) 表示可省略。
class TextIngestRequest(BaseModel):
    source: str = Field(default="手动输入", description="来源名称")  # 来源名，也是之后 /kb/sources 查询、更新、删除时的定位键；来自前端「入库」表单的来源输入框
    text: str = Field(..., description="要入库的原始文本")  # 必填；前端粘贴框里的原文
    strategy: str = Field(default="paragraph", description="切分策略：fixed/sentence/paragraph/markdown/semantic")  # 切分策略，来自前端策略下拉框；若 source 以 .md 结尾会被强制按 markdown 切
    role_key: str = Field(default="lawyer", description="角色 key：lawyer/psychologist/virtual_friend（决定写入哪个集合）")  # 角色 key，决定写入 rag_legal / rag_psychology / rag_companion 中的哪一个


class ChatRequest(BaseModel):
    user_id: int = Field(..., description="用户 ID")  # 必填；短期记忆和长期记忆都按 (user_id, role_id) 隔离，前端用固定的测试账号即可
    role_id: int = Field(..., description="角色 ID（从 /roles 获取）")  # 必填；MySQL 角色表主键，用来取出系统提示词和温度
    query: str = Field(..., description="用户消息")  # 必填；用户本轮说的话，同时当作检索 query
    top_k: int = Field(default=TOP_K_RERANK, ge=1, le=20, description="引用资料条数")  # 重排后保留的资料条数，限制 1~20（越大越慢、上下文越长）；前端「引用条数」控件
    use_local_llm: bool = Field(default=False, description="是否用本地部署模型")  # 默认走云端 DeepSeek；本地模型一般只作断网时的降级方案
    role_key: str = Field(default="", description="角色 key：lawyer/psychologist/virtual_friend（决定检索哪个集合，空则按 role_id 查 MySQL）")  # 决定检索哪个集合；留空时按角色名去 ROLE_TEMPLATES 反查


class CaseRequest(BaseModel):
    query: str = Field(..., description="查询关键词")  # 必填；检索用语，来自前端「检索台」输入框
    top_k: int = Field(default=TOP_K_RERANK, ge=1, le=20, description="返回条数")  # 返回条数，同样限制 1~20
    role_key: str = Field(default="lawyer", description="角色 key：决定检索哪个集合")  # 决定检索哪个集合，默认查法律库


class MemorySaveRequest(BaseModel):
    user_id: int = Field(...)  # 必填；这条长期记忆属于哪个用户
    role_id: int = Field(...)  # 必填；属于哪个角色（同一用户在不同角色下的记忆互不干扰）
    content: str = Field(..., description="要记住的内容")  # 必填；写进 Milvus 长期记忆集合的正文
    summary: str = Field(default="", description="摘要")  # 可选；一句话摘要，方便展示和过滤


class MemoryClearRequest(BaseModel):
    user_id: int = Field(...)  # 必填；要清空谁
    role_id: int = Field(...)  # 必填；只清空这个用户在这个角色下的短期记忆（Redis），长期记忆不受影响


class UpdateDocRequest(BaseModel):
    source: str = Field(..., description="要更新的来源文件名（必须与入库时一致）")  # 必填；必须和入库时的 source 完全一致，否则删不到旧 chunk，效果等于又新增了一份
    text: str = Field(..., description="新的完整文本内容")  # 必填；整篇新文本，不是增量 diff
    strategy: str = Field(default="paragraph", description="切分策略")  # 重新切分时用的策略


class RebuildRequest(BaseModel):
    confirm: bool = Field(default=False, description="必须传 true 才执行全量重建（防误操作）")  # 默认 false：重建会先 drop 整个集合，属于破坏性操作，必须显式确认


# ==================== 接口 ====================

@app.get("/", tags=["系统"])
def root():
    """根路径重定向到前端

    返回：一段 HTML，内容是一个 meta refresh，让浏览器立刻跳到 /app/index.html。
    用途：用户只输 http://127.0.0.1:8000 也能进到前端页面，不必记后面那一串路径。
    """
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/app/index.html">')  # 交给浏览器跳转，服务端不做 302，简单且不依赖前端路由

@app.get("/health", tags=["系统"])
def health():
    """健康检查

    返回：{"status": "ok", "online_users": [...]}
          status 固定为 "ok"（能返回就说明进程活着）；online_users 是 Redis 里记录的在线用户 ID 列表。
    说明：只读 Redis，不碰 Milvus 也不调 LLM，所以这个接口足够快，可以给前端定时轮询探活。
    """
    return {"status": "ok", "online_users": get_online_users()}


@app.get("/roles", tags=["角色"])
def get_roles():
    """列出所有可用角色（含 role_key 和建议问题）

    调用方：前端「对话」视图加载时调用，用来渲染角色列表和快捷提问。
    返回：{"roles": [{"id"（MySQL 主键，/chat 传的 role_id）,
                      "name"（角色名）,
                      "description"（角色简介）,
                      "temperature"（采样温度）,
                      "role_key"（决定检索哪个 Milvus 集合，匹配不到时为 None）,
                      "suggestions"（建议问题列表，前端展示为可点按钮）}, ...]}
    设计说明：角色基础信息存在 MySQL，而 role_key 与 suggestions 只写在 prompt_templates 的模板里，
              所以这里要按角色名把两边拼起来（角色数量很少，线性查找足够）。
    """
    roles = list_roles()  # 从 MySQL 取全部角色
    result = []
    for r in roles:
        # 从 ROLE_TEMPLATES 找到对应的 key 和建议
        role_key = None  # 模板里没匹配到就保持 None，前端需要兼容这种情况
        suggestions = []
        for k, tmpl in ROLE_TEMPLATES.items():
            if tmpl["name"] == r.name:  # 用角色名做两边对齐的关联键
                role_key = k
                suggestions = tmpl.get("suggestions", [])
                break  # 命中即停，模板数量很少，无需建索引
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
    """粘贴文本入库：按指定角色路由到对应集合

    参数（TextIngestRequest）：
        source   来源名称，同时是之后按来源查询/更新/删除的定位键；以 .md 结尾则按 Markdown 标题切分。
        text     要入库的原始文本，前端粘贴框内容。
        strategy 切分策略，默认 paragraph（按段落）。
        role_key 角色 key，决定写入哪个集合（lawyer -> rag_legal）。
    返回：{"message","chunks"（本次写入的块数）,"corpus_total"（该集合写入后的总条数）,
          "collection"（实际写入的集合名）,"role_key"}
    异常：切分结果为空时抛 400，不会写入 Milvus；Milvus 不可用时会向上冒泡为 500。
    注意：写库之后必须刷新 BM25，否则新文本只能被向量召回、关键词检索搜不到。
    """
    collection_name = get_collection_for_role(req.role_key)  # 1. 角色路由：角色 key -> Milvus 集合名
    is_md = req.source.lower().endswith(".md")  # 2. 按后缀判断是否 Markdown；用 lower 兼容 .MD
    docs = split_text(req.text, source=req.source, strategy=req.strategy, is_markdown=is_md)  # 切分成带元数据（source/section/chunk_index）的 Document
    if not docs:
        raise HTTPException(400, "切分结果为空，检查文本内容")  # 空文本直接拒绝，避免写进空 chunk 污染检索
    total = ingest_documents(docs, collection_name=collection_name)  # 3. 向量化 + 批量写入 Milvus
    refresh_bm25(collection_name)  # 刷新该集合的 BM25 索引：新增内容必须同时能被关键词召回
    return {"message": "入库成功", "chunks": len(docs), "corpus_total": total,
            "collection": collection_name, "role_key": req.role_key}


@app.post("/ingest/files", tags=["入库"])
def ingest_files(files: List[UploadFile] = File(...), role_key: str = "lawyer"):
    """上传文件入库：支持 txt/md/docx/wps/doc/pdf，按 role_key 路由集合

    参数：
        files    multipart 上传的文件列表（可多选），来自前端「入库」视图的文件选择控件；
                 单个文件解析失败或格式不支持会直接抛 400，不会「部分成功」。
        role_key 角色 key（表单字段，不是 JSON），决定写入的集合；默认 lawyer。
    返回：{"message","files"（成功入库的文件名列表）,"chunks"（本次总块数）,
          "corpus_total"（集合总条数）,"collection","role_key"}
    异常：文件后缀不在白名单抛 400；所有文件切分结果都为空抛 400。
    取舍说明：多个文件先在内存里解析切分、汇总后一次性写库，减少 Milvus 交互次数；
              代价是大文件会占用较多内存，教学场景的文件体量可以接受。
    """
    collection_name = get_collection_for_role(role_key)  # 1. 角色路由
    supported = (".txt", ".md", ".docx", ".wps", ".doc", ".pdf")  # 2. 后缀白名单，防止把二进制垃圾塞进库里
    all_docs = []
    names = []
    for upload in files:
        filename = upload.filename or "未命名.txt"  # 某些客户端可能不带文件名，给个兜底
        if not filename.lower().endswith(supported):
            raise HTTPException(400, f"暂不支持 {filename}，请上传 txt/md/docx/wps/doc/pdf")
        raw_bytes = upload.file.read()  # 一次性读入内存；doc_parser 内部按后缀选择解析器
        raw_text, is_md = extract_text(raw_bytes, filename)  # 解析：PDF 抽文字、docx/wps 走转换、md 标记格式
        docs = split_text(raw_text, source=filename, is_markdown=is_md)  # 用文件名作为 source，便于之后按来源管理
        all_docs.extend(docs)  # 攒起来统一入库
        names.append(filename)
    if not all_docs:
        raise HTTPException(400, "没有可入库的内容")  # 全部为空（例如扫描版 PDF）时明确报错
    total = ingest_documents(all_docs, collection_name=collection_name)  # 3. 统一向量化写入
    refresh_bm25(collection_name)  # 4. 刷新 BM25
    return {"message": "入库成功", "files": names, "chunks": len(all_docs),
            "corpus_total": total, "collection": collection_name, "role_key": role_key}


@app.post("/cases", tags=["检索"])
def query_cases(req: CaseRequest):
    """查判例/查原文：不调 LLM，直接返回检索重排后的文档片段

    参数（CaseRequest）：
        query    查询关键词，前端「检索台」输入框。
        top_k    返回条数（1~20），默认取配置里的 TOP_K_RERANK。
        role_key 角色 key，决定检索哪个集合。
    返回：{"query","count"（命中条数）,
          "documents": [{"index"（从 1 开始，方便前端编号展示）, "content"（整段原文，不截断）,
                         "source"（来源文件）, "section"（章节标题）}, ...]}
    设计取舍：
        1. 不调 LLM：这个接口是给前端看「检索到了什么」用的，人工核对引用来源时最需要的是原文，
           过一遍模型既费钱又费时，还可能改写成幻觉。
        2. 不写记忆：既不写 Redis 短期记忆也不写 MySQL 日志，所以可以随便重复调用，
           不会污染真正对话的上下文。
    异常：检索链路不可用（Milvus 未就绪等）时 hybrid_search 抛 RuntimeError，这里转成 503。
    """
    collection_name = get_collection_for_role(req.role_key)  # 1. 角色路由
    try:
        docs = hybrid_search(req.query, top_k=req.top_k, collection_name=collection_name)  # 2. 向量 + BM25 融合检索，再经重排、阈值过滤
    except RuntimeError as e:
        raise HTTPException(503, str(e))  # 依赖不可用时返回 503，前端可以提示「知识库服务不可用」
    return {"query": req.query, "count": len(docs), "documents": [  # 3. 只把内容与来源摊平成 JSON，Document 对象不能直接序列化
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

    参数（ChatRequest）：user_id 用户 ID；role_id 角色 ID；query 用户消息；
        top_k 引用资料条数（1~20）；use_local_llm 是否用本地模型；role_key 指定检索集合（可空）。
    返回：{"answer"（后处理后的回答）,"role_name"（角色名）,
          "sources": [{"index","content"（前 200 字预览，只给前端展示用）,"source"}, ...],
          "citations"（post_process 从回答里抽出的引用标记）}
    异常与降级：
        - 角色不存在：404；
        - 检索失败（知识库为空、Milvus 抖动）：不抛异常，docs 置空后照常作答（无资料对话）；
        - LLM 调用失败：503。
    说明：这是非流式版本，前端点击发送后等完整回答；需要打字机效果时用 /chat/stream。
    """
    role = get_role(req.role_id)  # 从 MySQL 取角色
    if not role:
        raise HTTPException(404, f"角色 ID {req.role_id} 不存在")

    # 1. 取短期记忆
    history = get_history_for_llm(req.user_id, req.role_id)  # 只取最近 MEMORY_WINDOW(=20) 条，控制提示词长度与成本

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
        docs = multi_route_recall(req.query, user_id=req.user_id, top_k=req.top_k, role_key=role_key)  # 知识库集合 + 该用户的长期记忆集合一起召回
    except RuntimeError as e:
        docs = []  # 知识库为空时不阻塞对话：没资料也让角色按自身人设回答
        logger.warning(f"检索失败：{e}")

    # 3. 拼装上下文
    # 注意编号沿用原代码（实际是第 4 步）：有召回结果就把「参考资料 + 问题」一起给模型，
    # 没有就只传问题，避免塞一段空的资料模板进去干扰模型。
    if docs:
        user_message = build_context(docs, req.query)  # 资料 + 问题
    else:
        user_message = req.query  # 无资料直接传问题

    # 4. 调 LLM
    try:
        raw_answer = chat(
            system_prompt=role.system_prompt,  # 角色人设（含回答风格约束）
            history=history,  # 短期记忆：让「他/对方/那个」这类代词能对上上文
            user_message=user_message,  # 本轮输入（含参考资料）
            temperature=role.temperature,  # 每个角色可配不同温度：法律顾问偏低、陪伴角色偏高
            use_local=req.use_local_llm,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))  # LLM 侧失败（超时/额度）直接告知前端，不写记忆

    # 5. 后处理
    source_names = [d.metadata.get("source", "") for d in docs]  # 去重前的来源列表，后处理会据此生成引用
    result = post_process(raw_answer, sources=source_names)  # 清理模型残留的思考过程、抽取引用
    answer = result["answer"]

    # 6. 存短期记忆（Redis）
    # 放在回答成功之后写：失败时不会留下「用户问了但角色没答」的残缺上下文
    add_message(req.user_id, req.role_id, "user", req.query)  # 先存用户问题
    add_message(req.user_id, req.role_id, "assistant", answer)  # 再存回答，保证顺序

    # 7. 存对话日志（MySQL）
    save_conversation(req.user_id, req.role_id, role.name, req.query, answer,
                      json.dumps(source_names, ensure_ascii=False))  # ensure_ascii=False：中文按原样存，便于直接看库

    # 8. 标记用户在线
    set_user_active(req.user_id)  # 写 Redis 活跃标记，/health 的 online_users 就是从这里来的

    return {
        "answer": answer,
        "role_name": role.name,
        "sources": [{"index": i + 1, "content": d.page_content[:200],  # 只截 200 字预览，完整原文走 /cases
                     "source": d.metadata.get("source", "")}
                    for i, d in enumerate(docs)],
        "citations": result["citations"],  # 回答里出现的引用标记，前端可高亮
    }


@app.post("/chat/stream", tags=["对话"])
def chat_stream(req: ChatRequest):
    """多轮对话流式输出（SSE）

    参数：同 /chat（ChatRequest），前端在流式模式下边收边渲染，得到打字机效果。
    返回：StreamingResponse，media_type="text/event-stream"，响应体是纯文本片段流，
          前端用 fetch + ReadableStream 逐段读取追加即可。
    与 /chat 的差异：
        1. 这里只写 Redis 短期记忆、不写 MySQL 对话日志（保持流式路径轻量，少一次数据库往返）；
        2. 检索与 LLM 的异常都在流内处理：检索失败降级为无资料作答，LLM 失败时把错误文本
           拼在流末尾返回，而不是抛 HTTPException——因为响应头早就发出去了，改不了状态码。
    关键取舍（为什么流结束后才写记忆）：
        generate() 是生成器，chunk 是边生成边吐给前端的。只有等循环正常结束，
        full_answer 才是完整的回答；若在流开始前就写记忆，或中途断开时写记忆，
        都会把半截回答塞进历史，导致下一轮对话上下文错乱。
    """
    role = get_role(req.role_id)
    if not role:
        raise HTTPException(404, f"角色 ID {req.role_id} 不存在")

    history = get_history_for_llm(req.user_id, req.role_id)  # 同样只取最近 MEMORY_WINDOW 条

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
        docs = []  # 与 /chat 一致：检索不可用也要能聊，只是没有参考资料

    user_message = build_context(docs, req.query) if docs else req.query  # 有资料拼上下文，无资料直接传问题

    def generate():
        """流式生成器：逐个 chunk 产出给前端，流结束后才落记忆

        返回：生成器，yield 的是 LLM 输出的一小段文本。
        异常：LLM 报错时不中断连接，而是 yield 一段 [错误] 文本让前端展示。
        """
        full_answer = ""  # 累积完整回答，用于最后写记忆
        try:
            for chunk in stream_chat(
                system_prompt=role.system_prompt,
                history=history,
                user_message=user_message,
                temperature=role.temperature,
                use_local=req.use_local_llm,
            ):
                full_answer += chunk  # 边收边攒
                yield chunk  # 逐 chunk 返回给前端
        except RuntimeError as e:
            yield f"\n[错误] {e}"  # 状态码已经发出去了，改不了，只能把错误当正文吐出来
            return  # 直接结束，不写记忆：这次回答是残缺的

        # 流完后存记忆
        add_message(req.user_id, req.role_id, "user", req.query)  # 到这里才确定回答完整，可以安全入库
        add_message(req.user_id, req.role_id, "assistant", full_answer)

    return StreamingResponse(generate(), media_type="text/event-stream")  # 生成器是惰性执行的：请求处理到这里就开始边算边发


@app.post("/memory/save", tags=["记忆"])
def save_memory(req: MemorySaveRequest):
    """保存长期记忆到 Milvus

    参数（MemorySaveRequest）：user_id 用户 ID；role_id 角色 ID；
        content 要记住的内容（会被向量化）；summary 摘要（可选，用于展示）。
    返回：{"message": "长期记忆已保存"}
    说明：长期记忆单独放在 rag_long_term_memory 集合，靠 user_id/role_id 元数据区分归属，
          检索时与知识库一起参与多路召回，让角色「记得」这个用户说过的事。
    """
    save_long_term_memory(req.user_id, req.role_id, req.content, req.summary)
    return {"message": "长期记忆已保存"}


@app.post("/memory/clear", tags=["记忆"])
def clear_memory(req: MemoryClearRequest):
    """清空短期记忆（Redis）

    参数（MemoryClearRequest）：user_id 用户 ID；role_id 角色 ID。
    返回：{"message": "短期记忆已清空"}
    说明：只清 Redis 里的对话窗口，Milvus 里的长期记忆不受影响；
          前端「新会话」按钮调的就是这个接口。
    """
    clear_session(req.user_id, req.role_id)  # 按 (用户, 角色) 删掉整个会话 key
    return {"message": "短期记忆已清空"}


# ==================== 知识库动态更新 ====================

@app.get("/kb/sources", tags=["知识库管理"])
def kb_list_sources(role_key: str = "lawyer"):
    """列出指定角色知识库中所有文档来源及其 chunk 数量

    参数：role_key 查询参数（不是 JSON body），默认 lawyer，来自前端「知识库」视图的角色切换。
    返回：{"total_sources"（来源文件个数）,"sources": [{"source"（来源名）,"chunks"（该来源的块数）}, ...],
          "collection"（实际查询的集合名）}
    实现说明：Milvus 没有 group by，list_sources 里用 offset 分页把 source 字段捞回来在 Python 侧计数，
              所以来源很多时这个接口会偏慢，属于教学项目的简化做法。
    """
    collection_name = get_collection_for_role(role_key)
    sources = list_sources(collection_name)  # 已在 kb_update 里按块数倒序排好，前端直接渲染
    return {"total_sources": len(sources), "sources": sources, "collection": collection_name}


@app.get("/kb/sources/{source}", tags=["知识库管理"])
def kb_get_source(source: str, role_key: str = "lawyer"):
    """查看某个来源的 chunk 列表（前 100 条，截取前 200 字预览）

    参数：source 路径参数，来源名（就是入库时的 source，注意含特殊字符时前端要做 URL 编码）；
          role_key 查询参数，决定查哪个集合，默认 lawyer。
    返回：{"source","count","chunks": [{"pk"（Milvus 主键）,"text"（前 200 字）,
          "section"（章节）,"chunk_index"（块序号）}, ...], "collection"}
    说明：只取前 100 条并截断文本，避免一次把整个来源的全文灌给前端。
    """
    collection_name = get_collection_for_role(role_key)
    detail = get_source_detail(source, collection_name)  # 集合不存在时返回空结构而不是报错
    return detail


@app.delete("/kb/sources/{source}", tags=["知识库管理"])
def kb_delete_source(source: str, role_key: str = "lawyer"):
    """按来源删除：删掉某个文件的所有 chunk（文档废弃时用）

    参数：source 来源名（路径参数）与入库时一致；role_key 决定操作哪个集合。
    返回：{"message","deleted"（删除条数）,"collection"}
    异常：该来源一条都没删到时返回 404，避免前端误以为删除成功。
    注意：删完必须刷新 BM25，否则缓存里还留着已删文本的倒排索引，会召回到不存在的文档。
    """
    collection_name = get_collection_for_role(role_key)
    deleted = delete_by_source(source, collection_name)
    if deleted == 0:
        raise HTTPException(404, f"来源 '{source}' 不在知识库中")
    refresh_bm25(collection_name)  # 索引与数据保持一致
    return {"message": f"已删除来源 '{source}' 的 {deleted} 条 chunk", "deleted": deleted,
            "collection": collection_name}


@app.put("/kb/update", tags=["知识库管理"])
def kb_update_doc(req: UpdateDocRequest, role_key: str = "lawyer"):
    """按来源更新：删旧 chunk -> 重新切分 -> 重新写入

    参数：
        req（UpdateDocRequest）：source 来源名（必须与入库时一致）；text 新的完整文本；strategy 切分策略。
        role_key 查询参数，决定操作哪个集合，默认 lawyer。
    返回：kb_update.update_by_source 的结果字典 {"source","deleted","inserted","corpus_total","collection"}；
          若新文本切分为空，则返回 {"source","deleted","inserted"=0,"warning"} 提示旧数据已删但未写入。
    注意：这实质是「先删后加」，如果新文本切分失败会出现短暂的「该来源无数据」窗口，
          是全量替换而非增量 patch，前端应提示用户传完整内容。
    """
    collection_name = get_collection_for_role(role_key)
    is_md = req.source.lower().endswith(".md")  # 与入库时保持同一套后缀判断，保证切分方式一致
    result = update_by_source(
        source=req.source,
        new_text=req.text,
        is_markdown=is_md,
        strategy=req.strategy,
        collection_name=collection_name,
    )
    refresh_bm25(collection_name)  # 内容变了，BM25 必须跟着重建
    return result


@app.post("/kb/rebuild", tags=["知识库管理"])
def kb_rebuild(req: RebuildRequest, role_key: str = "lawyer"):
    """全量重建：删集合 -> 建新集合 -> 全量写入

    参数：
        req（RebuildRequest）：confirm 必须传 true，否则直接拒绝（防误操作）。
        role_key 查询参数，决定重建哪个集合，默认 lawyer。
    返回：kb_update.rebuild_collection 的结果 {"message","deleted_old","inserted","corpus_total","collection"}。
    异常：未传 confirm=true 时抛 400，不做任何破坏性操作。
    使用场景：切分策略大改、向量模型或维度变化（EMBED_DIM）之后，旧集合的结构已经不匹配，
              只能 drop 重建；重建过程会短暂不可检索，建议在无人使用时执行。
    """
    if not req.confirm:
        raise HTTPException(400, "请确认操作：传 confirm=true 才执行全量重建")
    collection_name = get_collection_for_role(role_key)
    existing_docs = fetch_all_text(collection_name)  # 先把现有文本捞出来，重建后再灌回去，避免数据丢失
    result = rebuild_collection(existing_docs, collection_name)
    refresh_bm25(collection_name)  # 集合是新的，BM25 索引必须重建
    return result


# ==================== 启动入口 ====================
if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)  # 直接 python main.py 启动；host/port 来自 config，默认 0.0.0.0:8000
