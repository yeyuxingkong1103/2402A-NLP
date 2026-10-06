# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
RAG-PDF 问答系统 - FastAPI 主应用

完整链路：
    文档上传 → PDF 解析 → 切分 → 向量化 → 写入 Milvus
    用户提问 → 混合检索 → 重排 → 上下文拼装 → LLM 生成 → 返回

接口一览：
    POST /api/upload   上传 PDF 文件到 data 目录
    POST /api/ingest   解析 + 切分 + 入库 + 重建 BM25
    POST /api/chat     问答接口（检索 + LLM 生成），支持流式和非流式
    GET  /api/search   纯检索接口（不调 LLM）
    GET  /api/health   健康检查

本模块在系统中的位置：
    上游：前端页面（挂在根路径下的「web」目录，纯原生 html/css/js）通过
          HTTP 调这些接口；文件末尾的 __main__ 用 uvicorn 把 app 起在
          0.0.0.0:8000。
    下游：本模块只做「编排」，真正的算法都在各层模块里——
          db_milvus + retriever（向量召回 + BM25 融合 + 重排）、
          llm_client（DeepSeek 调用）、pdf_parser / text_splitter
          （入库解析与切分）。

关键设计取舍：
    1. 模型只加载一次：bge-m3 与 bge-reranker 体积大、加载慢，统一放在
       lifespan 启动阶段加载，之后靠各模块的模块级单例复用，绝不在请求
       处理里重新加载；
    2. 接口都写成同步 def：FastAPI 会把同步函数丢到线程池执行，不会阻塞
       事件循环；而 Milvus 客户端本身是同步库，写同步代码最直接；
    3. /api/search 与 /api/chat 分开：前者只检索不调 LLM，供前端「检索台」
       看引用原文；后者才走完整问答链路（检索 → 生成）。
"""

import os  # 路径操作
import json  # JSON 序列化
from contextlib import asynccontextmanager  # 生命周期钩子
from typing import List, Optional  # 类型标注

import uvicorn  # ASGI 服务器
from fastapi import FastAPI, UploadFile, File, HTTPException  # Web 框架
from fastapi.middleware.cors import CORSMiddleware  # 跨域支持（前端访问需要）
from fastapi.responses import StreamingResponse, HTMLResponse  # 流式响应 + 前端页面托管
from fastapi.staticfiles import StaticFiles  # 静态文件服务
from pydantic import BaseModel, Field  # 请求体模型

from config import API_HOST, API_PORT, TOP_K_RERANK  # 配置
from logger import get_logger  # 日志
from db_milvus import (  # Milvus 数据层
    get_embedding, get_vectorstore, get_milvus_client,
    ingest_documents, get_collection_count,
)
from pdf_parser import parse_pdf_to_document  # PDF 解析
from text_splitter import split_documents  # 文本切分
from retriever import hybrid_search, refresh_bm25, get_reranker  # 检索
from llm_client import generate_answer, stream_answer  # LLM 调用

logger = get_logger(__name__)  # 本模块 logger

# data 目录：存放上传的 PDF 文件
# 用 __file__ 拼绝对路径，任意工作目录启动都能找到 data 目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ==================== 生命周期：启动时加载模型 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动时一次性加载全部模型；关闭时什么都不用做

    这是 FastAPI 的 lifespan 生命周期钩子：yield 之前的代码在服务启动时
    执行，yield 之后的代码在进程收到关闭信号时执行。

    启动顺序不可随意调换，每一步都被后面的步骤依赖：
        向量模型 → Milvus 连接/建集合 → 重排模型 → BM25 索引。
        比如先把向量模型加载好，后面初始化 Milvus 集合时才能把 embedding
        函数绑上去；BM25 依赖「库里现在有哪些文本」，必须在集合就绪之后
        重建。

    参数：
        app：FastAPI 应用对象，由框架自动传入，本函数内不使用。

    异常与降级：
        任何一步失败（模型路径写错、Milvus 连不上）都会让服务启动直接
        失败——这是有意为之：带病启动只会让之后每个请求都报错，不如
        启动时就暴露问题。
    """
    logger.info("[启动] 开始初始化 RAG-PDF 问答系统 ...")

    # 0. 确保 data 目录存在（上传接口要用）
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. 向量模型（本地 bge-m3 权重，加载即用）
    get_embedding()  # 首次调用才真正读盘加载权重（约十几秒），之后全局复用同一实例
    logger.info("[启动] 本地 bge-m3 向量模型就绪")

    # 2. Milvus：初始化集合（不存在则由 LangChain 自动创建）
    get_milvus_client()  # 建立到 Milvus 的连接（模块级单例）
    get_vectorstore()  # 检查/创建集合并绑定向量模型
    logger.info("[启动] Milvus 集合连接就绪")

    # 3. 重排模型（首次加载约十几秒）
    try:
        get_reranker()  # CPU 上跑 HuggingFaceCrossEncoder，同样只加载一次
        logger.info("[启动] bge-reranker 重排模型就绪")
    except Exception as e:
        # 重排模型加载失败不阻断启动：用户可以暂时只用向量召回（检索质量会下降，但不至于完全不可用）
        logger.warning(f"[启动] 重排模型加载失败（{e}），后续检索将只用向量召回")

    # 4. BM25 索引重建（如果集合已有数据）
    try:
        refresh_bm25()  # 集合为空时内部会打 warning，不影响启动
        logger.info("[启动] BM25 索引就绪")
    except Exception as e:
        logger.warning(f"[启动] BM25 重建失败（{e}），首次入库后请重启或调用 /api/ingest")

    logger.info(f"[启动] 完成！打开 http://127.0.0.1:{API_PORT}/docs 测试")
    yield  # 服务运行期间停在这里：此后的代码只在进程关闭时才会执行
    logger.info("[关闭] 服务退出")


# ==================== FastAPI 应用 ====================
# lifespan 把上面写的启动/关闭逻辑注册进来；title/description/version 会显示在 /docs 页面上
app = FastAPI(
    title="RAG-PDF 问答系统",
    description="基于 PDF 文档的 RAG 问答：向量检索 + BM25 + RRF 重排 + DeepSeek LLM",
    version="1.0",
    lifespan=lifespan,
)

# CORS：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 教学项目图省事放开所有来源；生产环境应改成明确的白名单
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端静态文件托管：把 "web" 文件夹挂在根路径下
_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")  # 前端文件目录
if os.path.isdir(_frontend_dir):  # 前端目录存在才挂载，避免启动即崩
    app.mount("/web", StaticFiles(directory=_frontend_dir), name="frontend")  # /web/* 访问前端静态资源


# ==================== 请求体模型 ====================
class IngestRequest(BaseModel):
    """入库请求：指定 data 目录下要处理的文件名（不传则处理全部）"""
    files: Optional[List[str]] = Field(default=None, description="要处理的文件名列表（相对 data 目录）；不传则处理 data 下所有 PDF")


class ChatRequest(BaseModel):
    """问答请求"""
    query: str = Field(..., description="用户提问")  # 必填；用户本轮的问题，同时作为检索 query
    top_k: int = Field(default=TOP_K_RERANK, ge=1, le=20, description="引用资料条数")  # 重排后保留条数，1~20
    stream: bool = Field(default=False, description="是否流式返回（SSE）")  # 默认非流式


class SearchRequest(BaseModel):
    """纯检索请求（不调 LLM）"""
    query: str = Field(..., description="查询关键词")  # 必填
    top_k: int = Field(default=TOP_K_RERANK, ge=1, le=20, description="返回条数")  # 1~20


# ==================== 接口 ====================

@app.get("/", tags=["系统"])
def root():
    """根路径：返回简单引导页（如果 web 目录有 index.html 会重定向过去）"""
    index_html = os.path.join(_frontend_dir, "index.html") if os.path.isdir(_frontend_dir) else ""
    if index_html and os.path.isfile(index_html):
        return HTMLResponse('<meta http-equiv="refresh" content="0;url=/web/index.html">')
    return {"message": "RAG-PDF 问答系统", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health", tags=["系统"])
def health():
    """健康检查：返回集合文档数与各模块状态

    返回字段：
        status：整体状态（"ok" / "degraded"）
        collection_count：集合内语料总数
        data_dir：data 目录绝对路径
    """
    count = get_collection_count()
    return {
        "status": "ok",
        "collection_count": count,
        "data_dir": DATA_DIR,
    }


@app.post("/api/upload", tags=["入库"])
async def upload_pdf(file: UploadFile = File(..., description="PDF 文件")):
    """
    上传 PDF 文件到 data 目录

    参数：
        file：上传的 PDF 文件（multipart/form-data）。
    返回：
        上传成功后的文件名与绝对路径。
    异常：
        文件不是 .pdf 后缀时返回 400；写入失败时返回 500。
    说明：
        只做文件落盘，不做解析和入库——解析和入库走 /api/ingest，
        这样上传和入库可以解耦（先批量上传，再统一入库）。
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 .pdf 文件")

    os.makedirs(DATA_DIR, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(DATA_DIR, os.path.basename(file.filename))  # basename 防止路径穿越攻击
    try:
        content = await file.read()  # 读字节
        with open(save_path, "wb") as f:  # 二进制写
            f.write(content)
    except Exception as e:
        logger.error(f"上传文件失败：{file.filename}（{e}）")
        raise HTTPException(status_code=500, detail=f"保存文件失败：{e}")

    logger.info(f"上传成功：{file.filename} -> {save_path}（{len(content)} bytes）")
    return {"filename": file.filename, "saved_path": save_path, "size": len(content)}


@app.post("/api/ingest", tags=["入库"])
def ingest(req: IngestRequest):
    """
    解析 + 切分 + 入库 + 重建 BM25

    参数（请求体）：
        files：可选，指定要处理的文件名列表（相对 data 目录）；不传则
               处理 data 目录下所有 .pdf。
    返回：
        processed：本次处理的文件数
        ingested：成功入库的文件数
        collection_count：入库后集合总数
    异常与降级：
        单个文件失败不中断整批：失败文件只记 error，继续处理下一个；
        入库完成后无条件重建 BM25（即使没新文档，重建一次也能保证索引与
        集合一致）。
    """
    # 1. 确定文件列表
    if req.files:  # 指定了文件名
        pdf_files = [os.path.join(DATA_DIR, os.path.basename(n)) for n in req.files]
    else:  # 扫全部
        if not os.path.isdir(DATA_DIR):
            raise HTTPException(status_code=400, detail=f"data 目录不存在：{DATA_DIR}")
        pdf_files = [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.lower().endswith(".pdf")
        ]

    if not pdf_files:
        raise HTTPException(status_code=400, detail="data 目录下没有可处理的 PDF 文件")

    logger.info(f"开始入库：共 {len(pdf_files)} 个 PDF")
    ingested = 0  # 成功计数
    for i, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"[{i}/{len(pdf_files)}] 处理：{os.path.basename(pdf_path)}")
        if not os.path.isfile(pdf_path):
            logger.error(f"文件不存在：{pdf_path}")
            continue
        try:
            # 解析 → 切分 → 入库
            docs = parse_pdf_to_document(pdf_path)
            if not docs:
                logger.warning(f"{os.path.basename(pdf_path)} 解析为空，跳过")
                continue
            chunks = split_documents(docs)
            if not chunks:
                logger.warning(f"{os.path.basename(pdf_path)} 切分为空，跳过")
                continue
            ingest_documents(chunks)
            ingested += 1
        except Exception as e:
            logger.error(f"处理 {os.path.basename(pdf_path)} 失败：{e}")

    # 2. 重建 BM25（一次性，避免重复全量重建）
    try:
        refresh_bm25()
    except Exception as e:
        logger.error(f"重建 BM25 失败：{e}")

    count = get_collection_count()
    logger.info(f"入库完成：成功 {ingested}/{len(pdf_files)}，集合总数 {count}")
    return {
        "processed": len(pdf_files),
        "ingested": ingested,
        "collection_count": count,
    }


@app.get("/api/search", tags=["检索"])
def search(query: str, top_k: int = TOP_K_RERANK):
    """
    纯检索接口：向量 + BM25 + RRF + 重排，不调 LLM

    参数（query string）：
        query：查询关键词。
        top_k：返回条数，默认 TOP_K_RERANK（1~20，超过范围会被 FastAPI 校验拒绝）。
    返回：
        results：列表，每项含 page_content（原文）、score（重排分数）、
                 source（来源文件）、page_number（页码）。
        total：命中条数。
    异常：
        知识库未初始化时返回 400，提示先入库。
    说明：
        与 /api/chat 区别：本接口只做检索，不调 LLM 也不拼 prompt，
        用于前端「检索台」查看引用原文，或调试检索效果。
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k 必须在 1~20 之间")

    try:
        docs = hybrid_search(query, top_k=top_k)
    except RuntimeError as e:
        # 知识库没准备好（集合不存在 / BM25 未初始化）→ 400 提示用户先入库
        raise HTTPException(status_code=400, detail=str(e))

    results = []
    for doc in docs:
        results.append({
            "page_content": doc.page_content,
            "score": doc.metadata.get("score", 0.0),
            "source": doc.metadata.get("source", ""),
            "page_number": doc.metadata.get("page_number", 0),
        })
    logger.info(f"检索：query='{query[:30]}'，命中 {len(results)} 条")
    return {"query": query, "total": len(results), "results": results}


@app.post("/api/chat", tags=["问答"])
def chat(req: ChatRequest):
    """
    问答接口：检索 + LLM 生成

    请求体：
        query：用户提问。
        top_k：引用资料条数（1~20）。
        stream：是否流式返回（True 时返回 SSE 流）。
    返回：
        - 非流式：JSON，含 answer（回答文本）和 references（引用资料列表）。
        - 流式：SSE 流，每个 chunk 是一段文本；流末尾会附一个 [DONE] 标记。
    异常：
        知识库未初始化时返回 400；LLM 调用失败时返回 500。
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    # 1. 检索（失败转 400，提示先入库）
    try:
        docs = hybrid_search(req.query, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"问答：query='{req.query[:30]}'，引用 {len(docs)} 条资料，stream={req.stream}")

    # 2. 流式：返回 SSE
    if req.stream:
        def sse_generator():
            """SSE 生成器：把每个 chunk 包成 data: 事件"""
            try:
                for chunk in stream_answer(req.query, docs):
                    # SSE 格式：data: <json>\n\n；用 JSON 包一层便于前端统一解析
                    yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
                # 流结束发一个 DONE 标记
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"流式生成失败：{e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",  # SSE 必须是这个 MIME
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 关闭 nginx 缓冲，保证实时推送
        )

    # 3. 非流式：阻塞调用 LLM
    try:
        answer = generate_answer(req.query, docs)
    except Exception as e:
        logger.error(f"LLM 生成失败：{e}")
        raise HTTPException(status_code=500, detail=f"LLM 生成失败：{e}")

    # 附引用资料，便于前端展示
    references = [
        {
            "page_content": doc.page_content,
            "score": doc.metadata.get("score", 0.0),
            "source": doc.metadata.get("source", ""),
            "page_number": doc.metadata.get("page_number", 0),
        }
        for doc in docs
    ]
    return {"query": req.query, "answer": answer, "references": references}


# ==================== 启动入口 ====================
if __name__ == "__main__":
    # 用 uvicorn 启动：host/port 从 config 读
    # reload=True 适合开发：改代码自动重启；生产环境应去掉
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=False)
