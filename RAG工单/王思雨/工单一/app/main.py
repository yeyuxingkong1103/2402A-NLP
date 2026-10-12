# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本模块职责（HTTP 层）：路由、参数校验、SSE 流式输出、上传与建索引任务、反馈、评估、指标。
# 说明：接口一律用同步 def 定义，由 FastAPI 自动放入线程池执行，避免阻塞事件循环（利于高并发）。

import json                           # 组装 JSON 响应
import time                           # 计时与时间戳
import uuid                           # 生成任务号
import shutil                         # 保存上传文件
from contextlib import asynccontextmanager  # 生命周期上下文管理器
from pathlib import Path              # 路径处理

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile  # FastAPI 组件
from fastapi.middleware.cors import CORSMiddleware    # 跨域中间件
from fastapi.responses import FileResponse, StreamingResponse  # 静态页与流式响应
from fastapi.staticfiles import StaticFiles           # 静态资源托管（前端 JS 等）
from pydantic import BaseModel, Field                 # 请求体模型与校验

from . import config as C             # 全局配置
from . import feedback as FB          # 反馈与性能指标（APIRouter）
from . import llm as L               # 大模型调用与缓存（含容错）
from . import rag                     # 解析与语义分块
from . import store                   # 向量库与 BM25 存储层
from . import retrieval as R          # 在线问答侧

_TASKS: dict[str, dict] = {}          # 建索引/评估任务状态表（内存态，重启即清空）


def warmup() -> None:                         # 预热：把冷启动开销从首次请求挪到启动阶段
    """提前加载模型、跑一次真实前向、连接 Milvus、构建 BM25，避免首个用户请求变慢。

    注意：只加载模型是不够的——首次前向还要做 CUDA 上下文与卷积算法初始化，
    实测只加载不推理时，重启后的第一个请求仍要 4.17s。故这里做一次真实的重排调用。
    """
    try:                                      # 预热失败不应阻止服务启动
        rag.embed_texts(["预热"])             # 触发 bge-m3 加载并完成一次真实编码
        rag.tokenize("预热")                  # 触发 jieba 词典加载（首次约 0.5s，否则算进首个请求）
        dummy = {"content": "预热片段", "doc_id": "warmup", "chunk_index": 0,
                 "page_no": 1, "page_end": 1, "chunk_type": "text", "score": 0.0}  # 构造占位候选
        R.rerank("预热", [dummy], top_k=1)    # 触发 bge-reranker 加载并完成一次真实前向
        store.get_milvus()                    # 建立 Milvus 连接
        store.get_bm25()                      # 构建 BM25 索引
        print("[预热] 模型与服务依赖已就绪")
    except Exception as exc:                  # 任一项失败仅告警
        print(f"[预热] 部分组件预热失败（不影响启动，将在首次请求时重试）：{type(exc).__name__}: {exc}")


@asynccontextmanager                           # 用 lifespan 管理启动/关闭，替代已弃用的 on_event
async def lifespan(_app: FastAPI):             # 应用生命周期
    warmup()                                   # 启动时预热
    yield                                      # 交出控制权，服务运行中
    print("[关闭] 服务已停止")                   # 关闭日志


app = FastAPI(title="基于PDF文档的问答系统", version="1.0", lifespan=lifespan)   # 应用实例
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])                       # 本地演示放开跨域，便于前端 CDN 直连

app.include_router(FB.router)         # 挂载反馈与指标路由（含 /api/metrics）

if C.FRONTEND_DIR.exists():           # 前端目录存在才挂载，避免路径异常导致启动失败
    app.mount("/static", StaticFiles(directory=str(C.FRONTEND_DIR)), name="static")   # 托管前端静态资源


# ============================ 基础接口 ============================
@app.get("/")                                     # 根路径
def index():                                      # 返回前端单页
    """直接托管 frontend/index.html，无需 npm 构建。"""
    f = C.FRONTEND_DIR / "index.html"             # 前端文件路径
    if not f.exists():                            # 文件缺失
        return {"message": "前端文件缺失，请检查 frontend/index.html"}   # 给出提示而非 500
    return FileResponse(str(f))                   # 返回页面


@app.get("/api/health")                           # 健康检查
def health():                                     # 汇总各依赖状态
    """逐项探测 Milvus / Redis / 模型 / 索引，供高可用监控与演示使用。"""
    status = {"llm": L.llm_available(), "redis": False, "milvus": False,  # 初始状态
              "chunks": 0, "env": C.check_env()}    # 附带配置自检
    try:                                          # 探测 Milvus
        cli = store.get_milvus()                  # 取客户端（内部已含降级）
        status["milvus"] = cli.has_collection(C.MILVUS_COLLECTION)   # 集合是否已建
        if status["milvus"]:                      # 已建集合则统计条数
            status["chunks"] = len(store.list_documents())            # 文档数
    except Exception as exc:                      # 探测失败
        status["milvus_error"] = f"{type(exc).__name__}: {exc}"      # 记录原因
    status["redis"] = L.get_redis() is not None   # 探测 Redis（内部已容错）
    ok = status["milvus"] and status["chunks"] > 0                   # 可用的最低要求
    return {"status": "ok" if ok else "degraded", **status}          # 返回整体与明细


# ============================ 知识库管理 ============================
def _build_index(pdf_path: Path, task_id: str, use_mineru: bool) -> None:   # 后台建索引任务
    """解析 → 分块 → 向量化入库 → 写 BM25 语料，全程更新任务进度。"""
    t0 = time.time()                                                     # 计时
    try:                                                                 # 整体容错，任何异常都写入任务状态
        _TASKS[task_id].update(status="parsing", message="正在解析 PDF…")   # 更新进度
        blocks, parser = rag.parse_pdf(pdf_path, use_mineru)             # 解析（含降级）
        _TASKS[task_id].update(parser=parser, blocks=len(blocks),        # 记录解析器与块数
                               message=f"解析完成（{parser}），共 {len(blocks)} 个块，正在语义分块…")
        chunks = rag.semantic_chunk(blocks)                              # 语义分块
        _TASKS[task_id].update(status="indexing", chunks=len(chunks),    # 更新进度
                               message=f"分块完成，共 {len(chunks)} 块，正在向量化入库…")
        doc_id = rag.make_doc_id(pdf_path)                               # 内容指纹作为文档标识
        n = store.index_chunks(chunks, doc_id, pdf_path.name)            # 向量化 + 入库 + BM25 落盘
        _TASKS[task_id].update(status="done", doc_id=doc_id, indexed=n,  # 完成
                               parser=parser, chunks=len(chunks),
                               seconds=round(time.time() - t0, 1),
                               message=f"入库完成：{n} 条，解析器 {parser}")
    except Exception as exc:                                             # 解析/入库失败
        _TASKS[task_id].update(status="failed",                          # 标记失败
                               message=f"{type(exc).__name__}: {exc}")   # 保留原因供前端展示


@app.post("/api/kb/upload")                       # 上传 PDF 并建索引
def upload(background: BackgroundTasks, file: UploadFile = File(...),
           use_mineru: bool = Form(True)):        # 参数：文件 + 是否用 MinerU
    """保存上传的 PDF 并异步建立索引，立即返回任务号供轮询进度。"""
    name = Path(file.filename or "upload.pdf").name                    # 去掉路径，防目录穿越
    if not name.lower().endswith(".pdf"):                              # 类型校验
        raise HTTPException(400, "仅支持 PDF 文件")                     # 明确报错
    dst = C.UPLOAD_DIR / name                                          # 落盘路径
    with open(dst, "wb") as f:                                         # 二进制写入
        shutil.copyfileobj(file.file, f)                               # 流式拷贝，避免大文件吃内存
    if dst.stat().st_size == 0:                                        # 空文件校验
        raise HTTPException(400, "上传文件为空")                        # 明确报错
    task_id = uuid.uuid4().hex[:12]                                    # 生成任务号
    _TASKS[task_id] = {"status": "queued", "message": "排队中", "file": name}   # 初始化任务状态
    background.add_task(_build_index, dst, task_id, use_mineru)        # 交给后台线程执行
    return {"task_id": task_id, "file": name}                          # 立即返回，不阻塞


@app.get("/api/kb/task/{task_id}")                # 查询建索引进度
def task_status(task_id: str):                    # 路径参数：任务号
    """返回任务当前状态；任务不存在时返回 404。"""
    if task_id not in _TASKS:                     # 未知任务
        raise HTTPException(404, "任务不存在")     # 明确报错
    return _TASKS[task_id]                        # 返回状态字典


@app.get("/api/kb/documents")                     # 已入库文档列表
def documents():                                  # 无参数
    """列出知识库中的文档及其分块统计。"""
    try:                                          # Milvus 异常时返回空列表并附带原因
        return {"items": store.list_documents()}   # 正常返回
    except Exception as exc:                      # 连接失败等
        return {"items": [], "error": f"{type(exc).__name__}: {exc}"}   # 明确告知而非 500


@app.get("/api/kb/documents/{doc_id}/chunks")     # 查看解析后的分块内容
def chunks(doc_id: str, page: int = 1, size: int = 20, kind: str = ""):   # 分页与类型过滤
    """知识库管理：查看指定文档的分块内容，验证文字与表格解析结果。"""
    size = max(1, min(size, 100))                 # 限制每页条数，防止一次拉太多
    try:                                          # 容错
        return store.get_chunks(doc_id, page, size, kind)   # 返回分页结果
    except Exception as exc:                      # 查询失败
        return {"total": 0, "items": [], "error": f"{type(exc).__name__}: {exc}"}   # 明确告知


@app.delete("/api/kb/documents/{doc_id}")         # 删除文档
def delete_doc(doc_id: str):                      # 路径参数：文档标识
    """删除该文档的全部向量与 BM25 语料。"""
    try:                                          # 容错
        return {"deleted": store.delete_document(doc_id)}   # 返回删除条数
    except Exception as exc:                      # 删除失败
        raise HTTPException(500, f"删除失败：{type(exc).__name__}: {exc}")   # 明确报错


class ChatIn(BaseModel):                          # 问答请求体
    """问答接口入参，全部带默认值，前端可只传 question。"""
    question: str = Field(..., min_length=1, max_length=512)   # 问题：长度限制即为输入容错
    top_k: int | None = None                      # 每路召回条数，None 用配置默认
    use_rewrite: str = "auto"                     # auto/true/false：是否调 LLM 做 Query 改写
    use_rerank: bool = True                       # 是否启用重排
    use_cache: bool = True                        # 是否允许读缓存


@app.post("/api/chat")                            # 问答（非流式）
def chat(body: ChatIn):                           # 请求体
    """完整 RAG 问答，返回答案、引用片段与各阶段耗时，便于评测计时。"""
    q = body.question.strip()                     # 去空白
    if not q:                                     # 空问题
        raise HTTPException(400, "问题不能为空")    # 明确报错
    try:                                          # 主链路容错
        res = R.answer(q, body.top_k, body.use_rewrite, body.use_rerank, body.use_cache)   # 执行
    except Exception as exc:                      # 未预期异常
        raise HTTPException(500, f"问答失败：{type(exc).__name__}: {exc}")   # 返回明确错误
    # 非流式接口没有"首字"概念：答案一次性返回，首字时刻等于完整答案时刻，故两个参数同值。
    FB.record(res["latency"]["total"], res["latency"]["total"])   # 记录耗时（首字口径见 feedback.py）
    return res                                    # 返回结果


@app.post("/api/chat/stream")                     # 问答（SSE 流式）
def chat_stream(body: ChatIn):                    # 请求体同上
    """先推 meta（检索结果与耗时），再逐块推 delta，最后推 done；首字延迟即用户感知响应时间。"""
    q = body.question.strip()                     # 去空白
    if not q:                                     # 空问题
        raise HTTPException(400, "问题不能为空")    # 明确报错
    def gen():                                    # 同步生成器，FastAPI 会放入线程池执行
        t0 = time.time()                          # 总计时起点
        try:                                      # 检索阶段容错
            ret = R.retrieve(q, body.top_k, body.use_rewrite, body.use_rerank)   # 检索
        except Exception as exc:                  # 检索失败
            yield f"data: {json.dumps({'type': 'error', 'message': f'检索失败：{exc}'}, ensure_ascii=False)}\n\n"
            return                                # 结束
        meta = {"type": "meta", "contexts": [R.slim(c) for c in ret["contexts"]],    # 引用片段
                "query_understanding": ret["query_understanding"],                   # Query 理解
                "stats": {k: ret[k] for k in ("n_dense", "n_sparse", "n_fused")},    # 召回统计
                "retrieve_seconds": ret["timing"]["retrieve"]}                       # 检索耗时
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"   # 先发元信息，前端可立刻渲染引用
        buf, first = [], None                     # 累积答案与首字时刻
        try:                                      # 生成阶段容错
            for piece in L.llm_chat(R.build_prompt(q, ret["contexts"]), max_tokens=450,
                                        temperature=0.0, stream=True):   # 与线上一致的确定性生成
                if first is None:                 # 首个增量
                    first = time.time() - t0      # 记录首字延迟
                buf.append(piece)                 # 累积完整答案
                yield f"data: {json.dumps({'type': 'delta', 'text': piece}, ensure_ascii=False)}\n\n"
        except Exception as exc:                  # LLM 不可用
            text = R._fallback_answer(q, ret["contexts"], exc)   # 兜底摘录
            buf.append(text)                      # 计入答案
            first = first or (time.time() - t0)   # 记录首字时刻
            yield f"data: {json.dumps({'type': 'delta', 'text': text, 'degraded': True}, ensure_ascii=False)}\n\n"
        total = time.time() - t0                  # 完整答案耗时
        # 验收指标按首字口径记录（first 可能为 None，兜底取 total）；完整答案时长一并记录但只作参考。
        FB.record(first or total, total)          # 记录到指标
        done = {"type": "done", "latency": {**ret["timing"],          # 各阶段耗时
                "first_token": round(first or total, 3),              # 首字延迟
                "total": round(total, 3)},                            # 端到端
                "answer": "".join(buf)}                               # 完整答案文本
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"     # 收尾事件
        if L.get_redis():                                             # 有缓存才写
            L.cache_set(L.cache_key("stream", q), done["answer"], ttl=1800)    # 缓存答案文本
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})   # SSE


@app.post("/api/chat/llm_only")                   # 纯 LLM 对照组
def chat_llm_only(body: ChatIn):                  # 请求体复用
    """不传检索上下文直接问 LLM，用于工单要求的 RAG vs 纯 LLM 对比。"""
    return R.answer_llm_only(body.question.strip())   # 直接返回对照结果


# ============================ 评估与指标 ============================
@app.post("/api/eval/run")                        # 触发评估
def eval_run(background: BackgroundTasks, use_rerank: bool = True):   # 参数
    """后台运行工单 10 问的 RAG / 纯 LLM 对比与 RAGAS 评估。"""
    from . import eval as E                       # 延迟导入，避免拖慢启动
    task_id = uuid.uuid4().hex[:12]               # 任务号
    _TASKS[task_id] = {"status": "queued", "message": "评估排队中"}   # 初始化状态
    background.add_task(E.run_eval, task_id, _TASKS, use_rerank)      # 后台执行
    return {"task_id": task_id}                   # 返回任务号


@app.get("/api/eval/result")                      # 取最近一次评估结果
def eval_result():                                # 无参数
    """读取 eval/results/latest.json，不存在则返回空。"""
    p = C.EVAL_DIR / "results" / "latest.json"    # 结果文件
    if not p.exists():                            # 尚未评估
        return {"available": False, "message": "尚未运行评估"}   # 明确提示
    return {"available": True, **json.loads(p.read_text(encoding="utf-8"))}   # 返回结果
