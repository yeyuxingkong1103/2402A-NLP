# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：FastAPI 服务入口。对外提供 /chat、/feedback、/health 与文档管理接口；
#           启动时可预热模型与索引，把首请求的加载耗时前移，避免它吃掉 3 秒响应预算。
import logging                                                                  # 标准日志，替代复杂监控系统
import shutil                                                                   # 保存上传文件
import time                                                                     # 计时
from contextlib import asynccontextmanager                                      # 生命周期管理
from pathlib import Path                                                        # 路径处理

from fastapi import FastAPI, File, HTTPException, UploadFile                    # Web 框架
from fastapi.middleware.cors import CORSMiddleware                              # 跨域，供 Vue 前端直连
from fastapi.responses import JSONResponse                                      # 自定义响应

from . import config as C                                                       # 复用集中配置
from . import deepseek_llm, milvus_store, pipeline, redis_cache, reranker       # 各能力模块
from .schemas import (ChatRequest, ChatResponse, FeedbackRequest,               # 请求/响应模型
                      FeedbackResponse, HealthResponse)                         # 同上

logging.basicConfig(level=getattr(logging, C.LOG_LEVEL.upper(), logging.INFO),  # 日志级别来自配置
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")    # 统一日志格式
log = logging.getLogger("gd5")                                                  # 本服务日志器


@asynccontextmanager                                                             # 声明生命周期钩子
async def lifespan(app: FastAPI):                                               # 启动/关闭
    """启动时预热索引与模型；预热失败只告警不阻断，服务仍以降级模式提供问答。"""
    if C.WARMUP_ON_START:                                                       # 允许用配置关掉预热（快速调试）
        try:                                                                    # 预热失败不应让服务起不来
            t0 = time.time()                                                    # 计时
            pipeline.load()                                                     # 装载 BM25 与 Milvus
            pipeline.warmup()                                                   # 真正跑一次编码与重排，把首次推理开销前移
            log.info("预热完成，耗时 %.1fs，知识库块数=%d",                          # 打印结果
                     time.time() - t0, pipeline.stats()["chunks"])               # 块数
        except Exception as exc:                                                # 预热异常
            log.warning("预热失败（服务继续以降级模式运行）：%s: %s",                # 只告警
                        type(exc).__name__, exc)                                 # 打印原因
    yield                                                                       # 移交控制权给应用
    log.info("服务关闭")                                                          # 关闭日志


app = FastAPI(title="招股说明书多轮问答（工单五：Query 理解优化）",                     # 应用标题
              version="1.0.0", lifespan=lifespan)                                # 版本与生命周期
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],    # 演示用：允许任意来源
                   allow_headers=["*"])                                         # 允许任意头


@app.get("/health", response_model=HealthResponse)                              # 健康检查
def health() -> HealthResponse:                                                 # 逐个组件报状态
    """返回各依赖的可用性；任一关键组件不可用时 overall 置为 degraded，前端据此提示。"""
    st = pipeline.stats()                                                       # 知识库统计
    redis_ok = redis_cache.available()                                          # Redis
    milvus_ok = False                                                           # Milvus 默认不可用
    try:                                                                        # 连接可能失败
        cli = milvus_store.milvus_client()                                      # 建客户端
        milvus_ok = cli.has_collection(C.MILVUS_COLLECTION)                     # 集合存在即视为可用
    except Exception:                                                           # 连接失败
        milvus_ok = False                                                       # 置为不可用
    embed_ok = milvus_store._EMBED is not None                                   # 向量模型是否已加载
    rerank_ok = reranker.is_ready()                                             # 重排模型是否已加载
    llm_ok = deepseek_llm.llm_ready()                                           # 大模型密钥是否配置
    core_ok = bool(st["chunks"]) and llm_ok                                    # 核心可用条件：有知识库 + 有大模型
    detail = []                                                                 # 明细
    if not redis_ok: detail.append("Redis 不可用：会话历史与缓存已降级")             # 提示
    if not milvus_ok: detail.append("Milvus 不可用：仅关键词检索")                  # 提示
    if not embed_ok: detail.append("向量模型未加载（首次请求时加载）")                 # 提示
    if not rerank_ok: detail.append("重排模型未加载（重排降级为原始排序）")              # 提示
    if not llm_ok: detail.append("未配置 DEEPSEEK_API_KEY：无法生成回答")           # 提示
    if not st["chunks"]: detail.append("知识库为空：请先运行 scripts/ingest.py")     # 提示
    return HealthResponse(status="ok" if core_ok else "degraded",               # 总体状态
                          milvus=milvus_ok, redis=redis_ok, llm=llm_ok,         # 各组件
                          embed_model=embed_ok, rerank_model=rerank_ok,         # 模型
                          chunks=st["chunks"], docs=st["docs"],                 # 知识库
                          detail="；".join(detail))                             # 明细


@app.post("/chat", response_model=ChatResponse)                                 # 多轮问答
def chat(req: ChatRequest) -> ChatResponse:                                     # 入参：会话号 + 问题
    """一轮问答：改写 → 混合检索 → 重排 → 生成 → 引用；任一环节异常都转成明确的 HTTP 错误。"""
    question = (req.question or "").strip()                                     # 去空白
    if not question:                                                            # 空问题
        raise HTTPException(status_code=400, detail="问题不能为空")                # 参数错误
    try:                                                                        # 主流程可能抛异常
        t0 = time.time()                                                        # 计时
        result = pipeline.answer_question(req.session_id or "default", question)  # 执行
        log.info("chat sid=%s 耗时=%dms cached=%s degraded=%s",                  # 结构化日志
                 req.session_id, int((time.time() - t0) * 1000),                 # 耗时
                 result.get("cached"), result.get("degraded"))                   # 命中与降级
        return ChatResponse(**result)                                           # 组装响应
    except Exception as exc:                                                    # 任何未捕获异常
        log.exception("chat 处理失败")                                            # 打印堆栈
        raise HTTPException(status_code=500,                                # 明确告知失败原因
                            detail=f"问答处理失败：{type(exc).__name__}: {exc}")  # 细节


@app.post("/feedback", response_model=FeedbackResponse)                         # 用户反馈
def feedback(req: FeedbackRequest) -> FeedbackResponse:                         # 入参：评分与文字意见
    """记录点赞/点踩与文字反馈；写入失败如实返回失败，不假装成功。"""
    ok = redis_cache.save_feedback(req.model_dump())                            # 落库
    return FeedbackResponse(ok=ok,                                              # 结果
                            message="反馈已记录，感谢！" if ok else "反馈写入失败，请稍后重试")  # 提示文案


@app.get("/documents")                                                          # 预置文档列表
def documents() -> dict:                                                        # 返回知识库现状
    """列出 data/raw 下预置/上传的 PDF，供前端展示"当前知识库"。"""
    return {"docs": [p.name for p in C.raw_pdfs()],                             # PDF 文件名列表
            "indexed_chunks": pipeline.stats()["chunks"]}                       # 已入库块数


@app.post("/upload")                                                            # 上传 PDF（工单要求支持）
def upload(file: UploadFile = File(...)) -> dict:                               # 接收文件
    """保存 PDF 到 data/raw；随后需运行 scripts/ingest.py 重建索引（本接口只负责收文件）。"""
    if not (file.filename or "").lower().endswith(".pdf"):                      # 只收 PDF
        raise HTTPException(status_code=400, detail="只支持上传 PDF 文件")         # 参数错误
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)                                # 确保目录存在
    dst = C.RAW_DIR / Path(file.filename).name                                  # 目标路径（只取文件名，防目录穿越）
    try:                                                                        # 写盘可能失败
        with open(dst, "wb") as f:                                              # 二进制写入
            shutil.copyfileobj(file.file, f)                                    # 流式拷贝，避免整文件进内存
    except Exception as exc:                                                    # 写盘失败
        raise HTTPException(status_code=500, detail=f"保存失败：{exc}")           # 明确报错
    return {"ok": True, "saved": dst.name,                                      # 成功响应
            "next": "请运行 python scripts/ingest.py 重建索引后生效"}             # 明确后续步骤


@app.exception_handler(Exception)                                               # 兜底异常处理
async def on_error(request, exc: Exception) -> JSONResponse:                     # 统一错误响应
    """未预期的异常统一返回 500 与可读原因，避免前端拿到裸堆栈。"""
    log.exception("未处理异常")                                                   # 打印堆栈
    return JSONResponse(status_code=500,                                        # 状态码
                        content={"detail": f"服务内部错误：{type(exc).__name__}: {exc}"})  # 内容


def run() -> None:                                                              # 命令行启动
    """python -m backend.main 即可启动开发服务器。"""
    import uvicorn                                                              # 延迟导入
    uvicorn.run("backend.main:app", host=C.API_HOST, port=C.API_PORT,           # 监听地址与端口取自配置
                reload=False, log_level=C.LOG_LEVEL.lower())                     # 关闭热重载，避免模型被重复加载


if __name__ == "__main__":                                                      # 直接执行时
    run()                                                                       # 启动服务
