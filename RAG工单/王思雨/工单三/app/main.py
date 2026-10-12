# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：FastAPI 服务入口。提供 /upload、/parse、/ask、/evaluate、/health 五个接口，支持中英文问答。
import json                                                                        # 读取落盘索引里的规模统计
import time                                                                        # 统计接口层耗时
from contextlib import asynccontextmanager                                         # 启动预热钩子
from fastapi import FastAPI, File, HTTPException, Request, UploadFile              # Web 框架、文件上传、异常、请求对象、上传文件类型
from fastapi.middleware.cors import CORSMiddleware                                 # 允许 Vue 前端跨域调用
from fastapi.responses import StreamingResponse                                    # SSE 流式返回
from pydantic import BaseModel, Field                                              # 请求体校验
from . import cache as K                                                           # 缓存探测
from . import config as C                                                          # 集中配置
from . import rag_core as P                                                        # 问答流水线

@asynccontextmanager                                                                # 生命周期钩子
async def lifespan(_: FastAPI):                                                     # 启动时预热、关闭时无需额外处理
    """启动即载入索引与两个本地模型，避免第一个用户请求等 20 秒加载模型被当成卡死。"""
    print(f"[启动预热] {P.warmup()}")                                                # 预热失败只打印，服务照常起来
    yield                                                                           # 交由 FastAPI 处理请求


app = FastAPI(title="PDF文档的表格解析及检索优化 RAG 问答系统", version="1.0", lifespan=lifespan)  # 应用实例
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # 开发期放开跨域，便于 Vue 调试


class AskRequest(BaseModel):                                                       # 提问请求体
    """一次提问：问题文本、语言（zh / en / auto）、是否启用缓存、交给大模型的上下文条数。"""
    question: str = Field("", description="用户问题，支持中文或英文")                   # 问题文本
    lang: str = Field("auto", description="zh=中文，en=英文，auto=按问题自动判定")        # 语言
    use_cache: bool = Field(True, description="是否启用 Redis 缓存")                    # 缓存开关
    top_n: int = Field(0, description="交给大模型的上下文条数，0 表示用配置默认值")          # 上下文条数


class ParseRequest(BaseModel):                                                     # 解析请求体
    """触发解析与建索引：可只重建某篇文档，或关闭 MinerU 强制走降级链路。"""
    doc: str = Field("", description="只重建该文档（文件名去扩展名），留空表示全部重建")      # 单文档重建
    use_mineru: bool = Field(True, description="是否优先使用 MinerU 解析")               # 解析器开关


@app.post("/upload")                                                               # 上传 PDF
async def upload(file: UploadFile = File(...)) -> dict:                             # 接收上传文件
    """接收 PDF 存入 data/raw/，供 /parse 解析；非 PDF 或空文件返回 400 并说明原因。"""
    name = (file.filename or "").strip()                                           # 原始文件名
    if not name.lower().endswith(".pdf"): raise HTTPException(400, "只支持上传 .pdf 文件")  # 类型校验
    data = await file.read()                                                       # 读取上传内容
    if len(data) < 1024: raise HTTPException(400, "文件为空或已损坏，请重新选择")          # 体积校验
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)                                   # 目录不存在则创建
    path = C.RAW_DIR / name                                                        # 落盘路径
    path.write_bytes(data)                                                         # 写入磁盘
    return dict(ok=True, filename=name, bytes=len(data), path=str(path),            # 返回结果
                message="上传成功，调用 /parse 即可解析并建立索引")                      # 下一步提示


@app.post("/parse")                                                                # 解析并建索引
def parse(req: ParseRequest) -> dict:                                              # 触发解析
    """解析知识库 PDF（表格结构化 → 语义分块 → 向量入库 → 建 BM25）；失败时逐级降级并回报实际解析器。"""
    try:                                                                           # 解析异常统一转成 500 与中文原因
        stats = P.build_index(use_mineru=req.use_mineru, only=req.doc or None)      # 执行构建
        return dict(ok=True, stats=stats)                                          # 返回逐篇解析报告与块规模
    except Exception as exc:                                                       # 文件缺失、解析全失败等
        raise HTTPException(500, f"解析失败：{type(exc).__name__}: {exc}")            # 明确错误原因，不暴露堆栈


def _sse(event: str, data: dict) -> str:                                            # 组装一条 SSE 事件
    """SSE 帧格式：event 行 + data 行 + 空行；内容一律 JSON，中文不转义。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"        # 返回可直接写入响应流的一帧


def _ask_stream(req: AskRequest):                                                    # 流式问答主体
    """先推检索结果，再逐段推生成的 token。

    这样做的价值：检索约 0.9 秒就完成，前端立刻能看到来源片段与表格定位，
    用户感知的首字延迟从"整个端到端"降到"检索完成"，生成慢也不再像卡死。
    """
    t0 = time.time()                                                                # 计时起点
    try:                                                                            # 全程不抛栈，出错也走 SSE 事件
        prep = P.prepare(req.question, req.lang, req.use_cache, req.top_n or None)     # 前半程：校验 + 缓存 + 检索
        if prep.get("error"):                                                       # 校验失败或检索不可用
            yield _sse("error", {k: prep.get(k) for k in ("answer", "error", "detail")})  # 推出中文原因
            return                                                                  # 结束
        yield _sse("retrieval", {k: prep.get(k) for k in                              # 检索一完成立刻推：前端可马上渲染来源与定位
            ("contexts", "retrieval_ms", "retrieval_mode", "degraded", "degraded_reason", "cached")})
        if prep.get("cached"):                                                      # 缓存命中时答案已完整，直接一次性推完
            yield _sse("token", {"text": prep.get("answer", "")})                     # 推完整答案
            yield _sse("done", dict(prep, elapsed=round(time.time() - t0, 3), streamed=True))  # 收尾
            return                                                                  # 结束
        t_gen, acc, llm_error = time.time(), [], None                                # 生成计时、累积文本、错误类型
        try:                                                                        # 生成失败要退回"只给片段"
            for delta in P.generate_stream(prep["question"], prep["contexts"], prep["lang"]):  # 逐段取 token
                acc.append(delta)                                                    # 累积，供最终答案与缓存
                yield _sse("token", {"text": delta})                                  # 推给前端
        except Exception as exc:                                                     # 断网、限流、Key 失效、超时
            print(f"[生成降级] {type(exc).__name__}: {exc}")                            # 打印原因便于排障
            llm_error = type(exc).__name__                                            # 记录错误类型
        if llm_error:                                                               # 生成失败：把已推内容作废，改推完整片段提示
            yield _sse("reset", {"reason": _sse_reason(llm_error)})                    # 通知前端清屏
            yield _sse("token", {"text": "生成服务暂不可用，以下为检索到的原文片段，请自行查阅。"})  # 友好提示
        result = P._finish(prep, "".join(acc), llm_error, t_gen, t0)                   # 组装与非流式一致的返回体
        if req.use_cache: K.set(prep["_key"], {k: v for k, v in result.items() if k != "elapsed"})  # 写缓存，与非流式共用同一格式
        yield _sse("done", dict(result, streamed=True))                               # 收尾事件带完整信息
    except Exception as exc:                                                        # 兜底：任何意外都翻译成事件
        print(f"[流式异常] {type(exc).__name__}: {exc}")                               # 打印便于排障
        yield _sse("error", {"error": type(exc).__name__, "answer": "系统繁忙或依赖服务暂时不可用，请稍后重试。"})  # 中文提示


def _sse_reason(llm_error: str) -> str:                                              # 生成降级原因
    """与 _finish 里拼接的口径一致，供 reset 事件说明为什么清屏重来。"""
    return f"生成服务不可用（{llm_error}）"                                             # 与 done 事件里的措辞保持统一


@app.post("/ask")                                                                   # 问答接口
def ask(req: AskRequest, request: Request):                                         # 处理提问
    """返回答案、来源片段（含页码与表格定位）、分阶段耗时与是否命中缓存。

    **同一个接口、两种返回方式**：客户端带 `Accept: text/event-stream` 时走 SSE 流式，
    否则返回一次性 JSON。不是新增接口，只是同一接口的内容协商（工单未限定返回格式）。
    """
    t0 = time.time()                                                                # 接口层计时起点
    if "text/event-stream" in (request.headers.get("accept") or ""):                 # 客户端要流式
        return StreamingResponse(_ask_stream(req), media_type="text/event-stream",    # 流式返回
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})  # 关掉缓冲，保证逐帧到达
    out = P.answer(req.question, lang=req.lang, use_cache=req.use_cache, top_n=req.top_n or None)  # 非流式走原有链路
    return dict(out, api_elapsed=round(time.time() - t0, 3))                        # 附上接口层耗时


@app.post("/evaluate")                                                             # 评测接口
def evaluate(req: dict = None) -> dict:                                            # 跑工单 14 题
    """用工单 14 题跑一遍，返回准确率、检索精确度与响应时间；with_ragas=True 时附带 Ragas 四项。"""
    from . import evaluate_ragas as E                                              # 延迟导入，避免启动即拉起重依赖
    req = req or {}                                                                # 允许空请求体
    rows = E.run(with_cache=False)                                                 # 关缓存以测真实端到端耗时
    summary = E.summarize(rows)                                                    # 汇总准确率/精确度/耗时
    out = dict(summary=summary, rows=rows)                                         # 基础结果
    if req.get("with_ragas"): out["ragas"] = E.ragas_metrics(rows)                  # 可选：Ragas 四项指标
    return out                                                                     # 返回评测结果


@app.get("/health")                                                                # 健康检查
def health() -> dict:                                                              # 供长时间运行与排障使用
    """逐个探测依赖；任一依赖异常也返回 200，便于前端提示与监控告警。"""
    from . import vector_store as V                                                # 延迟导入
    try: V.milvus_client().list_collections(); milvus = "ok"                        # Milvus 探测：能列集合即连通
    except Exception as exc: milvus = f"error: {type(exc).__name__}"                # 记录错误类型而非堆栈
    stats = P.STATE.get("stats")                                                    # 优先用内存里的规模统计
    if not stats and C.INDEX_FILE.exists():                                         # 服务刚起还没提问过时内存为空
        try: stats = json.loads(C.INDEX_FILE.read_text(encoding="utf-8")).get("stats", {})  # 从落盘的索引文件读，避免显示空对象
        except Exception: stats = {}                                                # 文件损坏也不影响健康检查
    return dict(api="ok", llm_key=bool(C.DEEPSEEK_API_KEY), milvus=milvus,          # 服务、密钥、向量库
                redis=K.ping(), pdfs=[p.stem for p in C.docs()],                     # 缓存状态与知识库文件
                index=C.INDEX_FILE.exists(), stats=stats or {},                       # 索引是否已建与规模
                tables="ok")                                                       # 表格结构化模块已加载
