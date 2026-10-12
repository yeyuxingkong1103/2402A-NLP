# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""FastAPI 服务入口：问答、优化前后对比、健康检查与建索引，支持中文与英文问答。"""
import time                                                                                   # 统计接口层耗时
from fastapi import FastAPI                                                                   # Web 框架
from fastapi.middleware.cors import CORSMiddleware                                            # 允许 Vue 前端跨域调用
from pydantic import BaseModel, Field                                                         # 请求体校验
from . import config as C                                                                     # 复用集中配置
from . import rag_pipeline as P                                                               # 复用问答流水线

app = FastAPI(title="基于PDF文档的问答系统（优化版）", version="1.0")                             # 应用实例
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # 开发期放开跨域，便于 Vue 调试


class AskRequest(BaseModel):                                                                  # 提问请求体
    """一次提问的参数：问题、检索模式（优化后 / 优化前）、是否使用缓存。"""
    question: str = Field("", description="用户问题，支持中文或英文")                              # 问题文本
    mode: str = Field("optimized", description="optimized=优化后，baseline=优化前")               # 检索模式
    use_cache: bool = Field(True, description="是否启用 Redis 缓存")                              # 缓存开关


@app.post("/api/ask")                                                                         # 单模式问答接口
def ask(req: AskRequest) -> dict:                                                             # 处理提问
    """按指定模式检索并生成答案，返回答案、各阶段耗时与命中的上下文页码。"""
    t0 = time.time()                                                                          # 接口层计时起点
    return dict(P.answer(req.question, mode=req.mode, use_cache=req.use_cache), api_elapsed=round(time.time() - t0, 3))  # 附加接口耗时


@app.post("/api/compare")                                                                     # 优化前后对比接口
def compare(req: AskRequest) -> dict:                                                         # 同一问题跑两种模式
    """同一问题分别走优化前与优化后链路，输出两个答案与耗时，供演示对比。"""
    return dict(question=req.question, before=P.answer(req.question, "baseline", False),       # 优化前：仅向量检索
                after=P.answer(req.question, "optimized", False))                              # 优化后：混合检索 + 重排


@app.get("/api/health")                                                                       # 健康检查接口
def health() -> dict:                                                                         # 供长时间运行与排障使用
    """逐个探测依赖；任一依赖异常也返回 200，便于前端提示与监控告警。"""
    import redis as _r                                                                        # 延迟导入，避免启动即连库
    from . import retrieval as R                                                              # 延迟导入
    out = {"api": "ok", "llm_key": bool(C.DEEPSEEK_API_KEY), "pdf": C.PDF_PATH.exists(),       # 服务、密钥、知识库状态
           "index": (C.ROOT / "data" / "index.json").exists()}                                 # 索引是否已构建
    for name, probe in (("milvus", lambda: R.milvus_client().list_collections()),              # Milvus 探测
                        ("redis", lambda: _r.Redis.from_url(C.REDIS_URL, protocol=2, socket_timeout=0.5).ping())):  # Redis 探测
        try: probe(); out[name] = "ok"                                                         # 探测成功
        except Exception as exc: out[name] = f"error: {type(exc).__name__}"                     # 记录错误类型而非堆栈
    return out                                                                                # 返回状态汇总
# 说明：索引构建不做成接口 —— 工单未要求，且重建索引是运维动作，用 python -m src.rag_pipeline 即可（见 05-部署.md）
