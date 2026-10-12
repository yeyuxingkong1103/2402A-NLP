# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：FastAPI 入口。只做参数校验与路由分发，检索/重排/融合/问答逻辑全部在 rag_core。
from contextlib import asynccontextmanager                                          # 生命周期钩子
from typing import List, Optional                                                   # 类型标注
from fastapi import FastAPI, HTTPException                                          # Web 框架
from fastapi.middleware.cors import CORSMiddleware                                  # 跨域
from fastapi.staticfiles import StaticFiles                                         # 前端静态挂载
from pydantic import BaseModel, Field                                               # 请求体校验
from . import config as C                                                           # 全局配置
from . import rag_core as R                                                         # 核心逻辑

@asynccontextmanager                                                                 # 启动/关闭钩子
async def lifespan(app: FastAPI):                                                    # 应用生命周期
    """启动即载入索引与预热向量模型：把首次推理开销前移，避免首个请求独自承担以备达标 3 秒。"""
    try:                                                                             # 预热失败不应阻断启动
        R.bm25_load()                                                                # 载入 BM25 索引
        R.embed(["预热"])                                                             # 预热向量模型
        R.rerank_model()                                                             # 预热重排模型：首次加载约 3 秒，不预热就摊到首个真实请求上
        R.retrieve("预热查询", "hybrid", 3, rerankers=C.DEFAULT_RERANKERS)             # 端到端跑一次检索：把编码/Milvus/BM25/融合/重排打分的首次开销全部前移
        R.llm_chat([{"role": "user", "content": "hi"}], max_tokens=1, timeout=30)     # 预热大模型连接，付掉 DNS+TLS 握手
    except Exception as exc:                                                         # GPU/路径/网络异常
        print(f"[启动] 预热失败（服务继续以降级模式运行）：{type(exc).__name__}: {exc}")  # 如实上报
    yield                                                                            # 应用运行中

app = FastAPI(title="招股说明书混合检索问答（工单六）", version="1.0.0", lifespan=lifespan)  # 应用实例
app.add_middleware(CORSMiddleware, allow_origins=["*"],                            # 演示用：允许任意来源
                   allow_methods=["*"], allow_headers=["*"])                        # 允许全部方法与头

class SearchReq(BaseModel):                                                          # 检索请求
    question: str = Field(min_length=1)                                              # 用户问题
    mode: str = "hybrid"                                                             # vector / bm25 / hybrid
    top_k: Optional[int] = None                                                      # 返回条数
    vec_weight: Optional[float] = None                                               # 向量通道权重
    bm25_weight: Optional[float] = None                                              # 关键词通道权重
    fusion: str = "weighted"                                                         # weighted / rrf / vote
    rerankers: Optional[List[str]] = None                                            # 重排器顺序

class ChatReq(SearchReq):                                                            # 问答请求
    session_id: str = "default"                                                      # 会话号，用于多轮历史

class FeedbackReq(BaseModel):                                                        # 反馈请求
    chunk_id: str = ""                                                               # 被评价的片段
    doc: str = ""                                                                    # 所属文档
    rating: int = 0                                                                  # 1 点赞 / -1 点踩
    session_id: str = "default"                                                      # 会话号
    question: str = ""                                                               # 对应问题（便于留痕）

@app.get("/health")                                                                  # 健康检查
def health() -> dict:                                                                # 逐组件报状态
    """报告 Milvus/Redis/大模型/索引的可达性，任一降级都要如实标出而不是笼统回 OK。"""
    import redis as _redis                                                           # 延迟导入
    idx = R._STORE["chunks"] if R._STORE["chunks"] else R.bm25_load()                 # 载入索引
    out = {"status": "ok", "collection": C.MILVUS_COLLECTION, "chunks": len(idx),    # 基础信息
           "llm_key": bool(C.DEEPSEEK_API_KEY), "degraded": []}                       # 密钥是否配置
    try:                                                                             # Milvus 探活
        R.milvus_client().get_collection_stats(collection_name=C.MILVUS_COLLECTION)   # 取统计
        out["milvus"] = True                                                         # 可达
    except Exception as exc:                                                         # 不可达
        out["milvus"], out["status"] = False, "degraded"                             # 标记降级
        out["degraded"].append(f"milvus:{type(exc).__name__}")                       # 记原因
    try:                                                                             # Redis 探活
        _redis.Redis.from_url(C.REDIS_URL, protocol=C.REDIS_PROTOCOL).ping()          # 显式 RESP2
        out["redis"] = True                                                          # 可达
    except Exception as exc:                                                         # 不可达
        out["redis"], out["status"] = False, "degraded"                              # 标记降级
        out["degraded"].append(f"redis:{type(exc).__name__}")                        # 记原因
    return out                                                                       # 返回

@app.get("/config")                                                                  # 返回可用配置
def get_config() -> dict:                                                            # 供前端渲染下拉框
    """列出检索模式、融合算法、重排器、默认权重与 topK，前端据此生成交互控件。"""
    return {"modes": ["vector", "bm25", "hybrid"],                                   # 三种检索方式
            "fusions": ["weighted", "rrf", "vote"],                                  # 三种融合算法
            "fusion_labels": {"weighted": "加权平均", "rrf": "RRF 融合", "vote": "投票机制"},  # 中文名
            "rerankers": ["bge", "tfidf", "adaptive"],                               # 三种重排算法
            "reranker_labels": {"bge": "bge-reranker-v2-m3", "tfidf": "TF-IDF",      # 中文名
                                "adaptive": "用户反馈自适应"},                        # 同上
            "embed_models": list(C.EMBED_MODELS),                                    # 可用嵌入模型
            "defaults": {"mode": "hybrid", "fusion": "weighted",                     # 默认策略
                         "vec_weight": C.VEC_WEIGHT, "bm25_weight": C.BM25_WEIGHT,   # 默认权重
                         "top_k": C.FUSION_TOP_K, "rerank_top_n": C.RERANK_TOP_N,    # 默认条数
                         "rerankers": C.DEFAULT_RERANKERS}}                             # 默认重排顺序

@app.post("/search")                                                                 # 检索接口
def search(req: SearchReq) -> dict:                                                  # 只检索不生成
    """执行向量/全文/混合检索并返回片段，便于单独验证检索精度与排序。"""
    cands, deg = R.retrieve(req.question, req.mode, req.top_k, req.vec_weight,        # 调用统一检索入口
                            req.bm25_weight, req.fusion, req.rerankers)              # 传入全部配置
    return {"question": req.question, "mode": req.mode, "fusion": req.fusion,        # 回执配置
            "count": len(cands), "degraded": deg,                                    # 命中数与降级
            "results": [{"chunk_id": c.get("chunk_id"), "doc": c.get("doc"),         # 片段明细
                         "page": c.get("page_no"), "title": c.get("title"),          # 出处
                         "channel": c.get("channels"), "score": round(float(c.get("fused", 0)), 4),  # 通道与分数
                         "rank_detail": c.get("rank_detail"),                        # 两路原始名次
                         "snippet": (c.get("text") or "")[:200]} for c in cands]}    # 摘要

@app.post("/chat")                                                                   # 问答接口
def chat(req: ChatReq) -> dict:                                                      # 多轮问答
    """检索 + DeepSeek 生成，返回答案、引用来源与降级信息；历史写 Redis 支撑多轮指代消解。"""
    try:                                                                             # 兜底任何未预期异常
        return R.answer(req.question, req.session_id, req.mode, req.top_k,            # 调用问答核心
                        req.vec_weight, req.bm25_weight, req.fusion, req.rerankers)   # 传入检索配置
    except Exception as exc:                                                         # 内部错误
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")   # 明确回因

@app.post("/feedback")                                                               # 反馈接口
def feedback(req: FeedbackReq) -> dict:                                              # 写反馈
    """把点赞/点踩写入 Redis，供自适应重排器调整片段与文档权重。"""
    if req.rating not in (-1, 0, 1):                                                 # 校验取值
        raise HTTPException(status_code=400, detail="rating 只能是 1 / 0 / -1")        # 明确报错
    R.fb_add(req.chunk_id, req.rating, req.doc)                                      # 写入
    return {"ok": True, "message": "反馈已记录，自适应重排器将在后续检索中使用",           # 回执
            "chunk_id": req.chunk_id, "rating": req.rating}                          # 回显

@app.post("/ingest")                                                                 # 入库接口
def ingest_ep() -> dict:                                                             # 解析并入库
    """解析 data/raw 下全部 PDF、分块、重建 BM25 与 Milvus 集合；耗时较长，返回处理统计。"""
    from . import ingest as I                                                        # 延迟导入，避免启动即载解析依赖
    try:                                                                             # 解析可能失败
        return I.run()                                                               # 执行入库
    except Exception as exc:                                                         # 解析/写库失败
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")   # 明确回因

@app.post("/eval")                                                                   # 评估接口
def eval_ep(limit: Optional[int] = None) -> dict:                                    # 运行 RAGAS
    """跑 RAGAS 四指标并映射为准确率/召回率，结果落盘 results/。"""
    from . import eval_ragas as E                                                    # 延迟导入，避免启动即载 ragas
    return E.run(limit=limit)                                                        # 执行评估

_UI = C.ROOT / "frontend"                                                            # 前端目录
if _UI.exists():                                                                     # 存在才挂载
    app.mount("/", StaticFiles(directory=str(_UI), html=True), name="ui")            # 挂到根路径，路由优先

if __name__ == "__main__":                                                           # 允许 python -m app.main
    import uvicorn                                                                   # 服务器
    uvicorn.run("app.main:app", host=C.API_HOST, port=C.API_PORT,                    # 监听配置
                reload=False, log_level=C.LOG_LEVEL.lower())                         # 关热重载，避免模型被重复加载
