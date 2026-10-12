# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：FastAPI 服务。对外提供 /api/ask 问答接口与 /api/health 健康检查，并把前端单文件挂在根路径。
from contextlib import asynccontextmanager                                  # 生命周期上下文
from fastapi import FastAPI                                                 # Web 框架
from fastapi.middleware.cors import CORSMiddleware                          # 跨域，便于前端独立调试
from fastapi.responses import FileResponse, JSONResponse                     # 静态页与 JSON 响应
from pydantic import BaseModel, Field                                        # 入参校验
from backend import cache as K                                              # 缓存健康探测
from backend import rag as R                                                # 问答流水线
import config as C                                                          # 集中配置


@asynccontextmanager                                                         # 生命周期钩子（on_event 在新版 FastAPI 已弃用）
async def lifespan(app: FastAPI):                                            # 启动/关闭
    """启动即加载索引与本地模型：否则首个请求要多等十几秒，会被当成系统卡住。预热失败不阻断启动。"""
    print("[预热]", R.warmup())                                               # 预热
    yield                                                                   # 交给应用运行


app = FastAPI(title="招股说明书图文问答", version="1.0", lifespan=lifespan)    # 应用实例
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # 演示环境放开跨域


class Ask(BaseModel):                                                       # 问答入参
    """question 必填；lang 可选 zh/en/auto，缺省按问题字符自动判定。"""
    question: str = Field("", description="用户问题，中英文均可")                # 问题文本
    lang: str = Field("auto", description="zh / en / auto")                    # 语言
    # 压测/评测用：绕开服务端缓存取真实端到端耗时。
    # 坑：`test_questions.py --no-cache` 走 HTTP 时，补丁只打在**测试进程**里，
    # 缓存却在**服务端**——于是 16 题全部命中缓存、耗时 3 ms，性能数据完全失真且不报错。
    no_cache: bool = Field(False, description="true 时绕开问答缓存")            # 缓存旁路开关


@app.get("/api/health")                                                     # 健康检查
def health():                                                               # 依赖状态汇总
    """回报 Milvus / Redis 连通性与已载入块数，便于排障与验收。"""
    milvus = "ok" if R.STATE.get("client") is not None else "unavailable"    # Milvus 连接状态
    return dict(status="ok", milvus=milvus, redis=K.ping(),                  # 各依赖状态
                chunks=len(R.STATE.get("chunks", [])),                       # 已载入块数
                stats=R.STATE.get("stats") or {},                            # 索引规模统计
                device=C.DEVICE, collection=C.MILVUS_COLLECTION)             # 设备与集合名


@app.post("/api/ask")                                                       # 问答接口
def ask(body: Ask):                                                         # 入参：问题与语言
    """返回 {answer, sources, retrieval_precision, elapsed_ms}；任何异常都转成友好提示，不返回 5xx 崩溃栈。"""
    try:                                                                    # 全链路兜底
        r = R.answer(body.question, body.lang or "auto", use_cache=not body.no_cache)  # 走完整问答流水线
        return dict(answer=r["answer"],                                      # 答案
                    sources=[dict(doc=s["doc"], page_no=s["page_no"],         # 来源：文档与页码
                                  block_type=s["block_type"],                 # 块类型（image 表示来自图像解析）
                                  image_type=s.get("image_type", ""),         # ★图像语义类型
                                  locator=s.get("locator", ""),               # 可读定位串
                                  score=s["score"],                           # 相关度
                                  text=s["text"][:300]) for s in r["contexts"]],  # 片段摘要
                    retrieval_precision=r.get("retrieval_precision", 0.0),    # ★检索精确度
                    elapsed_ms=round(r.get("elapsed", 0) * 1000),             # ★端到端耗时（毫秒）
                    lang=r.get("lang"), cached=r.get("cached", False),        # 语言与是否命中缓存
                    retrieval_mode=r.get("retrieval_mode"),                   # 实际生效的检索通道
                    degraded=r.get("degraded", False),                        # 是否降级
                    degraded_reason=r.get("degraded_reason", ""),             # 降级说明
                    retrieval_ms=r.get("retrieval_ms"),                       # 检索+重排耗时
                    generate_ms=r.get("generate_ms"))                         # 生成耗时
    except Exception as e:                                                  # 未预期异常
        print(f"[接口异常] {type(e).__name__}: {e}")                          # 打印便于排障
        return JSONResponse(status_code=200,                                 # 仍返回 200 + 友好提示，前端统一处理
                            content=dict(answer="系统繁忙，请稍后重试。", sources=[],  # 空答案
                                         retrieval_precision=0.0, elapsed_ms=0,  # 指标置零
                                         error=type(e).__name__))            # 附错误类型


@app.get("/api/questions")                                                  # 工单问题列表
def questions():                                                            # 供前端"点击填入"与演示使用
    """返回工单四指定的 16 个检索问题（不含标准答案，答案仅用于离线评测）。"""
    import json                                                             # 延迟导入
    if not C.QUESTIONS.exists(): return []                                   # 列表缺失时返回空
    data = json.loads(C.QUESTIONS.read_text(encoding="utf-8"))                # 读取题库
    return [dict(id=x.get("id"), question=x.get("question")) for x in data]   # 只回问题，不回答案


@app.get("/")                                                               # 首页
def index():                                                                # 返回 Vue 单文件页面
    """前端为 CDN 单文件，直接由后端托管，部署时无需另起 Node 服务。"""
    f = C.ROOT / "frontend" / "index.html"                                   # 页面路径
    return FileResponse(f) if f.exists() else JSONResponse({"detail": "前端文件缺失"})


@app.get("/vue.global.prod.js")                                             # Vue 库的同源兜底
def vue_lib():                                                              # CDN 不可达时前端回落到本地副本
    """工单要求 Vue 走 CDN 单文件；但外部 CDN 偶发连接重置会让整页停在未渲染状态，
    故备一份同源副本，由 index.html 在 window.Vue 缺失时加载（nginx 部署下由静态目录直接提供）。"""
    return FileResponse(C.ROOT / "frontend" / "vue.global.prod.js")          # 本地副本


@app.get("/api/image")                                                      # 图像回显
def image(path: str):                                                       # 按相对路径取原图
    """来源里标了图像块时，前端可点开看原始插图，验证"答案是图里读出来的"。"""
    p = (C.ROOT / path).resolve()                                            # 解析绝对路径
    if not str(p).startswith(str(C.ROOT)) or not p.exists():                  # 防目录穿越
        return JSONResponse({"detail": "图像不存在"}, status_code=404)        # 越界或缺失
    return FileResponse(p)                                                  # 返回图片
