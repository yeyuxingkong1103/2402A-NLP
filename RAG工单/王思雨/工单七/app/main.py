# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— HTTP 层：FastAPI 路由，只做分发不含算法
import json, time                                                            # 读结果文件、计时
from fastapi import FastAPI, HTTPException                                   # Web 框架
from fastapi.responses import JSONResponse                                   # 显式 JSON 返回
from fastapi.staticfiles import StaticFiles                                  # 挂载前端静态页
from pydantic import BaseModel                                               # 请求体校验
from . import config as C                                                    # 配置
from .rag_core import retrieve, answer, milvus, rds, llm                      # 复用核心链路
# 说明：评估跑一轮约 15 分钟，不做成 HTTP 接口（会阻塞事件循环），由 scripts/run_eval.py 触发
app = FastAPI(title="RAG 功能测试及评估（工单七）", version="1.0")             # 应用实例

class Q(BaseModel):                                                          # 单问题请求体
    question: str                                                            # 问题文本
    mode: str = "hybrid"                                                     # 检索模式：hybrid/vector/bm25
    use_cache: bool = True                                                   # 是否允许命中缓存

@app.get("/health")                                                          # 健康检查：演示前先确认三件套是否在线
def health():                                                                # 无参
    st = {"milvus": "down", "redis": "down", "llm": "unknown", "collection": C.MILVUS_COLLECTION}  # 初始状态
    try:                                                                     # 逐项探测，任一失败不影响其它
        st["chunks"] = milvus().query(C.MILVUS_COLLECTION, filter="", output_fields=["count(*)"], consistency_level="Strong")[0]["count(*)"]  # 用 count(*)：get_collection_stats 的 row_count 在未落盘时是陈旧值
        st["milvus"] = "up"                                                  # 标记在线
    except Exception as e:                                                    # Milvus 不可用
        st["milvus"] = f"down: {type(e).__name__}"                            # 记录原因
    try:                                                                     # Redis 探测
        st["redis"] = "up" if rds().ping() else "down"                        # PING 通即在线
    except Exception as e:                                                    # Redis 不可用
        st["redis"] = f"down: {type(e).__name__}"                             # 记录原因
    st["llm"] = "configured" if C.LLM_API_KEY else "missing DEEPSEEK_API_KEY"  # 只报是否配置了 Key，不回显
    return st                                                                # 返回状态

@app.get("/questions")                                                       # 返回 10 题题面与标准答案
def questions():                                                             # 无参
    d = json.loads(C.QUESTIONS_FILE.read_text(encoding="utf-8"))               # 直接读题集文件（配置层只给路径，读法随调用方）
    return {"questions": d["questions"] if isinstance(d, dict) else d}         # 供前端列表渲染

@app.post("/retrieve")                                                       # 只跑检索（演示"检索结果"这一步）
def do_retrieve(q: Q):                                                       # 请求体含问题
    t0 = time.time()                                                         # 计时
    try:                                                                     # 链路异常要给前端可读信息
        hits, stat = retrieve(q.question, q.mode)                             # 混合检索 + 重排
    except Exception as e:                                                    # 失败
        raise HTTPException(500, f"检索失败：{type(e).__name__}: {e}")         # 明确报错
    return {"question": q.question, "stat": stat, "latency_ms": round((time.time() - t0) * 1000),
            "contexts": [{"n": i + 1, "company": h["company"], "year": h["year"], "page": h["page"],
                          "src": h["src"], "rscore": round(h.get("rscore", 0), 4),
                          "text": h["text"][:400]} for i, h in enumerate(hits)]}  # 截断正文便于展示

@app.post("/ask")                                                            # 完整问答
def do_ask(q: Q):                                                            # 请求体含问题
    try:                                                                     # 链路异常要给前端可读信息
        return answer(q.question, q.mode, q.use_cache)                        # 检索 + 生成 + 缓存
    except Exception as e:                                                    # 失败
        raise HTTPException(500, f"问答失败：{type(e).__name__}: {e}")         # 明确报错

@app.get("/results")                                                         # 列出历次评估结果
def list_results():                                                          # 无参
    fs = sorted(C.RESULTS.glob("eval_*.json"))                                # 只取正式评估文件
    return {"files": [f.name for f in fs]}                                    # 返回文件名列表

@app.get("/results/latest")                                                  # 取最近一次评估结果（前端主页数据源）
def latest():                                                                # 无参
    fs = sorted(C.RESULTS.glob("eval_*.json"))                                # 全部评估文件
    if not fs:                                                               # 还没跑过
        return JSONResponse({"empty": True, "hint": "先执行 python -m app.evaluate"}, status_code=200)  # 友好提示
    return json.loads(fs[-1].read_text(encoding="utf-8"))                     # 返回最后一次的完整结果

app.mount("/", StaticFiles(directory=str(C.ROOT / "frontend"), html=True), name="ui")  # 前端挂根路径（含 Vue）
