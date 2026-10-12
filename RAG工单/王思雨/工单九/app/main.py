# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# HTTP 层：FastAPI 路由，只做分发与展示，算法全在下层模块
import json                                                                     # 读结果文件
from fastapi import FastAPI, HTTPException, UploadFile, File                    # Web 框架与上传
from fastapi.responses import JSONResponse; from fastapi.staticfiles import StaticFiles  # JSON 返回 / 挂载 Vue 前端静态页
from . import config as C, graph_rag as G                                       # 配置 + 图谱层
from .schemas import Ask, Trip                                                  # 请求模型
from .retrieval import answer_after, answer_before; from .eval_ragas import parse_pdf, load_questions  # 两条问答链路 / 三级降级解析与题集加载
import 工单八.app.rag_core as R8                                                # 复用 08 的连接原语做健康探测
app = FastAPI(title="工单九 · GraphRAG 优化（金融问答）", version="1.0")          # 应用实例
@app.get("/health")                                                             # 健康检查：演示前先确认四件套是否在线
def health():                                                                   # 无参
    st = {"milvus": "down", "redis": "down", "graph": None, "collection": C.C8.MILVUS_COLLECTION, "llm": "configured" if C.C8.LLM_API_KEY else "missing DEEPSEEK_API_KEY"}  # 初始状态（只报是否配了 Key，不回显）
    try: st["chunks"] = R8.milvus().query(C.C8.MILVUS_COLLECTION, filter="", output_fields=["count(*)"], consistency_level="Strong")[0]["count(*)"]; st["milvus"] = "up"  # 用 count(*) 而非 row_count（后者未落盘时是陈旧值）
    except Exception as e: st["milvus"] = f"down: {type(e).__name__}"            # Milvus 不可用，记录原因
    try: st["redis"] = "up" if R8.rds().ping() else "down"                       # PING 通即在线
    except Exception as e: st["redis"] = f"down: {type(e).__name__}"             # Redis 不可用
    try: st["graph"] = G.stats()                                                 # 图谱规模与社区数
    except Exception as e: st["graph"] = f"missing: {type(e).__name__}（先跑 python -m app.graph_rag build）"  # 未建图时给出下一步命令
    try: R8.embed(["预热"]); R8.rerank("预热", [{"text": "预热"}]); st["models"] = "warm"  # 预热向量与重排模型：冷启动加载实测约 40 秒，不预热会吃掉首个请求的 3 秒预算并触发降级
    except Exception as e: st["models"] = f"warmup_failed: {type(e).__name__}"    # 预热失败如实上报，不阻断健康检查
    return st                                                                   # 返回状态
@app.post("/ask")                                                               # after 链路：工单九优化后的 GraphRAG 问答
def do_ask(q: Ask):                                                             # 请求体含问题/缓存开关/语言
    try: return answer_after(q.question, q.use_cache, q.lang)                    # 检索 + 生成 + 缓存 + 降级
    except Exception as e: raise HTTPException(500, f"问答失败：{type(e).__name__}: {e}")  # 明确报错
@app.post("/ask/before")                                                        # before 链路：08 原 GraphRAG（同题对比与现场演示用）
def do_before(q: Ask):                                                          # 请求体同上
    try: return answer_before(q.question, q.use_cache)                          # 直调 08 原链路
    except Exception as e: raise HTTPException(500, f"对比链路失败：{type(e).__name__}: {e}")  # 明确报错
@app.get("/graph")                                                              # 知识图谱可视化数据源
def graph(limit: int = 120):                                                    # limit=最多返回节点数（前端力导向图不宜过大）
    g = G.graph(); ns = sorted(g.nodes(data=True), key=lambda x: -x[1].get("freq", 1))[:limit]; ids = {n for n, _ in ns}  # 按频次取前 N（保证显示的是主干）并得到视图内节点集
    return {"nodes": [{"id": n, **a} for n, a in ns], "edges": [{"s": s, "t": t, **a} for s, t, a in g.edges(data=True) if s in ids and t in ids][:limit * 2], "stats": G.stats(), "communities": G._M.get("comm", [])}  # 节点 / 两端都在视图内的边 / 规模与社区摘要
@app.post("/graph/text")                                                        # 演示：贴一段文本 → DeepSeek 抽实体关系 → 并回图
def graph_text(t: Trip):                                                        # 请求体含正文与可选公司代码
    try: return G.add(t.text, t.code)                                           # 抽取 + 建图 + 落盘
    except Exception as e: raise HTTPException(500, f"建图失败：{type(e).__name__}: {e}")  # 明确报错
@app.post("/upload")                                                            # 上传 PDF → 三级降级解析（容错 E2）→ 抽实体关系入图
async def upload(file: UploadFile = File(...)):                                 # 表单文件
    try: p = C.DATA / "uploaded.pdf"; C.DATA.mkdir(parents=True, exist_ok=True); p.write_bytes(await file.read())  # 先确保 data/ 存在（缺失会让保存抛 FileNotFoundError、被误报成"保存失败"，实测踩到），再保存上传内容
    except Exception as e: raise HTTPException(500, f"保存上传文件失败：{type(e).__name__}: {e}")  # 保存失败明确报错
    try: text, parser = parse_pdf(str(p))                                       # 三级降级解析
    except Exception as e: raise HTTPException(400, f"解析失败（{type(e).__name__}）：{e}；请确认是带文本层的 PDF")  # 容错 E2：给出可读原因
    return {"parser": parser, "chars": len(text), "graph": G.add(text[:6000], "")}  # 只抽前 6000 字控成本，返回建图规模
@app.get("/questions")                                                          # 题集题面与参考答案（供前端展示）
def questions():                                                                # 无参
    try: return {"questions": load_questions()}                                  # 正常返回题集
    except Exception as e: return JSONResponse({"empty": True, "hint": str(e)}, status_code=200)  # PDF 未到位时返回可读提示而非 500
@app.get("/results/{which}")                                                    # before.json / after.json（前端对比页数据源）
def results(which: str):                                                        # which ∈ {before, after}
    p = C.EVAL / f"{which}.json"                                                # 结果文件
    if which not in ("before", "after") or not p.exists(): return JSONResponse({"empty": True, "hint": f"先跑 python -m app.eval_ragas（缺 eval/{which}.json）"}, status_code=200)  # 缺文件给友好提示
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))               # 返回结果
app.mount("/", StaticFiles(directory=str(C.FRONT), html=True), name="ui")        # 前端挂根路径（Vue 单页）
