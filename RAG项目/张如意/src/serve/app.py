# -*- coding: utf-8 -*-
"""农户问答前端（最小可用版）。

    python src/serve/app.py            # 起服务
    浏览器打开 http://127.0.0.1:8000

## 为什么不上一套构建链

你另一个项目（D:/zg6/rag）用的是 React + antd + vite，那是完整产品。
这一版是**最小可用**：单个 HTML + FastAPI，无 npm、无打包、无 CDN 依赖，
断网也能跑。跑通、拿到真实问法之后再决定要不要换成正经前端。

## 三条要求怎么落的

1. **高危提示视觉强化** —— 红色警示框 + ⚠️ 图标 + 粗边框，不只靠文字。
   `needs_verification_hint=true` 时**必定**渲染，且排在答案上方。
2. **「没找到答案？」入口** —— 每条回答下方都有，点开写原始问法，
   落盘到 `data/collected_queries.jsonl`，格式**直接对齐评估集**（追加即可重跑评估）。
3. **最小可用** —— 单输入框 + 作物选择 + 答案区，没有多余交互。
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "retrieve"))
sys.path.insert(0, str(ROOT / "src" / "generate"))

from fastapi import FastAPI                      # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel                   # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
COLLECTED = ROOT / "data" / "collected_queries.jsonl"
CROPS = ["黄瓜", "辣椒", "大蒜"]   # 前端作物下拉框的可选项（与语料覆盖的作物一致）

app = FastAPI(title="农业知识问答")
_searcher = None   # 全局单例缓存，避免每次请求都重新加载模型


def searcher():
    """懒加载全局 Searcher（Milvus + BGE-M3 + reranker）。

    模型加载约 20 秒，不能放在 import 时做；首次请求时加载一次并缓存。
    """
    global _searcher
    if _searcher is None:
        from search import Searcher
        _searcher = Searcher()
    return _searcher


class Ask(BaseModel):
    """POST /api/ask 的请求体：query 必填；crop 可选（None = 不限作物）。"""
    query: str
    crop: str | None = None


class Feedback(BaseModel):
    """POST /api/feedback 的请求体：原始问法 + 作物 + 反馈原因 + 系统当时的答案。"""
    query: str
    crop: str | None = None
    reason: str = "没找到答案"
    answer: str = ""


@app.get("/", response_class=HTMLResponse)
def index():
    """首页路由：返回 static/index.html 的原文内容（单文件前端，无模板引擎）。"""
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(a: Ask):
    """问答接口：检索（search）→ 生成（compose）→ 组装 JSON 返回。

    fallback_* 状态直接返回兜底话术；正常状态返回答案文本、去重后的
    来源列表、高危/需核实/暂无方案等标志位与耗时（秒，两位小数）。
    """
    from search import search, FALLBACK_MSG, VERIFIED, PENDING
    import answer as gen

    t0 = time.time()
    s = searcher()
    # Web 端固定取 Top-3：答案卡篇幅有限，多了农户看不完
    status, q2, intent, results, why = search(s, a.query, crop=a.crop, topk=3)

    if status.startswith("fallback"):
        return {
            "status": status, "query": a.query, "normalized": q2,
            "intent": intent["intent"] if intent else None,
            "answer": FALLBACK_MSG, "reason": why,
            "is_high_risk": False, "needs_verification": False,
            "show_noplan_notice": False, "sources": [], "chunks": [],
            "elapsed": round(time.time() - t0, 2),
        }

    # nq 传检索层的归一化问句：症状判据必须与 search() 的 coverage_check 同输入，
    # 否则「大蒜小白虫打什么药」（→蓟马）这类口语病名会被误判成症状问法
    text = gen.compose(a.query, a.crop, results, dry_run=False,
                       noplan=(status == "ok_noplan"), nq=q2)
    chunks = [gen._CARDS.get(r["chunk_id"]) for r in results]
    chunks = [c for c in chunks if c]

    # 来源去重
    seen, sources = set(), []
    for c in chunks:
        src = c["source"]
        k = (src["std_no"], src["section"], src["page"])
        if k in seen:
            continue
        seen.add(k)
        sources.append({"std_no": src["std_no"], "section": src["section"],
                        "page": src["page"]})

    return {
        "status": status, "query": a.query, "normalized": q2,
        "intent": intent["intent"] if intent else None,
        "answer": text, "reason": why,
        "is_high_risk": any(c["is_high_risk"] for c in chunks),
        # 生成层已带核实提示文案，这里再给一个标志位让前端做**视觉强化**
        "needs_verification": any(c["needs_verification_hint"] for c in chunks),
        "show_noplan_notice": status == "ok_noplan",
        "sources": sources,
        "chunks": [{"chunk_id": c["card_id"], "subtype": c["subtype"],
                    "crop": c["crop"], "section": c["source"]["section"],
                    "page": c["source"]["page"],
                    "needs_verification": c["needs_verification_hint"]}
                   for c in chunks],
        "elapsed": round(time.time() - t0, 2),
    }


@app.post("/api/feedback")
def api_feedback(f: Feedback):
    """「没找到答案？」——落盘成**评估集格式**，可直接追加去重跑评估。

    这是真实语料采集的入口：农户/试用者写的原始问法，比我们从卡片反推的
    问法有价值得多（能测出表述差异，auto 评估集测不到）。
    """
    COLLECTED.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "query": f.query,
        "crop": f.crop,
        # expect_cards 留空——真实问法的正确答案要人工标注，
        # **不能**用系统召回的卡当标准答案（那是循环论证）
        "expect_cards": [],
        "expect_fallback": None,
        "source": "collected",
        "is_high_risk": None,
        "note": f"{f.reason}；系统当时答：{f.answer[:60]}",
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(COLLECTED, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = sum(1 for _ in open(COLLECTED, encoding="utf-8"))
    return {"ok": True, "total": n, "path": str(COLLECTED.relative_to(ROOT))}


@app.get("/api/health")
def health():
    """给截图脚本用的就绪探针——模型加载完才返回 ok"""
    try:
        searcher()
        return {"ok": True, "crops": CROPS}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)


if __name__ == "__main__":
    import uvicorn
    print("载入模型…（首次约 20 秒）")
    searcher()
    print("就绪 → http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
