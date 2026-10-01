# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""评估接口：跑 10 题对比评估，读取历史报告。"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import settings
from app.core.evaluator import Evaluator
from app.schemas import EvalReport

router = APIRouter(tags=["evaluate"])

# 进程内的评估状态（10 题 × (RAG + 纯LLM + judge) ≈ 30 次 LLM 调用，约 2–5 分钟）
_state: dict = {"running": False, "done": 0, "total": 0, "message": "idle", "error": ""}
_last_report: dict | None = None


def _run_eval_sync(limit: int | None, with_no_rag: bool, with_judge: bool) -> dict:
    ev = Evaluator()

    def progress(i: int, total: int, item) -> None:
        _state["done"] = i
        _state["total"] = total
        _state["message"] = f"已评估 {i}/{total}（题 {item.id}）"

    report = asyncio.run(ev.evaluate_all(
        limit=limit, with_no_rag=with_no_rag, with_judge=with_judge,
        progress=progress,
    ))
    Evaluator.save_report(report)
    return report


async def _background(limit, with_no_rag, with_judge) -> None:
    global _last_report
    try:
        _last_report = await asyncio.to_thread(
            _run_eval_sync, limit, with_no_rag, with_judge)
        _state["message"] = "完成"
    except Exception as e:  # noqa: BLE001
        _state["error"] = f"{type(e).__name__}: {e}"
        _state["message"] = "失败"
    finally:
        _state["running"] = False


@router.post("/api/evaluate/run")
async def run_eval(background: BackgroundTasks, limit: int | None = None,
                   with_no_rag: bool = True, with_judge: bool = True) -> dict:
    if _state["running"]:
        raise HTTPException(409, "评估任务已在运行")
    _state.update(running=True, done=0, total=limit or 10,
                  message="启动中", error="")
    background.add_task(_background, limit, with_no_rag, with_judge)
    return {"ok": True, "state": dict(_state)}


@router.get("/api/evaluate/status")
async def eval_status() -> dict:
    return dict(_state)


@router.get("/api/evaluate/report", response_model=EvalReport)
async def eval_report() -> EvalReport:
    """返回最近一次报告：优先内存，其次磁盘上的 eval-latest.json。"""
    rep = _last_report
    if rep is None:
        p = settings.data_path / "eval" / "eval-latest.json"
        if not p.exists():
            raise HTTPException(404, "尚无评估报告，请先 POST /api/evaluate/run")
        rep = json.loads(p.read_text(encoding="utf-8"))
    return EvalReport(**rep)


@router.get("/api/evaluate/questions")
async def eval_questions() -> dict:
    """返回题面清单（不含真值关键词，避免前端剧透）。"""
    qs = Evaluator.load_questions()
    return {"items": [{"id": q["id"], "question": q["question"],
                       "category": q.get("category", "")} for q in qs]}
