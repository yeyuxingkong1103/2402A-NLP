# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：容错与长时间运行（硬性要求2「系统稳定性」）

两层能力：

  **容错**（常见异常逐条兜住，不让异常冒到用户面前）
    1. 空输入 / 超长输入        —— 入口校验，返回明确提示
    2. Ollama 不可用            —— 降级为抽取式答案（图谱事实 + 原文片段）
    3. 图谱加载失败             —— 退化为纯 RAG，标记 degraded
    4. 子图检索失败             —— 跳过图谱通道，仅用文本通道
    5. Qdrant 排他锁冲突        —— 给出可操作指引（换副本 / 关占用进程）
    6. Reranker 显存不足        —— 自动降级为按召回分排序
    7. 文档解析失败（上传 PDF） —— 返回可读错误，不中断会话
    8. 磁盘写满 / 日志写失败    —— 日志失败不影响主链路

  **长时间运行**
    · 日志轮转（8 MB × 5 份）
    · 内存监控（超阈值告警）
    · 健康检查（一键看全链路状态）
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import time
from logging.handlers import RotatingFileHandler

from w9 import config9

MAX_QUESTION_CHARS = 2000          # 超长输入上限（防注入超长上下文拖垮检索）
MIN_QUESTION_CHARS = 1


# ---------------------------------------------------------------------------
# 兼容守卫
# ---------------------------------------------------------------------------
class InputError(ValueError):
    """用户输入错误（可读提示，不是程序缺陷）。"""


def validate_question(question: str) -> str:
    """校验用户输入，返回清洗后的问题；非法则抛 InputError（带可读提示）。"""
    q = (question or "").strip()
    if len(q) < MIN_QUESTION_CHARS:
        raise InputError("请输入问题后再提交。")
    if len(q) > MAX_QUESTION_CHARS:
        raise InputError(f"问题过长（{len(q)} 字），请控制在 "
                         f"{MAX_QUESTION_CHARS} 字以内。")
    return q


# ---------------------------------------------------------------------------
# 日志（轮转）
# ---------------------------------------------------------------------------
_logger: logging.Logger | None = None


def get_logger(name: str = "w9") -> logging.Logger:
    """带轮转的文件日志（8 MB × 5）。日志初始化失败不抛异常。"""
    global _logger
    if _logger is not None:
        return _logger
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        try:
            os.makedirs(config9.LOG9, exist_ok=True)
            h = RotatingFileHandler(
                os.path.join(config9.LOG9, "w9.log"),
                maxBytes=config9.LOG_ROTATE_MAX_BYTES,
                backupCount=config9.LOG_ROTATE_BACKUPS,
                encoding="utf-8")
            h.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"))
            lg.addHandler(h)
        except Exception:                              # noqa: BLE001
            # 日志初始化失败（磁盘满/权限）不能影响主链路
            lg.addHandler(logging.NullHandler())
    _logger = lg
    return lg


# ---------------------------------------------------------------------------
# 内存监控
# ---------------------------------------------------------------------------
def memory_status() -> dict:
    """当前进程内存占用（psutil 缺失时优雅降级）。"""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        mb = p.memory_info().rss / (1024 * 1024)
        return {"rss_mb": round(mb, 1), "warn_mb": config9.MEMORY_WARN_MB,
                "over": mb > config9.MEMORY_WARN_MB}
    except Exception as exc:                           # noqa: BLE001
        return {"rss_mb": None, "warn_mb": config9.MEMORY_WARN_MB,
                "over": None, "error": str(exc)}


def disk_status() -> dict:
    """磁盘剩余空间。"""
    try:
        u = shutil.disk_usage(config9.DATA9)
        return {"free_gb": round(u.free / (1024 ** 3), 2),
                "total_gb": round(u.total / (1024 ** 3), 2)}
    except Exception as exc:                           # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
def health_check() -> dict:
    """全链路健康检查（界面「系统状态」与运维巡检用）。"""
    out: dict = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "workorder": config9.WORKORDER_ID}

    # 1) 图谱
    try:
        from w9.qa_v2 import get_kg9
        kg = get_kg9()
        out["graph"] = {
            "ok": kg is not None,
            "nodes": kg.g.number_of_nodes() if kg else 0,
            "edges": kg.g.number_of_edges() if kg else 0,
            "path": config9.KG9_PICKLE,
            "exists": os.path.isfile(config9.KG9_PICKLE),
        }
        if kg is not None:
            st = kg.stats()
            out["graph"]["by_relation"] = st.get("by_relation", {})
    except Exception as exc:                           # noqa: BLE001
        out["graph"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # 2) 向量库
    try:
        from src import vector_store
        n = vector_store.count() if hasattr(vector_store, "count") else None
        out["qdrant"] = {"ok": True, "path": config9.QDRANT9_LOCAL_PATH,
                         "is_copy": config9.QDRANT9_USING_COPY, "count": n}
    except Exception as exc:                           # noqa: BLE001
        out["qdrant"] = {"ok": False, "path": config9.QDRANT9_LOCAL_PATH,
                         "error": f"{type(exc).__name__}: {exc}",
                         "hint": "若为排他锁冲突，请关闭正在占用该目录的进程，"
                                 "或确认 工单9/data/qdrant 副本存在。"}

    # 3) LLM
    try:
        from src import llm
        h = llm.health()
        out["llm"] = {"ok": bool(h.get("ok")), "model": h.get("model"),
                      "error": h.get("error", "")}
    except Exception as exc:                           # noqa: BLE001
        out["llm"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "hint": "请先启动 Ollama：ollama serve"}

    # 4) 复用与只读性
    try:
        from w9 import compat
        out["reuse"] = compat.verify_reuse()
    except Exception as exc:                           # noqa: BLE001
        out["reuse"] = {"ok": False, "error": str(exc)}

    # 5) 资源
    out["memory"] = memory_status()
    out["disk"] = disk_status()
    out["ok"] = all(bool(out.get(k, {}).get("ok"))
                    for k in ("graph", "qdrant", "llm"))
    return out


# ---------------------------------------------------------------------------
# 故障清单（文档与界面共用）
# ---------------------------------------------------------------------------
FAULT_MATRIX: list[dict] = [
    {"id": 1, "fault": "空输入 / 超长输入",
     "symptom": "用户直接回车或粘贴整篇文档",
     "handling": "validate_question 入口校验，返回可读提示",
     "where": "w9/runtime9.py: validate_question"},
    {"id": 2, "fault": "Ollama 不可用 / 模型缺失",
     "symptom": "连接被拒、模型未 pull",
     "handling": "降级为抽取式答案（图谱事实 + 原文片段），如实标记 degraded",
     "where": "w9/qa_v2.py: _extractive_answer"},
    {"id": 3, "fault": "09 图谱加载失败",
     "symptom": "kg.pkl 缺失或损坏",
     "handling": "退化为纯文本 RAG，degraded=True 并给出运行指引",
     "where": "w9/qa_v2.py: get_kg9"},
    {"id": 4, "fault": "子图检索失败",
     "symptom": "图谱结构异常导致遍历报错",
     "handling": "跳过图谱通道，仅用文本通道，不中断问答",
     "where": "w9/retrieve_v2.py: retrieve_v2"},
    {"id": 5, "fault": "Qdrant 排他锁冲突",
     "symptom": "Storage folder is already accessed by another instance",
     "handling": "09 使用独立副本 data/qdrant，并在健康检查里给出明确指引",
     "where": "w9/config9.py / runtime9.health_check"},
    {"id": 6, "fault": "Reranker 显存不足 / 加载失败",
     "symptom": "CUDA OOM 或权重缺失",
     "handling": "rerank_final 捕获异常后按原召回分排序（不中断）",
     "where": "w9/retrieve_v2.py: rerank_final"},
    {"id": 7, "fault": "上传文档解析失败",
     "symptom": "加密 PDF / 扫描件无文字层",
     "handling": "返回可读错误并保留会话，不写入任何索引",
     "where": "app.py 上传处理"},
    {"id": 8, "fault": "磁盘写满 / 日志写失败",
     "symptom": "日志或产物写不进去",
     "handling": "日志初始化失败降级为 NullHandler，不影响主链路；"
                 "健康检查暴露磁盘余量",
     "where": "w9/runtime9.py: get_logger / disk_status"},
]


if __name__ == "__main__":
    import json
    print(json.dumps(health_check(), ensure_ascii=False, indent=2)[:2500])
