# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本模块职责：用户反馈（点赞/点踩）的收集与统计 + 响应时间指标（P50/P95/3 秒达标率）。
# 以 APIRouter 形式挂载，避免 app/main.py 超出「每个 .py 不超过 300 行」的约束。

import json                           # 反馈记录序列化
import time                           # 时间戳
from collections import deque         # 定长队列保存耗时样本

from fastapi import APIRouter         # 路由分组
from pydantic import BaseModel, Field  # 请求体校验

from . import config as C             # 全局配置
from . import llm as L                # 缓存（Redis）能力，自带容错

router = APIRouter()                  # 反馈与指标路由
# 口径（已定，不再讨论）：验收指标是「首字延迟」，即用户提问到 UI 渲染首字的时间。
# 完整答案时长单独统计、如实展示，但不纳入 3s 达标率 —— 理由是实测两者只差 0.37s，
# 用户看到首字后不到半秒就读完了，拿完整答案时长卡验收属于把工程指标当产品指标。
# 详见 docs/测试报告.md §5.3 与 docs/技术文档.md §2.1。
_LATENCY = deque(maxlen=500)          # 最近 500 次「首字延迟」样本（秒）—— 验收口径
_TOTAL = deque(maxlen=500)            # 最近 500 次「完整答案时长」样本（秒）—— 仅统计，不参与验收


def record(first_token: float, total: float = None) -> None:   # 记录一次问答的两个耗时
    """由问答接口调用。first_token 计入验收指标；total 只统计。

    非流式接口（/api/chat）没有"首字"概念，答案一次性返回，调用方把 total 同时传作 first_token。
    """
    _LATENCY.append(float(first_token))                        # 首字延迟，验收样本
    _TOTAL.append(float(total if total is not None else first_token))   # 完整答案时长，参考样本


def _pct(xs: list[float], p: float) -> float:   # 计算分位数
    if not xs:                                  # 空样本
        return 0.0                              # 返回 0，避免除零
    s = sorted(xs)                              # 升序排序
    return round(s[min(len(s) - 1, int(len(s) * p))], 3)   # 取最近秩，保留 3 位小数


class FeedbackIn(BaseModel):                      # 反馈请求体
    """点赞/点踩入参。"""
    question: str = Field(..., max_length=512)    # 原问题
    answer: str = Field("", max_length=4000)      # 被评价的回答
    vote: str = Field(..., pattern="^(up|down)$")  # 只允许 up / down
    comment: str = Field("", max_length=1000)     # 可选补充说明


@router.post("/api/feedback")                     # 提交反馈
def feedback(body: FeedbackIn):                   # 请求体
    """记录用户点赞/点踩；Redis 不可用时落本地 jsonl，保证不丢（容错要求）。"""
    item = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "vote": body.vote,   # 时间与票型
            "question": body.question, "answer": body.answer[:500], "comment": body.comment}   # 内容
    r = L.get_redis()                             # 取 Redis
    if r:                                         # Redis 可用
        try:                                      # 容错
            r.lpush("rag:feedback", json.dumps(item, ensure_ascii=False))   # 入列表
            r.ltrim("rag:feedback", 0, 999)       # 只保留最近 1000 条
        except Exception:                         # 写入失败
            r = None                              # 转本地落盘
    if not r:                                     # Redis 不可用
        with open(C.DATA_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:   # 追加写
            f.write(json.dumps(item, ensure_ascii=False) + "\n")               # 一行一条
    return {"ok": True, **feedback_stats()}       # 返回最新统计


@router.get("/api/feedback/stats")                # 反馈统计
def feedback_stats():                             # 无参数
    """统计点赞/点踩数量与最近若干条记录。"""
    r = L.get_redis()                             # 取 Redis
    items = []                                    # 记录列表
    if r:                                         # Redis 可用
        try:                                      # 容错
            items = [json.loads(x) for x in r.lrange("rag:feedback", 0, 49)]   # 最近 50 条
        except Exception:                         # 读取失败
            items = []                            # 视为空
    if not items:                                 # Redis 无数据
        p = C.DATA_DIR / "feedback.jsonl"         # 本地兜底文件
        if p.exists():                            # 存在
            items = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-50:] if x]   # 末 50 行
    up = sum(1 for i in items if i.get("vote") == "up")       # 点赞数
    down = sum(1 for i in items if i.get("vote") == "down")   # 点踩数
    return {"up": up, "down": down, "total": len(items), "recent": items[:10]}   # 汇总返回


@router.get("/api/metrics")                       # 响应时间指标
def metrics():                                    # 无参数
    """返回最近若干次问答的耗时指标（工单性能验收项）。

    顶层字段（p50/p95/max/under_3s_rate）一律指**首字延迟**，即验收口径；
    完整答案时长放在 total_* 字段里单独给出，不参与达标率计算。
    `basis` 字段把口径显式写进响应体，避免下次复审时对不上数。
    """
    xs = list(_LATENCY)                           # 复制首字样本
    ts = list(_TOTAL)                             # 复制完整答案样本
    ok = sum(1 for x in xs if x <= 3.0)           # 达标次数（按首字口径）
    return {"n": len(xs), "basis": "first_token",  # 样本数与口径声明
            "p50": _pct(xs, 0.5), "p95": _pct(xs, 0.95),                  # 首字分位数
            "max": round(max(xs), 3) if xs else 0.0,                      # 首字最大值
            "under_3s_rate": round(ok / len(xs), 4) if xs else 0.0,       # 3 秒达标率（首字口径）
            "total_p50": _pct(ts, 0.5), "total_p95": _pct(ts, 0.95),      # 完整答案分位数
            "total_max": round(max(ts), 3) if ts else 0.0}                # 完整答案最大值
