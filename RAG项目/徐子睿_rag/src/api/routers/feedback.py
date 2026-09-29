"""src/api/routers/feedback.py —— 用户反馈路由。

在链路中的位置：
    HTTP → 【本文件】 → src/models/database.py（feedback 表）

路由前缀 /api/v1/feedback：
    POST /feedback  对某条回答（或某次会话）提交评分与评论

定位：
    这是整条链路上最简单的接口，但它是"评测闭环"的数据入口 ——
    用户的真实评价比离线评测集更能反映线上质量，
    后续想"用反馈数据做检索/提示词优化"时，数据从这里来。
    当前实现只负责收集，不做自动化的分析与回灌（那是更大的一步）。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import current_user
from src.models.database import Feedback, User, db_session

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """反馈请求体。

    字段：
        session_id: 关联会话，可空（只对整体体验评分时不传）
        message_id: 关联到具体那条回答，可空
        value:      评分数值（必填）。用 float 而非枚举，是为了同时支持
                    "点赞=1/点踩=0"和"1-5 分"等多种前端交互，不必改接口
        comment:    文字评论，默认空
    """

    session_id: int | None = None
    message_id: int | None = None
    value: float
    comment: str = ""


@router.post("")
def feedback(request: FeedbackRequest, user: User = Depends(current_user)):
    """记录一条用户反馈。

    参数：
        request: FeedbackRequest
        user: 当前用户
    返回：
        {"ok": True, "feedback_id": 新记录的 id}

    只做写入、不做去重：
        同一个用户对同一条回答反复提交，会留下多条记录。
        这是有意的 —— 用户改主意（先点赞后取消）本身也是有用信息，
        而且反馈表是追加型的日志，不做覆盖更简单也更真实。

    user_id 取自登录身份而不是请求体：
        否则用户能伪造别人的反馈记录，数据就不可信了。

    db.refresh(item) 是为了拿到自增主键：
        commit 前 item.id 是 None，不 refresh 就返回不了 feedback_id。

    本接口不校验 session_id / message_id 是否真实存在、是否属于当前用户：
        当前实现的宽松点。后果是可能存进指向不存在对象的悬空记录。
        对"收集反馈"这个用途影响很小（分析时按 user_id 聚合即可），
        但如果将来要按反馈回溯到具体回答，这里就必须补上归属校验。
    """
    with db_session() as db:
        item = Feedback(user_id=user.id, session_id=request.session_id, message_id=request.message_id, value=request.value, comment=request.comment)
        db.add(item)
        db.commit()
        db.refresh(item)
        return {"ok": True, "feedback_id": item.id}
