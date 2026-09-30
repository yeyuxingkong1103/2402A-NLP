from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.deps import get_current_principal
from backend.app.schemas.feedback import FeedbackRequest, FeedbackResponse
from backend.app.services.feedback_service import FeedbackServiceError, submit_feedback

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


def _feedback_to_response(feedback) -> FeedbackResponse:
    # 只返回反馈处理结果，不回显内部告警元数据。
    return FeedbackResponse(**asdict(feedback))


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def create_feedback(body: FeedbackRequest, principal: dict = Depends(get_current_principal)) -> FeedbackResponse:
    # 用户身份来自 access token，不能由请求体指定。
    try:
        feedback = submit_feedback(principal["sub"], body.message_id, body.rating, body.reason, body.category)
    except FeedbackServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _feedback_to_response(feedback)
