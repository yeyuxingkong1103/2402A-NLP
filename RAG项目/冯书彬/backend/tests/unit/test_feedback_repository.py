from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.models.feedback import Feedback, FeedbackAlert
from backend.app.repositories.feedback_repository import SQLAlchemyFeedbackRepository, create_feedback_tables


def _repository() -> SQLAlchemyFeedbackRepository:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    create_feedback_tables(engine)
    return SQLAlchemyFeedbackRepository(engine)


def test_feedback_repository_round_trips_feedback_and_alert():
    repository = _repository()
    now = datetime.now(timezone.utc)
    feedback = Feedback(
        id="fb-1",
        user_id="user-1",
        message_id="msg-1",
        rating="down",
        category="wrong_legal_basis",
        reason="引用错误",
        risk_level="high",
        created_at=now,
    )
    alert = FeedbackAlert(
        id="alert-1",
        feedback_id=feedback.id,
        risk_level="high",
        category="wrong_legal_basis",
        metadata={"message_id": "msg-1"},
        notification_sent=True,
        created_at=now,
    )
    feedback.alert_id = alert.id

    repository.save_feedback(feedback)
    repository.save_alert(alert)
    repository.save_feedback(feedback)

    assert repository.get_feedback("fb-1") == feedback
    assert repository.get_alert("alert-1") == alert


def test_feedback_repository_deletes_user_feedback_and_alerts_only():
    repository = _repository()
    now = datetime.now(timezone.utc)
    owner = Feedback(id="fb-1", user_id="user-1", message_id="msg-1", rating="down", risk_level="high", created_at=now, alert_id="alert-1")
    other = Feedback(id="fb-2", user_id="user-2", message_id="msg-2", rating="up", risk_level="normal", created_at=now)
    alert = FeedbackAlert(id="alert-1", feedback_id="fb-1", risk_level="high", category="other", metadata={}, notification_sent=True, created_at=now)

    repository.save_feedback(owner)
    repository.save_feedback(other)
    repository.save_alert(alert)

    deleted = repository.delete_user_data("user-1")

    assert deleted == {"feedback": 1, "alerts": 1}
    assert repository.get_feedback("fb-1") is None
    assert repository.get_alert("alert-1") is None
    assert repository.get_feedback("fb-2") == other
