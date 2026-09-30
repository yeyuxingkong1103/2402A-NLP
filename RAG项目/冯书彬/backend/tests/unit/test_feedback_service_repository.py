import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.api.v1.chat import chat_service
from backend.app.models.message import Message
from backend.app.repositories.feedback_repository import SQLAlchemyFeedbackRepository, create_feedback_tables
from backend.app.services import feedback_service
from backend.app.services.feedback_service import delete_feedback_data_for_user, get_alert, get_feedback, submit_feedback


@pytest.fixture()
def feedback_repository_env():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    create_feedback_tables(engine)
    repository = SQLAlchemyFeedbackRepository(engine)
    feedback_service.set_feedback_repository(repository)
    chat_service.reset_for_tests()
    yield repository
    feedback_service.reset_feedback_store_for_tests()
    chat_service.reset_for_tests()


def test_feedback_service_persists_high_risk_alert(feedback_repository_env):
    chat_service.messages["msg-1"] = Message(id="msg-1", conversation_id="c1", user_id="user-1", user_text="问题", answer="回答", status="answered")

    feedback = submit_feedback("user-1", "msg-1", "down", "手机号13800138000泄露", "privacy_leak")

    restored_feedback = feedback_repository_env.get_feedback(feedback.id)
    restored_alert = feedback_repository_env.get_alert(feedback.alert_id)
    assert restored_feedback.reason == "手机号[手机号]泄露"
    assert restored_feedback.alert_id == feedback.alert_id
    assert restored_alert.notification_sent is True
    assert get_feedback(feedback.id) == restored_feedback
    assert get_alert(feedback.alert_id) == restored_alert


def test_feedback_service_repository_delete_user_data(feedback_repository_env):
    chat_service.messages["msg-1"] = Message(id="msg-1", conversation_id="c1", user_id="user-1", user_text="问题", answer="回答", status="answered")
    feedback = submit_feedback("user-1", "msg-1", "down", "引用错误", "wrong_legal_basis")

    deleted = delete_feedback_data_for_user("user-1")

    assert deleted == {"feedback": 1, "alerts": 1}
    assert feedback_repository_env.get_feedback(feedback.id) is None
    assert feedback_repository_env.get_alert(feedback.alert_id) is None
