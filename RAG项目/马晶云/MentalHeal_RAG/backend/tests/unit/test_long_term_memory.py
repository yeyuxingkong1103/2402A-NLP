from pydantic import ValidationError

from app.schemas.memories import LongTermMemoryCreateRequest
from app.security.safety import is_crisis_message


def test_memory_requires_explicit_confirmation_value() -> None:
    request = LongTermMemoryCreateRequest(content="希望回答简洁", confirmed=False)
    assert request.confirmed is False


def test_memory_rejects_blank_content_after_stripping() -> None:
    try:
        LongTermMemoryCreateRequest(content="   ", confirmed=True)
    except ValidationError:
        return
    raise AssertionError("blank memory content must be rejected")


def test_memory_strips_user_entered_values() -> None:
    request = LongTermMemoryCreateRequest(
        content="  希望回答简洁  ",
        memory_type="  preference  ",
        confirmed=True,
    )
    assert request.content == "希望回答简洁"
    assert request.memory_type == "preference"


def test_crisis_detection_remains_separate_from_memory_creation() -> None:
    assert is_crisis_message("我最近不想活了")
    assert not is_crisis_message("我希望回答简洁一些")
