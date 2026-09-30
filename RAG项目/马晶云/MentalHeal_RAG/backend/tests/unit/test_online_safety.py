from app.security.safety import is_crisis_message


def test_detects_chinese_self_harm_signal() -> None:
    assert is_crisis_message("我最近真的不想活了")


def test_detects_spaced_crisis_signal() -> None:
    assert is_crisis_message("我想 自 杀")


def test_does_not_flag_general_wellbeing_question() -> None:
    assert not is_crisis_message("最近睡眠不好，怎么调整？")
