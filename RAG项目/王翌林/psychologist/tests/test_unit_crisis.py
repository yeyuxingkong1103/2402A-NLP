"""单元测试：src.services.crisis_service 危机词检测、转介提示、敏感词脱敏、后处理。"""
import pytest

from src.core.config import settings
from src.services import crisis_service


# ---------------- 危机词检测 ----------------
def test_detect_crisis_negative():
    assert crisis_service.detect_crisis("我最近工作压力有点大") == (False, [])
    assert crisis_service.detect_crisis("") == (False, [])
    assert crisis_service.detect_crisis(None) == (False, [])


@pytest.mark.parametrize("text,keyword", [
    ("我觉得活着没意思", "活着没意思"),
    ("我不想活了", "不想活"),
    ("我想自杀", "自杀"),
    ("有时候真想一了百了，结束生命", "结束生命"),
    ("我控制不住想割腕", "割腕"),
    ("我想杀了他", "杀人"),  # 命中“杀了/杀人”类高危词表
    ("我想结束这一切", "结束这一切"),
    ("我想离开这个世界", "离开这个世界"),
])
def test_detect_crisis_keywords_hit(text, keyword):
    hit, keywords = crisis_service.detect_crisis(text)
    assert hit is True
    assert keywords, "命中危机时需返回命中词"
    assert any(k in text for k in keywords)


def test_detect_crisis_risk_pattern_without_keyword():
    hit, keywords = crisis_service.detect_crisis("我真的撑不下去了")
    assert hit is True
    assert keywords


def test_detect_crisis_multiple_keywords():
    hit, keywords = crisis_service.detect_crisis("我想死，也想过自杀")
    assert hit is True
    assert len(keywords) >= 2


# ---------------- 危机提示文案 ----------------
def test_crisis_notice_contains_help_lines():
    notice = crisis_service.crisis_notice()
    assert settings.crisis_hotline in notice      # 12356
    assert "120" in notice
    assert "110" in notice
    assert "请优先关注安全" in notice


def test_ensure_crisis_notice_appends_when_missing():
    result = crisis_service.ensure_crisis_notice("我理解你的痛苦。")
    assert crisis_service.crisis_notice().strip() in result


@pytest.mark.parametrize("answer", ["请拨打 12356", "如果需要请拨打 120", "可以报警 110"])
def test_ensure_crisis_notice_keeps_existing(answer):
    assert crisis_service.ensure_crisis_notice(answer) == answer


# ---------------- 敏感词脱敏 ----------------
def test_mask_sensitive_words():
    masked = crisis_service.mask_sensitive_words("有人在群里发赌博和刷单广告")
    assert "赌博" not in masked and "刷单" not in masked
    assert "****" in masked or "**" in masked


def test_mask_sensitive_words_custom_replacement():
    assert crisis_service.mask_sensitive_words("色情内容", replacement="#") == "##内容"


def test_mask_sensitive_words_no_hit_keeps_text():
    text = "我们聊聊你的情绪"
    assert crisis_service.mask_sensitive_words(text) == text


def test_sensitive_words_are_masked_in_post_process():
    assert "赌博" not in crisis_service.post_process("他沉迷赌博")


# ---------------- post_process ----------------
def test_post_process_empty():
    assert crisis_service.post_process("") == ""


def test_post_process_normalizes_whitespace():
    cleaned = crisis_service.post_process("第一段\n\n\n\n第二段   结束")
    assert "\n\n\n" not in cleaned
    assert "   " not in cleaned
    assert cleaned.startswith("第一段")
    assert cleaned.endswith("第二段 结束")


def test_post_process_blocks_diagnosis_claim():
    cleaned = crisis_service.post_process("作为一个心理医生，我可以为你诊断，你是抑郁症。")
    assert "我可以为你诊断" not in cleaned
    assert "我不能进行诊断或开药" in cleaned


def test_post_process_without_crisis_does_not_add_notice():
    cleaned = crisis_service.post_process("我听到你的疲惫。", crisis=False)
    assert settings.crisis_hotline not in cleaned


def test_post_process_with_crisis_adds_notice():
    cleaned = crisis_service.post_process("我听到你的痛苦。", crisis=True)
    assert settings.crisis_hotline in cleaned
    assert "120" in cleaned and "110" in cleaned


def test_post_process_with_crisis_does_not_duplicate_notice():
    answer = f"请立即拨打 {settings.crisis_hotline} 求助。"
    assert crisis_service.post_process(answer, crisis=True) == answer