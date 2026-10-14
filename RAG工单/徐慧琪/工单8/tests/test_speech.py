# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.speech import SPEECH_DEGRADATION_NOTE, browser_support_note, speech_component_html


def test_html_contains_webspeech_api():
    html = speech_component_html()
    assert "webkitSpeechRecognition" in html or "SpeechRecognition" in html
    assert "recognition.start()" in html


def test_html_sets_language():
    assert "zh-CN" in speech_component_html(lang="zh-CN")
    assert "en-US" in speech_component_html(lang="en-US")


def test_html_communicates_result_to_parent():
    """识别结果需通过 Streamlit 可读的方式回传（查询参数）。"""
    html = speech_component_html()
    assert "URLSearchParams" in html or "parent" in html


def test_html_handles_unsupported_browser():
    html = speech_component_html()
    assert "不支持" in html or "not supported" in html.lower()


def test_degradation_note_mentions_limitation():
    assert "Web Speech API" in SPEECH_DEGRADATION_NOTE
    assert "降级" in SPEECH_DEGRADATION_NOTE


def test_browser_support_note_is_str():
    assert isinstance(browser_support_note(), str)
    assert browser_support_note()
