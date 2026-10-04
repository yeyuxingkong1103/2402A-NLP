"""单元测试：分块、清洗、后处理、配置、角色预设（不依赖外部服务）。"""
from app.config import settings
from app.ingest.chunker import chunk_parent_child, chunk_text, split_sentences
from app.ingest.clean import clean_chunks, is_low_quality, normalize, remove_watermarks
from app.rag.postprocess import extract_citations, postprocess
from app.roles.presets import ROLE_PRESETS, domain_of, get_preset

SAMPLE = "第一条 为了加强保护。第二条 禁止猎捕。第三条 违者处罚。第四条 附则说明。"


def test_split_sentences():
    sents = split_sentences(SAMPLE)
    assert len(sents) >= 3


def test_chunk_sentence():
    parts = chunk_text(SAMPLE, strategy="sentence", size=30, overlap=0)
    assert parts and all(p["chunk_type"] == "sentence" for p in parts)


def test_chunk_fixed_overlap():
    parts = chunk_text("abcdefghij" * 10, strategy="fixed", size=20, overlap=5)
    assert len(parts) > 1


def test_chunk_parent_child():
    parts = chunk_parent_child(SAMPLE, strategy="sentence", size=20)
    assert parts and all("parent" in p for p in parts)


def test_normalize_removes_cjk_spaces():
    assert normalize("人 民 法 院") == "人民法院"


def test_remove_watermarks():
    text = "正文内容\n第 1 页\nhttp://example.com\n结尾"
    cleaned = remove_watermarks(text)
    assert "第 1 页" not in cleaned
    assert "example.com" not in cleaned


def test_is_low_quality():
    assert is_low_quality("短")
    assert not is_low_quality("这是一段足够长的正常中文内容用于通过质量检测。")


def test_clean_chunks_dedup():
    chunks = [{"text": "重复内容测试重复内容测试"}, {"text": "重复内容测试重复内容测试"}]
    assert len(clean_chunks(chunks)) == 1


def test_extract_citations():
    text = "依据《野生动物保护法》第四十八条，应予处罚。"
    assert extract_citations(text) == ["《野生动物保护法》第四十八条"]


def test_postprocess_verify_flag():
    reply = "依据《A法》第一条。"
    out = postprocess(reply, context="无关上下文")
    assert "未在知识库原文中检索到" in out


def test_role_presets_integrity():
    ids = [r["id"] for r in ROLE_PRESETS]
    assert len(ids) == len(set(ids)) >= 10
    assert domain_of(1) == "law"
    assert get_preset(999) is None


def test_settings_defaults():
    assert settings.embed_dim == 1024
    assert settings.rag_engine in ("native", "langchain", "llamaindex")
