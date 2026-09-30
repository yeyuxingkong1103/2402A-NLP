# 路由的测试只验"选对了模型、缺密钥会炸"，不真发请求——
# 真调用由 Task 13 的 CLI 冒烟覆盖。缺密钥必须抛错而不是降级到本地，
# 这是设计文档 4.5 定的口径：静默换模型会让口径名存实亡且难以排查。
import pytest

from app.generation.llm_router import (
    DEEPSEEK_BASE_URL, DEEPSEEK_KEY_ENV, DEEPSEEK_MODEL, OLLAMA_BASE_URL,
    OLLAMA_MODEL, REQUEST_TIMEOUT_S, MissingAPIKeyError, get_llm,
)
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC


def test_constants_match_agreed_endpoints():
    assert OLLAMA_BASE_URL == "http://127.0.0.1:11434"
    assert OLLAMA_MODEL == "qwen2.5:3b"
    assert DEEPSEEK_BASE_URL == "https://api.deepseek.com/v1"
    assert DEEPSEEK_MODEL == "deepseek-chat"
    assert DEEPSEEK_KEY_ENV == "api"


def test_internal_returns_ollama_client():
    llm = get_llm(SIDE_INTERNAL)
    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == OLLAMA_MODEL
    assert llm.format == "json", "不给 json 模式，3b 会回一大段解释性文字"
    # 超时必须真的落到底层 httpx 客户端：ChatOllama 没有 timeout 字段且
    # extra="ignore"，写成 timeout= 会被静默丢弃、底层退回无限等待，
    # 所以只查 client_kwargs 不够，要查 httpx 自己拿到的值
    assert llm._client._client.timeout.read == REQUEST_TIMEOUT_S, (
        "超时没传到 httpx 客户端；3b 卡死时整条问答链会无限等")


def test_public_returns_deepseek_client(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_KEY_ENV, "sk-test-not-a-real-key")
    llm = get_llm(SIDE_PUBLIC)
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == DEEPSEEK_MODEL
    assert str(llm.openai_api_base).rstrip("/") == DEEPSEEK_BASE_URL
    # 设计文档 4.5 定稿的接线：公众侧靠 response_format 约束成 JSON，
    # 丢了它 DeepSeek 会回自由文本，下游解析直接崩
    assert llm.model_kwargs.get("response_format") == {"type": "json_object"}


def test_public_without_key_raises(monkeypatch):
    monkeypatch.delenv(DEEPSEEK_KEY_ENV, raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_llm(SIDE_PUBLIC)


def test_temperature_is_low_for_both_sides(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_KEY_ENV, "sk-test-not-a-real-key")
    # 技术方案 6.1 定的是 0~0.2：法条引用要忠实，不创作
    assert get_llm(SIDE_INTERNAL).temperature == 0.1
    assert get_llm(SIDE_PUBLIC).temperature == 0.1


def test_unknown_side_raises():
    with pytest.raises(ValueError):
        get_llm("third_side")
