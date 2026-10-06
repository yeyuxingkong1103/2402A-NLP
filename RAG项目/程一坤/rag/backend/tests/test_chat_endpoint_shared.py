"""chat_endpoint 去重的锁定测试。

背景：`llm.py` 与 `qwen_vl.py` 的 `chat_endpoint` 属性原本是逐字相同的
3 行实现（纯拼接），清理时抽成 `http_retry.chat_completions_endpoint`
共用。这里的测试锁定两件事：
1. 两处行为逐字不变（输入带不带结尾斜杠、输出格式）；
2. 两处确实委托共用实现（patch 掉共用函数后行为跟着变，防止将来
   有人又抄一份本地实现回来）。
"""

from __future__ import annotations

import pytest

from app.models.http_retry import chat_completions_endpoint
from app.models.llm import OpenAiCompatibleChatClient
from app.models.qwen_vl import QwenVlClient

# 两类客户端的最小构造参数（不联网：不发起任何请求，只看属性）
_LLM_KWARGS = {"api_base_url": "http://example.com", "api_key": "k", "model": "m"}
_VL_KWARGS = {"api_base_url": "http://example.com", "api_key": "k"}


@pytest.mark.parametrize("base", ["http://example.com", "http://example.com/", "http://example.com//"])
def test_endpoint_identical_between_llm_and_qwen_vl(base: str) -> None:
    """同一 base_url 下，两个客户端的对话接口地址必须完全一致。"""
    llm = OpenAiCompatibleChatClient(api_base_url=base, api_key="k", model="m")
    vl = QwenVlClient(api_base_url=base, api_key="k")
    assert llm.chat_endpoint == vl.chat_endpoint


@pytest.mark.parametrize("base", ["http://example.com", "http://example.com/"])
def test_endpoint_behavior_unchanged(base: str) -> None:
    """行为逐字不变：始终是 `<base 去结尾斜杠>/chat/completions`。"""
    assert chat_completions_endpoint(base) == "http://example.com/chat/completions"
    llm = OpenAiCompatibleChatClient(api_base_url=base, api_key="k", model="m")
    vl = QwenVlClient(api_base_url=base, api_key="k")
    assert llm.chat_endpoint == "http://example.com/chat/completions"
    assert vl.chat_endpoint == "http://example.com/chat/completions"


def test_both_clients_delegate_to_shared_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """锁死"共用"这一事实：patch 共用实现后，两处属性都必须跟着变。"""
    import app.models.llm as llm_module
    import app.models.qwen_vl as vl_module

    marker = "http://patched.example/marker"
    monkeypatch.setattr(llm_module, "chat_completions_endpoint", lambda _base: marker)
    monkeypatch.setattr(vl_module, "chat_completions_endpoint", lambda _base: marker)

    llm = OpenAiCompatibleChatClient(**_LLM_KWARGS)  # type: ignore[arg-type]
    vl = QwenVlClient(**_VL_KWARGS)  # type: ignore[arg-type]
    assert llm.chat_endpoint == marker
    assert vl.chat_endpoint == marker
