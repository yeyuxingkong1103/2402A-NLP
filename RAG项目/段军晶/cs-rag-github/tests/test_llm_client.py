# -*- coding: utf-8 -*-
"""LLM 客户端空响应处理测试

背景：deepseek-v4-flash 是推理模型，先输出 reasoning_content 再输出 content。
当推理吃光 max_tokens 预算时 finish_reason 为 "length" 且 content 为空，
旧实现会静默返回空字符串，导致用户拿到空答案且无任何报错。
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from backend.llm_client import ChatLLM, LLMError


class _FakeCompletions:
    def __init__(self, response: Any) -> None:
        self._response = response

    def create(self, **kwargs: Any) -> Any:
        return self._response


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def _make_llm(content: str, finish_reason: str, reasoning: str = "") -> ChatLLM:
    """构造一个把 OpenAI 客户端替换为假对象的 ChatLLM"""
    llm = ChatLLM()
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )
    llm._client = _FakeClient(response)          # 绕过真实网络调用
    return llm


def test_正常响应原样返回():
    llm = _make_llm("这是答案", "stop")
    assert llm.chat([{"role": "user", "content": "问"}]) == "这是答案"


def test_截断且内容为空时抛出_LLMError():
    """推理吃光预算 -> finish_reason=length 且 content 为空 -> 必须报错"""
    llm = _make_llm("", "length", reasoning="模型的长篇推理过程……")
    with pytest.raises(LLMError) as exc:
        llm.chat([{"role": "user", "content": "问"}])
    assert "长度" in str(exc.value) or "截断" in str(exc.value)


def test_正常结束但内容为空也抛出_LLMError():
    llm = _make_llm("", "stop")
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "问"}])


def test_不向调用方返回_reasoning_content():
    """按 spec 要求：推理内容只进日志，绝不作为答案返回给用户"""
    llm = _make_llm("", "length", reasoning="应当只出现在日志里")
    with pytest.raises(LLMError) as exc:
        llm.chat([{"role": "user", "content": "问"}])
    assert "应当只出现在日志里" not in str(exc.value)


def test_仅含空白字符的响应同样视为空():
    llm = _make_llm("   \n  ", "stop")
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "问"}])
