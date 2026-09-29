# -*- coding: utf-8 -*-
"""验证 render_system 与 build_messages 使用同一套渲染逻辑。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import persona  # noqa: E402


class _FakeCharacter:
    """最小 Character 替身，只带渲染需要的字段。"""
    name = "测试角色"
    prompt_template = "{identity_block}\n【风格】\n{style_block}\n【知识】\n{context}\n【问题】\n{question}"
    identity_block = "你是测试角色。"
    description = "测试范围"
    style_json = {"tone": "简洁", "address": "朋友"}
    domain_constraints = "不编造。"
    kb_collection = "kb_medical"


def test_render_system_returns_str():
    out = persona.render_system(_FakeCharacter(), "问题?", [], [])
    assert isinstance(out, str)
    assert "测试角色" in out
    assert "问题?" in out


def test_build_messages_system_matches_render_system():
    """build_messages 的 system 必须与 render_system 完全一致。"""
    ch, q, hits, mem = _FakeCharacter(), "问题?", [], []
    assert persona.build_messages(ch, q, hits, mem)[0]["content"] == \
        persona.render_system(ch, q, hits, mem)


def test_render_system_includes_style():
    out = persona.render_system(_FakeCharacter(), "q", [], [])
    assert "简洁" in out
    assert "朋友" in out
