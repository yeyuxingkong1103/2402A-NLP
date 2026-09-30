"""角色注册表与护栏测试。"""

from __future__ import annotations

import pytest

from role_rag.errors import NotFoundError
from role_rag.roles import Role, RoleRegistry


@pytest.fixture(scope="module")
def registry():
    from role_rag.config import get_config

    return RoleRegistry.from_config(get_config())


def test_all_eleven_roles_defined(registry):
    assert len(registry) == 11
    expected = {
        "customer_service", "social", "npc", "doctor", "psychologist", "lawyer",
        "stock_analyst", "financial_planner", "scientist", "teacher", "english_tutor",
    }
    assert set(role.id for role in registry.all()) == expected


def test_randomly_selected_roles_enabled(registry):
    enabled = registry.enabled_ids()
    assert set(enabled) == {"financial_planner", "scientist", "lawyer"}
    assert len(enabled) == 3


def test_role_kb_scopes_include_shared(registry):
    role = registry.get("lawyer")
    assert role.kb_scopes == ["lawyer", "shared"]
    assert "shared" in registry.get("scientist").kb_scopes


def test_persona_block_contains_guardrails(registry):
    role = registry.get("financial_planner")
    block = role.persona_block()
    assert "金融理财师" in block or "理财规划师" in block
    assert "红线" in block
    assert "不承诺任何收益" in block


def test_disabled_role_rejected(registry):
    assert registry.get("doctor").enabled is False
    with pytest.raises(NotFoundError):
        registry.require_enabled("doctor")


def test_unknown_role_raises(registry):
    with pytest.raises(NotFoundError):
        registry.get("not_exists")


def test_guardrails_append_disclaimer(registry):
    role = registry.get("financial_planner")
    text, hits = registry.apply_guardrails(role, "这只基金稳赚不赔，建议满仓。")
    assert "稳赚" in hits and "满仓" in hits
    assert role.disclaimer.strip() in text
    # 幂等：已包含免责声明时不重复追加
    again, _ = registry.apply_guardrails(role, text)
    assert again.count("⚠️") == 1


def test_guardrails_no_hit(registry):
    role = registry.get("lawyer")
    text, hits = registry.apply_guardrails(role, "诉讼时效为三年。")
    assert hits == []
    assert text == "诉讼时效为三年。"


def test_safety_notice_only_when_question_hits(registry):
    role = registry.get("financial_planner")
    notice = registry.safety_notice(role, "帮我推荐一只稳赚不赔的基金")
    assert "安全纠正" in notice and "稳赚" in notice
    assert registry.safety_notice(role, "基金有哪些类型？") == ""


def test_role_public_dict_has_no_prompt(registry):
    payload = registry.get("scientist").to_public_dict()
    assert set(payload) >= {"id", "name", "avatar", "tagline", "followups"}
    assert "persona" not in payload and "guardrails" not in payload


def test_role_dataclass_defaults():
    role = Role(id="x", name="测试")
    assert role.kb_scopes == ["shared"]
    assert role.to_public_dict()["enabled"] is False
