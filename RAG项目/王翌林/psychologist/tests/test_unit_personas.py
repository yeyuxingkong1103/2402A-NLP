"""单元测试：三个心理医生角色的种子数据与数据库记录一一对应。"""
import pytest
from sqlalchemy import select

from src.models import CounselorPersona
from src.rag import prompt as prompt_utils
from src.services.persona_seed import KNOWLEDGE_DIRS, PERSONAS, SYS_ROLES

EXPECTED_CODES = {"humanistic_lin", "cbt_chen", "mindfulness_zhou"}
EXPECTED_IDS = {"humanistic_lin": 1, "cbt_chen": 2, "mindfulness_zhou": 3}


# ---------------- 种子数据 ----------------
def test_seed_has_three_personas():
    assert len(PERSONAS) == 3
    assert {p["persona_code"] for p in PERSONAS} == EXPECTED_CODES


def test_seed_greetings_and_prompts_are_distinct():
    greetings = [p["greeting"] for p in PERSONAS]
    prompts = [p["system_prompt"] for p in PERSONAS]
    assert len(set(greetings)) == 3
    assert len(set(prompts)) == 3
    assert all(g.strip() for g in greetings)


@pytest.mark.parametrize("persona", PERSONAS, ids=[p["persona_code"] for p in PERSONAS])
def test_seed_persona_fields(persona):
    assert persona["name"]
    assert persona["knowledge_scope"], "每个角色必须有独立的知识库范围"
    params = persona["model_params"]
    assert isinstance(params, dict)
    assert 0 <= params["temperature"] <= 1
    assert params["max_tokens"] > 0
    assert persona["status"] == 1
    assert persona["safety_boundary"]
    # 独立 system prompt 必须包含角色名与安全边界
    assert persona["name"] in persona["system_prompt"]
    assert "12356" in persona["system_prompt"]


def test_seed_prompts_match_prompt_module():
    for persona in PERSONAS:
        assert persona["system_prompt"] == prompt_utils.get_default_system_prompt(
            persona["persona_code"]
        )
        assert prompt_utils.get_default_system_prompt(persona["persona_code"]) in (
            prompt_utils.HUMANISTIC_SYSTEM_PROMPT,
            prompt_utils.CBT_SYSTEM_PROMPT,
            prompt_utils.MINDFULNESS_SYSTEM_PROMPT,
        )


def test_knowledge_dirs_cover_all_personas():
    assert set(KNOWLEDGE_DIRS) == EXPECTED_CODES
    for code, dirs in KNOWLEDGE_DIRS.items():
        assert dirs, f"{code} 未配置知识库目录"
        assert all(isinstance(d, str) and d for d in dirs)


def test_sys_roles():
    codes = {r["role_code"] for r in SYS_ROLES}
    assert codes == {"admin", "user"}


# ---------------- 数据库记录 ----------------
@pytest.mark.parametrize("code,expected_id", list(EXPECTED_IDS.items()))
def test_db_persona_matches_seed(db_session, code, expected_id):
    persona = db_session.execute(
        select(CounselorPersona).where(CounselorPersona.persona_code == code)
    ).scalars().first()
    assert persona is not None, f"数据库中缺少角色：{code}"
    assert persona.id == expected_id
    assert persona.status == 1

    seed = next(p for p in PERSONAS if p["persona_code"] == code)
    assert persona.name == seed["name"]
    assert persona.greeting == seed["greeting"]
    assert persona.system_prompt == seed["system_prompt"]
    assert persona.knowledge_scope == seed["knowledge_scope"]
    assert persona.model_params == seed["model_params"]
    assert persona.therapy_type == seed["therapy_type"]


def test_db_personas_exactly_three(db_session):
    rows = db_session.execute(
        select(CounselorPersona).where(CounselorPersona.persona_code.in_(EXPECTED_CODES))
    ).scalars().all()
    assert len(rows) == 3
    assert len({r.system_prompt for r in rows}) == 3
    assert len({r.greeting for r in rows}) == 3
    assert all(r.knowledge_scope for r in rows)
    assert all(r.model_params for r in rows)