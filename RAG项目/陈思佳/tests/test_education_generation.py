from unittest.mock import Mock

from src.edu_rag_ingest.generation.education_generation import EducationGenerationService


def test_lesson_plan_filters_include_subject_grade_and_chapter():
    service = object.__new__(EducationGenerationService)

    assert service._filters("语文", "九年级", "阅读") == {
        "subject": "语文",
        "grade": "九年级",
        "chapter": "阅读",
    }


def test_lesson_plan_prompt_contains_structured_education_requirements():
    service = object.__new__(EducationGenerationService)
    service.config = Mock()
    service.config.qa.max_context_chars = 2000
    service._lesson_plan_prompt = EducationGenerationService._lesson_plan_prompt.__get__(service)
    service._questions_prompt = EducationGenerationService._questions_prompt.__get__(service)

    prompt = service._lesson_plan_prompt("语文", "九年级", "阅读", 2, "突出合作学习", [])

    assert "教学目标" in prompt
    assert "教学重点" in prompt
    assert "突出合作学习" in prompt
    assert "不得编造课程标准" in prompt


def test_questions_prompt_contains_question_constraints():
    service = object.__new__(EducationGenerationService)
    service.config = Mock()
    service.config.qa.max_context_chars = 2000

    prompt = service._questions_prompt("语文", "九年级", "阅读", "人物形象", "选择题", 5, "中等", [])

    assert "题目数量：5" in prompt
    assert "人物形象" in prompt
    assert "每道题必须有明确答案" in prompt
