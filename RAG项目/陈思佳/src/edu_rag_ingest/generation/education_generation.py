from __future__ import annotations

"""教育内容生成服务，负责基于检索资料生成教案和试题。"""

from dataclasses import dataclass
from typing import Any

from ..config.config import AppConfig
from .llm import DashScopeChatClient
from ..retrieval.retriever import MilvusRetriever, build_citations, build_teacher_prompt


@dataclass(frozen=True)
class GenerationResult:
    content: str
    citations: list[dict[str, Any]]


class EducationGenerationService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.retriever = MilvusRetriever(config)
        self.llm = DashScopeChatClient(config.qa)

    def generate_lesson_plan(
        self,
        subject: str,
        grade: str,
        chapter: str,
        lesson_hours: int,
        requirements: str,
    ) -> GenerationResult:
        query = f"{subject} {grade} {chapter} 教学目标 教学重点 教学难点 教学过程"
        filters = self._filters(subject, grade, chapter)
        chunks = self.retriever.search(query, filters=filters, top_k=8)
        prompt = self._lesson_plan_prompt(subject, grade, chapter, lesson_hours, requirements, chunks)
        return GenerationResult(self.llm.generate(prompt), build_citations(chunks))

    def generate_questions(
        self,
        subject: str,
        grade: str,
        chapter: str,
        knowledge_point: str,
        question_type: str,
        question_count: int,
        difficulty: str,
    ) -> GenerationResult:
        query = f"{subject} {grade} {chapter} {knowledge_point} {question_type} 题目 答案 解析"
        filters = self._filters(subject, grade, chapter)
        chunks = self.retriever.search(query, filters=filters, top_k=8)
        prompt = self._questions_prompt(
            subject,
            grade,
            chapter,
            knowledge_point,
            question_type,
            question_count,
            difficulty,
            chunks,
        )
        return GenerationResult(self.llm.generate(prompt), build_citations(chunks))

    @staticmethod
    def _filters(subject: str, grade: str, chapter: str) -> dict[str, str]:
        return {
            key: value
            for key, value in {"subject": subject, "grade": grade, "chapter": chapter}.items()
            if value.strip()
        }

    def _lesson_plan_prompt(
        self,
        subject: str,
        grade: str,
        chapter: str,
        lesson_hours: int,
        requirements: str,
        chunks: list[Any],
    ) -> str:
        context = build_teacher_prompt(
            f"为{grade}{subject}{chapter}设计教学方案",
            chunks,
            self.config.qa.max_context_chars,
        )
        return f"""你是严谨的一线教师备课助手。请基于提供的知识库资料，为教师生成可执行的教学设计。
不得编造课程标准或教材事实；资料不足时明确标注需要教师补充的内容。

【教学信息】
学科：{subject}
年级：{grade}
章节：{chapter}
课时：{lesson_hours}
教师要求：{requirements or '无'}

【检索资料】
{context}

【输出格式】
1. 教学目标
2. 教学重点
3. 教学难点
4. 教学准备
5. 教学过程（按课时和环节展开）
6. 课堂活动
7. 巩固练习
8. 课堂小结
9. 作业设计
10. 板书设计
11. 教学反思建议

请输出完整教学设计，并在需要处注明资料依据。"""

    def _questions_prompt(
        self,
        subject: str,
        grade: str,
        chapter: str,
        knowledge_point: str,
        question_type: str,
        question_count: int,
        difficulty: str,
        chunks: list[Any],
    ) -> str:
        context = build_teacher_prompt(
            f"为{grade}{subject}{chapter}生成{knowledge_point}{question_type}练习题",
            chunks,
            self.config.qa.max_context_chars,
        )
        return f"""你是严谨的教师命题助手。请严格依据检索资料生成练习题，不要超出指定年级和知识点范围。
每道题必须有明确答案；无法从资料确认的内容不要编造。

【出题要求】
学科：{subject}
年级：{grade}
章节：{chapter}
知识点：{knowledge_point}
题型：{question_type}
题目数量：{question_count}
难度：{difficulty}

【检索资料】
{context}

【输出格式】
请按题号输出。每道题包含：题目、选项（如适用）、答案、解析、对应知识点、难度。
最后列出本组题目的资料依据。"""
