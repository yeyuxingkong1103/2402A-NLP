# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单

包：document_quality_assessment
知识库文档质量评估 Skill（DocumentQualityAssessmentSkill）。

三层结构对应关系（Claude Skills 规范）：
  * 元数据层 → ``SKILL.md`` 的 YAML frontmatter（name / description / 触发条件）
  * 指令层   → ``SKILL.md`` 正文（工作流程 SOP）
  * 资源层   → 本包内的 Python 模块 + ``研发/assessment_config.yaml`` + ``scripts/``

快速使用::

    from document_quality_assessment import DocumentQualityAssessmentSkill
    skill = DocumentQualityAssessmentSkill()
    result = skill.run(folder=r"E:\\gao6gongdan\\_共享输入\\original_problems\\documents", resume=True)
    print(result.to_summary())
"""
from __future__ import annotations

from .config import Config, ConfigError, REPO_ROOT, load_config
from .extractors import ExtractResult, extract
from .skill import (
    DocumentQualityAssessmentSkill,
    FileEntry,
    SKILL_NAME,
    SKILL_VERSION,
    SkillResult,
    collect_files,
    process_one,
)

__all__ = [
    "DocumentQualityAssessmentSkill",
    "SkillResult",
    "Config",
    "ConfigError",
    "load_config",
    "REPO_ROOT",
    "FileEntry",
    "collect_files",
    "process_one",
    "extract",
    "ExtractResult",
    "SKILL_NAME",
    "SKILL_VERSION",
]

__version__ = SKILL_VERSION
