# -*- coding: utf-8 -*-
"""
DocumentQualityAssessmentSkill · 文档质量评估技能
（工单 18：实现文档质量评估 Skill 并集成至智能体工作流）

五大功能：格式分布统计 / PDF 文字型·扫描型分流 / 文档长度分布 / MD5+SimHash 重复检测 / 敏感信息检测
配套：文档分类标签体系 + 工作流路由 + API 端点。
"""
from .config import Config
from .skill import assess
from . import report

__all__ = ["Config", "assess", "report"]
__version__ = "1.0.0"
