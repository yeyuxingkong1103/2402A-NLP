# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：DocumentQualityAssessmentSkill 包入口，对外暴露技能元数据与调用接口。

from .assessment import run_assessment, SKILL_NAME, SKILL_VERSION   # 主入口
from .config_loader import load_config                              # 配置加载

SKILL_META = {                                                      # 元数据层
    "name": SKILL_NAME,                                             # 技能名称
    "version": SKILL_VERSION,                                       # 版本
    "description": "知识库文档质量评估：格式分布统计、PDF 文字型/扫描型分流、"
                   "文档长度分布、MD5+SimHash 重复检测、敏感信息检测与文档分类标签。",
    "entry_point": "document_quality_assessment.run_assessment",    # 调用入口
    "inputs": "文件夹路径、文件列表或单个文件路径",                  # 输入
    "outputs": "结构化 JSON 报告 + HTML 简报（含待确认/待审核列表）",  # 输出
}

__all__ = ["run_assessment", "load_config", "SKILL_META"]           # 导出符号
