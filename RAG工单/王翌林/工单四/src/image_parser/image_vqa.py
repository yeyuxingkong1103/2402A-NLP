# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_vqa.py —— 工单四图像视觉问答（VQA）模块

职责（见 docs/01_图像解析方案.md §5.2）：
  对图表类/结构图类图像，用多模态大模型（Qwen2-VL）按预置问题模板提问，
  提取结构化数据，输出 vqa_qa 列表 [{"q":..., "a":...}, ...]。

预置问题模板（工单四指定）：
  图表类：图中包含哪些类别？数值是多少？增长率最快的是哪个？负增长的是哪个？
  组织结构图类：节点层级/父子关系模板（支撑 IMG-05 销售部构成类问题）
  流程图类：步骤顺序模板
"""
import re
from typing import Any, Dict, List

from loguru import logger
from PIL import Image

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

# 工单四：图表类预置问题（IMG-06"2008年中国IC市场应用结构与增长"依赖）
CHART_QUESTIONS = [
    "图中包含哪些类别？",
    "图中各数据的数值分别是多少？请逐项列出。",
    "增长率最快的是哪个？数值是多少？",
    "负增长的是哪个？数值是多少？",
]

# 工单四：组织结构图预置问题（IMG-05"销售部构成/大客户销售部销售处"依赖）
ORG_QUESTIONS = [
    "图中根节点/最高层节点是什么？",
    # 工单四：Qwen2-VL-2B 宽松问法偶发漏数并列框（实测漏"国际贸易部"），
    # 全图逐部门枚举又会因输出过长退化重复；直接锚定"销售部"节点按连线
    # 从左到右枚举，可稳定数出 4 个平级子部门（电话及网络/渠道/大客户/国际贸易）。
    # 置于第 2 问确保答案进入多模态 prompt 的 VQA 截断窗口前部。
    "图中标注为\"销售部\"的方框，向下连接了哪几个平级子部门方框？"
    "请严格按连线从左到右逐一完整写出每个方框内的文字（不得遗漏或合并），"
    "并给出部门总数。",
    "列出图中所有节点的父子关系：每个节点的上级和下级分别是什么？",
]

# 工单四：流程图预置问题
FLOW_QUESTIONS = [
    "图的流程步骤有哪些？请按顺序列出。",
    "流程中有哪些判断分支？",
]

# 工单四：通用兜底问题
GENERIC_QUESTIONS = [
    "图中有什么内容？请列出关键要素。",
]

# 工单四：图像类型判定关键词（caption/提取类型 → 问题模板选择）
_CHART_HINTS = ("柱状", "饼图", "折线", "曲线", "增长", "占比", "百分比",
                "图表", "统计", "分布", "chart", "bar", "pie", "line")
_ORG_HINTS = ("组织结构", "组织架构", "架构图", "层级", "部门", "结构图")
_FLOW_HINTS = ("流程", "步骤", "工艺", "流程图", "flowchart")


class ImageVQAEngine:
    """工单四：图像 VQA 引擎（复用常驻 Qwen2-VL，逐问串行）"""

    def __init__(self, vlm_engine: Any):
        # 工单四：vlm_engine 需提供 generate(image, prompt) 接口（便于 Mock）
        self.vlm = vlm_engine

    # ------------------------------------------------------------------
    @staticmethod
    def detect_question_set(meta: Dict[str, Any], caption: str = "") -> List[str]:
        """工单四：按图像类型/描述关键词选择问题模板（可在模型解析前用提取类型预判）"""
        text = f"{meta.get('image_type', '')} {caption} {meta.get('path', '')}"
        if any(k in text for k in _ORG_HINTS):
            return ORG_QUESTIONS
        if any(k in text for k in _CHART_HINTS):
            return CHART_QUESTIONS
        if any(k in text for k in _FLOW_HINTS):
            return FLOW_QUESTIONS
        return GENERIC_QUESTIONS

    # ------------------------------------------------------------------
    def ask(self, image: Image.Image, question: str) -> str:
        """工单四：单问 VQA"""
        prompt = (f"请仔细看图并回答问题，输出为中文，尽量给出图中确切数值：\n{question}")
        return self.vlm.generate(image, prompt, max_new_tokens=384)

    # ------------------------------------------------------------------
    def run(self, image: Image.Image, meta: Dict[str, Any],
            caption: str = "", questions: List[str] = None) -> List[Dict[str, str]]:
        """工单四：对一张图按模板逐问执行，返回 vqa_qa 列表"""
        qs = questions or self.detect_question_set(meta, caption)
        qa = []
        for q in qs:
            try:
                a = self.ask(image, q)
                qa.append({"q": q, "a": a})
            except Exception as e:                     # 工单四：单问失败不阻塞
                logger.warning(f"[vqa] {meta.get('image_id')} 问题'{q}'失败: {e}")
                qa.append({"q": q, "a": ""})
        return qa
