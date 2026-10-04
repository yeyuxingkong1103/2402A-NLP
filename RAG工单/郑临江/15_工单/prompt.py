# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程工单
提示工程：当检索上下文含图像信息时，明确指示模型结合图纸描述分析。
"""
from query_understanding import is_visual_query


def build_prompt(query, contexts):
    """构造发送给 LLM 的上下文 Prompt 模板。"""
    parts = ["请根据以下上下文回答问题。"]
    if is_visual_query(query) or any(c.get("is_image") for c in contexts):
        parts.append("请结合以下技术图纸的描述进行分析，注意部件编号与位置关系。")
    parts.append("\n".join(
        (c.get("image_desc") + "\n" if c.get("image_desc") else "") + c.get("text", "")
        for c in contexts
    ))
    parts.append(f"\n问题：{query}")
    return "\n".join(parts)
