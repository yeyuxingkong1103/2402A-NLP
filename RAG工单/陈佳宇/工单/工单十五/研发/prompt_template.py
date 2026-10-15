# -*- coding: utf-8 -*-
# prompt_template.py 工单15 两套Prompt模板：普通文本 / 图文图纸感知模板

# 普通文本问答模板
PROMPT_NORMAL = """
基于下面检索上下文回答用户问题。
上下文：
{context_text}

用户问题：{question}
回答：
"""

# 检测到图纸引用，图文专用Prompt
PROMPT_FIGURE_AWARE = """
下面包含文档检索得到的上下文，其中部分片段来自技术图纸解析结果。
请**优先结合技术图纸描述信息做推理**，重点关注部件编号、部件之间位置关系、装置动作逻辑。
不要编造图纸中不存在的部件与关系。

上下文：
{context_text}

用户问题：{question}
回答：
"""

def select_prompt_template(context_list:list, question:str):
    """自动选择模板：上下文包含图纸标记返回图文感知模板"""
    concat_context = "\n".join([x["content"] for x in context_list])
    if "图" in concat_context or "图示" in concat_context:
        return PROMPT_FIGURE_AWARE
    return PROMPT_NORMAL
