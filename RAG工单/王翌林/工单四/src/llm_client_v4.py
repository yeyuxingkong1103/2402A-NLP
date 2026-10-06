# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/llm_client_v4.py —— 工单四 LLM 客户端（新增文件）

增量原则：chat 复用工单三 llm_client_v3（DeepSeek-v4-flash），本文件只新增：
  1. RAG_V4_SYSTEM 多模态系统提示词；
  2. build_multimodal_prompt：文本 chunk + 表格 + 图像描述/VQA/OCR 融合上下文组装。
"""
from typing import Any, Dict, List

# 工单四：多模态 RAG 系统提示词（图文表三源引用规范）
RAG_V4_SYSTEM = (
    "你是一名专业的投资分析师。请仅依据提供的资料回答问题。"
    "资料包含：文本片段【资料N】、表格【表N】、图像【图N】（含图像描述、"
    "图内文字OCR、图表问答VQA）。回答时在关键结论后标注引用编号（如[图1][表2]）。"
    "若图像VQA/OCR中包含具体数值或组织层级，优先采信。"
    # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：金额表达完整性规范，
    # 例如"募集资金总额40,584.83万元"同时给出约合亿元，便于跨单位核对。
    "金额表述规范：资料以万元为单位且数值达到10,000万元（1亿元）及以上时，"
    "须同时给出约合亿元（1亿元=10,000万元，保留两位小数），"
    "形如\"40,584.83万元（约4.06亿元）\"；中英文提问均遵守此规范，"
    "英文回答可写作\"405.85 million RMB (about 0.41 billion RMB)\"式双单位表述。"
    # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：图内计数交叉核对规范。
    # VLM 描述偶发漏数并列框图元（实测漏列子部门），OCR 保留图内全部文字，
    # 故"有几个/数量/构成"类问题须以 OCR 文字与 VQA 枚举交叉核对：
    # 各来源条目不一致时取去重并集计数，并列举全部名称。
    "图内计数规范：回答组织结构图/流程图等图中\"有几个部门（机构、步骤）\"类问题时，"
    "必须把图像描述、VQA枚举与图内文字OCR中的并列条目交叉核对、去重后取并集计数，"
    "并逐一列出全部名称；不得只依据单一描述源的数量直接作答。"
    # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：时序数据完整性规范。
    # 实测"各年度军用收入"类问题证据已在上下文，但模型偶发只输出占比、漏掉
    # 逐年金额（6464.51/14414.16/18780.67万元），故强制逐年金额与占比并列。
    "时序数据规范：当问题含\"分别是多少、各年度/各期、报告期内\"等词语时，"
    "必须按年度（期间）逐一列出资料中出现的全部数值：金额（保留资料原单位与千分位）"
    "与占比须并列给出，不得只给占比、只给最近一期或做笼统概括；"
    "资料中确无对应数值时才说明未披露。"
)


def build_multimodal_prompt(query: str, text_chunks: List[Dict[str, Any]],
                            table_chunks: List[Dict[str, Any]],
                            image_hits: List[Dict[str, Any]],
                            max_chars: int = 9000) -> str:
    """工单四：三源上下文组装（图像块置顶——VQA 结构化事实对答题最关键）"""
    parts: List[str] = []

    # 1) 工单四：图像块（描述 + VQA + OCR）
    for i, img in enumerate(image_hits, 1):
        vqa = img.get("vqa_text") or ""
        ocr = (img.get("ocr_text") or "")[:600]
        block = [f"【图{i}】doc={img.get('doc_id', '?')} 第{img.get('page', '?')}页"
                 f" 路径={img.get('path', '')}"]
        if img.get("caption"):
            block.append(f"图像描述：{img['caption']}")
        if vqa:
            block.append(f"图表问答VQA：{vqa[:900]}")
        if ocr:
            block.append(f"图内文字OCR：{ocr}")
        parts.append("\n".join(block))

    # 2) 工单四：表格块（复用工单三 build_table_aware_prompt 的表格段落格式）
    for i, t in enumerate(table_chunks, 1):
        parts.append(f"【表{i}】doc={t.get('doc_id', '?')} 第{t.get('page', '?')}页\n"
                     f"{(t.get('content') or t.get('table_text') or '')[:800]}")

    # 3) 工单四：文本块
    # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：单块截断 600→1000。
    # 招股书 chunk 前部多为页眉页脚，600 字符会把报告期首年金额截掉
    # （实测"军用收入"题首年 6,464.51 万元因此不可见，模型只能答占比）。
    for i, c in enumerate(text_chunks, 1):
        parts.append(f"【资料{i}】doc={c.get('doc_id', '?')} 第{c.get('page', '?')}页\n"
                     f"{(c.get('content') or '')[:1000]}")

    ctx = "\n\n".join(parts)[:max_chars]
    return f"问题：{query}\n\n资料：\n{ctx}\n\n请依据以上资料回答问题。"


# 工单四：复用工单三 LLM 客户端（增量复用，不重写）
from src.llm_client_v3 import chat  # noqa: E402,F401
