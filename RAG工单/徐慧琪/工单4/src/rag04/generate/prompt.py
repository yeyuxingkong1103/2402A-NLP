# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""中英双语 prompt 模板与引用构造。"""
from __future__ import annotations

import re

from rag04.schema import Hit

_CJK = re.compile(r"[一-鿿]")

_TYPE_LABEL = {
    "text": "正文", "table": "表格", "image": "图像",
}


def detect_lang(text: str) -> str:
    return "zh" if _CJK.search(text or "") else "en"


def build_context(hits: list[Hit]) -> str:
    """把召回块拼成带溯源信息的上下文。"""
    if not hits:
        return "（无检索结果）"
    parts = []
    for i, h in enumerate(hits, 1):
        label = _TYPE_LABEL.get(h.block_type, h.block_type)
        parts.append(
            f"[片段{i}] 文档：{h.doc_id} | 页码：{h.page} | 类型：{label}({h.block_type})"
            f" | 来源ID：{h.source_id}\n{h.text}"
        )
    return "\n\n".join(parts)


_ZH_SYS = """你是招股说明书问答助手，只依据给定的检索片段作答。

严格要求：
1. 只使用检索片段中的信息作答，不得使用外部知识、不得推测、不得编造。
2. **必须用中文回答**（用户用中文提问）。
3. 如果问题是列举型（如"有哪些""哪几个""分别是多少"），
   必须**完整列出全部项目**，逐条列出，**不得省略、不得用"等"字概括**。
4. 若片段中的信息不足以回答，明确回答"根据文档内容无法确定"，不要编造。
5. 回答末尾附上引用来源（页码与类型）。
6. **输出格式（必须严格遵守）**：第一行只输出一个 JSON 对象——
   `{"answerable": true}` 或 `{"answerable": false}`。
   判定依据：检索片段中的证据是否足以回答该问题（true=足以作答；
   false=证据不足、只能拒答）。从第二行起输出答案正文；若 answerable 为
   false，正文写"根据文档内容无法确定"并简述缺什么证据。
   answerable 必须与正文结论一致：正文给出了答案就不得标 false，
   正文说无法确定就不得标 true。"""

_EN_SYS = """You are a prospectus Q&A assistant. Answer ONLY from the provided retrieved passages.

Strict requirements:
1. Use only the retrieved passages. Do not use outside knowledge, do not guess, do not fabricate.
2. **You must answer in English** (the user asked in English).
3. If the question is enumerative ("which", "what are", "list"), you MUST list
   EVERY item completely, one per line. Do NOT omit items and do NOT summarize with "etc.".
4. If the passages are insufficient, answer "Cannot be determined from the document." Do not fabricate.
5. Append the citation sources (page number and type) at the end of your answer.
6. **Output format (strict)**: the FIRST line must be a single JSON object:
   `{"answerable": true}` or `{"answerable": false}` (true = the passages are
   sufficient to answer; false = insufficient evidence, refusal only). The answer
   body starts on the second line. The flag must match the body: never set false
   when the body actually answers, never set true when the body refuses."""


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    """构造对话消息。按提问语言选择系统提示。"""
    lang = detect_lang(question)
    system = _ZH_SYS if lang == "zh" else _EN_SYS
    user = (
        f"检索片段：\n{build_context(hits)}\n\n"
        f"问题：{question}\n\n"
        + ("请用中文作答。" if lang == "zh" else "Please answer in English.")
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_citations(hits: list[Hit]) -> list[dict]:
    """结构化引用：页码 + block_type + 来源ID。"""
    out = []
    for h in hits:
        out.append({
            "doc_id": h.doc_id,
            "page": h.page,
            "block_type": h.block_type,
            "source_id": h.source_id,
            "chunk_id": h.chunk_id,
            "image_path": h.image_path,
            "score": round(h.score, 6),
            "channel": h.channel,
            "snippet": h.text[:200],
        })
    return out
