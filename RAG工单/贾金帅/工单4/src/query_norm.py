"""
Query 归一化（检索前置处理）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

对应工单功能需求「Query理解 → 消歧/分解」的最便宜那一层：
在调融合权重、换模型之前先做这一步，改动最小、收益最大。

三个实测有效的规则（顺序不可调换）：
  1) 切掉「输出格式指令」尾巴 —— 「……？用三句话回答」会把指令词送进 BM25，
     与无关文档产生关键词命中，稀释话题词权重。
  2) 去掉句末标点 —— 句号/问号会被 bge 当成独立 token 参与池化，
     把句向量往「泛问句」方向拉偏，实测 top5 命中率从 5/5 掉到 3/5。
  3) 口语 → 书面语替换（按 key 长度降序，避免短 key 遮蔽长 key）。

每一步都有**长度兜底**：切完/替完不足 4 字就退回原串，防止「帮我总结」这类
整句即指令的输入被清成空串。
"""
from __future__ import annotations

import re

from .config import WORK_ORDER_NO  # noqa: F401

# 句末标点（只去句末，句中标点保留）
_TRAILING_PUNCT = "？?。.！!；;，,、~～ \t\r\n"

# 「看起来像输出要求」的分句特征
_INSTRUCTION_RE = re.compile(
    r"用\s*[一二三四五六七八九十\d]+\s*句话|每句|一行|字数|不超过|"
    r"只(?:回答|说|输出|给)|不要(?:解释|展开)|简(?:要|单)|详细|"
    r"举例|罗列|列出|写出|给出|翻译|总结|概括|改写|请(?:用|以|帮我)|回答|输出|格式"
)

# 口语 → 书面语（用于提升与正式文档的词汇重合度）
_VERBAL_MAP = {
    "咋办": "如何处理",
    "咋样": "情况如何",
    "咋": "如何",
    "啥": "什么",
    "多少块": "金额",
    "多少钱": "金额",
    "挣了多少": "收入",
    "赚了多少": "收入",
    "老大": "法定代表人",
    "一把手": "法定代表人",
    "老板": "法定代表人",
    "注册资金": "注册资本",
    "本公司": "公司",
    "这家公司": "公司",
    "他们公司": "公司",
}


def _strip_instruction(query: str) -> str:
    """从尾部逐个吃掉「像输出要求」的分句。"""
    clauses = [c for c in re.split(r"(?<=[。！？；!?;，,\n])", query) if c.strip()]
    kept = list(clauses)
    while len(kept) > 1 and _INSTRUCTION_RE.search(kept[-1]):
        candidate = "".join(kept[:-1]).strip().rstrip("，,、 ")
        if len(candidate) < 4:  # 兜底：切完没剩东西就不切
            break
        kept.pop()
    return "".join(kept).strip() or query


def _strip_trailing_punct(text: str) -> str:
    stripped = text.rstrip(_TRAILING_PUNCT)
    return stripped if len(stripped) >= 4 else text


def _apply_verbal_map(text: str) -> str:
    """口语替换。**必须按 key 长度降序**，否则「咋」会先命中、
    把「咋办」替换成「如何办」，而「咋办」那条成为死条目。"""
    out = text
    for verbal, formal in sorted(_VERBAL_MAP.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(verbal, formal)
    return out or text


def normalize_query(query: str) -> str:
    """归一化用户 query，返回**真正用于检索**的字符串。"""
    q = (query or "").strip()
    if not q:
        return ""
    q = _strip_instruction(q)
    q = _strip_trailing_punct(q)
    q = _apply_verbal_map(q)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q or query.strip()
