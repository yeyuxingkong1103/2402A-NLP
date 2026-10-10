# -*- coding: utf-8 -*-
"""多语言支持：中文 / 英文问答。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

实现方式：
    1. 语言检测：依据字符集判断问句语种；
    2. 英文提问 → 中文检索：用术语对照表把英文关键词映射为中文，
       使英文问题也能命中中文招股说明书知识库；
    3. 英文作答：可选调用本地大模型把中文答案译为英文（默认开启于界面英文模式）。
"""
from __future__ import annotations

import re

from src import config

_CJK = re.compile(r"[\u4e00-\u9fa5]")

# 英文 → 中文术语对照（覆盖验收问题与常见字段）
EN2CN = {
    "military": "军用领域", "military field": "军用领域", "military revenue": "军用领域收入",
    "revenue": "收入", "income": "收入", "sales": "收入",
    "national science and technology progress award": "国家科技进步一等奖",
    "first prize": "国家科技进步一等奖", "award": "一等奖",
    "engineering": "工程", "project": "工程", "c4isr": "C4ISR",
    "legal representative": "法定代表人", "representative": "法定代表人",
    "chairman": "董事长", "general manager": "总经理",
    "organizational chart": "组织结构图", "organization chart": "组织结构图",
    "org chart": "组织结构图", "structure": "结构图",
    "sales department": "销售部", "sales office": "销售处", "sales offices": "销售处",
    "key account": "大客户销售部", "key account sales department": "大客户销售部",
    "xingtu": "兴图新科", "xingtuxinke": "兴图新科",
    "ps information": "力源信息", "liyuan": "力源信息", "p&s": "力源信息",
    "wuhan": "武汉", "beijing": "北京", "guangzhou": "广州", "chengdu": "成都",
    "shenzhen": "深圳", "zhuhai": "珠海", "shanghai": "上海",
    "registered capital": "注册资本", "shares": "发行股数",
    "prospectus": "招股说明书", "reporting period": "报告期内",
    "which": "哪些", "how many": "多少", "who": "谁", "what": "什么",
}


def detect_lang(text: str) -> str:
    """检测语种：'zh' 或 'en'。"""
    return "zh" if _CJK.search(text or "") else "en"


def en_to_cn_query(text: str) -> str:
    """英文问句 → 中文检索问句（术语替换）。"""
    t = (text or "").lower()
    # 长词优先替换
    for en in sorted(EN2CN, key=len, reverse=True):
        if en in t:
            t = t.replace(en, " " + EN2CN[en] + " ")
    t = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]+", " ", t).strip()
    return t or text


def to_retrieval_query(text: str) -> tuple[str, str]:
    """返回（语种, 用于检索的问句）。"""
    lang = detect_lang(text)
    if lang == "en":
        return lang, en_to_cn_query(text)
    return lang, text


def translate_answer(text: str, target: str = "en") -> str:
    """用本地大模型把答案翻译为目标语言（失败则原样返回）。"""
    if target != "en" or not text:
        return text
    if not _CJK.search(text):
        return text
    try:
        import requests
        prompt = ("Translate the following Chinese text into concise English. "
                  "Output only the translation.\n\n" + text)
        r = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}},
            timeout=config.LLM_TIMEOUT,
        )
        out = (r.json().get("response") or "").strip()
        out = re.sub(r"^.*?", "", out, flags=re.S).strip()
        return out or text
    except Exception:
        return text