# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/table_parser/query_router.py —— 工单三表格感知路由

职责（见 docs/02_表格检索优化方案.md §五）：
  判断用户问题是否涉及表格，输出检索路由信号：
    - text_only  ：纯文本问题（公司简介、主营业务描述等）
    - table_only：纯表格问题（发行股数、持股比例、收入金额等数字/表格类）
    - hybrid    ：既需表格数值又需文本语境（"募集资金用途及金额"）

路由信号：
  1. 关键词命中（股数/比例/注册资本/募集资金/关联方/持股/收入/金额/数量/前五大…）
  2. 数字 + 单位模式（万股 / 万元 / % / 元）
  3. 疑问词 + 表格实体（"…是多少" / "…比例" / "…金额"）
  4. 英文等价关键词（shares / ratio / amount / revenue / related party）
"""
import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

# 工单三：表格关键词（中文）
TABLE_KEYWORDS_CN = {
    # 数值/单位类
    "股数", "股份", "比例", "注册资本", "募集资金", "募集资金总额",
    "关联方", "关联交易", "持股", "持股比例", "持股数量", "收入", "营业收入",
    "金额", "数量", "总额", "净额", "占比", "前五大", "前五名", "前十",
    "股东", "控股股东", "实际控制人", "毛利率", "净利率", "资产负债",
    "总资产", "净资产", "总股本", "股本", "发行价", "发行价格", "发行数量",
    "投向", "募投", "募集资金投向", "资金用途",
    # 表格问句模板
    "是多少", "多少", "占比多少", "比例是多少", "金额是多少",
}
# 工单三：表格关键词（英文）
TABLE_KEYWORDS_EN = {
    "shares", "share", "ratio", "registered", "capital", "raised", "funds",
    "related", "party", "parties", "holding", "holdings", "revenue",
    "amount", "quantity", "total", "percentage", "top", "shareholder",
    "shareholders", "controller", "gross", "margin", "net", "asset", "assets",
    "price", "issued", "issuance", "investment", "project", "projects",
    "how many", "how much", "what is the",
}
# 工单三：数字+单位正则（"1,670万股" "98,765万元" "30.5%"）
_NUM_UNIT_RE = re.compile(
    r"[\d,]+\.?\d*\s*(?:万股|万元|亿元|元|%|％|万|亿)",
    re.UNICODE,
)
# 工单三：纯文本描述关键词（公司介绍、业务描述）
_TEXT_ONLY_HINTS = {
    "简介", "概况", "主营业务", "业务介绍", "公司概况", "发展历程",
    "公司是做什么的", "introduction", "overview", "background",
    "history", "description", "about",
}


@dataclass
class RouteResult:
    """工单三：路由结果"""
    route: str  # text_only / table_only / hybrid
    confidence: float  # 0.0 ~ 1.0
    matched_keywords: list  # 命中的关键词
    reason: str  # 路由理由（可读）

    def __repr__(self):
        return (f"RouteResult(route={self.route}, confidence={self.confidence:.2f}, "
                f"keywords={self.matched_keywords}, reason={self.reason})")


def route_query(query: str) -> RouteResult:
    """工单三：判断问题路由

    路由逻辑：
      1. 命中 _TEXT_ONLY_HINTS → text_only（高置信）
      2. 命中 ≥2 个 TABLE_KEYWORDS 或命中数字+单位模式 → table_only
      3. 命中 1 个 TABLE_KEYWORD 且无文本提示 → hybrid（保守，两者都查）
      4. 默认 → hybrid
    """
    if not query or not query.strip():
        return RouteResult("hybrid", 0.0, [], "空查询，默认 hybrid")

    q = query.strip()
    q_lower = q.lower()

    # 1. 纯文本提示
    text_hits = [k for k in _TEXT_ONLY_HINTS if k in q_lower]
    if text_hits:
        return RouteResult("text_only", 0.8, text_hits,
                           f"命中纯文本提示词 {text_hits}")

    # 2. 表格关键词命中
    cn_hits = [k for k in TABLE_KEYWORDS_CN if k in q]
    en_hits = [k for k in TABLE_KEYWORDS_EN if k in q_lower]
    table_hits = cn_hits + en_hits

    # 3. 数字+单位模式
    num_unit_hits = _NUM_UNIT_RE.findall(q)

    # 4. 路由决策
    # 工单三：score >= 5 才 table_only（短表 table_text 易被长表挤掉，hybrid 更稳）
    score = len(table_hits) + len(num_unit_hits) * 2
    if score >= 5:
        return RouteResult("table_only", 0.9, table_hits + num_unit_hits,
                           f"表格信号强 score={score}")
    if score >= 1:
        # 1-4 个表格信号 → hybrid（兼顾文本语境，避免短表被漏召回）
        return RouteResult("hybrid", 0.7, table_hits + num_unit_hits,
                           f"表格信号中等 score={score}，hybrid 兼顾语境")
    # 无任何信号 → 默认 hybrid（保守，避免漏召回）
    return RouteResult("hybrid", 0.5, [],
                       "无明确信号，默认 hybrid 保守召回")


def is_table_query(query: str) -> bool:
    """工单三：便捷函数——是否涉及表格（table_only 或 hybrid）"""
    r = route_query(query)
    return r.route in ("table_only", "hybrid")


if __name__ == "__main__":  # pragma: no cover
    import sys
    tests = [
        "本次发行的发行股数是多少？",
        "公司前五大股东的持股比例是多少？",
        "公司在军用领域的营业收入是多少？",
        "公司简介是什么？",
        "How many shares are issued in this offering?",
        "募集资金投向哪些项目？",
    ]
    for q in tests:
        print(f"{q}  →  {route_query(q)}")
