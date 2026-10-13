# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：中英文分词（倒排索引与 BM25 共用）

为什么单独成模块：
  全文检索（自研倒排索引，src/fulltext.py）与 BM25 稀疏召回（src/retriever.py）
  必须使用**完全一致的分词口径**，否则同一句话在两路索引里切出的词不同，
  融合后的排序会互相矛盾。把分词抽到独立模块，两边 import 同一份实现。

分词口径：
  · 数字（含小数/千分位/百分号）整体保留；
  · 英文单词整体保留并小写（保留连字符）；
  · 连续汉字交给 jieba，但**领域专名优先整体成词**（"发行后总股本""国家科技进步一等奖"
    这类词 jieba 会切错，而它们在招股书里是关键检索词）。
"""
from __future__ import annotations

import re

# 中英混排分词：数字(含小数/千分位) | 英文词(允许内嵌数字与连字符) | 连续汉字
# 英文词必须允许数字：实测 "BGE-M3" 若只允许字母会被切成 "bge-m" + "3"，
# 而模型名/指标名（M3E、PDF-Extract-Kit-1.0、C4ISR）在检索里是完整语义单元。
_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9._\-]*|[一-鿿]+")

# 作为整体出现的领域专名（jieba 可能切错）
_PROTECTED_TERMS: tuple[str, ...] = (
    "注册资本", "法定代表人", "主营业务收入", "军用领域", "国家科技进步一等奖",
    "募集资金", "补充流动资金", "重要供应商", "实际控制人", "发行股数",
    "招股意向书", "电子信息行业", "技术标准", "视频指挥系统",
    # --- 工单3 新增（表格列名 / 招股2 相关）---
    "发行后总股本", "每股面值", "每股发行价格", "发行股票类型", "拟上市证券交易所",
    "发行前总股本", "持股比例", "关联关系", "存在控制关系", "不存在控制关系",
    "关联方", "控股股东", "募集资金投资项目", "承销方式", "保荐机构",
    "主承销商", "限售安排", "武汉力源信息技术股份有限公司",
    # --- 工单6 新增（全文检索/混合检索重点问题词）---
    "组织结构图", "大客户销售部", "销售处", "市场应用结构", "增长率",
    "仓储及物流中心", "电子商务平台", "研发中心", "扩充产品种类和数量",
    "营运资金", "融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源",
    "力源贸易", "普芯达", "视频指挥", "军队视频指挥", "集成电路",
    "武汉兴图新科电子股份有限公司", "招股说明书",
)

# 索引时保留、但查询解析中不单独成词的连接词（避免 "和" "与" 制造全量命中）
QUERY_STOPWORDS = {"的", "了", "是", "和", "与", "及", "或", "在", "为", "对", "有", "被"}


def _split_by_protected(text: str) -> list[str]:
    """把一段汉字按保护词切分：保护词整体保留，其余碎片留给 jieba。"""
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(text):
        hit = None
        for term in _PROTECTED_TERMS:
            if text.startswith(term, i):
                hit = term
                break
        if hit:
            if buf:
                out.append(buf)
                buf = ""
            out.append(hit)
            i += len(hit)
        else:
            buf += text[i]
            i += 1
    if buf:
        out.append(buf)
    return out


def _number_forms(tok: str) -> list[str]:
    """数字 token 的等价形态（**数字口径归一**）。

    招股书里同一个数字有多种写法，检索时必须能互相命中：
        "5,520.00万元" ← 查询常写 "5,520"、"5520万"
        "25.04%"       ← 查询常写 "25.04%"
    实测教训（工单6 自测）：只按字面切词时，查询 "5,520" 与正文 "5,520.00"
    是两个不同的 term，全文检索直接漏检 —— 而招股书问题里数字恰恰是主干。
    因此对小数部分全为 0 的数字，额外产出「去小数」形态；
    千分位数字额外产出「无逗号」形态，两种都进倒排表。
    """
    out = [tok]
    base = tok.rstrip("%")
    if "." in base:
        int_part, frac = base.split(".", 1)
        if set(frac) <= {"0"}:
            out.append(int_part + ("%" if tok.endswith("%") else ""))
    if "," in base:
        out.append(base.replace(",", "") + ("%" if tok.endswith("%") else ""))
        if "." in base and set(base.split(".", 1)[1]) <= {"0"}:
            out.append(base.split(".", 1)[0].replace(",", "")
                       + ("%" if tok.endswith("%") else ""))
    # 去重且保持顺序
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def tokenize(text: str) -> list[str]:
    """中英混合分词：汉字走 jieba（保护词优先），英文/数字整体保留并小写。

    数字会产出等价形态（见 _number_forms），使 "5,520" 与 "5,520.00" 可互查。
    """
    import jieba

    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text or ""):
        tok = m.group(0)
        if tok[0].isascii():
            tokens.extend(_number_forms(tok.lower()))
            continue
        for piece in _split_by_protected(tok):
            if piece in _PROTECTED_TERMS:
                tokens.append(piece)
            else:
                tokens.extend(t for t in jieba.cut(piece) if t.strip())
    return tokens


def tokenize_query(text: str) -> list[str]:
    """查询侧分词：在通用分词基础上丢弃纯连接词（不丢保护词）。"""
    return [t for t in tokenize(text) if t not in QUERY_STOPWORDS]
