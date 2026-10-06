# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估

测试用例构建脚本 —— 把《sample_questions.pdf》整理成结构化测试集，并扩充到 10 个问题。

流程：
  1. 用 PyMuPDF 提取 sample_questions.pdf 全文，解析出 4 个示例问题及其参考答案；
  2. 示例问题里用「xx银行 / xx人寿」做了脱敏，脚本通过**在 9 份年报 txt 里检索
     特征短语**的方式自动定位真实出处（例如「逾越者联盟 / 咖啡零售」只出现在招商银行），
     定位结果作为证据留档，而不是拍脑袋硬编码；
  3. 在示例问题基础上，围绕 9 份年报正文扩充 6 个问题，凑成 10 个测试用例，
     覆盖：银行 / 保险 / 证券三个子行业，数值型 / 事实型 / 分析型 / 归纳型四类题型，
     单文档检索与多文档归纳两种检索形态；
  4. 每个用例标注：问题、题型、主责文档、期望页码、答案要点、参考答案（ground_truth）、
     自动判准关键词、是否需要表格、难度、设计说明；
  5. 落盘 results/test_cases.json 与 results/test_cases.md。

用法：
    python build_questions.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import config  # noqa: E402
from prepare_corpus import (WO_NO, ensure_results_dir, list_ccf_docs,  # noqa: E402
                            decode_garbled_filename, STOCK_CODE_MAP)

# ---------------------------------------------------------------------------
# 一、解析 sample_questions.pdf
# ---------------------------------------------------------------------------
#: 示例问题用的是「xx银行」这类占位符，这里用「特征短语在年报 txt 中的出现次数」定位真实出处。
#: 特征短语全部来自示例参考答案本身，可人工复核，不是臆测。
SAMPLE_EVIDENCE: list[dict] = [
    {
        "keywords": ["董事长致辞", "盈利增长", "关键因素"],
        "must_contain": ["逾期60天以上贷款占比", "大零售、大对公六四占比"],
        "candidates": ["平安银行2019年报", "招商银行2019年报", "邮储银行2019年报"],
        "expect_multi": False,
    },
    {
        "keywords": ["创新商业模式"],
        "must_contain": ["逾越者联盟", "咖啡零售", "出行预订", "影票销售"],
        "candidates": ["招商银行2019年报", "平安银行2019年报", "中信证券2020年报"],
        "expect_multi": False,
    },
    {
        "keywords": ["风险管理体系", "宏观经济周期波动", "拨备覆盖率", "贷款结构优化"],
        "must_contain": ["逾期60天以上贷款余额", "逾期60天以上贷款偏离度", "零售突破"],
        "candidates": ["平安银行2019年报"],
        "expect_multi": False,
    },
    {
        # 第 4 题原题就是「这些银行和保险公司」的跨机构归纳，
        # 特征短语在 9 份年报中普遍存在，因此直接标注为多文档题，
        # 定位结果只用于展示「哪些年报提供了证据」。
        "keywords": ["共同策略", "差异化策略", "经济周期波动", "绿色金融", "科技金融"],
        "must_contain": ["拨备覆盖率", "绿色金融", "资产负债匹配", "碳达峰"],
        "candidates": ["平安银行2019年报", "招商银行2019年报", "邮储银行2019年报",
                       "中国平安2019年报", "中国人寿2020年报", "中国太保2021年报",
                       "国泰君安2021年报"],
        "expect_multi": True,
    },
]


#: sample_questions.pdf 的文本层使用了「康熙部首 / CJK 部首补充」区的兼容字符，
#: 例如把「行」写成 U+2F8F（⼀样的字形，但码位不同）。这些字符不会影响人眼阅读，
#: 却会让「关键词匹配」「BM25 检索」直接失配——必须在入库前归一化。
#: NFKC 能处理康熙部首（U+2F00–U+2FDF），但处理不了 CJK 部首补充区（U+2E80–U+2EFF），
#: 因此对后者补一张小映射表。
_RADICAL_SUPPLEMENT = {
    "⺋": "雨", "⻓": "长", "⻛": "风", "⻔": "门",
    "⻅": "页", "⻌": "鱼", "⻗": "鸟", "⻣": "马",
    "⻑": "食", "⻅": "页",
}


def normalize_text(s: str) -> str:
    """
    归一化 PDF 提取文本。

    只对「部首区」字符做归一化，**不动标点**——
    整体跑 NFKC 会把全角逗号「，」也压成半角「,」，中文语料里反而更难读。
    """
    import unicodedata

    s = s.replace("　", " ").replace("\xa0", " ")
    out = []
    for ch in s:
        o = ord(ch)
        if 0x2E80 <= o <= 0x2FDF:            # CJK部首补充 + 康熙部首
            out.append(_RADICAL_SUPPLEMENT.get(ch, unicodedata.normalize("NFKC", ch)))
        else:
            out.append(ch)
    return "".join(out)


def extract_sample_text(pdf_path: Path | None = None) -> str:
    """用 PyMuPDF 提取 sample_questions.pdf 的全文文本，并做兼容字符归一化。"""
    import fitz

    pdf_path = Path(pdf_path or config.PDF_SAMPLE_QUESTIONS)
    if not pdf_path.exists():
        raise FileNotFoundError(f"未找到示例问题文件：{pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        text = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()
    return normalize_text(text)


def parse_sample_questions(text: str) -> list[dict]:
    """
    从全文里切分出示例问题。

    文本形如：
        问题：
        xx银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？
        答案：
        xx银行在2019年的盈利增长主要得益于以下因素：…
    """
    blocks = re.split(r"\n\s*问题\s*[:：]", text)
    out: list[dict] = []
    for b in blocks[1:]:
        m = re.search(r"\n\s*(参考答案|答案)\s*[:：]", b)
        if m:
            raw_q = b[:m.start()]
            raw_a = b[m.end():]
        else:
            raw_q, raw_a = b, ""
        question = _tidy(raw_q)
        # 文档占位名提示（要从清洗前的原文里找，例如 "xxx__2019年__年度报告"）
        hints = re.findall(r"[xX]+\w*__\d{4}年?__\S+", raw_a)
        answer = _clean_answer(raw_a)
        out.append({
            "question": question,
            "reference_answer": answer,
            "doc_hints": hints,
        })
    return [q for q in out if q["question"]]


def _tidy(s: str) -> str:
    """去掉换行与多余空白，拼回一行问题。"""
    return re.sub(r"\s+", "", s).strip()


def _clean_answer(a: str) -> str:
    """清掉文档占位名等噪声行，保留正文。"""
    lines = []
    for ln in a.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"[xX]+\w*__\d{4}年?__\S+", s):     # 如 xxx__2019年__年度报告
            continue
        if s.startswith("====="):                            # 分页标记
            continue
        lines.append(s)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 二、用年报正文定位示例问题真实出处
# ---------------------------------------------------------------------------
def load_report_texts(docs: list[dict] | None = None) -> dict[str, str]:
    """
    读取 9 份年报的 txt 文本（UTF-8），以规范文档名为键。

    txt 目录与 pdf 目录同名同构，是 CCF 赛题官方给出的解析结果，
    这里只用来**取证**（定位示例问题的真实出处、核对数字），不参与建索引。
    """
    docs = docs or [d.to_dict() for d in list_ccf_docs()]
    texts: dict[str, str] = {}
    for d in docs:
        p = Path(d["txt_path"])
        if not p.exists():
            continue
        raw = p.read_bytes()
        try:
            texts[d["doc_name"]] = raw.decode("utf-8")
        except UnicodeDecodeError:
            texts[d["doc_name"]] = raw.decode("gbk", errors="replace")
    return texts


def locate_sample_source(sample: dict, texts: dict[str, str],
                         evidence: dict) -> dict:
    """
    判定示例问题的真实出处。

    判定方式：统计「特征短语」在各候选年报中的出现次数，取最高分文档；
    命中数为 0 时判为「多文档问题」。
    """
    scores: dict[str, int] = {}
    for name, text in texts.items():
        scores[name] = sum(text.count(p) for p in evidence["must_contain"])
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    hits = [r for r in ranked if r[1] > 0]
    best_name, best_score = ranked[0] if ranked else ("", 0)

    if evidence.get("expect_multi"):
        detail = "、".join(f"{n} {s} 次" for n, s in hits[:5])
        return {"doc": "", "multi": True, "scores": scores,
                "reason": f"原题即为跨机构归纳题；特征短语在 {len(hits)} 份年报中均有命中"
                          f"（{detail}），需多文档联合作答"}

    if best_score <= 0:
        return {"doc": "", "multi": True, "scores": scores,
                "reason": "特征短语在所有年报中均未命中，判为多文档归纳题"}
    return {"doc": best_name, "multi": False, "scores": scores,
            "reason": f"特征短语在《{best_name}》命中 {best_score} 次，"
                      f"其余年报命中≤{ranked[1][1] if len(ranked) > 1 else 0} 次"}


# ---------------------------------------------------------------------------
# 三、10 个测试用例（4 个来自示例问题 + 6 个基于年报内容扩充）
# ---------------------------------------------------------------------------
def build_test_cases(samples: list[dict], texts: dict[str, str]) -> list[dict]:
    """组装 10 个测试用例。"""
    resolved = []
    for i, s in enumerate(samples[:4]):
        ev = SAMPLE_EVIDENCE[i] if i < len(SAMPLE_EVIDENCE) else SAMPLE_EVIDENCE[-1]
        loc = locate_sample_source(s, texts, ev)
        s = {**s, "source": loc}
        resolved.append(s)

    cases: list[dict] = []

    # ---------------- 用例 1（示例问题 1）----------------
    cases.append({
        "id": "Q01",
        "question": "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？",
        "question_type": "归纳型",
        "origin": "sample_questions.pdf（示例问题1，脱敏的「xx银行」已定位为平安银行）",
        "reference_doc": "平安银行2019年报",
        "reference_docs": ["平安银行2019年报"],
        "expected_pages": [5],
        "expected_answer_points": [
            "业务结构不断优化、更趋均衡：零售进一步巩固收入贡献的支柱地位，对公、资金同业走上快车道，大零售、大对公六四占比的均衡格局逐步形成",
            "盈利增长形成趋势且风险可控：2019年营收、净利润增速创转型以来新高，建立在扎实的资产质量基础上",
            "逾期60天以上贷款占比、偏离度不断下降，拨备覆盖率不断提升，成绩单有底气、可持续",
            "数据化经营能力不断提升：通过智慧经营平台打造智能化的决策大脑、风控大脑、服务大脑，把经验驱动转变为数据驱动",
        ],
        "ground_truth": (
            "平安银行2019年董事长致辞（《金融向善，履责前行》）把盈利增长归结为四点："
            "一是业务结构不断优化、更趋均衡，零售进一步巩固收入贡献的支柱地位，"
            "对公、资金同业也进一步理顺思路走上快车道，大零售、大对公六四占比的均衡格局逐步形成；"
            "二是盈利增长形成趋势且风险可控，2019年营收、净利润增速创转型以来新高，"
            "且建立在扎实的资产质量基础上，逾期60天以上贷款占比、偏离度不断下降，拨备覆盖率不断提升，"
            "这份成绩单有底气、可持续；三是数据化经营能力不断提升且渐显成效，通过智慧经营平台"
            "打造智能化的决策大脑、风控大脑、服务大脑，将经验驱动转变为数据驱动，"
            "并不断提升线上化运营能力，推动业务效率提升、成本降低；"
            "四是均衡结构推动盈利增长、资产质量夯实后方防线、数据化能力促进模式创新，既着眼当下又布局未来。"
        ),
        "answer_keywords": ["业务结构", "零售", "盈利增长", "风险可控",
                            "逾期60天以上贷款", "拨备覆盖率", "数据化经营", "智慧经营平台"],
        "need_table": False,
        "difficulty": "中",
        "top_k": 5,
        "sample_source": resolved[0]["source"],
        "design_note": "示例问题原题。定位结果：" + resolved[0]["source"]["reason"] +
                       "。考查「董事长致辞」这一非结构化章节的归纳能力，"
                       "验证 RAG 能否把散落在同一页致辞中的多条论点合并成结构化答案。",
    })

    # ---------------- 用例 2（示例问题 2）----------------
    cases.append({
        "id": "Q02",
        "question": "招商银行在其2019年年报中提到的创新商业模式有哪些？",
        "question_type": "事实型",
        "origin": "sample_questions.pdf（示例问题2，脱敏的「xx银行」已定位为招商银行）",
        "reference_doc": "招商银行2019年报",
        "reference_docs": ["招商银行2019年报"],
        "expected_pages": [5],
        "expected_answer_points": [
            "以共生共建的理念，聚焦高频生活场景，联合合作伙伴共同为用户提供优质服务",
            "召开首次由商业银行举办的合作伙伴大会",
            "发起设立「逾越者联盟」",
            "参与咖啡零售、出行预订、影票销售等，探索开放的新生态模式",
        ],
        "ground_truth": (
            "招商银行2019年度报告（董事长致辞）提出「锐意转型创新，培育创新土壤，探索创新商业模式」："
            "以共生共建的理念，聚焦高频生活场景，联合合作伙伴共同为用户提供优质服务；"
            "通过召开首次由商业银行举办的合作伙伴大会、发起设立「逾越者联盟」、"
            "参与咖啡零售、出行预订、影票销售等，探索开放的新生态模式；"
            "同时将金融科技投入、市场化选人用人机制和薪酬激励机制纳入公司章程，"
            "持续优化员工职业发展通道，加强金融科技人才吸引和培养，通过蛋壳平台建立「平视、包容」文化。"
        ),
        "answer_keywords": ["高频生活场景", "合作伙伴大会", "逾越者联盟",
                            "咖啡零售", "出行预订", "影票", "生态"],
        "need_table": False,
        "difficulty": "中",
        "top_k": 5,
        "sample_source": resolved[1]["source"],
        "design_note": "示例问题原题。定位结果：" + resolved[1]["source"]["reason"] +
                       "。特征词「逾越者联盟」「咖啡零售」是强区分度实体，"
                       "考查 RAG 对**专有名词**的精确召回——这正是 BM25 通道相对纯向量检索的优势场景。",
    })

    # ---------------- 用例 3（示例问题 3）----------------
    cases.append({
        "id": "Q03",
        "question": "平安银行的风险管理体系如何体现其应对宏观经济周期波动的能力？"
                    "结合年报中的拨备覆盖率动态调整、资产质量控制，以及贷款结构优化策略，"
                    "分析其应对经济下行周期的准备程度及可能的潜在压力。",
        "question_type": "分析型",
        "origin": "sample_questions.pdf（示例问题3）",
        "reference_doc": "平安银行2019年报",
        "reference_docs": ["平安银行2019年报"],
        "expected_pages": [5, 22, 27],
        "expected_answer_points": [
            "拨备覆盖率动态调整：2017-2019年末分别为151.08%、155.24%、183.12%，逐年提升，2019年较上年末提高27.88个百分点；拨贷比由2.57%提升至3.01%",
            "资产质量控制：2019年末不良贷款率1.65%、较上年末下降0.10个百分点；逾期60天以上贷款余额367.82亿元、占比1.58%（下降0.34个百分点），逾期60天以上/90天以上贷款偏离度均低于1",
            "贷款结构优化：坚持「零售突破」，新增资源重点投向资产质量较好的零售业务；持续「对公做精」，新增业务聚焦成长性好、符合国家战略方向的行业，加大问题资产清收处置力度",
            "准备程度：高拨备覆盖率与资产结构调整为应对经济波动提供缓冲；潜在压力：区域与行业集中度、风险暴露的滞后传导",
        ],
        "ground_truth": (
            "（1）拨备覆盖率动态调整：平安银行2017—2019年末拨备覆盖率分别为151.08%、155.24%、183.12%，"
            "逐年提高，2019年较上年末提升27.88个百分点，拨贷比由2.57%升至3.01%，"
            "损失吸收能力不断增强，为经济下行期的不良贷款风险预留了缓冲。"
            "（2）资产质量控制：2019年末不良贷款率1.65%，较上年末下降0.10个百分点；"
            "逾期贷款余额485.50亿元、占比2.09%，较上年末下降0.39个百分点；"
            "其中逾期60天以上贷款余额367.82亿元、占比1.58%，较上年末下降0.34个百分点；"
            "逾期90天以上贷款余额314.11亿元、占比1.35%，较上年末下降0.35个百分点；"
            "逾期60天以上和逾期90天以上贷款偏离度均低于1，资产质量指标全面持续改善。"
            "（3）贷款结构优化：坚持「零售突破」，新增资源重点投向资产质量较好的零售业务，"
            "加强零售客户准入标准和管理要求；持续「对公做精」，新增业务聚焦成长性好、"
            "符合国家战略发展方向的行业，集中优势资源投向高质量、高潜力客户，"
            "同时做好存量资产结构调整，加大问题资产清收处置力度。"
            "（4）准备程度与潜在压力：高拨备覆盖率与资产结构调整显示其对潜在经济下行风险的重视，"
            "为应对未来波动提供了缓冲；但区域与行业集中度偏高、不良风险暴露的滞后传导，"
            "仍可能对资产质量形成挑战。"
        ),
        "answer_keywords": ["183.12", "155.24", "151.08", "1.65", "367.82",
                            "拨备覆盖率", "逾期60天以上", "零售突破", "对公做精"],
        "need_table": True,
        "difficulty": "难",
        "top_k": 8,
        "sample_source": resolved[2]["source"],
        "design_note": "示例问题原题。定位结果：" + resolved[2]["source"]["reason"] +
                       "。典型「数值+分析」复合题：答案同时依赖监管指标表（第22页）、"
                       "资产质量明细段（第27页）和董事长致辞（第5页），"
                       "top_k 上调到 8，用来暴露「长文档中目标信息被稀释」的问题。",
    })

    # ---------------- 用例 4（示例问题 4，多文档归纳）----------------
    cases.append({
        "id": "Q04",
        "question": "分析这些银行和保险公司在面对经济周期波动时的共同策略与差异化策略，"
                    "重点关注其在风险管理、资本结构优化和新兴领域（如绿色金融或科技金融）投资上的表现。"
                    "如何评估这些策略在未来经济下行周期中的可持续性？",
        "question_type": "多文档归纳型",
        "origin": "sample_questions.pdf（示例问题4）",
        "reference_doc": "平安银行2019年报",
        "reference_docs": ["平安银行2019年报", "招商银行2019年报", "邮储银行2019年报",
                           "中国平安2019年报", "中国人寿2020年报", "中国太保2021年报",
                           "国泰君安2021年报"],
        "expected_pages": [],
        "expected_answer_points": [
            "共同策略-风险管理：动态调整拨备覆盖率（平安银行183.12%、邮储银行389.45%），加强不良与逾期贷款管理；引入大数据/智能风控（平安银行风控大脑、招商银行天秤系统）",
            "共同策略-资本结构优化：强调内生资本补充、利润留存优化核心资本充足率；保险公司强化资产负债匹配、降低流动性压力",
            "共同策略-新兴领域：绿色金融（中国太保绿色保险绿色投资、中国人寿新增绿色投资、国泰君安碳达峰碳中和行动方案）与科技金融（平安银行「金融+科技」、招商银行数字化经营、国泰君安「SMART投行」）",
            "差异化策略：银行侧重逾期贷款管理与信贷投放结构调整，保险侧重长期资产配置优化与负债久期管理，券商侧重绿色投融资与数字化转型",
            "可持续性评估：高拨备覆盖率和资本充足率提供短期缓冲、绿色与科技投入形成长期增长点，但新兴领域回报不确定、高风险行业敞口调整需要时间",
        ],
        "ground_truth": (
            "共同策略：（1）风险管理方面，各行普遍动态调整拨备覆盖率以增强损失吸收能力，"
            "例如平安银行2019年末拨备覆盖率提升至183.12%，邮储银行2019年末拨备覆盖率为389.45%，"
            "同时加强不良贷款和逾期贷款管理；多家机构引入大数据与智能风控工具用于早期识别风险客户。"
            "（2）资本结构优化方面，各公司均强调内生资本补充、通过利润留存优化资本充足率；"
            "保险公司则通过强化资产负债匹配来降低流动性压力。"
            "（3）新兴领域投资方面，绿色金融上中国太保推进绿色保险与绿色投资、"
            "中国人寿持续新增绿色投资、国泰君安发布碳达峰碳中和行动方案；"
            "科技金融上平安银行以「金融+科技」构建五大生态圈、招商银行推进数字化经营与开放银行、"
            "国泰君安提出「SMART投行」愿景。"
            "差异化策略：银行更多关注逾期贷款管理和信贷投放结构调整（如平安银行「零售突破、对公做精」），"
            "保险公司侧重长期资产配置优化和负债久期管理，证券公司侧重绿色投融资服务与数字化转型。"
            "可持续性评估：高拨备覆盖率与资本充足率为短期波动提供了缓冲，"
            "绿色金融和科技金融投入形成长期增长点；但绿色金融与科技金融的长期回报存在不确定性，"
            "部分高风险行业信贷敞口调整仍需时间，未来可能因经济下行承压。"
        ),
        "answer_keywords": ["拨备覆盖率", "绿色金融", "科技", "资产负债", "资本",
                            "不良贷款", "可持续"],
        "need_table": False,
        "difficulty": "难",
        "top_k": 12,
        "sample_source": resolved[3]["source"],
        "design_note": "示例问题原题（跨 7 份文档）。定位结果：" + resolved[3]["source"]["reason"] +
                       "。这是**单次检索最吃力**的一类题：top_k=12 也难覆盖 7 份文档，"
                       "预留用于验证「跨文档归纳覆盖不全」这一典型问题，"
                       "并作为工单08 Graph RAG 的改进基线。",
    })

    # ---------------- 用例 5（扩充：银行 · 数值型）----------------
    cases.append({
        "id": "Q05",
        "question": "平安银行2019年末的拨备覆盖率、不良贷款率和拨贷比分别是多少？"
                    "与2018年末、2017年末相比各是如何变化的？",
        "question_type": "数值型",
        "origin": "基于年报内容扩充（监管指标表）",
        "reference_doc": "平安银行2019年报",
        "reference_docs": ["平安银行2019年报"],
        "expected_pages": [22],
        "expected_answer_points": [
            "拨备覆盖率：2019年183.12%、2018年155.24%、2017年151.08%（监管标准≥150%）",
            "不良贷款率：2019年1.65%、2018年1.75%、2017年1.70%（监管标准≤5%）",
            "拨贷比：2019年3.01%、2018年2.71%、2017年2.57%（监管标准≥2.5%）",
            "2019年拨备覆盖率较上年提升27.88个百分点，不良率下降0.10个百分点，拨贷比上升0.30个百分点",
        ],
        "ground_truth": (
            "根据平安银行2019年年度报告「主要监管指标」：拨备覆盖率2017—2019年末分别为151.08%、"
            "155.24%、183.12%，监管标准为不低于150%，2019年较2018年上升27.88个百分点；"
            "不良贷款率2017—2019年末分别为1.70%、1.75%、1.65%，监管标准为不高于5%，"
            "2019年较2018年下降0.10个百分点；拨贷比2017—2019年末分别为2.57%、2.71%、3.01%，"
            "监管标准为不低于2.5%，2019年较2018年上升0.30个百分点。"
        ),
        "answer_keywords": ["183.12", "155.24", "151.08", "1.65", "1.75",
                            "3.01", "2.71", "拨备覆盖率", "不良贷款率", "拨贷比"],
        "need_table": True,
        "difficulty": "中",
        "top_k": 5,
        "sample_source": None,
        "design_note": "扩充题。答案**只存在于表格**（第22页监管指标表），"
                       "用来检验工单03 表格解析（表格→Markdown）是否真正生效；"
                       "三列年份紧邻，也是检验 LLM 是否「张冠李戴」的经典样本。",
    })

    # ---------------- 用例 6（扩充：银行 · 数值型）----------------
    cases.append({
        "id": "Q06",
        "question": "招商银行2019年实现归属于股东净利润是多少？同比增长多少？"
                    "加权平均净资产收益率（ROAE）为多少？",
        "question_type": "数值型",
        "origin": "基于年报内容扩充（董事长致辞）",
        "reference_doc": "招商银行2019年报",
        "reference_docs": ["招商银行2019年报"],
        "expected_pages": [5],
        "expected_answer_points": [
            "归属于股东净利润928.67亿元，同比增长15.28%",
            "加权平均净资产收益率（ROAE）16.84%，连续三年提升",
        ],
        "ground_truth": (
            "招商银行2019年实现归属于股东净利润928.67亿元，同比增长15.28%；"
            "加权平均净资产收益率（ROAE）16.84%，连续三年提升；"
            "2019年末A股、H股较年初分别上涨53%和43%，市值创历史新高，"
            "不良额和不良率连续三年实现「双降」。"
        ),
        "answer_keywords": ["928.67", "15.28", "16.84"],
        "need_table": False,
        "difficulty": "易",
        "top_k": 5,
        "sample_source": None,
        "design_note": "扩充题。数字集中在董事长致辞首段，用于验证**基础数值题的正确率上限**；"
                       "与 Q05/Q07 一起构成数值型题组，统计「数字抄错/串年份」的比例。",
    })

    # ---------------- 用例 7（扩充：保险 · 数值型）----------------
    cases.append({
        "id": "Q07",
        "question": "中国人寿2020年年报披露的保费收入、归属于母公司股东的净利润、"
                    "内含价值和一年新业务价值分别是多少？",
        "question_type": "数值型",
        "origin": "基于年报内容扩充（经营亮点指标）",
        "reference_doc": "中国人寿2020年报",
        "reference_docs": ["中国人寿2020年报"],
        "expected_pages": [7],
        "expected_answer_points": [
            "保费收入612,265百万元（约6122.65亿元）",
            "归属于母公司股东的净利润50,268百万元（约502.68亿元）",
            "内含价值1,072,140百万元（约10721.40亿元）",
            "一年新业务价值58,373百万元（约583.73亿元）",
        ],
        "ground_truth": (
            "中国人寿2020年年报「经营亮点指标」显示：保费收入612,265百万元（约6122.65亿元）；"
            "归属于母公司股东的净利润50,268百万元（约502.68亿元）；"
            "内含价值1,072,140百万元（约10721.40亿元）；一年新业务价值58,373百万元（约583.73亿元）；"
            "总资产4,252,410百万元，总投资收益198,596百万元。"
        ),
        "answer_keywords": ["612,265", "50,268", "1,072,140", "58,373"],
        "need_table": False,
        "difficulty": "中",
        "top_k": 5,
        "sample_source": None,
        "design_note": "扩充题，补上**保险行业**覆盖。"
                       "「经营亮点指标」数字被拆成多行文本（数值与单位分行），"
                       "是检验 PDF 文本层拼接顺序是否错乱的典型样本。",
    })

    # ---------------- 用例 8（扩充：银行 · 事实型）----------------
    cases.append({
        "id": "Q08",
        "question": "邮储银行2019年末的资产质量指标表现如何？"
                    "其经营理念和差异化的市场定位是什么？",
        "question_type": "事实型",
        "origin": "基于年报内容扩充（资产质量 + 经营理念）",
        "reference_doc": "邮储银行2019年报",
        "reference_docs": ["邮储银行2019年报"],
        "expected_pages": [13, 18],
        "expected_answer_points": [
            "2019年末不良贷款率0.86%，与2018年末持平（2017年末0.75%）",
            "2019年末拨备覆盖率389.45%（2018年末346.80%、2017年末324.77%），贷款拨备率3.35%",
            "2019年实现净利润610.36亿元，同比增长16.52%，净资产收益率13.10%，净利差2.45%、净利息收益率2.50%",
            "经营理念为「普之城乡，惠之于民」，在提供普惠金融服务、发展绿色金融、支持精准扶贫等方面履行社会责任",
        ],
        "ground_truth": (
            "邮储银行2019年年度报告显示：2019年末不良贷款率0.86%，与2018年末持平，"
            "2017年末为0.75%；拨备覆盖率389.45%，较2018年末的346.80%、2017年末的324.77%持续提升，"
            "贷款拨备率3.35%。2019年实现净利润610.36亿元，同比增长16.52%，净资产收益率13.10%，"
            "净利差2.45%，净利息收益率2.50%，继续保持行业领先水平；"
            "惠誉、穆迪、标普三大国际评级机构继续给予中国银行业最高的信用评级。"
            "本行坚持「普之城乡，惠之于民」的经营理念，在提供普惠金融服务、发展绿色金融、"
            "支持精准扶贫等方面积极履行社会责任，坚持风险为本，树立「全面、全程、全员」的风险管理理念。"
        ),
        "answer_keywords": ["0.86", "389.45", "610.36", "16.52", "普之城乡，惠之于民", "普惠金融"],
        "need_table": True,
        "difficulty": "中",
        "top_k": 6,
        "sample_source": None,
        "design_note": "扩充题。同一份年报里「资产质量」既在指标表（第13页）"
                       "又在董事长致辞（第18页）出现，两处数值口径不同（一个是比率、一个是绝对额），"
                       "用于检验多片段证据的**一致性核对**能力。",
    })

    # ---------------- 用例 9（扩充：证券 · 数值型）----------------
    cases.append({
        "id": "Q09",
        "question": "中信证券2020年实现营业收入和归属于母公司股东的净利润分别是多少？"
                    "较2019年的同比增减幅度如何？",
        "question_type": "数值型",
        "origin": "基于年报内容扩充（主要会计数据）",
        "reference_doc": "中信证券2020年报",
        "reference_docs": ["中信证券2020年报"],
        "expected_pages": [11],
        "expected_answer_points": [
            "2020年营业收入54,382,730,241.56元（约543.83亿元），同比增长26.06%",
            "2020年归属于母公司股东的净利润14,902,324,215.75元（约149.02亿元），同比增长21.86%",
            "2019年营业收入431.40亿元、归母净利润122.29亿元",
        ],
        "ground_truth": (
            "中信证券2020年年度报告「主要会计数据」显示：2020年营业收入54,382,730,241.56元"
            "（约543.83亿元），较2019年的43,139,697,642.01元同比增长26.06%；"
            "2020年归属于母公司股东的净利润14,902,324,215.75元（约149.02亿元），"
            "较2019年的12,228,609,723.82元同比增长21.86%。"
        ),
        "answer_keywords": ["54,382,730,241.56", "14,902,324,215.75", "26.06", "21.86"],
        "need_table": True,
        "difficulty": "难",
        "origin_note": "金额为「元」而非「亿元」，单位换算是主要出错点",
        "top_k": 5,
        "sample_source": None,
        "design_note": "扩充题，补上**证券行业**与「超长数字」场景。"
                       "参考答案是 20 位精度的金额（单位：元），"
                       "用于检验 LLM 是否会擅自换算单位或截断数字——这是金融问答的高频事故点。",
    })

    # ---------------- 用例 10（扩充：多文档对比 · 归纳型）----------------
    cases.append({
        "id": "Q10",
        "question": "对比国泰君安与招商证券2021年年报，两家证券公司在经营业绩"
                    "和数字化转型战略上有哪些共同点与差异？",
        "question_type": "多文档归纳型",
        "origin": "基于年报内容扩充（跨文档对比）",
        "reference_doc": "国泰君安2021年报",
        "reference_docs": ["国泰君安2021年报", "招商证券2021年报"],
        "expected_pages": [4, 22],
        "expected_answer_points": [
            "国泰君安2021年合并口径营业收入428亿元、归母净利润150亿元，同比分别增长22%和35%，ROE 11.05%、较上年上升2.51个百分点",
            "招商证券2021年营业总收入294.29亿元、同比增长21.22%，归母净利润116.45亿元、同比增长22.69%，加权平均净资产收益率11.52%",
            "共同点：经营业绩均创历史新高、营收与净利润增速均在20%以上，均强调数字化转型，均连续14年获得AA级监管评级",
            "差异：国泰君安规模更大并提出「SMART投行」愿景、开发新一代低延时国产化核心交易系统；招商证券以国企改革「双百行动」「揭榜挂帅」「六能」机制为抓手推进变革",
        ],
        "ground_truth": (
            "国泰君安2021年实现合并口径营业收入428亿元、归母净利润150亿元，同比分别增长22%和35%，"
            "ROE为11.05%、较上年上升2.51个百分点；公司聚焦理念、架构、技术三大要素，"
            "擘画「SMART投行」愿景，开发新一代低延时国产化核心交易系统，"
            "加快推进以战略目标驱动、IT与业务融合、组织体系配套、「人的转型」为核心的全面数字化转型，"
            "并连续14年获得A类AA级监管评级。"
            "招商证券2021年实现营业总收入294.29亿元、同比增长21.22%，"
            "归属于上市公司股东的净利润116.45亿元、同比增长22.69%，加权平均净资产收益率11.52%、"
            "同比增长0.67个百分点；公司深入推进国企改革「双百行动」「揭榜挂帅」等改革任务，"
            "「六能」机制建设、核心业务转型、数字化发展等关键变革实现重要突破，"
            "连续14年获得行业分类评价AA评级，并首批入围中国证监会公布的证券公司监管「白名单」。"
            "共同点：两家公司2021年营收与净利润均实现20%以上增长、经营业绩创历史新高，"
            "均把数字化转型作为核心战略，且均连续14年获得AA级监管评级。"
            "差异：国泰君安资产与收入规模更大（营业收入428亿元 vs 294.29亿元），"
            "战略提法上突出「SMART投行」与国产化核心交易系统；"
            "招商证券则更强调国企改革「双百行动」「六能」机制等体制机制变革。"
        ),
        "answer_keywords": ["428", "150", "294.29", "116.45", "22.69", "数字化",
                            "SMART投行", "双百行动"],
        "need_table": False,
        "difficulty": "难",
        "top_k": 10,
        "sample_source": None,
        "design_note": "扩充题，与 Q04 构成「多文档对比」题型对。"
                       "两份文档均为券商年报、章节结构高度相似（都有「经营情况讨论与分析」），"
                       "是检验「章节结构相似导致串公司」问题的最佳样本。",
    })

    # 统一补齐字段，确保 JSON 结构一致
    for c in cases:
        c.setdefault("origin_note", "")
        c["expected_pages"] = c.get("expected_pages", [])
    return cases


# ---------------------------------------------------------------------------
# 四、落盘
# ---------------------------------------------------------------------------
def save_cases(cases: list[dict], samples: list[dict]) -> None:
    out_dir = ensure_results_dir()
    payload = {
        "wo_no": WO_NO,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_pdf": str(config.PDF_SAMPLE_QUESTIONS),
        "n_sample_questions": len(samples),
        "n_cases": len(cases),
        "question_type_distribution": _dist(cases, "question_type"),
        "sector_distribution": _sector_dist(cases),
        "sample_questions": samples,
        "cases": cases,
    }
    (out_dir / "test_cases.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "test_cases.md").write_text(_render_md(payload), encoding="utf-8")


def _dist(cases: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for c in cases:
        out[c[key]] = out.get(c[key], 0) + 1
    return out


def _sector_dist(cases: list[dict]) -> dict:
    out: dict[str, int] = {}
    for c in cases:
        for d in c["reference_docs"]:
            for sector, comps in _SECTOR_OF.items():
                if any(d.startswith(x) for x in comps):
                    out[sector] = out.get(sector, 0) + 1
                    break
    return out


_SECTOR_OF = {
    "银行": ["平安银行", "招商银行", "邮储银行"],
    "保险": ["中国平安", "中国人寿", "中国太保"],
    "证券": ["中信证券", "招商证券", "国泰君安"],
}


def _render_md(payload: dict) -> str:
    cases = payload["cases"]
    lines = [
        f"# 测试用例集（{payload['n_cases']} 个问题）",
        "",
        f"> 工单编号：{WO_NO}　生成时间：{payload['generated_at']}",
        f"> 示例问题来源：`{payload['source_pdf']}`（共 {payload['n_sample_questions']} 个示例问题）",
        "",
        "## 一、用例总览",
        "",
        "| 编号 | 问题（简） | 题型 | 主责文档 | 期望页码 | 难度 | 是否来自示例 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in cases:
        q = c["question"] if len(c["question"]) <= 34 else c["question"][:34] + "…"
        pages = "、".join(str(p) for p in c["expected_pages"]) or "跨多页"
        from_sample = "是" if c["sample_source"] else "否（扩充）"
        lines.append(f"| {c['id']} | {q} | {c['question_type']} | {c['reference_doc']} | "
                     f"{pages} | {c['difficulty']} | {from_sample} |")

    lines += [
        "",
        f"**题型分布**：{payload['question_type_distribution']}",
        "",
        f"**行业覆盖**（按涉及文档计）：{payload['sector_distribution']}",
        "",
        "## 二、用例明细",
        "",
    ]
    for c in cases:
        lines += [
            f"### {c['id']}　{c['question']}",
            "",
            f"- **题型**：{c['question_type']}　**难度**：{c['difficulty']}　"
            f"**top_k**：{c['top_k']}　**是否需表格**：{'是' if c['need_table'] else '否'}",
            f"- **来源**：{c['origin']}",
            f"- **主责文档**：{'、'.join(c['reference_docs'])}",
            f"- **期望页码**：{'、'.join(str(p) for p in c['expected_pages']) or '（跨多页，不限定）'}",
            "",
            "**答案要点**：",
            "",
        ]
        lines += [f"1. {p}" for p in c["expected_answer_points"]]
        lines += [
            "",
            "**参考答案（ground_truth）**：",
            "",
            "> " + c["ground_truth"].replace("\n", "\n> "),
            "",
            f"**自动判准关键词**：`{'` / `'.join(c['answer_keywords'])}`",
            "",
            f"**设计说明**：{c['design_note']}",
            "",
        ]
        if c["sample_source"]:
            lines += [
                "**示例问题出处定位（证据）**：",
                "",
                f"- 判定结果：`{c['sample_source']['doc'] or '多文档'}`",
                f"- 依据：{c['sample_source']['reason']}",
                "",
            ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 五、主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=f"测试用例构建（{WO_NO}）")
    ap.add_argument("--sample-pdf", default=str(config.PDF_SAMPLE_QUESTIONS))
    ap.add_argument("--show", action="store_true", help="打印用例摘要")
    args = ap.parse_args()

    print(f"===== {WO_NO} · 测试用例构建 =====")
    raw_text = extract_sample_text(Path(args.sample_pdf))
    samples = parse_sample_questions(raw_text)
    print(f"[1/3] 从 sample_questions.pdf 解析出 {len(samples)} 个示例问题")

    docs = [d.to_dict() for d in list_ccf_docs()]
    texts = load_report_texts(docs)
    print(f"[2/3] 载入 {len(texts)} 份年报 txt 用于出处取证")

    cases = build_test_cases(samples, texts)
    save_cases(cases, samples)
    print(f"[3/3] 生成 {len(cases)} 个测试用例 → results/test_cases.json / test_cases.md")

    if args.show:
        for c in cases:
            print(f"  {c['id']} [{c['question_type']}/{c['difficulty']}] "
                  f"{c['question'][:44]}…")
    print("\n完成。")


if __name__ == "__main__":
    main()
