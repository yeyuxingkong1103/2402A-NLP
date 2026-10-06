# -*- coding: utf-8 -*-
"""T9 金标准构建：14 题标准答案与证据页 —— **全部逐字回溯到 PDF 原文**。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

产出（唯一权威落盘位置，设计/接口设计.md §6.2 的 ``--golden`` 默认值）：
    ``研发/data/eval/golden_qa.jsonl``          14 题金标准（id 1~4 属 PDF2，其余属 PDF1）
    ``优化/评估结果/golden_verification.json``  逐题 PDF 回溯校验（机器可读）
    ``优化/评估结果/golden_verification.md``    逐题 PDF 回溯校验（人读）

数据来源与纪律：
    * 题面（14 条）唯一来源 = ``测试/测试数据/eval_retrieval_14.jsonl``（T5 冻结题面，**禁止改字**）；
    * id 260/95/33/34/957/793/795/543/531/207：``answer``/``evidence``/``evidence_pages``
      **原样沿用** ``E:\\gao6gongdan\\工单1\\data\\eval\\golden_qa.jsonl``（只读，不得篡改）；
      其中题 207 的 ``evidence`` 是**合成串+编辑注记**（14 题中唯一非逐字原文），
      按 captain 裁定：``evidence`` 保留原值，新增 ``evidence_verbatim`` 填可判定原文句，判定优先用它。
    * id 1~4：答案与证据**由本脚本从 PDF2 页文本切片组装**，逐字可回溯（物理页 21/22/157/158/306）。
    * PDF 文件**自动发现**（``config.discover_pdfs``），再按「第 1 页发行人名称」映射到语料编号，
      **严禁硬编码文件名**（设计/需求分析.md §4.1 / FR-1）。
    * 全部结论以「归一化后子串命中」为唯一判据（``squash`` 去空白 + 全角标点归一），不靠人眼。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/build_golden_qa.py
    pwsh -NoProfile -File run_py.ps1 优化/脚本/build_golden_qa.py --print-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "研发"
sys.path.insert(0, str(DEV_DIR))

from app.core import text_utils  # noqa: E402
from app.core.config import PdfSource, discover_pdfs, get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402

REFERENCE_GOLDEN = Path(r"E:\gao6gongdan\工单1\data\eval\golden_qa.jsonl")   # 只读
QUESTION_SOURCE = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"  # 只读
GOLDEN_OUT = DEV_DIR / "data" / "eval" / "golden_qa.jsonl"
VERIFY_JSON = REPO_ROOT / "优化" / "评估结果" / "golden_verification.json"
VERIFY_MD = REPO_ROOT / "优化" / "评估结果" / "golden_verification.md"

# 语料编号 → 第 1 页必须出现的发行人全称（用于自动发现后映射，非文件名硬编码）
CORPUS_ISSUER = {"pdf1": "武汉兴图新科电子股份有限公司", "pdf2": "武汉力源信息技术股份有限公司"}

# PDF1 十题的 evidence_verbatim 人工候选（按优先级排在 golden evidence 之后尝试；全部仍须机器校验命中）
PDF1_VERBATIM_HINTS: dict[int, list[str]] = {
    260: ["报告期内，公司来自军用领域的收入分别为6,464.51 万元、14,414.16 万元、18,780.67 万元和4,627.14 万元"],
    95: ["公司目前已经成为军队视频指挥领域的重要供应商，参与制定了全军第一个视频指挥系统技术标准",
         "参与制定了全军第一个视频指挥系统技术标准（即2019年制订的《某视频指挥系统技术规范（1.0版）》）",
         "全军第一个视频指挥系统技术标准"],
    33: ["占主营业务收入比重分别为82.10%、97.31%、94.84%和94.34%"],
    34: ["电子信息行业的上游涉及信息系统相关的电子元器件制造企业，以及机箱、机柜等金属壳体制造企业，竞争充分，采购便利。"],
    957: ["兴图新科目前已经成为国防军队视频指挥领域的重要供应商",
          "已经成为军队视频指挥领域的重要供应商"],
    793: ["下游行业为各类终端用户，覆盖范围广泛，主要包括军队、政府机关、能源等行业企业。"],
    795: ["2014年12月，某大型研究所牵头承担的“某情报、指挥、控制与通信网络一体化工程”（即相当于美军的C4ISR系统）荣获国家科技进步一等奖。",
          "荣获国家科技进步一等奖"],
    543: ["注册资本：5,520 万元"],
    531: ["法定代表人：程家明"],
    207: ["拟使用本次发行募集资金15,000 万元用于补充流动资金"],
}

# 题 207 的 golden evidence 是合成串+编辑注记 → 判定不得用它，必须用 evidence_verbatim
EDITORIAL_MARKERS = ("（第", "同载明", "……", "注：", "见第")

# 逐题附加判据（与 设计/验收标准.md §2.1 强制负例 N-1/N-2/N-3、§3.3 负例 N-5/N-8 对齐）
EXTRA_FIELDS: dict[int, dict[str, Any]] = {
    1: {"category": "发行方案", "citation_quote": "1,670 万股，占发行后总股本的比例为25.04%",
        "required_substrings": ["1,670", "25.04"], "forbidden_substrings": [],
        "numeric_signal_expected": True, "subject_expectation": "any", "subject_gate_counted": None},
    2: {"category": "募投项目", "citation_quote": "其他与主营业务相关的营运资金",
        "required_substrings": ["3,393.40", "1,526.38", "2,492.78", "9,000.00", "其他与主营业务相关的营运资金"],
        # 负例口径：第 5 项计划总投资原文是占位符 [ ◆ ] → 不得当作 0（禁用串刻意避开「9,000.00 万元」的尾串）
        "forbidden_substrings": ["营运资金 0", "营运资金0", "营运资金为零"],
        "numeric_signal_expected": False, "subject_expectation": "any", "subject_gate_counted": None},
    3: {"category": "关联方", "citation_quote": "赵马克 42.35% 公司控股股东",
        "required_substrings": ["赵马克", "42.35"], "forbidden_substrings": ["融冰投资", "武汉博润", "上海博润",
                                                                     "听音投资", "联众聚源", "力源贸易", "普芯达"],
        "numeric_signal_expected": True, "subject_expectation": "person", "subject_gate_counted": True},
    4: {"category": "关联方", "citation_quote": "融冰投资",
        "required_substrings": ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达"],
        "forbidden_substrings": ["赵马克"],
        "numeric_signal_expected": False, "subject_expectation": "organization", "subject_gate_counted": True},
    260: {"category": "收入金额", "citation_quote": "6,464.51 万元",
          "required_substrings": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"], "forbidden_substrings": [],
          "numeric_signal_expected": True, "subject_expectation": "any", "subject_gate_counted": None},
    95: {"category": "技术标准", "citation_quote": "全军第一个视频指挥系统技术标准",
         "required_substrings": ["视频指挥系统技术标准"], "forbidden_substrings": [],
         "numeric_signal_expected": False, "subject_expectation": "any", "subject_gate_counted": None},
    33: {"category": "收入占比", "citation_quote": "82.10%、97.31%、94.84%和94.34%",
         "required_substrings": ["82.10%", "97.31%", "94.84%", "94.34%"], "forbidden_substrings": [],
         "numeric_signal_expected": True, "subject_expectation": "any", "subject_gate_counted": None},
    34: {"category": "上下游", "citation_quote": "电子元器件制造企业，以及机箱、机柜等金属壳体制造企业",
         "required_substrings": ["电子元器件制造企业", "金属壳体制造企业"], "forbidden_substrings": ["不清楚"],
         "numeric_signal_expected": False, "subject_expectation": "any", "subject_gate_counted": False},
    957: {"category": "行业地位", "citation_quote": "国防军队视频指挥领域的重要供应商",
          "required_substrings": ["视频指挥"], "forbidden_substrings": ["不清楚"],
          "numeric_signal_expected": False, "subject_expectation": "any", "subject_gate_counted": None},
    793: {"category": "上下游", "citation_quote": "下游行业为各类终端用户",
          "required_substrings": ["军队", "政府机关", "能源"], "forbidden_substrings": ["不清楚"],
          "numeric_signal_expected": False, "subject_expectation": "any", "subject_gate_counted": False},
    795: {"category": "荣誉奖项", "citation_quote": "国家科技进步一等奖",
          "required_substrings": ["2014", "国家科技进步一等奖"], "forbidden_substrings": ["不清楚"],
          "numeric_signal_expected": False, "subject_expectation": "any", "subject_gate_counted": None},
    543: {"category": "注册资本", "citation_quote": "注册资本：5,520 万元",
          "required_substrings": ["5,520"], "forbidden_substrings": [],
          "numeric_signal_expected": True, "subject_expectation": "any", "subject_gate_counted": None},
    531: {"category": "法定代表人", "citation_quote": "法定代表人：程家明",
          "required_substrings": ["程家明"], "forbidden_substrings": [],
          "numeric_signal_expected": False, "subject_expectation": "person", "subject_gate_counted": None},
    207: {"category": "募资用途", "citation_quote": "15,000 万元用于补充流动资金",
          "required_substrings": ["15,000"], "forbidden_substrings": ["300 万美元", "汉口银行"],
          "numeric_signal_expected": True, "subject_expectation": "any", "subject_gate_counted": None},
}

# PDF2 四题从原文切片的锚点（物理页号 1-based；锚点字符串必须逐字出现在页文本里，否则脚本报错退出）
PDF2_ANCHORS: dict[int, dict[str, Any]] = {
    1: {"page": 22, "anchor": "发行股数", "mode": "next_nonempty", "count": 1,
        "must_contain": ["1,670", "25.04"]},
    2: {"page": 22, "anchor": "本次募集资金拟投资以下项目：", "mode": "block",
        "stop": "本次股票发行所募集资金净额", "must_contain": ["仓储及物流中心", "研发中心", "电子商务平台",
                                                        "扩充产品种类和数量", "其他与主营业务相关的营运资金",
                                                        "3,393.40", "1,526.38", "2,492.78", "9,000.00"]},
    3: {"page": 157, "anchor": "1、存在控制关系的关联方", "mode": "next_nonempty", "count": 6,
        "must_contain": ["赵马克", "42.35%", "公司控股股东"],
        "alt": {"page": 21, "anchor": "赵马克(Mark Zhao)先生持有本公司股份", "mode": "block",
                "stop": "赵马克(Mark Zhao)，美国籍", "must_contain": ["2,117.70", "42.35%"]}},
    4: {"page": 157, "anchor": "2、不存在控制关系的关联方", "mode": "block", "stop": "3、报告期内曾为关联方",
        "must_contain": ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达",
                         "企业名称", "与本公司关系"]},
}

PDF2_DECLARED_PAGES: dict[int, list[int]] = {1: [22, 24], 2: [22, 30, 306, 144], 3: [157, 21, 158], 4: [157, 158]}
PDF2_ANSWER_TEXT: dict[int, str] = {
    1: "发行 1,670 万股，占发行后总股本的比例为 25.04%。",
    2: ("5 项：仓储及物流中心 3,393.40 万元、研发中心 1,526.38 万元、电子商务平台 2,492.78 万元、"
        "扩充产品种类和数量 9,000.00 万元、其他与主营业务相关的营运资金（原文为 [ ◆ ] → 未披露）。"),
    3: "赵马克（Mark Zhao），持股 42.35%，为公司控股股东及实际控制人。",
    4: ("7 家企业：融冰投资、武汉博润、上海博润、听音投资、联众聚源（均持有公司股份 5% 以上的股东）、"
        "力源贸易（同一实际控制人控制的企业）、普芯达（实际控制人近亲属控制的公司）。"),
}
PDF2_ANSWER_SOURCE: dict[int, str] = {
    1: "PDF2 物理 22「四、本次发行情况」表行「发行股数｜1,670 万股，占发行后总股本的比例为25.04%」逐字（数值与比例未改动）",
    2: "PDF2 物理 22「五、募集资金用途」表逐行逐字（5 个项目名与 4 个金额）；第 5 项计划总投资原文为占位符 [ ◆ ]",
    3: "PDF2 物理 157 表 1「关联方名称/持股比例/与本公司关系」行 + 物理 21 正文「占本公司总股本的42.35%，为本公司控股股东及实际控制人」",
    4: "PDF2 物理 157 表 2「企业名称/与本公司关系」7 行逐字（关系描述未改写）",
}

PUNCT_SQUASH = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"


def squash(text: str) -> str:
    """去空白 + 全角标点归一（判定「逐字命中」的唯一归一化口径）。"""
    body = str(text or "")
    for old, new in (("，", ","), ("（", "("), ("）", ")"), ("％", "%"), ("；", ";"), ("：", ":"),
                     (" ", ""), ("\u3000", ""), ("\n", ""), ("\r", ""), ("\t", "")):
        body = body.replace(old, new)
    return body.strip()


def sha256_16(path: Path) -> str:
    """文件 sha256 前 16 位（十六进制大写；用于「读了哪一版」留痕）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()[:16]


def page_lines(doc: Any, page: int) -> list[str]:
    """取「1-based 物理页」的文本行（去行尾空白，保留空行结构）。"""
    raw = doc[int(page) - 1].get_text()
    return [line.rstrip() for line in raw.splitlines()]


def page_text_map(doc: Any) -> dict[int, str]:
    """整本 PDF 的「物理页 → 页文本」映射（只读，用于全库回溯搜索）。"""
    return {index + 1: doc[index].get_text() for index in range(doc.page_count)}


def find_pages(texts: dict[int, str], needle: str) -> list[int]:
    """返回归一化后包含 ``needle`` 的物理页号列表（升序）。"""
    target = squash(needle)
    if not target:
        return []
    return sorted(page for page, text in texts.items() if target in squash(text))


def slice_facts(doc: Any, spec: dict[str, Any], *, log: Any) -> tuple[str, dict[str, Any]]:
    """按锚点从页文本切片，返回 ``(evidence_verbatim, 切片元数据)``。

    逐字（去行尾空白）拼接，不做任何改写；``must_contain`` 全部命中才算通过，
    否则抛 ``ValueError``（不做静默降级）。
    """
    page = int(spec["page"])
    lines = page_lines(doc, page)
    anchor = str(spec["anchor"])
    hits = [index for index, line in enumerate(lines) if anchor in line]
    if not hits:
        raise ValueError(f"PDF 物理 {page} 未找到锚点：{anchor!r}（自动发现结果为空 → 金标准无法构建）")
    start = hits[0]
    if spec["mode"] == "next_nonempty":
        picked: list[str] = []
        cursor = start + 1
        while cursor < len(lines) and len(picked) < int(spec["count"]):
            if lines[cursor].strip():
                picked.append(lines[cursor].strip())
            cursor += 1
    else:
        stop = str(spec["stop"])
        picked = []
        for line in lines[start + 1:]:
            if stop and stop in line:
                break
            if line.strip():
                picked.append(line.strip())
    verbatim = " ".join(picked)
    missing = [item for item in spec.get("must_contain", []) if squash(item) not in squash(verbatim)]
    if missing:
        raise ValueError(f"PDF 物理 {page} 切片缺关键片段 {missing}（锚点 {anchor!r}）")
    meta = {"page": page, "anchor": anchor, "mode": spec["mode"], "lines": len(picked),
            "chars": len(verbatim)}
    log.log_event("golden.slice", page=page, anchor=anchor, lines=len(picked), chars=len(verbatim))
    return verbatim, meta


def load_questions(*, log: Any) -> tuple[dict[int, dict[str, Any]], str]:
    """读取 14 题冻结题面（唯一来源 = 测试/测试数据/eval_retrieval_14.jsonl）。"""
    with log.enter("load_questions", {"path": str(QUESTION_SOURCE)}) as span:
        records = [json.loads(line) for line in QUESTION_SOURCE.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        out = {int(item["id"]): item for item in records}
        if len(out) != 14:
            raise ValueError(f"题面文件应有 14 题，实际 {len(out)} 题：{sorted(out)}")
        digest = sha256_16(QUESTION_SOURCE)
        span.set_output({"count": len(out), "sha256_16": digest})
        return out, digest


def load_reference(*, log: Any) -> tuple[dict[int, dict[str, Any]], str]:
    """只读读取工单1 金标准（10 题）——**绝不写入**。"""
    with log.enter("load_reference", {"path": str(REFERENCE_GOLDEN)}) as span:
        records = [json.loads(line) for line in REFERENCE_GOLDEN.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        out = {int(item["id"]): item for item in records}
        digest = sha256_16(REFERENCE_GOLDEN)
        span.set_output({"count": len(out), "sha256_16": digest, "ids": sorted(out)})
        return out, digest


def resolve_corpus(*, log: Any) -> tuple[dict[str, PdfSource], dict[str, int]]:
    """自动发现 PDF 并按「第 1 页发行人全称」映射到语料编号（**不硬编码文件名**）。"""
    with log.enter("resolve_corpus", {}) as span:
        import pymupdf

        sources = discover_pdfs(logger=log)
        if len(sources) < 2:
            raise ValueError(f"data/raw 至少需 2 份 PDF，实际发现 {len(sources)}：{[s.file_name for s in sources]}")
        mapping: dict[str, PdfSource] = {}
        for source in sources:
            doc = pymupdf.open(str(source.path))
            try:
                head = "".join(doc[index].get_text() for index in range(min(2, doc.page_count)))
            finally:
                doc.close()
            for corpus, issuer in CORPUS_ISSUER.items():
                if issuer in head and corpus not in mapping:
                    mapping[corpus] = source
        missing = sorted(set(CORPUS_ISSUER) - set(mapping))
        if missing:
            raise ValueError(f"无法由第 1 页发行人名称映射语料编号：{missing}")
        page_counts = {corpus: int(source.page_count or 0) for corpus, source in mapping.items()}
        span.set_output({"files": {c: s.file_name for c, s in mapping.items()}, "pages": page_counts})
        return mapping, page_counts


def resolve_verbatim(record: dict[str, Any], questions: dict[int, dict[str, Any]],
                     texts: dict[int, str], *, log: Any) -> dict[str, Any]:
    """为 PDF1 十题确定 ``evidence_verbatim``：候选按优先级取「第一个在 PDF 中逐字命中」的。"""
    qid = int(record["id"])
    evidence = str(record.get("evidence") or "")
    candidates: list[tuple[str, str]] = []
    if evidence and not any(marker in evidence for marker in EDITORIAL_MARKERS):
        candidates.append(("golden_evidence", evidence))
    for hint in PDF1_VERBATIM_HINTS.get(qid, []):
        candidates.append(("hint", hint))
    for line in evidence.splitlines():
        if squash(line) and len(squash(line)) >= 12:
            candidates.append(("golden_evidence_line", line.strip()))
    for source, candidate in candidates:
        pages = find_pages(texts, candidate)
        if pages:
            log.log_event("golden.verbatim_found", qid=qid, candidate_source=source, pages=pages,
                          chars=len(candidate))
            return {"evidence_verbatim": candidate, "candidate_source": source, "verbatim_pages": pages}
    raise ValueError(f"题 {qid} 的 evidence_verbatim 在 PDF 中找不到任何逐字候选（golden evidence 与候选提示均未命中）")


def build_pdf1_records(reference: dict[int, dict[str, Any]], questions: dict[int, dict[str, Any]],
                       texts: dict[int, str], *, log: Any) -> list[dict[str, Any]]:
    """构建 PDF1 十题金标准记录（answer/evidence/evidence_pages **原样沿用**工单1）。"""
    with log.enter("build_pdf1_records", {"count": len(reference)}) as span:
        out: list[dict[str, Any]] = []
        for qid in sorted(reference):
            ref = reference[qid]
            question = str(questions[qid]["question"])
            if squash(question) != squash(str(ref["question"])):
                raise ValueError(f"题 {qid} 题面与工单1 不一致：\n  T5 : {question}\n  工单1: {ref['question']}")
            found = resolve_verbatim(ref, questions, texts, log=log)
            declared = [int(page) for page in ref.get("evidence_pages") or []]
            extra = dict(EXTRA_FIELDS[qid])
            record = {
                "id": qid,
                "question": question,
                "answer": str(ref["answer"]),
                "evidence": str(ref["evidence"]),
                "evidence_verbatim": found["evidence_verbatim"],
                "evidence_verbatim_source": f"PDF1 逐字命中（候选来源 {found['candidate_source']}）",
                "evidence_pages": declared,
                "corpus": "pdf1",
                "category": str(ref.get("category") or extra["category"]),
                "should_be_unknown": bool(ref.get("should_be_unknown", False)),
                "citation_quote": extra["citation_quote"],
                "required_substrings": extra["required_substrings"],
                "forbidden_substrings": extra["forbidden_substrings"],
                "numeric_signal_expected": extra["numeric_signal_expected"],
                "subject_expectation": extra["subject_expectation"],
                "subject_gate_counted": extra["subject_gate_counted"],
                "min_hit_top_k": 5,
                "max_first_token_ms": 3000,
                "question_source": f"测试/测试数据/eval_retrieval_14.jsonl（T5 冻结题面，sha256_16={{qs}}）",
                "answer_source": "E:\\gao6gongdan\\工单1\\data\\eval\\golden_qa.jsonl（只读基准，answer 原样）",
                "pdf_verification": {"verbatim_pages": found["verbatim_pages"], "declared_pages": declared},
            }
            out.append(record)
        span.set_output({"built": len(out)})
        return out


def build_pdf2_records(questions: dict[int, dict[str, Any]], doc: Any, page_count: int, *,
                       log: Any) -> list[dict[str, Any]]:
    """构建 PDF2 四题金标准记录（证据逐字切片自 PDF2 物理页 21/22/157/158/306）。"""
    with log.enter("build_pdf2_records", {"page_count": page_count}) as span:
        out: list[dict[str, Any]] = []
        for qid in (1, 2, 3, 4):
            spec = PDF2_ANCHORS[qid]
            verbatim, meta = slice_facts(doc, spec, log=log)
            alt_text, alt_meta = "", None
            if spec.get("alt"):
                alt_text, alt_meta = slice_facts(doc, spec["alt"], log=log)
            extra = dict(EXTRA_FIELDS[qid])
            declared = [int(page) for page in PDF2_DECLARED_PAGES[qid]]
            for page in declared:
                if not 1 <= page <= page_count:
                    raise ValueError(f"题 {qid} 声明证据页 {page} 越界（PDF2 共 {page_count} 页）")
            record = {
                "id": qid,
                "question": str(questions[qid]["question"]),
                "answer": PDF2_ANSWER_TEXT[qid],
                "evidence": verbatim,
                "evidence_verbatim": verbatim,
                "evidence_verbatim_source": f"PDF2 物理 {meta['page']} 锚点「{meta['anchor']}」切片（{meta['lines']} 行逐字）",
                "evidence_verbatim_alt": alt_text,
                "evidence_pages": declared,
                "corpus": "pdf2",
                "category": extra["category"],
                "should_be_unknown": False,
                "citation_quote": extra["citation_quote"],
                "required_substrings": extra["required_substrings"],
                "forbidden_substrings": extra["forbidden_substrings"],
                "numeric_signal_expected": extra["numeric_signal_expected"],
                "subject_expectation": extra["subject_expectation"],
                "subject_gate_counted": extra["subject_gate_counted"],
                "min_hit_top_k": 5,
                "max_first_token_ms": 3000,
                "question_source": f"测试/测试数据/eval_retrieval_14.jsonl（T5 冻结题面，sha256_16={{qs}}）",
                "answer_source": PDF2_ANSWER_SOURCE[qid],
                "pdf_verification": {"verbatim_pages": [int(meta["page"])], "declared_pages": declared,
                                     "alt_page": (None if alt_meta is None else int(alt_meta["page"]))},
            }
            out.append(record)
        span.set_output({"built": len(out)})
        return out


def verify_records(records: list[dict[str, Any]], text_maps: dict[str, dict[int, str]], *, log: Any) -> dict[str, Any]:
    """逐题回溯校验：``evidence_verbatim`` 是否真的能在 PDF 逐字命中，且声明证据页是否承载本题事实。

    两条判据（任一成立即视为「声明证据页有效」，避免把「允许引用页 ≠ evidence_pages」误判为错误）：
        ① 逐字句本身出现在某个声明证据页上；
        ② 某个声明证据页同时含该题全部 ``required_substrings``（事实级回溯，如题 531/543 的
           「第二节 概览」表格页 22 是「注册资本 5,520.00 万元 / 法定代表人 程家明」单元格形式）。
    **硬门槛**：``evidence_verbatim`` 必须能在该 PDF 中逐字命中，否则抛错（金标准不得无原文支撑）。
    """
    with log.enter("verify_records", {"count": len(records)}) as span:
        rows: list[dict[str, Any]] = []
        for record in records:
            corpus = str(record["corpus"])
            texts = text_maps[corpus]
            verbatim = str(record["evidence_verbatim"])
            pages = find_pages(texts, verbatim)
            if not pages:
                raise ValueError(f"题 {record['id']} 的 evidence_verbatim 不在 {corpus} 任何页中（金标准无原文支撑）")
            declared = [int(page) for page in record["evidence_pages"]]
            on_declared = [page for page in declared if page in pages]
            required = list(record.get("required_substrings") or [])
            page_fact = {page: sorted(item for item in required if squash(item) in squash(texts.get(page, "")))
                         for page in declared}
            fact_pages = [page for page, hits in page_fact.items() if required and len(hits) == len(required)]
            ok_declared = bool(on_declared) or bool(fact_pages)
            if on_declared:
                reason = "逐字句命中声明页"
            elif fact_pages:
                reason = f"声明页以事实级形式承载本题（命中页 {fact_pages}）"
            else:
                reason = "声明页既无逐字句也无全部 required_substrings"
            row = {
                "id": int(record["id"]), "corpus": corpus,
                "evidence_pages_declared": declared,
                "evidence_verbatim_pages": pages,
                "verbatim_on_declared_page": on_declared,
                "declared_page_fact_hits": page_fact,
                "fact_level_declared_pages": fact_pages,
                "verbatim_chars": len(verbatim),
                "ok_verbatim_found": True,
                "ok_on_declared": ok_declared,
                "ok_on_declared_reason": reason,
            }
            rows.append(row)
            log.log_event("golden.verify", qid=row["id"], corpus=corpus, verbatim_pages=pages,
                          declared=declared, on_declared=on_declared, fact_pages=fact_pages,
                          ok=bool(ok_declared), reason=reason)
        bad = [row["id"] for row in rows if not row["ok_on_declared"]]
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total": len(rows), "verbatim_found_in_pdf": sum(1 for row in rows if row["ok_verbatim_found"]),
            "verbatim_on_declared_page": sum(1 for row in rows if row["ok_on_declared"]),
            "declared_page_mismatch_ids": bad,
            "rows": rows,
        }
        span.set_output({"total": len(rows), "declared_page_mismatch_ids": bad})
        return payload


def stamp_pages(records: list[dict[str, Any]], verification: dict[str, Any], *, log: Any) -> None:
    """把「逐字句实际命中页」与「声明页 ∪ 逐字页」回写到每条金标准记录（便于逐题核对）。"""
    with log.enter("stamp_pages", {"records": len(records)}) as span:
        by_id = {int(row["id"]): row for row in verification["rows"]}
        for record in records:
            row = by_id[int(record["id"])]
            declared = [int(page) for page in record["evidence_pages"]]
            verbatim_pages = [int(page) for page in row["evidence_verbatim_pages"]]
            record["evidence_verbatim_pages"] = verbatim_pages
            record["evidence_pages_union"] = sorted(set(declared) | set(verbatim_pages))
            record["evidence_pages_note"] = str(row["ok_on_declared_reason"])
        note = ("题 207：``evidence`` 与 ``evidence_pages`` 原样沿用只读基准（工单1），**不得篡改**；"
                "但该 evidence 是合成串+编辑注记（14 题唯一非逐字原文）→ 召回命中判定一律以 "
                "``evidence_verbatim``（PDF1 物理 490 逐字句）为准；主锚点 479 的表格行「补充流动资金 15,000.00」"
                "见 ``evidence_pages`` 第 2 项。")
        for record in records:
            if int(record["id"]) == 207:
                record["evidence_verbatim_note"] = note
        span.set_output({"stamped": len(records)})


def write_outputs(records: list[dict[str, Any]], verification: dict[str, Any],
                  question_sha: str, reference_sha: str, *, log: Any) -> dict[str, Any]:
    """落盘金标准 JSONL 与校验报告（JSON + Markdown）。"""
    with log.enter("write_outputs", {"golden": str(GOLDEN_OUT), "records": len(records)}) as span:
        GOLDEN_OUT.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for record in sorted(records, key=lambda item: int(item["id"])):
            payload = dict(record)
            payload["question_source"] = str(payload["question_source"]).replace("{qs}", question_sha)
            lines.append(json.dumps(payload, ensure_ascii=False))
        GOLDEN_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

        verification = dict(verification)
        verification.update({
            "golden_path": str(GOLDEN_OUT.relative_to(REPO_ROOT)).replace("\\", "/"),
            "golden_sha256_16": sha256_16(GOLDEN_OUT),
            "question_source_sha256_16": question_sha,
            "reference_golden_sha256_16": reference_sha,
            "work_order": WORK_ORDER,
        })
        VERIFY_JSON.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        md = ["# T9 金标准 PDF 回溯校验（14 题）", "",
              f"> 工单：{WORK_ORDER}",
              f"> 生成时间：{verification['generated_at']}　"
              f"金标准：`{verification['golden_path']}`（sha256_16 `{verification['golden_sha256_16']}`）", "",
              f"- 逐字证据可在 PDF 中命中：**{verification['verbatim_found_in_pdf']}/14**",
              f"- 逐字证据命中「声明证据页」的题：**{verification['verbatim_on_declared_page']}/14**"
              f"（不一致 id：{verification['declared_page_mismatch_ids'] or '无'}）", "",
              "| id | 语料 | 声明证据页 | evidence_verbatim 实际命中页 | 声明页内逐字命中 | 事实级承载页 | 判定依据 | 逐字证据字数 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in verification["rows"]:
            md.append(f"| {row['id']} | {row['corpus']} | {row['evidence_pages_declared']} | "
                      f"{row['evidence_verbatim_pages']} | {row['verbatim_on_declared_page']} | "
                      f"{row['fact_level_declared_pages']} | {row['ok_on_declared_reason']} | "
                      f"{row['verbatim_chars']} |")
        md += ["", "## 说明", "",
               "- 判据：把页文本与 `evidence_verbatim` 同时做「去空白 + 全角标点归一」后做子串包含判定，"
               "不靠人眼比对（唯一实现见 `优化/脚本/build_golden_qa.py::find_pages`）。",
               "- `evidence_verbatim` 的选取优先级：golden `evidence`（无编辑注记时）→ 逐题候选提示 → "
               "`evidence` 的单行切片；取**第一个在 PDF 中逐字命中**的候选，并把实际命中页写回校验表。",
               "- 声明证据页有效 = ①逐字句落在该页，**或** ②该页同时含本题全部 `required_substrings`（事实级）。"
               "题 531/543 即属 ②：声明页 22（「第二节 概览」）是表格单元形式"
               "（`注册资本 5,520.00 万元` / `法定代表人 程家明`），带冒号形式的逐字句在物理 52；"
               "两页都真实承载本题事实，与验收标准「允许引用页 ≠ evidence_pages」一致。",
               "- 题 207 的 golden `evidence` 是合成串 + 编辑注记（14 题中唯一非逐字原文，源自工单1 只读基准，"
               "**不得篡改**）→ 本表以其 `evidence_verbatim`（PDF1 物理 490 逐字句）为准；"
               "主锚点物理 479 的表格行「补充流动资金 15,000.00」另由声明页承载（事实级命中）。", ""]
        VERIFY_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
        span.set_output({"golden_bytes": GOLDEN_OUT.stat().st_size, "md_bytes": VERIFY_MD.stat().st_size})
        return {"golden_bytes": GOLDEN_OUT.stat().st_size, "golden_sha256_16": verification["golden_sha256_16"]}


def main(argv: Sequence[str] | None = None) -> int:
    """入口：构建金标准 → 逐题回溯校验 → 落盘（退出码 0 成功 / 2 入参或校验失败）。"""
    parser = argparse.ArgumentParser(description="T9 金标准构建（14 题，逐字回溯 PDF 原文）")
    parser.add_argument("--print-only", action="store_true", help="只打印校验结果，不落盘")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("build_golden_qa")
    started = time.perf_counter()
    with log.enter("main", {"print_only": args.print_only, "cwd": str(Path.cwd())}) as span:
        import pymupdf

        questions, question_sha = load_questions(log=log)
        reference, reference_sha = load_reference(log=log)
        mapping, page_counts = resolve_corpus(log=log)
        text_maps: dict[str, dict[int, str]] = {}
        docs: dict[str, Any] = {}
        for corpus, source in mapping.items():
            doc = pymupdf.open(str(source.path))
            docs[corpus] = doc
            text_maps[corpus] = page_text_map(doc)
            log.log_event("golden.corpus_loaded", corpus=corpus, file_name=source.file_name,
                          pages=int(source.page_count or 0), sha256_16=source.sha256_16)
        try:
            records = build_pdf1_records(reference, questions, text_maps["pdf1"], log=log)
            records += build_pdf2_records(questions, docs["pdf2"], page_counts["pdf2"], log=log)
        finally:
            for doc in docs.values():
                doc.close()
        records.sort(key=lambda item: int(item["id"]))
        missing_ids = sorted(set(questions) - {int(item["id"]) for item in records})
        if missing_ids:
            raise ValueError(f"金标准缺题：{missing_ids}")
        verification = verify_records(records, text_maps, log=log)
        stamp_pages(records, verification, log=log)
        if args.print_only:
            span.set_output({"print_only": True, "verified": verification["verbatim_found_in_pdf"]})
            print(json.dumps({k: v for k, v in verification.items() if k != "rows"}, ensure_ascii=False, indent=2))
            for row in verification["rows"]:
                print(f"  id={row['id']:<4} {row['corpus']} 声明={row['evidence_pages_declared']} "
                      f"逐字命中页={row['evidence_verbatim_pages']} 声明页内={row['verbatim_on_declared_page']}")
            shutdown_logging()
            return 0
        result = write_outputs(records, verification, question_sha, reference_sha, log=log)
        print(f"✅ 金标准已落盘：{GOLDEN_OUT}（{result['golden_bytes']} B / sha256_16 {result['golden_sha256_16']}）")
        print(f"✅ 校验报告：{VERIFY_JSON.relative_to(REPO_ROOT)} / {VERIFY_MD.relative_to(REPO_ROOT)}")
        print(f"   逐字证据命中 PDF：{verification['verbatim_found_in_pdf']}/14；"
              f"命中声明证据页：{verification['verbatim_on_declared_page']}/14；"
              f"不一致 id：{verification['declared_page_mismatch_ids'] or '无'}")
        print(f"   耗时 {round((time.perf_counter() - started) * 1000, 2)} ms")
        span.set_output({"golden_sha256_16": result["golden_sha256_16"], "ok": True})
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底：结构化日志 + 非零退出，绝不静默
        try:
            get_logger("build_golden_qa").log_event("golden.failed", level="ERROR",
                                                    error_type=type(exc).__name__, message=str(exc),
                                                    stack=__import__("traceback").format_exc())
        finally:
            import traceback

            traceback.print_exc()
        raise SystemExit(2)
