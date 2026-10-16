# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/common/golden.py
作者：测试工程师（T3）

用途：**16 题 golden 测试集的唯一判分库**。提供：
  1. ``normalize``  —— 中文/数值归一化（全角→半角、去千分位逗号、去空白、百分之→%）
  2. ``load_golden`` —— 读取冻结的 golden（研发/data/eval/golden_qa.jsonl 或 测试/测试数据/golden_qa_16.jsonl）
  3. ``judge_answer`` —— **确定性**判对/判错（关键字 + 数值等价 + 必含正则 + 禁止误答项 + 否定护栏）
  4. ``check_retrieval_hit`` —— **检索命中**判定（证据原文是否落在返回 chunk 中，与「回答判对」严格分离）
  5. ``validate_record_shape`` —— golden 记录字段契约校验（供离线测试使用）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
判分口径（与 设计/验收标准.md §3.4 一致，测试报告必须照抄这段）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **判对 = 确定性关键字判定**，不引用任何 LLM 自评：
     判对 ⟺  (全部 ``answer_keywords`` 命中)  ∧  (全部 ``answer_keywords_any_of`` 组各命中其一)
            ∧  (全部 ``answer_regex_required`` 命中)  ∧  (无 ``forbidden_keywords``/``forbidden_regex`` 命中)
2. **数值归一化**：去千分位逗号、全角转半角、去全部空白、``百分之X`` → ``X%``。
   带单位的数值按**基本单位**比较：1 亿元 = 10,000 万元 = 100,000,000 元；1 万股 = 10,000 股。
   故 ``6,464.51`` 与 ``6464.51`` 等价、``82.10%`` 与 ``82.1%`` 等价、``1.5亿元`` 与 ``15,000.00万元`` 等价。
   数值关键词**必须带单位**（万元/亿元/元/万股/股/%/倍）才走等价比较，防止「数字碰巧相同」误判。
3. **否定护栏**：禁止误答项若出现在否定语境中（命中点前后各 ``GUARD_WINDOW`` 字窗口内出现
   ``NEGATION_CUES`` 中**任一**否定词），**不判为违规**。
   ⚠️ 触发词全量以本模块常量 ``NEGATION_CUES`` 为**唯一实现来源**（勿在文档里另行罗列，以免文档比实现窄）。
4. **「检索命中」≠「回答判对」**，分别报告，互不代替：
     检索命中 ⟺ ``retrieval_evidence`` 中的**证据原文**（归一化后）落在**返回的 chunk** 中；
     **禁止**用「引用页码 == evidence_pages」冒充命中（会把无关页误计为命中、虚高）。
5. 图像题（id 5 框图 / id 6 图表）的检索证据来自**图像通道**：
   - id 5：页 39 框图文本层（列序连续，T1 已证需按 x 列聚类重建）；
   - id 6：页 72 条形图 xref 1692 的**行业-增长率配对**（像素级抽取/视觉复核，本机无 OCR）。
   配对命中 = 同一 chunk 内同时出现行业名与其增长率。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 调用约定（T5/T6 必读，避免 AttributeError）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
* 工单3 兼容字段（``required_substrings`` / ``forbidden_substrings`` / ``evidence`` /
  ``min_hit_top_k`` / ``max_first_token_ms`` …）一律是 **记录字段**：
      ✅ ``rec = load_golden_index()[1]; rec["min_hit_top_k"]``
      ❌ ``golden.min_hit_top_k``（本模块**没有**这个名字）
* 本模块顶层只有工具函数与少量常量；与记录字段同名的默认值以 ``DEFAULT_`` 前缀暴露：
  ``DEFAULT_MIN_HIT_TOP_K = 5``、``DEFAULT_MAX_FIRST_TOKEN_MS = 3000``。
* 记住两条 API：``judge_answer(record, answer)``、``check_retrieval_hit(record, chunks)``。
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

__all__ = [
    "WORKSPACE",
    "GOLDEN_CANONICAL",
    "GOLDEN_MIRROR",
    "EXPECTED_IDS",
    "EXPECTED_ORDER",
    "CATEGORIES",
    "PDFS",
    "normalize",
    "extract_numbers",
    "keyword_hit",
    "page_carries",
    "load_golden",
    "load_golden_index",
    "judge_answer",
    "check_retrieval_hit",
    "validate_record_shape",
    "REQUIRED_FIELDS",
    "GOLDEN3_ALIASES",
    "GOLDEN3_RECORD_FIELDS",
    "DEFAULT_MIN_HIT_TOP_K",
    "DEFAULT_MAX_FIRST_TOKEN_MS",
    "NEGATION_CUES",
    "GUARD_WINDOW",
    "ENUMERATION_POLICY",
    "NUMERIC_VS_ENUM_STRICTNESS",
    "DIAGNOSTIC_EVIDENCE",
    "DIAGNOSTIC_EVIDENCE_MIRROR",
    "load_diagnostic_evidence",
    "check_retrieval_hit_layers",
    "clause_fragments",
    "FRAGMENT_MIN_LEN",
    "HIT_LAYER_GUIDE",
]

WORKSPACE = Path(__file__).resolve().parents[2]
GOLDEN_CANONICAL = WORKSPACE / "研发" / "data" / "eval" / "golden_qa.jsonl"
GOLDEN_MIRROR = WORKSPACE / "测试" / "测试数据" / "golden_qa_16.jsonl"

#: 16 题 id（与 设计/工单需求原文.md §五 完全一致）
EXPECTED_ORDER = [5, 6, 1, 2, 3, 4, 260, 95, 33, 34, 957, 793, 795, 543, 531, 207]
EXPECTED_IDS = sorted(EXPECTED_ORDER)

CATEGORIES = ("文本事实", "表格数值", "图像结构", "图像图表")
PDFS = ("招股说明书1", "招股说明书2")

#: 诊断层证据（**派生自 golden 构建过程**，不参与冻结真值的哈希；含 supporting 页的逐字证据）
DIAGNOSTIC_EVIDENCE = WORKSPACE / "研发" / "data" / "eval" / "golden_诊断层证据.json"
DIAGNOSTIC_EVIDENCE_MIRROR = WORKSPACE / "测试" / "测试数据" / "golden_诊断层证据.json"

#: 检索命中的**三层口径**（captain 2026-10 裁定；命中率主指标只有第一层，另两层仅作诊断列）
HIT_LAYER_GUIDE = (
    "① 权威层「完整证据命中」`hit_complete`：chunk 含该题**完整证据原文**"
    "（`record['retrieval_evidence']`，取自 `evidence_pages_complete`）——**唯一命中率主指标**；"
    "② 诊断层A「任证证据命中」`hit_any_evidence`：chunk 含 complete ＋ supporting 的**任一**证据原文，"
    "**或其任意一个子句片段**（≥4 字的、按 。；，、 切分的子句）——补充诊断列；"
    "片段级放宽是必要的：id 5 的支持性枚举在真实索引里被切到两个块"
    "（`pdf2-p40-x1` 含部门名前半、`pdf2-p41-x0` 含「国际贸易部」），任一单块都不含整句；"
    "③ 诊断层B「页级命中」`hit_page_level`：chunk 的 `page` ∈ `record['evidence_pages']`——"
    "纯诊断，**禁止**用作命中率（§3.4 明令不得以「引用页 == 证据页」冒充命中）。"
    "三者**不可合并**成「宽松口径」一列：页级只要页码落在集合里，任证层要求证据原文（或子句）真的落在 chunk 里。"
)

#: 子句片段的最小长度（诊断层A 的片段级放宽阈值）
FRAGMENT_MIN_LEN = 4

#: golden 记录必备字段（缺一不可）
REQUIRED_FIELDS = (
    "id", "order", "question", "answer", "answer_keywords", "pdf", "corpus", "category",
    "evidence_pages", "evidence_pages_primary", "evidence_pages_complete",
    "evidence_verbatim", "retrieval_evidence",
    "forbidden_keywords", "answer_regex_required", "traps", "review",
    "required_substrings", "forbidden_substrings",   # 工单3 兼容别名（由 answer_keywords 派生）
    "evidence", "evidence_verbatim_alt", "should_be_unknown", "citation_quote",
    "judge_samples",
)

#: 工单3 格式兼容别名：{本 golden 字段: 工单3 字段}
GOLDEN3_ALIASES = {
    "required_substrings": "answer_keywords",
    "forbidden_substrings": "forbidden_keywords",
    "corpus": "corpus",
    "category": "category",
}

#: ⚠️ 工单3 兼容字段都存放在 **记录里**，不是本模块的顶层常量：
#:     正确用法 ``record["required_substrings"]`` / ``record["min_hit_top_k"]``
#:     错误用法 ``golden.required_substrings`` / ``golden.min_hit_top_k``（会 AttributeError）
#:     本模块只额外给出**默认值常量**（见下），供调用方兜底，不代表记录字段的来源。
GOLDEN3_RECORD_FIELDS = (
    "required_substrings", "forbidden_substrings", "evidence", "evidence_verbatim_alt",
    "should_be_unknown", "citation_quote", "min_hit_top_k", "max_first_token_ms",
)

#: 记录字段 ``record["min_hit_top_k"]`` 的默认值（检索命中 Top-K 口径）
DEFAULT_MIN_HIT_TOP_K = 5
#: 记录字段 ``record["max_first_token_ms"]`` 的默认值（首字延迟上限，毫秒）
DEFAULT_MAX_FIRST_TOKEN_MS = 3000

#: ⚠️ 否定护栏的**全部**触发词（唯一来源；报告与文档必须与本元组一致）
#: 判定机制：禁止误答项命中点**前后各 GUARD_WINDOW 字**的窗口内出现**任一**否定词 → 豁免（不判违规）。
NEGATION_CUES = ("并非", "不属于", "不包括", "不包含", "不计入", "未计入", "不算", "不含",
                 "除外", "而不是", "而非", "不是", "而是", "不在", "非本公司")
#: 否定护栏窗口半径（命中点前后各多少字）
GUARD_WINDOW = 14

#: 枚举完整性口径（写进测试报告；T5 据此在提示词里强制逐项枚举）
ENUMERATION_POLICY = (
    "answer_keywords 为**全集匹配**（缺一即判错），**刻意要求逐项完整枚举**："
    "枚举型题（见每题 answer_form 字段，如 id 2/4/5/33/260/793）漏一项即判错。"
    "answer_keywords_any_of 只用于**同一实体的不同写法**（id 95 全称/简称、"
    "id 2 仓储及物流中心/仓储物流中心、id 5 电话及网络销售部/电话网络销售部、"
    "id 795 工程名含不含掩码「某」与顿号），**不豁免枚举完整性**。"
    "⚠️ **答案字段与判分字段不同源**：golden 的 ``answer`` 只是**人工参考答案（供报告展示）**，"
    "判分只认 answer_keywords / answer_keywords_any_of / answer_regex_required / forbidden_*；"
    "**不得照 golden answer 的表述截断列举项**——枚举项一律用**证据原文全称**"
    "（例：id 2 第 ⑤ 项必须写全『其他与主营业务相关的营运资金』，写成『其他与主营业务相关』判错）。"
    "T5 的生成提示词必须显式要求『逐项完整枚举，不得概括省略』。"
)

#: 严格度不对称口径（写进测试报告；T5 据此区分「数值可自由格式化」与「列举项必须全称」）
NUMERIC_VS_ENUM_STRICTNESS = (
    "**数值宽松（只放格式，不放精度）**、**枚举严格**："
    "① 数值按基本单位等价比较：5,520.00万元 ≡ 5,520万元 ≡ 5520万元；"
    "15,000.00万元 ≡ 15,000万元 ≡ 1.5亿元；1,670万股 ≡ 1670万股；82.10% ≡ 82.1% ≡ 百分之82.1；"
    "**但精度不放宽**（6,464.5万元 ≠ 6,464.51万元，截断即判错），且数值关键字必须带单位才走等价比较。"
    "② 列举项按**证据原文全称**（或该题 answer_keywords_any_of 组内写法）匹配，"
    "截断、简写、用『等』概括即判错。"
)

_NEGATION_CUES = NEGATION_CUES          # 内部别名（保持向后兼容）
_GUARD_WINDOW = GUARD_WINDOW

# ---------------------------------------------------------------- 归一化
_QUOTE_MAP = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2014": "-", "\uff0d": "-", "\u2013": "-", "\u2212": "-",
    "\uff05": "%", "\u3001": "、",
}


def normalize(text: str) -> str:
    """归一化：NFKC → 引号/破折号统一 → 百分之X→X% → 去千分位逗号 → 去全部空白。

    >>> normalize("6,464.51 万元") == "6464.51万元"
    True
    >>> normalize("占 25.04 ％") == "占25.04%"
    True
    >>> normalize("百分之82.10") == "82.10%"
    True
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    for k, v in _QUOTE_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"百分之\s*(\d+(?:\.\d+)?)", r"\1%", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)     # 千分位逗号（1,670 / 6,464.51）
    s = re.sub(r"\s+", "", s)
    return s


#: 单位表：后缀 → (基本单位换算系数, 量纲)
_UNITS = (
    ("亿元", 1e8, "money"), ("万元", 1e4, "money"), ("元", 1.0, "money"),
    ("亿股", 1e8, "share"), ("万股", 1e4, "share"), ("股", 1.0, "share"),
    ("%", 1.0, "percent"), ("倍", 1.0, "ratio"),
    ("亿", 1e8, "unspecified"), ("万", 1e4, "unspecified"),
)
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)(" + "|".join(re.escape(u[0]) for u in _UNITS) + ")?")


def extract_numbers(text: str) -> list[dict]:
    """抽出 (数值, 单位) → 基本单位值。返回 [{raw, value, unit, kind, base}]。"""
    s = normalize(text)
    out = []
    for m in _NUM_RE.finditer(s):
        raw, unit = m.group(1), m.group(2) or ""
        val = float(raw)
        kind, factor = "pure", 1.0
        for suf, fac, k in _UNITS:
            if unit == suf:
                kind, factor = k, fac
                break
        if not unit and m.start() > 0 and s[m.start() - 1] == "%":
            kind, factor = "percent", 1.0
        out.append({"raw": raw + unit, "value": val, "unit": unit, "kind": kind,
                    "base": val * factor})
    return out


def _kind_compatible(kw_kind: str, cand_kind: str) -> bool:
    if kw_kind == cand_kind:
        return True
    return "unspecified" in (kw_kind, cand_kind) and set((kw_kind, cand_kind)) <= {
        "unspecified", "money", "share"}


def keyword_hit(answer: str, keyword: str, tol: float = 1e-9) -> tuple[bool, str]:
    """单个关键字命中判定：纯文本走子串；带单位数值走基本单位等价。"""
    na, nk = normalize(answer), normalize(keyword)
    if not nk:
        return False, "empty_keyword"
    if nk in na:
        return True, "substring"
    kws = extract_numbers(keyword)
    if kws and re.search(r"\d", nk) and re.search(
            r"(万元|亿元|元|万股|亿股|股|%|倍)$", nk):
        for kw in kws:
            for cand in extract_numbers(answer):
                if abs(kw["base"] - cand["base"]) <= max(tol * max(1.0, abs(kw["base"])), tol) \
                        and _kind_compatible(kw["kind"], cand["kind"]):
                    return True, f"numeric_equiv({kw['raw']}~{cand['raw']})"
        return False, "numeric_missing"
    return False, "not_found"


def page_carries(page_text: str, keyword: str) -> bool:
    """**页级证据归属**判定：该页是否承载该关键字。

    与 ``keyword_hit``（回答判对用，严格）的区别：表格页的金额列由表头声明单位
    （如『计划总投资(万元)』），数值后不跟单位，故对带单位数值关键词放宽为
    「该页出现同一数值」。**只用于证据页复核**，回答判对一律用 ``keyword_hit``。
    """
    if keyword_hit(page_text, keyword)[0]:
        return True
    nk = normalize(keyword)
    if not re.search(r"(万元|亿元|元|万股|亿股|股|%|倍)$", nk):
        return False
    for k in extract_numbers(keyword):
        for c in extract_numbers(page_text):
            if c["kind"] == "pure":
                if abs(k["value"] - c["value"]) <= 1e-9 * max(1.0, abs(k["value"])):
                    return True
            elif abs(k["base"] - c["base"]) <= 1e-9 * max(1.0, abs(k["base"])):
                return True
    return False


# ---------------------------------------------------------------- 读取
def load_golden(path: str | Path | None = None) -> list[dict]:
    """读取 golden 记录。默认优先 研发/data/eval/golden_qa.jsonl，缺失则退到 测试/测试数据/。"""
    if path is None:
        p = GOLDEN_CANONICAL if GOLDEN_CANONICAL.exists() else GOLDEN_MIRROR
    else:
        p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"golden 文件不存在: {p}")
    records = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:            # pragma: no cover
            raise ValueError(f"{p} 第 {i} 行不是合法 JSON: {exc}") from exc
    return records


def load_golden_index(path: str | Path | None = None) -> dict[int, dict]:
    return {r["id"]: r for r in load_golden(path)}


# ---------------------------------------------------------------- 判分
def _forbidden_hits(answer: str, keywords, regexes) -> tuple[list, list]:
    na = normalize(answer)
    fired, guarded = [], []
    for kw in keywords or []:
        nk = normalize(kw)
        if not nk:
            continue
        start = na.find(nk)
        while start >= 0:
            lo = max(0, start - _GUARD_WINDOW)
            ctx = na[lo:start + len(nk) + _GUARD_WINDOW]
            cue = next((c for c in _NEGATION_CUES if c in ctx), None)
            if cue:
                guarded.append({"keyword": kw, "cue": cue, "context": ctx})
            else:
                fired.append({"keyword": kw, "context": ctx})
            start = na.find(nk, start + 1)
    for item in regexes or []:
        rx = item["regex"] if isinstance(item, dict) else item
        label = item.get("label", rx) if isinstance(item, dict) else rx
        m = re.search(rx, na)
        if m:
            fired.append({"regex": label, "context": na[max(0, m.start() - 10):m.end() + 10]})
    return fired, guarded


def judge_answer(record: dict, answer: str) -> dict:
    """确定性判对/判错。返回结构化结果（含逐关键字明细），供测试报告逐题留痕。"""
    na = normalize(answer)
    kw_results = []
    for kw in record.get("answer_keywords", []):
        ok, mode = keyword_hit(answer, kw)
        kw_results.append({"keyword": kw, "hit": ok, "mode": mode})
    anyof_results = []
    for group in record.get("answer_keywords_any_of", []) or []:
        hits = [kw for kw in group if keyword_hit(answer, kw)[0]]
        anyof_results.append({"group": list(group), "hit": bool(hits), "hit_by": hits})
    rx_results = []
    for item in record.get("answer_regex_required", []) or []:
        rx = item["regex"] if isinstance(item, dict) else item
        label = item.get("label", rx) if isinstance(item, dict) else rx
        rx_results.append({"label": label, "regex": rx, "hit": bool(re.search(rx, na))})
    fired, guarded = _forbidden_hits(
        answer, record.get("forbidden_keywords"), record.get("forbidden_regex"))

    miss_kw = [r["keyword"] for r in kw_results if not r["hit"]]
    miss_anyof = [r["group"] for r in anyof_results if not r["hit"]]
    miss_rx = [r["label"] for r in rx_results if not r["hit"]]
    correct = not miss_kw and not miss_anyof and not miss_rx and not fired
    reason = "全部判分关键字/正则命中且无禁止误答项" if correct else (
        f"缺关键字={miss_kw}；缺任一组={miss_anyof}；缺正则={miss_rx}；"
        f"命中禁止项={[f.get('keyword') or f.get('regex') for f in fired]}")
    return {
        "id": record.get("id"), "correct": bool(correct), "reason": reason,
        "keyword_results": kw_results, "any_of_results": anyof_results,
        "regex_results": rx_results, "forbidden_fired": fired, "forbidden_guarded": guarded,
        "normalized_answer": na,
        "judged_by": "确定性关键字判定（测试/common/golden.py::judge_answer），非 LLM 自评",
    }


# ---------------------------------------------------------------- 检索命中
def _chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("content") or chunk.get("chunk") or "")
    return str(chunk)


def _chunk_page(chunk):
    """取 chunk 的真实物理页码（dict 形式支持 page / 页码；字符串形式返回 None）。"""
    if isinstance(chunk, dict):
        for key in ("page", "页码", "page_no", "pageNum"):
            v = chunk.get(key)
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.strip().isdigit():
                return int(v.strip())
    return None


def load_diagnostic_evidence(path: str | Path | None = None) -> dict:
    """读取诊断层证据（supporting 页的逐字证据），返回 {id: {"supporting": [...], ...}}。

    该文件由 `研发/data/eval/构建_golden_qa.py` 派生生成，**不改变冻结 golden 的哈希**。
    缺失时返回空字典（诊断层A 退化为仅 complete 层，不影响权威层判定）。
    """
    if path is None:
        p = DIAGNOSTIC_EVIDENCE if DIAGNOSTIC_EVIDENCE.exists() else DIAGNOSTIC_EVIDENCE_MIRROR
    else:
        p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {int(it["id"]): it for it in data.get("items", [])}


def clause_fragments(text: str, min_len: int = FRAGMENT_MIN_LEN, keywords=()) -> list[str]:
    """把证据原文切成子句片段（诊断层A 的片段级放宽用）。

    片段来源（去重、保序、归一化后长度 ≥ ``min_len``）：
      1. 按 ``。；！？，、`` 与换行切分的子句；
      2. **该题判分关键字（含 any_of 写法）中确实出现在该证据里的项**——保证「大客户销售部和国际贸易部」
         这类被连词粘在一起的枚举也能拆出 `国际贸易部` 这样的独立片段（不靠切「和/及」，避免切坏
         `电话及网络销售部` 这类含连词的真实名称）。
    例：``clause_fragments('销售部下设渠道销售部、电话及网络销售部、大客户销售部和国际贸易部',
                          keywords=['电话及网络销售部', '国际贸易部'])``
    → 含 ``'销售部下设渠道销售部'``/``'电话及网络销售部'``/``'国际贸易部'``。
    """
    s = normalize(text)
    parts = [p for p in re.split(r"[。；;！!？?，,、\n]+", s) if len(p) >= min_len]
    for kw in keywords or ():
        nk = normalize(kw)
        if len(nk) >= min_len and nk in s:
            parts.append(nk)
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _layer_hit(evidence_list, texts, allow_fragments: bool = False,
               fragment_keywords=()) -> tuple[bool, list, list, list]:
    """在给定 chunk 文本集合里按「证据原文落在 chunk 中」判定（配对类按配对同块判定）。

    ``allow_fragments=True``（仅诊断层A）：整句未命中时再按**子句片段**判定，并记录命中的片段。
    权威层（``allow_fragments=False``）**不启用**片段放宽。
    """
    matched, missed, frag_hits = [], [], []
    for ev in evidence_list or []:
        kind = ev.get("kind", "verbatim")
        hit_idx, how = None, "text"
        if kind in ("chart_pairs", "pairs", "labels"):
            pairs = ev.get("pairs") or [[lbl] for lbl in ev.get("labels", [])]
            for i, t in enumerate(texts):
                if all(all(normalize(p) in t for p in pair) for pair in pairs):
                    hit_idx = i
                    break
        else:
            needle = normalize(ev.get("text", ""))
            hit_idx = next((i for i, t in enumerate(texts) if needle and needle in t), None)
            if hit_idx is None and allow_fragments and ev.get("text"):
                for frag in clause_fragments(ev["text"], keywords=fragment_keywords):
                    j = next((i for i, t in enumerate(texts) if frag in t), None)
                    if j is not None:
                        hit_idx, how = j, "fragment"
                        frag_hits.append({"page": ev.get("page"), "fragment": frag, "chunk_index": j})
                        break
        rec = {"page": ev.get("page"), "kind": kind,
               "text": ev.get("text") or ev.get("pairs") or ev.get("labels"),
               "chunk_index": hit_idx, "matched_by": how}
        (matched if hit_idx is not None else missed).append(rec)
    return bool(matched), matched, missed, frag_hits


def check_retrieval_hit_layers(record: dict, chunks, diagnostic: dict | None = None,
                               top_k: int | None = None) -> dict:
    """**三层命中判定**（captain 裁定；见 ``HIT_LAYER_GUIDE``）：

    - ① ``hit_complete``（权威，唯一命中率主指标）：chunk 含**完整证据原文**（`retrieval_evidence`）；
    - ② ``hit_any_evidence``（诊断层A）：chunk 含 complete ＋ supporting 的**任一**证据原文，**或其子句片段**
      （片段级放宽是必要的：id 5 的支持性枚举在真实索引里被切到 `pdf2-p40-x1` 与 `pdf2-p41-x0` 两块，单块不含整句）；
    - ③ ``hit_page_level``（诊断层B）：chunk 的 `page` ∈ `record["evidence_pages"]`（**禁止**用作命中率）。

    ``diagnostic`` 取 ``load_diagnostic_evidence()``（提供 supporting 层证据）；缺省时自动加载。
    ``top_k`` 默认取 ``record["min_hit_top_k"]``（前 K 条内命中即算），三层的候选集合同一。
    """
    k = int(top_k if top_k is not None else record.get("min_hit_top_k", DEFAULT_MIN_HIT_TOP_K))
    all_chunks = list(chunks or [])
    considered = all_chunks[:k] if k > 0 else []
    texts = [normalize(_chunk_text(c)) for c in considered]
    pages = [_chunk_page(c) for c in considered]

    diag = diagnostic if diagnostic is not None else load_diagnostic_evidence()
    supporting = (diag.get(record.get("id"), {}) or {}).get("supporting", []) or []

    hit_complete, m1, miss1, _ = _layer_hit(record.get("retrieval_evidence", []), texts,
                                            allow_fragments=False)
    kw_pool = list(record.get("answer_keywords", []) or []) + [
        m for grp in (record.get("answer_keywords_any_of", []) or []) for m in grp]
    hit_any, m2, miss2, frags = _layer_hit(
        list(record.get("retrieval_evidence", [])) + list(supporting), texts,
        allow_fragments=True, fragment_keywords=kw_pool)
    ev_pages = set(record.get("evidence_pages", []) or [])
    page_hit_idx = next((i for i, p in enumerate(pages) if p is not None and p in ev_pages), None)
    return {
        "id": record.get("id"),
        # ① 权威层
        "hit_complete": hit_complete, "matched_complete": m1, "missed_complete": miss1,
        # ② 诊断层A
        "hit_any_evidence": hit_any, "matched_any_evidence": m2, "missed_any_evidence": miss2,
        "matched_fragments": frags, "n_supporting_evidence": len(supporting),
        # ③ 诊断层B
        "hit_page_level": page_hit_idx is not None,
        "page_level_chunk_index": page_hit_idx,
        "considered_chunk_pages": pages,
        "top_k": k, "n_chunks_passed": len(all_chunks), "n_chunks_considered": len(texts),
        "authoritative_layer": "hit_complete",
        "guide": HIT_LAYER_GUIDE,
    }


def check_retrieval_hit(record: dict, chunks, top_k: int | None = None) -> dict:
    """检索命中：证据原文（归一化）落在**前 K 条返回 chunk** 中。与「回答判对」严格分离。

    ``min_hit_top_k`` 的**精确语义**：
      1. 取检索结果的**前 K 条 chunk** 作为命中候选集合（K = ``record["min_hit_top_k"]``，golden 中恒为 5）；
         **其中任意一条**包含证据原文即算本题「检索命中」；
      2. 它不是「至少命中 K 条」，也不是「必须命中第 K 条」——是「**前 K 条内命中即算**」；
      3. 本函数默认按 ``record["min_hit_top_k"]`` **截断**（``chunks[:K]``）后再判定；
         若调用方只传了 3 条（例如生成阶段实际检索 Top-K=3），就只在这 3 条里判（本函数不补数据）。
    与用户提示词「检索 Top-K=3」的关系：
      - **生成阶段**的检索条数 K=3 由 T2/T5 按用户要求实现（可配置）；
      - **命中判定**口径 K=5 由 golden 固定，比生成口径宽 2 条，用于区分「检索没捞到证据」与「生成没写对」；
      - T6 报告必须**分别标注**两个 K 的取值与来源，不得混用同一个 K。
    """
    k = int(top_k if top_k is not None else record.get("min_hit_top_k", DEFAULT_MIN_HIT_TOP_K))
    all_texts = [_chunk_text(c) for c in (chunks or [])]
    considered = all_texts[:k] if k > 0 else []
    texts = [normalize(t) for t in considered]
    matched, missed = [], []
    for ev in record.get("retrieval_evidence", []) or []:
        kind = ev.get("kind", "verbatim")
        hit_idx = None
        if kind in ("chart_pairs", "pairs", "labels"):
            pairs = ev.get("pairs") or [[lbl] for lbl in ev.get("labels", [])]
            for i, t in enumerate(texts):
                if all(all(normalize(p) in t for p in pair) for pair in pairs):
                    hit_idx = i
                    break
        else:
            needle = normalize(ev.get("text", ""))
            hit_idx = next((i for i, t in enumerate(texts) if needle and needle in t), None)
        rec = {"page": ev.get("page"), "kind": kind,
               "text": ev.get("text") or ev.get("pairs") or ev.get("labels"),
               "chunk_index": hit_idx}
        (matched if hit_idx is not None else missed).append(rec)
    return {
        "id": record.get("id"), "hit": bool(matched),
        "matched": matched, "missed": missed,
        "top_k": k, "n_chunks_passed": len(all_texts), "n_chunks_considered": len(texts),
        "rule": "取检索结果前 K 条 chunk 作为候选集合，其中任一含证据原文即算命中；"
                "不得以「引用页码==证据页」冒充命中",
    }


# ---------------------------------------------------------------- 契约校验
def validate_record_shape(record: dict) -> list[str]:
    """字段契约校验，返回问题列表（空列表=通过）。"""
    problems: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in record:
            problems.append(f"缺字段 {f}")
    if problems:
        return problems
    rid = record["id"]
    if not isinstance(rid, int):
        problems.append("id 必须是 int")
    if record["order"] not in range(1, 17):
        problems.append("order 必须在 1..16")
    for f in ("question", "answer", "evidence_verbatim"):
        if not isinstance(record[f], str) or not record[f].strip():
            problems.append(f"{f} 不能为空")
    if record["category"] not in CATEGORIES:
        problems.append(f"category 非法: {record['category']}")
    if record["pdf"] not in PDFS:
        problems.append(f"pdf 非法: {record['pdf']}")
    if record["corpus"] not in ("pdf1", "pdf2"):
        problems.append(f"corpus 非法: {record['corpus']}")
    if not record["answer_keywords"]:
        problems.append("answer_keywords 不能为空")
    if not record["evidence_pages"]:
        problems.append("evidence_pages 不能为空")
    if not record["evidence_pages_complete"]:
        problems.append("evidence_pages_complete 不能为空")
    if not set(record["evidence_pages_complete"]) <= set(record["evidence_pages"]):
        problems.append("evidence_pages_complete 必须是 evidence_pages 的子集")
    if not set(record["evidence_pages_primary"]) <= set(record["evidence_pages_complete"]):
        problems.append("evidence_pages_primary 必须是 evidence_pages_complete 的子集")
    # 页级等式（防止口径漂移）：
    #   evidence_pages       = evidence_pages_complete + supporting_pages
    #   evidence_pages_union = evidence_pages + related_pages
    _complete = set(record["evidence_pages_complete"])
    _supporting = set(record.get("supporting_pages", []))
    _related = set(record.get("related_pages", []))
    if sorted(_complete | _supporting) != record["evidence_pages"]:
        problems.append("evidence_pages 必须等于 evidence_pages_complete + supporting_pages")
    if "evidence_pages_union" in record and \
            sorted(_complete | _supporting | _related) != record["evidence_pages_union"]:
        problems.append("evidence_pages_union 必须等于 complete + supporting + related")
    for p in record["evidence_pages"] + record.get("supporting_pages", []) \
            + record.get("related_pages", []):
        if not isinstance(p, int) or p < 1:
            problems.append(f"证据页必须是 1-based 正整数: {p!r}")
    if record["evidence_pages"] != sorted(set(record["evidence_pages"])):
        problems.append("evidence_pages 必须升序去重")
    if not record["retrieval_evidence"]:
        problems.append("retrieval_evidence 不能为空")
    for ev in record["retrieval_evidence"]:
        if "page" not in ev or "kind" not in ev:
            problems.append(f"retrieval_evidence 缺 page/kind: {ev}")
    if record["required_substrings"] != record["answer_keywords"]:
        problems.append("required_substrings（工单3 别名）必须等于 answer_keywords")
    if record["forbidden_substrings"] != record["forbidden_keywords"]:
        problems.append("forbidden_substrings（工单3 别名）必须等于 forbidden_keywords")
    if record["evidence"] != record["evidence_verbatim"]:
        problems.append("evidence（工单3 别名）必须等于 evidence_verbatim")
    if record["should_be_unknown"] is not False:
        problems.append("16 题均为可答问题，should_be_unknown 必须为 False")
    if not record["citation_quote"]:
        problems.append("citation_quote 不能为空（引用回溯抽查用）")
    js = record["judge_samples"]
    if not js.get("positive") or not js.get("negatives"):
        problems.append("judge_samples 必须含 positive 与 negatives")
    if not record["traps"]:
        problems.append("traps（判分陷阱口径）不能为空")
    rv = record["review"]
    for f in ("reviewed", "reviewed_by", "pdf_opened", "verdict"):
        if f not in rv:
            problems.append(f"review 缺字段 {f}")
    if rv.get("reviewed") is not True:
        problems.append("review.reviewed 必须为 True（已逐题打开 PDF 复核）")
    return problems
