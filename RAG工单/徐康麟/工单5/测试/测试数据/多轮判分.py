# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-Query 理解优化
文件：测试/测试数据/多轮判分.py

用途：**工单5 多轮对话 Query 理解的唯一确定性判分库**（不依赖网络、不依赖 LLM 自评）。
      与工单4 的 `测试/common/golden.py::judge_answer` 同源同风格：数值走
      「全角/半角 + 千分位逗号 + 空白」归一化等价比较，枚举项走归一化子串匹配。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
对外 API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    load_cases(path)                读 JSONL 用例（自动跳过 `_` 开头的元数据行）
    judge_case(case, result)        -> {"passed", "checks":[...], "reasons":[...]}
    judge_abnormal_case(case, resp) -> 异常输入的「不崩溃/非空/不 500」判分
    summarize_abnormal(items)       -> 异常输入汇总
    normalize / normalize_query     归一化（供报告与人工复核复算）
    COMPANIES / OTHER_COMPANY_OF    语料两家发行人的全称表（no_leak 检查用）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
判分维度（全部确定性；每个维度一条 check，可直接在报告里逐条展示）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. answer_present        答案非空且未抛异常（任何用例的**底线**）
 2. answer_must_contain   `must_contain` 每项都命中（归一化子串；数值带单位走基本单位等价）
 3. answer_must_contain_any  `must_contain_any` 每组至少命中一项（可接受分支）
 4. answer_must_not_contain  `must_not_contain` 每项都不得命中（带否定护栏，见 §护栏）
 5. refused_match         `refused` 与 `expect_refused` 一致
 6. strategy_match        `strategy` == `expected_strategy`（expected 为空则跳过）
 7. focus_match           `focus_entity` == `expected_focus`（expected 为空则跳过）
 8. rewrite_contains      `expected_rewrite` 非空时，`rewritten_query` 归一化后须包含它
 9. focus_in_rewrite      `rewritten_query` 归一化后须包含 `expected_focus`（**改写质量**主检查）
10. no_leak              **跨会话实体泄漏**：改写后的 query 里不得出现「另一个公司」的全称
11. checks_passed        以上全部通过 → `passed=True`

`passed_turns / total_turns` 是主准确率；`checks` 里每个维度的通过率单独汇总，
故报告既能给总分、也能定位是「改写没做对」还是「检索/生成没答对」。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
归一化口径（**为什么这样定**）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
语料原文是「6,464.51 万元」，模型可能输出「6464.51万元」「６４６４.５１ 万元」「6,464.51万元」，
**这些都必须判过**（放格式、不放精度）：

    ① NFKC：全角→半角（６４６４.５１ → 6464.51）
    ② 引号/破折号/％/全角逗号统一
    ③ 去掉**数字之间的千分位逗号**（6,464.51 → 6464.51，逗号仅作分隔符）
    ④ 百分之82.10 → 82.10%
    ⑤ 删除**全部空白**（含全角空格 U+3000、换行、制表符）

实现上：优先复用 `测试/common/golden.py::normalize`（**单一实现来源**，避免两处口径漂移），
在其结果之上再抹掉 CJK 逗号「，」（golden 的逗号切分用于子句片段，故未做此步）；
`golden.py` 不可用时退回本地等价实现，并在返回值里标注 `normalizer` 字段。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
护栏（防误判）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
* **否定护栏**：`must_not_contain` 的命中点前后 `GUARD_WINDOW` 字内出现 `NEGATION_CUES`
  任一否定词 → 不判违规（与 golden.judge_answer 同口径）。例如
  「大客户销售部下设 6 个销售处；销售部由 4 个部门构成，包括国际贸易部」中
  「国际贸易部」若出现在澄清语境（如「不包括上海分公司」）即被豁免。
* **数值等价**：`must_contain` 的项若形如 `<数字><单位>` 且带单位，则按基本单位比较
  （6,464.51万元 ≡ 6464.51万元），**精度不放宽**（6,464.5 ≠ 6,464.51）。
* **同义写法**：`must_contain_any` 用于「实体不同写法」与「可接受分支」两类场景。
"""
from __future__ import annotations

import importlib.util
import json
import re
import unicodedata
from pathlib import Path

__all__ = [
    "WORKSPACE", "DATA_DIR", "CASES_MULTITURN", "CASES_ISOLATED", "CASES_ABNORMAL",
    "COMPANIES", "OTHER_COMPANY_OF", "SHORT_NAMES",
    "GUARD_WINDOW", "NEGATION_CUES", "NORMALIZER_SOURCE",
    "load_cases", "load_multiturn_cases", "load_isolated_cases", "load_abnormal_cases",
    "normalize", "normalize_query", "keyword_hit",
    "judge_case", "judge_abnormal_case", "summarize_abnormal", "abnormal_expect_note",
    "CHECK_NAMES", "CHECK_TITLES", "MULTITURN_METRICS_GUIDE",
]

WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
CASES_MULTITURN = DATA_DIR / "工单5多轮用例.jsonl"
CASES_ISOLATED = DATA_DIR / "工单5多轮用例_独立会话.jsonl"
CASES_ABNORMAL = DATA_DIR / "异常输入用例.jsonl"

#: 语料两家发行人（**全称**，no_leak 只按全称判，避免简称误伤）
COMPANIES = ("武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司")
#: 用于报告的简称（仅展示，不参与判定）
SHORT_NAMES = {"武汉兴图新科电子股份有限公司": "兴图新科",
               "武汉力源信息技术股份有限公司": "力源信息"}


def other_company_of(entity: str | None) -> str | None:
    """给定焦点实体，返回语料里的**另一个**公司全称（无则 None）。

    ⚠️ 只能判「已知公司全称 C 是否就是焦点实体」（``C in entity`` 吸收 ``entity`` 里可能带的修饰），
    **不能**反向用 ``entity in C``——两家公司全称互为非子串，反向包含会让短的那个永远匹配上自己，
    导致 no_leak 恒被跳过（实测踩过的坑）。
    """
    if not entity:
        return None
    n_ent = normalize(entity)
    for c in COMPANIES:
        if normalize(c) not in n_ent:
            return c
    return None


OTHER_COMPANY_OF = other_company_of

#: 否定护栏（与 golden.NEGATION_CUES 一致；此处为兜底副本，能 import golden 时以 golden 为准）
NEGATION_CUES = ("并非", "不属于", "不包括", "不包含", "不计入", "未计入", "不算", "不含",
                 "除外", "而不是", "而非", "不是", "而是", "不在", "非本公司")
GUARD_WINDOW = 14

#: 归一化函数来源标记（运行时确定，写进判分结果便于复核）
NORMALIZER_SOURCE = "local"

#: 判分维度名（顺序 = 报告展示顺序）
CHECK_NAMES = ("answer_present", "answer_must_contain", "answer_must_contain_any",
               "answer_must_not_contain", "refused_match", "strategy_match", "focus_match",
               "rewrite_contains", "focus_in_rewrite", "no_leak")
CHECK_TITLES = {
    "answer_present": "答案非空",
    "answer_must_contain": "必含项（must_contain）",
    "answer_must_contain_any": "任含项（must_contain_any）",
    "answer_must_not_contain": "禁含项（must_not_contain）",
    "refused_match": "拒答标志一致",
    "strategy_match": "改写策略一致",
    "focus_match": "焦点实体一致",
    "rewrite_contains": "改写结果含期望问句",
    "focus_in_rewrite": "改写结果含焦点实体",
    "no_leak": "无跨会话实体泄漏",
}

MULTITURN_METRICS_GUIDE = (
    "多轮准确率主口径 = `passed_turns / total_turns`（逐轮判分，见 `多轮判分.py::judge_case`）；"
    "并列给出**会话整体口径** = 「该 session 的全部轮次都 passed 才算 1 个会话通过」"
    "（本工单 5 轮分属 2 个 session：seq1 三轮、seq2 两轮，故会话整体准确率分母为 2）。"
    "两套口径**必须同时报告**：逐轮口径便于定位，会话整体口径反映「多轮链路是否真的跑通」。"
    "独立会话对照实验（`工单5多轮用例_独立会话.jsonl`）的降级轮次**不计入**多轮准确率分母。"
)


# ---------------------------------------------------------------- 归一化
def _load_golden_module():
    """加载 测试/common/golden.py（工单4 遗留判分器）作为**归一化单一实现来源**。"""
    p = WORKSPACE / "测试" / "common" / "golden.py"
    if not p.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("gongdan5_golden_bridge", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore[union-attr]
        return mod
    except Exception:                          # noqa: BLE001 —— 归一化绝不能因桥接失败而中断
        return None


_GOLDEN = _load_golden_module()
if _GOLDEN is not None:
    NORMALIZER_SOURCE = "测试/common/golden.py::normalize（单一实现来源）+ 追加 CJK 逗号/全角空格规整"
    NEGATION_CUES = tuple(_GOLDEN.NEGATION_CUES)
    GUARD_WINDOW = int(_GOLDEN.GUARD_WINDOW)

_EXTRA_STRIP = ("，", "\u3000", "\u200b", "\ufeff", "\u00a0")
_QUOTE_MAP = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
              "\u2014": "-", "\uff0d": "-", "\u2013": "-", "\u2212": "-",
              "\uff05": "%", "％": "%", "﹪": "%"}


def normalize(text) -> str:
    """多轮判分专用归一化：NFKC → 引号/％/破折号统一 → 千分位逗号 → 百分之X→X% → 删全部空白。

    >>> normalize("6,464.51 万元") == "6464.51万元"
    True
    >>> normalize("６４６４.５１万元") == "6464.51万元"
    True
    >>> normalize("百分之82.10") == "82.10%"
    True
    >>> normalize("大客户 销售部") == "大客户销售部"
    True
    """
    if text is None:
        return ""
    s = str(text)
    if _GOLDEN is not None:
        s = _GOLDEN.normalize(s)
    else:                                                    # 兜底：与 golden 等价的本地方案
        s = unicodedata.normalize("NFKC", s)
        for k, v in _QUOTE_MAP.items():
            s = s.replace(k, v)
        s = re.sub(r"百分之\s*(\d+(?:\.\d+)?)", r"\1%", s)
        s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)
        s = re.sub(r"\s+", "", s)
    for ch in _EXTRA_STRIP:
        s = s.replace(ch, "")
    # 二次保险：NFKC 后仍可能残留的全角逗号/千分位模式
    s = re.sub(r"(?<=\d)[,，](?=\d{3}(\D|$))", "", s)
    return s


def normalize_query(text) -> str:
    """改写问句归一化：与 ``normalize`` 同口径（去全部空白 → 可直接做子串比较）。"""
    return normalize(text)


def keyword_hit(answer: str, keyword: str) -> tuple[bool, str]:
    """单项命中判定：优先 golden.keyword_hit（数值按基本单位等价），否则本地实现。"""
    if _GOLDEN is not None:
        try:
            return _GOLDEN.keyword_hit(answer or "", keyword or "")
        except Exception:                                    # noqa: BLE001
            pass
    na, nk = normalize(answer), normalize(keyword)
    if not nk:
        return False, "empty_keyword"
    if nk in na:
        return True, "substring"
    return False, "not_found"


def _forbidden_hits(answer: str, items) -> tuple[list, list]:
    """禁含项命中（带否定护栏）。返回 (fired, guarded)。"""
    na = normalize(answer)
    fired, guarded = [], []
    for kw in items or []:
        nk = normalize(kw)
        if not nk:
            continue
        start = na.find(nk)
        while start >= 0:
            lo = max(0, start - GUARD_WINDOW)
            ctx = na[lo:start + len(nk) + GUARD_WINDOW]
            cue = next((c for c in NEGATION_CUES if c in ctx), None)
            rec = {"keyword": kw, "context": ctx}
            if cue:
                rec["cue"] = cue
                guarded.append(rec)
            else:
                fired.append(rec)
            start = na.find(nk, start + 1)
    return fired, guarded


# ---------------------------------------------------------------- 用例读取
def load_cases(path) -> list[dict]:
    """读 JSONL 用例。**跳过** ``_`` 开头的元数据行与空行；行内非 JSON 直接报错定位。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"用例文件不存在: {p}")
    out = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} 第 {lineno} 行不是合法 JSON: {exc}") from exc
        if not isinstance(rec, dict) or "id" not in rec:
            continue                                          # `_work_order` 等元数据行
        out.append(rec)
    return out


def load_multiturn_cases(path=None) -> list[dict]:
    return load_cases(path or CASES_MULTITURN)


def load_isolated_cases(path=None) -> list[dict]:
    return load_cases(path or CASES_ISOLATED)


def load_abnormal_cases(path=None) -> list[dict]:
    return load_cases(path or CASES_ABNORMAL)


# ---------------------------------------------------------------- 判分
def _mk(name: str, passed: bool, detail: str, **extra) -> dict:
    rec = {"name": name, "title": CHECK_TITLES.get(name, name),
           "passed": bool(passed), "detail": detail}
    rec.update(extra)
    return rec


def judge_case(case: dict, result: dict, *, other_company: str | None = None) -> dict:
    """工单5 多轮用例的**确定性判分**。

    参数
    ----
    case   : `工单5多轮用例.jsonl` 的一行（含 must_contain / expected_strategy / …）
    result : 引擎一轮的输出，字段名兼容 `MultiTurnAnswer.to_dict()` 与 `POST /api/chat` 响应
             （至少含 answer；可选 rewritten_query / strategy / focus_entity / refused / error）
    other_company : 覆盖 no_leak 检查里「另一个公司」的全称；缺省按 `expected_focus` 自动推断。

    返回
    ----
    ``{"passed": bool, "checks": [{"name","title","passed","detail"}, …], "reasons": [str, …]}``

    **纯函数**：不读网络、不调 LLM、不写文件。
    """
    case = case or {}
    result = result or {}
    answer = result.get("answer") or ""
    rewritten = result.get("rewritten_query") or result.get("rewritten") or ""
    strategy = result.get("strategy")
    focus = result.get("focus_entity")
    refused = result.get("refused")
    checks: list[dict] = []

    # 1) 答案非空（含异常/未返回）
    err = result.get("error")
    ok_present = bool(str(answer).strip()) and not err
    checks.append(_mk("answer_present", ok_present,
                      "答案非空" if ok_present else f"答案为空或请求失败: {err or 'answer 字段为空'}"))

    # 2) must_contain
    must = list(case.get("must_contain") or [])
    miss = [kw for kw in must if not keyword_hit(answer, kw)[0]]
    checks.append(_mk("answer_must_contain", not miss,
                      f"全部 {len(must)} 项命中" if not miss else f"缺 {len(miss)} 项: {miss}",
                      required=must, missing=miss))

    # 3) must_contain_any（每组至少一项）
    groups = list(case.get("must_contain_any") or [])
    group_recs, miss_groups = [], []
    for grp in groups:
        hit_by = [kw for kw in grp if keyword_hit(answer, kw)[0]]
        group_recs.append({"group": list(grp), "hit": bool(hit_by), "hit_by": hit_by})
        if not hit_by:
            miss_groups.append(list(grp))
    checks.append(_mk("answer_must_contain_any", not miss_groups,
                      f"{len(groups)} 组均命中" if groups and not miss_groups
                      else ("未设任含项（跳过）" if not groups else f"未命中的组: {miss_groups}"),
                      groups=group_recs))

    # 4) must_not_contain（带否定护栏）
    forbid = list(case.get("must_not_contain") or [])
    fired, guarded = _forbidden_hits(answer, forbid)
    checks.append(_mk("answer_must_not_contain", not fired,
                      f"未命中任何禁含项（{len(forbid)} 项）" if not fired
                      else f"命中禁含项: {[f['keyword'] for f in fired]}",
                      forbidden=forbid, fired=fired, guarded=guarded))

    # 5) 拒答标志一致
    expect_refused = bool(case.get("expect_refused", False))
    ok_refused = (refused is not None) and (bool(refused) == expect_refused)
    checks.append(_mk("refused_match", ok_refused,
                      f"refused={refused!r} 与 expect_refused={expect_refused} 一致" if ok_refused
                      else f"refused={refused!r} ≠ expect_refused={expect_refused}"))

    # 6) 策略一致
    exp_strategy = case.get("expected_strategy")
    if exp_strategy is None:
        checks.append(_mk("strategy_match", True, "用例未指定 expected_strategy（跳过）"))
    else:
        checks.append(_mk("strategy_match", strategy == exp_strategy,
                          f"strategy={strategy!r} == 期望 {exp_strategy!r}" if strategy == exp_strategy
                          else f"strategy={strategy!r} ≠ 期望 {exp_strategy!r}"))

    # 7) 焦点实体一致
    exp_focus = case.get("expected_focus")
    if exp_focus is None:
        checks.append(_mk("focus_match", True, "用例未指定 expected_focus（跳过）"))
    else:
        checks.append(_mk("focus_match", focus == exp_focus,
                          f"focus_entity={focus!r} == 期望 {exp_focus!r}" if focus == exp_focus
                          else f"focus_entity={focus!r} ≠ 期望 {exp_focus!r}"))

    # 8) 改写结果包含期望问句
    exp_rw = case.get("expected_rewrite")
    n_rw = normalize_query(rewritten)
    if exp_rw:
        ok_rw = normalize_query(exp_rw) in n_rw
        checks.append(_mk("rewrite_contains", ok_rw,
                          "改写结果包含期望问句" if ok_rw
                          else f"改写结果 {rewritten!r} 未包含期望 {exp_rw!r}",
                          expected_rewrite=exp_rw))
    else:
        checks.append(_mk("rewrite_contains", True, "用例未指定 expected_rewrite（跳过）"))

    # 9) 改写结果包含焦点实体（改写质量主检查；expected_focus 为空则跳过）
    if exp_focus:
        ok_focus_in = normalize_query(exp_focus) in n_rw
        checks.append(_mk("focus_in_rewrite", ok_focus_in,
                          "改写结果含焦点实体" if ok_focus_in
                          else f"改写结果 {rewritten!r} 不含焦点实体 {exp_focus!r}"
                               + ("（首轮 expected_rewrite=null 时该检查仍要求 query 含实体）"
                                  if not exp_rw else "")))
    else:
        checks.append(_mk("focus_in_rewrite", True, "用例未指定 expected_focus（跳过）"))

    # 10) 跨会话实体泄漏
    other = other_company if other_company is not None else other_company_of(exp_focus)
    if not n_rw:
        checks.append(_mk("no_leak", True,
                          "无改写结果可比对（rewritten_query 为空）——不判泄漏" if not str(rewritten).strip()
                          else "改写结果为空串"))
    elif other:
        leaked = normalize_query(other) in n_rw
        same_focus = bool(exp_focus) and normalize_query(exp_focus) in n_rw
        checks.append(_mk("no_leak", not leaked,
                          (f"未出现另一公司「{other}」"
                           + ("（且含本会话焦点实体）" if same_focus else "")) if not leaked
                          else f"改写结果泄漏了另一公司全称「{other}」：{rewritten!r}",
                          other_company=other, rewritten=rewritten))
    else:
        checks.append(_mk("no_leak", True, f"未能推断「另一公司」（expected_focus={exp_focus!r}），跳过"))

    reasons = [f"[{c['name']}] {c['detail']}" for c in checks if not c["passed"]]
    return {"id": case.get("id"), "passed": not reasons, "checks": checks, "reasons": reasons,
            "normalizer": NORMALIZER_SOURCE}


# ---------------------------------------------------------------- 异常输入判分
_ABNORMAL_EXPECT_NOTE = {
    "friendly": "应友好处理：给出『不清楚/无法确定』或向用户追问澄清；不得编造、不得崩溃",
    "refused": "应走拒答闸门：refused=true 且答案说明语料未披露",
    "no_crash": "底线口径：不崩溃、返回非空字符串、HTTP 层不 500（内容不作强判）",
}


def abnormal_expect_note(expect: str) -> str:
    return _ABNORMAL_EXPECT_NOTE.get(expect, _ABNORMAL_EXPECT_NOTE["no_crash"])


def judge_abnormal_case(case: dict, resp: dict) -> dict:
    """异常输入的判分：**不崩溃 + 非空 + HTTP 不 500**（+ 友好处理的软证据）。

    ``resp`` 字段：``{"http_status": int|None, "answer": str|None, "error": str|None,
                     "elapsed_ms": float|None, "refused": bool|None}``
    """
    case, resp = case or {}, resp or {}
    status = resp.get("http_status")
    answer = resp.get("answer")
    err = resp.get("error")
    checks = []

    no_500 = (status is None) or (int(status) < 500)
    checks.append(_mk("http_not_5xx", no_500,
                      f"HTTP {status}（非 5xx）" if status is not None else "无 HTTP 状态（local 直调）"
                      if no_500 else f"HTTP {status} 为 5xx"))
    no_exc = not err
    checks.append(_mk("no_exception", no_exc, "未抛异常" if no_exc else f"抛异常: {err}"))
    nonempty = bool(str(answer or "").strip())
    checks.append(_mk("answer_nonempty", nonempty,
                      f"答案非空（{len(str(answer or ''))} 字）" if nonempty else "答案为空"))

    friendly_markers = ("不清楚", "不知道", "无法确定", "未披露", "无法回答", "抱歉",
                        "请输入", "请说明", "请补充", "哪家公司", "哪一家", "具体", "未能理解",
                        "无法理解", "没听懂", "换个问法")
    na = normalize(answer or "")
    friendly_hit = [m for m in friendly_markers if normalize(m) in na]
    expect = case.get("expect", "no_crash")
    if expect == "no_crash":
        semantic_ok = True
    elif expect == "refused":
        semantic_ok = bool(resp.get("refused")) or bool(friendly_hit)
    else:                       # friendly
        semantic_ok = bool(friendly_hit) or bool(str(answer or "").strip())
    checks.append(_mk("friendly_handled", semantic_ok,
                      (f"命中友好表达: {friendly_hit[:4]}" if friendly_hit
                       else "未命中友好表达词，但答案非空（异常输入不作强判）") if semantic_ok
                      else "既未友好追问也无有效答复"))

    reasons = [f"[{c['name']}] {c['detail']}" for c in checks if not c["passed"]]
    return {"id": case.get("id"), "expect": expect, "expect_note": abnormal_expect_note(expect),
            "note": case.get("note"), "passed": not reasons, "checks": checks, "reasons": reasons}


def summarize_abnormal(items: list[dict]) -> dict:
    """异常输入汇总：通过数/总数 + 逐维度通过率。"""
    items = list(items or [])
    n = len(items)
    dims: dict[str, int] = {}
    for it in items:
        for c in it.get("checks") or []:
            dims.setdefault(c["name"], 0)
            if c["passed"]:
                dims[c["name"]] += 1
    return {
        "n_cases": n,
        "n_passed": sum(1 for it in items if it.get("passed") is True),
        "pass_rate": round(sum(1 for it in items if it.get("passed") is True) / n, 4) if n else None,
        "dimension_pass": {k: {"passed": v, "total": n,
                               "rate": round(v / n, 4) if n else None} for k, v in dims.items()},
        "criterion": "不崩溃（无异常）+ 返回非空字符串 + HTTP 层不 5xx；"
                     "能给出「不清楚」或友好追问均算通过；不比对语料真值",
    }


if __name__ == "__main__":                                    # 手工自检：仅打印口径，不跑网络
    print(f"[多轮判分] 归一化来源: {NORMALIZER_SOURCE}")
    print(f"[多轮判分] 语料实体: {COMPANIES}")
    for name, path in (("多轮", CASES_MULTITURN), ("独立会话", CASES_ISOLATED), ("异常输入", CASES_ABNORMAL)):
        try:
            cases = load_cases(path)
            print(f"[多轮判分] {name}用例 {len(cases)} 条 ← {path}")
        except Exception as exc:                              # noqa: BLE001
            print(f"[多轮判分] {name}用例读取失败: {type(exc).__name__}: {exc}")
    demo = {"answer": "大客户销售部有 6 个销售处：北京销售处、深圳销售处、广州销售处、"
                      "成都销售处、珠海销售处、武汉销售处。",
            "rewritten_query": "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？",
            "strategy": "none", "focus_entity": "武汉力源信息技术股份有限公司", "refused": False}
    print(json.dumps(judge_case(load_multiturn_cases()[4], demo), ensure_ascii=False, indent=1))
