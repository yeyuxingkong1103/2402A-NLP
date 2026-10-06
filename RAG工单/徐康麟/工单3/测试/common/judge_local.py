# -*- coding: utf-8 -*-
"""判分口径（§2.1 五步）的三级取用：产品 bridge → 工单1 权威 Evaluator（只读）→ 本地副本。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

背景（实测）：产品侧冻结模块 ``研发/app/core/evaluator_bridge.py``（``设计/接口设计.md`` §3.23）
在本轮交付中**不存在**，因此 ``assertions.judge_answer`` 无法走产品口径。为了不让「14 题准确率」
失去判据，这里按 §3.23 规定的降级链取判分口径，并**始终标注来源**：

    ① ``app.core.evaluator_bridge.check_answer`` —— 产品冻结口径（首选）；
    ② 工单1 ``研发/app/core/evaluator.py::Evaluator.check_answer`` —— 权威口径本体，
       **只读**动态加载（``sys.dont_write_bytecode = True``，绝不在工单1 下写任何文件）；
    ③ 本地副本 ``check_answer_local`` —— §2.1 五步口径的等价实现（仅当 ①② 都不可用时使用）。

口径（``设计/验收标准.md`` §2.1，``FUZZY_THRESHOLD = 0.62``）：
    ① 参考答案是答案的子串（忽略标点/空白，双向）；
    ② 参考答案中的**全部数值**命中（含千分位/单位归一）；
    ③ 关键实体（引号内专有名称）有交集；
    ④ 比例（百分比）全部命中；
    ⑤ 在不缺任何数字的前提下，字符二元组 Jaccard ≥ 0.62；否则判错。
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import paths

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
FUZZY_THRESHOLD = 0.62

NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
ALL_NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
ENTITY_PATTERN = re.compile(r"[“\"]([^”\"]{4,60})[”\"]")
PUNCT_TO_STRIP = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"


def _normalize(text: str) -> str:
    """归一化：统一逗号/括号/百分号并去空白（与工单1 Evaluator 同口径）。"""
    body = str(text or "")
    for old, new in (("，", ","), ("（", "("), ("）", ")"), ("％", "%"), (" ", ""), ("\u3000", "")):
        body = body.replace(old, new)
    return body.strip()


def _bigrams(text: str) -> set[str]:
    """字符二元组集合（中文无需分词即可度量字面重叠）。"""
    cleaned = "".join(char for char in text if char not in PUNCT_TO_STRIP)
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard 相似度。"""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _bigram_number(raw: str) -> str:
    """数字字面量规范化（去千分位、去多余小数零）。"""
    try:
        value = float(str(raw).replace(",", ""))
    except ValueError:
        return str(raw)
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def normalize_number_strings(text: str) -> set[str]:
    """把文本里的**带单位**数值归一成可比较集合（万元/亿元/美元 → 元）。

    与工单1 ``app/core/number_utils.py::normalize_number_strings`` 同口径：**只收录带单位的数字**。
    若把裸数字也收进来，「5 项」里的 5 会被当成金额，导致「答对但未写『5 项』」被误判为错
    ——本副本初版正是这样，导致题 2 被误判（已按权威实现修正）。
    """
    factors = {"亿美元": 1e8, "亿元": 1e8, "万美元": 1e4, "万元": 1e4, "元": 1.0, "美元": 1.0}
    out: set[str] = set()
    for match in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(亿美元|亿元|万美元|万元|元|美元)", text or ""):
        raw, unit = match.group(1).replace(",", ""), match.group(2)
        try:
            value = float(raw)
        except ValueError:                                     # pragma: no cover - 正则已限数字
            continue
        out.add(f"{value * factors.get(unit, 1.0):.6f}")
    return out


def check_answer_local(answer: str, golden: str) -> tuple[bool, str]:
    """§2.1 五步口径的本地副本（返回 ``(是否正确, 判分说明)``）。"""
    if not answer or not golden:
        return False, "答案或参考答案为空"
    norm_answer, norm_golden = _normalize(answer), _normalize(golden)

    # ① 忽略标点的包含关系（双向）
    plain_answer = "".join(char for char in norm_answer if char not in PUNCT_TO_STRIP)
    plain_golden = "".join(char for char in norm_golden if char not in PUNCT_TO_STRIP)
    if plain_golden and plain_golden in plain_answer:
        return True, "参考答案为答案子串（忽略标点）"
    if plain_answer and plain_answer in plain_golden:
        return True, "答案为参考答案子串（忽略标点）"

    # ② 金额全部命中（含单位折算与等价写法）
    golden_amounts = normalize_number_strings(norm_golden)
    answer_amounts = normalize_number_strings(norm_answer)
    if golden_amounts and golden_amounts.issubset(answer_amounts):
        return True, f"金额全部命中({len(golden_amounts)}个)"

    # ③ 关键实体命中
    golden_entities = set(ENTITY_PATTERN.findall(golden))
    answer_entities = set(ENTITY_PATTERN.findall(answer))
    if golden_entities and golden_entities & answer_entities:
        return True, "关键实体命中"

    # ④ 比例全部命中
    golden_percents = {pct.replace(" ", "") for pct in PERCENT_PATTERN.findall(norm_golden)}
    answer_percents = {pct.replace(" ", "") for pct in PERCENT_PATTERN.findall(norm_answer)}
    if golden_percents and golden_percents.issubset(answer_percents):
        return True, "比例全部命中"

    # ⑤ 模糊相似度：必须在「不缺任何数字」的前提下才启用（防止漏答一半被判对）
    golden_all = {_bigram_number(m) for m in ALL_NUMBER_PATTERN.findall(norm_golden)}
    answer_all = {_bigram_number(m) for m in ALL_NUMBER_PATTERN.findall(norm_answer)}
    missing_all = golden_all - answer_all
    missing_amounts = golden_amounts - answer_amounts
    if not missing_amounts and not missing_all:
        similarity = _jaccard(_bigrams(norm_answer), _bigrams(norm_golden))
        if similarity >= FUZZY_THRESHOLD:
            return True, f"字符二元组相似度 {similarity:.2f} ≥ {FUZZY_THRESHOLD}"
    missing = sorted(missing_all) or sorted(missing_amounts)
    return False, (f"未命中：缺少数值 {missing[:6]}" if missing else "未命中：文本与数值均不匹配")


@lru_cache(maxsize=1)
def load_authoritative() -> Any:
    """只读加载工单1 的权威 ``Evaluator``，返回可直接调用的 ``check_answer``（失败返回 ``None``）。

    实现要点：工单1 的判分是 ``Evaluator.check_answer(self, answer, golden)`` **实例方法**
    （模块级没有同名函数 —— 这正是本副本首版加载失败的原因）。判分只用到 ``self._normalize``
    （staticmethod），因此用 ``object.__new__`` 造一个**不跑 __init__** 的实例：既拿到权威口径，
    又不会去连库/写盘。红线：全程 ``sys.dont_write_bytecode = True``，绝不在工单1 落任何文件。
    """
    reference = paths.REFERENCE_DIRS[0] / "研发"
    if not (reference / "app" / "core" / "evaluator.py").is_file():
        return None
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                              # 红线：不得在工单1 写 .pyc
    try:
        if str(reference) not in sys.path:
            sys.path.insert(0, str(reference))
        from app.core.evaluator import Evaluator              # noqa: PLC0415

        instance = object.__new__(Evaluator)                  # 不触发 __init__（不连库、不写盘）
        return instance.check_answer
    except Exception:                                            # noqa: BLE001 —— 显式降级
        return None
    finally:
        sys.dont_write_bytecode = previous


def judge(answer_text: str, golden: str) -> tuple[bool, str, str]:
    """返回 ``(是否正确, 判分说明, 口径来源)`` —— 来源必须写进测试报告。"""
    paths.ensure_dev_on_path()
    # ① 产品冻结口径
    try:
        from app.core import evaluator_bridge                        # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 —— 模块缺失属显式降级，来源里注明
        bridge_error = f"{type(exc).__name__}: {exc}"
    else:
        try:
            ok, reason = evaluator_bridge.check_answer(answer_text, golden)
            return bool(ok), str(reason), "产品口径 app.core.evaluator_bridge"
        except Exception as exc:  # noqa: BLE001 —— 动态加载失败 → 降级并注明
            bridge_error = f"{type(exc).__name__}: {exc}"
    # ② 工单1 权威口径（只读）
    caller = load_authoritative()
    if caller is not None:
        try:
            ok, reason = caller(answer_text, golden)
            return bool(ok), str(reason), f"工单1 权威 Evaluator（只读加载；产品 bridge 不可用：{bridge_error}）"
        except Exception as exc:  # noqa: BLE001
            bridge_error = f"{bridge_error}；工单1 调用失败 {type(exc).__name__}: {exc}"
    # ③ 本地副本
    ok, reason = check_answer_local(answer_text, golden)
    return bool(ok), f"{reason}（判分口径来源：测试侧本地副本；{bridge_error}）", "测试侧本地副本（§2.1 五步）"


def judge_source_available() -> tuple[bool, str]:
    """产品冻结判分模块是否可用 ``(可用, 说明)`` —— 供报告判定「口径是否合规」。"""
    paths.ensure_dev_on_path()
    module_path = paths.DEV_DIR / "app" / "core" / "evaluator_bridge.py"
    try:
        from app.core import evaluator_bridge  # noqa: PLC0415,F401
    except Exception as exc:  # noqa: BLE001
        return False, (f"缺少冻结模块 {paths.display(module_path)}（设计/接口设计.md §3.23）："
                       f"{type(exc).__name__}: {exc}")
    return True, paths.display(module_path)


def placeholder_negative_cases() -> list[tuple[str, str, bool, str]]:
    """§2.1 的强制负例（N-1/N-2/N-3）：``(名称, 答案, 期望判对?, golden)``。"""
    return [
        ("N-1 题33 只答 82.10%", "82.10%", False,
         "报告期内，公司来自军用领域的收入占主营业务收入的比重分别为82.10%、97.31%、94.84%和94.34%。"),
        ("N-2 题543 答 5520 万元（缺千分位/小数）", "注册资本 5520 万元", True, "注册资本为5,520万元。"),
        ("N-3 题95 答 未知。", "未知。", False,
         "公司参与制定了全军第一个视频指挥系统技术标准（即2019年制订的《某视频指挥系统技术规范（1.0版）》）。"),
    ]


def negative_case_report() -> list[dict[str, Any]]:
    """跑一遍强制负例，返回逐条结果（供离线用例断言口径未放宽）。"""
    rows: list[dict[str, Any]] = []
    for name, answer, expected, golden in placeholder_negative_cases():
        ok, reason, source = judge(answer, golden)
        rows.append({"case": name, "answer": answer, "expected": expected,
                     "judged": bool(ok), "reason": reason, "source": source,
                     "passed": bool(ok) is bool(expected)})
    return rows


def _unused(_: Path) -> None:  # pragma: no cover - 保留类型引用，避免误删 Path 导入
    """占位函数（保持 ``Path`` 导入语义清晰）。"""
    return None
