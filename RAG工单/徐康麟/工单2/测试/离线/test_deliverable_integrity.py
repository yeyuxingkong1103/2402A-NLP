"""交付完整性守卫：对比报告与事实必须一致（防止交付物被冒烟运行覆盖）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：测试 / 离线（交付完整性守卫，T4 评审发现 F1 的回归用例）
提出者：designer（t4 批判性评审）

## 本用例守的是什么（务必读清定位）

本用例**不是性能回归检测**，而是**交付完整性守卫**：它守的是「**报告与事实一致**」，
即「对比报告声称的题目数与准确率，必须与真实运行的逐题证据一致」。

**它钉住的真实失效（2026-10-03 实测，评审发现）**：
`优化/脚本/compare_optimization.py --limit 1` 被当作冒烟运行使用，而该脚本**默认写入正式交付路径**，
于是四个核心交付物在同秒（16:55:34）被 1 题运行覆盖：
- ``accuracy_report.json`` 的 ``after.count`` 变成 1、``per_question`` 只剩 1 条；
- ``optimization_compare.md`` 出现「答案准确率（**10 题**）0.5000 → **1.0000**」与
  §5.1「1.00（**1/1**）」**同文件互相矛盾**的表述，并把 **10 题口径的 before 与 1 题口径的 after 混比**；
- ``optimization_compare.csv`` / ``ragas_report.md`` 同步被覆盖。

该失效**在代码与测试全绿的情况下静默发生**，直到人工评审才发现——因此必须由用例钉住。

## 断言口径（4 条，对应工单 §7 与 设计/验收标准 验收1/验收7）

1. ``accuracy_report.json``：``after.count == 10`` 且 ``per_question`` 长度 == 10；
2. ``optimization_compare.md``：**题目数表述全文一致**（不得一处「10 题」、另一处「1/1」）；
3. ``optimization_compare.md`` / ``optimization_compare.csv`` 与 ``eval_records.json`` 的
   accuracy 一致（同为 0.90，即 9/10，恰达工单 ≥90% 门槛）；
4. ``优化/评估结果/`` 与 ``优化/脚本/`` 下**不存在任何以 ``_`` 开头的文件或子目录**
   （文件豁免 ``__init__.py``、目录豁免 ``__pycache__``）。

## 纪律

- 只**读**交付物，不修改/不重写任何报告（修复由实现方完成，本用例只负责判红）；
- 本文件是**新增**用例，不改动 tester 既有任何断言；
- 失败时的报错信息给出「实际值 + 修复命令」，避免后人误以为是性能问题。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_RESULTS = PROJECT_ROOT / "优化" / "评估结果"
ACCURACY_REPORT = EVAL_RESULTS / "accuracy_report.json"
COMPARE_MD = EVAL_RESULTS / "optimization_compare.md"
COMPARE_CSV = EVAL_RESULTS / "optimization_compare.csv"
EVAL_RECORDS = EVAL_RESULTS / "eval_records.json"

#: 交付物必须覆盖的工单问题数（golden_qa.jsonl 的 10 条，工单 §5-1 / §12-1）
EXPECTED_QUESTIONS = 10
#: 工单 ≥90% 门槛下本次实测的准确率（9/10；Q95 为已裁定的方案 A 遗留，不作为失败原因）
EXPECTED_ACCURACY = 0.90
#: 交付目录（正式命名产物/脚本的归档处；不得出现下划线前缀的临时件）
GUARDED_DIRS = ("优化/评估结果", "优化/脚本")
#: 下划线前缀文件的唯一豁免
UNDERSCORE_EXEMPT = ("__init__.py",)
#: 下划线前缀**目录**的豁免（``__pycache__`` 是 Python 自身产物，非临时件命名习惯）
UNDERSCORE_DIR_EXEMPT = ("__pycache__",)

_REPAIR_HINT = (
    "修复方式（实现方执行，本用例只判红）：\n"
    "  全量重跑（不得带 --limit）：\n"
    "    pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py\n"
    "  并给该脚本加防覆盖保护：--limit/部分运行只许写临时目录，不得写 优化/评估结果/。"
)


# ==========================================================================
# 辅助：从三类产物中抽取判分（口径：确定性 check_answer，阈值 0.62）
# ==========================================================================
def _load_json(path: Path) -> dict:
    """读取 JSON；文件缺失或损坏时给出明确的修复指引，而不是模糊报错。"""
    assert path.exists(), f"缺少交付物 {path}（{_REPAIR_HINT}）"
    return json.loads(path.read_text(encoding="utf-8"))


def _accuracy_row(text: str) -> tuple[str, list[str]] | None:
    """定位「答案准确率」所在的表格行，返回 ``(行文本, 各列)``。

    容错：表格首列可能被省略（只写「准确率」），也可能写成「答案准确率（10 题）」。
    """
    for line in text.splitlines():
        if "准确率" in line and "|" in line and "---" not in line:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 3:
                return line.strip(), cells
    return None


def _parse_ratio(raw: str) -> float | None:
    """把 ``0.9000`` / ``90%`` / ``9/10`` 三种写法解析为 0~1 的小数。"""
    text = (raw or "").strip()
    fraction = re.search(r"\b(\d+)\s*/\s*(\d+)\b", text)
    if fraction:
        denom = float(fraction.group(2))
        return float(fraction.group(1)) / denom if denom else None
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if percent:
        return float(percent.group(1)) / 100.0
    decimal = re.search(r"(\d+(?:\.\d+)?)", text)
    if not decimal:
        return None
    value = float(decimal.group(1))
    return value / 100.0 if value > 1.0 else value


def _accuracy_from_md(text: str) -> tuple[float | None, int | None, str]:
    """从 md 的「答案准确率…」行抽取 ``(优化后准确率, 分母题数, 标签)``。

    注意取的是「**优化后**」列（第 3 列），不是「优化前」列——这正是评审第一版
    正则写错的地方（把 0.5000 当成优化后值），故此处把列序写死并做打印留痕。
    """
    row = _accuracy_row(text)
    if row is None:
        return None, None, ""
    line, cells = row
    label, after = cells[0], cells[2]
    total = re.search(r"(\d+)\s*题", label)
    return _parse_ratio(after), (int(total.group(1)) if total else None), label


def _accuracy_from_csv(text: str) -> float | None:
    """从长表 CSV 的 ``summary,ALL,,accuracy,before,after,delta,note`` 行取 after 值。"""
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) >= 6 and parts[0] == "summary" and parts[3] == "accuracy":
            try:
                return float(parts[5])
            except ValueError:
                return None
    return None


def _accuracy_from_eval_records() -> tuple[float, int]:
    """由逐题记录**重算**准确率（事实侧真值，不引用任何报告里的数字）。"""
    payload = _load_json(EVAL_RECORDS)
    records = payload.get("records") or []
    assert records, f"{EVAL_RECORDS} 无 records（{_REPAIR_HINT}）"
    correct = sum(1 for item in records if item.get("is_correct"))
    return correct / len(records), len(records)


# ==========================================================================
# 1 / 4：机器可读产物的题目数 与 交付目录整洁度
# ==========================================================================
def test_accuracy_report_covers_all_ten_questions():
    """交付汇总必须覆盖 10 题：``after.count == 10`` 且 ``per_question`` 长度 == 10。

    这一条正是 1 题冒烟运行会打破的断言（失效时实测 count=1、per_question=1）。
    """
    payload = _load_json(ACCURACY_REPORT)
    after = payload.get("after") or {}
    per_question = payload.get("per_question") or []
    count = after.get("count")
    print(f"\n[交付完整性] accuracy_report.json：after.count={count}、per_question 长度={len(per_question)}")
    assert count == EXPECTED_QUESTIONS, (
        f"accuracy_report.json 的 after.count={count!r}，应为 {EXPECTED_QUESTIONS}——"
        f"交付汇总被部分运行覆盖。\n{_REPAIR_HINT}"
    )
    assert len(per_question) == EXPECTED_QUESTIONS, (
        f"accuracy_report.json 的 per_question 只有 {len(per_question)} 条，应为 {EXPECTED_QUESTIONS} 条。\n{_REPAIR_HINT}"
    )


def underscore_prefixed_files(directory: Path) -> list[str]:
    """**递归**返回目录下以 ``_`` 开头的文件（豁免 ``__init__.py``），路径相对 ``directory``。

    为什么必须递归（2026-10-03 第 2 轮评审 finding #1，captain 实测确认）：
    原实现用 ``iterdir()`` **只看一层**，子目录里的 ``_nested_probe.py`` 之类会**漏网**。
    "交付目录不得有临时命名件"是**规则式**判据，只覆盖顶层等于留了半个口子。
    """
    if not directory.exists():
        return []
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name.startswith("_") and path.name not in UNDERSCORE_EXEMPT
    )


def underscore_prefixed_dirs(directory: Path) -> list[str]:
    """**递归**返回目录下以 ``_`` 开头的子目录（豁免 ``__pycache__``），路径相对 ``directory``。

    与文件规则同源：临时件命名习惯同样会用在目录上（如 ``_smoke/``、``sub/_scratch_dir/``）；
    范围**只限交付目录** ``优化/评估结果`` 与 ``优化/脚本``——
    工作区根目录的 ``.tmp_review/``（点号前缀）按 captain 裁定保留到第 2 轮复审后，不在此判据内。
    """
    if not directory.exists():
        return []
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_dir() and path.name.startswith("_") and path.name not in UNDERSCORE_DIR_EXEMPT
    )


def test_no_temporary_run_artifacts_in_eval_results():
    """交付目录不得残留**任何以下划线开头的文件或子目录**（规则式判据，**递归**扫描）。

    ## 为什么从"黑名单"改成"规则式"（2026-10-03，captain 裁定，tester 执行）

    原判据只认 ``_v3_`` / ``_tmp`` / ``_probe`` 三种前缀。captain 全仓扫描发现归档队伍遗留的
    ``优化/脚本/_t9_verify_judge_and_citation.py`` **三种前缀一个都不匹配** ——
    于是「临时件已清零」只对"已知命名习惯"成立，对"任意临时命名"是**假结论**。
    正式命名的脚本与产物不该以下划线开头，故改为规则式判据。

    ## 为什么还要**递归**（第 2 轮评审 finding #1，captain 实测确认）

    规则式判据若只用 ``iterdir()``，子目录里的 ``_nested_probe.py`` 依然会**漏网** ——
    规则式判据必须**覆盖整棵子树**才成立。本用例末尾的\"嵌套自检\"即该缺口的回归证据
    （种子树里放 ``sub/_nested_probe.py`` 与 ``sub/_scratch_dir/``：应被抓；
    ``sub/formal_script.py``、``sub/__init__.py``、``sub/__pycache__``：应放行）。

    ## 本次处置（同一裁定的一部分）

    ``_t9_verify_judge_and_citation.py`` → **转正**为 ``优化/脚本/verify_judge_and_citation.py``
    （内容未变，仅去下划线前缀 + 同步 docstring 用法路径）。选择转正而非删除的理由：
    它是**独立核验脚本**（判分口径负例 + 引用页码可回查 chunk），删掉会丢证据价值；
    与 verifier 的 ``verify_evaluator_gauge.py`` / ``verify_citation_integrity.py`` 有部分重叠，
    但重叠部分属"两方独立复算"，按纪律**不做合并**（合并会减少独立证据）。
    """
    report: list[str] = []
    offenders: list[str] = []
    for rel in GUARDED_DIRS:
        directory = PROJECT_ROOT / rel
        files = underscore_prefixed_files(directory)
        dirs = underscore_prefixed_dirs(directory)
        report.append(f"{rel}: 文件{files or '（无）'} / 目录{dirs or '（无）'}")
        offenders += [f"{rel}/{name}（文件）" for name in files]
        offenders += [f"{rel}/{name}/（目录）" for name in dirs]
    print("\n[交付完整性] 下划线前缀文件/目录**递归**扫描 → " + "；".join(report))

    # ---- 嵌套自检（finding #1 回归证据；种子树建在工作区内并 finally 清理）----
    # 不用 pytest 的 tmp_path：本机系统临时目录被沙箱拒绝
    # （实测 PermissionError: ...\AppData\Local\Temp\dsh-*\pytest-of-*），会让整条用例 ERROR。
    probe = PROJECT_ROOT / "测试" / ".guard_probe_tmp"
    shutil.rmtree(probe, ignore_errors=True)
    try:
        (probe / "sub").mkdir(parents=True)
        (probe / "sub" / "_nested_probe.py").write_text("# 临时件", encoding="utf-8")
        (probe / "sub" / "formal_script.py").write_text("# 正式命名", encoding="utf-8")
        (probe / "sub" / "__init__.py").write_text("", encoding="utf-8")
        (probe / "sub" / "_scratch_dir").mkdir()
        (probe / "sub" / "__pycache__").mkdir()
        nested_files = underscore_prefixed_files(probe)
        nested_dirs = underscore_prefixed_dirs(probe)
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    print(f"[交付完整性] 嵌套自检种子树 → 命中文件{nested_files} / 命中目录{nested_dirs}"
          f"（期望 ['sub/_nested_probe.py'] / ['sub/_scratch_dir']；formal_script.py、__init__.py、__pycache__ 应放行）")
    assert nested_files == ["sub/_nested_probe.py"], (
        f"递归扫描未抓到嵌套临时件：{nested_files}（期望 ['sub/_nested_probe.py']）——"
        f"若为 [] 说明又退回了非递归扫描（finding #1 复发）"
    )
    assert nested_dirs == ["sub/_scratch_dir"], (
        f"递归扫描未抓到嵌套临时目录：{nested_dirs}（期望 ['sub/_scratch_dir']）"
    )

    assert not offenders, (
        f"交付目录残留下划线前缀文件/目录 {offenders}——临时件命名习惯（``_v3_*``/``_tmp*``/``_probe*``/``_t9_*``…）"
        f"不得进入 优化/评估结果/ 与 优化/脚本/（文件豁免 ``__init__.py``、目录豁免 ``__pycache__``；**递归到子树**）。\n"
        f"处置方式：**转正命名**（去掉 ``_`` 前缀并同步 docstring 用法路径）或删除；不得默默留着。\n"
        f"依据：工单 §1「产出物按五类归档」与 设计/验收标准 验收11。"
    )


# ==========================================================================
# 2：报告内部题目数表述一致（不得一处 10 题、另一处 1/1）
# ==========================================================================
def test_compare_md_question_count_statement_is_consistent():
    """对比报告的题目数表述必须全文一致，不得自相矛盾。"""
    text = COMPARE_MD.read_text(encoding="utf-8")
    value, total, label = _accuracy_from_md(text)
    print(f"\n[交付完整性] 对比表准确率行标签 = {label!r}（解析分母 {total} 题 / 优化后值 {value}）")
    assert label, f"{COMPARE_MD} 里找不到「准确率」行，报告结构已变（{_REPAIR_HINT}）"

    # 口径 A：题目数必须显式声明为 10 题（由题目数而非行文推断得出）
    declared_total = total if total is not None else 0
    assert declared_total == EXPECTED_QUESTIONS, (
        f"对比表未声明「{EXPECTED_QUESTIONS} 题」口径（实测标签 {label!r}，推出分母 {declared_total}）。\n{_REPAIR_HINT}"
    )
    # 口径 B：不得同时出现「10 题」与「1/1」这类互相矛盾的表述
    has_one_of_one = re.search(r"\b1\s*/\s*1\b", text) is not None
    has_ten = re.search(rf"{EXPECTED_QUESTIONS}\s*题", text) is not None
    assert not (has_one_of_one and has_ten), (
        f"{COMPARE_MD} 同时出现「{EXPECTED_QUESTIONS} 题」与「1/1」——报告自相矛盾（1 题运行覆盖了 10 题结论）。\n"
        f"{_REPAIR_HINT}"
    )
    # 口径 C：不得出现「只有 1 题」的结论式表述
    assert "单题 **1/1**" not in text, f"{COMPARE_MD} 仍保留「单题 1/1」的 1 题运行结论。\n{_REPAIR_HINT}"


# ==========================================================================
# 3：三处产物准确率一致（与逐题事实同源）
# ==========================================================================
def test_accuracy_is_consistent_across_reports_and_records():
    """md / csv / 逐题记录三处准确率必须一致（本次实测 0.90，即 9/10）。"""
    md_text = COMPARE_MD.read_text(encoding="utf-8")
    md_value, md_total, md_label = _accuracy_from_md(md_text)
    csv_value = _accuracy_from_csv(COMPARE_CSV.read_text(encoding="utf-8"))
    fact_value, fact_total = _accuracy_from_eval_records()

    print(
        f"\n[交付完整性] 准确率核对：md={md_value}（分母 {md_total} 题）、csv={csv_value}、"
        f"逐题重算={fact_value:.4f}（{round(fact_value * fact_total)}/{fact_total}）"
    )
    assert md_total == EXPECTED_QUESTIONS, (
        f"对比报告的准确率分母为 {md_total} 题，应为 {EXPECTED_QUESTIONS} 题。\n{_REPAIR_HINT}"
    )
    assert fact_total == EXPECTED_QUESTIONS, f"逐题证据只有 {fact_total} 条，应为 {EXPECTED_QUESTIONS} 条。"
    assert abs(fact_value - EXPECTED_ACCURACY) < 1e-6, (
        f"逐题重算准确率 {fact_value:.4f}，与工单门槛下的实测值 {EXPECTED_ACCURACY} 不符——"
        f"若这是真实的新结果，请先复核判分口径是否被改动（环境事实 §4.2 明令不得放宽）。"
    )
    assert md_value is not None and abs(md_value - fact_value) < 1e-6, (
        f"对比报告准确率 {md_value} 与逐题重算 {fact_value:.4f} 不一致。\n{_REPAIR_HINT}"
    )
    assert csv_value is not None and abs(csv_value - fact_value) < 1e-6, (
        f"optimization_compare.csv 的 summary.accuracy={csv_value} 与逐题重算 {fact_value:.4f} 不一致。\n{_REPAIR_HINT}"
    )
