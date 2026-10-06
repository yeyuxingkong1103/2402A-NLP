# -*- coding: utf-8 -*-
"""工单3 判分口径桥（设计/接口设计.md §3.23 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

职责：把**工单1 的权威 Evaluator.check_answer**（`FUZZY_THRESHOLD = 0.62`）只读复用到本工单，
供在线/用户级测试与评估报告使用同一套判分口径，避免「产品一套、测试一套」。

```python
def load_evaluator() -> Any        # 动态加载 工单1 的 Evaluator（只读）
def check_answer(answer_text, golden) -> tuple[bool, str]          # 首选权威口径，失败降级本地副本
def check_answer_local(answer_text, golden) -> tuple[bool, str]    # 口径副本（离线兜底）
def FUZZY_THRESHOLD() -> float     # 0.62
```

只读纪律（§3.23 硬约束）：
    * 调用前设 ``sys.dont_write_bytecode = True`` 并给子进程传 ``PYTHONDONTWRITEBYTECODE=1``；
    * **不在** ``工单1`` 下创建任何文件（含 ``__pycache__``）；子进程里用 ``object.__new__(Evaluator)``
      绕过 ``__init__``（其 ``__init__`` 会 ``mkdir`` 结果目录并打开 SQLite，属写入行为，必须避免）。

为什么走子进程：工单1 与工单3 的包名都叫 ``app``，本进程已 ``import app``（工单3），
再 ``from app.core.evaluator import ...`` 必然解析到工单3 的包（实测 ImportError）。
子进程在工单1 的 ``sys.path`` 下独立导入，才是**真·权威口径**，且互不污染。

日志事件：``evaluator.load``(source,ok,elapsed_ms) / ``evaluator.check``(is_correct,reason,source) /
``evaluator.degrade``(reason,fallback)。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 工单1 源码与解释器（只读引用；不得写入）
GONGDAN1_ROOT = Path(r"E:\gao6gongdan\工单1")
GONGDAN1_DEV = GONGDAN1_ROOT / "研发"
FROZEN_FUZZY_THRESHOLD = 0.62
_SUBPROCESS_TIMEOUT_S = 60.0

# 与工单1 `app/core/evaluator.py` 完全同源的判定常数（口径副本用）
PUNCT_TO_STRIP = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"
NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
ENTITY_PATTERN = re.compile(r"[“\"]([^”\"]{4,60})[”\"]")
ALL_NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
PERCENT_ONLY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# 子进程脚本：只读加载工单1 Evaluator，规避 __init__ 的写入行为
_FROZEN_SCRIPT = r"""
import json, sys
sys.dont_write_bytecode = True                      # 不向工单1 写 .pyc
sys.path.insert(0, r"{dev}")
from app.core.evaluator import Evaluator            # 工单1 的权威实现
proxy = object.__new__(Evaluator)                   # 不调 __init__：避免 mkdir/SQLite 写入
ok, reason = Evaluator.check_answer(proxy, sys.argv[1], sys.argv[2])
print(json.dumps({{"ok": bool(ok), "reason": str(reason)}}, ensure_ascii=False))
"""


def _lazy_logger(logger: Any, module: str = "evaluator_bridge") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class FrozenJudgeInfo:
    """权威判分器的探测结果（供报告标注口径来源）。"""

    ok: bool
    source: str
    threshold: float
    elapsed_ms: float
    error: str = ""


def FUZZY_THRESHOLD() -> float:
    """模糊相似度阈值（来源：工单1 ``evaluator.py``，固定 0.62）。"""
    return FROZEN_FUZZY_THRESHOLD


# ---------------------------------------------------------------------------
# 口径副本（离线兜底）：与工单1 `Evaluator.check_answer` 同源的五步判定
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """归一化：去空白、统一逗号与括号（与工单1 同名实现一致）。"""
    if not text:
        return ""
    return (str(text).replace("，", ",").replace("（", "(").replace("）", ")")
            .replace("％", "%").replace(" ", "").replace("\u3000", "").strip())


def _bigrams(text: str) -> set[str]:
    """字符二元组集合（中文无需分词即可度量字面重叠）。"""
    cleaned = "".join(char for char in text if char not in PUNCT_TO_STRIP)
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def _bigram_number(raw: str) -> str:
    """数字字面量规范化（去千分位与多余小数零）。"""
    try:
        value = float(str(raw).replace(",", ""))
    except ValueError:
        return str(raw)
    return str(int(value)) if value == int(value) else f"{value:.6f}".rstrip("0").rstrip(".")


def _jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard 相似度。"""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _amounts(text: str) -> set[str]:
    """金额集合（元为单位的规范值；这里退化为「规范化数字字面量」集合）。"""
    return {_bigram_number(match) for match in NUMBER_PATTERN.findall(text)}


def check_answer_local(answer_text: str, golden: str) -> tuple[bool, str]:
    """口径副本：与工单1 ``Evaluator.check_answer`` 等价的确定性五步判定（仅当权威不可用时使用）。"""
    if not answer_text or not golden:
        return False, "答案或参考答案为空（本地副本）"
    norm_answer, norm_golden = _normalize(answer_text), _normalize(golden)
    plain_answer = "".join(ch for ch in norm_answer if ch not in PUNCT_TO_STRIP)
    plain_golden = "".join(ch for ch in norm_golden if ch not in PUNCT_TO_STRIP)
    if plain_golden and plain_golden in plain_answer:
        return True, "参考答案为答案子串（忽略标点；本地副本）"
    if plain_answer and plain_answer in plain_golden:
        return True, "答案为参考答案子串（忽略标点；本地副本）"
    golden_amounts, answer_amounts = _amounts(norm_golden), _amounts(norm_answer)
    if golden_amounts and golden_amounts.issubset(answer_amounts):
        return True, f"金额全部命中({len(golden_amounts)}个；本地副本)"
    golden_entities = set(ENTITY_PATTERN.findall(golden))
    if golden_entities and golden_entities & set(ENTITY_PATTERN.findall(answer_text)):
        return True, "关键实体命中（本地副本）"
    golden_percents = {p.replace(" ", "") for p in PERCENT_PATTERN.findall(norm_golden)}
    if golden_percents and golden_percents.issubset({p.replace(" ", "")
                                                     for p in PERCENT_PATTERN.findall(norm_answer)}):
        return True, "比例全部命中（本地副本）"
    golden_all = {_bigram_number(m) for m in ALL_NUMBER_PATTERN.findall(norm_golden)}
    answer_all = {_bigram_number(m) for m in ALL_NUMBER_PATTERN.findall(norm_answer)}
    if not (golden_amounts - answer_amounts) and not (golden_all - answer_all):
        similarity = _jaccard(_bigrams(norm_answer), _bigrams(norm_golden))
        if similarity >= FROZEN_FUZZY_THRESHOLD:
            return True, f"字符二元组相似度 {similarity:.2f} ≥ {FROZEN_FUZZY_THRESHOLD}（本地副本）"
    missing = sorted((golden_all - answer_all))[:5]
    return False, f"未命中（本地副本）：缺少数值 {missing}" if missing else "未命中（本地副本）：文本与数值均不匹配"


# ---------------------------------------------------------------------------
# 权威口径（子进程隔离，只读）
# ---------------------------------------------------------------------------
class _FrozenEvaluator:
    """工单1 权威 Evaluator 的只读代理（每次调用起一个隔离子进程）。"""

    def __init__(self, *, python: str | None = None, logger: Any = None) -> None:
        self.python = python or sys.executable
        self.log = _lazy_logger(logger)
        self.cache: dict[tuple[str, str], tuple[bool, str]] = {}
        self.source = "工单1 app.core.evaluator.Evaluator.check_answer（只读子进程）"

    def _script(self) -> str:
        """拼装子进程脚本（路径注入，避免转义问题）。"""
        return _FROZEN_SCRIPT.format(dev=str(GONGDAN1_DEV))

    def check_answer(self, answer_text: str, golden: str) -> tuple[bool, str]:
        """调用权威判分（带进程内缓存）；失败抛出异常由上层降级。"""
        key = (str(answer_text), str(golden))
        if key in self.cache:
            return self.cache[key]
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"                 # 只读保护：不写 .pyc
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [self.python, "-c", self._script(), str(answer_text), str(golden)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_SUBPROCESS_TIMEOUT_S, env=env, cwd=str(GONGDAN1_DEV),
        )
        if completed.returncode != 0:
            raise RuntimeError(f"权威判分子进程退出码 {completed.returncode}："
                               f"{(completed.stderr or '')[-300:]}")
        line = next((ln for ln in reversed((completed.stdout or "").splitlines()) if ln.strip().startswith("{")),
                    "")
        if not line:
            raise RuntimeError(f"权威判分子进程无 JSON 输出：{(completed.stdout or '')[-200:]}")
        payload = json.loads(line)
        result = (bool(payload.get("ok")), str(payload.get("reason") or ""))
        self.cache[key] = result
        return result

    def check_answer_batch(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[bool, str]]:
        """批量判分（逐条调用，命中缓存即复用）。"""
        return [self.check_answer(answer, golden) for answer, golden in pairs]


_instance: _FrozenEvaluator | None = None
_info: FrozenJudgeInfo | None = None


def load_evaluator(*, logger: Any = None, probe: tuple[str, str] | None = None) -> Any:
    """动态加载工单1 的 Evaluator（只读）；返回代理对象，不可用时抛 ``RuntimeError``。

    ``probe`` 默认用一对已知正确的样本做连通性探测（探测结果缓存，日志 ``evaluator.load``）。
    """
    global _instance, _info
    import time

    log = _lazy_logger(logger)
    started = time.perf_counter()
    with log.enter("load_evaluator", {"gongdan1_dev": str(GONGDAN1_DEV), "exists": GONGDAN1_DEV.exists()}) as span:
        if not GONGDAN1_DEV.exists():
            raise RuntimeError(f"工单1 源码目录不存在：{GONGDAN1_DEV}")
        if _instance is None:
            _instance = _FrozenEvaluator(logger=log)
        sample = probe or ("公司的注册资本为 5,520.00 万元。", "注册资本为5,520万元。")
        try:
            ok, reason = _instance.check_answer(*sample)
            _info = FrozenJudgeInfo(ok=True, source=_instance.source, threshold=FROZEN_FUZZY_THRESHOLD,
                                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            log.log_event("evaluator.load", source=_instance.source, ok=True,
                          probe_ok=ok, probe_reason=reason,
                          elapsed_ms=_info.elapsed_ms)
            span.set_output({"ok": True, "probe": ok})
            return _instance
        except Exception as exc:  # noqa: BLE001 —— 加载失败必须留痕并降级（不静默）
            _info = FrozenJudgeInfo(ok=False, source="本地副本 check_answer_local",
                                    threshold=FROZEN_FUZZY_THRESHOLD,
                                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                                    error=f"{type(exc).__name__}: {exc}")
            log.log_event("evaluator.load", level="ERROR", source="本地副本", ok=False,
                          error_type=type(exc).__name__, message=str(exc),
                          fallback="改用 check_answer_local（口径副本）")
            span.set_output({"ok": False, "error": str(exc)})
            raise


def check_answer(answer_text: str, golden: str, *, logger: Any = None) -> tuple[bool, str]:
    """判分首选口径：工单1 权威 Evaluator（只读子进程）；不可用时**显式降级**为口径副本。"""
    log = _lazy_logger(logger)
    global _info
    with log.enter("check_answer", {"answer_chars": len(str(answer_text or "")),
                                    "golden_chars": len(str(golden or ""))}) as span:
        try:
            evaluator = load_evaluator(logger=log)
            ok, reason = evaluator.check_answer(answer_text, golden)
            log.log_event("evaluator.check", is_correct=bool(ok), reason=str(reason)[:80],
                          source=(_info.source if _info else "工单1 权威口径"))
            span.set_output({"ok": bool(ok), "source": "frozen"})
            return bool(ok), f"{reason}（判分口径来源：工单1 权威 Evaluator {FROZEN_FUZZY_THRESHOLD}）"
        except Exception as exc:  # noqa: BLE001 —— 降级路径必须留痕，返回口径副本结果
            log.log_event("evaluator.degrade", level="WARNING", error_type=type(exc).__name__,
                          message=str(exc), fallback="check_answer_local（口径副本）")
            ok, reason = check_answer_local(answer_text, golden)
            span.set_output({"ok": bool(ok), "source": "local_copy"})
            return bool(ok), f"{reason}（判分口径来源：本地副本；{type(exc).__name__}: {exc}）"


def cross_check(pairs: Sequence[tuple[str, str]] | None = None, *,
                logger: Any = None) -> list[dict[str, Any]]:
    """与工单1 的对照断言：逐对比较「权威口径」与「本地副本」，供报告标注一致性。"""
    log = _lazy_logger(logger)
    samples = list(pairs or [
        ("注册资本为5,520万元。", "注册资本为 5,520 万元。"),
        ("1,670万股，占发行后总股本的比例为25.04%", "发行 1,670 万股，占发行后总股本的比例为 25.04%。"),
        ("电子信息行业的上游涉及信息系统相关的电子元器件制造企业，以及机箱、机柜等金属壳体制造企业。",
         "电子信息行业的上游涉及信息系统相关的电子元器件制造企业，以及机箱、机柜等金属壳体制造企业，竞争充分、采购便利。"),
        ("赵马克 42.35% 公司控股股东", "赵马克（Mark Zhao），持股 42.35%，为公司控股股东（及实际控制人）。"),
        ("银河系外文明的量子计算机专利申请数量是多少", "注册资本为 5,520 万元。"),
    ])
    rows: list[dict[str, Any]] = []
    with log.enter("cross_check", {"pairs": len(samples)}) as span:
        for index, (answer, golden) in enumerate(samples, start=1):
            frozen_ok, frozen_reason = check_answer(answer, golden, logger=log)
            local_ok, local_reason = check_answer_local(answer, golden)
            rows.append({"pair": index, "frozen_ok": frozen_ok, "frozen_reason": frozen_reason[:80],
                         "local_ok": local_ok, "local_reason": local_reason[:80],
                         "agree": frozen_ok == local_ok})
        agree = sum(1 for row in rows if row["agree"])
        log.log_event("evaluator.cross_check", pairs=len(rows), agree=agree)
        span.set_output({"pairs": len(rows), "agree": agree})
    return rows


def judge_info() -> FrozenJudgeInfo | None:
    """最近一次 ``load_evaluator`` 的口径来源信息（未探测过返回 None）。"""
    return _info


__all__ = ["load_evaluator", "check_answer", "check_answer_local", "FUZZY_THRESHOLD", "cross_check",
           "judge_info", "FrozenJudgeInfo", "FROZEN_FUZZY_THRESHOLD"]
