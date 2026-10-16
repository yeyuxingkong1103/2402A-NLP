# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-Query 理解优化
文件：测试/多轮/run_multiturn_eval.py

用途：**工单5 多轮 Query 理解的端到端评测跑分器**（local 直调引擎 / http 打服务皆可），一次跑三套：
  ① 多轮序列       ← 测试/测试数据/工单5多轮用例.jsonl（权威金标 5 轮 / 2 session）
  ② 独立会话对照   ← 测试/测试数据/工单5多轮用例_独立会话.jsonl（每轮新 session，验证降级属预期）
  ③ 单轮回归 16 题 ← 测试/测试数据/golden_qa_16.jsonl（工单4 能力未退化）
  ④ 异常输入       ← 测试/测试数据/异常输入用例.jsonl（不崩溃 / 非空 / 不 5xx）

产物（默认 `测试/测试数据/多轮评测结果/<run-tag>/`）：
    results.jsonl        多轮逐轮明细（含每个判分维度）
    isolated.jsonl       独立会话对照逐轮明细
    summary.json         准确率 / 各维度通过率 / 策略分布 / 延迟统计 / run-meta
    report.md            可读报告（逐轮表格）
    regression16.json    单轮回归（准确率 + 逐题明细）
    abnormal.json        异常输入（逐条 + 汇总）
    run_meta.json        运行口径（预热、冷启动、缓存、版本）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测量纪律（**先定义后测量**）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
* **冷启动数字不算验收数字**：脚本先对第 1 轮做一次**预热**并单独记录 `cold_start_ms`，
  该次结果**不进入**正式统计；随后才按 session 顺序正式提问并计时（`--no-warmup` 为调试口径，报告须标注）。
* 延迟主口径 = **客户端 wall**（请求发出 → 完整答案返回）；服务端 `latency_ms.total` 作辅助列。
* **改写阶段单独计时**：取 `latency_ms.rewrite`，与检索/生成分列，便于定位多轮开销。
* 逐题 `≤3000ms` 单独判定并列出超线题号；统计量给 **P50 / P95 / max / min**，禁止只报平均值。
* 准确率主口径 `passed_turns / total_turns`；并列**会话整体口径**（该 session 全轮通过才算 1 个会话通过）。

运行：
    pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine local --run-tag t7-local
    pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine http --base-url http://127.0.0.1:8123

退出码：0 成功；2 引擎/判分库导入失败；3 服务不可用；4 运行中发现异常（详见 stderr 中文报错）。
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- 路径与延迟导入
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]                      # 仓库根 E:\gao6gongdan\工单5
DATA_DIR = ROOT / "测试" / "测试数据"
DEFAULT_OUT_ROOT = DATA_DIR / "多轮评测结果"
PLACEHOLDER = "未运行"

EXIT_OK, EXIT_IMPORT, EXIT_SERVICE, EXIT_RUNTIME = 0, 2, 3, 4


class EvalError(RuntimeError):
    """带中文指引的评测错误（不抛裸 traceback 给验收人）。"""


def _bootstrap_syspath() -> None:
    """让 `研发/app` 包与 `测试/` 下的判分库都可被 import（Windows 中文路径安全）。"""
    for p in (ROOT / "研发", ROOT):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def load_judge():
    """延迟加载 测试/测试数据/多轮判分.py（唯一确定性判分来源）。"""
    path = DATA_DIR / "多轮判分.py"
    if not path.exists():
        raise EvalError(f"找不到判分库：{path}\n"
                        f"→ 请确认 测试/测试数据/多轮判分.py 已就位（本脚本不自行实现判分逻辑）。")
    try:
        spec = importlib.util.spec_from_file_location("gongdan5_judge", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                          # type: ignore[union-attr]
        return mod
    except Exception as exc:                                  # noqa: BLE001
        raise EvalError(f"判分库加载失败：{path}\n→ 原因：{type(exc).__name__}: {exc}") from exc


def load_golden_lib():
    """延迟加载 测试/common/golden.py（工单4 判分逻辑，单轮回归复用）。"""
    path = ROOT / "测试" / "common" / "golden.py"
    if not path.exists():
        raise EvalError(f"找不到工单4 判分库：{path}\n→ 单轮回归无法进行，请确认仓库完整性。")
    try:
        spec = importlib.util.spec_from_file_location("gongdan5_golden", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                          # type: ignore[union-attr]
        return mod
    except Exception as exc:                                  # noqa: BLE001
        raise EvalError(f"工单4 判分库加载失败：{path}\n→ 原因：{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------- 小工具
def pctl(values, q: float):
    """分位数（线性取序，单点值与最近秩法一致；空列表返回 None）。"""
    vals = [v for v in (values or []) if v is not None]
    if not vals:
        return None
    ordered = sorted(vals)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(float(ordered[idx]), 1)


def lat_stats(values) -> dict:
    vals = [float(v) for v in (values or []) if v is not None]
    if not vals:
        return {"n": 0, "p50": None, "p95": None, "max": None, "min": None, "mean": None}
    return {"n": len(vals), "p50": pctl(vals, 0.50), "p95": pctl(vals, 0.95),
            "max": round(max(vals), 1), "min": round(min(vals), 1),
            "mean": round(statistics.fmean(vals), 1)}


def pick(d: dict, *keys, default=None):
    """按多个候选键取值（兼容 MultiTurnAnswer.to_dict() 与 /api/chat 响应两种命名）。"""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


# ---------------------------------------------------------------- 运行器
class LocalRunner:
    """local 模式：直接 import 研发/app 的 MultiTurnEngine（延迟导入 + 友好报错）。"""

    name = "local"

    def __init__(self, args) -> None:
        _bootstrap_syspath()
        self.args = args
        self.engine = None
        self.module_path = ROOT / "研发" / "app" / "multiturn_engine.py"
        self.health_info: dict = {}

    def prepare(self) -> None:
        if not self.module_path.exists():
            raise EvalError(
                "local 模式需要 `研发/app/multiturn_engine.py`，但该文件**尚不存在**：\n"
                f"    期望路径：{self.module_path}\n"
                "→ 这是 captain 正在并行实现的核心模块（连同 研发/app/core/dialogue_state.py、\n"
                "  研发/app/core/query_rewriter.py）。请等它落盘后重跑，或改用 http 模式：\n"
                "    pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine http")
        try:
            mod = importlib.import_module("app.multiturn_engine")
        except Exception as exc:                              # noqa: BLE001
            raise EvalError(
                "导入 `app.multiturn_engine` 失败（引擎或其依赖尚未就绪）：\n"
                f"    原因：{type(exc).__name__}: {exc}\n"
                "→ 请先确认：① run_py.ps1 已把 `研发` 加入 PYTHONPATH；\n"
                "  ② 研发/app/core/dialogue_state.py 与 query_rewriter.py 已落盘；\n"
                "  ③ 依赖（faiss / jieba / requests 等）已装。") from exc
        cls = getattr(mod, "MultiTurnEngine", None)
        if cls is None:
            raise EvalError("`app.multiturn_engine` 里找不到类 `MultiTurnEngine`，"
                            "接口契约见 设计/接口契约_冻结.md §1.3。")
        try:
            self.engine = cls(warmup=False)
        except Exception as exc:                              # noqa: BLE001
            raise EvalError("构造 `MultiTurnEngine(...)` 失败：\n"
                            f"    原因：{type(exc).__name__}: {exc}") from exc
        try:
            h = self.engine.health()
            self.health_info = h if isinstance(h, dict) else {"raw": str(h)}
        except Exception as exc:                              # noqa: BLE001
            self.health_info = {"error": f"{type(exc).__name__}: {exc}"}

    def ask(self, question: str, session_id: str, use_cache: bool) -> dict:
        t0 = time.perf_counter()
        try:
            ans = self.engine.ask(question, session_id=session_id, lang="zh", use_cache=use_cache)
        except TypeError:
            # 允许实现尚未支持 lang/use_cache 的过渡期签名（如实降级并记录）
            ans = self.engine.ask(question, session_id=session_id)
        wall = (time.perf_counter() - t0) * 1000.0
        d = ans.to_dict() if hasattr(ans, "to_dict") else dict(ans)
        d.setdefault("question", question)
        return {"resp": d, "wall_ms": round(wall, 1), "http_status": None, "error": None}

    def reset(self, session_id: str) -> None:
        try:
            self.engine.reset(session_id)
        except Exception:                                     # noqa: BLE001
            pass


class HttpRunner:
    """http 模式：POST {base_url}/api/chat（契约见 设计/接口契约_冻结.md §2）。"""

    name = "http"

    def __init__(self, args) -> None:
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        self.health_info: dict = {}

    def prepare(self) -> None:
        url = self.base_url + "/health"
        try:
            with urllib.request.urlopen(url, timeout=min(30.0, self.args.timeout)) as r:
                self.health_info = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise EvalError(
                f"服务不可用：GET {url} 失败（{exc.reason if hasattr(exc, 'reason') else exc}）。\n"
                "→ 请先启动服务（部署/脚本/serve_api.py，默认 8600；本脚本默认 8123）：\n"
                "    pwsh -NoProfile -File run_py.ps1 部署/脚本/serve_api.py\n"
                "  或用 --base-url 指定实际地址/端口。") from exc
        except Exception as exc:                              # noqa: BLE001
            raise EvalError(f"健康检查失败：GET {url}\n→ 原因：{type(exc).__name__}: {exc}") from exc

    def ask(self, question: str, session_id: str, use_cache: bool) -> dict:
        body = json.dumps({"question": question, "session_id": session_id,
                           "language": "auto", "use_cache": bool(use_cache)},
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json; charset=utf-8"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.args.timeout) as resp:
                raw = resp.read().decode("utf-8")
                status = getattr(resp, "status", 200)
        except urllib.error.HTTPError as exc:
            wall = (time.perf_counter() - t0) * 1000.0
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:                                 # noqa: BLE001
                pass
            return {"resp": {}, "wall_ms": round(wall, 1), "http_status": exc.code,
                    "error": f"HTTP {exc.code}: {detail}"}
        except Exception as exc:                              # noqa: BLE001
            wall = (time.perf_counter() - t0) * 1000.0
            return {"resp": {}, "wall_ms": round(wall, 1), "http_status": None,
                    "error": f"{type(exc).__name__}: {exc}"}
        wall = (time.perf_counter() - t0) * 1000.0
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {"resp": {}, "wall_ms": round(wall, 1), "http_status": status,
                    "error": f"响应不是合法 JSON：{exc}; 前 200 字符={raw[:200]!r}"}
        return {"resp": data, "wall_ms": round(wall, 1), "http_status": status, "error": None}

    def reset(self, session_id: str) -> None:                 # 无重置端点：换 session_id 即可
        return None


# ---------------------------------------------------------------- 合成数据自检运行器
_DRY_STRATEGY_BY_ID = {
    "MT1": "none", "MT2": "pronoun", "MT3": "pronoun",
    "MT4": "ellipsis_entity_switch", "MT5": "none",
    "MT1_iso": "none", "MT2_iso": "pronoun", "MT3_iso": "pronoun",
    "MT4_iso": "ellipsis_entity_switch", "MT5_iso": "none",
}


class DryRunRunner:
    """**只有在没有真引擎时**用于自检「脚本管道（判分→落盘→报告）」的合成数据运行器。

    ⚠️ 它不是被测系统：**不加载任何模型、不检索、不生成**，只把「用例的期望值」当作
    「系统输出」回放，因此其产物**永远不能当作验收数字**。双重防呆：
      ① `--engine dry-run` 要求 `--run-tag` 必须以 `dryrun` 开头；
      ② 所有产物带 `synthetic: true` / `NOT_FOR_ACCEPTANCE: true`，报告正文顶部有醒目横幅。
    用途：金标用例写错、判分器口径漂移、报告渲染崩掉这类**管道问题**可以在几秒内被发现。
    """

    name = "dry-run"

    def __init__(self, args) -> None:
        self.args = args
        self.health_info = {"synthetic": True,
                            "warning": "合成数据自检运行器：不加载模型，产物不得作为验收数字"}
        self._t0 = time.perf_counter()

    def prepare(self) -> None:
        if not str(self.args.run_tag).startswith("dryrun"):
            raise EvalError(
                "`--engine dry-run` 只用于管道自检，为避免产物被误当实测数据，\n"
                "必须把 --run-tag 写成以 `dryrun` 开头，例如：\n"
                "    --engine dry-run --run-tag dryrun-selfcheck")

    def ask(self, question: str, session_id: str, use_cache: bool = True) -> dict:
        """回放：答案取 `case` 的期望值（由调用方通过 `ask(..., case=...)` 注入）。"""
        case = getattr(self, "_dry_current", None) or {}
        rewrite = case.get("expected_rewrite")
        if not rewrite:
            rewrite = case.get("question") or ""
        focus = case.get("expected_focus")
        if focus and rewrite and _norm_lite(focus) not in _norm_lite(rewrite):
            rewrite = f"{focus}：{rewrite}"
        if not case.get("must_contain"):
            answer = "（合成占位答案：该用例无 must_contain，dry-run 不判答案内容）"
        else:
            head = case.get("expected_answer") or case.get("question") or ""
            answer = f"{head}（合成答案，含必含项：{'、'.join(case['must_contain'])}）"
        lat = {"rewrite": 0.0, "retrieve": 0.0, "generate": 0.0, "total": 0.0}
        resp = {
            "question": question, "session_id": session_id, "answer": answer,
            "rewritten_query": rewrite,
            "strategy": case.get("expected_strategy") or _DRY_STRATEGY_BY_ID.get(case.get("id"), "none"),
            "focus_entity": focus,
            "refused": bool(case.get("expect_refused", False)),
            "latency_ms": lat, "cited_pages": [], "synthetic": True,
            "dry_run_note": "合成数据：由用例期望值回放，非模型输出",
        }
        time.sleep(0.002)
        return {"resp": resp, "wall_ms": 0.0, "http_status": None, "error": None}

    def reset(self, session_id: str) -> None:
        return None


def _norm_lite(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def ask_case(runner, case: dict, session_id: str, use_cache: bool) -> dict:
    """统一提问入口：dry-run 需要把用例期望值注入运行器，真引擎不需要。"""
    if getattr(runner, "name", "") == "dry-run":
        runner._dry_current = case
        return runner.ask(case.get("question") or "", session_id, use_cache)
    return runner.ask(case.get("question") or "", session_id, use_cache)


# ---------------------------------------------------------------- 逐轮记录
def build_row(case: dict, run: dict, judge, wall_ms: float) -> dict:
    """把一次请求结果 + 判分结果拼成一行逐轮记录（报告直接消费）。"""
    resp = run.get("resp") or {}
    lat = pick(resp, "latency_ms", default={}) or {}
    if not isinstance(lat, dict):
        lat = {"total": lat}
    answer = pick(resp, "answer", default="") or ""
    rewritten = pick(resp, "rewritten_query", "rewritten", default="") or ""
    row = {
        "id": case.get("id"), "seq": case.get("seq"), "session": case.get("session"),
        "turn": case.get("turn"), "question": case.get("question"),
        "rewritten_query": rewritten,
        "strategy": pick(resp, "strategy"),
        "focus_entity": pick(resp, "focus_entity"),
        "answer": answer,
        "refused": pick(resp, "refused"),
        "latency_ms": lat,
        "latency_wall_ms": round(float(wall_ms), 1) if wall_ms is not None else None,
        "latency_rewrite_ms": lat.get("rewrite"),
        "latency_retrieve_ms": lat.get("retrieve"),
        "latency_generate_ms": lat.get("generate"),
        "latency_total_ms": lat.get("total"),
        "cited_pages": pick(resp, "cited_pages", default=[]),
        "http_status": run.get("http_status"),
        "error": run.get("error"),
        "expected_answer": case.get("expected_answer"),
        "expected_rewrite": case.get("expected_rewrite"),
        "expected_strategy": case.get("expected_strategy"),
        "expected_focus": case.get("expected_focus"),
        "expect_degraded": case.get("expect_degraded"),
        "note": case.get("note"),
    }
    verdict = judge.judge_case(case, {**resp, "answer": answer, "rewritten_query": rewritten,
                                      "error": run.get("error")})
    row["passed"] = verdict["passed"]
    row["checks"] = {c["name"]: c["passed"] for c in verdict["checks"]}
    row["check_details"] = {c["name"]: c["detail"] for c in verdict["checks"]}
    row["reasons"] = verdict["reasons"]
    row["judged_by"] = "确定性判分（测试/测试数据/多轮判分.py::judge_case），非 LLM 自评"
    return row


def summarize_turns(rows: list[dict], judge) -> dict:
    """多轮逐轮汇总：准确率（两套口径）+ 各维度通过率 + 策略分布 + 延迟统计。"""
    rows = list(rows or [])
    n = len(rows)
    passed = [r for r in rows if r.get("passed") is True]
    dims = {}
    for name in judge.CHECK_NAMES:
        judged = [r for r in rows if name in (r.get("checks") or {})]
        ok = sum(1 for r in judged if r["checks"][name])
        dims[name] = {"title": judge.CHECK_TITLES.get(name, name), "passed": ok,
                      "total": len(judged), "rate": round(ok / len(judged), 4) if judged else None}
    # 会话整体口径：该 session 全部轮次都 passed 才算 1 个会话通过
    sessions = {}
    for r in rows:
        sessions.setdefault(r.get("session") or "default", []).append(r)
    session_rows = []
    for sid, group in sessions.items():
        ok = all(r.get("passed") is True for r in group)
        session_rows.append({"session": sid, "n_turns": len(group),
                             "n_passed": sum(1 for r in group if r.get("passed") is True),
                             "all_passed": ok,
                             "turns": [r.get("id") for r in group]})
    n_sess = len(session_rows)
    n_sess_ok = sum(1 for s in session_rows if s["all_passed"])
    wall = [r.get("latency_wall_ms") for r in rows]
    rw = [r.get("latency_rewrite_ms") for r in rows]
    over = [r.get("id") for r in rows if (r.get("latency_wall_ms") or 0) > 3000.0]
    strategy_dist: dict[str, int] = {}
    for r in rows:
        key = str(r.get("strategy"))
        strategy_dist[key] = strategy_dist.get(key, 0) + 1
    return {
        "n_turns": n, "n_passed": len(passed),
        "accuracy": round(len(passed) / n, 4) if n else None,
        "accuracy_definition": "passed_turns / total_turns（逐轮判分）",
        "n_sessions": n_sess, "n_sessions_all_passed": n_sess_ok,
        "session_accuracy": round(n_sess_ok / n_sess, 4) if n_sess else None,
        "session_accuracy_definition": "该 session 全部轮次均 passed 才算 1 个会话通过；"
                                       "本工单 5 轮分属 2 个 session（seq1 ×3 轮、seq2 ×2 轮）",
        "session_detail": session_rows,
        "dimension_pass": dims,
        "strategy_distribution": strategy_dist,
        "latency_wall": {**lat_stats(wall), "n_over_3000ms": len(over), "over_3000_ids": over},
        "latency_rewrite": lat_stats(rw),
        "latency_definition": "主口径=客户端 wall（请求发出→完整答案返回）；"
                              "latency_rewrite=服务端 latency_ms.rewrite（改写阶段单独计时）",
        "metrics_guide": judge.MULTITURN_METRICS_GUIDE,
    }


# ---------------------------------------------------------------- 三套跑批
def run_multiturn(runner, cases, judge, args, label="results") -> tuple[list[dict], dict, dict]:
    """按 **session 顺序** 提问（同一 session 内按 turn 升序），返回 (rows, cold_start, meta)。"""
    rows: list[dict] = []
    cold_start: dict = {}
    meta: dict = {"warmup_done": False, "warmup_error": None, "cold_start_ms": None,
                  "warmup_id": None, "timing_scope": "预热 1 次（不计入统计）+ 正式跑 1 次（计时）"}
    if not cases:
        return rows, cold_start, meta
    sessions: dict[str, list[dict]] = {}
    for c in cases:
        sessions.setdefault(c.get("session") or "default", []).append(c)
    for sid in sessions:
        sessions[sid].sort(key=lambda c: (c.get("turn") or 0))

    # ---- 预热：单独跑一次，只记 cold_start_ms，结果不进入正式统计 ----
    # 预热必须用**独立会话 + 不在被测集合里的问题**：
    #   ① 用真实 session 会把状态写进被测会话；
    #   ② 用被测问题会把它写进**问题级答案缓存**（QAEngine._cache_put 是无条件的），
    #      使该轮正式测量退化成缓存命中 —— 实测踩坑：MT1 的 wall 被压成 2.3 ms，完全失真。
    _WARMUP_QUESTION = "武汉兴图新科电子股份有限公司的注册资本是多少？"
    if not args.no_warmup:
        first_sid = next(iter(sessions))
        first_case = sessions[first_sid][0]
        is_dry = getattr(runner, "name", "") == "dry-run"
        warm_question = first_case["question"] if is_dry else _WARMUP_QUESTION
        if args.engine == "local" and getattr(runner, "reset", None):
            runner.reset("__warmup__")
        if is_dry:
            runner._dry_current = first_case
        w = runner.ask(warm_question, "__warmup__", False)
        meta["warmup_done"] = w.get("error") is None
        meta["warmup_error"] = w.get("error")
        meta["warmup_id"] = f"{first_case.get('id')}·预热"
        meta["warmup_question"] = warm_question
        meta["cold_start_ms"] = w.get("wall_ms")
        meta["cold_start_scope"] = ("预热轮 wall 耗时（含冷启动：模型/索引/首轮加载），"
                                    "**不计入正式延迟统计**")
        cold_start = {"id": meta["warmup_id"], "question": warm_question,
                      "wall_ms": w.get("wall_ms"), "error": w.get("error"),
                      "latency_ms": (w.get("resp") or {}).get("latency_ms")}
        print(f"[预热] id={first_case.get('id')}·预热 wall={w.get('wall_ms')} ms "
              f"（冷启动；不计入统计） error={w.get('error')}")
        if args.engine == "local" and getattr(runner, "reset", None):
            runner.reset("__warmup__")
    else:
        meta["cold_start_scope"] = "已用 --no-warmup 跳过预热（**调试口径**，报告须标注冷启动未分离）"

    # ---- 正式跑（计时口径）----
    for sid, group in sessions.items():
        if args.engine == "local" and getattr(runner, "reset", None):
            runner.reset(sid)                                  # 同一 session 从干净状态开始
        for case in group:
            run = ask_case(runner, case, sid, not args.no_cache)
            row = build_row(case, run, judge, run.get("wall_ms"))
            rows.append(row)
            mark = "✅" if row["passed"] else "❌"
            print(f"[{label} {case.get('id')}] session={sid} turn={case.get('turn')} {mark} "
                  f"strategy={row.get('strategy')} wall={row.get('latency_wall_ms')}ms "
                  f"rewrite={row.get('latency_rewrite_ms')}ms "
                  f"答案={str(row.get('answer') or '')[:36]!r}")
            if not row["passed"]:
                for reason in row["reasons"]:
                    print(f"        ↳ {reason}")
    return rows, cold_start, meta


def run_isolated(runner, cases, judge, args) -> list[dict]:
    """独立会话对照：**每轮都换新 session**，用于验证 Q2/Q3/Q4 单轮无法消解属预期。"""
    rows = []
    for case in cases:
        sid = case.get("session") or f"iso-{case.get('id')}"
        if args.engine == "local" and getattr(runner, "reset", None):
            runner.reset(sid)
        run = ask_case(runner, case, sid, not args.no_cache)
        row = build_row(case, run, judge, run.get("wall_ms"))
        row["degraded_expected"] = bool(case.get("expect_degraded"))
        row["counts_toward_multiturn_accuracy"] = not bool(case.get("expect_degraded"))
        rows.append(row)
        print(f"[对照 {case.get('id')}] 预期降级={row['degraded_expected']!s:<5} "
              f"通过={row['passed']!s:<5} strategy={row.get('strategy')} "
              f"答案={str(answer_preview(row))!r}")
    return rows


def answer_preview(row: dict, n: int = 36) -> str:
    return str(row.get("answer") or "")[:n]


def _dry_case_of_golden(rec: dict) -> dict:
    """把工单4 golden 记录映射成 dry-run 可回放的最小用例（仅用于管道自检，不影响真跑口径）。"""
    return {"id": f"reg-{rec.get('id')}", "question": rec.get("question"),
            "expected_answer": rec.get("answer"),
            "must_contain": list(rec.get("answer_keywords") or []),
            "expected_strategy": "none", "expect_refused": False}


def run_regression16(runner, golden, args) -> dict:
    """单轮回归：逐题**新 session** 提问，用 测试/common/golden.py 判分（工单4 口径原样复用）。"""
    records = golden.load_golden()
    rows = []
    for rec in records:
        sid = f"reg16-{rec['id']}"
        if args.engine == "local" and getattr(runner, "reset", None):
            runner.reset(sid)
        run = ask_case(runner, _dry_case_of_golden(rec), sid, not args.no_cache)
        resp = run.get("resp") or {}
        answer = pick(resp, "answer", default="") or ""
        verdict = golden.judge_answer(rec, answer)
        rows.append({
            "order": rec.get("order"), "id": rec.get("id"), "category": rec.get("category"),
            "pdf": rec.get("pdf"), "question": rec.get("question"),
            "golden_answer": rec.get("answer"), "system_answer": answer,
            "correct": verdict["correct"], "judge_reason": verdict["reason"],
            "missing_keywords": [k["keyword"] for k in verdict["keyword_results"] if not k["hit"]],
            "forbidden_fired": [f.get("keyword") or f.get("regex") for f in verdict["forbidden_fired"]],
            "cited_pages": pick(resp, "cited_pages", default=[]),
            "refused": pick(resp, "refused"),
            "latency_wall_ms": run.get("wall_ms"),
            "latency_total_ms": (pick(resp, "latency_ms", default={}) or {}).get("total")
            if isinstance(pick(resp, "latency_ms", default={}), dict) else None,
            "http_status": run.get("http_status"), "error": run.get("error"),
        })
        print(f"[回归 {rec.get('order')}/16] id={rec.get('id')} 判对={verdict['correct']!s:<5} "
              f"wall={run.get('wall_ms')}ms 缺关键字={rows[-1]['missing_keywords'][:3]}")
    judged = [r for r in rows if r.get("correct") is not None]
    n_ok = sum(1 for r in judged if r["correct"])
    wall = [r.get("latency_wall_ms") for r in rows if r.get("latency_wall_ms") is not None]
    return {
        "suite": "单轮回归（工单4 golden_qa_16）",
        "judge": "测试/common/golden.py::judge_answer（确定性关键字判定，非 LLM 自评）",
        "golden_path": str(golden.GOLDEN_CANONICAL if golden.GOLDEN_CANONICAL.exists()
                           else golden.GOLDEN_MIRROR),
        "n_questions": len(rows), "n_judged": len(judged), "n_correct": n_ok,
        "accuracy": round(n_ok / len(judged), 4) if judged else None,
        "baseline_t4": "工单4 实测 16/16 = 100%（见 测试/测试报告.md §1）；本项用于确认**未退化**",
        "regression_status": ("未退化" if judged and n_ok == len(judged)
                              else ("未运行" if not judged else "**有退化，须逐题解释**")),
        "n_error": sum(1 for r in rows if r.get("error")),
        "latency_wall": lat_stats(wall),
        "items": rows,
    }


def run_abnormal(runner, cases, judge, args) -> dict:
    """异常输入：不崩溃 / 非空 / 不 5xx（local 模式以捕获异常代替 HTTP 状态）。"""
    items = []
    for case in cases:
        sid = f"abn-{case.get('id')}"
        if args.engine == "local" and getattr(runner, "reset", None):
            runner.reset(sid)
        t0 = time.perf_counter()
        try:
            run = ask_case(runner, case, sid, not args.no_cache)
            resp = run.get("resp") or {}
            outcome = {"http_status": run.get("http_status"),
                       "answer": pick(resp, "answer", default=""),
                       "refused": pick(resp, "refused"),
                       "error": run.get("error"),
                       "elapsed_ms": run.get("wall_ms")}
        except Exception as exc:                              # noqa: BLE001 —— 崩溃即判失败，不终止跑批
            outcome = {"http_status": None, "answer": None, "refused": None,
                       "error": f"{type(exc).__name__}: {exc}",
                       "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
        verdict = judge.judge_abnormal_case(case, outcome)
        items.append({**verdict, "question_repr": repr(case.get("question")),
                      "question_len": len(case.get("question") or ""),
                      "answer": outcome.get("answer"), "refused": outcome.get("refused"),
                      "http_status": outcome.get("http_status"),
                      "error": outcome.get("error"), "elapsed_ms": outcome.get("elapsed_ms")})
        print(f"[异常 {case.get('id')}] expect={case.get('expect'):<9} passed={verdict['passed']!s:<5} "
              f"http={outcome.get('http_status')} 用时={outcome.get('elapsed_ms')}ms "
              f"答案={str(outcome.get('answer') or '')[:30]!r}")
    summary = judge.summarize_abnormal(items)
    return {"suite": "异常/边界输入",
            "criterion": summary["criterion"],
            "source": str(judge.CASES_ABNORMAL),
            **summary, "items": items}


# ---------------------------------------------------------------- 报告
def _fmt(v, dash=PLACEHOLDER):
    return dash if v is None or v == "" else v


def _yn(v):
    return "—" if v is None else ("是" if v else "否")


def render_report(summary: dict, rows: list[dict], isolated: list[dict], regression: dict,
                  abnormal: dict, meta: dict, judge) -> str:
    """生成可读报告（**只写实跑产物**；未跑的套件写 `未运行`，绝不预填数字）。"""
    s = summary
    m = meta
    L: list[str] = []
    A = L.append
    A("# 工单5 多轮对话 Query 理解 —— 测试报告（实跑产物）")
    A("")
    if m.get("synthetic"):
        A("> ## ⛔ 合成数据警报：本报告**不是实测结果**")
        A(">")
        A("> 本次运行使用 `--engine dry-run`（合成数据自检运行器）：**不加载模型、不检索、不生成**，")
        A("> 答案与改写全部由用例的期望值回放。其唯一用途是验证「判分 → 落盘 → 报告」管道，")
        A("> **其中的任何数字都不得写进验收报告**。真实验收必须用 `--engine local` 或 `--engine http`。")
        A("")
    A("> **工单编号**：人工智能NLP-RAG-Query 理解优化  ")
    A(f"> **run-tag**：`{_fmt(m.get('run_tag'))}`　**模式**：`{_fmt(m.get('engine'))}`"
      f"　**base_url**：`{_fmt(m.get('base_url'), '（local 直调，无服务）')}`  ")
    A(f"> **开始**：{_fmt(m.get('started_at'))}　**结束**：{_fmt(m.get('finished_at'))}  ")
    A(f"> **判分**：`测试/测试数据/多轮判分.py::judge_case`（确定性，**非 LLM 自评**）  ")
    A(f"> **预热**：冷启动 `{_fmt(m.get('cold_start_ms'))}` ms（{_fmt(m.get('cold_start_scope'))}）  ")
    A(f"> **缓存**：use_cache={_fmt(m.get('use_cache'))}（`--no-cache` 时为 False）")
    A("")
    A("## 1. 多轮序列逐轮结果")
    A("")
    A("| id | session | turn | 问题 | 改写后 query | 策略 | 焦点实体 | 答案（摘要） | 逐维度 | 判过 | wall(ms) | 改写(ms) |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        dims = "".join("✅" if v else "❌" for v in (r.get("checks") or {}).values())
        A(f"| {_fmt(r.get('id'))} | {_fmt(r.get('session'))} | {_fmt(r.get('turn'))} | "
          f"{str(r.get('question') or '')[:28]} | {str(r.get('rewritten_query') or '')[:34]} | "
          f"{_fmt(r.get('strategy'))} | {_fmt(r.get('focus_entity'))} | "
          f"{str(r.get('answer') or '').replace('|', '/')[:48]} | {dims} | "
          f"{'✅' if r.get('passed') else '❌'} | {_fmt(r.get('latency_wall_ms'))} | "
          f"{_fmt(r.get('latency_rewrite_ms'))} |")
    A("")
    A("> 「逐维度」列顺序 = " + " / ".join(judge.CHECK_NAMES) + "；未通过项的逐条原因见 `results.jsonl` 的 `reasons`。")
    A("")
    A("## 2. 准确率汇总")
    A("")
    A("| 指标 | 取值 | 口径 |")
    A("| --- | --- | --- |")
    A(f"| **逐轮准确率（主口径）** | **{_fmt(s.get('n_passed'))}/{_fmt(s.get('n_turns'))} = "
      f"{_fmt(s.get('accuracy'))}** | `passed_turns / total_turns` |")
    A(f"| **会话整体准确率** | **{_fmt(s.get('n_sessions_all_passed'))}/{_fmt(s.get('n_sessions'))} = "
      f"{_fmt(s.get('session_accuracy'))}** | 该 session 全部轮次均通过才算 1 个会话通过 |")
    A("")
    for sid in s.get("session_detail") or []:
        A(f"- session `{sid['session']}`：{sid['n_passed']}/{sid['n_turns']} 轮通过 → "
          f"{'**整体通过**' if sid['all_passed'] else '整体未通过'}"
          f"（轮次 {', '.join(str(t) for t in sid['turns'])}）")
    A("")
    A("### 2.1 各判分维度通过率")
    A("")
    A("| 维度 | 说明 | 通过/总数 | 通过率 |")
    A("| --- | --- | --- | --- |")
    for name, d in (s.get("dimension_pass") or {}).items():
        A(f"| `{name}` | {d.get('title')} | {d.get('passed')}/{d.get('total')} | {d.get('rate')} |")
    A("")
    A("### 2.2 策略分布（实跑 strategy 计数）")
    A("")
    A("| strategy | 轮次数 |")
    A("| --- | --- |")
    for k, v in (s.get("strategy_distribution") or {}).items():
        A(f"| `{k}` | {v} |")
    A("")
    A("## 3. 指代消解准确率（多轮核心能力）")
    A("")
    A("指代消解 = 需要改写才能答对的轮次（`expected_strategy != none`）。逐维度拆开看：")
    A("")
    A("| 口径 | 通过/应通过 | 通过率 | 说明 |")
    A("| --- | --- | --- | --- |")
    ref = [r for r in rows if r.get("expected_strategy") not in (None, "none")]
    n_ref = len(ref)
    n_ref_pass = sum(1 for r in ref if r.get("passed"))
    n_ref_strategy = sum(1 for r in ref if (r.get("checks") or {}).get("strategy_match"))
    n_ref_no_leak = sum(1 for r in ref if (r.get("checks") or {}).get("no_leak"))
    n_ref_rewrite = sum(1 for r in ref if (r.get("checks") or {}).get("focus_in_rewrite"))
    A(f"| 端到端判过 | {n_ref_pass}/{n_ref} | {round(n_ref_pass / n_ref, 4) if n_ref else PLACEHOLDER} | "
      f"该轮全部维度通过 |")
    A(f"| 策略判定正确 | {n_ref_strategy}/{n_ref} | "
      f"{round(n_ref_strategy / n_ref, 4) if n_ref else PLACEHOLDER} | `strategy_match` |")
    A(f"| 改写后含焦点实体 | {n_ref_rewrite}/{n_ref} | "
      f"{round(n_ref_rewrite / n_ref, 4) if n_ref else PLACEHOLDER} | `focus_in_rewrite` |")
    A(f"| 无跨会话实体泄漏 | {n_ref_no_leak}/{n_ref} | "
      f"{round(n_ref_no_leak / n_ref, 4) if n_ref else PLACEHOLDER} | `no_leak` |")
    A("")
    det = [r for r in rows if r.get("expect_degraded")]
    if det:
        A("按指代类型细分（实跑）：")
        A("")
        A("| id | 问句 | 期望策略 | 实跑策略 | 期望焦点 | 实跑焦点 | 改写后 query | 判过 |")
        A("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in ref:
            A(f"| {r.get('id')} | {str(r.get('question') or '')[:24]} | {_fmt(r.get('expected_strategy'))} | "
              f"{_fmt(r.get('strategy'))} | {_fmt(r.get('expected_focus'))} | {_fmt(r.get('focus_entity'))} | "
              f"{str(r.get('rewritten_query') or '')[:36]} | {'✅' if r.get('passed') else '❌'} |")
        A("")
    A("### 3.1 独立会话对照实验（每轮新 session）")
    A("")
    A("| id | 问句 | 预期降级 | 实跑策略 | 判过 | 计入多轮准确率 |")
    A("| --- | --- | --- | --- | --- | --- |")
    for r in isolated or []:
        A(f"| {r.get('id')} | {str(r.get('question') or '')[:26]} | "
          f"{_yn(r.get('degraded_expected'))} | {_fmt(r.get('strategy'))} | "
          f"{'✅' if r.get('passed') else '❌'} | "
          f"{_yn(r.get('counts_toward_multiturn_accuracy'))} |")
    if not isolated:
        A(f"| {PLACEHOLDER} | — | — | — | — | — |")
    A("")
    A("> 对照实验结论口径：**单轮下 Q2/Q3/Q4 无法消解属预期行为**，其降级轮次**不作为多轮准确率的分母**；")
    A("> 该表的用途是证明「多轮的提升确实来自上下文继承」，而不是「单轮本身就能答对」。")
    A("")
    A("## 4. 响应时间")
    A("")
    A("| 口径 | n | P50 | P95 | max | min | 均值 | >3000ms |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    w = s.get("latency_wall") or {}
    rw = s.get("latency_rewrite") or {}
    A(f"| **客户端 wall（主口径）** | {_fmt(w.get('n'))} | **{_fmt(w.get('p50'))}** | "
      f"**{_fmt(w.get('p95'))}** | **{_fmt(w.get('max'))}** | {_fmt(w.get('min'))} | "
      f"{_fmt(w.get('mean'))} | {_fmt(w.get('n_over_3000ms'))} |")
    A(f"| 服务端 total（辅助） | — | — | — | — | — | — | — |")
    A(f"| **改写阶段 rewrite** | {_fmt(rw.get('n'))} | {_fmt(rw.get('p50'))} | {_fmt(rw.get('p95'))} | "
      f"{_fmt(rw.get('max'))} | {_fmt(rw.get('min'))} | {_fmt(rw.get('mean'))} | — |")
    A("")
    A(f"- 冷启动（预热 1 次，**不计入**统计）：`{_fmt(m.get('cold_start_ms'))}` ms（id "
      f"{_fmt(m.get('warmup_id'))}）")
    A(f"- 超过 3000ms 的轮次：{_fmt((w.get('over_3000_ids') or None) or '无')}")
    A(f"- 口径：{_fmt(s.get('latency_definition'))}")
    A("")
    A("## 5. 单轮回归（工单4 能力未退化验证）")
    A("")
    A(f"| 指标 | 取值 |")
    A("| --- | --- |")
    A(f"| 判对 / 判分 | {_fmt(regression.get('n_correct'))} / {_fmt(regression.get('n_judged'))} |")
    A(f"| 准确率 | {_fmt(regression.get('accuracy'))} |")
    A(f"| 基线（工单4） | {_fmt(regression.get('baseline_t4'))} |")
    A(f"| 回归结论 | {_fmt(regression.get('regression_status'))} |")
    A(f"| 请求失败条数 | {_fmt(regression.get('n_error'))} |")
    A(f"| 延迟 P50 / P95 / max | {_fmt((regression.get('latency_wall') or {}).get('p50'))} / "
      f"{_fmt((regression.get('latency_wall') or {}).get('p95'))} / "
      f"{_fmt((regression.get('latency_wall') or {}).get('max'))} |")
    A("")
    A("| # | id | 题型 | 问题（摘要） | 判对 | 缺失关键字 | wall(ms) |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    for r in regression.get("items") or []:
        A(f"| {_fmt(r.get('order'))} | {_fmt(r.get('id'))} | {_fmt(r.get('category'))} | "
          f"{str(r.get('question') or '')[:30]} | {'✅' if r.get('correct') else '❌'} | "
          f"{', '.join(r.get('missing_keywords') or []) or '—'} | {_fmt(r.get('latency_wall_ms'))} |")
    if not (regression.get("items")):
        A(f"| {PLACEHOLDER} | — | — | {PLACEHOLDER} | — | — | — |")
    A("")
    A("## 6. 异常输入")
    A("")
    A(f"判据：{_fmt(abnormal.get('criterion'))}")
    A("")
    A("| 指标 | 取值 |")
    A("| --- | --- |")
    A(f"| 通过 / 总数 | {_fmt(abnormal.get('n_passed'))} / {_fmt(abnormal.get('n_cases'))} |")
    A(f"| 通过率 | {_fmt(abnormal.get('pass_rate'))} |")
    A("")
    A("| id | 输入（repr） | 期望 | HTTP | 非空 | 无异常 | 判过 | 答案（摘要） |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for it in abnormal.get("items") or []:
        ck = {c["name"]: c["passed"] for c in (it.get("checks") or [])}
        A(f"| {_fmt(it.get('id'))} | `{str(it.get('question_repr') or '')[:34]}` | {_fmt(it.get('expect'))} | "
          f"{_fmt(it.get('http_status'), '—')} | {_yn(ck.get('answer_nonempty'))} | "
          f"{_yn(ck.get('no_exception'))} | {'✅' if it.get('passed') else '❌'} | "
          f"{str(it.get('answer') or '').replace('|', '/')[:36]} |")
    if not (abnormal.get("items")):
        A(f"| {PLACEHOLDER} | — | — | — | — | — | — | — |")
    A("")
    A("## 7. 截图清单")
    A("")
    A("> 截图必须**浏览器真实渲染**（禁止 PIL 画假图），存 `测试/测试截图/`。")
    A("")
    A("| 序号 | 文件名 | 内容 | 状态 |")
    A("| --- | --- | --- | --- |")
    shot_list = (("Q1_军用领域收入.png", "MT1 军用领域收入（4 个数值）"),
                 ("Q2_科技进步一等奖.png", "MT2 指代消解：他参与的工程"),
                 ("Q3_法定代表人.png", "MT3 指代消解：这个公司的法定代表人"),
                 ("Q4_力源法定代表人.png", "MT4 省略式实体切换：那武汉力源呢"),
                 ("Q5_销售部销售处.png", "MT5 大客户销售部 6 个销售处"),
                 ("MT_准确率汇总.png", "多轮准确率 / 维度通过率汇总页"),
                 ("MT_响应时间.png", "延迟 P50/P95/max 与改写阶段耗时"),
                 ("MT_异常输入.png", "异常输入 12 条通过情况"))
    for i, (fn, desc) in enumerate(shot_list, 1):
        A(f"| {i} | `{fn}` | {desc} | {PLACEHOLDER} |")
    A("")
    A("## 8. 结论与已知限制")
    A("")
    A("### 8.1 结论（由本报告数字直接得出，逐条对号）")
    A("")
    A(f"- 多轮逐轮准确率：**{_fmt(s.get('n_passed'))}/{_fmt(s.get('n_turns'))}**"
      f"（会话整体 {_fmt(s.get('n_sessions_all_passed'))}/{_fmt(s.get('n_sessions'))}）。")
    A(f"- 指代消解端到端：**{n_ref_pass}/{n_ref}**；策略判定 {n_ref_strategy}/{n_ref}；"
      f"跨会话泄漏防护 {n_ref_no_leak}/{n_ref}。")
    A(f"- 响应时间：wall P50 **{_fmt(w.get('p50'))}** ms / P95 **{_fmt(w.get('p95'))}** ms / "
      f"max **{_fmt(w.get('max'))}** ms；超 3000ms **{_fmt(w.get('n_over_3000ms'))}** 轮。")
    A(f"- 单轮回归：**{_fmt(regression.get('n_correct'))}/{_fmt(regression.get('n_judged'))}**"
      f"（{_fmt(regression.get('regression_status'))}）。")
    A(f"- 异常输入：**{_fmt(abnormal.get('n_passed'))}/{_fmt(abnormal.get('n_cases'))}** 通过。")
    A("")
    A("### 8.2 已知限制（如实登记，不得删改）")
    A("")
    A(f"- {PLACEHOLDER}：本报告中凡标 `{PLACEHOLDER}` 的单元格表示该项**未跑**；")
    A("  任何数字必须是实跑产物（见 `设计/接口契约_冻结.md` §3.4）。")
    A("- 独立会话对照的降级轮次（`expect_degraded=true`）**不计入**多轮准确率分母，"
      "报告必须同时给出「计入 / 不计入」两种口径下的数字。")
    A("- `no_leak` 只按**公司全称**判泄漏；简称（兴图新科 / 力源信息 / 力源）造成的歧义不在本检查范围内。")
    A(f"- 归一化来源：{_fmt(judge.NORMALIZER_SOURCE)}。")
    A("")
    A("---")
    A("")
    A("## 附录 A：运行命令与产物")
    A("")
    A("```powershell")
    A("cd E:\\gao6gongdan\\工单5")
    A(f"pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine {_fmt(m.get('engine'))} "
      f"--run-tag {_fmt(m.get('run_tag'))}")
    A("```")
    A("")
    A(f"- 产物目录：`{_fmt(m.get('out_dir'))}`")
    A("- 逐轮明细 `results.jsonl` / 汇总 `summary.json` / 对照 `isolated.jsonl` / "
      "回归 `regression16.json` / 异常 `abnormal.json` / 口径 `run_meta.json`")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="工单5 多轮 Query 理解评测跑分器（多轮 + 独立会话对照 + 单轮回归 + 异常输入）")
    ap.add_argument("--engine", choices=("local", "http", "dry-run"), default="local",
                    help="local=直调 app.multiturn_engine；http=POST {base_url}/api/chat；"
                         "dry-run=**合成数据自检**（不加载模型，产物禁止当验收数字，"
                         "--run-tag 必须以 dryrun 开头）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8123", help="http 模式的服务地址")
    ap.add_argument("--run-tag", default="mt-local", help="产物子目录名（默认 mt-local）")
    ap.add_argument("--no-cache", action="store_true", default=False,
                    help="禁用缓存（延迟测量口径；默认**启用**缓存，需与报告口径一致）")
    ap.add_argument("--out-dir", default=None,
                    help=f"产物根目录（默认 {DEFAULT_OUT_ROOT}，脚本会在其下建 <run-tag>/）")
    ap.add_argument("--timeout", type=float, default=180.0, help="单次请求超时（秒）")
    ap.add_argument("--no-warmup", action="store_true",
                    help="跳过预热（**调试口径**，报告须标注冷启动未分离）")
    ap.add_argument("--suites", default="multiturn,isolated,regression,abnormal",
                    help="逗号分隔：multiturn,isolated,regression,abnormal")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = datetime.now()
    out_root = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT
    out_dir = out_root / args.run_tag
    suites = {s.strip() for s in (args.suites or "").split(",") if s.strip()}

    print("=" * 78)
    print("工单5 多轮 Query 理解评测  |  工单：人工智能NLP-RAG-Query 理解优化")
    print(f"模式={args.engine}  缓存={'禁用(--no-cache)' if args.no_cache else '启用'}  "
          f"预热={'跳过(--no-warmup, 调试口径)' if args.no_warmup else '进行'}")
    print(f"产物 → {out_dir}")
    print("=" * 78)

    # 1) 判分库（本地，无网络）
    try:
        judge = load_judge()
    except EvalError as exc:
        print(f"\n[错误] {exc}\n", file=sys.stderr)
        return EXIT_IMPORT

    # 2) 运行器（延迟导入 + 友好报错）
    if args.engine == "local":
        runner = LocalRunner(args)
    elif args.engine == "http":
        runner = HttpRunner(args)
    else:
        runner = DryRunRunner(args)
    try:
        runner.prepare()
    except EvalError as exc:
        print(f"\n[错误] {exc}\n", file=sys.stderr)
        return EXIT_SERVICE if args.engine == "http" else EXIT_IMPORT

    try:
        cases = judge.load_multiturn_cases()
        iso_cases = judge.load_isolated_cases()
        abn_cases = judge.load_abnormal_cases()
    except Exception as exc:                                  # noqa: BLE001
        print(f"\n[错误] 用例文件读取失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "work_order": "人工智能NLP-RAG-Query 理解优化",
        "run_tag": args.run_tag, "engine": args.engine, "base_url": args.base_url,
        "started_at": started.isoformat(timespec="seconds"), "finished_at": None,
        "use_cache": not args.no_cache, "warmup": not args.no_warmup,
        "out_dir": str(out_dir), "timeout_s": args.timeout,
        "python": sys.version.split()[0], "executable": sys.executable,
        "suites": sorted(suites),
        "cases": {"multiturn": str(judge.CASES_MULTITURN), "isolated": str(judge.CASES_ISOLATED),
                  "abnormal": str(judge.CASES_ABNORMAL)},
        "normalizer": judge.NORMALIZER_SOURCE,
        "judge": "测试/测试数据/多轮判分.py::judge_case（确定性，非 LLM 自评）",
        "skeleton_ready": None,
        "synthetic": args.engine == "dry-run",
        "NOT_FOR_ACCEPTANCE": args.engine == "dry-run",
        "synthetic_warning": ("合成数据自检运行器：不加载模型，答案由用例期望值回放，"
                              "**不得作为验收数字**") if args.engine == "dry-run" else None,
    }
    try:
        for name in ("app", "app.multiturn_engine", "app.core.dialogue_state", "app.core.query_rewriter"):
            try:
                importlib.import_module(name)
                meta.setdefault("module_import", {})[name] = "ok"
            except Exception as exc:                          # noqa: BLE001
                meta.setdefault("module_import", {})[name] = f"{type(exc).__name__}: {exc}"
        meta["skeleton_ready"] = all(v == "ok" for v in meta["module_import"].values())
    except Exception:                                         # noqa: BLE001
        pass
    if getattr(runner, "health_info", None):
        meta["health"] = runner.health_info

    results = {"rows": [], "isolated": [], "regression": {}, "abnormal": {}}
    cold_start: dict = {}
    try:
        if "multiturn" in suites:
            rows, cold_start, run_meta = run_multiturn(runner, cases, judge, args)
            results["rows"] = rows
            meta.update({k: v for k, v in run_meta.items()})
        if "isolated" in suites and iso_cases:
            print("\n-- 独立会话对照（每轮新 session；预期降级轮次不计入多轮准确率）--")
            results["isolated"] = run_isolated(runner, iso_cases, judge, args)
        if "regression" in suites:
            print("\n-- 单轮回归（工单4 golden_qa_16，判分复用 测试/common/golden.py）--")
            try:
                results["regression"] = run_regression16(runner, load_golden_lib(), args)
            except EvalError as exc:
                print(f"[警告] 单轮回归跳过：{exc}", file=sys.stderr)
                results["regression"] = {"suite": "单轮回归（工单4 golden_qa_16）",
                                         "regression_status": PLACEHOLDER,
                                         "error": str(exc), "items": []}
        if "abnormal" in suites:
            print("\n-- 异常输入（不崩溃 / 非空 / 不 5xx）--")
            results["abnormal"] = run_abnormal(runner, abn_cases, judge, args)
    except EvalError as exc:
        print(f"\n[错误] {exc}\n", file=sys.stderr)
        return EXIT_SERVICE if args.engine == "http" else EXIT_IMPORT
    except KeyboardInterrupt:
        print("\n[中断] 用户取消；已跑完的轮次不会落盘（重跑请加 --run-tag 区分）。", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as exc:                                  # noqa: BLE001
        print(f"\n[错误] 评测过程中出现未预期异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        print("→ 请把上面的 traceback 一并反馈；已完成的轮次未落盘。", file=sys.stderr)
        return EXIT_RUNTIME

    summary = summarize_turns(results["rows"], judge)
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["duration_s"] = round((datetime.now() - started).total_seconds(), 1)

    # ---- 落盘 ----
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results["rows"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "isolated.jsonl").open("w", encoding="utf-8") as fh:
        for r in results["isolated"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(
        {"run_meta": meta, "summary": summary, "cold_start": cold_start,
         "isolated_summary": {
             "n": len(results["isolated"]),
             "n_passed": sum(1 for r in results["isolated"] if r.get("passed")),
             "n_degraded_expected": sum(1 for r in results["isolated"] if r.get("degraded_expected")),
             "note": "对照实验：预期降级轮次不计入多轮准确率"},
         "regression16": {k: v for k, v in (results["regression"] or {}).items() if k != "items"},
         "abnormal": {k: v for k, v in (results["abnormal"] or {}).items() if k != "items"}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "regression16.json").write_text(json.dumps(
        results["regression"] or {"regression_status": PLACEHOLDER}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "abnormal.json").write_text(json.dumps(
        results["abnormal"] or {"suite": "异常/边界输入", "n_cases": 0}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (out_dir / "report.md").write_text(
        render_report(summary, results["rows"], results["isolated"], results["regression"] or {},
                      results["abnormal"] or {}, meta, judge), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"[完成] 产物 → {out_dir}")
    print("  results.jsonl / summary.json / report.md / isolated.jsonl / "
          "regression16.json / abnormal.json / run_meta.json")
    print(f"  多轮逐轮：{summary['n_passed']}/{summary['n_turns']} = {summary['accuracy']}；"
          f"会话整体：{summary['n_sessions_all_passed']}/{summary['n_sessions']} = "
          f"{summary['session_accuracy']}")
    w = summary["latency_wall"]
    print(f"  延迟 wall：P50 {w['p50']} / P95 {w['p95']} / max {w['max']} ms；"
          f"超 3000ms {w['n_over_3000ms']} 轮；冷启动 {meta.get('cold_start_ms')} ms（不计入）")
    print(f"  改写阶段：P50 {summary['latency_rewrite']['p50']} / "
          f"P95 {summary['latency_rewrite']['p95']} ms")
    if results["regression"]:
        rg = results["regression"]
        print(f"  单轮回归：{rg.get('n_correct')}/{rg.get('n_judged')} = {rg.get('accuracy')}"
              f"（{rg.get('regression_status')}）")
    if results["abnormal"]:
        ab = results["abnormal"]
        print(f"  异常输入：{ab.get('n_passed')}/{ab.get('n_cases')} = {ab.get('pass_rate')}")
    print("=" * 78)
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EvalError as exc:                                  # 兜底：绝不裸抛 traceback
        print(f"\n[错误] {exc}\n", file=sys.stderr)
        sys.exit(EXIT_IMPORT)
