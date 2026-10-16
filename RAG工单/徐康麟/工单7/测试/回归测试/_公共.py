# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 回归测试公共支撑模块。

本模块被 ``test_单轮问答.py`` / ``test_多轮对话.py`` / ``test_异常输入.py`` / ``run_all.py``
共同引用，提供四件事：

1. **环境引导**：设置 ``RAG_DATA__*`` / ``RAG_LOG__DIR`` 等环境变量并把 ``研发`` 加入
   ``sys.path``（不依赖调用方预先设好环境）。
2. **真实判分适配**：直接 ``importlib`` 加载 ``研发/评估/判分.py`` 的纯函数
   （``normalize`` / ``keyword_coverage`` / ``is_correct`` / ``recall``），**不另造口径**，
   保证回归结论与评测官口径同源可比。
3. **逐题结果落盘**：把每个用例的真实结果写成 JSON（``回归结果.json``），
   供 ``测试/生成截图.py`` 渲染真实截图（禁止手绘假数据）。
4. **CJK 宽度感知的控制台表格渲染**：打印 PASS/FAIL 表（中英文混排也能对齐）。

只读红线：本模块只写入 ``测试/`` 下的产物目录，绝不写 ``工单1/2/6``。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- 常量
WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
WORK_ORDER_SHORT = "工单7"

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "研发"
EVAL_DIR = APP_DIR / "评估"
TEST_DIR = REPO_ROOT / "测试"
TEST_DATA_DIR = TEST_DIR / "测试数据"
SHOT_DIR = TEST_DIR / "测试截图"
INDEX_MANIFEST = APP_DIR / "data" / "index" / "index_manifest.json"

PY = r"E:\gao6gongdan\工单1\.venv\Scripts\python.exe"

#: 公共前置：与 设计/测试用例设计.md §0.1 一致（不传参时由本模块兜底设置）
ENV_DEFAULTS: dict[str, str] = {
    "RAG_DATA__RAW_DIR": str(APP_DIR / "data" / "raw"),
    "RAG_DATA__PROCESSED_DIR": str(APP_DIR / "data" / "processed"),
    "RAG_DATA__INDEX_DIR": str(APP_DIR / "data" / "index"),
    "RAG_LOG__DIR": str(REPO_ROOT / "部署" / "日志"),
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}

#: 题型 -> 组名（截图与报告分组用）
GROUPS = ("单轮问答", "多轮对话", "异常输入")


# --------------------------------------------------------------------------- 环境
def bootstrap_env() -> dict[str, str]:
    """设置环境变量与 ``sys.path``；返回实际生效的关键变量（供报告记录）。"""
    for key, value in ENV_DEFAULTS.items():
        if key.startswith("RAG_"):
            os.environ[key] = value           # 数据/日志路径以本工单为准，强制覆盖
        else:
            os.environ.setdefault(key, value)
    for path in (str(APP_DIR), str(EVAL_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:  # 控制台 UTF-8（Windows 默认 GBK 会吃掉中文表格）
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 —— 非 TTY 环境忽略
        pass
    return {k: os.environ.get(k, "") for k in ENV_DEFAULTS}


def index_ready() -> tuple[bool, dict[str, Any]]:
    """索引是否就绪（返回 ``(就绪, 清单摘要)``）。"""
    if not INDEX_MANIFEST.exists():
        return False, {"manifest": str(INDEX_MANIFEST), "exists": False}
    try:
        data = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, {"manifest": str(INDEX_MANIFEST), "exists": True,
                       "error": f"{type(exc).__name__}: {exc}"}
    return True, {
        "manifest": str(INDEX_MANIFEST), "exists": True,
        "chunk_total": data.get("chunk_total") or data.get("chunks_total"),
        "embed_model": data.get("embed_model") or data.get("embedding_model"),
        "run_id": data.get("run_id"),
        "page_basis": data.get("page_basis"),
    }


def page_limits() -> dict[str, int]:
    """语料文件名 -> 页数（取自索引清单 ``files[].page_count``，用于引用页合法性校验）。"""
    if not INDEX_MANIFEST.exists():
        return {}
    try:
        data = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    limits: dict[str, int] = {}
    for item in data.get("files") or []:
        name = str(item.get("file_name") or "")
        count = item.get("page_count")
        if name and isinstance(count, int):
            limits[name] = count
    return limits


def corpus_short_names() -> list[str]:
    """从索引清单派生语料公司简称（``...__601318__中国平安__2019年__...`` → ``中国平安``）。

    用于「答案是否引用了语料内公司的数据」这类判定，**不硬编码公司名**。
    """
    if not INDEX_MANIFEST.exists():
        return []
    try:
        data = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    names: list[str] = []
    for item in data.get("files") or []:
        parts = str(item.get("file_name") or "").split("__")
        if len(parts) >= 4 and parts[3]:
            names.append(parts[3])
    return sorted(set(names))


def env_facts() -> dict[str, Any]:
    """环境事实（报告与截图页脚都引用这里，避免手抄）。"""
    import platform

    return {
        "解释器": sys.executable,
        "解释器版本": sys.version.split()[0],
        "平台": platform.platform(),
        "工作目录": str(Path.cwd()),
        "仓库根": str(REPO_ROOT),
        "索引清单": str(INDEX_MANIFEST),
    }


# --------------------------------------------------------------------------- 判分
_JUDGE: Any = None


def judge_module() -> Any:
    """加载 ``研发/评估/判分.py``（纯函数，无索引/模型依赖）。"""
    global _JUDGE
    if _JUDGE is None:
        spec = importlib.util.spec_from_file_location("gd7_judge", EVAL_DIR / "判分.py")
        if spec is None or spec.loader is None:  # pragma: no cover
            raise RuntimeError(f"无法加载判分模块：{EVAL_DIR / '判分.py'}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _JUDGE = module
    return _JUDGE


def coverage_of(answer: Any, keywords: Iterable[Any] | None) -> dict[str, Any]:
    """关键词覆盖率（与评测官同口径：归一化 + 数字边界）。"""
    judge = judge_module()
    hit, total, cov, hit_list, miss_list = judge.keyword_coverage(answer, keywords)
    return {"命中数": hit, "总数": total, "覆盖率": cov,
            "命中关键词": hit_list, "缺失关键词": miss_list}


def contains(answer: Any, keyword: Any) -> bool:
    """归一化 + 数字边界的子串包含判定（多轮断言用）。"""
    return bool(judge_module().text_matches(answer, keyword))


def load_question_set() -> dict[str, dict[str, Any]]:
    """读取 ``研发/评估/测试问题集.jsonl``（17 题，逐字题面与关键词）。"""
    path = EVAL_DIR / "测试问题集.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[str(row.get("id"))] = row
    return rows


# --------------------------------------------------------------------------- 结果
def new_case(case_id: str, name: str, *, group: str, question: str = "",
             expect: str = "") -> dict[str, Any]:
    """构造一条用例结果记录。"""
    return {
        "用例编号": case_id,
        "用例名称": name,
        "分组": group,
        "输入": question,
        "预期": expect,
        "通过": None,
        "断言": [],          # [{"名称","通过","实测"}]
        "耗时ms": None,
        "答案": "",
        "答案字数": 0,
        "拒答": None,
        "拒答原因": None,
        "引用页": [],
        "引用语料": [],
        "生成方式": None,
        "关键实测": {},
        "备注": "",
        "异常": None,
    }


def add_assert(case: dict[str, Any], name: str, ok: Any, measured: Any = "") -> bool:
    """追加一条断言（返回布尔化后的结果，便于调用方串联）。"""
    passed = bool(ok)
    case["断言"].append({"名称": name, "通过": passed, "实测": measured if measured != "" else passed})
    return passed


def finalize_case(case: dict[str, Any]) -> dict[str, Any]:
    """结算：全部断言通过 ⇒ 用例通过；无断言 ⇒ 未通过（防止「空断言即通过」）。"""
    asserts = case.get("断言") or []
    case["通过"] = bool(asserts) and all(a["通过"] for a in asserts)
    return case


def summarize(group: str, cases: Sequence[Mapping[str, Any]], elapsed_ms: float) -> dict[str, Any]:
    total = len(cases)
    passed = sum(1 for c in cases if c.get("通过"))
    return {
        "分组": group,
        "用例数": total,
        "通过": passed,
        "失败": total - passed,
        "通过率": round(passed / total, 4) if total else 0.0,
        "耗时ms": round(float(elapsed_ms), 2),
        "用例": list(cases),
    }


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- 表格
def width_of(text: Any) -> int:
    """CJK 宽度（全角记 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def pad(text: Any, width: int, align: str = "left") -> str:
    text = str(text)
    fill = max(0, width - width_of(text))
    if align == "right":
        return " " * fill + text
    if align == "center":
        left = fill // 2
        return " " * left + text + " " * (fill - left)
    return text + " " * fill


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
                 aligns: Sequence[str] | None = None) -> str:
    """渲染等宽表格（列宽按内容自适应）。"""
    cols = len(headers)
    aligns = list(aligns or ["left"] * cols)
    widths = [width_of(h) for h in headers]
    for row in rows:
        for idx in range(cols):
            cell = row[idx] if idx < len(row) else ""
            widths[idx] = max(widths[idx], width_of(cell))
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line,
           "|" + "|".join(f" {pad(h, widths[i], 'center')} " for i, h in enumerate(headers)) + "|",
           line]
    for row in rows:
        cells = []
        for idx in range(cols):
            cell = row[idx] if idx < len(row) else ""
            cells.append(f" {pad(cell, widths[idx], aligns[idx])} ")
        out.append("|" + "|".join(cells) + "|")
    out.append(line)
    return "\n".join(out)


def print_case_table(cases: Sequence[Mapping[str, Any]], title: str = "") -> None:
    """打印 PASS/FAIL 表（每行一个用例）。"""
    if title:
        print(f"\n=== {title} ===")
    rows = []
    for case in cases:
        assert_total = len(case.get("断言") or [])
        assert_ok = sum(1 for a in case.get("断言") or [] if a["通过"])
        first_fail = next((a["名称"] for a in case.get("断言") or [] if not a["通过"]), "")
        rows.append([
            case.get("用例编号", ""),
            case.get("用例名称", ""),
            "PASS" if case.get("通过") else "FAIL",
            f"{assert_ok}/{assert_total}",
            f"{case.get('耗时ms') if case.get('耗时ms') is not None else '-'}",
            (case.get("答案字数") or 0),
            first_fail[:26],
        ])
    print(render_table(["用例编号", "用例名称", "结论", "断言", "耗时ms", "答案字数", "首个失败断言"],
                       rows, aligns=["left", "left", "center", "center", "right", "right", "left"]))


def fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "-"


__all__ = [
    "WORK_ORDER", "WORK_ORDER_SHORT", "REPO_ROOT", "APP_DIR", "EVAL_DIR", "TEST_DIR",
    "TEST_DATA_DIR", "SHOT_DIR", "INDEX_MANIFEST", "PY", "GROUPS",
    "bootstrap_env", "index_ready", "env_facts", "judge_module", "coverage_of", "contains",
    "load_question_set", "new_case", "add_assert", "finalize_case", "summarize",
    "save_json", "now_str", "width_of", "pad", "render_table", "print_case_table", "fmt_ms",
]
