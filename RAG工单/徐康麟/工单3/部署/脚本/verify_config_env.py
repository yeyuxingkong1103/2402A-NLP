# -*- coding: utf-8 -*-
"""工单3 基座校验：解释器 / 依赖 / PDF 语料 / Ollama 连通性 四项【实测】。

工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
阶段：T0 基座（部署/脚本）
定位：不依赖任何第三方包（只用标准库），逐项打印实测结论，失败快速返回。
      PyMuPDF(fitz) 只在 PDF 段落内延迟导入并显式降级，绝不 import 缺失包。

用法（工作目录 = E:\\gao6gongdan\\工单3）：
    pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py
    pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py --deep
    pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py --table-pages 129,152

常用开关：
    --deep              真实 import 重型依赖（torch/transformers 等，几十秒），默认只做 find_spec + 版本元数据探测
    --no-ollama         跳过 Ollama 段（此时 Ollama 不参与总体判定）
    --no-pdf            跳过 PDF 段
    --table-pages A,B   指定做 find_tables 实测的页码（默认 1,中段,末段前，二份 PDF 通用，不硬编码文件名）
    --no-write          不落盘 JSON 报告（默认写 部署/配置/verify_env_report.json）
    --json PATH         自定义 JSON 报告路径

退出码：0 = 全部必检项通过；1 = 存在必检项失败；2 = 脚本自身参数错误。
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import platform
import struct
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# 0. 常量：路径与工单标识（全部结论以实测为准，禁止编造）
# ---------------------------------------------------------------------------
WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "研发" / "data" / "raw"
LOG_DIR = REPO_ROOT / "部署" / "日志"
CONFIG_DIR = REPO_ROOT / "部署" / "配置"
LOG_PATH = LOG_DIR / "verify_env.jsonl"
DEFAULT_REPORT_PATH = CONFIG_DIR / "verify_env_report.json"
ENTRY_SCRIPT = REPO_ROOT / "run_py.ps1"
DEFAULT_INTERPRETER = Path(r"E:\gao6gongdan\工单1\.venv\Scripts\python.exe")

# 五文件夹骨架（T0 交付物之一）
SKELETON_DIRS = (
    "设计",
    "研发",
    "研发/app/core",
    "研发/app/ui",
    "研发/data/raw",
    "研发/scripts",
    "测试",
    "测试/离线",
    "测试/在线",
    "优化",
    "优化/基线",
    "优化/脚本",
    "优化/评估结果",
    "部署",
    "部署/脚本",
    "部署/配置",
    "部署/日志",
    "部署/docker",
)

# 已装可用（实测 import 成功）：必须存在，缺失即视为基座失败
REQUIRED_PACKAGES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("pymupdf", ("pymupdf", "PyMuPDF"), "PDF 文本 + 表格提取（page.find_tables，1.28 起的正式包名）"),
    ("fitz", ("pymupdf", "PyMuPDF"), "PyMuPDF 的历史别名（仍可用，导入时会打印废弃告警）"),
    ("numpy", ("numpy",), "向量精确检索 / 打分"),
    ("jieba", ("jieba",), "自实现 BM25 的中文分词"),
    ("sklearn", ("scikit-learn",), "TF-IDF / 指标计算"),
    ("faiss", ("faiss-cpu", "faiss"), "可选向量加速（非必需路径）"),
    ("torch", ("torch",), "CPU 版深度模型运行时（无 CUDA）"),
    ("transformers", ("transformers",), "本地模型加载"),
    ("sentence_transformers", ("sentence-transformers",), "本地嵌入降级后端"),
    ("openai", ("openai",), "vLLM / SGLang 兼容后端客户端"),
    ("fastapi", ("fastapi",), "在线服务框架"),
    ("uvicorn", ("uvicorn",), "ASGI 服务器"),
    ("pytest", ("pytest",), "测试框架"),
    ("requests", ("requests",), "HTTP 客户端"),
    ("httpx", ("httpx",), "HTTP 客户端"),
    ("sentencepiece", ("sentencepiece",), "分词器依赖"),
)

# 缺失且本机断网无法安装：必须【保持缺失】，存在即为可回收的意外发现（INFO，不算失败）
EXPECTED_MISSING: tuple[tuple[str, str], ...] = (
    ("streamlit", "改用自实现 研发/app/ui/serve_fallback.py（http.server）；streamlit_app.py 仍按真 Streamlit 应用交付"),
    ("loguru", "改用自实现 loguru 风格 JSON Lines logger"),
    ("pdfplumber", "改用 pymupdf page.find_tables()"),
    ("chromadb", "改用 numpy 精确检索（faiss 可选）"),
    ("langchain", "自实现检索/生成链路"),
    ("rank_bm25", "自实现 BM25（jieba 分词）"),
    ("ragas", "不可用：报告必须标注「RAGAS 未运行（依赖不可用，本机断网）」并给确定性指标替代"),
    ("pandas", "改用标准库 csv 写 qa_results.csv"),
    ("gradio", "改用 http.server 备用界面"),
    ("markdown", "自实现 Markdown 渲染 / 直接输出对齐文本"),
    ("bs4", "改用标准库 html.parser"),
    ("lxml", "改用标准库 xml.etree / html.parser"),
)

OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434"
OLLAMA_PROBE_TIMEOUT_S = 0.5  # 后端探测超时 ≤ 0.5 s 且不重试（硬性要求）
OLLAMA_EMBED_TIMEOUT_S = 20.0  # 功能实测预算（含冷启动加载模型）
OLLAMA_GEN_TIMEOUT_S = 30.0
FIRST_TOKEN_BUDGET_MS = 3000.0  # 首字响应硬指标 ≤ 3 s


# ---------------------------------------------------------------------------
# 1. 自实现 loguru 风格 JSON Lines logger（禁止 import loguru）
# ---------------------------------------------------------------------------
class JsonLineLogger:
    """工单3 自实现结构化 logger：JSON Lines 落盘 + stderr 告警。

    字段固定为：ts / level / event / func / work_order / input / output /
    elapsed_ms / exception / traceback / message，满足「入口出口都要有
    函数名、输入摘要、输出摘要、耗时、异常堆栈」的硬性要求。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.degraded_reason: str | None = None
        self._fh = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")
        except OSError as exc:
            # 显式降级：不静默，明确告知日志不可落盘
            self.degraded_reason = f"日志文件不可写（{path}）：{type(exc).__name__}: {exc}"
            print(f"[logger 降级] {self.degraded_reason}", file=sys.stderr)

    def log(self, event: str, func: str, level: str = "INFO", **fields: Any) -> None:
        """写一条结构化日志；落盘失败时降级到 stderr，绝不静默丢弃。"""
        record: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
            "func": func,
            "work_order": WORK_ORDER,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
                return
            except OSError as exc:
                print(f"[logger 降级] 写日志失败：{type(exc).__name__}: {exc}", file=sys.stderr)
                self.degraded_reason = f"写日志失败：{type(exc).__name__}: {exc}"
                self._fh = None
        if level in ("ERROR", "WARN"):
            print(f"[{level}] {line}", file=sys.stderr)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError as exc:
                print(f"[logger 降级] 关闭日志失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                self._fh = None


LOGGER = JsonLineLogger(LOG_PATH)


def setup_stdio() -> None:
    """把 stdout/stderr 切成 UTF-8，保证中文与 ✅/❌ 在 Windows 控制台不乱码。

    说明：本函数是日志链路自身的引导步骤，位于 `traced` 装饰器定义之前，
    且其内部已对每一步失败显式 LOGGER.log 降级，故不再套 @traced（套了会 NameError）。
    """
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError) as exc:
            LOGGER.log(
                "degrade",
                "setup_stdio",
                level="WARN",
                message=f"{name} 无法切换 UTF-8，已降级为平台默认编码",
                exception=type(exc).__name__,
                traceback=traceback.format_exc(),
            )


def brief(value: Any, limit: int = 400) -> str:
    """生成输入/输出摘要字符串；摘要生成失败时返回显式降级标记（不静默）。"""
    try:
        if isinstance(value, CheckItem):
            text = f"CheckItem(key={value.key}, status={value.status}, detail={value.detail})"
        elif isinstance(value, (list, tuple)):
            items = [asdict(v) if isinstance(v, CheckItem) else v for v in value]
            text = f"len={len(items)} " + json.dumps(items, ensure_ascii=False, default=str)
        elif isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            text = repr(value)
    except Exception as exc:  # noqa: BLE001 —— 摘要失败不得影响主流程，但要显式标注
        text = f"<摘要降级 {type(exc).__name__}: {exc}>"
    return text if len(text) <= limit else f"{text[:limit]}…(截断,共{len(text)}字符)"


def traced(func: Callable[..., Any]) -> Callable[..., Any]:
    """装饰器：函数入口/出口/异常全部写结构化日志（含耗时与异常堆栈）。"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        LOGGER.log("enter", func.__name__, input=brief({"args": args, "kwargs": kwargs}))
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 —— 记录堆栈后原样传播，禁止静默失败
            LOGGER.log(
                "error",
                func.__name__,
                level="ERROR",
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
                exception=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            raise
        LOGGER.log(
            "exit",
            func.__name__,
            output=brief(result),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
        return result

    return wrapper


# ---------------------------------------------------------------------------
# 2. 检查项数据结构与渲染
# ---------------------------------------------------------------------------
@dataclass
class CheckItem:
    """一条可核对的实测结论。status: passed / failed / degraded / info。"""

    key: str
    title: str
    status: str
    detail: str
    required: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)


ICON = {"passed": "✅", "failed": "❌", "degraded": "⚠️", "info": "ℹ️"}


@traced
def render_report(items: Sequence[CheckItem]) -> str:
    """把检查项渲染成人可读清单（逐项打印实测结论）。"""
    lines: list[str] = []
    current = None
    for item in items:
        if item.title != current:
            current = item.title
            lines.append("")
            lines.append(f"── {item.title} " + "─" * max(0, 46 - len(item.title) * 2))
        icon = ICON.get(item.status, "?")
        flag = "" if item.required else "（参考项，不参与判定）"
        lines.append(f"  {icon} [{item.key}]{flag} {item.detail}")
        for key, value in item.evidence.items():
            if isinstance(value, (dict, list)):
                payload = json.dumps(value, ensure_ascii=False, default=str)
                if len(payload) > 300:
                    payload = payload[:300] + "…"
                lines.append(f"        · {key} = {payload}")
            else:
                lines.append(f"        · {key} = {value}")
    return "\n".join(lines)


@traced
def summarize(items: Sequence[CheckItem]) -> dict[str, Any]:
    """统计各状态数量并给出总体判定（必检项无 failed 即通过）。"""
    counts: dict[str, int] = {"passed": 0, "failed": 0, "degraded": 0, "info": 0}
    required_failed: list[str] = []
    degraded: list[str] = []
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
        if item.required and item.status == "failed":
            required_failed.append(item.key)
        if item.required and item.status == "degraded":
            degraded.append(item.key)
    return {
        "counts": counts,
        "required_failed": required_failed,
        "degraded": degraded,
        "verdict": "passed" if not required_failed else "failed",
    }


# ---------------------------------------------------------------------------
# 3. 通用工具
# ---------------------------------------------------------------------------
@traced
def sha256_prefix(path: Path, prefix_hex: int = 16, chunk: int = 1 << 20) -> str:
    """计算文件 SHA256 的前若干位十六进制（用于与已核验哈希比对）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:prefix_hex].upper()


@traced
def dist_version(candidates: Sequence[str]) -> str | None:
    """按候选发行名逐个查元数据版本，全部失败返回 None。"""
    for name in candidates:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 —— 元数据异常不致命，记录后继续下一个候选名
            LOGGER.log(
                "degrade",
                "dist_version",
                level="WARN",
                message=f"读取 {name} 元数据失败",
                exception=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            continue
    return None


@traced
def env_or(name: str, default: str) -> str:
    """读环境变量，空串视为未设置。"""
    value = os.environ.get(name, "").strip()
    return value or default


# ---------------------------------------------------------------------------
# 4. 逐段检查
# ---------------------------------------------------------------------------
@traced
def check_interpreter() -> CheckItem:
    """实测当前解释器身份，并与 run_py.ps1 的解析结果对照。"""
    override = os.environ.get("RAG_SCHEDULER_PYTHON", "").strip()
    expected = Path(override) if override else DEFAULT_INTERPRETER
    expected_norm = os.path.normcase(os.path.abspath(str(expected)))
    actual_norm = os.path.normcase(os.path.abspath(sys.executable))
    same = expected_norm == actual_norm
    evidence = {
        "python 版本": platform.python_version(),
        "编译器": platform.python_compiler(),
        "sys.executable": sys.executable,
        "run_py.ps1 期望解释器": str(expected) + ("（来自 RAG_SCHEDULER_PYTHON）" if override else "（默认工单1 venv）"),
        "解释器是否存在": expected.exists(),
        "位数": f"{struct.calcsize('P') * 8} bit",
        "sys.prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "是否为 venv": getattr(sys, "base_prefix", sys.prefix) != sys.prefix,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "<未设置>"),
        "PYTHONUTF8": os.environ.get("PYTHONUTF8", "<未设置>"),
    }
    if not expected.exists():
        return CheckItem(
            "interpreter",
            "一、解释器（实测）",
            "failed",
            f"run_py.ps1 期望的解释器不存在：{expected}",
            evidence=evidence,
        )
    if not same:
        return CheckItem(
            "interpreter",
            "一、解释器（实测）",
            "degraded",
            f"本进程解释器与 run_py.ps1 解析结果不一致：本进程 {sys.executable}",
            evidence=evidence,
        )
    return CheckItem(
        "interpreter",
        "一、解释器（实测）",
        "passed",
        f"{sys.version.split()[0]} / {sys.executable}（与 run_py.ps1 一致）",
        evidence=evidence,
    )


@traced
def check_entry_script() -> list[CheckItem]:
    """实测 run_py.ps1 是否就位、是否带工单编号注释。"""
    items: list[CheckItem] = []
    if not ENTRY_SCRIPT.exists():
        items.append(
            CheckItem(
                "run_py.ps1",
                "二、统一入口脚本（实测）",
                "failed",
                f"缺失：{ENTRY_SCRIPT}",
                evidence={"路径": str(ENTRY_SCRIPT)},
            )
        )
        return items
    text = ENTRY_SCRIPT.read_text(encoding="utf-8", errors="replace")
    has_wo = WORK_ORDER in text
    has_env_override = "RAG_SCHEDULER_PYTHON" in text
    evidence = {
        "路径": str(ENTRY_SCRIPT),
        "字节数": ENTRY_SCRIPT.stat().st_size,
        "SHA256(前16)": sha256_prefix(ENTRY_SCRIPT),
        "含工单编号注释": has_wo,
        "支持 RAG_SCHEDULER_PYTHON 覆盖": has_env_override,
        "默认解释器串": str(DEFAULT_INTERPRETER),
    }
    status = "passed" if (has_wo and has_env_override) else "failed"
    detail = (
        "run_py.ps1 就位，含工单编号注释与 RAG_SCHEDULER_PYTHON 覆盖入口"
        if status == "passed"
        else "run_py.ps1 存在但缺少工单编号注释或解释器覆盖入口"
    )
    items.append(CheckItem("run_py.ps1", "二、统一入口脚本（实测）", status, detail, evidence=evidence))
    return items


@traced
def check_skeleton() -> CheckItem:
    """实测五文件夹骨架是否齐备。"""
    missing = [rel for rel in SKELETON_DIRS if not (REPO_ROOT / rel).is_dir()]
    evidence = {
        "根目录": str(REPO_ROOT),
        "要求目录数": len(SKELETON_DIRS),
        "缺失目录": missing,
        "五文件夹": [d for d in ("设计", "研发", "测试", "优化", "部署") if (REPO_ROOT / d).is_dir()],
    }
    if missing:
        return CheckItem("skeleton", "三、目录骨架（实测）", "failed", f"缺少 {len(missing)} 个目录：{missing}", evidence=evidence)
    return CheckItem("skeleton", "三、目录骨架（实测）", "passed", f"{len(SKELETON_DIRS)} 个目录全部就位", evidence=evidence)


@traced
def check_dependencies(deep: bool) -> list[CheckItem]:
    """逐包探测依赖：默认 find_spec + 元数据版本；--deep 时真实 import。"""
    items: list[CheckItem] = []

    rows: list[dict[str, Any]] = []
    missing_required: list[str] = []
    for mod, dists, role in REQUIRED_PACKAGES:
        row: dict[str, Any] = {"模块": mod, "用途": role, "发行名": dists[0]}
        t0 = time.perf_counter()
        try:
            spec = importlib.util.find_spec(mod)
        except Exception as exc:  # noqa: BLE001 —— 探测异常必须显式记录
            spec = None
            LOGGER.log(
                "degrade",
                "check_dependencies",
                level="WARN",
                message=f"find_spec({mod}) 抛异常",
                exception=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            row["探测异常"] = f"{type(exc).__name__}: {exc}"
        row["可导入"] = spec is not None
        row["版本"] = dist_version(dists)
        row["探测耗时ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if spec is None:
            missing_required.append(mod)
        if deep and spec is not None:
            t1 = time.perf_counter()
            cached = mod in sys.modules
            try:
                importlib.import_module(mod)
                row["真实import"] = "成功"
            except Exception as exc:  # noqa: BLE001 —— 真实导入失败即降级为缺失
                row["真实import"] = f"失败：{type(exc).__name__}: {exc}"
                missing_required.append(mod)
                LOGGER.log(
                    "degrade",
                    "check_dependencies",
                    level="WARN",
                    message=f"--deep 真实 import({mod}) 失败",
                    exception=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            row["import耗时ms"] = round((time.perf_counter() - t1) * 1000, 2)
            row["导入前已在sys.modules"] = cached
        rows.append(row)

    mode = "find_spec + 真实 import" if deep else "find_spec + 元数据版本（未真实 import）"
    if missing_required:
        items.append(
            CheckItem(
                "deps.required",
                "四、依赖探测（实测）",
                "failed",
                f"缺失必需依赖：{sorted(set(missing_required))}",
                evidence={"探测模式": mode, "必需包": rows},
            )
        )
    else:
        items.append(
            CheckItem(
                "deps.required",
                "四、依赖探测（实测）",
                "passed",
                f"{len(rows)} 个必需依赖全部可用",
                evidence={"探测模式": mode, "必需包": rows},
            )
        )

    absent_rows: list[dict[str, Any]] = []
    unexpected_present: list[str] = []
    present_but_broken: list[str] = []
    for mod, workaround in EXPECTED_MISSING:
        try:
            spec = importlib.util.find_spec(mod)
        except Exception as exc:  # noqa: BLE001 —— 同上，显式记录
            spec = None
            LOGGER.log(
                "degrade",
                "check_dependencies",
                level="WARN",
                message=f"find_spec({mod}) 抛异常",
                exception=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        # 关键：find_spec 只说明「目录在」，必须真实 import 才能判定「能不能用」。
        # 实测教训：loguru 目录存在（元数据 0.7.3）但 import 因缺 win32_setctime 直接失败。
        usable = False
        note = ""
        if spec is not None:
            try:
                module = importlib.import_module(mod)
                usable = True
                note = f"真实 import 成功，版本 {getattr(module, '__version__', None) or dist_version((mod,))}"
            except Exception as exc:  # noqa: BLE001 —— 存在但不可用，必须如实记录
                note = f"目录存在但真实 import 失败：{type(exc).__name__}: {exc}"
                present_but_broken.append(f"{mod}（{note}）")
                LOGGER.log(
                    "degrade",
                    "check_dependencies",
                    level="WARN",
                    message=f"{mod} 目录存在但不可用，仍按缺失处理",
                    exception=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
        absent_rows.append(
            {
                "模块": mod,
                "find_spec 命中": spec is not None,
                "真实 import 可用": usable,
                "说明": note or "未发现（完全缺失）",
                "预期替代方案": workaround,
            }
        )
        if usable:
            unexpected_present.append(f"{mod}({dist_version((mod,))})")

    detail_bits = [f"{len(EXPECTED_MISSING)} 个包确认缺失或不可用"]
    if present_but_broken:
        detail_bits.append(f"其中存在但 import 失败：{present_but_broken}")
    if unexpected_present:
        detail_bits.append(f"实际可用：{unexpected_present}")
    if unexpected_present:
        items.append(
            CheckItem(
                "deps.missing",
                "五、预期缺失依赖（实测）",
                "info",
                "；".join(detail_bits) + " —— 可考虑启用官方实现，但不改本工单既定降级方案",
                required=False,
                evidence={"缺失清单（预期）": absent_rows},
            )
        )
    else:
        items.append(
            CheckItem(
                "deps.missing",
                "五、预期缺失依赖（实测）",
                "passed",
                "；".join(detail_bits) + " —— 降级方案（自实现/替代库）为唯一可行路径",
                evidence={"缺失清单（预期）": absent_rows},
            )
        )
    return items


@traced
def probe_one_pdf(path: Path, table_pages: Sequence[int], do_table: bool) -> dict[str, Any]:
    """用 PyMuPDF 实测单个 PDF：页数、文本层、随机页表格提取与两类缺陷。"""
    # 延迟导入：优先 1.28 起的正式包名 pymupdf，缺失时回退历史别名 fitz；
    # 两者都缺失则由上层捕获 ImportError 并显式降级（禁止静默失败）。
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore[no-redef]  # 旧版本仅有 fitz 别名

    info: dict[str, Any] = {
        "文件名": path.name,
        "字节数": path.stat().st_size,
        "SHA256(前16)": sha256_prefix(path),
    }
    doc = fitz.open(str(path))
    try:
        info["页数"] = doc.page_count
        meta = doc.metadata or {}
        info["producer"] = meta.get("producer")
        info["标题"] = (meta.get("title") or "").strip()[:80]

        text_probe_idx = sorted({0, doc.page_count // 2, max(0, doc.page_count - 1)})
        text_chars: dict[str, int] = {}
        for idx in text_probe_idx:
            raw = doc.load_page(idx).get_text("text") or ""
            text_chars[f"第{idx + 1}页"] = len("".join(raw.split()))
        info["文本层字符数(去空白)"] = text_chars
        info["是否文本版"] = all(v >= 100 for v in text_chars.values())

        if do_table:
            probes: list[dict[str, Any]] = []
            for page_no in table_pages:
                idx = page_no - 1
                if idx < 0 or idx >= doc.page_count:
                    probes.append({"页": page_no, "跳过原因": "超出页数范围"})
                    continue
                page = doc.load_page(idx)
                t0 = time.perf_counter()
                finder = page.find_tables()
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                tables = list(getattr(finder, "tables", []) or [])
                entry: dict[str, Any] = {"页": page_no, "表数": len(tables), "find_tables耗时ms": elapsed}
                if tables:
                    data = tables[0].extract()
                    none_cells = sum(1 for row in data for cell in row if cell is None)
                    newline_cells = sum(
                        1 for row in data for cell in row if isinstance(cell, str) and "\n" in cell
                    )
                    entry["首表形状"] = f"{len(data)}行×{max(len(r) for r in data)}列"
                    entry["首表None格数"] = none_cells
                    entry["首表含换行格数"] = newline_cells
                    entry["首表表头"] = [c if c is None else str(c).replace("\n", "\\n")[:24] for c in data[0]][:12]
                probes.append(entry)
            info["表格实测"] = probes
    finally:
        doc.close()
    return info


@traced
def check_pdfs(table_pages: Sequence[int], do_table: bool) -> list[CheckItem]:
    """自动发现 研发/data/raw/*.pdf（严禁硬编码文件名）并逐份实测。"""
    if not RAW_DIR.is_dir():
        return [
            CheckItem(
                "pdf.dir",
                "六、PDF 语料（实测）",
                "failed",
                f"语料目录不存在：{RAW_DIR}",
                evidence={"目录": str(RAW_DIR)},
            )
        ]
    pdfs = sorted(p for p in RAW_DIR.glob("*.pdf") if p.is_file())
    if not pdfs:
        return [
            CheckItem(
                "pdf.discover",
                "六、PDF 语料（实测）",
                "failed",
                f"{RAW_DIR} 下未发现任何 *.pdf（系统按通用多 PDF 设计，不得硬编码文件名）",
                evidence={"目录": str(RAW_DIR)},
            )
        ]
    if do_table and not any(p == 1 for p in table_pages):
        table_pages = [1, *table_pages]

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in pdfs:
        try:
            info = probe_one_pdf(path, table_pages, do_table)
        except ImportError as exc:
            LOGGER.log(
                "degrade",
                "check_pdfs",
                level="WARN",
                message="PyMuPDF 不可用，PDF 段降级",
                exception=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            return [
                CheckItem(
                    "pdf.probe",
                    "六、PDF 语料（实测）",
                    "degraded",
                    f"PyMuPDF(fitz) 不可用，无法实测 PDF：{exc}",
                    evidence={"发现文件": [p.name for p in pdfs]},
                )
            ]
        except Exception as exc:  # noqa: BLE001 —— 单份 PDF 失败不阻断其余，但要显式记录
            LOGGER.log(
                "error",
                "check_pdfs",
                level="ERROR",
                message=f"解析 {path.name} 失败",
                exception=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        rows.append(info)
        if not info.get("是否文本版", False):
            failures.append(f"{path.name}: 疑似扫描件（文本层字符数不足，需 OCR）")

    if failures:
        return [
            CheckItem(
                "pdf.probe",
                "六、PDF 语料（实测）",
                "failed",
                f"PDF 实测存在失败项：{failures}",
                evidence={"发现文件": [p.name for p in pdfs], "明细": rows},
            )
        ]
    total_pages = sum(int(r.get("页数", 0)) for r in rows)
    table_note = "表格提取已实测" if do_table else "仅测页数与文本层（已跳过表格实测）"
    return [
        CheckItem(
            "pdf.probe",
            "六、PDF 语料（实测）",
            "passed",
            f"自动发现 {len(pdfs)} 份 PDF，共 {total_pages} 页，均为文本版；{table_note}",
            evidence={
                "发现文件（自动发现，非硬编码）": [p.name for p in pdfs],
                "表格实测页码": list(table_pages) if do_table else "已跳过",
                "明细": rows,
            },
        )
    ]


@traced
@traced
def http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    """极简 JSON HTTP 调用（标准库，无重试）。"""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


@traced
def ollama_probe_endpoint(base: str) -> dict[str, Any]:
    """后端探测：GET /api/tags，超时 0.5 s 且不重试（硬性要求）。"""
    t0 = time.perf_counter()
    data = http_json(f"{base}/api/tags", None, OLLAMA_PROBE_TIMEOUT_S)
    return {
        "耗时ms": round((time.perf_counter() - t0) * 1000, 1),
        "模型": [m.get("name") for m in (data.get("models") or [])],
    }


@traced
def ollama_probe_embed(base: str, model: str, text: str = "环境校验探针：招股说明书的军用收入与募集资金") -> dict[str, Any]:
    """功能实测：POST /api/embed，记录向量维度与耗时（冷启动会包含加载模型的时间）。"""
    t0 = time.perf_counter()
    data = http_json(
        f"{base}/api/embed",
        {"model": model, "input": text},
        OLLAMA_EMBED_TIMEOUT_S,
    )
    embeddings = data.get("embeddings") or []
    dims = len(embeddings[0]) if embeddings else 0
    return {
        "模型": model,
        "维度": dims,
        "耗时ms": round((time.perf_counter() - t0) * 1000, 1),
        "向量条数": len(embeddings),
    }


@traced
def ollama_probe_generate(
    base: str,
    model: str,
    prompt: str = "用一句话说明：招股说明书中军用收入为何波动？",
    num_predict: int = 48,
) -> dict[str, Any]:
    """功能实测：POST /api/generate（stream=true）逐块计时，得出首字延迟。"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        f"{base}/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_ms: float | None = None
    chunk_count = 0
    text = ""
    with urllib.request.urlopen(request, timeout=OLLAMA_GEN_TIMEOUT_S) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.log(
                    "degrade",
                    "ollama_probe_generate",
                    level="WARN",
                    message="流式分块非 JSON，跳过该块",
                    exception=f"{type(exc).__name__}: {exc}",
                    raw=line[:120],
                )
                continue
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
            chunk_count += 1
            text += obj.get("response") or ""
            if obj.get("done"):
                break
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "模型": model,
        "首字ms": round(first_ms, 1) if first_ms is not None else None,
        "总耗时ms": round(total_ms, 1),
        "分块数": chunk_count,
        "输出字符数": len(text),
        "输出摘要": text.strip().replace("\n", " ")[:80],
    }


@traced
def check_ollama(base: str, embed_model: str, llm_model: str) -> list[CheckItem]:
    """Ollama 连通性：先 0.5 s 快速探测，失败即快速返回；成功再做功能实测。"""
    items: list[CheckItem] = []
    try:
        probe = ollama_probe_endpoint(base)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        LOGGER.log(
            "error",
            "check_ollama",
            level="ERROR",
            message="Ollama 端点探测失败（0.5 s 内不重试）",
            exception=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        return [
            CheckItem(
                "ollama.endpoint",
                "七、Ollama 连通性（实测）",
                "failed",
                f"{base} 探测失败（超时 {OLLAMA_PROBE_TIMEOUT_S}s 且不重试）：{type(exc).__name__}: {exc}",
                evidence={"探测地址": base, "探测超时s": OLLAMA_PROBE_TIMEOUT_S},
            ),
            CheckItem(
                "ollama.functional",
                "七、Ollama 连通性（实测）",
                "info",
                "端点不可达，已跳过嵌入/生成功能实测（系统此时应自动降级到 extractive 后端）",
                required=False,
            ),
        ]

    models = probe["模型"]
    items.append(
        CheckItem(
            "ollama.endpoint",
            "七、Ollama 连通性（实测）",
            "passed",
            f"{base} 可达，探测耗时 {probe['耗时ms']} ms（超时上限 {OLLAMA_PROBE_TIMEOUT_S}s，无重试），可用模型 {len(models)} 个",
            evidence={"可用模型": models},
        )
    )

    missing_models = [m for m in (embed_model, llm_model) if not any(str(x).startswith(m.split(":")[0]) for x in models)]
    items.append(
        CheckItem(
            "ollama.models",
            "七、Ollama 连通性（实测）",
            "degraded" if missing_models else "passed",
            (
                f"配置模型缺失：{missing_models}（需 ollama pull，或改用上表已有模型）"
                if missing_models
                else f"配置模型均在位：嵌入 {embed_model}、生成 {llm_model}"
            ),
            evidence={"配置嵌入模型": embed_model, "配置生成模型": llm_model},
        )
    )

    # 嵌入实测：先做一次短文预热（模型冷加载不进首字预算），再实测并如实并列两者耗时
    warm_embed: dict[str, Any] | None = None
    try:
        warm_embed = ollama_probe_embed(base, embed_model, "预热")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        LOGGER.log(
            "degrade",
            "check_ollama",
            level="WARN",
            message="嵌入预热调用失败，将直接实测（冷启动耗时照实记录）",
            exception=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    try:
        embed = ollama_probe_embed(base, embed_model)
        embed["冷启动耗时ms"] = warm_embed["耗时ms"] if warm_embed else "预热失败"
        ok = embed["维度"] > 0
        items.append(
            CheckItem(
                "ollama.embed",
                "七、Ollama 连通性（实测）",
                "passed" if ok else "failed",
                f"{embed_model} 返回 {embed['维度']} 维向量，预热后耗时 {embed['耗时ms']} ms（冷启动 {embed['冷启动耗时ms']} ms）",
                evidence=embed,
            )
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        LOGGER.log(
            "error",
            "check_ollama",
            level="ERROR",
            message="嵌入实测失败",
            exception=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        items.append(
            CheckItem(
                "ollama.embed",
                "七、Ollama 连通性（实测）",
                "failed",
                f"嵌入实测失败（{embed_model}）：{type(exc).__name__}: {exc}",
                evidence={"模型": embed_model, "超时s": OLLAMA_EMBED_TIMEOUT_S},
            )
        )

    # 生成实测：同样先预热（冷启动加载模型），再实测首字延迟；冷/热两组数据都写入证据
    warm_gen: dict[str, Any] | None = None
    try:
        warm_gen = ollama_probe_generate(base, llm_model, prompt="你好", num_predict=1)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        LOGGER.log(
            "degrade",
            "check_ollama",
            level="WARN",
            message="生成预热调用失败，将直接实测（冷启动耗时照实记录）",
            exception=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    try:
        gen = ollama_probe_generate(base, llm_model)
        first = gen["首字ms"]
        cold_first = warm_gen["首字ms"] if warm_gen else None
        gen["冷启动首字ms"] = cold_first
        within = first is not None and first <= FIRST_TOKEN_BUDGET_MS
        items.append(
            CheckItem(
                "ollama.generate",
                "七、Ollama 连通性（实测）",
                "passed" if within else "failed",
                (
                    f"{llm_model} 预热后首字 {first} ms（预算 ≤ {FIRST_TOKEN_BUDGET_MS:.0f} ms；冷启动首字 {cold_first} ms），总耗时 {gen['总耗时ms']} ms"
                    if within
                    else f"{llm_model} 预热后首字 {first} ms 超出预算 {FIRST_TOKEN_BUDGET_MS:.0f} ms"
                ),
                evidence=gen,
            )
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        LOGGER.log(
            "error",
            "check_ollama",
            level="ERROR",
            message="生成实测失败",
            exception=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        items.append(
            CheckItem(
                "ollama.generate",
                "七、Ollama 连通性（实测）",
                "failed",
                f"生成实测失败（{llm_model}）：{type(exc).__name__}: {exc}",
                evidence={"模型": llm_model, "超时s": OLLAMA_GEN_TIMEOUT_S},
            )
        )
    return items


@traced
def check_config_baseline() -> list[CheckItem]:
    """记录 config 基线物是否就位（T0 阶段 config.py 尚未落地，仅作存在性参考）。"""
    env_file = CONFIG_DIR / "config.example.env"
    cfg_py = REPO_ROOT / "研发" / "app" / "core" / "config.py"
    evidence: dict[str, Any] = {
        "部署/配置/config.example.env": env_file.exists(),
        "研发/app/core/config.py": cfg_py.exists(),
        "研发/data/eval/golden_qa.jsonl": (REPO_ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl").exists(),
    }
    if env_file.exists():
        try:
            keys = [
                line.split("=", 1)[0].strip()
                for line in env_file.read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("RAG_") and "=" in line
            ]
            evidence["config.example.env 中 RAG_* 变量数"] = len(keys)
        except OSError as exc:
            LOGGER.log(
                "degrade",
                "check_config_baseline",
                level="WARN",
                message="读取 config.example.env 失败，已降级为仅报告存在性",
                exception=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            evidence["config.example.env 读取"] = f"降级：{type(exc).__name__}: {exc}"
    present = [k for k, v in evidence.items() if v is True]
    return [
        CheckItem(
            "config.baseline",
            "八、配置基线（参考）",
            "info",
            f"已就位：{present}；缺失项由后续任务补齐（config.py 一致性校验在研发任务落地后追加）",
            required=False,
            evidence=evidence,
        )
    ]


@traced
def check_reference_materials() -> list[CheckItem]:
    """只读核对工单1/工单2 参考基线可访问性（严禁修改，仅记录）。"""
    golden = Path(r"E:\gao6gongdan\工单1\data\eval\golden_qa.jsonl")
    baseline = Path(r"E:\gao6gongdan\工单2\优化\基线\baseline_metrics.json")
    evidence: dict[str, Any] = {"工单1 golden_qa.jsonl": str(golden), "工单2 baseline_metrics.json": str(baseline)}
    notes: list[str] = []
    try:
        lines = [ln for ln in golden.read_text(encoding="utf-8").splitlines() if ln.strip()]
        evidence["golden 题数"] = len(lines)
        ids = []
        for ln in lines:
            try:
                ids.append(json.loads(ln).get("id"))
            except json.JSONDecodeError as exc:
                LOGGER.log(
                    "degrade",
                    "check_reference_materials",
                    level="WARN",
                    message="golden_qa.jsonl 存在非 JSON 行",
                    exception=f"{type(exc).__name__}: {exc}",
                    raw=ln[:120],
                )
        evidence["golden 题号"] = ids
        notes.append(f"工单1 golden 可读，{len(lines)} 题")
    except OSError as exc:
        evidence["golden 读取"] = f"失败：{type(exc).__name__}: {exc}"
        notes.append("工单1 golden 不可读")
        LOGGER.log(
            "degrade",
            "check_reference_materials",
            level="WARN",
            message="工单1 golden_qa.jsonl 不可读，参考项降级",
            exception=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    try:
        data = json.loads(baseline.read_text(encoding="utf-8"))
        evidence["工单2 基线条目"] = sorted(data.keys())[:20]
        notes.append("工单2 基线 JSON 可读")
    except (OSError, json.JSONDecodeError) as exc:
        evidence["基线读取"] = f"失败：{type(exc).__name__}: {exc}"
        notes.append("工单2 基线不可读")
        LOGGER.log(
            "degrade",
            "check_reference_materials",
            level="WARN",
            message="工单2 baseline_metrics.json 不可读，参考项降级",
            exception=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    return [
        CheckItem(
            "reference.materials",
            "九、只读参考基线（参考）",
            "info",
            "；".join(notes) if notes else "未发现可读参考物",
            required=False,
            evidence=evidence,
        )
    ]


@traced
def check_self_import_audit() -> CheckItem:
    """AST 静态自检：本脚本模块级只许 import 标准库，延迟 import 只许用必需依赖。"""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    def collect(nodes: Sequence[ast.AST]) -> set[str]:
        """从给定节点集合收集顶层模块名。"""
        found: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    found.add(node.module.split(".")[0])
        return found

    module_level = collect(list(tree.body))
    all_imports = collect(list(ast.walk(tree)))
    deferred = all_imports - module_level

    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    forbidden = {mod for mod, _ in EXPECTED_MISSING}
    allowed_deferred = {mod for mod, _, _ in REQUIRED_PACKAGES}

    violation = sorted(all_imports & forbidden)
    module_non_std = sorted(module_level - stdlib)
    deferred_outside = sorted(deferred - stdlib - allowed_deferred)

    evidence = {
        "模块级 import": sorted(module_level),
        "延迟（函数内）import": sorted(deferred),
        "允许的延迟 import（必需依赖）": sorted(allowed_deferred),
        "模块级非标准库 import": module_non_std,
        "越权延迟 import": deferred_outside,
        "命中禁用清单": violation,
        "禁用清单": sorted(forbidden),
    }
    if violation or module_non_std:
        return CheckItem(
            "self.import",
            "十、脚本自身 import 自检（实测）",
            "failed",
            f"违规 import：禁用包 {violation}；模块级非标准库 {module_non_std}",
            evidence=evidence,
        )
    if deferred_outside:
        return CheckItem(
            "self.import",
            "十、脚本自身 import 自检（实测）",
            "degraded",
            f"存在清单外的延迟 import：{deferred_outside}（须确认已列为必需依赖）",
            evidence=evidence,
        )
    return CheckItem(
        "self.import",
        "十、脚本自身 import 自检（实测）",
        "passed",
        f"模块级仅标准库（{len(module_level)} 个）；延迟 import 仅必需依赖 {sorted(deferred)}；未触碰任何缺失包",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
@traced
def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description=f"工单3 基座校验（{WORK_ORDER}）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--deep", action="store_true", help="真实 import 重型依赖（torch/transformers），耗时较长")
    parser.add_argument("--no-ollama", action="store_true", help="跳过 Ollama 探测（该段不参与判定）")
    parser.add_argument("--no-pdf", action="store_true", help="跳过 PDF 实测")
    parser.add_argument("--no-table", action="store_true", help="PDF 段不做 find_tables 实测（仅页数/文本层）")
    parser.add_argument("--table-pages", default="", help="逗号分隔的实测页码，默认 1,中段,末段前")
    parser.add_argument("--ollama-base", default=env_or("RAG_LLM__BASE_URL", OLLAMA_DEFAULT_BASE), help="Ollama 地址")
    parser.add_argument("--embed-model", default=env_or("RAG_EMBEDDING__MODEL", "bge-m3:latest"), help="嵌入模型")
    parser.add_argument("--llm-model", default=env_or("RAG_LLM__MODEL", "qwen2.5:3b"), help="生成模型")
    parser.add_argument("--json", dest="json_path", default=str(DEFAULT_REPORT_PATH), help="JSON 报告输出路径")
    parser.add_argument("--no-write", action="store_true", help="不落盘 JSON 报告")
    return parser


@traced
def resolve_table_pages(raw: str) -> list[int]:
    """解析 --table-pages；默认返回 1,中段,末段前 的通用页码（不依赖具体文件名）。"""
    if not raw.strip():
        return [1, 130, 200]
    pages: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        pages.append(int(token))
    return sorted(set(pages))


@traced
def run_checks(args: argparse.Namespace) -> list[CheckItem]:
    """按顺序执行全部检查段，返回检查项列表。"""
    items: list[CheckItem] = []
    items.append(check_interpreter())
    items.extend(check_entry_script())
    items.append(check_skeleton())
    items.extend(check_dependencies(args.deep))
    if not args.no_pdf:
        items.extend(check_pdfs(resolve_table_pages(args.table_pages), not args.no_table))
    if not args.no_ollama:
        items.extend(check_ollama(args.ollama_base, args.embed_model, args.llm_model))
    items.extend(check_config_baseline())
    items.extend(check_reference_materials())
    items.append(check_self_import_audit())
    return items


@traced
def write_report(path: Path, items: Sequence[CheckItem], summary: dict[str, Any], argv: Sequence[str]) -> str:
    """把检查结果写成 JSON 报告（供测试/优化阶段留痕）。"""
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "work_order": WORK_ORDER,
        "script": str(Path(__file__).resolve()),
        "script_sha256_prefix16": sha256_prefix(Path(__file__).resolve()),
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "argv": list(argv),
        "cwd": os.getcwd(),
        "summary": summary,
        "items": [asdict(item) for item in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


@traced
def main(argv: Sequence[str] | None = None) -> int:
    """入口：解析参数 → 逐段实测 → 打印结论 → 落盘报告 → 返回退出码。"""
    setup_stdio()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    LOGGER.log(
        "start",
        "main",
        input=brief({"argv": sys.argv, "cwd": os.getcwd(), "log": str(LOG_PATH)}),
        logger_degraded=LOGGER.degraded_reason,
    )

    print("=" * 72)
    print(f"工单3 基座校验｜{WORK_ORDER}")
    print(f"根目录：{REPO_ROOT}")
    print(f"脚本  ：{Path(__file__).resolve()}（SHA256 前16 {sha256_prefix(Path(__file__).resolve())}）")
    print(f"日志  ：{LOG_PATH}")
    print("=" * 72)

    items = run_checks(args)
    summary = summarize(items)
    print(render_report(items))

    print("")
    print("─" * 72)
    counts = summary["counts"]
    print(
        f"统计：✅{counts.get('passed', 0)}  ❌{counts.get('failed', 0)}  "
        f"⚠️{counts.get('degraded', 0)}  ℹ️{counts.get('info', 0)}"
    )
    if summary["degraded"]:
        print(f"降级（不阻断）：{summary['degraded']}")
    if summary["required_failed"]:
        print(f"失败项：{summary['required_failed']}")
    print(f"总体判定：{'✅ 通过' if summary['verdict'] == 'passed' else '❌ 不通过'}")

    if not args.no_write:
        path = write_report(Path(args.json_path), items, summary, sys.argv)
        print(f"JSON 报告：{path}")

    LOGGER.log(
        "finish",
        "main",
        output=brief(summary),
        exit_code=0 if summary["verdict"] == "passed" else 1,
    )
    return 0 if summary["verdict"] == "passed" else 1


if __name__ == "__main__":
    try:
        CODE = main()
    except Exception:  # noqa: BLE001 —— 顶层兜底：记录完整堆栈后以退出码 1 结束，禁止静默
        LOGGER.log("fatal", "main", level="ERROR", traceback=traceback.format_exc())
        print("\n❌ 校验脚本自身异常终止，详见上方堆栈与日志：", file=sys.stderr)
        traceback.print_exc()
        CODE = 1
    finally:
        LOGGER.close()
    raise SystemExit(CODE)
