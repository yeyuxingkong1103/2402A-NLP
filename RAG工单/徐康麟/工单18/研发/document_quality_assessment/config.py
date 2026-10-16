# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单

模块：document_quality_assessment.config
职责：加载 `研发/assessment_config.yaml`，提供点号路径取值 + 环境变量覆盖 + 路径锚定。

设计约束（验收项）：
  * Skill 的所有判断阈值只能来自本配置文件，代码里**不允许出现阈值字面量**。
  * 环境变量覆盖规则：``DQA_`` + 配置路径大写、``.`` → ``__``。
    例：``DQA_SCAN_DETECTION__SCAN_RATIO_THRESHOLD=0.6``
        ``DQA_PERFORMANCE__MAX_WORKERS=8``
  * 所有相对路径锚定到**仓库根**（本文件上溯两级 = 工单18 根目录），
    这样脚本在任意 cwd 下运行都写到同一个地方。
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

try:  # PyYAML 是本解释器已装包（见 _共享约定.md §2）
    import yaml
except ImportError as exc:  # pragma: no cover - 环境缺包时给出明确指引
    raise RuntimeError(
        "缺少 PyYAML。请用 run_py.ps1 指定的解释器运行（工单1 venv 已自带 PyYAML）。"
    ) from exc

# 本文件路径：<仓库根>/研发/document_quality_assessment/config.py → parents[2] = 仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "研发" / "assessment_config.yaml"

_ENV_PREFIX = "DQA_"


class ConfigError(RuntimeError):
    """配置缺失 / 类型错误。"""


def _coerce(raw: str) -> Any:
    """把环境变量字符串转成 YAML 标量（用 YAML 解析器保证与配置文件同语义）。"""
    try:
        value = yaml.safe_load(raw)
    except Exception:
        return raw
    return raw if value is None and raw.strip().lower() not in ("null", "~") else value


def _apply_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> list[str]:
    """把 DQA_ 环境变量写入配置树，返回生效的覆盖项列表（用于日志可追溯）。"""
    applied: list[str] = []
    for key, raw in env.items():
        if not key.startswith(_ENV_PREFIX) or key == "DQA_CONFIG":
            continue
        path = [p.lower() for p in key[len(_ENV_PREFIX):].split("__") if p]
        if not path:
            continue
        cursor: dict[str, Any] = data
        ok = True
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce(raw)
        applied.append(f"{key} -> {'.'.join(path)}={cursor[path[-1]]!r}")
    return applied


class Config:
    """配置对象：``cfg.get("scan_detection.scan_ratio_threshold", 0.7)``。

    ``default`` 只在配置文件**缺少该键**时使用；所有阈值仍以配置为准。
    """

    __slots__ = ("data", "path", "env_overrides")

    def __init__(self, data: dict[str, Any], path: Path, env_overrides: list[str] | None = None) -> None:
        self.data = data
        self.path = path
        self.env_overrides = env_overrides or []

    # -- 取值 ---------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        missing = object()
        value = self.get(dotted, missing)
        if value is missing:
            raise ConfigError(f"配置缺少必需项：{dotted}（文件：{self.path}）")
        return value

    def section(self, name: str) -> dict[str, Any]:
        node = self.get(name, {})
        if not isinstance(node, dict):
            raise ConfigError(f"配置节 {name} 应为映射，实际为 {type(node).__name__}")
        return node

    def number(self, dotted: str, default: float | None = None) -> float:
        value = self.get(dotted, default)
        if value is None:
            raise ConfigError(f"配置项 {dotted} 缺失且无默认值")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"配置项 {dotted} 应为数值，实际为 {value!r}") from exc

    def integer(self, dotted: str, default: int | None = None) -> int:
        return int(self.number(dotted, default))

    def boolean(self, dotted: str, default: bool = False) -> bool:
        value = self.get(dotted, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "是")
        return bool(value)

    # -- 路径 ---------------------------------------------------------------
    def path_of(self, dotted: str) -> Path:
        """取配置里的相对路径并锚定到仓库根。"""
        raw = str(self.require(dotted))
        p = Path(raw)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def resolve_path(self, raw: str | os.PathLike[str]) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (REPO_ROOT / p)

    # -- 调试 ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def describe_thresholds(self) -> dict[str, Any]:
        """给报告用：把本 Skill 的全部判断阈值摘出来，便于人工复核。"""
        keys = [
            "scan_detection.page_char_threshold",
            "scan_detection.scan_ratio_threshold",
            "scan_detection.mixed_ratio_threshold",
            "scan_detection.image_area_ratio_threshold",
            "scan_detection.max_pages_sampled",
            "length_stats.too_short_chars",
            "length_stats.too_long_chars",
            "duplicate_detection.simhash.enabled",
            "duplicate_detection.simhash.shingle_size",
            "duplicate_detection.simhash.hamming_threshold",
            "duplicate_detection.length_ratio_tolerance",
            "sensitive_detection.context_chars",
            "performance.max_workers",
            "performance.per_file_timeout_sec",
        ]
        return {k: self.get(k) for k in keys}


_CACHE: dict[str, Config] = {}


def load_config(path: str | os.PathLike[str] | None = None, *, use_cache: bool = True) -> Config:
    """加载配置。``path`` 为空时依次尝试 $DQA_CONFIG、仓库默认路径。"""
    raw_path = path or os.environ.get("DQA_CONFIG") or DEFAULT_CONFIG_PATH
    cfg_path = Path(raw_path)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    key = str(cfg_path)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    if not cfg_path.exists():
        raise ConfigError(f"配置文件不存在：{cfg_path}")
    try:
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 YAML 解析失败：{cfg_path} :: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"配置文件根节点应为映射：{cfg_path}")

    overrides = _apply_env_overrides(loaded, os.environ)
    cfg = Config(loaded, cfg_path, overrides)
    if use_cache:
        _CACHE[key] = cfg
    return cfg
