# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""日志与通用工具。

约定：
  * 所有长日志写 ``部署/日志/<name>_<date>.log``（UTF-8），不在终端刷屏；
  * :func:`timed` 用于记录真实耗时，禁止估算；
  * :func:`write_json` / :func:`write_jsonl` 统一落盘口径（ensure_ascii=False）。
"""
from __future__ import annotations

import functools
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .config import Cfg

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, cfg: Cfg | None = None, *, console: bool = False) -> logging.Logger:
    """取（并缓存）一个写文件的 logger。"""
    key = f"{name}:{bool(console)}"
    if key in _LOGGERS:
        return _LOGGERS[key]
    logger = logging.getLogger(f"lightrag.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        if cfg is not None:
            try:
                log_dir = cfg.path("log_dir")
                fh = logging.FileHandler(
                    Path(log_dir) / f"{name}_{time.strftime('%Y%m%d')}.log", encoding="utf-8")
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except Exception:  # noqa: BLE001 - 日志不可用不应阻断主流程
                pass
        if console:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
    _LOGGERS[key] = logger
    return logger


@contextmanager
def timed(sink: dict[str, Any], key: str) -> Iterator[None]:
    """把真实耗时写进 sink[key]。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sink[key] = round(time.perf_counter() - t0, 4)


def write_json(path: Path | str, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path | str, rows: Sequence[dict[str, Any]], *, columns: Sequence[str] | None = None) -> Path:
    """写 CSV（标准库实现，不依赖 pandas）。"""
    import csv

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8-sig")
        return out
    cols = list(columns or rows[0].keys())
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in cols})
    return out


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def safe_call(fn: Callable[..., Any], *args: Any, default: Any = None, logger: logging.Logger | None = None,
              **kwargs: Any) -> Any:
    """容错包装：异常记录 + 返回默认值，避免裸崩。"""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 明确要求兜底
        if logger is not None:
            logger.warning("safe_call(%s) 失败：%s: %s", getattr(fn, "__name__", fn), type(exc).__name__, exc)
        return default


def retry(times: int = 3, sleep_s: float = 0.5) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """简单重试装饰器（用于不稳定的外部调用）。"""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last: Exception | None = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if attempt + 1 < times:
                        time.sleep(sleep_s * (attempt + 1))
            raise last  # type: ignore[misc]

        return wrapper

    return deco


def human_seconds(value: float) -> str:
    if value < 60:
        return f"{value:.2f}s"
    minutes, seconds = divmod(value, 60)
    if minutes < 60:
        return f"{int(minutes)}m{seconds:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes}m{seconds:.0f}s"
