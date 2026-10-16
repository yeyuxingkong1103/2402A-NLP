# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化
工单6 增补：人工智能NLP-RAG-混合检索任务
    —— ``make_key(..., strategy_fp=...)`` 新增形参：把检索策略指纹并入会话内缓存键，
    默认 ``""`` 时键值与 工单5 完全一致（既有会话缓存不失效）。

多轮对话级缓存（``DialogueCache``）—— 缓存「改写结果 + 完整多轮答案」，避免同一会话里
重复提问走完整链路（改写 → 检索 → 生成）。

## 设计要点（与工单4 的问题级缓存 ``qa_engine._memory_cache`` 的区别）
| 维度 | 工单4 问题级缓存 | 本模块（对话级缓存） |
| --- | --- | --- |
| 缓存对象 | 单轮 ``Answer`` | **整轮**多轮结果（含 rewritten_query / strategy / focus_entity） |
| 键的构成 | ``PIPELINE_VERSION \\| question + lang + top_k`` | ``rewritten_query + language + **session_id**`` |
| 能否跨会话复用 | 能（同一问句答同一条） | **不能**：同一问句在不同会话里上下文不同，焦点实体可能不同 |

**键为什么必须含 ``session_id``**：多轮问答里「这个公司的法定代表人是谁？」在第 1 个会话可能
指兴图新科、在第 2 个会话可能指力源信息；若跨会话复用，就会把 A 公司的答案发给 B 会话。
本模块因此把 ``session_id`` 编进 sha1，**宁可少命中，也绝不串台**；「清空对话」时调用
``invalidate_session`` 精确剔除该会话条目。

## 可靠性约定（缓存故障绝不影响问答）
* 日志：``logger`` 可能为 ``None`` —— 用 ``_NullLogger`` 兜底，**不抛异常**；
* 落盘：``persist_path`` 非空时写 JSON（**先写 ``.tmp`` 再 ``replace``，原子**）；
  读写盘任何异常只记 ``WARNING`` 并降级为纯内存，**绝不向上抛**；
* 线程安全：单把 ``RLock`` 保护全部状态（命中/淘汰/落盘计数一致）。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"

#: 落盘格式版本（将来改结构时用它识别旧文件）
PERSIST_VERSION = 1


def _lazy_logger(logger: Any, module: str = "dialogue_cache") -> Any:
    """延迟取 logger，避免与 logging_conf 循环导入。"""
    if logger is not None:
        return logger
    try:
        from .logging_conf import get_logger

        return get_logger(module)
    except Exception:  # noqa: BLE001 —— 日志不可用时不得阻断对话
        return _NullLogger()


class _NullLogger:
    """日志后端不可用时的空实现（保证问答链路永不因日志而失败）。"""

    def log_event(self, *_a: Any, **_kw: Any) -> None:
        return None

    def bind(self, **_kw: Any) -> "_NullLogger":
        return self

    class _Span:
        def set_output(self, *_a: Any, **_kw: Any) -> None:
            return None

        def __enter__(self) -> "_NullLogger._Span":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

    def enter(self, *_a: Any, **_kw: Any) -> "_NullLogger._Span":
        return _NullLogger._Span()


def _safe_log(logger: Any, event: str, **fields: Any) -> None:
    """记日志；**日志后端本身抛异常也必须吞掉**（缓存故障绝不影响问答）。"""
    try:
        logger.log_event(event, **fields)
    except Exception:  # noqa: BLE001 —— 日志失败不是业务失败
        return None


class DialogueCache:
    """多轮对话级缓存：改写结果 + 完整多轮答案。LRU + TTL + 可选落盘。线程安全。

    参数：
        max_size:     条目上限（LRU，超出淘汰最久未使用，计 ``evictions``）。
        ttl_s:        条目存活秒数（过期即 miss，计 ``expired``）。``<=0`` 表示永不过期。
        persist_path: 非空则落盘 JSON（原子写）；构造时尝试 ``load()`` 回填。
        logger:       任意带 ``log_event(**kwargs)`` 的对象；``None`` 时用空实现兜底。
    """

    def __init__(self, *, max_size: int = 512, ttl_s: float = 1800.0,
                 persist_path: str | Path | None = None, logger: Any = None) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl_s = float(ttl_s)
        self.log = _lazy_logger(logger)
        self.persist_path = Path(persist_path) if persist_path else None
        self._entries: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()
        self._session_keys: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0,
                       "puts": 0, "invalidations": 0, "load_errors": 0, "save_errors": 0}
        self._loaded = 0
        if self.persist_path is not None:
            self.load()

    # ------------------------------------------------------------------ 键
    @staticmethod
    def make_key(rewritten_query: str, language: str, session_id: str,
                 strategy_fp: str = "") -> str:
        """会话内缓存键：``sha1(改写后问句 + 语言 + session_id)`` 前 16 位。

        **必须含 session_id**：同一问句在不同会话里上下文不同、焦点实体可能不同，
        不能跨会话复用（否则 A 公司的答案会串到 B 会话）。

        工单6 新增 ``strategy_fp``：把**检索策略指纹**并入键。同一问句在「向量检索」
        与「全文检索」下证据不同、答案可能不同，不区分就会互相污染
        （策略对比测试会因此得出「三模式答案完全一样」的错误结论）。
        默认 ``""`` 时键值与 工单5 完全一致 —— 既有会话缓存不失效。
        """
        raw = "\x1f".join([
            (rewritten_query or "").strip(),
            (language or "auto").strip().lower(),
            (session_id or "default").strip() or "default",
        ])
        if strategy_fp:
            raw = raw + "\x1f" + strategy_fp
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ TTL
    def _is_expired(self, key: str, ts: float, now: float) -> bool:
        if self.ttl_s <= 0:
            return False
        return (now - float(ts)) > self.ttl_s

    def _drop(self, key: str) -> None:
        """删除条目并同步会话索引（**调用方须已持锁**）。"""
        self._entries.pop(key, None)
        for sid, keys in list(self._session_keys.items()):
            keys.discard(key)
            if not keys:
                self._session_keys.pop(sid, None)

    # ------------------------------------------------------------------ 主接口
    def get(self, key: str) -> dict[str, Any] | None:
        """命中返回**深拷贝**（调用方可能改）；过期按 miss 处理并删除。"""
        if not key:
            with self._lock:
                self._stats["misses"] += 1
            return None
        now = time.time()
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                self._stats["misses"] += 1
                return None
            ts, value = item
            if self._is_expired(key, ts, now):
                self._drop(key)
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                _safe_log(self.log, "dialogue_cache.expired", key=key, ttl_s=self.ttl_s)
                return None
            self._entries.move_to_end(key)        # LRU：命中即刷新为「最新」
            self._stats["hits"] += 1
            return deepcopy(value)

    def put(self, key: str, value: dict[str, Any]) -> None:
        """写入一条（``value`` 会深拷贝留底；超上限淘汰最旧并计 ``evictions``）。"""
        if not key or not isinstance(value, dict):
            return
        now = time.time()
        with self._lock:
            payload = deepcopy(value)
            self._entries[key] = (now, payload)
            self._entries.move_to_end(key)
            self._stats["puts"] += 1
            sid = str(payload.get("session_id") or "").strip()
            if sid:
                self._session_keys.setdefault(sid, set()).add(key)
            while len(self._entries) > self.max_size:
                old_key, _ = self._entries.popitem(last=False)   # 淘汰最久未使用
                self._stats["evictions"] += 1
                for keys in self._session_keys.values():
                    keys.discard(old_key)
                _safe_log(self.log, "dialogue_cache.evict", key=old_key,
                          size=len(self._entries), max_size=self.max_size)
        self._save()

    def invalidate_session(self, session_id: str) -> int:
        """清掉某会话的全部条目，返回删除条数（「清空对话」时必须调用）。"""
        sid = (session_id or "default").strip() or "default"
        with self._lock:
            keys = sorted(self._session_keys.pop(sid, set()))
            removed = 0
            for key in keys:
                if self._entries.pop(key, None) is not None:
                    removed += 1
            # 兜底：value.session_id 未写入时按值再扫一遍（保证语义正确）
            for key, (_ts, value) in list(self._entries.items()):
                if str(value.get("session_id") or "") == sid:
                    self._entries.pop(key, None)
                    removed += 1
            if removed:
                self._stats["invalidations"] += removed
        if removed:
            self._save()
            _safe_log(self.log, "dialogue_cache.invalidate_session", session_id=sid,
                      removed=removed, size=self.size())
        return removed

    def clear(self) -> None:
        """清空全部条目（不动累计统计）。"""
        with self._lock:
            self._entries.clear()
            self._session_keys.clear()
        self._save()
        _safe_log(self.log, "dialogue_cache.clear")

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        """命中/未命中/命中率/条目数/淘汰数/过期数/上限/TTL。"""
        with self._lock:
            hits, misses = self._stats["hits"], self._stats["misses"]
            total = hits + misses
            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 4) if total else 0.0,
                "size": len(self._entries),
                "evictions": self._stats["evictions"],
                "expired": self._stats["expired"],
                "max_size": self.max_size,
                "ttl_s": self.ttl_s,
                "puts": self._stats["puts"],
                "invalidations": self._stats["invalidations"],
                "sessions": len(self._session_keys),
                "loaded": self._loaded,
                "load_errors": self._stats["load_errors"],
                "save_errors": self._stats["save_errors"],
                "persist_path": str(self.persist_path) if self.persist_path else None,
            }

    # ------------------------------------------------------------------ 落盘
    def load(self) -> int:
        """从 ``persist_path`` 回填（进程启动时调用）；返回回填条数。

        文件不存在 / 损坏 / 无权限 → 只记 WARNING 并返回 0，**不抛异常**。
        """
        path = self.persist_path
        if path is None or not path.is_file():
            return 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            entries = raw.get("entries") if isinstance(raw, dict) else None
            if not isinstance(entries, list):
                entries = []
            now = time.time()
            restored = 0
            with self._lock:
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("key") or "")
                    value = item.get("value")
                    if not key or not isinstance(value, dict):
                        continue
                    ts = float(item.get("ts") or now)
                    if self._is_expired(key, ts, now):
                        self._stats["expired"] += 1
                        continue
                    self._entries[key] = (ts, value)
                    sid = str(value.get("session_id") or "").strip()
                    if sid:
                        self._session_keys.setdefault(sid, set()).add(key)
                    restored += 1
                while len(self._entries) > self.max_size:
                    self._entries.popitem(last=False)
                    self._stats["evictions"] += 1
                self._loaded = restored
            _safe_log(self.log, "dialogue_cache.loaded", path=str(path), restored=restored)
            return restored
        except Exception as exc:  # noqa: BLE001 —— 缓存读盘失败绝不影响问答
            with self._lock:
                self._stats["load_errors"] += 1
            _safe_log(self.log, "dialogue_cache.load_failed", level="WARNING",
                      path=str(path), error=f"{type(exc).__name__}: {exc}")
            return 0

    def save(self) -> bool:
        """把当前条目原子落盘（先写 ``.tmp`` 再 ``replace``）；失败只记 WARNING。"""
        path = self.persist_path
        if path is None:
            return False
        try:
            if path.exists() and not path.is_file():
                raise IsADirectoryError(f"落盘路径不是文件：{path}")
            with self._lock:
                entries = [{"key": k, "ts": float(ts), "value": value}
                           for k, (ts, value) in self._entries.items()]
            payload = {"version": PERSIST_VERSION, "work_order": WORK_ORDER,
                       "saved_at": time.time(), "ttl_s": self.ttl_s,
                       "max_size": self.max_size, "entries": entries}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)          # 原子替换：读者永远看不到半截文件
            return True
        except Exception as exc:  # noqa: BLE001 —— 缓存写盘失败绝不影响问答
            with self._lock:
                self._stats["save_errors"] += 1
            _safe_log(self.log, "dialogue_cache.save_failed", level="WARNING",
                      path=str(path), error=f"{type(exc).__name__}: {exc}")
            return False

    def _save(self) -> None:
        """内部落盘钩子（写失败已由 ``save`` 吞掉）。"""
        self.save()


__all__ = ["WORK_ORDER", "PERSIST_VERSION", "DialogueCache"]
