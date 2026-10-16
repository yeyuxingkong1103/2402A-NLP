# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

会话共享存储：``研发/service/session_store.py``。

验收项「容器数据管理 → 支持容器间数据共享」的落地实现。两条后端：

====================  ==========================================================
后端                  说明
====================  ==========================================================
``redis``             首选。``rag-api`` / ``rag-ui`` / 后续新容器连同一 Redis，
                      会话跨容器、跨副本共享（键前缀 ``DEPLOY_REDIS_KEY_PREFIX``）。
``file``              兜底。落盘到**具名卷/绑定挂载目录**（``DEPLOY_SESSION_DIR``），
                      同宿主机多容器挂同一卷即可共享，且 Redis 挂了不丢数据。
====================  ==========================================================

关键设计点（三条，都是踩坑换来的，见 ``部署/过程问题记录.md``）：

1. **Redis 不可用不能拖垮问答**：Redis 客户端连接失败在部分版本里会阻塞较久，
   因此这里做「惰性探测 + 熔断」（``DEPLOY_REDIS_TIMEOUT_S`` 秒超时，
   连续失败 ``_BREAKER_THRESHOLD`` 次后转文件后端，``_BREAKER_COOLDOWN_S`` 秒后半开重试）。
2. **双写**：即便 Redis 可用，也同时写文件卷 —— 这样「数据不丢失」验收不依赖任何外部服务，
   拔掉 Redis 容器后会话仍可从卷里读回。
3. **redis 客户端为可选依赖**：本机解释器未装 ``redis``（实测 ``ModuleNotFoundError``），
   导入失败必须降级而不是崩，容器内则按需 ``pip install redis``。
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import DeployConfig

#: 熔断阈值（常量，属于实现细节而非业务阈值，故不放环境变量）
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_S = 30.0

try:  # pragma: no cover —— 依赖可用性随环境而变
    import redis as _redis_module  # type: ignore
except Exception:  # noqa: BLE001 —— 没装 redis 包也不影响：走本模块自带的 RESP 客户端
    _redis_module = None


# ====================================================================== #
# 零依赖 Redis 客户端（标准库 socket 实现的 RESP 协议子集）
#
# 为什么自己写（**实测动机**）：
#   本机解释器**未安装 redis 包**（`ModuleNotFoundError: No module named 'redis'`），
#   但 Redis 服务**是在跑的** —— 实测 `6379` 端口发 `PING\r\n` 收到 `+PONG\r\n`。
#   若强依赖 redis 包，就会出现"Redis 明明可用、服务却只能用文件后端"的**假降级**。
#   本项目只需 RESP 的一个子集：PING / AUTH / SELECT / RPUSH / LTRIM / EXPIRE /
#   LRANGE / LLEN / SCAN / TYPE / DEL / FLUSHDB，130 行足够且零外部依赖。
#   容器内若装了 redis 包，仍**优先**用官方包（见 RedisSessionStore.__init__）。
# ====================================================================== #
class RedisError(RuntimeError):
    """RESP 层错误（含服务端 ``-ERR`` 响应）。"""


class MiniRedis:
    """单连接、线程安全的 RESP 客户端（够用即止：不做集群/管道/哨兵）。"""

    def __init__(self, host: str, port: int = 6379, *, db: int = 0, password: str | None = None,
                 timeout_s: float = 0.5) -> None:
        self.host = host
        self.port = int(port)
        self.db = int(db)
        self.password = password or None
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None
        self._fp: Any = None
        self._lock = threading.Lock()

    # -- 连接管理 ------------------------------------------------------ #
    def _connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._sock = sock
        self._fp = sock.makefile("rb")
        if self.password:
            self._command("AUTH", self.password)
        if self.db:
            self._command("SELECT", str(self.db))

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        for handle in (self._fp, self._sock):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
        self._fp = None
        self._sock = None

    # -- RESP 编解码 --------------------------------------------------- #
    @staticmethod
    def _encode(*args: Any) -> bytes:
        out = [f"*{len(args)}\r\n".encode()]
        for arg in args:
            payload = str(arg).encode("utf-8")
            out.append(f"${len(payload)}\r\n".encode() + payload + b"\r\n")
        return b"".join(out)

    def _read_reply(self) -> Any:
        line = self._fp.readline()
        if not line:
            raise RedisError("连接已被服务端关闭（空响应）")
        kind, body = line[:1], line[1:-2]
        if kind == b"+":
            return body.decode("utf-8", "replace")
        if kind == b"-":
            raise RedisError(body.decode("utf-8", "replace"))
        if kind == b":":
            return int(body)
        if kind == b"$":
            length = int(body)
            if length == -1:
                return None
            data = self._fp.read(length + 2)
            return data[:-2].decode("utf-8", "replace")
        if kind == b"*":
            count = int(body)
            if count == -1:
                return None
            return [self._read_reply() for _ in range(count)]
        raise RedisError(f"无法识别的 RESP 类型：{kind!r}")

    def _command(self, *args: Any) -> Any:
        """发送一条命令并读回响应（内部使用，调用方需持锁）。"""
        if self._sock is None:
            self._connect()
        self._sock.sendall(self._encode(*args))   # type: ignore[union-attr]
        return self._read_reply()

    def do(self, *args: Any) -> Any:
        """线程安全执行；连接被对端断开时重建一次再重试。"""
        with self._lock:
            try:
                return self._command(*args)
            except RedisError as exc:
                if str(exc).split(" ", 1)[0] in {"ERR", "WRONGTYPE", "NOAUTH", "NOPERM", "EXECABORT"}:
                    raise           # 服务端语义错误，重连无用
                self._close_locked()
                return self._command(*args)
            except OSError:
                self._close_locked()
                return self._command(*args)

    # -- 便利方法 ------------------------------------------------------ #
    def ping(self) -> bool:
        return str(self.do("PING")).upper() == "PONG"

    @staticmethod
    def parse_url(url: str, *, default_timeout: float = 0.5) -> dict[str, Any]:
        """解析 ``redis://[:password@]host:port/db``（兼容无 scheme 的 ``host:port``）。"""
        raw = url.strip()
        if "://" not in raw:
            raw = "redis://" + raw
        parsed = urlparse(raw)
        db = 0
        if parsed.path and parsed.path.strip("/"):
            try:
                db = int(parsed.path.strip("/").split("/")[0])
            except ValueError:
                db = 0
        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": int(parsed.port or 6379),
            "db": db,
            "password": unquote(parsed.password) if parsed.password else None,
            "timeout_s": default_timeout,
        }


@dataclass
class Turn:
    """一轮对话（问 + 答 + 引用页）。"""

    role: str
    question: str = ""
    answer: str = ""
    refused: bool = False
    cited_pages: list[int] = field(default_factory=list)
    trace_id: str = ""
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "cited_pages": list(self.cited_pages),
            "trace_id": self.trace_id,
            "ts": self.ts,
        }


class FileSessionStore:
    """文件后端：每条会话一个 ``<session_id>.json``，落在会话卷里。"""

    def __init__(self, directory: str | Path, *, ttl_s: int = 86400, max_turns: int = 20) -> None:
        self.dir = Path(directory)
        self.ttl_s = int(ttl_s)
        self.max_turns = int(max_turns)
        self._lock = threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- 内部 ---------------------------------------------------------- #
    def _path(self, session_id: str) -> Path:
        # 会话 id 来自外部请求，必须做文件名消毒，防目录穿越（``../``）与非法字符
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(session_id))[:80]
        return self.dir / f"{safe or 'anon'}.json"

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"session_id": path.stem, "turns": [], "updated_at": 0.0}
        except (json.JSONDecodeError, OSError):
            # 损坏文件按空会话处理（不抛错：会话损坏不该让问答 500）
            return {"session_id": path.stem, "turns": [], "updated_at": 0.0, "corrupt": True}

    # -- 对外 ---------------------------------------------------------- #
    def append(self, session_id: str, turn: Turn) -> dict[str, Any]:
        with self._lock:
            path = self._path(session_id)
            data = self._read(path)
            turns = list(data.get("turns") or [])
            turns.append(turn.to_dict())
            if len(turns) > self.max_turns:
                turns = turns[-self.max_turns:]
            data.update({"session_id": path.stem, "turns": turns, "updated_at": time.time()})
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)   # 原子替换：容器被 kill 也不会留下半截 JSON
            return data

    def load(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._read(self._path(session_id))
        if self.ttl_s > 0:
            updated = float(data.get("updated_at") or 0.0)
            if updated and (time.time() - updated) > self.ttl_s:
                data["expired"] = True
        return data

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = self._read(path)
            turns = data.get("turns") or []
            rows.append(
                {
                    "session_id": data.get("session_id") or path.stem,
                    "turns": len(turns),
                    "updated_at": data.get("updated_at") or 0.0,
                    "last_question": (turns[-1].get("question") if turns else ""),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def count(self) -> int:
        return len(list(self.dir.glob("*.json")))

    def health(self) -> dict[str, Any]:
        try:
            probe = self.dir / ".health"
            probe.write_text(str(time.time()), encoding="utf-8")
            probe.unlink(missing_ok=True)
            return {"ok": True, "detail": f"可写：{self.dir}", "sessions": self.count()}
        except OSError as exc:
            return {"ok": False, "detail": f"不可写：{exc}"}


class RedisSessionStore:
    """Redis 后端（前缀 + TTL），把每轮对话 append 到 LIST。

    客户端选择（按可用性自动挑，**不再强制依赖 redis 包**）：
        * 装了 ``redis`` 包 → 用官方客户端（支持连接池）；
        * 没装 → 用本模块自带的 :class:`MiniRedis`（标准库 socket + RESP 子集）。
    两者暴露同一套调用面（``do`` / ``ping``），上层代码不区分。
    """

    def __init__(self, url: str, *, prefix: str = "rag:finqa:", timeout_s: float = 0.5,
                 ttl_s: int = 86400, max_turns: int = 20) -> None:
        self.url = url
        self.prefix = prefix
        self.timeout_s = float(timeout_s)
        self.ttl_s = int(ttl_s)
        self.max_turns = int(max_turns)
        self.client_kind = "miniredis"
        if _redis_module is not None:
            self._client = _redis_module.Redis.from_url(
                url, socket_timeout=self.timeout_s, socket_connect_timeout=self.timeout_s,
                decode_responses=True,
            )
            self.client_kind = "redis-py"
        else:
            params = MiniRedis.parse_url(url, default_timeout=self.timeout_s)
            self._client = MiniRedis(**params)

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}session:{session_id}"

    def append(self, session_id: str, turn: Turn) -> dict[str, Any]:
        key = self._key(session_id)
        payload = json.dumps(turn.to_dict(), ensure_ascii=False)
        if self.client_kind == "redis-py":
            pipe = self._client.pipeline()
            pipe.rpush(key, payload)
            pipe.ltrim(key, -self.max_turns, -1)
            if self.ttl_s > 0:
                pipe.expire(key, self.ttl_s)
            pipe.execute()
        else:
            self._client.do("RPUSH", key, payload)
            self._client.do("LTRIM", key, -self.max_turns, -1)
            if self.ttl_s > 0:
                self._client.do("EXPIRE", key, self.ttl_s)
        return self.load(session_id)

    def load(self, session_id: str) -> dict[str, Any]:
        key = self._key(session_id)
        if self.client_kind == "redis-py":
            raw = self._client.lrange(key, 0, -1) or []
        else:
            raw = self._client.do("LRANGE", key, 0, -1) or []
        turns = []
        for item in raw:
            try:
                turns.append(json.loads(item))
            except (json.JSONDecodeError, TypeError):
                continue
        return {"session_id": session_id, "turns": turns, "updated_at": time.time()}

    def _scan_keys(self, limit: int) -> list[str]:
        """SCAN 遍历（不用 KEYS：生产环境阻塞风险）。"""
        if self.client_kind == "redis-py":
            keys = []
            for key in self._client.scan_iter(match=f"{self.prefix}session:*", count=100):
                keys.append(key)
                if len(keys) >= limit:
                    break
            return keys
        keys: list[str] = []
        cursor = "0"
        pattern = f"{self.prefix}session:*"
        while True:
            reply = self._client.do("SCAN", cursor, "MATCH", pattern, "COUNT", 100)
            cursor, batch = str(reply[0]), list(reply[1] or [])
            keys.extend(batch)
            if cursor == "0" or len(keys) >= limit:
                break
        return keys[:limit]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in self._scan_keys(limit):
            if self.client_kind == "redis-py":
                turns = self._client.lrange(key, -1, -1) or []
                count = int(self._client.llen(key) or 0)
            else:
                turns = self._client.do("LRANGE", key, -1, -1) or []
                count = int(self._client.do("LLEN", key) or 0)
            last_question = ""
            if turns:
                try:
                    last_question = json.loads(turns[0]).get("question", "")
                except (json.JSONDecodeError, TypeError):
                    last_question = ""
            rows.append(
                {
                    "session_id": key.split("session:", 1)[-1],
                    "turns": count,
                    "updated_at": time.time(),
                    "last_question": last_question,
                }
            )
        return rows

    def health(self) -> dict[str, Any]:
        try:
            if self.client_kind == "redis-py":
                self._client.ping()
            else:
                self._client.ping()
            info = f"PING 成功：{self.url}"
            try:
                if self.client_kind == "miniredis":
                    info += f"（客户端：内置 MiniRedis/RESP，无需 redis 包）"
            except Exception:  # noqa: BLE001
                pass
            return {"ok": True, "detail": info, "client": self.client_kind}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "client": self.client_kind}

    def info(self) -> dict[str, Any]:
        """运维用：客户端种类 + 该前缀下的会话键数（用于验证跨容器共享）。"""
        keys = self._scan_keys(1000)
        return {"client": self.client_kind, "url": self.url, "prefix": self.prefix,
                "session_keys": len(keys), "sample": keys[:5]}


class SharedSessionStore:
    """对外的统一门面：``auto`` 模式 Redis 优先、文件兜底，且始终双写文件卷。"""

    def __init__(self, cfg: DeployConfig) -> None:
        self.cfg = cfg
        self.file = FileSessionStore(cfg.session_dir, ttl_s=cfg.session_ttl_s,
                                     max_turns=cfg.session_max_turns)
        self.redis: RedisSessionStore | None = None
        self.backend = "file"
        self.degraded_reason = ""
        self._failures = 0
        self._breaker_until = 0.0
        self._lock = threading.Lock()
        self._maybe_init_redis()

    # -- Redis 可用性（惰性 + 熔断） ------------------------------------ #
    def _maybe_init_redis(self) -> None:
        want_redis = self.cfg.resolved_session_backend == "redis"
        if not want_redis:
            self.degraded_reason = "DEPLOY_SESSION_BACKEND=file（显式指定文件后端）"
            return
        try:
            store = RedisSessionStore(
                self.cfg.redis_url, prefix=self.cfg.redis_key_prefix,
                timeout_s=self.cfg.redis_timeout_s, ttl_s=self.cfg.session_ttl_s,
                max_turns=self.cfg.session_max_turns,
            )
            health = store.health()
            if health["ok"]:
                self.redis = store
                self.backend = "redis+file"
            else:
                self.degraded_reason = f"Redis 不可达：{health['detail']}"
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"Redis 初始化失败：{type(exc).__name__}: {exc}"

    def _redis_ok(self) -> bool:
        if self.redis is None:
            return False
        if time.time() < self._breaker_until:
            return False
        return True

    def _trip(self, exc: Exception) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= _BREAKER_THRESHOLD:
                self._breaker_until = time.time() + _BREAKER_COOLDOWN_S
                self._failures = 0
            self.backend = "file"
            self.degraded_reason = f"Redis 运行期失败转文件后端：{type(exc).__name__}: {exc}"

    # -- 对外 ---------------------------------------------------------- #
    def append(self, session_id: str, turn: Turn) -> dict[str, Any]:
        # ① 文件卷必写（数据不丢失的硬保障）
        data = self.file.append(session_id, turn)
        # ② Redis 尽力写（跨容器/跨副本实时共享）
        if self._redis_ok():
            try:
                self.redis.append(session_id, turn)   # type: ignore[union-attr]
                self.backend = "redis+file"
            except Exception as exc:  # noqa: BLE001
                self._trip(exc)
        return data

    def load(self, session_id: str) -> dict[str, Any]:
        if self._redis_ok():
            try:
                data = self.redis.load(session_id)   # type: ignore[union-attr]
                if data.get("turns"):
                    return data
            except Exception as exc:  # noqa: BLE001
                self._trip(exc)
        return self.file.load(session_id)

    def history(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        turns = list(self.load(session_id).get("turns") or [])
        return turns[-limit:] if limit else turns

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if self._redis_ok():
            try:
                return self.redis.recent(limit)   # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                self._trip(exc)
        return self.file.recent(limit)

    def health(self) -> dict[str, Any]:
        file_health = self.file.health()
        if self.redis is not None:
            redis_health = self.redis.health()
            try:
                redis_health.update(self.redis.info())      # 前缀下会话键数（共享证据）
            except Exception as exc:  # noqa: BLE001
                redis_health["info_error"] = f"{type(exc).__name__}: {exc}"
        else:
            redis_health = {"ok": False, "detail": self.degraded_reason or "未启用"}
        return {
            "backend": self.backend,
            "file": file_health,
            "redis": redis_health,
            "degraded_reason": self.degraded_reason,
            "session_dir": str(self.cfg.session_dir),
            "sessions_on_disk": self.file.count(),
            "redis_enabled": bool(self.cfg.redis_enabled),
        }
