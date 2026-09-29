# -*- coding: utf-8 -*-
"""
RoleManager · 角色卡管理器
- 从 roles/*.yaml 加载角色，字段校验 + 类型转换 + 默认值兜底
- 单例、线程安全、支持热重载
- 兼容旧接口：.roles / .get() / .list_roles()
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    raise ImportError("请先安装 PyYAML：pip install pyyaml")

import config as C
from logger import get_logger

log = get_logger("RoleManager")

ROLE_DIR = Path(getattr(C, "ROLE_DIR", Path(__file__).parent / "roles"))
DEFAULT_ROLE = getattr(C, "DEFAULT_ROLE", "lawyer_friend")


# ======================== 数据模型 ========================
@dataclass
class Role:
    id: str
    name: str
    type: str = ""
    greeting: str = ""
    persona: str = ""
    personality: str = ""
    speaking_style: str = ""
    expertise: str = ""
    goals: List[str] = field(default_factory=list)
    boundaries: List[str] = field(default_factory=list)
    temperature: float = 0.5
    rag_top_k: int = 5
    disclaimer: str = ""
    category: str = "general"      # entry / expert / adversary / general
    avatar: str = ""
    tags: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    # --- 让 Role 兼容 dict 语义，保持向后兼容 ---
    def __getitem__(self, k):
        try:
            return getattr(self, k)
        except AttributeError:
            if k in self.raw:
                return self.raw[k]
            raise KeyError(k)

    def get(self, k, default=None):
        if hasattr(self, k):
            return getattr(self, k)
        return self.raw.get(k, default)

    def __contains__(self, k):
        return hasattr(self, k) or k in self.raw

    def keys(self):   return self.to_dict().keys()
    def items(self):  return self.to_dict().items()
    def values(self): return self.to_dict().values()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d

    def brief(self) -> Dict[str, Any]:
        """前端只需要这些字段"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "greeting": self.greeting,
            "category": self.category,
            "avatar": self.avatar,
        }


class RoleError(Exception):
    """角色卡格式错误"""


# ======================== 管理器 ========================
class RoleManager:
    _instance = None
    _cls_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, role_dir: Optional[Path] = None, auto_reload: bool = False):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.role_dir = Path(role_dir) if role_dir else ROLE_DIR
        self.auto_reload = auto_reload
        self._lock = threading.RLock()
        self._mtimes: Dict[str, float] = {}
        self.roles: Dict[str, Role] = {}
        self.load()

    # ---------- 加载 ----------
    def load(self) -> int:
        with self._lock:
            if not self.role_dir.exists():
                log.warning("角色目录不存在: %s", self.role_dir)
                return 0
            self.roles.clear()
            self._mtimes.clear()

            files = sorted(list(self.role_dir.glob("*.yaml")) + list(self.role_dir.glob("*.yml")))
            ok, fail = 0, 0
            for p in files:
                try:
                    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    role = self._parse(p, data)
                    self.roles[role.id] = role
                    self._mtimes[str(p)] = p.stat().st_mtime
                    ok += 1
                except RoleError as e:
                    log.warning("跳过角色文件 %s：%s", p.name, e)
                    fail += 1
                except Exception as e:
                    log.exception("加载角色失败 %s：%s", p.name, e)
                    fail += 1
            log.info("角色加载完成：成功 %d，失败 %d", ok, fail)
            return ok

    def reload_if_changed(self) -> bool:
        """检查磁盘是否变化，有变化则重载"""
        with self._lock:
            if not self.role_dir.exists():
                return False
            files = list(self.role_dir.glob("*.y*ml"))
            changed = len(self._mtimes) != len(files)
            if not changed:
                for p in files:
                    if self._mtimes.get(str(p)) != p.stat().st_mtime:
                        changed = True
                        break
            if changed:
                log.info("检测到角色卡变化，重新加载")
                self.load()
            return changed

    # ---------- 解析与校验 ----------
    _REQUIRED = ("id", "name")

    def _parse(self, path: Path, data: Dict[str, Any]) -> Role:
        if not isinstance(data, dict):
            raise RoleError("不是有效的 YAML 对象")
        for k in self._REQUIRED:
            if not data.get(k):
                raise RoleError(f"缺少必填字段：{k}")

        role_id = str(data["id"]).strip()
        if not role_id.replace("_", "").replace("-", "").isalnum():
            raise RoleError(f"id 只能包含字母数字和下划线：{role_id}")

        def _list(v):
            if v is None: return []
            if isinstance(v, str):
                return [x.strip() for x in v.replace("；", ";").split(";") if x.strip()]
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
            return [str(v)]

        def _f(v, d):
            try: return float(v)
            except (TypeError, ValueError): return d

        def _i(v, d):
            try: return int(v)
            except (TypeError, ValueError): return d

        return Role(
            id=role_id,
            name=str(data.get("name")).strip(),
            type=str(data.get("type", "")).strip(),
            greeting=str(data.get("greeting", "")).strip(),
            persona=str(data.get("persona", "")).strip(),
            personality=str(data.get("personality", "")).strip(),
            speaking_style=str(data.get("speaking_style", "")).strip(),
            expertise=str(data.get("expertise", "")).strip(),
            goals=_list(data.get("goals")),
            boundaries=_list(data.get("boundaries")),
            temperature=_f(data.get("temperature"), 0.5),
            rag_top_k=_i(data.get("rag_top_k"), 5),
            disclaimer=str(data.get("disclaimer", "")).strip(),
            category=str(data.get("category", "general")).strip() or "general",
            avatar=str(data.get("avatar", "")).strip(),
            tags=_list(data.get("tags")),
            raw=data,
        )

    # ---------- 查询 ----------
    def get(self, role_id: str) -> Role:
        with self._lock:
            if role_id and role_id in self.roles:
                return self.roles[role_id]
            if DEFAULT_ROLE in self.roles:
                return self.roles[DEFAULT_ROLE]
            if self.roles:
                return next(iter(self.roles.values()))
        return Role(
            id="default", name="AI助手", type="助手",
            greeting="你好，有什么可以帮你？",
            persona="你是一个乐于助人的AI助手。",
            personality="友好、专业", speaking_style="简洁清晰",
            expertise="通用", temperature=0.5,
        )

    def has(self, role_id: str) -> bool:
        return role_id in self.roles

    def ids(self) -> List[str]:
        return list(self.roles.keys())

    def list_roles(self) -> List[Dict[str, Any]]:
        """兼容旧接口：给前端的精简列表"""
        with self._lock:
            return [r.brief() for r in self.roles.values()]

    def list_full(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self.roles.values()]

    def by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        with self._lock:
            for r in self.roles.values():
                out.setdefault(r.category, []).append(r.brief())
        return out

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        kw = (keyword or "").strip().lower()
        if not kw:
            return self.list_roles()
        hits = []
        with self._lock:
            for r in self.roles.values():
                hay = " ".join([
                    r.id, r.name, r.type, r.persona, r.expertise,
                    " ".join(r.tags), " ".join(r.goals),
                ]).lower()
                if kw in hay:
                    hits.append(r.brief())
        return hits

    # ---------- 语法糖 ----------
    def __len__(self): return len(self.roles)
    def __iter__(self): return iter(self.roles.values())
    def __repr__(self): return f"<RoleManager dir={self.role_dir} count={len(self.roles)}>"


# ---------- 供外部直接用（可选） ----------
_default_manager: Optional[RoleManager] = None
def get_manager() -> RoleManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = RoleManager()
    return _default_manager