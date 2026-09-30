"""版本过滤：同一份文件的多个版本里，只让**现行有效**的那一版进候选。

为什么（真机发现，见 ``handoff/BACKLOG.md`` 附录 P7）：``knowledge/lawyer`` 里有
**411 个** ``X__<hash>.md`` 与基名文件**没有一对字节相同** —— 是 crawler 从
flk.npc.gov.cn 抓到的**不同版本**（基名 = 2020-12-29/时效性 3；副本 = 2006-11-23/时效性 2）。
旧版留在库里就会被召回，模型可能引一条**已被修改/已废止**的条文 —— 对法律 RAG 是高危。

分工：
* ``scripts/audit_corpus_versions.py`` 负责**盘点**（只读，按 时效性 → 公布日期 排序，
  给每族一个"建议保留"，产出 ``index/version_inventory.json``）；
* 本模块负责**检索侧执行**：读那份盘点，把非保留版本从候选里剔掉。

三条硬性口径：
1. **不删任何语料文件** —— 过滤只发生在检索出口，随时可关、可回退；
2. **fail-open**：盘点文件缺失/损坏/为空 ⇒ 一律不过滤（宁可不生效，也不误伤召回），
   并在日志里说清"没生效 + 怎么生成"（按项目规则，降级必须看得见）；
3. **只按 basename 比对**（语料里存的就是 basename；路径差异不该影响判定）。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: 盘点文件的默认名（与 ``scripts/audit_corpus_versions.py --out`` 一致）
DEFAULT_INVENTORY_NAME = "version_inventory.json"
#: 生成盘点的命令（缺失时原样打给运维，别让人猜）
REBUILD_HINT = ("python scripts/audit_corpus_versions.py "
                "--corpus knowledge/lawyer --out index/version_inventory.json")


@dataclass
class VersionFilter:
    """非现行版本的黑名单（+ 每个文件名族的现行版，便于日志/自检）。

    ``pending`` 是**尚未生效**的新版（如 2026-06-26 公布、2027-01-01 施行的《商标法》）。
    用户 2026-09-23 拍板的口径：**回答现行问题用现行有效版**；涉及尚未施行的新规要**提示**。
    所以 pending **不进黑名单**（不能因为"不是现行版"就把新规藏起来），但要能被识别出来
    以便在答案里加一句"这是尚未施行的新版"。
    """

    stale: frozenset[str] = frozenset()
    keep: dict[str, str] = field(default_factory=dict)
    pending: frozenset[str] = frozenset()
    #: 未生效版的**施行日期**（basename -> ``YYYY-MM-DD``）：给选择器/提示词用，
    #: 让"这条是尚未施行的新版"能带上具体日期（用户口径：涉新规要提示）。
    pending_dates: dict[str, str] = field(default_factory=dict)

    # ---------- 构造 ----------
    @classmethod
    def from_inventory(cls, payload: dict[str, Any]) -> "VersionFilter":
        """按盘点结果构造：每族取排好序的第一项为保留，其余进黑名单。

        兼容三种形状：
        * ``families``（脚本产物，每族是**已排序**的行列表；行里带 ``role`` 时按角色分流，
          ``pending`` 进"未生效"集合而**不进黑名单**）；
        * ``keep_map`` + ``pending_map``（新版脚本直接给出的两张小表，最明确）；
        * ``stale_files``（手写的简单黑名单）。
        """
        keep: dict[str, str] = {}
        stale: set[str] = set()
        pending: set[str] = set()
        pending_dates: dict[str, str] = {}

        # 形状 A：显式两表（新版脚本产物）
        for family, name in dict(payload.get("keep_map") or {}).items():
            if name:
                keep[str(family)] = str(name)
        for name in (payload.get("pending_map") or {}).values():
            if name:
                pending.add(Path(str(name)).name)

        families = payload.get("families") or {}
        if isinstance(families, dict):
            for family, rows in families.items():
                if not isinstance(rows, list) or not rows:
                    continue
                first = rows[0]
                file = str((first or {}).get("file") or "")
                if not file:
                    continue
                keep.setdefault(str(family), file)
                for row in rows[1:]:
                    name = str((row or {}).get("file") or "")
                    if not name or name == file:
                        continue
                    if str((row or {}).get("role") or "") == "pending":
                        pending.add(Path(name).name)     # 未生效：留着，但会提示
                        date = str((row or {}).get("effective_date")
                                   or (row or {}).get("effective") or "").strip()
                        if date:
                            pending_dates[Path(name).name] = date
                        continue
                    stale.add(name)

        for name in payload.get("stale_files") or []:
            text = str(name or "").strip()
            if text:
                stale.add(Path(text).name)

        # 明确的 keep_map 里点名的现行版，绝不进黑名单（防止新旧盘点形状混用时自相矛盾）
        stale -= {Path(name).name for name in keep.values()}
        pending -= stale
        pending_dates = {name: date for name, date in pending_dates.items() if name in pending}
        return cls(stale=frozenset(stale), keep=keep, pending=frozenset(pending),
                   pending_dates=pending_dates)

    @classmethod
    def load(cls, path: str | Path) -> "VersionFilter | None":
        """读盘点文件；**任何**异常都返回 ``None``（调用方据此保持不过滤）。"""
        target = Path(path)
        if not target.is_file():
            logger.info("版本过滤未生效：盘点文件不存在（%s）。生成命令：%s",
                        target, REBUILD_HINT)
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("版本过滤未生效：盘点文件读不了（%s：%s: %s）。生成命令：%s",
                           target, type(exc).__name__, exc, REBUILD_HINT)
            return None
        if not isinstance(payload, dict):
            logger.warning("版本过滤未生效：盘点文件不是对象（%s）", target)
            return None

        result = cls.from_inventory(payload)
        if not result.stale and not result.pending:
            # 既没有"非现行版本"也没有"尚未生效版" ⇒ 没什么可做的（保持 fail-open）
            logger.info("版本过滤未生效：盘点里既没有『非现行版本』也没有『尚未生效版』（%s）", target)
            return None
        logger.info("版本过滤已就绪：%d 个非现行版本将被剔除；另有 %d 个**尚未生效**的新版"
                    "（保留但会在答案里提示）（来自 %s，覆盖 %d 个文件名族）",
                    len(result.stale), len(result.pending), target, len(result.keep))
        return result

    @classmethod
    def load_or_none(cls, index_dir: str | Path) -> "VersionFilter | None":
        return cls.load(Path(index_dir) / DEFAULT_INVENTORY_NAME)

    # ---------- 查询 ----------
    def is_stale(self, source: str) -> bool:
        """按 basename 判：未知文件一律**不**算旧版（fail-open）。"""
        name = Path(str(source or "").replace("\\", "/")).name
        return bool(name) and name in self.stale

    def is_pending(self, source: str) -> bool:
        """这一条来自**尚未生效**的新版？"""
        name = Path(str(source or "").replace("\\", "/")).name
        return bool(name) and name in self.pending

    def pending_sources(self, hits: Iterable[Any]) -> list[str]:
        """命中里来自"尚未生效"版本的来源（去重、保序）—— 供上层加**可见提示**。"""
        found: list[str] = []
        for hit in hits:
            source = str(getattr(getattr(hit, "chunk", None), "source", "") or "")
            if self.is_pending(source) and source not in found:
                found.append(source)
        return found

    def filter_hits(self, hits: Iterable[Any]) -> tuple[list[Any], int]:
        """剔掉旧版命中，返回 ``(保留的命中, 剔掉条数)``；顺序不变。"""
        kept: list[Any] = []
        dropped = 0
        for hit in hits:
            source = getattr(getattr(hit, "chunk", None), "source", "")
            if self.is_stale(source):
                dropped += 1
                continue
            kept.append(hit)
        return kept, dropped

    def __len__(self) -> int:
        return len(self.stale)
