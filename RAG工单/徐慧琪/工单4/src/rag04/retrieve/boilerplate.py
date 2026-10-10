# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""跨页重复样板（页眉/页脚/表格表头）判据 —— 免重建（RC1）。

判据（可量化、可测试、可解释）：
  1) 归一化：删除文本中的全部空白（含全角空格）；
  2) 若同一归一化文本出现在 >= ``min_pages`` 个不同 (doc_id, page) 上，
     判为跨页样板；
  3) 长度护栏 ``max_chars``：超过该长度的文本不判 —— 正文长段即便复现
     也是实义内容，不得当版式样板删除。

``min_pages=20`` 的选取依据（实测现有索引 27,271 块，证据见
``_scratch/rc10_boilerplate_c20.out.txt``）：复现页数分布为「……11、12、13 页
各 1~7 种，14~19 页为空，20 页起再现」；复现 <=13 页的都是正文（其中含
id957/id793 的答案句），>=20 页的 9 种文本全部是页眉/页脚/表头。K=20 落在
实测分布的自然间隙内：判出 1,310 块（4.80%），16 题中含答案要点的块零误判。

缓存：判据只依赖现有索引 payload（不改 chunker/loader 产物），扫描一次后落盘
``data/boilerplate.json``；查询期只做 set 成员判断，不重复扫描 27k 块。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger("rag04.boilerplate")

DEFAULT_MIN_PAGES = 20
DEFAULT_MAX_CHARS = 120
CACHE_VERSION = 1

_WS = re.compile(r"[\s　]+")


def normalize_text(text: str) -> str:
    """归一化：删除全部空白（含全角空格）。页眉块的空白填充不影响判据。"""
    return _WS.sub("", text or "")


@dataclass(frozen=True)
class BoilerplateInfo:
    """样板判据结果。ids 为被判为样板的 chunk_id 集合。"""

    ids: frozenset[str]
    min_pages: int = DEFAULT_MIN_PAGES
    max_chars: int = DEFAULT_MAX_CHARS
    n_scanned: int = 0
    n_flagged: int = 0
    n_groups: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CACHE_VERSION,
            "min_pages": self.min_pages,
            "max_chars": self.max_chars,
            "n_scanned": self.n_scanned,
            "n_flagged": self.n_flagged,
            "n_groups": self.n_groups,
            "counts": dict(self.counts),
            "ids": sorted(self.ids),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BoilerplateInfo":
        return cls(
            ids=frozenset(d.get("ids", ())),
            min_pages=int(d["min_pages"]),
            max_chars=int(d["max_chars"]),
            n_scanned=int(d.get("n_scanned", 0)),
            n_flagged=int(d.get("n_flagged", 0)),
            n_groups=int(d.get("n_groups", 0)),
            counts=dict(d.get("counts", {})),
        )

    def is_boilerplate(self, chunk_id: str) -> bool:
        return chunk_id in self.ids


def flagged_texts(
    blocks: Iterable[Any],
    min_pages: int = DEFAULT_MIN_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> set[str]:
    """判据核心：返回被判为跨页样板的**归一化文本**集合。

    只需 block 具备 doc_id/page/text 属性（Chunk 与 TextBlock 皆可），
    因此同一判据可同时服务检索侧（chunk）与入库侧（TextBlock，RC1-in）。
    """
    if min_pages < 2:
        raise ValueError(f"min_pages 必须 >= 2（单页不构成跨页样板），收到 {min_pages}")

    pages_of: dict[str, set[tuple[str, int]]] = {}
    norm_len: dict[str, int] = {}
    for c in blocks:
        key = normalize_text(getattr(c, "text", "") or "")
        if len(key) < 2:
            continue
        pid = (getattr(c, "doc_id", ""), int(getattr(c, "page", 0)))
        pages_of.setdefault(key, set()).add(pid)
        norm_len.setdefault(key, len(key))

    return {key for key, pages in pages_of.items()
            if len(pages) >= min_pages and norm_len[key] <= max_chars}


def find_boilerplate(
    chunks: Iterable[Any],
    min_pages: int = DEFAULT_MIN_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
    counts: dict[str, int] | None = None,
) -> BoilerplateInfo:
    """从块序列推导样板集合。``chunks`` 只需具备 chunk_id/doc_id/page/text 属性。"""
    blocks = list(chunks)
    flagged = flagged_texts(blocks, min_pages=min_pages, max_chars=max_chars)
    ids = frozenset(
        str(getattr(c, "chunk_id", "")) for c in blocks
        if getattr(c, "chunk_id", "")
        and normalize_text(getattr(c, "text", "") or "") in flagged
    )
    return BoilerplateInfo(
        ids=ids, min_pages=min_pages, max_chars=max_chars,
        n_scanned=len(blocks), n_flagged=len(ids), n_groups=len(flagged),
        counts=dict(counts or {}),
    )


def filter_boilerplate_blocks(
    blocks: Iterable[Any],
    min_pages: int = DEFAULT_MIN_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
    enabled: bool = True,
) -> tuple[list[Any], int]:
    """RC1-in：入库侧按同一判据剔除页眉/页脚/表头文本块。

    重建阶段在分块**之前**生效：这些块根本不会进入 text_chunks / BM25，
    从根上消除它们对词频/IDF 与向量库容量的污染（第一阶段只能在检索侧
    过滤已入库的 1310 块）。返回 (保留块, 剔除数)；``enabled=False`` 时
    原样返回（baseline_03 对照组）。
    """
    items = list(blocks)
    if not enabled:
        return items, 0
    flagged = flagged_texts(items, min_pages=min_pages, max_chars=max_chars)
    if not flagged:
        return items, 0
    kept = [b for b in items
            if normalize_text(getattr(b, "text", "") or "") not in flagged]
    return kept, len(items) - len(kept)


def collect_chunks(store) -> list[Any]:
    """从现有索引读取三库 payload 转成块对象（不重建索引，只读）。

    文本/表格块与 BM25 的来源一致；图像块只在 image_chunks，其 payload 文本
    （CLIP 路的「文本侧」）同样纳入判据。
    """
    from rag04.ingest.store import COLL_IMAGE, COLL_TABLE, COLL_TEXT
    from rag04.schema import Chunk

    out: list[Any] = []
    for coll in (COLL_TEXT, COLL_TABLE, COLL_IMAGE):
        recs, _ = store.client.scroll(coll, limit=100_000, with_payload=True)
        for r in recs:
            pl = r.payload or {}
            out.append(Chunk(
                chunk_id=pl.get("chunk_id", str(r.id)),
                doc_id=pl.get("doc_id", ""),
                page=int(pl.get("page", 0)),
                block_type=pl.get("block_type", "text"),
                source_id=pl.get("source_id", ""),
                text=pl.get("text", ""),
            ))
    return out


def load_or_build(
    path: Path,
    chunks: Sequence[Any] | Callable[[], Sequence[Any]] | None = None,
    min_pages: int = DEFAULT_MIN_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
    counts: dict[str, int] | None = None,
) -> BoilerplateInfo:
    """读缓存；缺失/参数或语料规模不符时重建并落盘。

    ``chunks`` 也可以是零参可调用对象——只在确定需要重建时才扫描索引，
    缓存命中路径不付任何扫描成本。
    ``counts`` 为 collection→点数快照：与缓存记录不一致说明索引已变，强制重建，
    避免拿着过期样板集合过滤新库。
    """
    path = Path(path)
    if path.exists():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            info = BoilerplateInfo.from_dict(d)
            params_ok = (d.get("version") == CACHE_VERSION
                         and info.min_pages == min_pages
                         and info.max_chars == max_chars)
            counts_ok = counts is None or info.counts == dict(counts)
            if params_ok and counts_ok:
                logger.info("样板缓存命中：%d 块（判据 K>=%d, len<=%d）",
                            info.n_flagged, min_pages, max_chars)
                return info
            logger.info("样板缓存与当前索引/参数不符，重建：%s", path)
        except Exception as e:                       # 缓存损坏不阻断主链路
            logger.warning("样板缓存读取失败（%s），将重建：%s", e, path)

    if chunks is None:
        raise ValueError("样板缓存缺失且未提供 chunks，无法推导判据")
    if callable(chunks):
        chunks = chunks()
    info = find_boilerplate(chunks, min_pages=min_pages, max_chars=max_chars,
                            counts=counts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info.to_dict(), ensure_ascii=False),
                    encoding="utf-8")
    logger.info("样板判据已建立并缓存：%d/%d 块（%.2f%%），%d 种文本 → %s",
                info.n_flagged, info.n_scanned,
                100 * info.n_flagged / max(info.n_scanned, 1),
                info.n_groups, path)
    return info
