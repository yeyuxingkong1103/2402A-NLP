# -*- coding: utf-8 -*-
"""检索侧「最新代」过滤（t52）：**每个文件只返回最新一代的块**。

## 为什么需要它（不依赖 Milvus 的删除语义）

t40 attempt 4 实测（`handoff/t40-CONVERGENCE-r4.md`）：对 15,116 个旧代子块发起
Milvus 删除，服务端**逐批报删 15,116 / 主键路径 `get()` 全部取不到**，但
`count(*)`/`scan_rows()` 仍有 **13,760** 行可见、且**旧代块用原向量照样被召回**
（生态环境法典旧代 `ca=1789646114` 以 score=0.9031 进入 top-3）。对同一批 id
**再删一次也不改变可见性** ⇒ 删除对 query/search 路径不生效，**这不是最终一致性**。

所以本模块把「只保留最新一代」这件事**移到检索侧**：无论 Milvus 里还留着多少旧行，
**进入召回与引用的一律只有最新一代**。

## 判定规则（确定性、可复算）

* **归组键**：默认 `source`（配置 `RAG_GENERATION_FILTER_KEY`，可选 `source` / `doc_id`）。
  一个文件在库里同时有 `source` 与 `doc_id`，二者对本库是 1:1（见 t40 盘点），
  用 `source` 是因为它就是检索结果里直接可读到的字段。
* **最新代**：同一归组键下，`created_at` 的**最大值**（严格数值比较，`float`）。
  同一代内父/子块共用同一个 `created_at`（`legal_rag/ingest/chunker.py`：每个
  document 取一次 `now = now_ts()`），所以「最新代」= 「该文件最后一趟入库写的那批块」。
* **并列**：同一归组键下 `created_at` 并列的，全部算最新代（不丢数据）。
* **边界：`created_at == 0`**（当前全库 0 行，仍需写明）——0 是一个**普通的代值**：
  某文件的**全部**行都是 0 时，那一代就是最新代（全部保留）；某文件既有 0 又有 >0
  的行时，0 那一代是**更早代**（会被过滤）。**不做特判放行**，以免"未知代"变成后门。
* **未知归组键**（`source` 为空，或该文件不在索引里——例如刚上传、索引还没重建）：
  **默认放行**（`unknown_policy="allow"`），即"宁可多留，不可错杀好数据"；
  可配置成 `"drop"`。被放行的未知键会计数（`unknown_keys`），便于观测。
* **总开关**：**默认开启**（`RAG_GENERATION_FILTER`，见下）——用户要的正是"旧代碎片
  不再进入召回与引用"，所以**默认生效**；置 `RAG_GENERATION_FILTER=0` 可**一键回到
  改前行为**（召回范围逐字节不变）。索引文件缺失/不可读时：若
  `RAG_GENERATION_FILTER_REQUIRE_INDEX=1`，则**报错拒绝继续**（fail-closed，可选）；
  默认 `require_index=0` 时**跳过过滤 + 告警一次**（fail-open：绝不让"索引没建好"
  变成检索不可用或报错）。
* **索引陈旧也安全**：判定用 `created_at >= 该文件最新代`，所以**索引构建之后新入库的
  那一代照样通过**（不必每次入库都重建索引）。代价：若某文件的时钟回拨（新一代的
  `created_at` 反而更小），那一代会被当成旧代过滤掉 —— 处置 = 重建索引（`--rebuild`）。

## 用法（运维/工具）

```powershell
# 只读重建索引（全库扫一遍 created_at，写 index/latest_generations.json）
.venv\\Scripts\\python.exe -m legal_rag.retrieve.generation_filter --rebuild
# 只看现状（不写盘）
.venv\\Scripts\\python.exe -m legal_rag.retrieve.generation_filter --status
```

索引文件：`index/latest_generations.json`（默认路径；可用
`RAG_GENERATION_FILTER_PATH` 覆盖）。内含 `key_field` / `built_at` /
`rows_scanned` / `sources`（`{归组键: 最新代 created_at}`），便于复核与重放。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

#: 归组键字段（可配置）
DEFAULT_KEY_FIELD = "source"
#: 索引文件默认名（放在 config.index_dir 下）
DEFAULT_INDEX_NAME = "latest_generations.json"
#: 未知归组键的默认处置：放行（宁可多留，不可错杀）
DEFAULT_UNKNOWN_POLICY = "allow"


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on", "y")


def filter_enabled() -> bool:
    """总开关（**默认开启**；置 `RAG_GENERATION_FILTER=0` 回到改前行为）。"""
    return _env_bool("RAG_GENERATION_FILTER", True)


def key_field() -> str:
    value = _env("RAG_GENERATION_FILTER_KEY", DEFAULT_KEY_FIELD).lower()
    return value if value in ("source", "doc_id") else DEFAULT_KEY_FIELD


def unknown_policy() -> str:
    value = _env("RAG_GENERATION_FILTER_UNKNOWN", DEFAULT_UNKNOWN_POLICY).lower()
    return value if value in ("allow", "drop") else DEFAULT_UNKNOWN_POLICY


def require_index() -> bool:
    return _env_bool("RAG_GENERATION_FILTER_REQUIRE_INDEX", False)


def index_path(index_dir: str | Path | None = None) -> Path:
    override = _env("RAG_GENERATION_FILTER_PATH")
    if override:
        return Path(override)
    base = Path(index_dir) if index_dir else Path("index")
    return base / DEFAULT_INDEX_NAME


def generation_of(row: Any) -> float:
    """读一条记录的 ``created_at``（dict 或对象都支持；缺失/非法 → 0.0）。"""
    if isinstance(row, dict):
        raw = row.get("created_at", 0.0)
    else:
        raw = getattr(row, "created_at", 0.0)
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def source_of(row: Any, field: str = DEFAULT_KEY_FIELD) -> str:
    """读归组键（dict 或对象；缺失 → 空串，代表"未知键"）。"""
    if isinstance(row, dict):
        raw = row.get(field, "")
    else:
        raw = getattr(row, field, "")
    return str(raw or "")


def build_latest_map(rows: Iterable[Any], field: str = DEFAULT_KEY_FIELD) -> dict[str, float]:
    """从行集合算出「归组键 → 最新代 created_at」（纯计算，不碰存储）。"""
    latest: dict[str, float] = {}
    for row in rows:
        key = source_of(row, field)
        if not key:
            continue
        created = generation_of(row)
        current = latest.get(key)
        if current is None or created > current:
            latest[key] = created
    return latest


class LatestGenerationIndex:
    """「每个文件只保留最新一代」的判定器（纯内存 + 一个 JSON 文件）。

    * :meth:`from_rows` —— 由已扫到的行直接构造（工具/测试用）；
    * :meth:`load` —— 从 `index/latest_generations.json` 读；
    * :meth:`check` —— 判定一条记录是否来自最新代。
    """

    def __init__(self, latest: dict[str, float], *, field: str = DEFAULT_KEY_FIELD,
                 built_at: str = "", rows_scanned: int = 0,
                 unknown_policy_: str = DEFAULT_UNKNOWN_POLICY) -> None:
        self.latest = {str(k): float(v) for k, v in (latest or {}).items()}
        self.field = field
        self.built_at = built_at
        self.rows_scanned = int(rows_scanned or 0)
        self.unknown_policy = unknown_policy_
        #: 统计（便于观测"到底拦掉了多少"）
        self.kept = 0
        self.dropped = 0
        self.unknown_keys_seen: set[str] = set()

    # ---------- 构造 ----------
    @classmethod
    def from_rows(cls, rows: Iterable[Any], *, field: str = DEFAULT_KEY_FIELD,
                  unknown_policy_: str = DEFAULT_UNKNOWN_POLICY) -> "LatestGenerationIndex":
        rows = list(rows)
        return cls(build_latest_map(rows, field), field=field,
                   built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   rows_scanned=len(rows), unknown_policy_=unknown_policy_)

    @classmethod
    def load(cls, path: str | Path | None = None, *,
             index_dir: str | Path | None = None) -> "LatestGenerationIndex | None":
        target = Path(path) if path else index_path(index_dir)
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[GEN-FILTER] 索引文件不可读（%s）：%s —— 本次不做代数过滤",
                           target, exc)
            return None
        sources = payload.get("sources")
        if not isinstance(sources, dict):
            logger.warning("[GEN-FILTER] 索引文件缺 sources 字段：%s —— 本次不做代数过滤", target)
            return None
        return cls(sources, field=str(payload.get("key_field") or DEFAULT_KEY_FIELD),
                   built_at=str(payload.get("built_at") or ""),
                   rows_scanned=int(payload.get("rows_scanned") or 0),
                   unknown_policy_=unknown_policy())

    def dump(self, path: str | Path | None = None, *,
             index_dir: str | Path | None = None) -> Path:
        target = Path(path) if path else index_path(index_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "key_field": self.field,
            "built_at": self.built_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows_scanned": self.rows_scanned,
            "distinct_generations": len({v for v in self.latest.values()}),
            "sources": {k: v for k, v in sorted(self.latest.items())},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    # ---------- 判定 ----------
    @property
    def size(self) -> int:
        return len(self.latest)

    def latest_of(self, key: str) -> float | None:
        return self.latest.get(str(key))

    def is_latest(self, row: Any) -> bool:
        """一条记录是否属于其文件的最新代。"""
        key = source_of(row, self.field)
        if not key or key not in self.latest:
            self.unknown_keys_seen.add(key or "<空键>")
            return self.unknown_policy == "allow"
        if generation_of(row) >= self.latest[key]:
            return True
        return False

    def filter_rows(self, rows: Sequence[Any]) -> list[Any]:
        """按序过滤（保持输入顺序）。"""
        return [row for row in rows if self.is_latest(row)]

    def split(self, rows: Sequence[Any]) -> tuple[list[Any], list[Any]]:
        """返回 ``(keep, drop)`` 两个列表（保持各自顺序），便于取证/对照。"""
        keep: list[Any] = []
        drop: list[Any] = []
        for row in rows:
            (keep if self.is_latest(row) else drop).append(row)
        return keep, drop

    def reset_stats(self) -> None:
        self.kept = 0
        self.dropped = 0
        self.unknown_keys_seen = set()

    def stats(self) -> dict:
        return {"index_size": self.size, "built_at": self.built_at,
                "rows_scanned": self.rows_scanned, "key_field": self.field,
                "unknown_policy": self.unknown_policy,
                "kept": self.kept, "dropped": self.dropped,
                "unknown_keys": len(self.unknown_keys_seen)}


def build_from_store(store: Any, *, field: str = DEFAULT_KEY_FIELD,
                     max_rows: int = 0) -> LatestGenerationIndex:
    """**只读**扫一遍 store，构造最新代索引（只取标量字段，不取 text/vector）。"""
    scanner = getattr(store, "scan_rows", None)
    if not callable(scanner):
        raise TypeError("store 不支持 scan_rows()（只读盘点接口），无法重建代数索引")
    rows = scanner(fields=("id", field, "created_at", "is_parent", "role_id"),
                   max_rows=max_rows)
    logger.info("[GEN-FILTER] 只读扫描完成：%d 行 → 归组键 %d 个", len(rows), 0)
    index = LatestGenerationIndex.from_rows(rows, field=field)
    logger.info("[GEN-FILTER] 最新代索引：%d 个归组键（扫了 %d 行）",
                index.size, index.rows_scanned)
    return index


# ---------- 工具入口（只读 / 写索引文件，绝不碰 Milvus 数据） ----------
def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索侧「最新代」索引（只读 Milvus）")
    parser.add_argument("--rebuild", action="store_true",
                        help="只读扫描全库并写 index/latest_generations.json")
    parser.add_argument("--status", action="store_true", help="只看现状，不写盘")
    parser.add_argument("--key", default="", help="归组键字段（默认 source）")
    parser.add_argument("--path", default="", help="索引文件路径（默认 index/latest_generations.json）")
    parser.add_argument("--max-rows", type=int, default=0, help="只扫前 N 行（调试用）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    field = (args.key or key_field()).lower()
    target = Path(args.path) if args.path else index_path()

    if args.status or not args.rebuild:
        loaded = LatestGenerationIndex.load(target)
        if loaded is None:
            print(f"[status] 索引文件不存在：{target}")
            return 1
        print(f"[status] path={target}")
        print(f"[status] key_field={loaded.field} built_at={loaded.built_at} "
              f"rows_scanned={loaded.rows_scanned} sources={loaded.size}")
        return 0

    from legal_rag.config import RagConfig
    from legal_rag.store.milvus_store import MilvusVectorStore
    # 必须 from_env()：命令行工具若不读环境变量，Milvus Lite 的 URI
    # （LEGAL_RAG_MILVUS_URI）会被丢掉，转而连默认 127.0.0.1:19530 并失败
    # —— 真机踩坑（云端没有独立 Milvus 服务，只有 Lite 文件库）。
    cfg = RagConfig.from_env()
    manifest = Path(cfg.index_dir) / "milvus_manifest.json"
    collection, dim = "", 0
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        collection, dim = str(payload.get("collection") or ""), int(payload.get("dim") or 0)
    store = MilvusVectorStore(cfg, dim=dim, collection=collection,
                              manifest_dir=cfg.index_dir, recreate_on_dim_mismatch=False)
    store.connect(create=False)
    before = store.count()
    index = build_from_store(store, field=field, max_rows=args.max_rows)
    after = store.count()
    written = index.dump(target)
    print(f"[rebuild] count(*) 前 {before} → 后 {after}（必须相等：本命令零写数据）")
    print(f"[rebuild] 归组键 {index.size} 个 / 扫描 {index.rows_scanned} 行")
    print(f"[rebuild] 已写 {written}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
