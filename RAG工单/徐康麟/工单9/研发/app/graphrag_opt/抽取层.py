# -*- coding: utf-8 -*-
"""工单9 优化 · **实体/关系构建层**：别名归并 + 关系方向与类型规范化 + 噪声剪枝 + 置信度过滤。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

输入 = 工单8 已产出的 ``研发/data/graph/{entities,relations}.jsonl``（**只读**）；
输出 = ``研发/data/图谱优化后/{entities,relations}.jsonl + 图谱优化统计.json``。

------------------------------------------------------------
现状与问题（**逐条都是工单8 产物上的实测现象，不是设想**）
------------------------------------------------------------
在 工单8 的图谱（9817 节点 / 4861 边）上实测到四类质量问题：

1. **同一实体分裂**（别名未归并）
   * ``何肖``(count=1) 与 ``何肖锋``(count≥1) 同时存在 —— PDF 抽取在换行处截断姓名；
   * 公司全称/简称并存：``平安银行`` 与 ``平安银行股份有限公司``；
   * 数值型实体写法不一：``2020年度`` / ``2020 年`` / ``2020年``。
   后果：实体链接时两条都是弱节点、任何一条单独都不足以锁定语料 ⇒ 图谱通道召回不出来。

2. **关系方向/类型不规范**
   * ``安邦保险集团股份有限公司 --任职--> 何肖`` —— 方向反了（schema 规定 ``任职`` 是 person→company）；
   * 同义关系名未归并时会出现 ``持有``/``控股``/``持股`` 三名并存。

3. **通用刻度噪声占据枢纽位置**
   * ``2020年度`` count=593、``2021年12月31日`` count=528 —— 这类日期/比例是
     财务表格的**通用刻度**，几乎每页都有。它们度极大但零信息量，
     两跳扩展会把整张图拉进来（工单8 的可视化图中心就是这些节点）。

4. **置信度与提及数未设有效下限**
   * 工单8 的 ``min_confidence=0.55`` 会放行 ``person_bare``(0.55) 这类弱抽取；
   * 每实体最多 6 个提及块 ⇒ 图谱通道单次扩展要扫更多块，检索耗时上升。

------------------------------------------------------------
优化方案（对应本模块的四个函数）
------------------------------------------------------------
* :func:`merge_aliases`    —— 并查集归并：公司后缀剥离、人名前缀补全、数值写法归一；保留提及最多的写法为规范名；
* :func:`normalize_relations` —— 关系名归并到 schema 规范名 + head/tail 类型校验与**自动反转** + 自环剪除 + 去重；
* :func:`prune_noise`      —— 通用刻度剪枝（低值整数比例 / 超高度数日期）+ 度数上限 + 置信度过滤；
* :func:`refine_graph`     —— 主入口，流式读写（边跑边 append），单进程，无进程池。

资源纪律：全程**单进程、流式**，实体表与关系表都按行处理；
9817 节点 / 4861 边的内存占用实测 < 100 MB（远低于 2 GB 上限）。
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..graphrag.schema import RELATION_TYPES, entity_type_zh, is_valid_relation, relation_canonical

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

#: 低值整数比例（0% / 1% / 5% / 10% / 1倍）—— 财务表格的通用刻度
_TRIVIAL_RATIO = re.compile(r"^(\d+(?:\.\d+)?)\s*(%|个百分点|倍)$")
#: 日期写法：2020年度 / 2020 年 / 2019年12月31日 / 2021-12-31
_DATE_LIKE = re.compile(r"^(?:19|20)\d{2}\s*年(?:度|末|初|底)?(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日?)?)?$")


def norm_name(name: Any) -> str:
    """实体名归一化（NFKC + 去空白 + 全角括号统一）。

    ⚠️ NFKC 不是可选项：``sample_questions.pdf`` 文本层用 Kangxi 部首码位，
    不归一化会出现「看起来一样、码位不同」的两套实体（模块 docstring §1 的现象之一）。
    """
    text = unicodedata.normalize("NFKC", str(name or ""))
    text = re.sub(r"[\s\u3000]+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    return text.strip()


def _org_core(name: str, suffixes: Iterable[str]) -> str:
    """公司/机构名 → 核心名（剥离「股份有限公司」这类后缀，用于全称/简称归并）。"""
    text = norm_name(name)
    for suffix in sorted(suffixes, key=len, reverse=True):
        if text.endswith(suffix) and len(text) - len(suffix) >= 2:
            return text[: -len(suffix)]
    return text


def _numeric_key(name: str) -> str:
    """数值型实体名的归一化键（``2020 年度`` → ``2020年``；``5 ％`` → ``5%``）。"""
    text = norm_name(name)
    text = text.replace("年度", "年").replace("年末", "年末").replace("%", "%")
    text = re.sub(r"(?<=\d)\s+(?=年|月|日|%|个百分点|倍|元|亿元|万元)", "", text)
    return text


class _UnionFind:
    """并查集（归并别名用；路径压缩 + 按秩合并）。"""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        self.rank.setdefault(key, 0)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:          # 路径压缩
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key in list(self.parent):
            out.setdefault(self.find(key), []).append(key)
        return out


# ---------------------------------------------------------------------------
# 读 / 写
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """流式读 jsonl（逐行 yield，不把整个文件读进内存）。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue                          # 坏行跳过（不因一行坏数据整体失败）


# ---------------------------------------------------------------------------
# ① 别名归并
# ---------------------------------------------------------------------------
def merge_aliases(entities: list[dict[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """实体别名归并：返回 ``{"alias_of": {原名: 规范名}, "groups": [...], "stats": {...}}``。

    三条归并规则（**互不冲突，按优先级依次施加**）：
        R1 公司/机构后缀剥离：``平安银行股份有限公司`` ≡ ``平安银行``；
        R2 人名前缀补全：``何肖`` → ``何肖锋``（长度差 ≤ ``person_prefix_len_diff``、被并入者提及少）；
        R3 数值写法归一：``2020 年度`` ≡ ``2020年`` ≡ ``2020年度``。
    """
    suffixes = list(cfg.get("org_suffixes") or [])
    min_conf = float(cfg.get("min_confidence") or 0.0)
    uf = _UnionFind()
    by_name: dict[str, dict[str, Any]] = {}

    for row in entities:
        name = norm_name(row.get("name") or row.get("id"))
        if not name:
            continue
        row["name"] = name
        row["id"] = norm_name(row.get("id") or name) or name
        by_name.setdefault(name, row)
        uf.find(name)

    buckets: dict[tuple[str, str], list[str]] = {}
    for name, row in by_name.items():
        etype = str(row.get("type") or "unknown")
        if etype in ("company", "institution"):
            key = ("org", _org_core(name, suffixes))
        elif etype == "person":
            key = ("person", name)
        elif etype in ("date", "ratio", "amount"):
            key = (etype, _numeric_key(name))
        else:
            key = (etype, name)
        buckets.setdefault(key, []).append(name)

    stats = {"R1_公司后缀归并组": 0, "R2_人名前缀归并组": 0, "R3_数值写法归并组": 0}
    # R1 / R3：同桶即同实体
    for (kind, _key), names in buckets.items():
        if len(names) < 2:
            continue
        base = names[0]
        for other in names[1:]:
            uf.union(base, other)
        if kind == "org":
            stats["R1_公司后缀归并组"] += 1
        elif kind in ("date", "ratio", "amount"):
            stats["R3_数值写法归并组"] += 1

    # R2：人名前缀补全（同姓氏前缀、长度差 ≤ diff）
    # ⚠️ 实测踩坑（由 pytest 用例 `test_别名归并_公司后缀与人名前缀` 抓出）：
    #   第一版把 persons 按 count 降序后，用 `lname.startswith(sname)` 判断前缀 ——
    #   但排序后 index 小的往往是**更长的名字**，`何肖`.startswith(`何肖锋`) 恒为 False，
    #   规则**永远不会触发**（实测 `何肖` 未被并入 `何肖锋`）。
    #   正确做法：先按长度定出 (短, 长)，再判前缀。
    persons = [row for row in by_name.values() if str(row.get("type")) == "person"]
    persons.sort(key=lambda r: (-int(r.get("count") or 0), -len(str(r.get("name")))))
    diff = int(cfg.get("person_prefix_len_diff") or 1)
    ratio = float(cfg.get("person_prefix_count_ratio") or 0.3)
    for idx, first in enumerate(persons):
        fname = str(first.get("name"))
        for second in persons[idx + 1:]:
            sname = str(second.get("name"))
            if fname == sname:
                continue
            shorter, longer = (fname, sname) if len(fname) < len(sname) else (sname, fname)
            if not (0 < len(longer) - len(shorter) <= diff):
                continue
            if not longer.startswith(shorter):
                continue
            short_row = first if shorter == fname else second
            long_row = second if shorter == fname else first
            if int(long_row.get("count") or 0) >= int(short_row.get("count") or 0) * ratio:
                uf.union(longer, shorter)             # 统一并入更完整的写法
                stats["R2_人名前缀归并组"] += 1
                break

    groups = {root: members for root, members in uf.groups().items() if len(members) > 1}
    alias_of: dict[str, str] = {}
    group_rows: list[dict[str, Any]] = []
    for members in groups.values():
        # 规范名 = 提及数最多；平手时取更长（信息更全）的名字
        canonical = max(members, key=lambda n: (int(by_name.get(n, {}).get("count") or 0), len(n)))
        aliases = sorted(m for m in members if m != canonical)
        for member in members:
            alias_of[member] = canonical
        group_rows.append({"canonical": canonical, "aliases": aliases,
                           "canonical_count": int(by_name.get(canonical, {}).get("count") or 0),
                           "type": str(by_name.get(canonical, {}).get("type") or "")})
    group_rows.sort(key=lambda g: (-len(g["aliases"]), g["canonical"]))
    stats.update({"归并组数": len(group_rows), "被并入实体数": len(alias_of)})
    return {"alias_of": alias_of, "groups": group_rows, "stats": stats,
            "by_name": by_name, "entities": entities}


def apply_alias(alias_of: Mapping[str, str], name: Any) -> str:
    """把任意名字映射到规范名（不在表内则原样返回归一化名）。"""
    key = norm_name(name)
    seen = set()
    while key in alias_of and key not in seen:
        seen.add(key)
        key = alias_of[key]
    return key


def merged_entities(entities: list[dict[str, Any]], alias_of: Mapping[str, str], *,
                    mentions_per_entity: int, min_confidence: float) -> list[dict[str, Any]]:
    """按别名表合并实体（提及块去重、置信度取最大、count 求和）。"""
    merged: dict[str, dict[str, Any]] = {}
    for row in entities:
        name = norm_name(row.get("name") or row.get("id"))
        canonical = apply_alias(alias_of, name)
        target = merged.get(canonical)
        if target is None:
            target = {"id": canonical, "name": canonical, "type": str(row.get("type") or "unknown"),
                      "type_zh": entity_type_zh(str(row.get("type") or "unknown")),
                      "confidence": 0.0, "count": 0, "aliases": [], "chunks": []}
            merged[canonical] = target
        target["count"] = int(target["count"]) + int(row.get("count") or 0)
        target["confidence"] = max(float(target["confidence"]), float(row.get("confidence") or 0.0))
        if name != canonical and name not in target["aliases"]:
            target["aliases"].append(name)
        for alias in (row.get("aliases") or []):
            key = norm_name(alias)
            if key and key != canonical and key not in target["aliases"]:
                target["aliases"].append(key)
        for chunk in (row.get("chunks") or []):
            target["chunks"].append(chunk)

    out: list[dict[str, Any]] = []
    for node in merged.values():
        if float(node["confidence"]) < float(min_confidence):
            continue
        dedup: dict[str, dict[str, Any]] = {}
        for chunk in node["chunks"]:
            cid = str(chunk.get("chunk_id") or "")
            if not cid:
                continue
            old = dedup.get(cid)
            if old is None or float(chunk.get("confidence") or 0) > float(old.get("confidence") or 0):
                dedup[cid] = chunk
        chunks = sorted(dedup.values(), key=lambda c: (-float(c.get("confidence") or 0),
                                                       int(c.get("page") or 0)))
        node["chunks"] = chunks[: max(int(mentions_per_entity), 1)]
        node["mention_total"] = len(dedup)
        out.append(node)
    out.sort(key=lambda n: (-int(n["count"]), str(n["name"])))
    return out


# ---------------------------------------------------------------------------
# ② 关系方向与类型规范化
# ---------------------------------------------------------------------------
def _direction_ok(head_type: str, tail_type: str, relation: str) -> bool:
    spec = RELATION_TYPES.get(relation)
    if not spec:
        return False
    return head_type in set(spec.get("head") or []) and tail_type in set(spec.get("tail") or [])


def normalize_relations(relations: list[dict[str, Any]], *, node_type: Mapping[str, str],
                        alias_of: Mapping[str, str], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """关系规范化：类型归并 → 方向修正 → 自环/违规剪除 → 去重。"""
    canon_on = bool(cfg.get("relation_canonical", True))
    fix_dir = bool(cfg.get("relation_direction_fix", True))
    drop_invalid = bool(cfg.get("drop_invalid_relation", True))
    drop_loop = bool(cfg.get("drop_self_loop", True))
    dedupe = bool(cfg.get("dedupe_edges", True))
    min_edge_conf = float(cfg.get("min_edge_confidence") or 0.0)

    stats = {"输入边数": len(relations), "关系名归并数": 0, "方向反转数": 0, "方向无法判定保留数": 0,
             "自环剪除数": 0, "非法关系剪除数": 0, "低置信度剪除数": 0, "重复边合并数": 0,
             "规范关系覆盖率": 1.0}
    out_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    dropped_direction: list[dict[str, Any]] = []

    for row in relations:
        head = apply_alias(alias_of, row.get("head"))
        tail = apply_alias(alias_of, row.get("tail"))
        if not head or not tail:
            continue
        raw_rel = str(row.get("relation") or "")
        relation = relation_canonical(raw_rel) if canon_on else raw_rel
        if relation != raw_rel:
            stats["关系名归并数"] += 1
        if drop_invalid and not is_valid_relation(relation):
            stats["非法关系剪除数"] += 1
            continue
        if drop_loop and head == tail:
            stats["自环剪除数"] += 1
            continue
        if float(row.get("confidence") or 0.0) < min_edge_conf:
            stats["低置信度剪除数"] += 1
            continue

        htype = str(node_type.get(head) or row.get("head_type") or "")
        ttype = str(node_type.get(tail) or row.get("tail_type") or "")
        if fix_dir and not _direction_ok(htype, ttype, relation):
            if _direction_ok(ttype, htype, relation):
                head, tail, htype, ttype = tail, head, ttype, htype
                stats["方向反转数"] += 1
            else:
                stats["方向无法判定保留数"] += 1
                dropped_direction.append({"head": head, "relation": relation, "tail": tail,
                                          "head_type": htype, "tail_type": ttype})

        key = (head, relation, tail)
        edge = {
            "head": head, "relation": relation, "tail": tail,
            "head_type": htype, "tail_type": ttype,
            "confidence": float(row.get("confidence") or 0.0),
            "extractor": str(row.get("extractor") or "rule"),
            "evidence": str(row.get("evidence") or "")[:200],
            "chunk_id": str(row.get("chunk_id") or ""), "page": int(row.get("page") or 0),
            "file_name": str(row.get("file_name") or ""), "support": 1,
        }
        old = out_map.get(key)
        if old is None:
            out_map[key] = edge
            continue
        if dedupe:
            stats["重复边合并数"] += 1
            old["support"] = int(old.get("support") or 1) + 1
            if edge["confidence"] > old["confidence"]:
                old.update({"confidence": edge["confidence"], "evidence": edge["evidence"],
                            "chunk_id": edge["chunk_id"], "page": edge["page"],
                            "file_name": edge["file_name"], "extractor": edge["extractor"]})
        else:
            stats["重复边合并数"] += 1

    edges = list(out_map.values())
    valid = sum(1 for e in edges if is_valid_relation(str(e["relation"])))
    stats["规范关系覆盖率"] = round(valid / len(edges), 4) if edges else 1.0
    stats["输出边数"] = len(edges)
    return {"edges": edges, "stats": stats, "direction_undecided": dropped_direction[:50]}


# ---------------------------------------------------------------------------
# ③ 噪声剪枝
# ---------------------------------------------------------------------------
def prune_noise(nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
                cfg: Mapping[str, Any], *, chunks_scanned: int) -> dict[str, Any]:
    """通用刻度剪枝 + 抽取碎片剪枝 + 度数上限 + 孤立节点清理。"""
    ratio_max = float(cfg.get("uninformative_ratio_max") or 10.0)
    prune_types = set(cfg.get("prune_types") or [])
    max_doc_ratio = float(cfg.get("max_doc_ratio") or 0.05)
    min_abs_count = int(cfg.get("min_absolute_count") or 200)
    max_degree = int(cfg.get("max_degree") or 400)
    min_len = int(cfg.get("min_entity_len") or 2)
    prune_frag = bool(cfg.get("prune_fragments", True))
    frag_max_len = int(cfg.get("fragment_max_len") or 16)
    frag_markers = list(cfg.get("fragment_markers") or [])
    frag_joiners = list(cfg.get("fragment_joiners") or [])
    frag_joiner_len = int(cfg.get("fragment_joiner_min_len") or 12)

    degree: dict[str, int] = {}
    for edge in edges:
        degree[str(edge["head"])] = degree.get(str(edge["head"]), 0) + 1
        degree[str(edge["tail"])] = degree.get(str(edge["tail"]), 0) + 1

    threshold = max(int(chunks_scanned * max_doc_ratio), 30)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    stats = {"通用刻度剪除数": 0, "抽取碎片剪除数": 0, "度数超限剪除数": 0,
             "过短实体剪除数": 0, "孤立节点剪除数": 0}

    for node in nodes:
        name = str(node.get("name"))
        etype = str(node.get("type"))
        count = int(node.get("count") or 0)
        mention_total = int(node.get("mention_total") or len(node.get("chunks") or []))
        reason = ""
        if len(name) < min_len:
            reason = "实体名过短"
            stats["过短实体剪除数"] += 1
        elif etype == "ratio":
            m = _TRIVIAL_RATIO.match(name)
            if m and float(m.group(1)) <= ratio_max and float(m.group(1)) == int(float(m.group(1))):
                reason = f"通用刻度比例（≤{ratio_max} 的整数百分比）"
                stats["通用刻度剪除数"] += 1
        if not reason and etype in prune_types:
            # 用 count（真实出现次数）判定，不是保留的提及块数（见 配置.py 的实测注释）
            if count > threshold and count >= min_abs_count:
                reason = f"通用刻度（出现 {count} 次 > 阈值 {max(threshold, min_abs_count)}，类型 {etype}）"
                stats["通用刻度剪除数"] += 1
            elif degree.get(name, 0) > max_degree:
                reason = f"枢纽噪声（度数 {degree.get(name, 0)} > {max_degree}）"
                stats["度数超限剪除数"] += 1
        if not reason and prune_frag and etype in ("company", "institution"):
            if len(name) > frag_max_len:
                reason = f"抽取碎片（公司/机构名 {len(name)} 字 > {frag_max_len}）"
                stats["抽取碎片剪除数"] += 1
            else:
                marker = next((mk for mk in frag_markers if mk in name), "")
                if marker:
                    reason = f"抽取碎片（含句级连词/指示词「{marker}」）"
                    stats["抽取碎片剪除数"] += 1
                else:
                    joiner = next((mk for mk in frag_joiners if mk in name), "")
                    if joiner and (len(name) >= frag_joiner_len or name.count(joiner) >= 2):
                        reason = (f"抽取碎片（含并列连词「{joiner}」且长度 {len(name)} ≥ {frag_joiner_len}）")
                        stats["抽取碎片剪除数"] += 1
        if not reason and degree.get(name, 0) == 0 and count <= 1:
            reason = "孤立节点（无任何边且仅出现 1 次）"
            stats["孤立节点剪除数"] += 1
        if reason:
            removed.append({"name": name, "type": etype, "degree": degree.get(name, 0),
                            "count": count, "mentions": mention_total, "reason": reason})
            continue
        node["degree"] = degree.get(name, 0)
        kept.append(node)

    keep_names = {str(n["name"]) for n in kept}
    kept_edges = [e for e in edges
                  if str(e["head"]) in keep_names and str(e["tail"]) in keep_names]
    stats["随节点剪除的边数"] = len(edges) - len(kept_edges)
    stats["输出节点数"] = len(kept)
    stats["输出边数"] = len(kept_edges)
    return {"nodes": kept, "edges": kept_edges, "removed": removed, "stats": stats,
            "degree_threshold": threshold}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def refine_graph(cfg: Any, *, logger: Any = None, base_graph_dir: Path | None = None,
                 out_dir: Path | None = None) -> dict[str, Any]:
    """跑完整的抽取层优化，产出优化后图谱与统计。"""
    started = time.perf_counter()
    ecfg = dict(getattr(cfg, "extract", {}) or {})
    src = Path(base_graph_dir or getattr(cfg, "base_graph_dir"))
    dst = Path(out_dir or getattr(cfg, "opt_graph_dir"))
    dst.mkdir(parents=True, exist_ok=True)

    entities = list(read_jsonl(src / "entities.jsonl"))
    relations = list(read_jsonl(src / "relations.jsonl"))
    stats_in: dict[str, Any] = {}
    if (src / "图谱构建统计.json").is_file():
        try:
            stats_in = json.loads((src / "图谱构建统计.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— 统计缺失不影响优化
            stats_in = {}
    chunks_scanned = int(stats_in.get("chunks_scanned") or 7430)

    raw_nodes = len(entities)
    raw_edges = len(relations)
    raw_degree = _mean_degree(entities, relations)

    merged = merge_aliases(entities, ecfg) if ecfg.get("alias_merge", True) else {
        "alias_of": {}, "groups": [], "stats": {}, "by_name": {}, "entities": entities}
    alias_of = dict(merged["alias_of"])

    nodes = merged_entities(entities, alias_of,
                            mentions_per_entity=int(ecfg.get("mentions_per_entity") or 4),
                            min_confidence=float(ecfg.get("min_confidence") or 0.0))
    node_type = {str(n["name"]): str(n["type"]) for n in nodes}
    rel = normalize_relations(relations, node_type=node_type, alias_of=alias_of, cfg=ecfg)
    pruned = prune_noise(nodes, rel["edges"], ecfg, chunks_scanned=chunks_scanned)

    final_nodes = pruned["nodes"]
    final_edges = pruned["edges"]
    dst.mkdir(parents=True, exist_ok=True)
    _write_jsonl(dst / "entities.jsonl", final_nodes)
    _write_jsonl(dst / "relations.jsonl", final_edges)

    by_type: dict[str, int] = {}
    for node in final_nodes:
        key = str(node.get("type_zh") or node.get("type"))
        by_type[key] = by_type.get(key, 0) + 1
    by_rel: dict[str, int] = {}
    for edge in final_edges:
        by_rel[str(edge["relation"])] = by_rel.get(str(edge["relation"]), 0) + 1

    stats = {
        "work_order": WORK_ORDER,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "输入": {"graph_dir": str(src), "节点": raw_nodes, "边": raw_edges, "平均度数": raw_degree},
        "输出": {"graph_dir": str(dst), "节点": len(final_nodes), "边": len(final_edges),
                 "平均度数": _mean_degree(final_nodes, final_edges)},
        "别名归并": merged["stats"],
        "关系规范化": rel["stats"],
        "噪声剪枝": pruned["stats"],
        "度数阈值": pruned["degree_threshold"],
        "entities_by_type": by_type,
        "relations_by_type": by_rel,
        "归并组样例": merged["groups"][:30],
        "剪除样例": pruned["removed"][:30],
        "方向无法判定样例": rel["direction_undecided"][:10],
        "配置": ecfg,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    (dst / "图谱优化统计.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    if logger is not None:
        logger.log_event("graphopt.refine_done", nodes_in=raw_nodes, nodes_out=len(final_nodes),
                         edges_in=raw_edges, edges_out=len(final_edges),
                         groups=merged["stats"].get("归并组数"), elapsed_ms=stats["elapsed_ms"])
    return {"stats": stats, "stats_path": str(dst / "图谱优化统计.json"),
            "entities_path": str(dst / "entities.jsonl"), "relations_path": str(dst / "relations.jsonl"),
            "nodes": len(final_nodes), "edges": len(final_edges)}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:                              # 边跑边 append，不堆内存
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def _mean_degree(nodes: Iterable[Mapping[str, Any]], edges: Iterable[Mapping[str, Any]]) -> float:
    nodes = list(nodes)
    edges = list(edges)
    if not nodes:
        return 0.0
    return round(2 * len(edges) / len(nodes), 4)
