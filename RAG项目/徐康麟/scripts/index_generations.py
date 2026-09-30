# -*- coding: utf-8 -*-
"""索引**代数盘点**与多代收敛工具（t40）。

背景：同一个文件被重入过多次时，Milvus 里会**多代块并存**（每代一个 ``created_at``），
而旧代的碎块仍然对 ``query``/检索可见 —— 引用条号可能来自旧策略产物（跨条率高、文本差）。
本工具把这件事变成**可核对的数字**，并提供"每个文件只保留最新一代"的收敛动作。

    # ① 只读盘点（默认就是 dry-run，**绝不写数据**）
    python scripts/index_generations.py
    python scripts/index_generations.py --sample 20 --seed 20260917

    # ② 收敛（必须显式加 --apply；会先把被删主键清单落盘，再按批逻辑删除）
    python scripts/index_generations.py --apply

行为要点（对应任务书"先盘点、再收敛、后验证"）：

* **只读扫描**：``MilvusVectorStore.scan_rows`` 只取标量字段（不含 text/vector），
  并把去重后的行数与权威 ``count(*)`` 一起打印（服务端少行时默认可见）；
* **"最新一代"的定义**：同一文件内 ``created_at`` **最大**的那一代；若该文件只有一代，
  则不动它（即使那一代是 ``created_at == 0`` 的未知代 —— 见报告里的口径说明）；
* **零误伤硬约束**：只处理 ``role_id == "lawyer"`` 的行；plan 里出现任何非 lawyer 行
  就**拒绝执行**（断言式保护）；不碰其它 collection；**不调用** compact/flush；
* **留痕**：删前把被删主键清单（按文件分组 + 计数 + 分代明细）落盘成 JSON，
  语料在磁盘上可重灌；删后重新盘点并给出前后对比。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.config import RagConfig                    # noqa: E402
from legal_rag.logging_setup import setup_logging         # noqa: E402
from legal_rag.store.milvus_store import (                # noqa: E402
    INVENTORY_FIELDS,
    MilvusVectorStore,
)

#: 复验（t26）点名的两个文件：报告里必须单独给出它们的分代明细
NAMED_FILES = ("中华人民共和国民法典", "中华人民共和国生态环境法典")

DEFAULT_INVENTORY = "handoff/t40-index-generations.json"
DEFAULT_PLAN = "handoff/t40-delete-plan.json"
#: 机器可读的**全量明细** dump（逐文件逐代）。人工阅读的报告另有一份
#: `handoff/t40-INDEX-GENERATIONS.md`（结论 + 关键表 + 证据索引），不会被本脚本覆盖。
DEFAULT_REPORT = "handoff/t40-index-dump.md"


def file_key(row: dict) -> str:
    """文件的归组键：优先 ``doc_id``（重入同一文件时它稳定），空则退回 ``source``。"""
    return str(row.get("doc_id") or row.get("source") or "<无 doc_id/source>")


def resolve_collection(config, override: str = "") -> str:
    """确定要盘点的 collection 名（**只读解析，不创建任何东西**）。

    关键坑（真踩过）：``RagConfig().collection`` 的默认值是历史遗留的 ``legal_kb``，
    而真实索引名在 ``config.milvus.collection``（``legal_rag_chunks_bge_m3_1024``）。
    ``legal_rag/store/base.py::_build_milvus_store`` 有一条"把 ``legal_kb`` 换成
    本项目默认名"的兼容改写；**直接 ``MilvusVectorStore(config)`` 会绕过它**，
    于是指向一个不存在的 collection（connect 还会顺手把它建出来）。
    所以这里按同样的规则解析，并**以磁盘上的 manifest 为准**（它记录真实写入的 collection）。
    """
    if override:
        return override
    manifest_path = Path(config.index_dir) / "milvus_manifest.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("collection"):
                return str(payload["collection"])
        except (OSError, json.JSONDecodeError):
            pass
    name = str(getattr(config, "collection", "") or "")
    if not name or name in ("legal_kb", "legal_kb_default"):
        name = str(getattr(getattr(config, "milvus", None), "collection", "") or "")
    return name or "legal_rag_chunks_bge_m3_1024"


def resolve_dim(config, override: int = 0) -> tuple[int, str]:
    """确定向量维度：``--dim`` > manifest > 配置兜底提示（并说明来源）。"""
    if override:
        return override, "--dim 参数"
    manifest_path = Path(config.index_dir) / "milvus_manifest.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            dim = int(payload.get("dim") or 0)
            if dim > 0:
                return dim, f"manifest({manifest_path.name})"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    dim = int(getattr(config, "embedding_dim", 0) or 0)
    return dim, "config.embedding_dim（兜底提示，可能不准）"


def generation_of(row: dict) -> float:
    try:
        return float(row.get("created_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_inventory(rows: list[dict]) -> dict:
    """把行列表汇总成"每文件 → 每代"的盘点结构（纯计算，不碰存储）。"""
    generations: dict[str, dict[float, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"rows": 0, "parents": 0, "children": 0,
                                     "roles": set(), "sources": set()}))
    role_rows: dict[str, int] = defaultdict(int)
    for row in rows:
        role = str(row.get("role_id") or "")
        role_rows[role] += 1
        key = file_key(row)
        created = generation_of(row)
        slot = generations[key][created]
        slot["rows"] += 1
        if bool(row.get("is_parent")):
            slot["parents"] += 1
        else:
            slot["children"] += 1
        slot["roles"].add(role)
        slot["sources"].add(str(row.get("source") or ""))

    files: list[dict] = []
    for key, by_generation in generations.items():
        entries = []
        for created in sorted(by_generation):
            slot = by_generation[created]
            entries.append({
                "created_at": created,
                "rows": slot["rows"],
                "parents": slot["parents"],
                "children": slot["children"],
                "roles": sorted(slot["roles"]),
                "sources": sorted(s for s in slot["sources"] if s),
            })
        latest = max(by_generation)
        sources = sorted({s for entry in entries for s in entry["sources"]})
        files.append({
            "doc_id" if key != "<无 doc_id/source>" else "source": key,
            "sources": sources,
            "generations": len(entries),
            "rows_total": sum(e["rows"] for e in entries),
            "latest_created_at": latest,
            "latest_rows": next(e["rows"] for e in entries if e["created_at"] == latest),
            "older_rows": sum(e["rows"] for e in entries if e["created_at"] < latest),
            "detail": entries,
        })
    files.sort(key=lambda item: (-item["generations"], -item["rows_total"], item["doc_id"]))
    return {
        "scanned_rows": len(rows),
        "files": files,
        "role_rows": dict(sorted(role_rows.items())),
        "files_with_multiple_generations": sum(1 for f in files if f["generations"] > 1),
        "rows_in_older_generations": sum(f["older_rows"] for f in files),
    }


def build_plan(rows: list[dict], inventory: dict) -> dict:
    """算出"每个文件保留最新一代"要删掉的**主键清单**（纯计算，不做删除）。

    只有 ``role_id == "lawyer"`` 的行会进 plan；其它角色即使多代也**原样保留**
    （本任务不碰非 lawyer 行）。
    """
    keep_latest = {f["doc_id"]: f["latest_created_at"] for f in inventory["files"]}
    per_file: dict[str, dict] = {}
    ids: list[str] = []
    non_lawyer: list[str] = []
    missing_created: list[str] = []
    for row in rows:
        role = str(row.get("role_id") or "")
        key = file_key(row)
        created = generation_of(row)
        latest = keep_latest.get(key)
        if latest is None or created >= latest:
            continue
        chunk_id = str(row.get("id") or "")
        if not chunk_id:
            continue
        if role != "lawyer":
            non_lawyer.append(chunk_id)
            continue
        if created == 0.0:
            missing_created.append(chunk_id)
        slot = per_file.setdefault(key, {"doc_id": key,
                                         "keep_created_at": latest,
                                         "delete_ids": [],
                                         "delete_by_created_at": defaultdict(int)})
        slot["delete_ids"].append(chunk_id)
        slot["delete_by_created_at"][created] += 1
        ids.append(chunk_id)
    plan_files = []
    for key, slot in sorted(per_file.items()):
        plan_files.append({
            "doc_id": key,
            "keep_created_at": slot["keep_created_at"],
            "delete_total": len(slot["delete_ids"]),
            "delete_by_created_at": {str(k): v for k, v in
                                     sorted(slot["delete_by_created_at"].items())},
            "delete_ids": slot["delete_ids"],
        })
    return {
        "policy": "每个文件只保留 created_at 最大的一代；更早代逻辑删除（不 compact/flush）",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "delete_total": len(ids),
        "files_touched": len(plan_files),
        "non_lawyer_rows_excluded": len(non_lawyer),
        "rows_with_created_at_zero": len(missing_created),
        "ids": ids,
        "files": plan_files,
    }


def apply_plan(store: MilvusVectorStore, plan: dict, *, batch: int = 500) -> dict:
    """执行逻辑删除（**调用方必须已确认 plan**）。返回对账数字。

    安全闸：plan 里若混进非 lawyer 行，直接抛错**拒绝执行**（宁可不动，也不误伤）。
    """
    if plan["non_lawyer_rows_excluded"]:
        raise RuntimeError("plan 里出现非 lawyer 行，拒绝执行（请先查清来源）")
    todos = list(plan["ids"])
    before = store.count()
    removed = 0
    for start in range(0, len(todos), batch):
        piece = todos[start:start + batch]
        removed += store.delete(piece)
    after = store.count()
    return {"before": before, "after": after, "requested": len(todos),
            "removed": removed, "delta": before - after,
            "matched": removed == len(todos)}


def _fmt_generation(entry: dict) -> str:
    return (f"`{entry['created_at']:.0f}` → {entry['rows']} 行"
            f"（父 {entry['parents']} / 子 {entry['children']}）")


def render_report(inventory: dict, plan: dict, *, sample: list[dict], named: list[dict],
                  source_count: int | None, generated: str) -> str:
    lines: list[str] = []
    add = lines.append
    add("# t40 索引多代块盘点与收敛报告\n")
    add(f"- 生成时间：{generated}")
    add(f"- 扫描口径：`{', '.join(INVENTORY_FIELDS)}`（**不含 text/vector**）")
    add(f"- 扫描到的行数（按 id 去重）：**{inventory['scanned_rows']}**"
        f"；权威 `count(*)`：**{source_count if source_count is not None else '未取到'}**")
    add(f"- 角色分布：{json.dumps(inventory['role_rows'], ensure_ascii=False)}")
    add(f"- **多代文件数：{inventory['files_with_multiple_generations']}**"
        f" / 共 {len(inventory['files'])} 个文件；"
        f"**落在旧代的行：{inventory['rows_in_older_generations']}**")
    add(f"- 待删（plan）行数：**{plan['delete_total']}**，涉及 {plan['files_touched']} 个文件；"
        f"被排除的非 lawyer 行：{plan['non_lawyer_rows_excluded']}")
    add("")
    add("> 「最新一代」= 同一文件内 `created_at` **最大**的一代；只有一代的文件不动。")
    add("> 被删主键清单（可据此重灌）：见同目录 `t40-delete-plan.json`（key = `ids`，"
        "并按文件分组在 `files[].delete_ids`）。\n")

    add("## 复验点名文件\n")
    for item in named:
        add(f"### {item.get('doc_id')}（{item['generations']} 代，共 {item['rows_total']} 行）")
        for entry in item["detail"]:
            mark = " ← **保留**" if entry["created_at"] == item["latest_created_at"] else " ← 待删"
            add(f"- {_fmt_generation(entry)}{mark}")
        add("")
    if not named:
        add("（未在索引里找到点名文件：按 `doc_id`/`source` 子串匹配，逐字记录）\n")

    add("## 随机抽样文件\n")
    for item in sample:
        add(f"- **{item.get('doc_id')}**：{item['generations']} 代 / {item['rows_total']} 行 → "
            + "；".join(_fmt_generation(e) for e in item["detail"]))
    add("")
    add("## 待删明细（按文件）\n")
    add("| 文件 | 保留代 created_at | 删除行数 | 各代删除数 |")
    add("|---|---|---|---|")
    for entry in plan["files"]:
        by_gen = "，".join(f"{k}→{v}" for k, v in entry["delete_by_created_at"].items())
        add(f"| {entry['doc_id']} | {entry['keep_created_at']:.0f} | {entry['delete_total']} | {by_gen} |")
    add("")
    add("## 边界与口径\n")
    add("- 只处理 `role_id == \"lawyer\"` 的行；plan 里若混进非 lawyer 行会**拒绝执行**。")
    add("- 不碰其它 collection（`five` / `legal_rag_memory`）；不删源文件、`index/` 原始资料、`uploads/`。")
    add("- **不做物理 compact/flush**：Milvus 的逻辑删除在查询侧立即生效，但**物理空间仍待用户决定**。")
    add("- 重灌路径：语料在磁盘上，按原入库流程重跑即可（被删主键清单给出「删掉了哪些 id」）。")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="索引代数盘点与多代收敛（默认只读）")
    parser.add_argument("--apply", action="store_true",
                        help="**显式**执行逻辑删除（默认不删；不做 compact/flush）")
    parser.add_argument("--sample", type=int, default=20, help="随机抽样文件数（默认 20）")
    parser.add_argument("--seed", type=int, default=20260917, help="抽样随机种子（可复现）")
    parser.add_argument("--inventory-out", default=DEFAULT_INVENTORY)
    parser.add_argument("--plan-out", default=DEFAULT_PLAN)
    parser.add_argument("--report-out", default=DEFAULT_REPORT)
    parser.add_argument("--max-rows", type=int, default=0, help="只扫前 N 行（调试用）")
    parser.add_argument("--collection", default="", help="覆盖 collection 名（默认按 manifest/配置解析）")
    parser.add_argument("--dim", type=int, default=0, help="覆盖向量维度（默认按 manifest 取）")
    args = parser.parse_args(argv)

    setup_logging()
    # from_env()：否则 MILVUS_URI / LEGAL_RAG_MILVUS_URI 等环境变量被忽略，
    # 命令行工具会去连默认 127.0.0.1:19530（真机踩坑，同 generation_filter）。
    config = RagConfig.from_env()
    collection = resolve_collection(config, args.collection)
    dim, dim_source = resolve_dim(config, args.dim)

    # ⚠️ 存在性预检：**只读列举**，不存在就退出 —— 绝不让 connect() 顺手建出一个空 collection
    from pymilvus import MilvusClient
    probe = MilvusClient(uri=MilvusVectorStore(config).uri)
    existing = sorted(probe.list_collections())
    if collection not in existing:
        print(f"[中止] collection {collection!r} 不存在（现有：{existing}）—— 不做任何操作。")
        return 2
    print(f"collection = {collection}（现有 collection：{existing}）")
    print(f"dim = {dim}（来源：{dim_source}）")

    store = MilvusVectorStore(config, dim=dim, collection=collection,
                              manifest_dir=config.index_dir,
                              recreate_on_dim_mismatch=False)
    store.connect(create=False)          # create=False：不存在就报错，绝不建表
    print(f"已连接：uri={store.uri} collection={store.collection}")

    authoritative = store.count()
    rows = store.scan_rows(max_rows=args.max_rows)
    print(f"[只读扫描] 去重后 {len(rows)} 行；权威 count(*) = {authoritative}"
          f"（差 {authoritative - len(rows)} 行 —— 少行是服务端痕迹，见 MILVUS-2200-NOTES）")

    inventory = build_inventory(rows)
    plan = build_plan(rows, inventory)

    named = []
    for name in NAMED_FILES:
        hit = [f for f in inventory["files"]
               if name in f["doc_id"] or any(name in s for s in f["sources"])]
        for item in hit:
            named.append(item)
            print(f"[点名] {item['doc_id']}: {item['generations']} 代 / {item['rows_total']} 行")
            for entry in item["detail"]:
                print(f"        created_at={entry['created_at']:.0f} rows={entry['rows']} "
                      f"parents={entry['parents']} children={entry['children']}")
    if not named:
        print(f"[点名] 未在索引里找到 {NAMED_FILES}（按 doc_id/source 子串匹配）")

    sample_pool = [f for f in inventory["files"] if f["generations"] > 1]
    rng = random.Random(args.seed)
    sample = rng.sample(sample_pool, min(args.sample, len(sample_pool)))

    print(f"\n[盘点] 文件 {len(inventory['files'])} 个；多代文件 "
          f"{inventory['files_with_multiple_generations']} 个；旧代行 "
          f"{inventory['rows_in_older_generations']} 行")
    print(f"[plan] 待删 {plan['delete_total']} 行（{plan['files_touched']} 个文件）；"
          f"排除的非 lawyer 行 {plan['non_lawyer_rows_excluded']}；"
          f"created_at=0 的行 {plan['rows_with_created_at_zero']}")

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(args.inventory_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.inventory_out).write_text(
        json.dumps({"generated_at": generated, "collection": store.collection,
                    "authoritative_count": authoritative, **inventory,
                    "named_files": [f["doc_id"] for f in named]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.plan_out).write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    Path(args.report_out).write_text(
        render_report(inventory, plan, sample=sample, named=named,
                      source_count=authoritative, generated=generated), encoding="utf-8")
    print(f"[落盘] {args.inventory_out}")
    print(f"[落盘] {args.plan_out}（被删主键清单：{plan['delete_total']} 个 id）")
    print(f"[落盘] {args.report_out}")

    if not args.apply:
        print("\n[模式] 只读盘点（dry-run）：**没有删除任何行**。"
              "确认后加 `--apply` 执行逻辑删除。")
        return 0

    print("\n[模式] --apply：开始按 plan 逻辑删除（不 compact/flush）")
    result = apply_plan(store, plan)
    print(f"[对账] 前 {result['before']} 行 → 后 {result['after']} 行；"
          f"请求删 {result['requested']}，Milvus 报删 {result['removed']}，"
          f"count 差 {result['delta']}（三者应一致）")
    after_rows = store.scan_rows(max_rows=args.max_rows)
    after_inventory = build_inventory(after_rows)
    print(f"[复盘点] 去重后 {len(after_rows)} 行；多代文件 "
          f"{after_inventory['files_with_multiple_generations']} 个；旧代行 "
          f"{after_inventory['rows_in_older_generations']} 行")
    Path(str(args.inventory_out).replace(".json", "-after.json")).write_text(
        json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "collection": store.collection,
                    "authoritative_count": store.count(),
                    "apply_result": result, **after_inventory},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
