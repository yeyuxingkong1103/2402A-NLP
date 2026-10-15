# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：工单08「只读复用」取证工具

用途：工单09 声称「08 目录物理零改动，永远是可复现的优化前基线」。
      本脚本把该声称变成**可验证的证据**：对 08 的关键产物做快照（路径 + 大小 +
      mtime + 内容哈希），事后比对，任何写痕迹都会被抓出来。

用法：
    python scripts/wo8_guard.py snapshot   # 跑 09 之前：落一份快照
    python scripts/wo8_guard.py verify     # 跑 09 之后：与快照比对
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w9 import compat  # noqa: E402

SNAP_PATH = os.path.join(compat.wo9_data(), "wo8_snapshot.json")

# 08 的关键产物与源码（源码也要纳入：证明没被改过）
TARGETS = [
    compat.wo8_data("kg", "kg.pkl"),
    compat.wo8_data("kg", "kg.graphml"),
    compat.wo8_data("kg", "kg_stats.json"),
    compat.wo8_data("kg", "kg_entities.jsonl"),
    compat.wo8_data("kg", "kg_relations.jsonl"),
    compat.wo8_data("kg", "llm_extract_results.jsonl"),
    compat.wo8_data("chunks", "chunks.json"),
    compat.wo8_data("eval", "ground_truth.json"),
    compat.wo8_data("qdrant", "meta.json"),
]
# 整目录签名：直接对 **08 工程根**做一次全树扫描，覆盖 data/ 产物、src/ 算法实现、
# prompts/ 与根目录散落文件。
#
# 为什么不只挑几个关键文件：实测发现跑一次基线会把 `data/index/fulltext.pkl`
# 重新持久化 —— 只盯预先想到的文件必然漏掉这类写入。全树扫一遍才不留死角。
# 成本很低：只 stat（大小 + mtime），不读内容，46 MB 的 zip 也只是两条 stat。
SRC_DIRS = [compat.workorder8_root()]

# 排除：版本库、缓存、日志（日志本来就会被写，不属于「产物被改动」）
WALK_EXCLUDE = {".git", "__pycache__", ".pytest_cache", "_scratch", "logs"}


def _file_sig(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {"exists": False}
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"exists": True, "size": st.st_size,
            "mtime": round(st.st_mtime, 3), "sha256": h.hexdigest()}


def _dir_sig(path: str) -> dict:
    """目录签名：文件数 + 「相对路径|大小|mtime_ns」汇总哈希。

    为什么必须带上 mtime_ns：只比大小会漏掉「**内容变了但大小没变**」的重写。
    实测教训 —— 跑一次 08 基线会把 `data/index/fulltext.pkl`（31.7 MB）重新持久化，
    其大小可能与旧文件完全一致；只有 mtime 变化能抓到这次写入。
    （仍不读文件内容：97 MB 的 Qdrant 目录全量哈希太慢，而写入必然改 mtime。）
    """
    if not os.path.isdir(path):
        return {"exists": False}
    h = hashlib.sha256()
    n = 0
    for dp, dn, fn in os.walk(path):
        dn[:] = [d for d in dn if d not in WALK_EXCLUDE]
        for f in sorted(fn):
            if f.endswith((".pyc", ".pyo")):
                continue
            fp = os.path.join(dp, f)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, path).replace("\\", "/")
            h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
            n += 1
    return {"exists": True, "files": n, "sha256": h.hexdigest()}


def take_snapshot() -> dict:
    return {
        "workorder8_root": compat.workorder8_root(),
        "files": {os.path.relpath(p, compat.workorder8_root()): _file_sig(p)
                  for p in TARGETS if p},
        "dirs": {os.path.relpath(p, compat.workorder8_root()): _dir_sig(p)
                 for p in SRC_DIRS},
    }


def cmd_snapshot() -> int:
    snap = take_snapshot()
    with io.open(SNAP_PATH, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    print(f"已落快照：{SNAP_PATH}")
    print(f"  文件 {len(snap['files'])} 个，源码目录 {len(snap['dirs'])} 个")
    return 0


def cmd_verify() -> int:
    if not os.path.isfile(SNAP_PATH):
        print(f"未找到快照 {SNAP_PATH}，请先运行 snapshot")
        return 2
    with io.open(SNAP_PATH, encoding="utf-8") as f:
        old = json.load(f)
    new = take_snapshot()

    diffs: list[str] = []
    for group in ("files", "dirs"):
        for k, ov in old[group].items():
            nv = new[group].get(k, {})
            if ov.get("sha256") != nv.get("sha256"):
                diffs.append(f"[{group}] 内容变化：{k}")
            elif ov.get("mtime") != nv.get("mtime"):
                diffs.append(f"[{group}] mtime 变化（内容未变）：{k}")

    print("=" * 68)
    print("工单08 只读复用取证")
    print("=" * 68)
    if diffs:
        print(f"❌ 检测到 {len(diffs)} 处变化 —— 08 目录**不是**只读的：")
        for d in diffs:
            print("   ·", d)
        return 1
    print("✅ 工单08 全部产物与源码**逐字节未变** —— 只读复用成立。")
    print(f"   （比对文件 {len(old['files'])} 个，源码目录 {len(old['dirs'])} 个）")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    sys.exit(cmd_snapshot() if cmd == "snapshot" else cmd_verify())
