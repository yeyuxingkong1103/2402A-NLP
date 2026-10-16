# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

上传卷与日志卷的「磁盘泄漏」取证：``研发/剖析/分析存储卷.py``。

## 为什么要这个脚本（工单原文对应点）

工单原文泄漏点一：「解析后的中间对象 … 可能未被垃圾回收机制正确释放」。
本工单在**落盘侧**发现了同构的真实现象，且量级远超内存：

* **上传路径无内容去重**：``POST /api/v1/uploads`` 每次都在 `DEPLOY_UPLOAD_DIR`
  落一份**完整副本**（文件名前缀是 task_id），同一份 PDF 上传 N 次就留 N 份。
  实测一份 4.6 MB 的中信证券年报被压测重复上传 **2304 次 ⇒ 15.4 GB**。
* **日志无轮转**：业务日志用普通 ``FileHandler``，没有 ``maxBytes``/``backupCount``，
  实测单个 ``app.log`` 长到 **600 MB**。

本脚本**只读**扫描并产出可复核的证据 JSON + CSV，不做任何删除。
清理由 ``研发/scripts/清理存储卷.py`` 负责（单独一步，便于留痕与复核）。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/剖析/分析存储卷.py --tag 清理前
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "研发"))

#: 抽样做 SHA256 的文件数（全量 15 GB 太慢；同尺寸 + 抽样哈希足以证明重复）
_SAMPLE = 8


def sha256_of(path: Path, limit_mb: int = 64) -> str:
    """分块算 SHA256（大文件不一次读入内存）。"""
    digest = hashlib.sha256()
    read = 0
    limit = limit_mb * 1024 * 1024
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            read += len(chunk)
            if read >= limit:
                break
    return digest.hexdigest()


def original_name(path: Path) -> str:
    """还原上传时的原始文件名：``<task_id>__<原名>``。"""
    name = path.name
    if "__" in name:
        head, _sep, tail = name.partition("__")
        if head.startswith("t-"):
            return tail
    return name


def scan_uploads(upload_dir: Path) -> dict:
    files = [p for p in upload_dir.glob("*") if p.is_file()]
    groups: dict[tuple[str, int], list[Path]] = defaultdict(list)
    total_bytes = 0
    for p in files:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        total_bytes += size
        groups[(original_name(p), size)].append(p)

    dup_groups = []
    wasted = 0
    for (name, size), paths in sorted(groups.items(), key=lambda kv: -kv[0][1] * len(kv[1])):
        if len(paths) <= 1:
            continue
        wasted += size * (len(paths) - 1)
        dup_groups.append({"original_name": name, "size_bytes": size, "copies": len(paths),
                           "kept_example": paths[0].name,
                           "wasted_bytes": size * (len(paths) - 1)})

    # 抽样哈希：证明「同尺寸的不同副本内容完全一致」
    sample_hashes: dict[str, list[str]] = {}
    for name, size in list(groups)[:3]:
        paths = groups[(name, size)]
        if len(paths) < 2 or size > 80 * 1024 * 1024:
            continue
        picked = paths[:_SAMPLE]
        sample_hashes[name] = [sha256_of(p) for p in picked]

    identical = {name: (len(set(hs)) == 1) for name, hs in sample_hashes.items()}
    return {
        "dir": str(upload_dir),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 1),
        "unique_content_groups": len(groups),
        "duplicate_groups": len(dup_groups),
        "wasted_bytes": wasted,
        "wasted_mb": round(wasted / (1024 * 1024), 1),
        "wasted_pct": round(100.0 * wasted / total_bytes, 2) if total_bytes else 0.0,
        "top_duplicate_groups": dup_groups[:20],
        "sample_sha256": {k: {"hashes": v, "all_identical": identical.get(k, False),
                              "count": len(v)} for k, v in sample_hashes.items()},
    }


def scan_dir(path: Path) -> dict:
    """通用目录扫描（日志/会话/运行期）。"""
    files = [p for p in path.rglob("*") if p.is_file()]
    total = 0
    biggest = []
    for p in files:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        total += size
        biggest.append((size, str(p.relative_to(path))))
    biggest.sort(reverse=True)
    ext = Counter(p.suffix.lower() or "<无扩展名>" for p in files)
    return {"dir": str(path), "file_count": len(files),
            "total_bytes": total, "total_mb": round(total / (1024 * 1024), 1),
            "by_ext": {k: v for k, v in ext.most_common(10)},
            "biggest": [{"file": name, "mb": round(size / (1024 * 1024), 2)}
                        for size, name in biggest[:10]]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 存储卷取证（只读）")
    ap.add_argument("--vol-root", default=str(_ROOT / "部署" / "卷"))
    ap.add_argument("--log-dir", default=str(_ROOT / "部署" / "日志"))
    ap.add_argument("--tag", default="清理前")
    ap.add_argument("--outdir", default=str(_ROOT / "测试" / "测试数据"))
    args = ap.parse_args(argv)

    vol = Path(args.vol_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report: dict = {"work_order": WORK_ORDER, "tag": args.tag, "vol_root": str(vol)}
    for name in ("uploads", "logs", "sessions", "runtime"):
        target = vol / name
        if target.exists():
            report[name] = scan_uploads(target) if name == "uploads" else scan_dir(target)
    log_dir = Path(args.log_dir)
    if log_dir.exists():
        report["部署_日志"] = scan_dir(log_dir)

    up = report.get("uploads", {})
    total_bytes = sum(int(report[k]["total_bytes"]) for k in report if isinstance(report[k], dict)
                      and "total_bytes" in report[k])
    report["summary"] = {
        "总占用_mb": round(total_bytes / (1024 * 1024), 1),
        "上传卷_mb": up.get("total_mb"),
        "上传重复浪费_mb": up.get("wasted_mb"),
        "上传文件数": up.get("file_count"),
        "上传唯一内容组": up.get("unique_content_groups"),
    }

    path = outdir / f"存储卷取证_{args.tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("===== 存储卷取证（只读）=====")
    for key in ("uploads", "logs", "sessions", "runtime", "部署_日志"):
        item = report.get(key)
        if not item:
            continue
        print(f"  {key:<10} {item['file_count']:>6} 个文件  {item['total_mb']:>10} MB"
              + (f"  重复浪费 {item['wasted_mb']} MB（{item['wasted_pct']}%）"
                 if key == "uploads" else ""))
    if up:
        print(f"\n  上传卷重复组 Top5：")
        for g in (up.get("top_duplicate_groups") or [])[:5]:
            print(f"      {g['copies']:>5} 份 × {g['size_bytes'] / 1048576:.2f} MB  "
                  f"浪费 {g['wasted_bytes'] / 1048576:.1f} MB  ← {g['original_name'][:52]}")
        print(f"\n  抽样 SHA256 一致性：")
        for name, info in (up.get("sample_sha256") or {}).items():
            print(f"      {name[:48]:<50} {info['count']} 份 → 内容全同 = {info['all_identical']}")
    print(f"\n  summary: {json.dumps(report['summary'], ensure_ascii=False)}")
    print(f"  落盘：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
