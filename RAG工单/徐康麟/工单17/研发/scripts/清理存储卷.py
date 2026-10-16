# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

存储卷清理（**在取证之后执行**）：``研发/scripts/清理存储卷.py``。

## 前置条件

1. **服务必须已停止**（``pwsh -NoProfile -File 部署/脚本/本地停止.ps1``）——
   否则正在写的日志/上传文件会被占用；
2. 已跑过 ``研发/剖析/分析存储卷.py``（产出 ``存储卷取证_清理前.json``）。

## 清理策略（**只删可再生的重复物，保留证据**）

| 目标 | 动作 | 保留 |
| --- | --- | --- |
| ``卷/uploads`` | 每「原始文件名 + 字节数」组**只留 1 份** | 每种唯一内容 1 份样本 |
| ``卷/logs`` | 已轮转的历史日志全部删除；当前 ``app.log`` 截断到最近 N 行 | 当前日志 + ``日志样本_*.txt`` |
| ``卷/sessions`` | 默认保留（体量小、且是会话功能的证据） | 全部 |
| ``卷/runtime`` | 默认保留；``--vacuum`` 时对 sqlite 做一次 VACUUM 收缩 | 全部 |

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/清理存储卷.py --dry-run     # 先看要删什么
    pwsh -NoProfile -File run_py.ps1 研发/scripts/清理存储卷.py --keep-lines 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"
_ROOT = Path(__file__).resolve().parents[2]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def original_name(path: Path) -> str:
    name = path.name
    if "__" in name:
        head, _sep, tail = name.partition("__")
        if head.startswith("t-"):
            return tail
    return name


def clean_uploads(upload_dir: Path, *, dry_run: bool) -> dict:
    if not upload_dir.exists():
        return {"dir": str(upload_dir), "removed": 0, "freed": 0}
    groups: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for p in upload_dir.glob("*"):
        if not p.is_file() or p.suffix.lower() != ".pdf":
            continue
        try:
            groups[(original_name(p), p.stat().st_size)].append(p)
        except OSError:
            continue
    removed = 0
    freed = 0
    kept = []
    for (name, size), paths in groups.items():
        paths.sort(key=lambda p: p.name)          # 稳定：留字典序第一个
        kept.append({"original_name": name, "size_bytes": size, "kept": paths[0].name})
        for extra in paths[1:]:
            freed += extra.stat().st_size
            removed += 1
            if not dry_run:
                try:
                    extra.unlink()
                except OSError as exc:
                    print(f"  [警告] 删除失败 {extra.name}：{exc}")
    return {"dir": str(upload_dir), "groups": len(groups), "removed": removed,
            "freed": freed, "kept": kept}


def clean_logs(log_dir: Path, *, keep_lines: int, dry_run: bool) -> dict:
    """删掉已轮转的历史日志；当前日志截断到最近 ``keep_lines`` 行并另存样本。"""
    if not log_dir.exists():
        return {"dir": str(log_dir), "removed": 0, "freed": 0}
    removed = 0
    freed = 0
    samples = []
    for p in sorted(log_dir.rglob("*")):
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        # 已轮转的备份：app.log.1 / app.log.2 …
        if any(p.name.endswith(f".{i}") for i in range(1, 100)):
            sample = log_dir / f"日志样本_{p.name}.最近{keep_lines}行.txt"
            if not dry_run and not sample.exists():
                try:
                    with p.open("r", encoding="utf-8", errors="replace") as fp:
                        lines = fp.readlines()
                    sample.write_text("".join(lines[-keep_lines:]), encoding="utf-8")
                    samples.append(sample.name)
                except OSError:
                    pass
            freed += size
            removed += 1
            if not dry_run:
                try:
                    p.unlink()
                except OSError as exc:
                    print(f"  [警告] 删除失败 {p.name}：{exc}")
            continue
        # 当前日志：超过 32 MB 就截断到最近 keep_lines 行
        if size > 32 * 1024 * 1024:
            sample = log_dir / f"日志样本_{p.name}.最近{keep_lines}行.txt"
            try:
                with p.open("r", encoding="utf-8", errors="replace") as fp:
                    lines = fp.readlines()
                tail = "".join(lines[-keep_lines:])
                if not dry_run:
                    if not sample.exists():
                        sample.write_text(tail, encoding="utf-8")
                        samples.append(sample.name)
                    p.write_text(tail, encoding="utf-8")
                freed += size - len(tail.encode("utf-8"))
                removed += 1
            except OSError as exc:
                print(f"  [警告] 截断失败 {p.name}：{exc}")
    return {"dir": str(log_dir), "removed": removed, "freed": freed, "samples": samples}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 存储卷清理")
    ap.add_argument("--vol-root", default=str(_ROOT / "部署" / "卷"))
    ap.add_argument("--keep-lines", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=str(_ROOT / "测试" / "测试数据"))
    args = ap.parse_args(argv)

    vol = Path(args.vol_root)
    if not vol.exists():
        print(f"[跳过] 卷目录不存在：{vol}")
        return 0

    # 服务必须已停止
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:8200/healthz", timeout=3) as resp:
            if resp.status == 200:
                print("[中止] 8200 端口仍有服务在响应 —— 先停服务再清理")
                return 1
    except Exception:  # noqa: BLE001
        pass

    before_total = sum(p.stat().st_size for p in vol.rglob("*") if p.is_file())

    print(f"===== 存储卷清理{'（DRY RUN）' if args.dry_run else ''} =====")
    up = clean_uploads(vol / "uploads", dry_run=args.dry_run)
    print(f"  uploads：{up.get('groups')} 组唯一内容，删除 {up.get('removed')} 份重复副本，"
          f"释放 {human(up.get('freed', 0))}")
    lg = clean_logs(vol / "logs", keep_lines=args.keep_lines, dry_run=args.dry_run)
    print(f"  logs   ：处理 {lg.get('removed')} 个大/历史日志，释放 {human(lg.get('freed', 0))}，"
          f"样本 {lg.get('samples')}")
    if args.dry_run:
        print("  [DRY RUN] 未做任何删除")
        return 0

    after_total = sum(p.stat().st_size for p in vol.rglob("*") if p.is_file())
    freed = before_total - after_total
    report = {
        "work_order": WORK_ORDER, "vol_root": str(vol),
        "before_bytes": before_total, "after_bytes": after_total, "freed_bytes": freed,
        "before_mb": round(before_total / 1048576, 1), "after_mb": round(after_total / 1048576, 1),
        "freed_mb": round(freed / 1048576, 1), "freed_gb": round(freed / 1073741824, 3),
        "uploads": up, "logs": lg,
        "kept_samples": [k["kept"] for k in up.get("kept", [])],
        "note": "只删除可再生的重复副本与历史日志；每种唯一内容保留 1 份样本，日志保留最近行样本。",
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "存储卷取证_清理后.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n  清理前 {report['before_mb']} MB → 清理后 {report['after_mb']} MB")
    print(f"  **释放 {report['freed_mb']} MB（{report['freed_gb']} GB）**")
    print(f"  保留样本：{len(report['kept_samples'])} 份唯一内容 PDF")
    print(f"  落盘：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
