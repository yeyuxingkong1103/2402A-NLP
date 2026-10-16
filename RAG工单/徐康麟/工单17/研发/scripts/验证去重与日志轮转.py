# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

F11/F12 前后对比实测：``研发/scripts/验证去重与日志轮转.py``。

## 测什么

工单原文泄漏点一「全局缓存或会话未释放」在**落盘侧**的同构现象（本工单实测发现）：

| 现象 | 实测证据（清理前） | 对应修复 |
| --- | --- | --- |
| 上传路径**无内容去重**：同一份 PDF 上传 N 次留 N 份副本 | `2304` 份文件 / `15430.2 MB`，只对应 **10 组唯一内容**，浪费 `15369.9 MB`（**99.61 %**） | F12 `PERF_UPLOAD_DEDUP`（SHA256 索引命中即复用） |
| 业务日志**无轮转**：单文件只增不减 | `部署/卷/logs/app.log` 单文件 **600 MB**（日志卷合计 665 MB） | F11 `PERF_LOG_MAX_MB` / `PERF_LOG_BACKUP_COUNT`（写前按大小轮转） |

## 怎么测

**要求真实服务在跑**（默认 `http://127.0.0.1:8200`）。
用同一份真实年报 PDF 连续上传 N 次，前后各测一次磁盘占用与日志体积。

    # 优化前口径（PERF_ENABLED=0：不去重、不轮转）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/验证去重与日志轮转.py --label 优化前 --n 100

    # 优化后口径（PERF_ENABLED=1 + PERF_UPLOAD_DEDUP=1 + PERF_LOG_MAX_MB=1）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/验证去重与日志轮转.py --label 优化后 --n 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "测试" / "测试数据"
PDF_DIR = Path(r"E:\gao6gongdan\工单7\研发\data\raw")


def dir_stats(path: Path) -> dict[str, Any]:
    """目录体量与文件数（含最大单文件）。"""
    if not path.exists():
        return {"dir": str(path), "files": 0, "bytes": 0, "mb": 0.0, "biggest_mb": 0.0,
                "biggest": None}
    files = [p for p in path.rglob("*") if p.is_file()]
    total = 0
    biggest = (0, "")
    for p in files:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        total += size
        if size > biggest[0]:
            biggest = (size, p.name)
    return {"dir": str(path), "files": len(files), "bytes": total,
            "mb": round(total / (1024 * 1024), 2),
            "biggest_mb": round(biggest[0] / (1024 * 1024), 2),
            "biggest": biggest[1] or None}


def upload(base: str, filename: str, content: bytes, *, timeout: float = 120.0) -> tuple[int, dict, str]:
    boundary = "----dedup" + str(int(time.time() * 1000))
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/pdf\r\n\r\n").encode("utf-8") + content + \
           f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(base + "/api/v1/uploads", data=body, method="POST",
                                headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                         "Content-Length": str(len(body))})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), ""
            except json.JSONDecodeError:
                return resp.status, {}, raw[:200]
    except urllib.error.HTTPError as exc:
        text = ""
        try:
            text = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        return int(exc.code), {}, text[:160]
    except Exception as exc:  # noqa: BLE001
        return 0, {}, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 去重与日志轮转前后对比")
    ap.add_argument("--base-url", default="http://127.0.0.1:8200")
    ap.add_argument("--label", default="优化前")
    ap.add_argument("--n", type=int, default=100, help="同一份 PDF 的重复上传次数")
    ap.add_argument("--upload-dir", default=str(_ROOT / "部署" / "卷" / "uploads"))
    ap.add_argument("--log-dir", default=str(_ROOT / "部署" / "卷" / "logs"))
    ap.add_argument("--outdir", default=str(_DATA))
    args = ap.parse_args(argv)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[错误] 找不到上传语料：{PDF_DIR}")
        return 2
    pdf = pdfs[0]
    content = pdf.read_bytes()

    upload_dir, log_dir = Path(args.upload_dir), Path(args.log_dir)
    before_up, before_log = dir_stats(upload_dir), dir_stats(log_dir)

    print(f"===== {args.label}：同一份 PDF 连续上传 {args.n} 次 =====")
    print(f"  语料：{pdf.name}（{len(content) / 1048576:.2f} MB）")
    print(f"  上传前：uploads {before_up['files']} 文件 / {before_up['mb']} MB；"
          f"logs {before_log['files']} 文件 / {before_log['mb']} MB")

    started = time.perf_counter()
    ok = 0
    codes: dict[str, int] = {}
    for i in range(max(1, args.n)):
        status, _body, _err = upload(args.base_url, pdf.name, content)
        codes[str(status)] = codes.get(str(status), 0) + 1
        if status in (200, 202):
            ok += 1
    elapsed = round(time.perf_counter() - started, 2)

    # 给解析 worker 一点时间把队列消费掉（不影响磁盘统计，但要看到 done）
    task_stats: dict[str, Any] = {}
    try:
        import urllib.request as _u

        with _u.urlopen(args.base_url + "/api/v1/tasks", timeout=15) as resp:
            task_stats = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        task_stats = {}

    after_up, after_log = dir_stats(upload_dir), dir_stats(log_dir)
    delta_up = after_up["bytes"] - before_up["bytes"]
    delta_log = after_log["bytes"] - before_log["bytes"]
    per_upload = delta_up / max(1, args.n)

    dedup = (task_stats.get("dedup") or {}) if task_stats else {}
    report = {
        "work_order": WORK_ORDER, "label": args.label, "base_url": args.base_url,
        "pdf": pdf.name, "pdf_bytes": len(content), "uploads": max(1, args.n),
        "http_ok": ok, "http_codes": codes, "elapsed_s": elapsed,
        "uploads_dir_before": before_up, "uploads_dir_after": after_up,
        "logs_dir_before": before_log, "logs_dir_after": after_log,
        "delta_uploads_bytes": delta_up, "delta_uploads_mb": round(delta_up / 1048576, 2),
        "delta_logs_bytes": delta_log, "delta_logs_mb": round(delta_log / 1048576, 2),
        "bytes_per_upload": round(per_upload, 1),
        "ratio_to_pdf_size": round(per_upload / len(content), 4),
        "task_queue_dedup": dedup,
        "task_queue_counts": {k: task_stats.get(k) for k in ("submitted", "done", "failed", "rejected")}
        if task_stats else {},
        "perm_enabled": task_stats.get("dedup", {}).get("enabled") if task_stats else None,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"去重与轮转_{args.label}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"  HTTP 成功 {ok}/{args.n}（{codes}），耗时 {elapsed} s")
    print(f"  上传后：uploads {after_up['files']} 文件 / {after_up['mb']} MB")
    print(f"  **磁盘增量 {report['delta_uploads_mb']} MB（每份上传 {report['bytes_per_upload']} 字节，"
          f"= PDF 体积的 {report['ratio_to_pdf_size']} 倍）**")
    print(f"  日志卷：{before_log['mb']} → {after_log['mb']} MB（增量 {report['delta_logs_mb']} MB，"
          f"最大单文件 {after_log['biggest_mb']} MB ← {after_log['biggest']}）")
    if dedup:
        print(f"  去重统计：enabled={dedup.get('enabled')} 命中={dedup.get('hits')} "
              f"实写={dedup.get('writes')} 省下={dedup.get('saved_mb')} MB")
    print(f"  落盘：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
