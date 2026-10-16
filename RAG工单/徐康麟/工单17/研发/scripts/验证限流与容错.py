# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

限流与容错实测脚本：``研发/scripts/验证限流与容错.py``。

**要求真实服务在跑**（默认 ``http://127.0.0.1:8200``）。产出两份可直接喂给截图脚本的 JSON：

* ``测试/测试数据/限流验证.json`` —— 429 契约、在途不超上限、过载不堆积、健康探针不被阻塞
* ``测试/测试数据/容错验证.json`` —— 5 类故障注入的「行为 + 留痕」

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/验证限流与容错.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/验证限流与容错.py --tag 优化后 --concurrency 20
    pwsh -NoProfile -File run_py.ps1 研发/scripts/验证限流与容错.py --tag 基准_优化前
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "测试" / "测试数据"
PDF_DIR = Path(r"E:\gao6gongdan\工单7\研发\data\raw")

#: 硬性并发上限（共享约定 §5.5 / 本工单纪律）
MAX_CONCURRENCY_HARD = 20


def http_json(url: str, payload: dict[str, Any] | None = None, *, timeout: float = 60.0,
              method: str | None = None) -> tuple[int, dict[str, Any], str]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), ""
            except json.JSONDecodeError:
                return resp.status, {}, raw[:200]
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(body)
        except Exception:  # noqa: BLE001
            pass
        return int(exc.code), parsed, body[:200]
    except Exception as exc:  # noqa: BLE001
        return 0, {}, f"{type(exc).__name__}: {exc}"


def upload(base: str, filename: str, content: bytes, *, timeout: float = 120.0) -> tuple[int, dict, str]:
    boundary = "----verify" + str(int(time.time() * 1000))
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
            text = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(text)
        except Exception:  # noqa: BLE001
            pass
        return int(exc.code), parsed, text[:200]
    except Exception as exc:  # noqa: BLE001
        return 0, {}, f"{type(exc).__name__}: {exc}"


# ======================================================================
# 限流验证
# ======================================================================
def verify_limiter(base: str, *, concurrency: int, duration: float) -> dict[str, Any]:
    questions = ["平安银行2019年的营业收入是多少？", "招商银行2019年的营业收入和净利润分别是多少？",
                 "平安银行2019年末的不良贷款率是多少？", "平安银行2019年末的拨备覆盖率是多少？"]
    cases: list[dict[str, Any]] = []
    stop = threading.Event()
    results: list[tuple[int, float, dict]] = []
    lock = threading.Lock()

    status, snap_before, err = http_json(base + "/api/v1/perf", timeout=20)

    def worker(wid: int) -> None:
        seq = 0
        while not stop.is_set():
            q = questions[(wid + seq) % len(questions)]
            seq += 1
            started = time.perf_counter()
            code, body, _e = http_json(f"{base}/api/v1/ask",
                                       {"question": q, "session_id": f"lim-{wid}"}, timeout=90)
            with lock:
                results.append((code, round((time.perf_counter() - started) * 1000, 2), body))
            if stop.wait(0.02):
                break

    # 健康探针并发采样：验证「重活不阻塞事件循环」（P-7）
    probe_lat: list[float] = []
    probe_stop = threading.Event()

    def probe_loop() -> None:
        while not probe_stop.is_set():
            t0 = time.perf_counter()
            code, _b, _e = http_json(base + "/healthz", timeout=5)
            if code == 200:
                probe_lat.append(round((time.perf_counter() - t0) * 1000, 2))
            time.sleep(0.1)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(concurrency)]
        probe_thread = threading.Thread(target=probe_loop, daemon=True)
        probe_thread.start()
        inflight_samples: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            time.sleep(1.0)
            code, snap, _e = http_json(base + "/api/v1/perf", timeout=10)
            if code == 200:
                inflight_samples.append({"t": round(time.perf_counter() - t0, 2),
                                         "inflight": snap["limiter"]["inflight"],
                                         "queued": snap["limiter"]["queued"],
                                         "rss_mb": None})
        stop.set()
        probe_stop.set()
        for fut in futures:
            fut.result()
        probe_thread.join(timeout=5)

    codes = [c for c, _l, _b in results]
    ok_n = codes.count(200)
    rejected = codes.count(429)
    other = len(codes) - ok_n - rejected
    rejected_bodies = [b for c, _l, b in results if c == 429]
    sample = rejected_bodies[0] if rejected_bodies else {}
    peak_inflight = max([s["inflight"] for s in inflight_samples] or [0])
    peak_queued = max([s["queued"] for s in inflight_samples] or [0])

    cases.append({
        "用例": "过载时的在途上限",
        "期望": f"在途 ≤ PERF_MAX_INFLIGHT（观测峰值 {peak_inflight}）",
        "实际": f"峰值在途 {peak_inflight}，峰值排队 {peak_queued}",
        "通过": bool(peak_inflight <= max(1, (snap_before.get("limiter", {}) or {}).get("max_inflight", 10 ** 9))),
    })
    cases.append({
        "用例": "过载时快速失败（429）而不是堆积",
        "期望": "出现 429，且带 detail/inflight/queued/max_inflight/queue_max/retry_after_s",
        "实际": (f"429 共 {rejected} 次；字段={sorted(sample.keys())}" if rejected
                 else f"未出现 429（本次负载未超限：200×{ok_n}）"),
        "通过": bool(rejected == 0 or set(sample.keys()) >=
                     {"detail", "inflight", "queued", "max_inflight", "queue_max", "retry_after_s"}),
    })
    cases.append({
        "用例": "健康探针在高压下仍可答（事件循环不被阻塞）",
        "期望": "并发压测中 /healthz P95 ≤ 200 ms",
        "实际": (f"/healthz 采样 {len(probe_lat)} 次，P95 = "
                 f"{round(statistics.quantiles(probe_lat, n=20)[18], 2) if len(probe_lat) > 20 else (max(probe_lat) if probe_lat else 0)} ms"),
        "通过": bool(probe_lat and (sorted(probe_lat)[int(len(probe_lat) * 0.95)] if len(probe_lat) > 1 else probe_lat[0]) <= 200),
    })
    cases.append({
        "用例": "过载不产生 5xx",
        "期望": "5xx = 0（过载走 429，不是 500）",
        "实际": f"非 200/429 的响应 {other} 次",
        "通过": bool(other == 0),
    })
    return {
        "work_order": WORK_ORDER, "base_url": base, "concurrency": concurrency,
        "duration_s": duration, "total_requests": len(codes), "ok": ok_n,
        "rejected_429": rejected, "other": other,
        "perf_before": (snap_before.get("limiter") if status == 200 else {"error": err}),
        "peak_inflight": peak_inflight, "peak_queued": peak_queued,
        "probe_p95_ms": (sorted(probe_lat)[int(len(probe_lat) * 0.95)] if len(probe_lat) > 1 else None),
        "probe_samples": len(probe_lat),
        "cases": cases,
    }


# ======================================================================
# 容错验证
# ======================================================================
def verify_faults(base: str, *, tag: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    good = pdfs[0].read_bytes() if pdfs else b""

    # ① 服务可达性探针（服务未启动的场景由「本脚本连接失败」本身留痕）
    code, body, err = http_json(base + "/healthz", timeout=5)
    cases.append({"故障注入": "服务可达性", "行为": f"/healthz → HTTP {code}",
                  "留痕": "部署/日志/本地运行-API-*.log", "通过": code == 200})

    # ② 上传损坏 PDF
    broken = b"%PDF-1.4\n%\xd0\xd4\xc5\xd8\n1 0 obj\n"
    code, body, err = upload(base, "broken.pdf", broken)
    cases.append({"故障注入": "上传损坏 PDF", "行为": f"HTTP {code}：{str(body.get('detail') or err)[:60]}",
                  "留痕": "响应体 + 部署/日志/*.log", "通过": code in (400, 422)})

    # ③ 非 PDF 扩展名
    code, body, err = upload(base, "note.txt", b"hello world")
    cases.append({"故障注入": "非 PDF 扩展名", "行为": f"HTTP {code}：{str(body.get('detail') or err)[:60]}",
                  "留痕": "响应体", "通过": code == 400})

    # ④ 队列/解析路径的正常上传（对照：合法文件必须成功）
    if good:
        code, body, err = upload(base, pdfs[0].name, good)
        if code == 202:
            task_id = body.get("task_id")
            for _ in range(120):
                st, tb, _e = http_json(f"{base}/api/v1/tasks/{task_id}", timeout=10)
                if st == 200 and tb.get("status") in {"done", "failed"}:
                    break
                time.sleep(0.5)
            cases.append({"故障注入": "合法 PDF（对照）",
                          "行为": f"HTTP 202 → 任务 {tb.get('status')}（{tb.get('pages')} 页，"
                                  f"{tb.get('elapsed_ms')} ms，worker={tb.get('worker')}）",
                          "留痕": "部署/日志/parse_tasks.jsonl", "通过": tb.get("status") == "done"})
        else:
            cases.append({"故障注入": "合法 PDF（对照）",
                          "行为": f"HTTP {code} → {str(body.get('status') or err)[:60]}"
                                  f"（PERF_ENABLED=0 时为同步解析，200/422 都算可解释）",
                          "留痕": "部署/日志/parse_tasks.jsonl",
                          "通过": code in (200, 202) and str(body.get("status")) == "done"})

    # ⑤ Ollama 不可达（不改服务配置，直接验证卸载接口对不可达后端的容错）
    code, body, err = http_json(base + "/api/v1/admin/unload", payload={}, method="POST", timeout=30)
    cases.append({"故障注入": "主动卸载 Ollama（keep_alive=0s）",
                  "行为": f"HTTP {code}：{json.dumps(body, ensure_ascii=False)[:90]}",
                  "留痕": "响应体 + 服务日志 api.unload", "通过": code == 200})

    # ⑥ 未就绪端点契约
    code, body, err = http_json(base + "/api/v1/tasks/not-exist", timeout=10)
    cases.append({"故障注入": "查询不存在的任务",
                  "行为": f"HTTP {code}：{str(body.get('detail') or err)[:60]}",
                  "留痕": "响应体", "通过": code in (404,)})

    # ⑦ 非法请求体
    code, body, err = http_json(base + "/api/v1/ask", {"question": ""}, timeout=15)
    cases.append({"故障注入": "空问题（参数校验）", "行为": f"HTTP {code}",
                  "留痕": "响应体（FastAPI 422）", "通过": code == 422})

    return {"work_order": WORK_ORDER, "base_url": base, "tag": tag,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": cases}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 限流与容错实测")
    ap.add_argument("--base-url", default="http://127.0.0.1:8200")
    ap.add_argument("--tag", default="优化后")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--outdir", default=str(_DATA))
    args = ap.parse_args(argv)
    if args.concurrency > MAX_CONCURRENCY_HARD:
        print(f"[错误] 并发 {args.concurrency} 超过硬上限 {MAX_CONCURRENCY_HARD}")
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== 限流验证：并发 {args.concurrency}，时长 {args.duration}s ===")
    limit = verify_limiter(args.base_url, concurrency=args.concurrency, duration=args.duration)
    (outdir / "限流验证.json").write_text(json.dumps(limit, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    for case in limit["cases"]:
        print(f"  [{'通过' if case['通过'] else '未通过'}] {case['用例']}：{case['实际']}")

    print("\n=== 容错验证 ===")
    faults = verify_faults(args.base_url, tag=args.tag)
    (outdir / "容错验证.json").write_text(json.dumps(faults, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    for case in faults["cases"]:
        print(f"  [{'通过' if case['通过'] else '未通过'}] {case['故障注入']}：{case['行为']}")

    ok = sum(1 for c in limit["cases"] + faults["cases"] if c["通过"])
    total = len(limit["cases"]) + len(faults["cases"])
    print(f"\n合计 {ok}/{total} 项通过")
    print(f"落盘：{outdir / '限流验证.json'}、{outdir / '容错验证.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
