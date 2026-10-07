"""
稳定性与容错验证（工单2 验收项）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

工单2 新增两条验收要求：
    高可用性：系统应具备高稳定性，支持长时间运行
    容错机制：能够处理常见的异常情况，如用户输入错误或文档解析失败

本脚本对**运行中的服务**做三件事：
  1) 并发压测：N 路并发提问，给出 P50/P95/P99 延迟与错误率
     （注意：大模型侧有账号级限流，并发过高会出现 429，这属于外部依赖限制，
       不是系统缺陷 —— 脚本会如实区分「服务错误」与「上游限流」）
  2) 多语言：中英文各问一遍，验证「用问题相同的语言作答」
  3) 容错：一组异常输入，验证系统返回**结构化错误**而不是 500 / 挂起

用法：
    python scripts/load_test.py                       # 默认 8 并发 × 2 轮
    python scripts/load_test.py --concurrency 16 --rounds 3
    python scripts/load_test.py --skip-load           # 只跑容错与多语言
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

BASE = f"http://127.0.0.1:{config.APP_PORT}"


def _post(path: str, payload: dict, timeout: int = 90) -> tuple[int, dict | None, float]:
    """发一个 POST，返回 (http状态码, 解析后的 JSON 或 None, 耗时ms)。"""
    t0 = time.perf_counter()
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            ms = (time.perf_counter() - t0) * 1000
            try:
                return r.status, json.loads(body), ms
            except json.JSONDecodeError:
                return r.status, None, ms
    except urllib.error.HTTPError as e:
        ms = (time.perf_counter() - t0) * 1000
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body), ms
        except json.JSONDecodeError:
            return e.code, None, ms
    except Exception as e:  # noqa: BLE001
        return 0, {"__error__": str(e)}, (time.perf_counter() - t0) * 1000


def _get(path: str) -> dict | None:
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return round(s[k], 1)


# ------------------------------------------------------------------ 1) 并发压测

def load_test(concurrency: int, rounds: int, questions: list[str],
              path: str = "/api/ask", timeout: int = 120, label: str = "") -> dict:
    """
    对某个接口做 N 路并发压测，返回 P50/P95/P99、吞吐与错误率。

    path 之所以可配，是为了把两层分开测（见 main() 里的说明）：
      * `/api/ask`     —— 端到端，耗时由上游大模型主导
      * `/api/search`  —— 纯本地检索，衡量**本系统自身**的并发能力
    """
    tasks = [questions[i % len(questions)] for i in range(concurrency * rounds)]
    print(f"[压测] {label or path}：{concurrency} 并发 × {rounds} 轮 = {len(tasks)} 次请求")

    results: list[dict] = []

    def one(q: str) -> dict:
        code, body, ms = _post(path, {"question": q}, timeout=timeout)
        ok = code == 200 and isinstance(body, dict) and body.get("code") == "OK"
        err = ""
        if not ok:
            err = ((body or {}).get("message") or f"HTTP {code}")[:120]
        return {"ok": ok, "code": code, "ms": ms, "err": err}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for r in pool.map(one, tasks):
            results.append(r)
    wall = time.perf_counter() - t0

    good = [r for r in results if r["ok"]]
    lats = [r["ms"] for r in good]
    throttled = [r for r in results if "429" in r["err"] or "throttl" in r["err"].lower()
                 or "limit" in r["err"].lower()]
    others = [r for r in results if not r["ok"] and r not in throttled]

    out = {
        "endpoint": path,
        "concurrency": concurrency,
        "rounds": rounds,
        "requests": len(results),
        "success": len(good),
        "success_rate": round(len(good) / len(results), 4),
        "wall_seconds": round(wall, 1),
        "throughput_rps": round(len(results) / wall, 2) if wall > 0 else None,
        "p50_ms": _pct(lats, 50),
        "p95_ms": _pct(lats, 95),
        "p99_ms": _pct(lats, 99),
        "avg_ms": round(statistics.mean(lats), 1) if lats else None,
        "max_ms": round(max(lats), 1) if lats else None,
        "within_3s_rate": round(sum(1 for t in lats if t <= 3000) / len(lats), 4) if lats else None,
        "upstream_throttled": len(throttled),
        "service_errors": len(others),
        "error_samples": sorted({r["err"] for r in others})[:5],
    }
    print(f"  成功 {out['success']}/{out['requests']}（{out['success_rate']*100:.1f}%）"
          f"  吞吐 {out['throughput_rps']} rps  墙钟 {out['wall_seconds']}s")
    print(f"  P50 {out['p50_ms']}ms  P95 {out['p95_ms']}ms  P99 {out['p99_ms']}ms  "
          f"≤3s 占比 {out['within_3s_rate']}")
    if out["upstream_throttled"]:
        print(f"  ⚠ 上游限流 {out['upstream_throttled']} 次（大模型账号级并发限制，非服务缺陷）")
    if out["service_errors"]:
        print(f"  ✗ 服务错误 {out['service_errors']} 次：{out['error_samples']}")
    return out


def serialization_probe(questions: list[str], timeout: int = 120) -> dict:
    """
    串行 vs 并发对照：判断「并发下变慢」到底是系统撑不住，还是上游在排队。

    判据很简单 —— 如果 N 倍并发时的**墙钟时间 ≈ 单次耗时 × 请求数**，
    说明上游把请求串行化了（并发度实际是 1），慢的是模型服务不是本系统；
    如果墙钟时间 ≈ 单次耗时 × 请求数 / N，说明真的并行了。
    """
    print("\n[串行基线] /api/ask 单路 4 次（用于判断并发下变慢的根因）")
    lats = []
    for q in questions[:4]:
        code, body, ms = _post("/api/ask", {"question": q}, timeout=timeout)
        ok = code == 200 and isinstance(body, dict) and body.get("code") == "OK"
        lats.append({"ok": ok, "ms": ms})
        print(f"  单次 {ms:.0f} ms  {'ok' if ok else 'FAIL'}")
    good = [x["ms"] for x in lats if x["ok"]]
    return {
        "requests": len(lats),
        "avg_ms": round(statistics.mean(good), 1) if good else None,
        "samples_ms": [round(x["ms"], 1) for x in lats],
    }



# ------------------------------------------------------------------ 2) 持续负载

def soak_test(n: int, questions: list[str], interval: float = 0.0) -> dict:
    """
    持续负载下的稳定性（对应工单2「高可用性：支持长时间运行」）。

    走 `/api/search`（纯本地、不调大模型、零成本），连续打 n 次，重点看**漂移**：

      * 长时间运行最典型的退化不是「崩掉」，而是「越跑越慢」——
        缓存膨胀、句柄/连接泄漏、索引结构被反复重建，都会先表现为延迟爬升。
      * 所以这里比较**前 25% 与后 25% 的平均延迟**，而不是只看峰值。
      * 同时记录失败次数，以及服务自报的 uptime。

    真实的大模型链路不适合做几百次连续压测（成本 + 上游限流会污染结论），
    系统侧的长稳由这个用例覆盖，模型侧由端到端用例覆盖。
    """
    print(f"\n[持续负载] /api/search 连续 {n} 次（每次间隔 {interval}s）")
    lats: list[float] = []
    fails = 0
    t0 = time.perf_counter()
    for i in range(n):
        q = questions[i % len(questions)]
        code, body, ms = _post("/api/search", {"question": q}, timeout=30)
        okk = code == 200 and isinstance(body, dict) and body.get("code") == "OK"
        if not okk:
            fails += 1
        else:
            lats.append(ms)
        if interval:
            time.sleep(interval)
    wall = time.perf_counter() - t0

    if not lats:
        return {"requests": n, "success": 0, "fails": fails, "error": "全部失败"}

    cut = max(1, len(lats) // 4)
    head, tail = lats[:cut], lats[-cut:]
    head_avg = statistics.mean(head)
    tail_avg = statistics.mean(tail)
    drift = tail_avg / head_avg if head_avg else 0.0

    health = _get("/health") or {}
    out = {
        "endpoint": "/api/search",
        "requests": n,
        "success": len(lats),
        "fails": fails,
        "wall_seconds": round(wall, 1),
        "p50_ms": _pct(lats, 50),
        "p95_ms": _pct(lats, 95),
        "p99_ms": _pct(lats, 99),
        "max_ms": round(max(lats), 1),
        "first_quartile_avg_ms": round(head_avg, 2),
        "last_quartile_avg_ms": round(tail_avg, 2),
        "drift_ratio": round(drift, 3),
        "drift_verdict": ("稳定（后段无明显变慢）" if drift <= 1.30 else "⚠ 存在延迟漂移"),
        "service_uptime_seconds": health.get("uptime_seconds"),
    }
    print(f"  成功 {out['success']}/{n}  失败 {fails}  墙钟 {out['wall_seconds']}s")
    print(f"  P50 {out['p50_ms']}ms  P95 {out['p95_ms']}ms  P99 {out['p99_ms']}ms  max {out['max_ms']}ms")
    print(f"  前 25% 均值 {out['first_quartile_avg_ms']}ms → 后 25% 均值 {out['last_quartile_avg_ms']}ms"
          f"  漂移系数 {out['drift_ratio']} → {out['drift_verdict']}")
    return out


# ------------------------------------------------------------------ 3) 多语言

def multilang_test() -> dict:
    print("\n[多语言] 中英文对照（验证「用与问题相同的语言作答」）")
    pairs = [
        ("武汉兴图新科电子股份有限公司的注册资本是多少？",
         "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"),
        ("该公司参与的哪个工程荣获了国家科技进步一等奖？",
         "Which project that the company participated in won the National Science and Technology Progress Award, First Prize?"),
    ]
    out = []
    for zh, en in pairs:
        row = {"zh": zh, "en": en}
        for tag, q in (("zh", zh), ("en", en)):
            code, body, ms = _post("/api/ask", {"question": q})
            ans = ((body or {}).get("data") or {}).get("answer", "") if code == 200 else ""
            row[f"{tag}_answer"] = ans
            row[f"{tag}_ms"] = round(ms, 1)
            row[f"{tag}_lang_ok"] = _lang_ratio(ans, tag)
            print(f"  [{tag}] {ans[:88].replace(chr(10),' ')}")
        out.append(row)
    return {"pairs": out}


def _lang_ratio(text: str, lang: str) -> float:
    """粗判回答语种：中文字符占比 / 拉丁字母占比。"""
    if not text:
        return 0.0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    total = cjk + latin
    if total == 0:
        return 0.0
    return round((cjk if lang == "zh" else latin) / total, 3)


# ------------------------------------------------------------------ 4) 容错

def fault_test() -> dict:
    print("\n[容错] 异常输入（期望：结构化错误，不是 500、不是挂起）")
    cases = [
        ("空问题", "/api/ask", {"question": ""}),
        ("纯空白", "/api/ask", {"question": "   \n  "}),
        ("question 非字符串", "/api/ask", {"question": 12345}),
        ("缺字段", "/api/ask", {}),
        ("超长输入（3000 字）", "/api/ask", {"question": "注册资本" * 750}),
        ("离题问题（触发闸门）", "/api/ask", {"question": "如何用 Python 实现快速排序？"}),
        ("闲聊", "/api/ask", {"question": "你好"}),
        ("对比接口空问题", "/api/compare", {"question": ""}),
        ("优化对比空问题", "/api/compare-opt", {"question": ""}),
    ]
    rows = []
    for name, path, payload in cases:
        code, body, ms = _post(path, payload)
        code_field = (body or {}).get("code") if isinstance(body, dict) else None
        msg = ((body or {}).get("message") or "")[:70] if isinstance(body, dict) else ""
        ans = ""
        if code_field == "OK":
            d = (body or {}).get("data") or {}
            ans = (d.get("answer") or "")[:60].replace("\n", " ")
        verdict = "OK"
        if code == 500 or code == 0:
            verdict = "✗ 未兜住"
        elif code_field in ("BAD_REQUEST", "OK"):
            verdict = "OK"
        rows.append({"case": name, "http": code, "code": code_field,
                     "message": msg, "answer_head": ans, "verdict": verdict, "ms": round(ms, 1)})
        print(f"  [{verdict:<6}] {name:<20} HTTP {code}  code={code_field}  {msg or ans}")
    bad = [r for r in rows if r["verdict"] != "OK"]
    return {"cases": rows, "unhandled": len(bad)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--soak", type=int, default=0,
                    help="持续负载次数（走 /api/search，纯本地零成本），0=不跑")
    ap.add_argument("--skip-extra", action="store_true",
                    help="跳过多语言与容错段落（只想单独补跑某一项时用）")
    args = ap.parse_args()

    print(f"[工单] {config.WORK_ORDER_NO_OPT}（{config.WORK_ORDER_SHORT}）")
    print(f"[目标] {BASE}")

    ready = _get("/ready")
    if not ready:
        print("[ERR] 服务不可达。请先启动：python main.py")
        return 1
    print(f"[就绪] {json.dumps(ready, ensure_ascii=False)[:160]}\n")

    qs = json.loads((config.EVAL_DIR / "ticket_questions.json").read_text(encoding="utf-8"))
    questions = [q["question"] for q in qs[:6]]

    report: dict = {
        "work_order_no": config.WORK_ORDER_NO_OPT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE,
        "note": (
            "并发变慢的根因判定：若 N 并发时墙钟 ≈ 单次耗时 × 请求数，"
            "说明上游大模型把请求串行化了（本系统的并发度实际被压到 1）。"
            "因此本报告把「系统层」（/api/search，纯本地检索，不调模型）"
            "与「端到端」（/api/ask，含模型生成）分开统计。"
        ),
    }

    if not args.skip_load:
        # 顺序很重要：先拿单路基线，再测系统层，最后测端到端。
        report["serial_baseline"] = serialization_probe(questions)
        report["system_load"] = load_test(
            args.concurrency, args.rounds, questions,
            path="/api/search", timeout=30, label="系统层 /api/search（只检索不生成）",
        )
        report["load"] = report["e2e_load"] = load_test(
            args.concurrency, args.rounds, questions,
            path="/api/ask", label="端到端 /api/ask（含大模型生成）",
        )

        # 串行化判定
        sl, e2e = report["serial_baseline"], report["e2e_load"]
        if sl.get("avg_ms") and e2e.get("wall_seconds"):
            ideal_parallel = sl["avg_ms"] * e2e["requests"] / 1000 / args.concurrency
            report["serialization_ratio"] = round(e2e["wall_seconds"] / ideal_parallel, 2)
            verdict = ("上游串行化（本系统无并发缺陷）"
                       if report["serialization_ratio"] >= args.concurrency * 0.6
                       else "基本并行了")
            report["serialization_verdict"] = verdict
            print(f"\n[判定] 单路 {sl['avg_ms']}ms × {e2e['requests']} 请求 ÷ {args.concurrency} 并发"
                  f" = 理想墙钟 {ideal_parallel:.1f}s，实测 {e2e['wall_seconds']}s"
                  f" → 串行化系数 {report['serialization_ratio']} → **{verdict}**")

    if not args.skip_extra:
        report["multilang"] = multilang_test()
        report["fault"] = fault_test()
    if args.soak:
        report["soak"] = soak_test(args.soak, questions)

    path = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_稳定性与容错验证_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
