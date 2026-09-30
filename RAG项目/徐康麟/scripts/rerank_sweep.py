#!/usr/bin/env python3
"""重排器扫描（P8）：预取候选 → 让它们打**同一批候选** → 出对比表 + 判定。

为什么一个编排脚本：换重排器是 P8 的高频小改动，必须能"一条命令跑完、全程留痕、结论可复核"。
留痕（都落 `eval/results/`）：
* ``rerankers-availability.json`` —— 每个候选是否下载成功、能否被 CrossEncoder 加载；
* ``retrieval-rerank.json`` —— 每个重排器的 Recall@5 / MRR / 零命中 / 逐题排名；
* 控制台打印对比表 + **按写死标准**的判定（见 ``docs/RERANK-EXPERIMENT.md``）。

判定（先写死，避免事后找理由）：
1. **P8 家族**（L07/L13/L22/L23/V01）里 ≥3 道把期望来源拉进前 5；
2. MRR 不降；3. 零命中不变长。达标者才值得再跑整轮 107 题确认答案侧不倒退。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
#: P8 家族（法条级精度长期不过的那几道 + 版本陷阱题）
P8_ITEMS = ("L07", "L13", "L22", "L23", "V01")
#: 现役重排器（对照）
INCUMBENT = "/root/autodl-tmp/modelscope/models/BAAI--bge-reranker-v2-m3/snapshots/master"


def _run(cmd: list[str], *, timeout: float = 3600) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, timeout=timeout).returncode


def _stop_services() -> None:
    """先停服务：Milvus Lite 是单进程独占（API 在跑时第二个进程打不开库），且省显存。"""
    print("=== 0) 停服务（API/vLLM/Redis）", flush=True)
    script = Path("/root/stop_cloud_all.sh")
    if script.is_file():
        subprocess.run(["bash", str(script)], timeout=600)
    else:
        print("  （云端没有 stop_cloud_all.sh，跳过；若 API 在跑请先手动停）")


def _fetch(models: list[str], out_root: str) -> list[dict]:
    print("=== 1) 预取候选重排器", flush=True)
    import fetch_rerankers  # noqa: PLC0415 - 同目录脚本

    results: list[dict] = []
    for model in models:
        print(f"--- {model}", flush=True)
        started = time.perf_counter()
        ok, detail = fetch_rerankers._download(model, out_root=out_root)
        entry = {"model": model, "downloaded": ok, "path": detail if ok else "",
                 "error": "" if ok else detail,
                 "seconds": round(time.perf_counter() - started, 1)}
        if ok and Path(detail).is_dir():
            loadable, why = fetch_rerankers._check_loadable(detail)
            entry["cross_encoder"] = loadable
            entry["load_detail"] = why
        print(f"    {'OK  ' if ok else 'FAIL'} {str(detail)[:150]}", flush=True)
        results.append(entry)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rerankers-availability.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _specs(availability: list[dict]) -> list[str]:
    specs = ["cosine="]
    if Path(INCUMBENT).is_dir():
        specs.append(f"m3={INCUMBENT}")
    for entry in availability:
        if entry.get("downloaded") and entry.get("cross_encoder") and entry.get("path"):
            short = entry["model"].split("/")[-1].replace("-", "").replace(".", "")[:14].lower()
            specs.append(f"{short}={entry['path']}")
    return specs


def _judge(report: dict, base_name: str) -> None:
    rankers = report.get("rankers") or {}
    base = rankers.get(base_name) or {}
    base_ranks = base.get("ranks") or {}
    print("\n=== 判定（按 docs/RERANK-EXPERIMENT.md 写死的标准）", flush=True)
    for name, summary in rankers.items():
        if name == base_name:
            continue
        ranks = summary.get("ranks") or {}
        wins = [item for item in P8_ITEMS
                if ranks.get(item) is not None and (ranks.get(item) or 99) <= 5
                and (base_ranks.get(item) is None or (base_ranks.get(item) or 99) > 5)]
        detail = " ".join(f"{item}:{base_ranks.get(item)}→{ranks.get(item)}" for item in P8_ITEMS)
        verdict = []
        if len(wins) >= 3:
            verdict.append(f"P8 转正 {len(wins)} 道 ✅")
        if summary.get("mrr", 0) < base.get("mrr", 0):
            verdict.append("MRR 下降 ❌")
        if len(summary.get("zero_hit") or []) > len(base.get("zero_hit") or []):
            verdict.append("零命中变长 ❌")
        print(f"  {name:<14} {detail}")
        print(f"                 → {'；'.join(verdict) or '未达门槛（P8 转正不足 3 道）'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="重排器扫描（P8）")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--out-root", default="/root/autodl-tmp/rerankers")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-stop", action="store_true")
    args = parser.parse_args()

    if not args.skip_stop:
        _stop_services()

    availability: list[dict] = []
    if args.skip_fetch:
        path = RESULTS / "rerankers-availability.json"
        availability = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    else:
        import fetch_rerankers  # noqa: PLC0415

        models = args.models or list(fetch_rerankers.DEFAULT_MODELS)
        import os

        os.environ["MODELSCOPE_CACHE"] = str(Path(args.out_root) / "modelscope")
        availability = _fetch(models, args.out_root)

    specs = _specs(availability)
    print(f"\n=== 2) 检索侧扫描：{len(specs)} 个重排器打同一批候选", flush=True)
    for spec in specs:
        print(f"    {spec}", flush=True)
    out = RESULTS / "retrieval-rerank.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_retrieval.py"),
           "--qa-file", str(ROOT / "eval" / "qa_set.jsonl"), "--role", "lawyer",
           "--limit", str(args.limit), "--out", str(out)]
    for spec in specs:
        cmd += ["--ranker", spec]
    code = _run(cmd, timeout=7200)
    if code != 0:
        print(f"!! eval_retrieval 退出码 {code}（看上面输出）", file=sys.stderr)
        return code

    report = json.loads(out.read_text(encoding="utf-8"))
    _judge(report, "cosine")
    print(f"\n留痕：{RESULTS / 'rerankers-availability.json'}")
    print(f"      {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
