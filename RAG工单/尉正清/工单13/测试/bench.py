# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""串行基准测试：量单请求的延迟，并给出分阶段占比

和 JMeter 的分工：
  · **本脚本**量「一个请求花在哪」，串行发、不互相干扰，看的是**延迟构成**
  · **JMeter** 量「并发上去了会怎样」，看的是**吞吐与尾延迟**

两者缺一不可：只压测不知道时间花在哪，只profiling不知道负载下会怎样。

用法：
    D:/Anaconda/envs/rag_gd/python.exe bench.py [--n 20] [--warmup 3] [--tag before]
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "研发"
sys.path.insert(0, str(DEV))

URL = "http://127.0.0.1:8000/query"

# 覆盖不同题型：事实型 / 跨文档 / 分析型。题型不同，召回路数与上下文长度都不同
QUESTIONS = [
    "平安银行2019年末的拨备覆盖率是多少？",
    "招商银行2020年的营业收入是多少？",
    "中国平安2019年的归母净利润同比增速是多少？",
    "中信证券2020年末的总资产规模是多少？",
    "邮储银行2019年的不良贷款率是多少？",
    "中国人寿2020年的保费收入是多少？",
    "中国太保2021年的综合偿付能力充足率是多少？",
    "招商证券2021年的营业收入是多少？",
    "国泰君安2021年的归母净利润是多少？",
    "比较平安银行和招商银行2019年的净息差",
]


def ask(question, rid=None, timeout=180):
    body = json.dumps({"question": question, "request_id": rid}).encode("utf-8")
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out, time.perf_counter() - t


def pct(vals, p):
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(round((p / 100) * len(vals) + 0.5)) - 1))
    return vals[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="测量请求数")
    ap.add_argument("--warmup", type=int, default=3, help="预热请求数（不计入统计）")
    ap.add_argument("--tag", default="run", help="标记本轮，写进结果文件名")
    args = ap.parse_args()

    print(f"预热 {args.warmup} 次 ...", flush=True)
    for i in range(args.warmup):
        try:
            ask(QUESTIONS[i % len(QUESTIONS)], f"warm-{args.tag}-{i}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  预热失败：{exc}")

    print(f"测量 {args.n} 次 ...", flush=True)
    wall, retr, stages, ok = [], [], {}, 0
    for i in range(args.n):
        q = QUESTIONS[i % len(QUESTIONS)]
        try:
            out, w = ask(q, f"bench-{args.tag}-{i}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  [{i+1}] 失败：{exc}")
            continue
        if out.get("status") != "ok":
            continue
        ok += 1
        wall.append(w)
        retr.append(out.get("retrieve_seconds", 0.0))
        for k, v in out.get("stages", {}).items():
            stages.setdefault(k, []).append(v)
        print(f"  [{i+1:>2}/{args.n}] 端到端 {w:5.2f}s | 检索 "
              f"{out.get('retrieve_seconds', 0):5.2f}s | 召回 {out.get('recall_count')}",
              flush=True)

    if not wall:
        print("没有成功的请求")
        return 1

    result = {
        "tag": args.tag, "n": ok,
        "wall": {"mean": statistics.mean(wall), "median": statistics.median(wall),
                 "p95": pct(wall, 95), "min": min(wall), "max": max(wall)},
        "retrieve": {"mean": statistics.mean(retr), "median": statistics.median(retr),
                     "p95": pct(retr, 95)},
        "stages": {k: {"mean": statistics.mean(v), "p95": pct(v, 95)}
                   for k, v in stages.items()},
    }
    print("\n===== 汇总 =====")
    w = result["wall"]
    print(f"  端到端  均值 {w['mean']:.2f}s  中位 {w['median']:.2f}s  "
          f"P95 {w['p95']:.2f}s  最小 {w['min']:.2f}s  最大 {w['max']:.2f}s")
    print(f"  检索段  均值 {result['retrieve']['mean']:.2f}s  "
          f"P95 {result['retrieve']['p95']:.2f}s")
    print("  各阶段均值（占端到端比例）:")
    for k in sorted(result["stages"]):
        m = result["stages"][k]["mean"]
        print(f"    {k:18} {m:6.3f}s  {m / w['mean'] * 100:5.1f}%")

    out_file = HERE / "results" / f"bench_{args.tag}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n已写入 {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
