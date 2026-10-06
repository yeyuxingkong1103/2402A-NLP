# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化
# 模块：evaluation/coldstart_bench —— 冷启动 / 预热 / 缓存 对“首问时延”的影响（工单 13）
# 说明：① 先把 Ollama 上的 bge-m3 卸载（keep_alive=0）制造冷态；② 测“冷态首问”；
#       ③ 调用 llm.warmup()；④ 测“热态首问”；⑤ 再测“命中缓存的同问”。
# 用法：python evaluation/coldstart_bench.py
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import engine        # noqa: E402
import kb as kb_mod  # noqa: E402
import llm           # noqa: E402
from config import OLLAMA_BASE, EMBED_MODEL  # noqa: E402

Q = "武汉兴图新科电子股份有限公司注册资本是多少？"
Q2 = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"


def unload(model):
    data = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_BASE + "/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except Exception as e:  # noqa: BLE001
        print("unload warn:", e)
        return False


def timed_retrieve(q, label):
    t0 = time.perf_counter()
    hits, conf = engine.retrieve(q, top_k=5)
    ms = (time.perf_counter() - t0) * 1000
    print("  %-22s %8.0f ms   top1=p%-3s conf=%.3f" % (label, ms, hits[0]["page"] if hits else "-", conf))
    return ms


def main():
    kb_mod.get_kb()
    res = {}
    print("1) 制造冷态（卸载 %s）..." % EMBED_MODEL)
    unload(EMBED_MODEL)
    time.sleep(1)
    res["cold_first"] = timed_retrieve(Q, "冷态首问")

    print("2) 启动预热 llm.warmup() ...")
    t0 = time.perf_counter()
    info = llm.warmup(gen=True)
    res["warmup_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print("   warmup:", info)

    res["warm_first"] = timed_retrieve(Q2, "预热后首问(新问题)")
    res["cache_hit"] = timed_retrieve(Q2, "同问（命中缓存）")
    res["warm_no_cache"] = timed_retrieve(Q, "预热后又一新问题")
    res["cache_stat"] = llm.cache_stat()

    print("\n==== 结果 ====")
    print("  冷态首问        : %.0f ms" % res["cold_first"])
    print("  预热耗时(一次)  : %.0f ms" % res["warmup_ms"])
    print("  预热后首问(新题): %.0f ms" % res["warm_first"])
    print("  又一新问题      : %.0f ms" % res["warm_no_cache"])
    print("  同问命中缓存    : %.0f ms" % res["cache_hit"])
    print("  缓存统计        :", res["cache_stat"])
    json.dump(res, open(os.path.join(HERE, "冷启动对比结果.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved -> evaluation/冷启动对比结果.json")


if __name__ == "__main__":
    main()
