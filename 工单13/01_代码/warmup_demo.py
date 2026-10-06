# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化
# 模块：evaluation/warmup_demo —— 冷启动 vs 预热 vs 缓存 的量化对比（工单 13）
# 说明：① 先把 bge-m3 从 Ollama 卸下（模拟服务刚启动/模型被回收）→ 量“首问冷启动”；
#       ② 调 llm.warmup() 预热 → 再量“首问”；
#       ③ 同一问题重复检索 → 量嵌入缓存带来的收益。
# 用法：python evaluation/warmup_demo.py
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import engine       # noqa: E402
import llm          # noqa: E402
from config import OLLAMA_BASE, EMBED_MODEL  # noqa: E402

OUT = os.path.join(HERE, "warmup_result.json")
Q = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"


def unload(model):
    """把模型从 Ollama 卸载（keep_alive=0），用于复现冷启动。"""
    try:
        req = urllib.request.Request(OLLAMA_BASE + "/api/generate",
                                     data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except Exception as e:  # noqa: BLE001
        print("unload 失败:", e)
        return False


def t_embed(text, use_cache=True):
    t0 = time.perf_counter()
    llm.embed([text], use_cache=use_cache)
    return round((time.perf_counter() - t0) * 1000, 1)


def t_retrieve(q):
    t0 = time.perf_counter()
    hits, conf = engine.retrieve(q, top_k=5)
    return round((time.perf_counter() - t0) * 1000, 1), hits, conf


def main():
    res = {}
    print("① 卸载 bge-m3 模拟冷启动 …", unload(EMBED_MODEL))
    time.sleep(2)
    res["cold_first_embed_ms"] = t_embed("冷启动测试", use_cache=False)
    print("   冷启动首个嵌入：%.1f ms" % res["cold_first_embed_ms"])

    print("② 服务预热 llm.warmup() …")
    t0 = time.perf_counter()
    info = llm.warmup(gen=False)
    res["warmup_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    res["warmup_info"] = info
    print("   预热耗时 %.1f ms" % res["warmup_ms"])

    res["warm_first_embed_ms"] = t_embed("预热后首个嵌入", use_cache=False)
    print("   预热后首个嵌入：%.1f ms" % res["warm_first_embed_ms"])

    print("③ 检索：冷缓存 / 热缓存")
    llm._EMB_CACHE.clear()
    ms1, hits, conf = t_retrieve(Q)
    ms2, _, _ = t_retrieve(Q)
    res["retrieve_cold_cache_ms"] = ms1
    res["retrieve_warm_cache_ms"] = ms2
    res["top1_page"] = hits[0]["page"] if hits else None
    res["confidence"] = round(conf, 4)
    res["embed_cache"] = llm.cache_stat()
    print("   首次检索 %.1f ms / 再次检索（嵌入缓存命中）%.1f ms" % (ms1, ms2))
    print("   命中页 p%s conf=%.3f；缓存统计 %s" % (res["top1_page"], conf, res["embed_cache"]))

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n==== 汇总 ====")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("saved -> evaluation/warmup_result.json")


if __name__ == "__main__":
    main()
