# 工单编号：人工智能NLP-RAG-混合检索任务
"""PDF 逐项验收的**加压复核**（把三项证据偏弱的检查用正确对照重做）。

第一版探针的教训（已修正）：
  · 模糊匹配对照组选了"注删资本"——无 `~` 时 jieba 会把它切成 [注删, 资本]，
    照样命中"注册资本"的文档，对照组本身失效 → 改用**整词错别字**并加否定对照；
  · 权重对照只看 rag.ask 的 Top-1 —— 精排会把两条路的差异抹平 →
    改为**关掉精排**直接比较融合候选序（strategies.retrieve(do_rerank=False)）。

用法：python scripts/acceptance_pdf_probe.py
产出：logs/acceptance_probe.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import bootstrap  # noqa: E402

OUT = os.path.join("logs", "acceptance_probe.json")


def _clean_probe_feedback() -> dict:
    """清掉验收脚本写入的探针反馈（保持反馈库干净）。"""
    from src import rerankers
    path = rerankers._feedback_path()
    if not os.path.isfile(path):
        return {"removed": 0, "reason": "no file"}
    rows, removed = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                rows.append(line)
                continue
            if row.get("chunk_id") == "acceptance#probe":
                removed += 1
            else:
                rows.append(line)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + ("\n" if rows else ""))
    rerankers.load_feedback(force=True)
    return {"removed": removed, "remaining": rerankers.feedback_stats()}


def probe_fuzzy() -> dict:
    """整词错别字对照：'法定代理人'（错）→ 应为 0 命中；加 `~` 应命中'法定代表人'。

    另加否定对照：`zzzqqq~` 必须 0 命中，证明 `~` 不是"匹配一切"的万能符。
    """
    from src import fulltext

    def one(q: str) -> dict:
        hits = fulltext.search(q, top_k=5)
        texts = " ".join((h.get("text") or "") for h in hits)
        return {"query": q, "n": len(hits),
                "top1": (hits[0].get("chunk_id") if hits else None),
                "hits_contain_target": "法定代表人" in texts if hits else False}

    res = {
        "typo_plain": one("法定代理人"),        # 错别字，无 ~
        "typo_fuzzy": one("法定代理人~"),       # 错别字 + ~（编辑距离 1）
        "exact": one("法定代表人"),             # 正确词做基准
        "nonsense_fuzzy": one("zzzqqq~"),       # 否定对照：必须 0
        "exact2": one("注册资本"),
        "typo2_fuzzy": one("注删资本~"),
    }
    res["fuzzy_effective"] = (
        res["typo_fuzzy"]["n"] > 0
        and res["typo_fuzzy"]["hits_contain_target"]
        and res["nonsense_fuzzy"]["n"] == 0
    )
    return res


def probe_weights() -> dict:
    """关掉精排，比较不同权重下**融合候选序**是否真的不同。"""
    from src import strategies, runtime

    q = "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"
    out: dict = {}
    for vw, fw in ((0.9, 0.1), (0.5, 0.5), (0.1, 0.9)):
        cfg = strategies.with_overrides(
            runtime.get(), strategy="hybrid", fusion="weighted",
            vector_weight=vw, fulltext_weight=fw, rerank_enabled=False)
        res = strategies.retrieve(q, cfg, do_rerank=False)
        hits = res.get("hits") or []
        out[f"v{vw}_f{fw}"] = {
            "weights": res.get("info", {}).get("weights"),
            "top10": [h.get("chunk_id") for h in hits][:10],
            "top1": (hits[0].get("chunk_id") if hits else None),
        }
    tops = {v["top1"] for v in out.values()}
    out["distinct_top1"] = len(tops)
    out["weights_effective"] = len(tops) > 1
    return out


def probe_english() -> dict:
    """换一道答案含成句散文的问题，验证英文提问真的得到英文答案。"""
    from src import rag, runtime
    runtime.update(strategy="hybrid", rerank_algorithm="model")
    q = ("Which technical standard did Wuhan Xingtu Xinke Electronics "
         "participate in drafting?")
    r = rag.ask(q, use_cache=False)
    ans = r.answer or ""
    cjk = sum(1 for ch in ans if "一" <= ch <= "鿿")
    return {"question": q, "answer": ans[:300],
            "lang": (r.analysis or {}).get("lang"),
            "cjk_chars": cjk, "len": len(ans),
            "cjk_ratio": round(cjk / max(1, len(ans)), 3),
            "total_ms": (r.timings or {}).get("total_ms"),
            "english_prose": cjk / max(1, len(ans)) < 0.2}


def main() -> int:
    from src import rag
    rag.warmup()
    report = {
        "fuzzy": probe_fuzzy(),
        "weights": probe_weights(),
        "english": probe_english(),
        "feedback_cleanup": _clean_probe_feedback(),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("PROBE done ->", OUT)
    print("fuzzy_effective =", report["fuzzy"].get("fuzzy_effective"))
    print("weights_effective =", report["weights"].get("weights_effective"))
    print("english_prose =", report["english"].get("english_prose"))
    return 0


if __name__ == "__main__":
    raise SystemExit(bootstrap.run_with_large_stack(main))
