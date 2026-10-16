# -*- coding: utf-8 -*-
"""工单13 完善 A3：Q10 单题跨模型对比（同题同提示，可直接判断能否换模型）。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

背景：优化后唯一超线题 Q10（3136.6 ms，检索仅 65 ms、生成 3062 ms）。
用途：在**同一道题、同一提示、同一采样参数**下比较 3 个候选生成模型，
      同时给出**答案关键词命中率**，避免「换个更快但答错的模型」这种假优化。

产出：``优化/评估结果/Q10跨模型对比.json``
用法：& $py 研发/scripts/对比Q10跨模型.py
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "优化" / "评估结果" / "Q10跨模型对比.json"
MODELS = ["qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:0.5b"]
SLA_MS = 3000

QUESTION = ("中信证券2020年年度报告中，公司全年营业收入、归属于公司股东净利润、"
            "年末总资产和净资产收益率分别是多少？")
REF_KW = ["营业收入", "净利润", "总资产", "净资产收益率"]
PROMPT = (f"请用中文回答，只给出结论不要解释。\n{QUESTION}\n答案：")


def kw_hit(pred: str) -> float:
    """参考关键词命中率（4 个指标名是否出现）。"""
    p = str(pred or "")
    return sum(1 for k in REF_KW if k in p) / len(REF_KW)


def call(model: str, prompt: str, timeout: int = 300) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
        "keep_alive": "0s",
    }).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        ms = (time.perf_counter() - t0) * 1000
        txt = str(d.get("response", "")).strip()
        return {"模型": model, "成功": True, "端到端ms": round(ms, 1),
                "输出字符数": len(txt), "关键词命中率": round(kw_hit(txt), 4),
                "是否达标SLA": ms <= SLA_MS, "回答摘录": txt[:160]}
    except Exception as e:
        return {"模型": model, "成功": False, "错误": f"{type(e).__name__}: {e}"}


def main() -> int:
    rows = [call(m, PROMPT) for m in MODELS]
    ok = [r for r in rows if r.get("成功")]

    # 只有「达标 且 关键词命中率不低于 3b」才算可替换
    base = next((r for r in ok if r["模型"] == "qwen2.5:3b"), None)
   建议 = []
    for r in ok:
        if r["模型"] == "qwen2.5:3b":
            continue
        if base and r["是否达标SLA"] and r["关键词命中率"] >= base["关键词命中率"]:
            建议.append(f"{r['模型']}：{r['端到端ms']:.0f} ms、关键词命中 "
                        f"{r['关键词命中率']:.0%}（基线 3b 为 {base['端到端ms']:.0f} ms / "
                        f"{base['关键词命中率']:.0%}）→ **可考虑替换**")
    payload = {
        "工单编号": "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化",
        "完善项": "A3 Q10 跨模型对比（同题同提示）",
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "口径": f"同一题干与提示、temperature=0、num_predict=256；SLA={SLA_MS} ms；"
                "答案质量用参考关键词命中率（营业收入/净利润/总资产/净资产收益率）",
        "SLA_ms": SLA_MS,
        "结果": rows,
        "可替换建议": 建议 or ["无模型同时满足「达标 SLA」与「答案质量不低于基线」——不建议为压延迟牺牲答案质量"],
        "结论": "",
    }
    if base:
        payload["结论"] = (
            f"基线 qwen2.5:3b 端到端 {base['端到端ms']:.0f} ms（{'达标' if base['是否达标SLA'] else '超线'}）、"
            f"关键词命中 {base['关键词命中率']:.0%}。"
            + ("；".join(建议) if 建议 else
               "；候选模型中未能同时兼顾延迟与答案质量，故**维持现有模型**，"
               "不在检索侧做无效优化（检索仅 65 ms，已无空间）。")
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {OUT}")
    for r in rows:
        if r.get("成功"):
            print(f"   {r['模型']:16} {r['端到端ms']:8.0f} ms  "
                  f"{'达标' if r['是否达标SLA'] else '超线'}  输出 {r['输出字符数']:3} 字  "
                  f"关键词命中 {r['关键词命中率']:.0%}")
        else:
            print(f"   {r['模型']:16} 失败：{r.get('错误')}")
    print(f"   可替换建议：{payload['可替换建议'][0][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
