# -*- coding: utf-8 -*-
"""工单13 完善 A3：CPU 推理吐字速度标定（跨模型 A/B）。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

背景：优化后唯一超线题 Q10（3136.6 ms）中，**检索仅 65 ms，生成 3062 ms**，
首字仅 215 ms —— 瓶颈是纯输出长度。本脚本标定不同模型在同一提示下的吐字速度，
给出「CPU 部署下压延迟的真实杠杆」结论（换模型 / 压输出），
而不是继续在检索侧做无效优化。

产出：``优化/评估结果/吐字速度标定.json``
用法：& $py 研发/scripts/标定吐字速度.py
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "优化" / "评估结果" / "吐字速度标定.json"
MODELS = ["qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:0.5b"]
PROMPT = ("请用中文回答，只给出结论不要解释。\n"
          "中信证券2020年年度报告中，公司全年营业收入、归属于公司股东净利润、"
          "年末总资产和净资产收益率分别是多少？\n答案：")


def run(model: str, prompt: str, timeout: int = 300) -> dict:
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
        txt = str(d.get("response", ""))
        ev = d.get("eval_count") or 0
        ed = (d.get("eval_duration") or 0) / 1e9  # 秒
        return {
            "模型": model, "成功": True, "墙钟ms": round(ms, 1),
            "输出字符数": len(txt.strip()), "eval_count": ev,
            "eval秒": round(ed, 3),
            "吐字速度_字符每秒": round(len(txt.strip()) / (ms / 1000), 1) if ms else 0,
            "模型侧tok每秒": round(ev / ed, 1) if ed else None,
            "回答摘录": txt.strip()[:120],
        }
    except Exception as e:
        return {"模型": model, "成功": False, "错误": f"{type(e).__name__}: {e}"}


def main() -> int:
    rows = []
    for m in MODELS:
        print(f"  跑 {m} …")
        rows.append(run(m, PROMPT))
        time.sleep(0.5)

    ok = [r for r in rows if r.get("成功")]
    fastest = min(ok, key=lambda r: r["墙钟ms"]) if ok else None
    payload = {
        "工单编号": "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化",
        "完善项": "A3 CPU 吐字速度标定（跨模型 A/B）",
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "口径": "同一提示词、temperature=0、num_predict=256、keep_alive=0s；单模型串行避免互相抢 CPU",
        "SLA_ms": 3000,
        "结果": rows,
        "最快模型": fastest["模型"] if fastest else None,
        "结论": "",
    }
    if fastest:
        payload["结论"] = (
            f"同一提示下最快为 {fastest['模型']}（{fastest['墙钟ms']:.0f} ms，"
            f"{fastest['吐字速度_字符每秒']} 字符/秒）。"
            "Q10 的 3062 ms 生成耗时 = 177 字符 ÷ 该速度量级 → **延迟由输出长度与 CPU 吐字速度决定**，"
            "检索侧已无优化空间（仅 65 ms）。CPU 部署下要达标只有两条路："
            "① 换更小/更快的生成模型；② 压缩输出长度（须先做答案质量等价性验证）。"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {OUT}")
    for r in rows:
        if r.get("成功"):
            print(f"   {r['模型']:16} {r['墙钟ms']:8.0f} ms  {r['吐字速度_字符每秒']:6.1f} 字符/秒  "
                  f"输出 {r['输出字符数']} 字")
        else:
            print(f"   {r['模型']:16} 失败：{r.get('错误')}")
    print(f"   结论：{payload['结论'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
