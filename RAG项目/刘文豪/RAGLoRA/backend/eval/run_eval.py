# -*- coding: utf-8 -*-
"""RAG 评测：检索命中率 / 答案覆盖 / 引用率 / 拒答正确性 / 延迟。

评测维度说明：
    来源命中  —— 期望的知识来源是否出现在精排结果里（衡量检索质量）
    答案覆盖  —— 期望关键词是否出现在回答里（衡量生成质量）
    引用标注  —— 回答是否带 [n] 来源编号（衡量指令遵循）
    拒答正确  —— 越界问题是否被礼貌拒绝

用法：
    D:\\anaconda3\\envs\\rag_env\\python.exe eval\\run_eval.py
    D:\\anaconda3\\envs\\rag_env\\python.exe eval\\run_eval.py --role medical --limit 5   # 快速冒烟
    D:\\anaconda3\\envs\\rag_env\\python.exe eval\\run_eval.py --report ../docs/eval_report.md
"""
import argparse
import json
import secrets
import statistics
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# 后端地址可用环境变量覆盖。默认 8000；本机实测 8000 可能被其它应用（如酷狗）
# 抢占，此时可用 `RAGLORA_API_BASE=http://127.0.0.1:8001/api` 指向备用端口。
BASE = os.environ.get("RAGLORA_API_BASE", "http://127.0.0.1:8000/api")
EVAL_DIR = Path(__file__).resolve().parent

# 角色 slug → 评测集键
ROLE_MAP = {"doctor": "medical", "lawyer": "legal"}

# 拒答判定：出现任一表述即视为「守住了专业边界」。
# 判定词放宽过一次——初版只认「无法确定/无法回答」，把
# 「我无法直接进行癌症诊断」「与我的专业领域不符」这类同样合格的拒答判成了失败。
REFUSAL_MARKERS = [
    "无法确定", "不能确定", "无法回答", "无法提供", "无法直接", "无法进行",
    "无法为您", "无法预测", "不能预测", "不便",
    "建议您咨询", "建议咨询", "建议您尽快前往", "建议尽快前往", "建议就诊",
    "超出", "与我的专业领域不符", "不属于我的专业", "不在我的专业", "专业范围",
    "不构成", "不是我的专业",
]


def load_qa() -> dict:
    return json.loads((EVAL_DIR / "qa_set.json").read_text(encoding="utf-8"))


def any_hit(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["medical", "legal", "refusal", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="每个角色只跑前 N 条（0=全部）")
    ap.add_argument("--report", default=str(EVAL_DIR / "eval_report.md"))
    args = ap.parse_args()

    qa = load_qa()
    c = httpx.Client(timeout=600)

    # ---- 登录 / 注册 ----
    uname = f"eval_{int(time.time())}"
    pw = secrets.token_hex(8)   # 临时用户密码每次随机生成，不写死
    r = c.post(f"{BASE}/auth/register", json={"username": uname, "password": pw})
    if r.status_code != 200:
        r = c.post(f"{BASE}/auth/login", json={"username": uname, "password": pw})
    H = {"Authorization": f"Bearer {r.json()['access_token']}"}

    chars = {ch["slug"]: ch for ch in c.get(f"{BASE}/characters", headers=H).json()}

    print("=" * 78)
    print(f"RAGLoRA 评测  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    rows: list[dict] = []
    roles = ["medical", "legal"] if args.role == "all" else [args.role]

    for role in roles:
        if role == "refusal":
            continue
        slug = next((k for k, v in ROLE_MAP.items() if v == role), None)
        if not slug or slug not in chars:
            continue
        items = qa[role]
        if args.limit:
            items = items[:args.limit]

        print(f"\n▶ {role}（角色：{chars[slug]['name']}，{len(items)} 题）")
        print("-" * 78)

        for i, item in enumerate(items, 1):
            # ⚠️ 每题用独立会话。
            # 早期版本整个角色共用一个会话，跑到第 9 题时上下文里已堆了 8 轮问答，
            # 模型的回答风格被历史带偏、不再标注 [n]，导致医学角色引用率被误判为 30%。
            conv = c.post(f"{BASE}/conversations", headers=H,
                          json={"character_id": chars[slug]["id"]}).json()
            t0 = time.time()
            try:
                res = c.post(f"{BASE}/chat/completions", headers=H, json={
                    "conversation_id": conv["id"], "question": item["q"],
                }).json()
            except Exception as e:
                print(f"  {i:>2}. !! 请求失败: {e}")
                continue
            dt = time.time() - t0

            answer = res.get("answer", "")
            sources = res.get("sources", [])
            src_text = " ".join(s.get("source", "") + " " + (s.get("law_name") or "")
                                for s in sources)

            # 语料缺口题：期望的法律不在知识库里，来源命中不计入分母
            corpus_gap = bool(item.get("corpus_gap"))
            hit_src = None if corpus_gap else any_hit(src_text, item.get("expect_sources", []))
            hit_ans = any_hit(answer, item.get("expect_keywords", []))
            cited = "[" in answer and "]" in answer

            rows.append({
                "role": role, "q": item["q"], "answer": answer,
                "source_hit": hit_src, "answer_hit": hit_ans, "cited": cited,
                "corpus_gap": corpus_gap,
                "latency": dt, "n_sources": len(sources),
                "top_source": sources[0]["source"] if sources else "",
                "rerank_top": (sources[0].get("rerank_score") if sources else None),
                "trace": res.get("trace", {}),
            })

            mark = lambda b: "—" if b is None else ("✓" if b else "✗")
            gap = " [语料缺口]" if corpus_gap else ""
            print(f"  {i:>2}. 来源{mark(hit_src)} 答案{mark(hit_ans)} 引用{mark(cited)} "
                  f"{dt:>5.1f}s  {item['q'][:34]}{gap}")

    # ---- 拒答评测 ----
    if args.role in ("all", "refusal"):
        print(f"\n▶ 拒答（{len(qa['refusal'])} 题）")
        print("-" * 78)
        for i, item in enumerate(qa["refusal"], 1):
            slug = item["role"]
            if slug not in chars:
                continue
            conv = c.post(f"{BASE}/conversations", headers=H,
                          json={"character_id": chars[slug]["id"]}).json()
            t0 = time.time()
            res = c.post(f"{BASE}/chat/completions", headers=H, json={
                "conversation_id": conv["id"], "question": item["q"],
            }).json()
            dt = time.time() - t0
            answer = res.get("answer", "")
            refused = any(m in answer for m in REFUSAL_MARKERS)
            rows.append({
                "role": "refusal", "q": item["q"], "answer": answer,
                "source_hit": None, "answer_hit": refused, "cited": "[" in answer,
                "latency": dt, "n_sources": len(res.get("sources", [])),
                "top_source": "", "rerank_top": None, "trace": res.get("trace", {}),
            })
            print(f"  {i:>2}. 拒答{'✓' if refused else '✗'}  {dt:>5.1f}s  {item['q'][:38]}")

    # ---- 汇总 ----
    def pct(sel, key):
        vals = [r[key] for r in sel if r[key] is not None]
        return (sum(vals) / len(vals) * 100) if vals else 0.0

    print()
    print("=" * 78)
    print("结果汇总")
    print("=" * 78)
    print(f"{'角色':<10} {'题数':>5} {'来源命中':>9} {'答案覆盖':>9} {'引用标注':>9} {'平均延迟':>9}")
    print("-" * 78)

    summary = {}
    for role in ["medical", "legal", "refusal"]:
        sel = [r for r in rows if r["role"] == role]
        if not sel:
            continue
        lat = statistics.mean(r["latency"] for r in sel)
        s = {
            "n": len(sel),
            "source_hit": pct(sel, "source_hit"),
            "answer_hit": pct(sel, "answer_hit"),
            "cited": pct(sel, "cited"),
            "latency": lat,
        }
        summary[role] = s
        label = {"refusal": "拒答"}.get(role, role)
        print(f"{label:<10} {s['n']:>5} {s['source_hit']:>8.1f}% {s['answer_hit']:>8.1f}% "
              f"{s['cited']:>8.1f}% {lat:>8.1f}s")

    traced = [r for r in rows if r["trace"].get("total_retrieval_ms")]
    if traced:
        print()
        print("链路耗时分解（平均）：")
        for k, label in [("rewrite_ms", "查询改写"), ("recall_ms", "混合检索"),
                         ("rerank_ms", "精排"), ("generate_ms", "生成")]:
            vals = [r["trace"][k] for r in traced if r["trace"].get(k) is not None]
            if vals:
                print(f"  {label:<8} {statistics.mean(vals):>7.0f} ms")
        skipped = sum(1 for r in traced if r["trace"].get("rewrite_skipped"))
        print(f"  改写跳过  {skipped}/{len(traced)} 次（完整问句无需指代消解）")

    # ---- 生成报告 ----
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# RAGLoRA 评测报告",
            "",
            f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
            f"> 评测用户：{uname}　样本量：{len(rows)} 题",
            "",
            "## 汇总指标",
            "",
            "| 角色 | 题数 | 来源命中率 | 答案覆盖率 | 引用标注率 | 平均延迟 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for role, s in summary.items():
            label = {"refusal": "拒答"}.get(role, role)
            sh = "—" if role == "refusal" else f"{s['source_hit']:.1f}%"
            lines.append(f"| {label} | {s['n']} | {sh} | {s['answer_hit']:.1f}% | "
                         f"{s['cited']:.1f}% | {s['latency']:.1f}s |")

        if traced:
            lines += ["", "## 链路耗时分解", "",
                      "| 环节 | 平均耗时 |", "|---|---:|"]
            for k, label in [("rewrite_ms", "查询改写"), ("recall_ms", "混合检索"),
                             ("rerank_ms", "精排"), ("generate_ms", "生成")]:
                vals = [r["trace"][k] for r in traced if r["trace"].get(k) is not None]
                if vals:
                    lines.append(f"| {label} | {statistics.mean(vals):.0f} ms |")

        lines += ["", "## 逐题明细", "",
                  "| # | 角色 | 问题 | 来源命中 | 答案覆盖 | 引用 | 延迟 | 首位来源 |",
                  "|---:|---|---|:-:|:-:|:-:|---:|---|"]
        for i, r in enumerate(rows, 1):
            mk = lambda v: "—" if v is None else ("✓" if v else "✗")
            lines.append(
                f"| {i} | {r['role']} | {r['q']} | {mk(r['source_hit'])} | "
                f"{mk(r['answer_hit'])} | {mk(r['cited'])} | {r['latency']:.1f}s | "
                f"{r['top_source']} |"
            )

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n报告已写入 {out}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
