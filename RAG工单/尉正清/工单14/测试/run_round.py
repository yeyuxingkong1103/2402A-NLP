# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""跑一轮 6 题测试，判分并落盘 —— 工单要求「多轮测试结果」，每轮都用这个脚本保证可比

与之前几轮的区别：**按数据集 questions.jsonl 的原格式提问**（题干 + A/B/C/D 选项 +
「只回答选项字母」的指令）。前几轮只发题干，模型面对「以下哪个描述符合」无从选择，
这不是模型能力问题，是提问格式没对齐评测集。

用法：
    D:/Anaconda/envs/rag_gd/python.exe run_round.py --round 2 --tag "带选项"
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ragflow_client import RAGFlow

HERE = Path(__file__).parent
IDS = json.loads((HERE / ".ragflow_ids.json").read_text(encoding="utf-8"))
QJSON = HERE / "测试问题.json"

INSTRUCTION = "\n\n请先阅读知识库内容，再从上面四个选项中选出唯一正确的一项。只回答选项字母（A、B、C 或 D），不要解释。"


def build_prompt(q):
    """数据集原格式：题干 + 选项"""
    lines = [q["question"].rstrip("：:") + "："]
    lines += q["options"]
    return "\n".join(lines) + INSTRUCTION


def build_messages(q):
    """把「检索查询」和「作答要求」拆成两条消息 —— 最终采用的问法

    RAGFlow 用**最后一轮用户消息**去做检索（dialog_service.py:315,350）。
    选项和「只回答选项字母」这种通用指令对检索是纯噪声：实测同一条问题，
    带上它们以后第 1~4 题的召回从 10 条掉到 1 条。所以选项和作答要求放在
    前一轮，最后一轮只留问题本身，让检索查询保持干净。
    """
    opts = "以下是一道单选题的选项：\n" + "\n".join(q["options"]) + \
           "\n\n作答要求：只回答选项字母（A、B、C 或 D），不要解释。"
    return [
        {"role": "user", "content": opts},
        {"role": "assistant", "content": "好的，我已了解选项和作答要求。"},
        {"role": "user", "content": q["question"]},
    ]


def parse_letter(ans):
    """从回答里抠出选项字母。优先找「答案：X」「选 X」这类显式结论，否则取首个独立字母。"""
    if not ans:
        return ""
    m = re.search(r"(?:答案|选项|选择|应选|正确)\s*[:：是为]*\s*\**\s*([ABCD])\b", ans)
    if m:
        return m.group(1)
    m = re.search(r"^\s*\**\s*([ABCD])\b", ans.strip())
    if m:
        return m.group(1)
    letters = re.findall(r"\b([ABCD])\b", ans)
    return letters[0] if letters else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-options", action="store_true",
                    help="复现前几轮：只发题干不带选项")
    ap.add_argument("--joined", action="store_true",
                    help="复现第 2~5 轮：选项和作答要求拼在同一条消息里")
    args = ap.parse_args()

    qs = json.loads(QJSON.read_text(encoding="utf-8"))
    c = RAGFlow()
    c.load_token()

    tag = args.tag or ("无选项" if args.no_options else "带选项")
    rows, t_all = [], time.time()
    for n, q in enumerate(qs, 1):
        if args.no_options:
            prompt = q["question"]
        elif args.joined:
            prompt = build_prompt(q)
        else:
            prompt = build_messages(q)
        ans, sec = c.ask(IDS["chat_id"], prompt)
        got = parse_letter(ans)
        ok = got == q["answer"].strip()
        rows.append({
            "round": f"第{args.round}轮-{tag}",
            "id": n, "group": q.get("group", 0), "q": q["question"],
            "expect": q["answer"],
            "expect_text": next((o for o in q["options"] if o.startswith(q["answer"])), ""),
            "got": got, "ok": ok, "ans": ans, "sec": round(sec, 1),
        })
        print(f"  {'✓' if ok else '✗'} 期望 {q['answer']} 得到 {got or '(空)'} "
              f"{sec:5.1f}s  {q['question'][:28]}")

    n_ok = sum(r["ok"] for r in rows)
    print(f"\n=== 第{args.round}轮（{tag}）准确率 {n_ok}/{len(rows)} "
          f"= {n_ok/len(rows):.0%}，总耗时 {time.time()-t_all:.0f}s ===")
    avg = sum(r["sec"] for r in rows) / len(rows)
    print(f"    平均响应 {avg:.1f}s，最慢 {max(r['sec'] for r in rows):.1f}s")

    out = HERE / "results" / f"round{args.round}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    已存 {out}")


if __name__ == "__main__":
    main()
