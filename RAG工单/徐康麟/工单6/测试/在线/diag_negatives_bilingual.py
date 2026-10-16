# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/在线/diag_negatives_bilingual.py（T6 诊断：拒答闸门范围 与 英文路径表现）

用途（**只做诊断，不改引擎、不产成绩**）：
  1. 刻画拒答闸门对「不存在实体」的覆盖范围（已知名实体 vs 泛化名「某某科技有限公司」）；
  2. 打印英文路径的完整答案，判断「判分口径不适用」与「真的答不出」两种情况；
  3. 顺带核实 N1 失败用例的引用页（是否真的含被编造的数字）。
运行：pwsh -NoProfile -File run_py.ps1 测试/在线/diag_negatives_bilingual.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8123"


def ask(question: str, lang: str = "zh") -> dict:
    body = json.dumps({"question": question, "lang": lang, "use_cache": False, "top_k": 5},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BASE + "/api/ask", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    print("=== 1) N1 失败用例根因：引用页 招股说明书1 第64页 ===")
    with fitz.open(ROOT / "研发" / "data" / "raw" / "招股说明书1.pdf") as doc:
        text = doc[63].get_text("text")
    print(f"  含「某某」={'某某' in text}  含注册资本={'注册资本' in text}  "
          f"含100万={'100 万' in text or '100万' in text}")
    for m in re.finditer(r".{0,22}100\s*万.{0,22}", text):
        print("  片段:", m.group(0).replace("\n", " "))

    print("\n=== 2) 拒答闸门范围探针（不存在实体）===")
    for q in ["某某科技有限公司的注册资本是多少？",
              "特斯拉公司的注册资本是多少？",
              "苹果公司的注册资本是多少？",
              "武汉兴图新科电子股份有限公司的注册资本是多少？"]:
        d = ask(q)
        print(f"  Q: {q}")
        print(f"     refused={d.get('refused')} reason={d.get('refusal_reason')} "
              f"cited={d.get('cited_pages')} A={str(d.get('answer'))[:60]!r}")

    print("\n=== 3) 英文路径完整答案（区分「口径不适用」与「真的答不出」）===")
    cases = [
        ("543", "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"),
        ("5", "In the organization chart of Wuhan P&S Information Technology Co., Ltd., how many "
              "departments make up the Sales Department, and how many sales offices make up the "
              "Key Account Sales Department?"),
        ("6", "According to the prospectus of Wuhan P&S Information Technology Co., Ltd., based on the "
              "2008 China IC market application structure and growth chart, which industry has the "
              "fastest growth rate and which has negative growth?"),
    ]
    for qid, q in cases:
        d = ask(q, "en")
        print(f"  id {qid}: refused={d.get('refused')} reason={d.get('refusal_reason')} "
              f"cited={d.get('cited_pages')} lang={d.get('language')}")
        print("     A:", str(d.get("answer")).replace("\n", " / ")[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
