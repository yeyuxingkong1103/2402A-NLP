# -*- coding: utf-8 -*-
"""工单7 / 人工智能NLP-RAG-功能测试及评估 —— answer_keywords 关键词核对脚本

作用：用 pymupdf 打开语料中的原始 PDF，逐页搜索每题 answer_keywords 里的每个关键词，
      确认关键词真实存在于对应 PDF 中，并记录命中页码与前 120 字原文片段。

匹配规则（归一化后比较）：
  1) 去空白：空格 / 换行 / 制表符 / 全角空格全部删除；
  2) 去千分位逗号：仅删除「数字,数字」形式的逗号（"1,357.79" -> "1357.79"）；
  3) 全角转半角：全角数字/字母/％（）．，－ 转半角；
  4) 因此 "1,357.79" 与 "1357.79"、"1,357.79 亿元" 与 "1,357.79亿元" 视为同一串。
原文片段通过「归一化字符 -> 原始字符」位置映射回原文取得，因此片段是真正的 PDF 原文
（仅把连续空白压成单个空格）。

输入：E:\\gao6gongdan\\工单7\\研发\\评估\\测试问题集.jsonl
      （source_file 字段为 ";" 分隔的语料文件名，相对于 研发\\data\\raw\\）
输出：E:\\gao6gongdan\\工单7\\测试\\测试数据\\关键词核对结果.json
      核不到的关键词 命中=false，必须从题库删除或替换；目标通过率 100%。

本脚本只读 PDF、只写上面的 JSON，不联网、不调用任何模型、不写语料目录。
"""
import json
import os
import re
import sys
import time

import pymupdf

PROJECT = "人工智能NLP-RAG-功能测试及评估"
BASE = r"E:\gao6gongdan\工单7"
RAW = os.path.join(BASE, "研发", "data", "raw")
QUESTIONS = os.path.join(BASE, "研发", "评估", "测试问题集.jsonl")
OUT = os.path.join(BASE, "测试", "测试数据", "关键词核对结果.json")
SNIPPET = 120

# 全角 -> 半角映射表（数字、字母、常用标点）
_FULLWIDTH = {ord(c): ord(c) - 0xFEE0 for c in
              "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
              "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ％（）．，－"}


def normalize(text: str) -> str:
    """归一化：全角转半角 -> 去空白 -> 去千分位逗号（用于比较）。"""
    text = text.translate(_FULLWIDTH)
    text = re.sub(r"[\s\u00a0\u3000]+", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return text


def normalize_with_map(text: str):
    """归一化并保留「归一化字符 -> 原文下标」映射，便于回取原文片段。"""
    mid_chars, mid_map = [], []
    for i, ch in enumerate(text):
        for c in ch.translate(_FULLWIDTH):
            if c.isspace() or c == "\u00a0":
                continue
            mid_chars.append(c)
            mid_map.append(i)
    out_chars, out_map = [], []
    n = len(mid_chars)
    for j, c in enumerate(mid_chars):
        if c == "," and 0 < j < n - 1 and mid_chars[j - 1].isdigit() and mid_chars[j + 1].isdigit():
            continue  # 千分位逗号
        out_chars.append(c)
        out_map.append(mid_map[j])
    return "".join(out_chars), out_map


def load_pages(filename: str):
    """返回 [(页码, 归一化文本, 归一化->原文下标映射, 原文), ...]"""
    path = os.path.join(RAW, filename)
    if not os.path.isfile(path):
        raise SystemExit(f"[FATAL] 语料文件不存在: {path}")
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        raw = page.get_text("text")
        npage, imap = normalize_with_map(raw)
        pages.append((i + 1, npage, imap, raw))
    doc.close()
    return pages


def search(pages, keyword: str):
    """在页面里找归一化关键词，返回 [(页码, 原文片段), ...]（一页最多记 1 条）。"""
    needle = normalize(keyword)
    if not needle:
        return []
    hits = []
    for pno, npage, imap, raw in pages:
        pos = npage.find(needle)
        if pos < 0:
            continue
        char_start = imap[pos]
        char_end = imap[min(pos + len(needle), len(imap)) - 1] + 1
        snippet = re.sub(r"\s+", " ", raw[max(0, char_start - 40): char_end + 80]).strip()
        hits.append((pno, snippet[:SNIPPET + 40]))
    return hits


def main() -> int:
    if not os.path.isfile(QUESTIONS):
        raise SystemExit(f"[FATAL] 问题集不存在: {QUESTIONS}")
    with open(QUESTIONS, encoding="utf-8") as fh:
        questions = [json.loads(line) for line in fh if line.strip()]

    page_cache, out_questions = {}, []
    total = hit_total = 0
    failed = []

    for q in questions:
        filenames = [f for f in q["source_file"].split(";") if f]
        entries = []
        for kw in q["answer_keywords"]:
            total += 1
            detail, all_pages, first_snippet = [], [], ""
            for fn in filenames:
                if fn not in page_cache:
                    page_cache[fn] = load_pages(fn)
                hits = search(page_cache[fn], kw)
                if not hits:
                    continue
                detail.append({
                    "文件": fn,
                    "页码": [p for p, _ in hits],
                    "原文片段": hits[0][1],
                })
                all_pages.extend(p for p, _ in hits)
                if not first_snippet:
                    first_snippet = hits[0][1]
            ok = bool(all_pages)
            if ok:
                hit_total += 1
            else:
                failed.append({"id": q["id"], "keyword": kw, "files": filenames})
            entries.append({
                "keyword": kw,
                "命中": ok,
                "页码": sorted(set(all_pages)),
                "原文片段": first_snippet,
                # 附加字段（不改变必需结构）：命中文件与逐文件页码，供 expected_pages_hint 与人工复核使用
                "命中文件": [d["文件"] for d in detail],
                "命中明细": detail,
            })
        out_questions.append({"id": q["id"], "关键词": entries})

    result = {
        "工单编号": PROJECT,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "语料文件数": len([f for f in os.listdir(RAW) if f.lower().endswith(".pdf")]),
        "题目数": len(out_questions),
        "题目": out_questions,
        "统计": {
            "关键词总数": total,
            "命中数": hit_total,
            "通过率": round(hit_total / total, 4) if total else 0.0,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"题目数={len(out_questions)} 关键词总数={total} 命中数={hit_total} "
          f"通过率={result['统计']['通过率']:.2%}")
    if failed:
        print("!! 未通过关键词（必须删除或替换）：")
        for f in failed:
            print("   ", f["id"], repr(f["keyword"]), f["files"])
    else:
        print("全部关键词命中，通过率 100%")
    print("OUT=", OUT)

    # 复核 expected_pages_hint：逐文件取该题全部关键词命中页的最小值-最大值
    print("-" * 60)
    by_id = {x["id"]: x for x in out_questions}
    bad = 0
    for q in questions:
        segs = []
        for fn in [f for f in q["source_file"].split(";") if f]:
            pgs = sorted({p for e in by_id[q["id"]]["关键词"] for d in e["命中明细"]
                          if d["文件"] == fn for p in d["页码"]})
            if pgs:
                segs.append(f"{pgs[0]}-{pgs[-1]}" if pgs[0] != pgs[-1] else f"{pgs[0]}")
            else:
                segs.append("未命中")
        calc = ";".join(segs)
        same = calc == q["expected_pages_hint"]
        bad += 0 if same else 1
        print(f"{'OK ' if same else '差异'} {q['id']} 题库hint={q['expected_pages_hint']} 实测={calc}")
    print(f"expected_pages_hint 与实测不一致题数：{bad}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
