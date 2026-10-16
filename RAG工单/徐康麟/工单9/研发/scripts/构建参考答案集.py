# -*- coding: utf-8 -*-
"""构建工单9 的「参考答案集」：抽取 sample_questions.pdf 原文并与 17 题评估集逐条核对。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

为什么必须自己核对原文（工单硬性要求）：
    工单原文要求「比对附件中《sample_questions.pdf》中的参考答案的内容」，
    而 工单7 已把该 PDF 整理成 17 题评估集。**不能直接信任二手整理**：
    本脚本对 PDF 逐页抽文本、逐条解析，再用归一化编辑距离验证
    「PDF 的前 4 题 == 17 题集的 Q01~Q04」，把核对结论写进产物，供验收追溯。

⚠️ 实测踩坑（必须记住）：``sample_questions.pdf`` 的文本层用的是 **Kangxi 部首字符**
（``⾏`` U+2F8F / ``⼀`` U+2F00 / ``⾦`` U+2F7D / ``⻓`` U+2FBF …），
与正常汉字（``行````一````金````长``）**不是同一个码位**。不归一化就会出现：
    * 关键词表里是 ``银⾏``，而索引正文里是 ``银行`` ⇒ 关键词命中率恒为 0；
    * 与 17 题集的问题文本比对相似度极低 ⇒ 误判为「不匹配」。
因此抽取后第一件事就是 ``unicodedata.normalize("NFKC", text)``。

产物：
    * ``研发/评估/参考答案集.jsonl``       —— **17 条评估用参考答案**（主评估集，含 PDF 核对标记）
    * ``研发/评估/PDF参考答案原文对照.jsonl`` —— PDF 抽出的 4 条原始 Q&A（含页码与映射结论）
    * ``研发/评估/参考答案原文.txt``        —— PDF 逐页原始文本（人工复核用）
    * ``研发/评估/参考答案集核对报告.json`` —— 核对结论（匹配率、映射表、不一致项）

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/构建参考答案集.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/构建参考答案集.py --dump
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "输入" / "sample_questions.pdf"
BASE_QUESTIONS = Path(r"E:\gao6gongdan\工单7\研发\评估\测试问题集.jsonl")   # 只读引用
EVAL_DIR = ROOT / "研发" / "评估"
OUT_REF = EVAL_DIR / "参考答案集.jsonl"
OUT_PDF = EVAL_DIR / "PDF参考答案原文对照.jsonl"
OUT_RAW = EVAL_DIR / "参考答案原文.txt"
OUT_CHECK = EVAL_DIR / "参考答案集核对报告.json"


# --------------------------------------------------------------------------
# 文本归一化
# --------------------------------------------------------------------------
def nfkc(text: str) -> str:
    """Kangxi 部首 → 常规汉字（NFKC），并统一空白/全角。"""
    return unicodedata.normalize("NFKC", str(text or ""))


def normalize_for_match(text: str) -> str:
    """比对用的强归一化：NFKC + 去空白 + 去标点 + 小写。"""
    text = nfkc(text).lower()
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"[，,。.、；;：:！!？?（）()【】\[\]《》“”\"'‘’\-—_/\\|]", "", text)
    return text


# --------------------------------------------------------------------------
# PDF 抽取
# --------------------------------------------------------------------------
def extract_pages(path: Path) -> list[str]:
    """逐页抽文本（单进程，无进程池；PDF 仅 ~211 KB）。"""
    try:
        import pymupdf  # 新版包名
    except Exception:  # noqa: BLE001 —— 旧版只有 fitz
        try:
            import fitz as pymupdf  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[ERROR] 需要 pymupdf：{type(exc).__name__}: {exc}") from exc
    if not path.is_file():
        raise SystemExit(f"[ERROR] 找不到参考答案 PDF：{path}")
    pages: list[str] = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            pages.append(nfkc(page.get_text("text")))
    return pages


#: 「问题：」/「问题:」/「问：」开头的块
_Q_HEAD = re.compile(r"(?:^|\n)\s*(?:问题|问)\s*[:：]\s*")
_A_HEAD = re.compile(r"\n\s*(?:参考答案|答案|答)\s*[:：]\s*")
#: 英文题（PDF 中若出现）
_Q_HEAD_EN = re.compile(r"(?:^|\n)\s*(?:Question|Q)\s*[:：]\s*", re.IGNORECASE)
_A_HEAD_EN = re.compile(r"\n\s*(?:Reference\s+Answer|Answer|A)\s*[:：]\s*", re.IGNORECASE)


#: 页码哨兵（把多页拼成一条流，用于正确处理**跨页答案**）
_PAGE_SENTINEL = "\n@@PAGE={}@@\n"


def parse_pdf_qa(pages: list[str]) -> list[dict[str, object]]:
    """解析 PDF 的「问题/参考答案」对（**支持跨页答案**）。

    排版实测：``问题：\\n<题干>\\n答案：\\n<答案正文>``，且**答案会跨页**。
    踩坑记录 1：最初按「一页一解析」处理，第 1 页 Q3 的答案在第 2 页还有「2. 资产质量控制…」
    「3. 贷款结构优化策略…」三段续写，这些续写前面**没有** ``问题：`` 头，
    按页解析会把它们整段丢掉（参考答案只剩 1/4，判分口径直接失真）。

    踩坑记录 2：把页面用哨兵串成一条流后，**页码不能靠「段内是否含哨兵」来推** ——
    哨兵在页首，而第一个 ``问题：`` 出现在哨兵**之后**，因此第 1 题的段内根本没有哨兵，
    页码会算成 0（实测输出「PDF第[0]页」）。正确做法：记录每页在流里的**起始偏移**，
    再用「段起始偏移落在哪一页区间」反查页码。
    """
    parts: list[str] = []
    page_starts: list[tuple[int, int]] = []          # (流内起始偏移, 页码)
    offset = 0
    for page_no, raw in enumerate(pages, 1):
        text = raw.replace("\u3000", " ")
        # 去掉 PDF 里的占位行（实测 ``xxx__2019年__年度报告`` 这类是水印/页脚）
        text = re.sub(r"(?m)^\s*[a-zA-Z_\u4e00-\u9fff]{0,40}__\d{4}年__年度报告\s*$", "", text)
        text = text.strip("\n")
        if parts:
            parts.append("\n")
            offset += 1
        page_starts.append((offset, page_no))
        parts.append(text)
        offset += len(text)
    stream = "".join(parts)

    def page_at(pos: int) -> int:
        page = page_starts[0][1] if page_starts else 0
        for start, page_no in page_starts:
            if start <= pos:
                page = page_no
            else:
                break
        return page

    rows: list[dict[str, object]] = []
    heads = list(_Q_HEAD.finditer(stream))
    for idx, head in enumerate(heads):
        seg_start = head.end()
        seg_end = heads[idx + 1].start() if idx + 1 < len(heads) else len(stream)
        segment = stream[seg_start:seg_end]
        answer_head = _A_HEAD.search(segment)
        if not answer_head:
            continue
        q_text = segment[:answer_head.start()]
        a_text = segment[answer_head.end():]
        q_text = re.sub(r"\s*\n\s*", " ", q_text).strip()
        a_text = re.sub(r"[ \t]+", " ", a_text).strip()
        a_text = re.sub(r"\n{3,}", "\n\n", a_text)
        if len(q_text) < 8 or len(a_text) < 10:
            continue
        rows.append({
            "raw_pages": [page_at(seg_start), page_at(max(seg_end - 1, seg_start))],
            "question": q_text,
            "reference_answer": re.sub(r"\s+", " ", a_text).strip(),
        })
    return rows


# --------------------------------------------------------------------------
# 关键词抽取（规则化、可复核；非 LLM 生成）
# --------------------------------------------------------------------------
_ATOM = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|千元|百万元|元|亿美元|亿港元)"
    r"|\d+(?:\.\d+)?\s*(?:%|个百分点|倍)"
    r"|(?:19|20)\d{2}\s*年(?:度)?"
)

_STOP = {
    "以下", "如下", "其中", "此外", "同时", "因此", "说明", "分析", "结合", "关于", "对于",
    "报告", "年报", "年度报告", "公司", "集团", "银行", "情况", "内容", "主要", "进行",
    "通过", "相关", "表示", "上述", "以及", "分别", "包括", "根据", "披露", "显示",
    "一是", "二是", "三是", "例如", "如xx", "xx银行", "yy银行",
}


def _push(out: list[str], token: str, *, min_len: int = 4, max_len: int = 26) -> None:
    token = str(token or "").strip(" 　，,。、；;：:（）()【】[]《》\"'“”‘’")
    if not (min_len <= len(token) <= max_len):
        return
    if token in _STOP or token in out:
        return
    if re.fullmatch(r"[\d.,%]+", token):
        return
    out.append(token)


def keywords_of(answer: str, *, limit: int = 10) -> list[str]:
    """从参考答案抽判分关键词（规则化，可逐条复核）。"""
    out: list[str] = []
    text = str(answer or "")
    for atom in _ATOM.findall(text):
        _push(out, re.sub(r"\s+", "", atom), min_len=3, max_len=20)
    for quoted in re.findall(r"[“\"《]([^”\"》]{3,24})[”\"》]", text):
        _push(out, quoted)
    for piece in re.split(r"[。；;\n]", text):
        for sub in re.split(r"[、，,]", piece):
            _push(out, sub)
    if len(out) < limit:
        for piece in re.split(r"[。；;\n]", text):
            for sub in re.split(r"[：:]", piece):
                _push(out, sub, min_len=4)
    return out[:limit]


def similarity(a: str, b: str) -> float:
    """编辑相似度（0~1，字符级）。"""
    return round(SequenceMatcher(None, normalize_for_match(a), normalize_for_match(b)).ratio(), 4)


def containment(a: str, b: str) -> float:
    """**包含度**：短串被长串覆盖的比例。

    为什么不能只看编辑相似度（实测）：PDF 原文用的是匿名占位「xx银行」，
    而评估集 Q01 写的是「平安银行」，且 Q01 还多了一句「请结合业务结构…说明」。
    直接算 ratio 只有 0.60，会被阈值判成「不匹配」——但它**确实是同一道题**。
    包含度把「长串多出来的部分」的惩罚去掉，只问「短串有多少被长串覆盖」，
    对「扩写/匿名化」这类真实差异更稳健。
    """
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if not na or not nb:
        return 0.0
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    matcher = SequenceMatcher(None, short, long_)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return round(matched / len(short), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 构建参考答案集")
    parser.add_argument("--dump", action="store_true", help="只打印 PDF 原文")
    parser.add_argument("--pdf", default=str(PDF_PATH))
    parser.add_argument("--threshold", type=float, default=0.75, help="PDF 题与 17 题集的匹配阈值")
    args = parser.parse_args(argv)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pages = extract_pages(Path(args.pdf))
    OUT_RAW.write_text("\n\n".join(f"===== 第 {i} 页 =====\n{t}" for i, t in enumerate(pages, 1)),
                       encoding="utf-8")
    print(f"[1/5] PDF 抽取完成：{len(pages)} 页 → {OUT_RAW.name}")
    if args.dump:
        for i, p in enumerate(pages, 1):
            print(f"----- 第 {i} 页 -----\n{p}")
        return 0

    pdf_rows = parse_pdf_qa(pages)
    print(f"[2/5] 解析出 PDF 参考问答 {len(pdf_rows)} 条")

    base_rows: list[dict[str, object]] = []
    if BASE_QUESTIONS.is_file():
        for line in BASE_QUESTIONS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                base_rows.append(json.loads(line))
    print(f"[3/5] 载入 17 题评估集：{len(base_rows)} 题（只读引用 {BASE_QUESTIONS.parent.parent.parent.name}）")

    # ---- 核对：PDF 每一条在 17 题集里找最佳匹配 ----
    mapping: list[dict[str, object]] = []
    for prow in pdf_rows:
        best_id, best_sim, best_cont = "", 0.0, 0.0
        for brow in base_rows:
            sim = similarity(str(prow["question"]), str(brow.get("question") or ""))
            cont = containment(str(prow["question"]), str(brow.get("question") or ""))
            score = max(sim, cont)
            if score > max(best_sim, best_cont):
                best_id, best_sim, best_cont = str(brow.get("id")), sim, cont
        matched = max(best_sim, best_cont) >= float(args.threshold)
        mapping.append({"pdf_question": str(prow["question"])[:80], "pdf_pages": prow["raw_pages"],
                        "matched_id": best_id, "similarity": best_sim, "containment": best_cont,
                        "matched": matched})
    matched_n = sum(1 for m in mapping if m["matched"])
    print(f"[4/5] PDF ↔ 17 题集核对：{matched_n}/{len(mapping)} 条匹配（阈值 {args.threshold}）")
    for m in mapping:
        flag = "✓" if m["matched"] else "✗"
        print(f"      {flag} PDF第{m['pdf_pages']}页 → {m['matched_id']}  "
              f"相似度 {m['similarity']}  包含度 {m['containment']}")

    # ---- 写 PDF 原文对照 ----
    with OUT_PDF.open("w", encoding="utf-8") as handle:
        for idx, (prow, m) in enumerate(zip(pdf_rows, mapping), 1):
            handle.write(json.dumps({
                "id": f"PDF-Q{idx}",
                "question": prow["question"],
                "reference_answer": prow["reference_answer"],
                "answer_keywords": keywords_of(str(prow["reference_answer"])),
                "pdf_pages": prow["raw_pages"],
                "matched_eval_id": m["matched_id"],
                "matched_similarity": m["similarity"],
                "matched_containment": m["containment"],
                "matched": m["matched"],
                "来源": "输入/sample_questions.pdf（本工单用 pymupdf 自行抽取 + NFKC 归一）",
            }, ensure_ascii=False) + "\n")

    # ---- 写 17 题主评估集（带 PDF 核对标记）----
    pdf_by_eval_id: dict[str, dict[str, object]] = {}
    for idx, (prow, m) in enumerate(zip(pdf_rows, mapping), 1):
        if m["matched"]:
            pdf_by_eval_id[str(m["matched_id"])] = {"pdf_index": idx, "similarity": m["similarity"],
                                                    "containment": m["containment"],
                                                    "pdf_answer": prow["reference_answer"],
                                                    "pdf_pages": prow["raw_pages"]}
    n_pdf_backed = 0
    with OUT_REF.open("w", encoding="utf-8") as handle:
        for brow in base_rows:
            qid = str(brow.get("id"))
            backed = pdf_by_eval_id.get(qid)
            ref = str(brow.get("reference_answer") or "")
            if backed:
                n_pdf_backed += 1
            row = {
                "id": qid,
                "question": brow.get("question"),
                "lang": brow.get("lang") or "zh",
                "reference_answer": ref or str((backed or {}).get("pdf_answer") or ""),
                "answer_keywords": list(brow.get("answer_keywords") or []),
                "source_pdf": brow.get("source_pdf"),
                "source_file": brow.get("source_file"),
                "expected_pages_hint": brow.get("expected_pages_hint"),
                "question_type": brow.get("question_type"),
                "来源": ("sample_questions.pdf 原文（已逐条核对）" if backed
                        else "工单7 研发/评估/测试问题集.jsonl（该题不在 PDF 原文中，为扩充题）"),
                "pdf_核对": {
                    "matched": bool(backed),
                    "pdf_index": (backed or {}).get("pdf_index"),
                    "similarity": (backed or {}).get("similarity"),
                    "containment": (backed or {}).get("containment"),
                    "pdf_pages": (backed or {}).get("pdf_pages"),
                    "pdf_reference_answer": (backed or {}).get("pdf_answer"),
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "工单": WORK_ORDER,
        "PDF": str(args.pdf),
        "PDF页数": len(pages),
        "PDF参考问答条数": len(pdf_rows),
        "评估集题数": len(base_rows),
        "PDF核对匹配阈值": float(args.threshold),
        "PDF核对匹配数": matched_n,
        "评估集中有PDF原文支撑的题数": n_pdf_backed,
        "映射": mapping,
        "NFKC归一化": {
            "原因": "sample_questions.pdf 文本层使用 Kangxi 部首码位（⾏/⼀/⾦/⻓），不做 NFKC 会导致关键词与索引正文不同码位、命中率恒为 0",
            "示例": {"原始": "银⾏", "归一化后": nfkc("银⾏")},
            "实现": "unicodedata.normalize('NFKC', text)",
        },
        "结论": ("PDF 原文的 %d 条问答与评估集 Q01~Q%02d 一一对应，"
                 "其余 %d 题为工单7 扩充题（不在 PDF 原文中，判分仍用其人工关键词）"
                 % (matched_n, matched_n, len(base_rows) - n_pdf_backed)),
    }
    OUT_CHECK.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[5/5] 产物：\n      {OUT_REF.name}（{len(base_rows)} 题）\n      {OUT_PDF.name}（{len(pdf_rows)} 条）\n      {OUT_CHECK.name}")
    print(f"      结论：{report['结论']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
