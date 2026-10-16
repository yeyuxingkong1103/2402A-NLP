# -*- coding: utf-8 -*-
"""工单14 测试用例（离线，不联网、不调 LLM）。

工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单

覆盖：扫描件性质实证、附图标记恢复率、六题答案与证据自洽、产物齐全性。
运行：先 cd E:\\gao6gongdan\\工单14，再
      & 'E:\\gao6gongdan\\工单1\\.venv\\Scripts\\python.exe' -m pytest 测试 -q
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "研发" / "data" / "解析实证.json"
RES = ROOT / "研发" / "评估" / "六题结果.json"
Q = ROOT / "测试" / "测试数据" / "测试问题集.jsonl"
SHOT = ROOT / "测试" / "测试截图"
RENDER = ROOT / "研发" / "data" / "_render"
DESIGN = ROOT / "设计"

CN_PNG = re.compile(r"^[\u4e00-\u9fff0-9A-Za-z_\-（）()]+\.png$")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- 实证类
def test_文本层探针证明是纯扫描件():
    ev = load(EV)
    assert ev, "缺 研发/data/解析实证.json"
    p = ev["1_文本层探针"]
    assert p["pages"] == 8, f"页数应为 8，实得 {p['pages']}"
    assert p["text_chars_total"] == 0, f"文本层应为 0 字符，实得 {p['text_chars_total']}"
    assert "扫描件" in p["verdict"]
    assert all(x["images"] >= 1 for x in p["per_page"]), "每页应至少有 1 张位图"


def test_页面渲染产物齐全():
    pngs = sorted(RENDER.glob("page*.png"))
    assert len(pngs) == 8, f"应渲染 8 页，实得 {len(pngs)}"
    for p in pngs:
        assert p.stat().st_size > 10000, f"{p.name} 疑似空图（{p.stat().st_size} B）"


def test_附图标记恢复率可追溯():
    ev = load(EV)
    cov = ev["4_标记覆盖"]
    assert cov["needed_count"] == 14
    assert cov["recovered_count"] == 7, f"报告口径为 7/14，实得 {cov['recovered_count']}"
    # 缺失集合必须与报告一致（这是根因分析的核心证据）
    assert set(cov["missing"]) == {"6'", "6''", "X1", "X2", "X3", "h1", "h2"}, cov["missing"]
    assert set(cov["recovered"]) >= {"4", "5", "6", "D", "7", "10"}
    # 像素 OCR 确实跑了（有耗时与命中数）
    for k, v in ev["3_像素OCR"].items():
        assert v["hits_scored"] > 0 and v["elapsed_ms"] > 0, f"{k} 无有效 OCR 记录"


# ---------------------------------------------------------------- 答题类
def test_问题集为六题且答案键完整():
    rows = load_jsonl(Q)
    assert len(rows) == 6, f"应为 6 题，实得 {len(rows)}"
    for r in rows:
        assert r.get("answer") in {"A", "B", "C", "D"}, f"答案键异常：{r.get('answer')}"
        opts = r.get("options") or []
        assert len(opts) == 4, f"应为四选一，实得 {len(opts)} 个选项"
        assert r.get("document") == "CN100342976C.pdf"


def test_六题结果与问题集答案键一致():
    res, rows = load(RES), load_jsonl(Q)
    assert res and rows, "缺 六题结果.json 或 问题集"
    by_seq = {x["序号"]: x for x in res["逐题"]}
    assert len(by_seq) == 6
    for i, r in enumerate(rows, 1):
        x = by_seq[i]
        assert x["question"] == r["question"], f"第{i}题题干不一致"
        assert x["answer"] == r["answer"], f"第{i}题答案键不一致：{x['answer']} vs {r['answer']}"
        assert x["predicted"] == r["answer"], f"第{i}题判定错误"
        assert x["correct"] is True


def test_六题准确率为100且图内题全覆盖():
    res = load(RES)
    s = res["汇总"]
    assert s["题数"] == 6 and s["答对"] == 6
    assert abs(s["准确率"] - 1.0) < 1e-9
    assert s["需图内空间信息的题数"] == 4, "题 3/4/5/6 共 4 题依赖图内空间信息"
    assert s["图内题答对"] == 4
    groups = {x["group"] for x in res["逐题"]}
    assert groups == {1, 2, 3}, f"应覆盖三个 group，实得 {groups}"


def test_每题都带证据页与证据原文():
    res = load(RES)
    for x in res["逐题"]:
        assert isinstance(x.get("evidence_page"), int) and 1 <= x["evidence_page"] <= 8
        assert len(str(x.get("evidence", "")).strip()) >= 15, f"第{x['序号']}题证据过短"
    pages = sorted(x["evidence_page"] for x in res["逐题"])
    assert pages[:2] == [1, 2], "题 1/2 的证据页应为 1/2"
    assert pages[2:] == [7, 7, 7, 7], "题 3~6 的证据页应全为第 7 页"


# ---------------------------------------------------------------- 交付类
def test_测试截图存在非空且中文命名():
    pngs = sorted(SHOT.glob("*.png"))
    assert len(pngs) >= 2, f"截图应 ≥2 张，实得 {len(pngs)}"
    for p in pngs:
        assert p.stat().st_size > 5000, f"{p.name} 疑似空图"
        assert CN_PNG.match(p.name), f"{p.name} 命名不合规"


def test_设计文档与RAGFlow分析齐备():
    for name in ("工单需求原文.md", "RAGFlow技术分析.md"):
        f = DESIGN / name
        assert f.exists() and f.stat().st_size > 3000, f"缺设计文档 {name}"
    txt = (DESIGN / "RAGFlow技术分析.md").read_text(encoding="utf-8")
    # 三问的关键锚点必须在文中（防止空壳文档）
    for key in ("FACTORY", "queue_product", "do_handle_task", "deepdoc/parser", "page_size"):
        assert key in txt, f"RAGFlow 分析缺少关键锚点 {key}"


# ==================== 完善期新增（守护百度 OCR 结论） ====================
OCR_DIR = ROOT / "研发" / "data" / "ocr"
OCR_ANSWER = OCR_DIR / "答题复现.json"

NEEDED_MARKS = ["4","5","6","6'","6''","X1","X2","X3","h1","h2","D","7","10","1"]


def _ocr_text() -> str:
    parts = []
    for f in sorted(OCR_DIR.glob("page*.ocr.json")):
        parts.append(json.loads(f.read_text(encoding="utf-8"))["text"])
    return "\n".join(parts)


def test_百度OCR产物齐全():
    files = sorted(OCR_DIR.glob("page*.ocr.json"))
    assert len(files) == 8, f"应识别 8 页（工单14 全页），实得 {len(files)}"


def test_OCR里程碑_附图标记恢复不少于13个():
    """守护 2026-10-16 的结论：百度 OCR 把所需标记恢复从 7/14 提升到 13/14。"""
    txt = _ocr_text()
    got = [m for m in NEEDED_MARKS if m in txt]
    assert len(got) >= 13, f"恢复数退化：{len(got)}/14（交付时 7/14，接入 OCR 后 13/14），缺 {[m for m in NEEDED_MARKS if m not in txt]}"


def test_OCR里程碑_六题可机器判定且全对():
    d = json.loads(OCR_ANSWER.read_text(encoding="utf-8"))
    s = d["汇总"]
    assert s["题数"] == 6 and s["答对"] == 6, f"机器判定应 6/6，实得 {s['答对']}/{s['题数']}"
    assert d["附图标记恢复"]["本次恢复数"] >= 13


def test_坐标推理证据存在():
    d = json.loads(OCR_ANSWER.read_text(encoding="utf-8"))
    assert "坐标" in d["坐标推理证据"], "题3 的判定依据应为坐标比较"