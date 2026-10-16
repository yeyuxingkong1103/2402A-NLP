# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/离线/test_acceptance_offline.py（T6：离线验收层，不依赖网络/LLM/服务）

覆盖：
  A. golden 冻结版本锚定（第 4 版哈希）与 16 题完整性
  B. 索引版本一致性（2676 / image 77 / vectors 行数 / page_basis）与两道图像题证据块在场
  C. 引用可回溯（引用的物理页必须在 PDF 页数内）
  D. 交付物齐备（报告 / CSV / JSON / 25 张截图 / README）与「不得声称 OCR」措辞检查
运行：pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -q
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from conftest import DATA, GOLDEN_SHA256_ANCHOR, INDEX_CHUNK_TOTAL, INDEX_IMAGE_CHUNKS, ROOT, SHOTS

#: 命名口径：题号NN = **工单题 id 升序序号**（01=id1 发行股数 … 16=id957 重要供应商领域）
EXPECTED_SHOTS = [
    "题号01_发行股数_回答.png", "题号02_募投项目_回答.png", "题号03_控制关系关联方_回答.png",
    "题号04_非控制关联方_回答.png", "题号05_组织结构图_回答.png", "题号06_IC市场增长率_回答.png",
    "题号07_军用收入占比_回答.png", "题号08_行业上游_回答.png", "题号09_技术标准_回答.png",
    "题号10_补充流动资金_回答.png", "题号11_军用领域收入_回答.png", "题号12_法定代表人_回答.png",
    "题号13_注册资本_回答.png", "题号14_行业下游_回答.png", "题号15_国家科技进步一等奖_回答.png",
    "题号16_重要供应商领域_回答.png",
    "题号05_组织结构图_检索精度.png", "题号06_IC市场增长率_检索精度.png",
    "拒答红线.png", "检索精确度.png", "准确率汇总.png", "响应时间.png",
    "图像解析结果.png", "中英文问答.png", "日志可审计性.png",
]


# ---------------------------------------------------------------- A. golden 版本锚定
def test_golden_anchor(records):
    """A：golden 必须仍是第 4 版（哈希锚定），16 题齐备。"""
    a = (ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl").read_bytes()
    b = (ROOT / "测试" / "测试数据" / "golden_qa_16.jsonl").read_bytes()
    assert a == b, "golden 主真值与镜像不一致"
    assert hashlib.sha256(a).hexdigest() == GOLDEN_SHA256_ANCHOR, \
        "golden 版本已变——评测结论必须重跑（T6 锚定第 4 版）"
    assert len(records) == 16
    assert [r["id"] for r in records] == [5, 6, 1, 2, 3, 4, 260, 95, 33, 34, 957, 793, 795, 543, 531, 207]


# ---------------------------------------------------------------- B. 索引版本与图像证据块
def test_index_version_and_image_chunks(index_meta):
    """B：索引为 20:42:06 重建版（2676 / image 77），且三处计数自洽（供版本追溯）。"""
    assert index_meta.get("chunk_total") == INDEX_CHUNK_TOTAL
    assert (index_meta.get("chunk_by_modality") or {}).get("image") == INDEX_IMAGE_CHUNKS
    assert index_meta.get("bm25", {}).get("count") == INDEX_CHUNK_TOTAL
    assert index_meta.get("vector", {}).get("count") == INDEX_CHUNK_TOTAL
    assert index_meta.get("page_basis") == "1-based-physical"


def test_image_evidence_chunks_present(golden, by_id, diagnostic):
    """B：两道图像题的**完整证据块**在索引里在场，且其正文满足权威层 hit_complete。"""
    chunks_path = ROOT / "研发" / "data" / "index" / "bge-m3_1024" / "chunks_indexed.jsonl"
    assert chunks_path.exists(), f"缺 {chunks_path}"
    want = {"pdf2-p39-img-x-0": 5, "pdf2-p72-img-1692-0": 6}
    found = {}
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cid = d.get("chunk_id") or d.get("id")
            if cid in want:
                found[cid] = d
    assert set(found) == set(want), f"索引中缺图像证据块：{set(want) - set(found)}"
    for cid, qid in want.items():
        rec = by_id[qid]
        text = found[cid].get("content") or found[cid].get("text") or ""
        layers = golden.check_retrieval_hit_layers(rec, [{"page": found[cid].get("page"), "text": text}],
                                                   diagnostic=diagnostic)
        assert layers["hit_complete"] is True, f"{cid} 未满足权威层（id {qid}）"


# ---------------------------------------------------------------- C. 引用可回溯（离线复核）
def test_citation_pages_renderable(artifacts, pdfs):
    """C：逐题引用页必须落在 PDF 页数内（可渲染回原文），且引用合格口径为「该页承载证据」。"""
    rows = artifacts["qa_results.jsonl"]
    assert len(rows) == 16
    for r in rows:
        pdf_name = r["pdf"] + ".pdf"
        assert pdf_name in pdfs, f"未知语料 {pdf_name}"
        total = pdfs[pdf_name].page_count
        for p in r.get("cited_pages") or []:
            assert isinstance(p, int) and 1 <= p <= total, f"id {r['id']}: 引用页 {p} 超出 {pdf_name} {total} 页"


def test_artifact_consistency(artifacts):
    """C：CSV / JSONL / accuracy_report 三处数字必须一致（防止报告与数据脱节）。"""
    rows = artifacts["qa_results.jsonl"]
    summary = artifacts["accuracy_report.json"]["summary"]
    with (DATA / "qa_results.csv").open(encoding="utf-8-sig", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(rows) == 16
    assert summary["n_questions"] == 16
    assert summary["n_correct"] == sum(1 for r in rows if r.get("correct") is True)
    assert summary["n_hit_complete"] == sum(1 for r in rows if r.get("hit_complete"))
    assert summary["accuracy"] == pytest.approx(summary["n_correct"] / 16, abs=1e-4)
    for c, j in zip(csv_rows, rows):
        assert c["id"] == str(j["id"])
        assert c["correct"] == str(j["correct"])


# ---------------------------------------------------------------- D. 交付物齐备
def test_deliverables_exist(artifacts):
    """D：测试报告 / CSV / JSON / 测试数据结构齐备。"""
    for p in (ROOT / "测试" / "测试报告.md", DATA / "qa_results.csv", DATA / "accuracy_report.json",
              DATA / "negatives.jsonl", DATA / "bilingual.jsonl", DATA / "run_meta.json",
              ROOT / "README.md", ROOT / "测试" / "README.md"):
        assert p.exists(), f"缺交付物 {p}"
    report = (ROOT / "测试" / "测试报告.md").read_text(encoding="utf-8")
    for must in ("16 题", "准确率", "P50", "P95", "检索命中", "拒答", "中英文", "判分口径"):
        assert must in report, f"测试报告缺关键小节/字段：{must}"


def test_screenshots_complete_and_real():
    """D：25 张截图齐备、为真实 PNG（魔数 + 体积下限），命名对应测试内容。"""
    missing = [n for n in EXPECTED_SHOTS if not (SHOTS / n).exists()]
    assert not missing, f"缺截图：{missing}"
    for name in EXPECTED_SHOTS:
        p = SHOTS / name
        data = p.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{name} 不是 PNG（疑似伪造）"
        assert len(data) > 20000, f"{name} 仅 {len(data)} B，疑似空白/占位图"
    assert (SHOTS / "题号05_组织结构图_检索精度.png").stat().st_size > 20000
    # 同图双名/双编号冲突回归：不得存在字节完全相同的两张截图
    import hashlib
    digests = {}
    for f in SHOTS.glob("*.png"):
        d = hashlib.sha256(f.read_bytes()).hexdigest()
        assert d not in digests, f"截图内容重复（双编号冲突）：{f.name} ≡ {digests[d]}"
        digests[d] = f.name


def test_no_fake_ocr_claim():
    """D：交付物不得声称使用 OCR（本机无 OCR 依赖，T2 登记为已知限制）。"""
    texts = {p.name: p.read_text(encoding="utf-8", errors="ignore")
             for p in [ROOT / "测试" / "测试报告.md", ROOT / "README.md"]}
    report_text = texts["测试报告.md"]
    for name, text in texts.items():
        for m in re.finditer(r"[^。\n]{0,40}OCR[^。\n]{0,40}", text):
            frag = m.group(0)
            ok = any(k in frag for k in ("无 OCR", "不得声称", "不依赖", "字形模板匹配", "未使用 OCR",
                                         "非 OCR", "本机无"))
            assert ok, f"{name} 中出现可能构成「声称使用 OCR」的表述：{frag}"
    assert "字形模板" in report_text or "无 OCR" in report_text
