# -*- coding: utf-8 -*-
"""DocumentQualityAssessmentSkill 单元测试。

覆盖：格式统计、PDF 三分类、长度分位、MD5/SimHash、敏感信息、
损坏文件容错、标签路由、API 端点端到端调用。
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import pymupdf

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from document_quality_assessment import DocumentQualityAssessment
from document_quality_assessment.duplicate_detector import hamming, md5_groups, simhash
from document_quality_assessment.format_stats import collect_files, format_distribution
from document_quality_assessment.length_analysis import length_distribution
from document_quality_assessment.sensitive_detector import detect_sensitive
from document_quality_assessment.workflow import DocumentIngestionWorkflow
from document_quality_assessment.report import render_html
from api_server import QualityHandler


# ---------------- 夹具与工具 ----------------
_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_ID_CHECK = "10X98765432"


def valid_idcard(front17: str) -> str:
    s = sum(int(front17[i]) * _ID_WEIGHTS[i] for i in range(17))
    return front17 + _ID_CHECK[s % 11]


def make_pdf(path: Path, text_pages: int = 0, blank_pages: int = 0,
             line: str = "This is a patent text line for testing quality. "):
    doc = pymupdf.open()
    for _ in range(text_pages):
        page = doc.new_page()
        # 插入足够多字符（>100 非空白字符）
        page.insert_text((50, 50), line * 5, fontsize=11)
    for _ in range(blank_pages):
        doc.new_page()  # 空白页模拟扫描页
    doc.save(str(path))
    doc.close()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    make_pdf(tmp_path / "text.pdf", text_pages=3)
    make_pdf(tmp_path / "scan.pdf", blank_pages=4)
    make_pdf(tmp_path / "mixed.pdf", text_pages=2, blank_pages=2)
    (tmp_path / "note.md").write_text("# 标题\n" + "中文内容" * 100, encoding="utf-8")
    (tmp_path / "plain.txt").write_text("普通文本" * 50, encoding="utf-8")
    # 损坏 PDF
    (tmp_path / "broken.pdf").write_bytes(b"not a real pdf content" * 20)
    # 完全重复文件
    make_pdf(tmp_path / "dup_a.pdf", text_pages=1)
    import shutil
    shutil.copy(tmp_path / "dup_a.pdf", tmp_path / "dup_b.pdf")
    return tmp_path


def _run_skill(dataset: Path, tmp_path: Path) -> dict:
    skill = DocumentQualityAssessment()
    return skill.assess(target=str(dataset), output_dir=str(tmp_path / "out"),
                        resume=False).to_dict()


# ---------------- 测试用例 ----------------
def test_format_distribution(dataset):
    files, unsupported = collect_files(dataset, [".pdf", ".md", ".txt"], True)
    assert len(files) == 8
    dist = format_distribution(files)
    assert dist["total_files"] == 8
    pdf_item = next(x for x in dist["by_format"] if x["format"] == ".pdf")
    assert pdf_item["count"] == 6


def test_pdf_classification(dataset, tmp_path):
    report = _run_skill(dataset, tmp_path)
    by_name = {Path(r["name"]).name: r for r in report["records"]}
    assert "Text_PDF" in by_name["text.pdf"]["tags"]
    assert "Scan_PDF" in by_name["scan.pdf"]["tags"]
    assert "Hybrid_PDF" in by_name["mixed.pdf"]["tags"]
    assert by_name["scan.pdf"]["pdf"]["scanned_ratio"] == 1.0


def test_corrupt_pdf_no_crash(dataset, tmp_path):
    report = _run_skill(dataset, tmp_path)
    by_name = {Path(r["name"]).name: r for r in report["records"]}
    assert "Corrupt_PDF" in by_name["broken.pdf"]["tags"]
    assert by_name["broken.pdf"]["read_error"]


def test_length_percentiles(dataset):
    dist = length_distribution([100, 200, 300, 400, 1000],
                               {"percentiles": [25, 50, 75, 90, 99],
                                "bins": [500, 2000], "bin_labels": ["短", "中", "长"],
                                "empty_threshold": 10})
    assert dist["percentiles"]["P50"] == 300
    assert dist["bins_distribution"][0]["count"] == 4


def test_md5_groups(dataset, tmp_path):
    report = _run_skill(dataset, tmp_path)
    groups = report["duplicates"]["md5_groups"]
    assert len(groups) == 1
    names = sorted(Path(f).name for f in groups[0]["files"])
    assert names == ["dup_a.pdf", "dup_b.pdf"]


def test_simhash_distance():
    cfg = {"simhash_bits": 64, "ngram": 4, "ngram_step": 1, "max_chars_per_doc": 200000}
    base = "一种静电除尘器的制造方法包括壳体阳极板阴极线" * 20
    h1 = simhash(base, cfg)
    assert hamming(h1, simhash(base, cfg)) == 0
    h2 = simhash(base[: len(base) - 30], cfg)  # 截断一点
    assert hamming(h1, h2) <= 10
    h3 = simhash("完全不同的内容关于汽车发动机的传动系统设计" * 20, cfg)
    assert hamming(h1, h3) > 3


def test_sensitive_detection():
    cfg = {"phone": True, "email": True, "idcard": True, "bankcard": False,
           "context_chars": 40, "max_findings_per_type": 50}
    idc = valid_idcard("11010119900307051")
    text = f"联系电话 13812345678 或邮件 张三test@example.com 身份证 {idc} 结束"
    findings = detect_sensitive(text, cfg)
    types = {f["type"] for f in findings}
    assert {"手机号", "邮箱", "身份证"} <= types
    # 每条必须带上下文
    assert all(len(f["context"]) > 0 for f in findings)
    # 银行卡默认关闭
    assert not any(f["type"] == "银行卡" for f in findings)
    # 18 位非法随机数字不应命中身份证
    assert detect_sensitive("编号 123456789012345678 ok", cfg) == []


def test_bankcard_switch():
    cfg = {"phone": False, "email": False, "idcard": False, "bankcard": False,
           "context_chars": 40, "max_findings_per_type": 50}
    # 49927398716 是经典 Luhn 有效号（11位，银行卡正则要16位）
    num = "6212262201001234567"  # 19 位随机，多半 Luhn 不过
    assert detect_sensitive(num, cfg) == []


def test_routing_scan_to_ocr(dataset, tmp_path):
    report = _run_skill(dataset, tmp_path)
    wf = DocumentIngestionWorkflow(report)
    out = wf.run(str(dataset / "scan.pdf"))
    assert out["route"] == "OCRParser"
    nodes = [t["node"] for t in out["trace"]]
    assert nodes == ["TriggerNode", "DocumentQualityAssessmentSkill", "DecisionNode", "OCRParser"]
    assert out["result"]["status"] == "OCR完成并入库"


def test_html_report(dataset, tmp_path):
    report = _run_skill(dataset, tmp_path)
    h = render_html(report)
    assert "<html" in h and "待办1" in h


def test_api_endpoint(dataset, tmp_path):
    # 起服务（空闲端口）
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = ThreadingHTTPServer(("127.0.0.1", port), QualityHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.5)
    try:
        import urllib.request
        body = json.dumps({"path": str(dataset),
                           "output_dir": str(tmp_path / "api_out")}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/document/quality-inspection",
            data=body, headers={"Content-Type": "application/json"}, method="POST")
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert resp["summary"]["total_files"] == 8
        assert "action_lists" in resp
    finally:
        server.shutdown()


def test_checkpoint_resume(dataset, tmp_path):
    skill = DocumentQualityAssessment()
    out1 = str(tmp_path / "ck")
    skill.assess(target=str(dataset), output_dir=out1, resume=True)
    # 第二次应从 checkpoint 恢复且结果一致
    r2 = skill.assess(target=str(dataset), output_dir=out1, resume=True).to_dict()
    assert r2["summary"]["total_files"] == 8
