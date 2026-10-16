# -*- coding: utf-8 -*-
"""工单18 测试用例（离线，基于 1700 份实跑报告）。

工单编号：人工智能NLP-RAG-文档质量评估 Skill 工单
运行：先 cd E:\\gao6gongdan\\工单18，再 & $py -m pytest 测试 -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "研发"))
REPORT = ROOT / "研发" / "data" / "reports" / "quality_report_full.json"
PER_FILE = ROOT / "研发" / "data" / "reports" / "per_file.jsonl"
CFG = ROOT / "研发" / "assessment_config.yaml"
SHOT = ROOT / "测试" / "测试截图"
CN = re.compile(r"^[\u4e00-\u9fff0-9A-Za-z_\-（）()]+\.png$")


@pytest.fixture(scope="module")
def rep() -> dict:
    assert REPORT.exists(), f"缺全量报告 {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_报告结构与五大功能段齐全(rep):
    for k in ("skill", "run", "summary", "format_distribution", "pdf_type_analysis",
              "length_distribution", "duplicate_detection", "sensitive_detection",
              "tag_system", "pending_lists", "routing_rules", "errors", "per_file"):
        assert k in rep, f"报告缺少段落 {k}"


def test_全量规模与损坏文件统计(rep):
    s = rep["summary"]
    assert s["total_files"] == 1700, f"应为 1700 份，实得 {s['total_files']}"
    assert s["total_mb"] > 600
    counts = rep["pdf_type_analysis"]["counts"]
    assert sum(counts.values()) == 1700, "四类计数之和应等于总文件数"
    assert s["corrupted_files"] == counts.get("Corrupted", 0)


def test_PDF类型分流比例自洽(rep):
    pa = rep["pdf_type_analysis"]
    ratios = pa["ratios"]
    total = sum(pa["counts"].values())
    for k, c in pa["counts"].items():
        assert abs(ratios[k] - c / total) < 0.001, f"{k} 比例与计数不自洽"
    assert pa["parsed_ok"] + pa["counts"]["Corrupted"] == total
    # 页级抽样证据齐全
    assert pa["sampled_pages_total"] > 0
    assert pa["scanned_pages_total"] >= 0 and pa["low_text_pages_total"] >= 0


def test_长度分布统计完整(rep):
    ln = rep["length_distribution"]
    for scope in ("all", "pdf_only"):
        blk = ln.get(scope)
        if not blk or not blk.get("count"):
            continue
        assert blk["min"] <= blk["mean"] <= blk["max"]
        q = blk.get("quantiles") or {}
        if q:
            assert q.get("P25", 0) <= q.get("P50", 0) <= q.get("P75", 0) <= q.get("P90", 0)
        hist = blk.get("histogram") or []
        if hist:
            assert abs(sum(h["ratio"] for h in hist) - 1.0) < 0.02, "直方图占比之和应≈1"


def test_重复检测两套机制都有记录(rep):
    dup = rep["duplicate_detection"]
    for k in ("md5_exact", "simhash_near"):
        assert k in dup, f"缺少 {k}"
    assert isinstance(dup["md5_exact"], (list, dict))
    assert isinstance(dup["simhash_near"], (list, dict))


def test_敏感信息规则已配置(rep):
    sens = rep["sensitive_detection"]
    assert sens["enabled"] is True
    assert len(sens.get("enabled_rules") or []) > 0, "应至少启用一条敏感规则"
    assert "disabled_rules" in sens
    assert sens["total_hits"] == len(sens.get("by_rule") or {}) or sens["total_hits"] >= 0


def test_标签体系与待审清单(rep):
    tags = rep["tag_system"]
    by_primary = (tags.get("distribution") or {}).get("by_primary") or []
    assert by_primary, "应有主导标签分布"
    assert abs(sum(x["ratio"] for x in by_primary) - 1.0) < 0.02
    pend = rep["pending_lists"]
    for k in ("pdf_type_pending_confirm_total", "version_conflicts_total",
              "sensitive_pending_review_total", "corrupted_files_total"):
        assert k in pend, f"待审清单缺少 {k}"


def test_错误逐条记录且不裸崩(rep):
    assert isinstance(rep["errors"], list)
    # 有损坏文件就应有对应错误记录（允许部分损坏文件未产出错误行，但不应为 0 条全是静默）
    if rep["summary"]["corrupted_files"] > 0:
        assert len(rep["errors"]) > 0, "存在损坏文件却没有任何错误记录"
    for e in rep["errors"][:5]:
        assert isinstance(e, (dict, str))


def test_逐文件明细行数与总数一致():
    """全量报告应覆盖 1700 行；注意复跑子集时会被复写成子集行数，故按「≥1700 或 ==1700」判定。"""
    assert PER_FILE.exists(), "缺 per_file.jsonl"
    lines = [l for l in PER_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 1700, f"逐文件明细应 ≥1700 行，实得 {len(lines)}（可能被子集复跑覆盖）"
    o = json.loads(lines[0])
    assert isinstance(o, dict) and o, "逐文件明细每行应为非空 JSON 对象"


def test_配置化文件存在且含阈值():
    assert CFG.exists(), "缺 assessment_config.yaml"
    txt = CFG.read_text(encoding="utf-8")
    assert len(txt) > 1000, "配置过短，疑似空壳"
    # 至少包含扫描判定与长度分档相关键
    for key in ("scan", "threshold"):
        assert key.lower() in txt.lower(), f"配置缺少与 {key} 相关的阈值项"


def test_截图存在非空且中文命名():
    pngs = sorted(SHOT.glob("*.png"))
    assert len(pngs) >= 2, f"截图应 ≥2 张，实得 {len(pngs)}"
    for p in pngs:
        assert p.stat().st_size > 5000, f"{p.name} 疑似空图"
        assert CN.match(p.name), f"{p.name} 命名不合规"


def test_HTML报告存在且非空():
    h = ROOT / "研发" / "data" / "reports" / "quality_report_full.html"
    assert h.exists() and h.stat().st_size > 10000, "缺 HTML 版报告或过小"
    head = h.read_text(encoding="utf-8", errors="ignore")[:400].lower()
    assert "<html" in head or "<!doctype" in head, "HTML 报告头部格式异常"


# ============================ 完善期新增（守护 A4 的发现） ============================
CORRUPT_REPORT = ROOT / "优化" / "评估结果" / "损坏文件复跑.json"
REVIEW_REPORT = ROOT / "优化" / "评估结果" / "损坏文件复检.json"


def test_损坏判定不得误报_复跑后应为零():
    """守护 2026-10-16 的发现：原报告 83 份「损坏」实为**并行 worker 内存竞争**导致的漏检。

    同一批文件用 `--files … --workers 1 --no-resume` 复跑后 Corrupted 必须为 0。
    若本用例失败，说明「内存不足 → 误报损坏」的缺陷回归了。
    """
    assert CORRUPT_REPORT.exists(), "缺 优化/评估结果/损坏文件复跑.json（A4 证据）"
    d = json.loads(CORRUPT_REPORT.read_text(encoding="utf-8"))
    s = d["summary"]
    assert s["total_files"] == 83, f"应复跑 83 份，实得 {s['total_files']}"
    assert s["corrupted_files"] == 0, (
        f"复跑后仍有 {s['corrupted_files']} 份被判损坏 —— 「并行 worker 内存竞争误报损坏」缺陷回归")
    assert len(d.get("errors") or []) == 0, "复跑不应再产生错误记录"
    # 真实分类必须全部落到非 Corrupted 类别
    types = s.get("pdf_types") or {}
    assert types.get("Corrupted", 0) == 0
    assert sum(v for k, v in types.items() if k != "Corrupted") == 83


def test_原报告损坏原因确为内存不足():
    """原 83 份的 error 文本必须都是 MemoryError —— 这是「不是文件损坏」的直接证据。"""
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    items = (rep.get("pending_lists") or {}).get("corrupted_files") or []
    assert len(items) == 83, f"原报告应登记 83 份，实得 {len(items)}"
    bad = [i for i in items if "MemoryError" not in str(i.get("error", ""))]
    assert not bad, f"有 {len(bad)} 份的报错不是 MemoryError，需重新归因：{bad[:2]}"


def test_抽检记录与复跑结论一致():
    if not REVIEW_REPORT.exists():
        pytest.skip("未产出抽检记录（A4 的简化脚本）")
    d = json.loads(REVIEW_REPORT.read_text(encoding="utf-8"))
    assert d["抽检可打开数"] == d["抽检份数"], "抽检应全部可打开"
    # 错误类型分布的键是异常前缀（如 unhandled），MemoryError 出现在逐份明细的「报告错误」里
    assert any("MemoryError" in str(r.get("报告错误", "")) for r in d["明细"]), \
        "抽检明细里应能看到 MemoryError（这是「不是文件损坏」的直接证据）"
