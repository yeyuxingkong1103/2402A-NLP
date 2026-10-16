# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/用户/test_user_acceptance.py（T6：用户验收层——模拟验收人照 README 走一遍）

覆盖（用户视角，不看内部实现）：
  · README 里写明的命令行方式能否一次跑通（CLI 提问 → 答案 + 中文页码引用）
  · CLI --health 是否 ok；Web /health 是否 ok
  · 五个中文文件夹与 README 是否齐备；交付物（报告/CSV/截图）是否齐备
  · 16 题准确率 ≥90%、延迟 ≤3s、拒答与中英文结论是否与报告一致
运行：pwsh -NoProfile -File run_py.ps1 -m pytest 测试/用户 -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import DATA, ROOT, SHOTS


def run_cli(args: list[str], timeout: float = 240.0) -> subprocess.CompletedProcess:
    """按 README 的方式调用 CLI（run_py.ps1 透传入参）。"""
    cmd = ["pwsh", "-NoProfile", "-File", str(ROOT / "run_py.ps1"), *args]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def test_deployment_readme_present():
    """跨环节交付物：部署/README.md（属部署环节交付；T6 只做存在性核对，不代写）。"""
    assert (ROOT / "部署" / "README.md").exists()


def test_readme_folders_and_readme():
    """用户视角：五个中文文件夹 + 顶层 README + 部署 README 齐备。"""
    for name in ("设计", "研发", "测试", "优化", "部署"):
        assert (ROOT / name).is_dir(), f"缺文件夹 {name}"
    assert (ROOT / "README.md").exists()
    assert (ROOT / "部署" / "日志" / "app.log").exists(), "缺 部署/日志/app.log（结构化日志证据）"


def test_cli_ask_once():
    """用户视角：一条命令问一道题，必须返回答案与 [文件名: 第N页] 形式的引用。"""
    cp = run_cli(["研发/app/main.py", "--ask", "武汉兴图新科电子股份有限公司注册资本是多少？"])
    assert cp.returncode == 0, f"CLI 退出码 {cp.returncode}\n{cp.stdout[-400:]}\n{cp.stderr[-400:]}"
    out = cp.stdout
    assert "5,520" in out, f"答案未见 5,520：{out[-300:]}"
    assert "第22页" in out or "第 22 页" in out, f"引用未见页码：{out[-300:]}"


def test_cli_health_ok():
    """用户视角：CLI --health 必须 status=ok（否则用户无法判断服务可用性）。"""
    cp = run_cli(["研发/app/main.py", "--health"])
    assert cp.returncode == 0, f"--health 退出码 {cp.returncode}"
    assert '"status": "ok"' in cp.stdout or '"status":"ok"' in cp.stdout, cp.stdout[-400:]


def test_web_health_ok(service):
    """用户视角：Web /health 必须 status=ok 且报出索引规模（版本可追溯）。"""
    h = service.health()
    assert h.get("status") == "ok"
    idx = h.get("checks", {}).get("index", {})
    assert idx.get("chunks") == 2676
    assert idx.get("image_chunks") == 77


def test_user_deliverables_complete():
    """用户视角：报告 / CSV / 截图 / 逐题明细齐备（验收要看得到东西）。"""
    assert (ROOT / "测试" / "测试报告.md").exists()
    assert (DATA / "qa_results.csv").exists()
    assert len(list(SHOTS.glob("*.png"))) >= 25, f"截图不足 25 张：{len(list(SHOTS.glob('*.png')))}"
    rows = [json.loads(x) for x in (DATA / "qa_results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 16, "逐题明细必须是 16 题"


def test_user_acceptance_numbers(artifacts):
    """用户视角：验收判据（准确率 ≥90%、逐题 ≤3s、拒答与中英文状态如实）与报告一致。"""
    s = artifacts["accuracy_report.json"]["summary"]
    assert s["accuracy"] >= 0.90, f"准确率 {s['accuracy']} 未达 90%"
    assert s["latency_wall_max_ms"] <= 3000.0
    assert s["n_latency_over_3000ms"] == 0
    neg = artifacts["negatives.jsonl"]
    assert len(neg) == 4, "负例必须 4 条"
    bi = artifacts["bilingual.jsonl"]
    assert len(bi) == 6, "中英文必须各 3 题"


def test_report_states_known_issues(artifacts):
    """用户视角：报告必须如实写明已知缺陷（N1 拒答缺陷 / 英文路径缺口），不得只报喜。"""
    report = (ROOT / "测试" / "测试报告.md").read_text(encoding="utf-8")
    assert "某某科技" in report or "泛化名" in report, "报告未写明 N1 拒答缺陷"
    assert "英文" in report and ("未通过" in report or "缺口" in report or "不完整" in report), \
        "报告未如实说明英文路径缺口"
    assert "无 OCR" in report or "不得声称" in report, "报告未写明无 OCR 限制"
