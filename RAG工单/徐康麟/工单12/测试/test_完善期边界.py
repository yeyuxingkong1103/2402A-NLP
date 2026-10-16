# -*- coding: utf-8 -*-
"""工单12 完善期新增测试：边界与异常路径（2026-10-16）。

工单编号：人工智能NLP-RAG 项目-LightRAG 优化
运行：先 cd E:\\gao6gongdan\\工单12，再 & $py -m pytest 测试 -q

覆盖：JSONL 坏行容错、图谱悬挂边、文档引用截图一致性、lightrag 与工单7 基座解耦、
以及 A 项处置（未接线的继承 app 是否已移出并有说明）。
"""
from __future__ import annotations

import json
import pathlib as _pl
import re as _re
import subprocess as _sp
import sys as _sys

import pytest

REPO = _pl.Path(__file__).resolve().parents[1]   # 工单12 根
DEV = REPO / "研发"                               # 研发目录
SHOT = REPO / "测试" / "测试截图"


def test_JSONL加载_坏行不裸崩():
    """产物 JSONL 应逐行可解析（本用例同时是「坏行」的守门人）。"""
    for rel in ("data/processed/chunks.jsonl", "data/graph/nodes.jsonl", "data/graph/edges.jsonl"):
        p = DEV / rel
        if not p.exists():
            continue
        bad, n = [], 0
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if not line.strip():
                continue
            n += 1
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                bad.append(f"{rel}:{i} {e.msg}")
        assert n > 0, f"{rel} 为空"
        assert not bad, f"存在无法解析的行：{bad[:3]}"


def test_图谱_边端点在节点集中闭合():
    """完整性：关系两端实体必须存在于节点集合（防悬挂边）。"""
    nodes_p = DEV / "data" / "graph" / "nodes.jsonl"
    edges_p = DEV / "data" / "graph" / "edges.jsonl"
    if not (nodes_p.exists() and edges_p.exists()):
        pytest.skip("缺图谱产物")
    names = set()
    for line in nodes_p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            o = json.loads(line)
            for k in ("name", "canonical_name", "id"):
                if o.get(k):
                    names.add(str(o[k]))
    dangling = []
    for line in edges_p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        for k in ("source", "target", "head", "tail", "src", "dst"):
            v = o.get(k)
            if v is not None and str(v) not in names:
                dangling.append(f"{k}={v}")
    assert not dangling, f"存在悬挂边端点：{dangling[:5]}"


def test_截图_文档引用文件名与磁盘一致():
    """联动：文档提到的 .png 必须真实存在（防引用悬空）。"""
    names = {p.name for p in SHOT.glob("*.png")}
    assert names, "截图目录为空"
    docs = (list(REPO.glob("*.md")) + list((REPO / "优化").rglob("*.md"))
            + list((REPO / "测试").rglob("*.md")) + list((REPO / "部署").rglob("*.md")))
    missing = []
    for doc in docs:
        for m in _re.finditer(r"([\u4e00-\u9fff\w\-]+\.png)",
                              doc.read_text(encoding="utf-8", errors="ignore")):
            if m.group(1) not in names:
                missing.append(f"{doc.name} → {m.group(1)}")
    assert not missing, f"文档引用的截图不存在：{missing[:4]}"


def test_lightrag导入_不牵连工单7基座():
    """解耦验证：干净子进程里只加 研发/ 到 sys.path，导入 lightrag 不应依赖工单7 的 app。"""
    code = ("import sys; sys.path.insert(0, r'%s'); import lightrag; "
            "print('OK', len(getattr(lightrag, '__all__', [])))" % str(DEV))
    r = _sp.run([_sys.executable, "-c", code], capture_output=True, text=True,
                timeout=90, cwd=str(REPO))
    assert r.returncode == 0, f"lightrag 导入失败：{r.stderr[-300:]}"
    assert "OK" in r.stdout


def test_未接线app_已移出且有说明():
    """守护 A 项处置：继承自工单7 的 app 已移出研发根，并留有说明文档。"""
    moved = DEV / "_未接线_app_继承自工单7"
    assert moved.is_dir(), "app 应已移出到 _未接线_app_继承自工单7/"
    assert not (DEV / "app").exists(), "研发根下不应再有 app/（会误导为可用）"
    note = moved / "README_为何未接线.md"
    assert note.exists() and note.stat().st_size > 500, "应有说明文档解释为何未接线"
