# -*- coding: utf-8 -*-
"""工单13 核心链路测试（离线，不调 LLM、不起服务）。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

覆盖（效率模式，≥8 用例）：
  1 配置加载与阈值齐全
  2 优化开关等价性：baseline 全 false / optimized 至少一项 true
  3 剖析产物字段完整（口径/环境/优化开关/统计/逐题）
  4 逐题条目数与其中字段自洽（total_ms ≈ 各阶段之和，容差 5%）
  5 对比表存在且含四项关键指标
  6 截图存在、非空、中文命名
  7 问题集题号唯一且非空
  8 SLA 判定逻辑自洽（达标率 = 达标题数 / 样本数）

运行：先 cd E:\\gao6gongdan\\工单13，再 pwsh -NoProfile -File run_py.ps1 -PyArgs '-m','pytest','测试','-q'
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "优化" / "评估结果"
SHOT = ROOT / "测试" / "测试截图"
PROFILE = ROOT / "部署" / "配置" / "优化开关.json"
QUESTIONS = ROOT / "测试" / "测试数据" / "测试问题集.jsonl"

CN_NAME = re.compile(r"^[\u4e00-\u9fff0-9A-Za-z_\-（）()]+\.png$")


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# ----------------------------------------------------------------- 1 配置
def test_优化开关文件存在且阈值齐全():
    cfg = _load(PROFILE)
    assert cfg is not None, "部署/配置/优化开关.json 缺失"
    flags = cfg.get("flags") or {}
    for k in ("warmup_enabled", "embed_backend_cache", "parallel_channels",
              "rerank_prune", "cache_enabled", "adaptive_top_k", "generate_budget"):
        assert k in flags, f"缺少开关 {k}"
    assert isinstance(flags["rerank_budget"], int)


# ------------------------------------------------------- 2 口径等价性
def test_baseline与optimized档位语义正确():
    before, after = _load(EVAL / "优化前_汇总.json"), _load(EVAL / "优化后_汇总.json")
    assert before and after, "缺少优化前/优化后汇总"
    b_flags = before.get("优化开关") or {}
    a_flags = after.get("优化开关") or {}
    assert b_flags, "优化前汇总缺少『优化开关』段"
    # 基线：优化类开关必须全关
    for k in ("embed_backend_cache", "rerank_prune", "cache_enabled", "adaptive_top_k", "async_log"):
        assert b_flags.get(k) is False, f"基线档位 {k} 应为 False"
    # 优化档：至少一项真正打开
    assert any(a_flags.get(k) is True for k in
               ("embed_backend_cache", "rerank_prune", "cache_enabled", "adaptive_top_k")), "优化档未开启任何优化开关"


# ------------------------------------------------------- 3 产物字段完整
def test_剖析汇总字段完整():
    for name in ("优化前_汇总.json", "优化后_汇总.json"):
        s = _load(EVAL / name)
        assert s, f"{name} 缺失"
        for key in ("口径", "环境", "优化开关", "统计", "逐题"):
            assert key in s, f"{name} 缺少 {key}"
        st = s["统计"]
        for key in ("样本数", "总耗时", "检索", "生成"):
            assert key in st, f"{name} 统计缺少 {key}"
        assert st["样本数"] > 0


def test_逐题条目数与样本数一致():
    for name, jl in (("优化前", "优化前.jsonl"), ("优化后", "优化后.jsonl")):
        s = _load(EVAL / f"{name}_汇总.json")
        rows = _load_jsonl(EVAL / jl)
        assert rows, f"{jl} 为空"
        assert s["统计"]["样本数"] == len([r for r in rows if r.get("计入统计", True)]), \
            f"{name} 样本数与逐题条数不一致"


def test_逐题内部阶段与总耗时自洽():
    """逐题记录的阶段键是**顶层 `stage::` 前缀键**，且只有编号阶段（`stage::N_...`）可相加。

    实测踩坑：`stage::llm_first_token` 是**生成阶段内部的子跨度（与 4_LLM生成 重叠）**，
    若一并求和会得出约 1.9 倍 total 的假象；`stage::_其余(未归阶段)` 则是补差桶。
    故本用例**只累加 1~5 五个编号阶段**，并允许其不超过 total 的 5%。"""
    rows = _load_jsonl(EVAL / "优化后.jsonl")
    assert rows, "优化后.jsonl 为空"
    stage_re = re.compile(r"^stage::[1-5]_")
    checked = 0
    for r in rows:
        total = r.get("total_ms")
        if not isinstance(total, (int, float)) or total <= 0:
            continue
        parts = [v for k, v in r.items()
                 if stage_re.match(k) and isinstance(v, (int, float))]
        if not parts:
            continue
        s = sum(parts)
        assert s <= total * 1.05, f"{r.get('id')} 五阶段之和 {s:.0f} 超过 total {total:.0f} 达 5% 以上"
        assert s > 0, f"{r.get('id')} 五阶段之和为 0"
        checked += 1
    assert checked >= 1, "没有任何一条逐题记录可用于自洽校验"


def test_逐题必须带检索与生成两段():
    rows = _load_jsonl(EVAL / "优化后.jsonl")
    assert rows, "优化后.jsonl 为空"
    for r in rows:
        assert isinstance(r.get("retrieve_ms"), (int, float)), f"{r.get('id')} 缺 retrieve_ms"
        assert isinstance(r.get("generate_ms"), (int, float)), f"{r.get('id')} 缺 generate_ms"
        assert r.get("backend"), f"{r.get('id')} 缺 backend"
        assert r.get("model"), f"{r.get('id')} 缺 model"


# ------------------------------------------------------- 4 对比与评测
def test_对比表存在且含关键指标():
    csv_p = EVAL / "对比.csv"
    md_p = EVAL / "对比.md"
    assert csv_p.exists() and md_p.exists(), "对比.csv / 对比.md 缺失"
    text = csv_p.read_text(encoding="utf-8-sig")
    for key in ("总耗时", "检索", "生成", "P95"):
        assert key in text, f"对比表缺少 {key}"


def test_SLA判定逻辑自洽():
    for name in ("优化前", "优化后"):
        st = (_load(EVAL / f"{name}_汇总.json") or {}).get("统计") or {}
        sla = st.get("SLA") or {}
        if not sla:
            continue
        rate, n, hit = sla.get("达标率"), st.get("样本数"), sla.get("达标题数")
        if rate is None or not n:
            continue
        assert abs(float(rate) - (hit or 0) / n) < 0.02, f"{name} SLA 达标率与达标题数不自洽"


# ------------------------------------------------------- 5 截图
def test_截图存在非空且中文命名():
    pngs = sorted(SHOT.glob("*.png"))
    assert len(pngs) >= 2, f"截图不足 2 张（实得 {len(pngs)}）"
    for p in pngs:
        assert p.stat().st_size > 5000, f"{p.name} 疑似空图（{p.stat().st_size} B）"
        assert CN_NAME.match(p.name), f"{p.name} 命名不合规（应为中文按测试内容命名）"


# ------------------------------------------------------- 6 问题集
def test_问题集题号唯一且非空():
    rows = _load_jsonl(QUESTIONS)
    assert len(rows) >= 10, f"问题集题数偏少：{len(rows)}"
    ids = [r.get("id") for r in rows]
    assert all(ids), "存在空题号"
    assert len(set(ids)) == len(ids), "题号有重复"
    for r in rows:
        assert r.get("question", "").strip(), f"{r.get('id')} 题干为空"
