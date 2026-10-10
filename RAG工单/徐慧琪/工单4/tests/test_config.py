# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import pytest
from pathlib import Path
from rag04.config import (
    PROJECT_ROOT,
    WORK_ORDER_ID,
    get_settings,
    latency_verdict,
)


def test_work_order_id_matches_spec():
    assert WORK_ORDER_ID == "人工智能NLP-RAG-图像内容解析及检索优化"


def test_default_mode_is_full_04():
    s = get_settings()
    assert s.pipeline_mode == "full_04"


def test_baseline_mode_switches_flags():
    s = get_settings("baseline_03")
    assert s.pipeline_mode == "baseline_03"
    assert s.use_figures is False
    assert s.use_rerank is False
    assert s.use_hybrid is False
    assert s.fixed_chunk_size == 512


def test_full_mode_flags():
    s = get_settings("full_04")
    assert s.use_figures is True
    assert s.use_rerank is True
    assert s.use_hybrid is True


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="pipeline_mode"):
        get_settings("nope")


def test_global_constraints_come_from_settings():
    s = get_settings()
    assert s.vlm_max_tokens >= 2000      # 硬约束：过小会被思维链截断
    assert s.crop_margin >= 0.15         # 硬约束：防静默丢内容
    assert 200 <= s.render_dpi <= 220    # 硬约束：竖排小字可识别区间
    assert s.embed_batch == 16
    assert s.rerank_top_n == 30
    assert s.final_top_k == 5
    assert s.k_report == (3, 5, 10)
    assert s.answer_acc_threshold == 0.80


def test_paths_are_under_project_root():
    s = get_settings()
    assert s.data_dir.is_absolute()
    assert str(s.data_dir).startswith(str(PROJECT_ROOT))


# --- 终审修正B：时延判定唯一实现（接口 / 界面 / 压测共用，勿各自比较）---


def test_latency_verdict_rounds_before_comparing():
    """先取到 1 位小数（展示精度）再比较：3000.04 显示 3000.0 就必须算达标。"""
    assert latency_verdict(3000.04, 3000.0) == (3000.0, True)
    assert latency_verdict(2999.96, 3000.0) == (3000.0, True)   # 进位后仍达标
    assert latency_verdict(2500.04, 3000.0) == (2500.0, True)


def test_latency_verdict_over_budget_is_rounded_not_clamped():
    """超预算只做展示取整，绝不放宽成达标，也不截断到预算内。"""
    assert latency_verdict(3000.4, 3000.0) == (3000.4, False)
    assert latency_verdict(3000.06, 3000.0) == (3000.1, False)
    assert latency_verdict(3888.64, 3000.0) == (3888.6, False)   # 实测 p50 档位
    assert latency_verdict(13385.0, 3000.0) == (13385.0, False)  # 实测 p95 档位


def test_latency_verdict_boundary_equals_budget_is_within():
    """边界含等号（与历史口径一致：latency_ms == budget 视为达标）。"""
    assert latency_verdict(3000.0, 3000.0) == (3000.0, True)
    assert latency_verdict(0.0, 3000.0) == (0.0, True)


# --- 时延硬指标的隐藏杀手：Ollama 地址必须是 IPv4 字面量 ---


def test_ollama_url_is_ipv4_literal_not_localhost():
    """localhost 在本机解析为 ::1 优先，而 Ollama 只监听 IPv4。

    embed.py 的 _post_embed 用未复用连接的 requests.post，所以每次嵌入都要
    先等 IPv6 连接失败再回退：实测 localhost 2205/2269ms vs
    127.0.0.1 229/229/211ms，每问固定多付约 2 秒（占 3 秒预算的 2/3）。
    这个默认值一旦被改回 localhost，端到端时延会静默劣化 —— 用测试钉住。
    """
    s = get_settings()
    assert s.ollama_url.startswith("http://127.0.0.1:"), (
        f"ollama_url 应使用 IPv4 字面量，当前 {s.ollama_url!r}；"
        "写 localhost 会在 Windows 上每次嵌入多付约 2 秒"
    )
    assert "localhost" not in s.ollama_url
