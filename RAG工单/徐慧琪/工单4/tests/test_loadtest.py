# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""并发压测脚本纯单测：分位数、错误率、预算计数与报告渲染。

本文件**不导入服务、不建 socket、不读索引**：`run_level` / `_one_request` 的
网络行为不在单测范围内（属 phase 2 实跑）。为让预算判定可单测，统计与汇总被拆成
纯函数 `_summarize`，压测循环只负责收集原始样本。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from loadtest import (  # noqa: E402
    DEFAULT_PER_LEVEL_FACTOR,
    BenchResult,
    _percentile,
    _server_latency,
    _server_refused,
    _summarize,
    budget_source,
    default_budget_ms,
    latency_verdict,
    resolve_per_level,
    write_report,
)


# ---------------------------------------------------------------- 分位数

def test_percentile_basic():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _percentile(data, 50) == 5.5
    assert _percentile(data, 100) == 10
    assert _percentile([], 95) == 0.0


def test_percentile_edge_cases():
    assert _percentile([7], 99) == 7.0                    # 单元素
    assert _percentile([10, 20, 30, 40], 0) == 10         # P0 = 最小值
    assert _percentile([40, 10, 30, 20], 100) == 40       # 乱序输入不影响
    assert _percentile(list(range(1, 101)), 95) == 95.05  # 线性插值


# ---------------------------------------------------------------- 数据类

def test_bench_result_positional_order_matches_brief():
    """brief 的位置参数顺序是外部契约（脚本与测试按位构造）。"""
    r = BenchResult(10, 100, 100, 0, 0.0, 10.0, 100, 200, 300, 150, 500, None)
    assert (r.concurrency, r.n_requests, r.ok, r.errors) == (10, 100, 100, 0)
    assert (r.error_rate, r.qps) == (0.0, 10.0)
    assert (r.p50_ms, r.p95_ms, r.p99_ms, r.mean_ms) == (100, 200, 300, 150)
    assert (r.rss_mb, r.gpu_mb) == (500, None)


def test_bench_result_error_rate():
    r = BenchResult(concurrency=10, n_requests=100, ok=90, errors=10,
                    error_rate=0.1, qps=5.0, p50_ms=1, p95_ms=2, p99_ms=3,
                    mean_ms=1.5, rss_mb=100, gpu_mb=None)
    assert r.error_rate == 0.1


def test_client_and_server_latency_are_not_conflated():
    """主口径是客户端墙钟；服务端上报并列留存，两者绝不互相顶替。"""
    r = BenchResult(10, 100, 100, 0, 0.0, 10.0, 1234.0, 2345.0, 3456.0, 1500.0,
                    500, None, budget_ms=3000.0, n_within_budget=88,
                    server_p50_ms=700.0, server_p95_ms=1900.0,
                    server_p99_ms=2900.0, server_mean_ms=900.0)
    assert (r.client_p50_ms, r.client_p95_ms) == (1234.0, 2345.0)
    assert (r.client_p99_ms, r.client_mean_ms) == (3456.0, 1500.0)
    assert (r.server_p50_ms, r.server_p99_ms) == (700.0, 2900.0)
    d = r.as_dict()
    assert d["client_p50_ms"] == 1234.0 and d["server_p50_ms"] == 700.0
    assert d["client_p99_ms"] == 3456.0 and d["server_p99_ms"] == 2900.0


# ---------------------------------------------------------------- 汇总（纯函数）

def test_summarize_counts_budget_on_client_side():
    """达标与否只看客户端墙钟：服务端再快也不能把超预算的请求判成达标。"""
    r = _summarize(
        concurrency=10, n_requests=4, ok_count=3,
        client_ms=[100.0, 2900.0, 3100.0, 5000.0],
        client_ok_ms=[100.0, 2900.0, 3100.0],
        server_ms=[800.0, 900.0, 1200.0],     # 服务端全部 < 3000
        elapsed_s=2.0, rss_mb=512.0, gpu_mb=None, budget_ms=3000.0,
    )
    assert r.n_within_budget == 2                     # 仅 100 / 2900
    assert r.within_budget_rate == 0.5
    assert r.errors == 1 and r.error_rate == 0.25
    assert r.qps == 2.0
    assert r.p50_ms == 3000.0                         # [100,2900,3100,5000] 线性插值
    assert r.server_p50_ms == 900.0
    assert r.gpu_mb is None                           # 测不到显存就如实为 None


def test_summarize_without_server_samples_reports_none():
    r = _summarize(concurrency=1, n_requests=2, ok_count=2,
                   client_ms=[10.0, 20.0], client_ok_ms=[10.0, 20.0],
                   server_ms=[], elapsed_s=0.5, rss_mb=1.0, gpu_mb=None,
                   budget_ms=3000.0)
    assert r.server_p50_ms is None and r.server_mean_ms is None
    assert r.server_p95_ms is None and r.server_p99_ms is None


def test_summarize_zero_requests_is_safe():
    r = _summarize(concurrency=1, n_requests=0, ok_count=0,
                   client_ms=[], client_ok_ms=[], server_ms=[], elapsed_s=0.0,
                   rss_mb=0.0, gpu_mb=None, budget_ms=3000.0)
    assert (r.error_rate, r.qps) == (0.0, 0.0)
    assert (r.p50_ms, r.p95_ms, r.p99_ms, r.mean_ms) == (0.0, 0.0, 0.0, 0.0)
    assert r.n_within_budget == 0 and r.within_budget_rate == 0.0


def test_summarize_never_fabricates_gpu_memory():
    r = _summarize(concurrency=1, n_requests=1, ok_count=1,
                   client_ms=[1.0], client_ok_ms=[1.0], server_ms=[1.0],
                   elapsed_s=1.0, rss_mb=10.0, gpu_mb=None, budget_ms=3000.0)
    assert r.gpu_mb is None and r.rss_mb == 10.0


def test_server_latency_parsing_is_defensive():
    assert _server_latency({"latency_ms": 123.4}) == 123.4
    assert _server_latency({"latency_ms": 200}) == 200.0        # int 可用
    assert _server_latency({}) is None                          # 字段缺失
    assert _server_latency({"latency_ms": None}) is None
    assert _server_latency({"latency_ms": "123"}) is None       # 字符串不猜
    assert _server_latency({"latency_ms": True}) is None        # bool 不是时延


def test_server_refused_parsing_is_strict():
    assert _server_refused({"refused": True}) is True
    assert _server_refused({"refused": False}) is False
    assert _server_refused({"refused": "true"}) is False        # 字符串不猜
    assert _server_refused({}) is False and _server_refused(None) is False


def test_summarize_records_inflight_refused_and_server_samples():
    r = _summarize(concurrency=100, n_requests=3, ok_count=3,
                   client_ms=[10.0, 20.0, 30.0], client_ok_ms=[10.0, 20.0, 30.0],
                   server_ms=[9.0, 19.0], elapsed_s=1.0, rss_mb=1.0, gpu_mb=None,
                   budget_ms=3000.0, max_inflight=61, refused_count=2)
    assert r.max_inflight == 61          # 真正施加过的并发上限
    assert r.refused == 2
    assert r.n_server_samples == 2       # 服务端列的分母
    d = r.as_dict()
    assert (d["max_inflight"], d["refused"], d["n_server_samples"]) == (61, 2, 2)


def test_loadtest_reuses_the_config_verdict_helper():
    """终审修正B：达标判定只有一份实现 —— 压测必须直接用 config 里的那个函数，
    不是「看起来等价」的本地副本（否则接口/界面/报告三处迟早分叉）。"""
    import rag04.config as cfg

    assert latency_verdict is cfg.latency_verdict
    assert latency_verdict(3000.04, 3000.0) == (3000.0, True)
    assert latency_verdict(3000.06, 3000.0) == (3000.1, False)


def test_summarize_counts_budget_with_the_shared_rounded_verdict():
    """3000.04ms 在接口里显示 3000.0 且判达标：压测报告必须同判（旧实现按原值判超标）。"""
    r = _summarize(concurrency=1, n_requests=3, ok_count=3,
                   client_ms=[3000.04, 3000.06, 3000.0],
                   client_ok_ms=[3000.04, 3000.06, 3000.0],
                   server_ms=[], elapsed_s=1.0, rss_mb=1.0, gpu_mb=None,
                   budget_ms=3000.0)
    assert r.n_within_budget == 2          # 3000.04→3000.0 ✓、3000.0 ✓、3000.06→3000.1 ✗
    assert r.within_budget_rate == pytest.approx(2 / 3)


def test_summarize_rejects_sample_count_mismatch():
    """样本数与计数不自洽时必须报错，绝不产出对不上的报告。"""
    base = {"concurrency": 1, "elapsed_s": 1.0, "rss_mb": 0.0, "gpu_mb": None,
            "budget_ms": 3000.0}
    with pytest.raises(ValueError):                     # 客户端样本少了
        _summarize(n_requests=2, ok_count=2, client_ms=[1.0],
                   client_ok_ms=[1.0], server_ms=[], **base)
    with pytest.raises(ValueError):                     # 成功样本与成功数不符
        _summarize(n_requests=1, ok_count=1, client_ms=[1.0],
                   client_ok_ms=[], server_ms=[], **base)
    with pytest.raises(ValueError):                     # 服务端样本多于成功数
        _summarize(n_requests=1, ok_count=1, client_ms=[1.0],
                   client_ok_ms=[1.0], server_ms=[1.0, 2.0], **base)


def test_default_budget_ms_comes_from_settings():
    from rag04.config import get_settings
    budget, label = budget_source()
    assert budget == get_settings().latency_budget_ms > 0
    assert label == "Settings.latency_budget_ms"        # 出处如实标注
    assert default_budget_ms() == budget


def test_budget_source_labels_fallback(monkeypatch):
    """配置读不到时必须自称兜底，不能冒充 Settings 值。"""
    monkeypatch.setitem(sys.modules, "rag04.config", None)
    value, label = budget_source()
    assert value == 3000.0 and "兜底" in label
    assert default_budget_ms() == 3000.0


def test_resolve_per_level_defaults_and_clamps():
    assert resolve_per_level(None, [10, 50, 100]) == (
        DEFAULT_PER_LEVEL_FACTOR * 100, "")              # 缺省 300，够 100 并发在飞
    assert resolve_per_level(300, [10, 50, 100]) == (300, "")
    value, warning = resolve_per_level(60, [10, 50, 100])
    assert value == 100 and "上调" in warning            # brief 的 60 会被上调到 100
    assert resolve_per_level(5, []) == (5, "")           # 空档位不炸


# ---------------------------------------------------------------- 报告渲染

def test_write_report_contains_all_levels(tmp_path):
    rs = [
        BenchResult(10, 100, 100, 0, 0.0, 10.0, 100, 200, 300, 150, 500, None),
        BenchResult(50, 100, 98, 2, 0.02, 8.0, 300, 600, 900, 400, 600, 512.0),
        BenchResult(100, 100, 95, 5, 0.05, 6.0, 500, 1200, 2000, 700, 700, 600.0),
    ]
    p = write_report(rs, tmp_path / "bench.md")
    txt = p.read_text(encoding="utf-8")
    for lv in ("10", "50", "100"):
        assert lv in txt
    assert "错误率" in txt
    assert "P95" in txt


def test_write_report_creates_parent_dir(tmp_path):
    r = _summarize(concurrency=1, n_requests=1, ok_count=1, client_ms=[1.0],
                   client_ok_ms=[1.0], server_ms=[1.0], elapsed_s=1.0,
                   rss_mb=1.0, gpu_mb=None, budget_ms=3000.0)
    p = write_report([r], tmp_path / "nested" / "deep" / "bench.md")
    assert p.exists() and p.read_text(encoding="utf-8").startswith("# 高并发压测报告")


def _result(concurrency: int, n_requests: int, ok: int, *, p50: float = 100.0,
            p95: float = 200.0, p99: float = 300.0, within: int | None = 0,
            budget: float = 3000.0, server_p95: float | None = 250.0,
            max_inflight: int | None = None, refused: int = 0,
            server_samples: int = 1, rss_mb: float = 100.0) -> BenchResult:
    """报告测试用的合成结果（不构造真实样本、不发请求）。

    ``max_inflight`` 缺省取标称并发（即「并发施加到位」的正常档位），需要模拟
    施加不足时显式传小值；``rss_mb=0`` 模拟 ``mem_snapshot`` 未测得。
    """
    inflight = concurrency if max_inflight is None else max_inflight
    return BenchResult(concurrency, n_requests, ok, n_requests - ok,
                       round((n_requests - ok) / n_requests, 4), 1.0, p50, p95, p99,
                       p50, rss_mb, None, budget_ms=budget, n_within_budget=within,
                       server_p50_ms=server_p95, server_p95_ms=server_p95,
                       server_p99_ms=server_p95, server_mean_ms=server_p95,
                       max_inflight=inflight, refused=refused,
                       n_server_samples=server_samples)


def _row(txt: str, level: int, n_cells: int) -> list[str]:
    """按列数取出某档那一行的单元格（结果表 17 列、预算表 8 列，靠列数区分）。"""
    for line in txt.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == n_cells and cells[0] == str(level):
            return cells
    raise AssertionError(f"报告里找不到并发 {level} 的 {n_cells} 列表格行")


def test_write_report_states_budget_verdict_per_level(tmp_path):
    """逐级给出「预算内请求数 / 总请求数」与达标率。"""
    rs = [_result(10, 8, 8, within=8), _result(50, 6, 5, within=2)]
    txt = write_report(rs, tmp_path / "bench.md").read_text(encoding="utf-8")
    assert "8/8" in txt and "2/6" in txt          # 逐级达标计数
    assert "33.3%" in txt                          # 2/6 达标率
    assert "100.0%" in txt                         # 8/8 达标率
    assert "预算" in txt


def test_write_report_names_single_threshold_once(tmp_path):
    """所有档位阈值一致时，标题写明该值并标注出处。"""
    rs = [_result(10, 10, 10, within=10), _result(50, 10, 10, within=10)]
    txt = write_report(rs, tmp_path / "bench.md").read_text(encoding="utf-8")
    assert "## 三、3000 ms 预算逐级判定" in txt
    assert "来源：Settings.latency_budget_ms" in txt


def test_write_report_mixed_thresholds_are_not_silently_unified(tmp_path):
    """逐级阈值不同时不得在标题里写单一数字（否则与逐行阈值互相矛盾）。"""
    rs = [_result(10, 10, 10, within=10, budget=3000.0),
          _result(50, 10, 10, within=10, budget=2000.0)]
    txt = write_report(rs, tmp_path / "bench.md").read_text(encoding="utf-8")
    assert "逐级不同（10→3000、50→2000）" in txt
    assert "## 三、3000 ms" not in txt
    assert _row(txt, 10, n_cells=8)[1] == "3000"
    assert _row(txt, 50, n_cells=8)[1] == "2000"


def test_write_report_verdict_uses_client_side_latency(tmp_path):
    """服务端 P95 达标、客户端 P95 超预算时，判定列必须按客户端判为部分达标。"""
    r = _result(10, 10, 10, p95=4500.0, within=1, server_p95=800.0)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    cells = _row(txt, 10, n_cells=8)                # 预算判定表那一行
    assert cells[1] == "3000"                       # 阈值列
    assert cells[2] == "1/10"                       # 达标数（客户端逐请求计数）
    assert cells[6] == "部分达标"                    # 判定列
    assert cells[7] == "4500.0"                     # 判定依据：客户端 P95
    res = _row(txt, 10, n_cells=17)                 # 结果表同时保留两组数
    assert res[9] == "4500.0" and res[13] == "800.0"


def test_write_report_flags_underdelivered_concurrency(tmp_path):
    """标称 100 并发、在飞峰值只到 61 时必须显式告警，不得写成 100 并发成绩。"""
    r = _result(100, 100, 100, within=100, max_inflight=61)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    assert "在飞峰值仅 61" in txt
    assert "未真正施加标称并发" in txt
    assert _row(txt, 100, n_cells=17)[1] == "61"    # 结果表在飞峰值列


def test_write_report_notes_unrecorded_inflight(tmp_path):
    """手工构造（未记录在飞峰值）时不得默认声称并发施加到位。"""
    r = _result(10, 10, 10, within=10, max_inflight=0)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    assert "在飞峰值未记录" in txt
    assert "真实施加 10 并发" in txt            # 数字前留空格（终审修正B）
    assert _row(txt, 10, n_cells=17)[1] == "-"


def test_write_report_ladder_comes_from_results(tmp_path):
    """并发梯度必须由实际档位生成，不能写死 10/50/100。"""
    rs = [_result(5, 5, 5, within=5), _result(7, 7, 7, within=7)]
    txt = write_report(rs, tmp_path / "bench.md").read_text(encoding="utf-8")
    assert "- 并发梯度：5 / 7" in txt
    assert "10 / 50 / 100" not in txt


# 与 tests/test_docs.py 的 `_NEGATED_THROTTLE` 同口径：代码里没有限流，
# 报告凡提及「限流」必须带否定语。
_NEGATED_THROTTLE = ("未接线", "尚未接线", "没有实现", "没有请求限流", "无请求限流",
                     "不要声称")


def test_write_report_never_claims_unimplemented_throttling(tmp_path):
    """`llm_max_concurrency` 只是预留配置项，报告不得声称 Semaphore 限流。"""
    rs = [_result(10, 10, 10, within=10), _result(50, 10, 10, within=10)]
    txt = write_report(rs, tmp_path / "bench.md").read_text(encoding="utf-8")
    assert "Semaphore" not in txt and "semaphore" not in txt
    assert "asyncio" not in txt
    throttle_lines = [ln for ln in txt.splitlines() if "限流" in ln]
    assert throttle_lines, "报告应显式说明「没有请求限流」，而不是沉默"
    for line in throttle_lines:
        assert any(neg in line for neg in _NEGATED_THROTTLE), line
    llm_row = next(ln for ln in txt.splitlines() if ln.startswith("| LLM 并发控制"))
    assert "未接线" in llm_row and "线程池" in llm_row


def test_write_report_never_blames_upstream_throttling_for_low_inflight(tmp_path):
    """终审修正B：施加不足的归因必须与能力表一致。

    报告自己写着「没有请求限流」，同一份报告的告警行却写「线程启动错峰或上游
    限流」——自相矛盾。夹具取 ``max_inflight < concurrency``（旧限流守卫测试的
    夹具都是 ``max_inflight == concurrency``，根本走不到这条分支，才漏掉了它）。
    """
    r = _result(100, 100, 100, within=100, max_inflight=61)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    note = next(ln for ln in txt.splitlines() if "在飞峰值仅 61" in ln)
    assert "上游限流" not in note
    assert any(neg in note for neg in _NEGATED_THROTTLE), note
    assert "只代表 61 并发下" in note            # 数字前必须有空格
    assert "只代表61" not in txt
    assert _row(txt, 100, n_cells=17)[1] == "61"


def test_write_report_underdelivery_from_short_request_count(tmp_path):
    """另一半成因（请求数少于并发数）同样不得提限流：本服务没有限流实现。"""
    r = _result(100, 60, 60, within=60, max_inflight=40)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    note = next(ln for ln in txt.splitlines() if "在飞峰值仅 40" in ln)
    assert "请求数少于并发数" in note
    assert "限流" not in note


def test_write_report_renders_unmeasured_rss_as_text(tmp_path):
    """RSS 未测得（mem_snapshot 返回 0.0）时必须写「未测得」，不得打印占位数字。"""
    r = _result(10, 4, 4, within=4, rss_mb=0.0)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    row = _row(txt, 10, n_cells=17)
    assert row[15] == "未测得"                   # RSS 列
    assert "0.0" not in row[15]
    assert "不是真实占用为 0" in txt
    assert "未测得" in txt


def test_write_report_keeps_measured_rss_as_number(tmp_path):
    """反向守卫：真测到 RSS 时照常给数字（别把所有档都写成「未测得」）。"""
    r = _result(10, 4, 4, within=4, rss_mb=512.3)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    assert _row(txt, 10, n_cells=17)[15] == "512.3"


def test_write_report_describes_clip_and_cache_truthfully(tmp_path):
    """CLIP 常驻（查询期也用文本塔）、查询侧无缓存：不得写成入库后释放/双重缓存。"""
    r = _result(10, 10, 10, within=10)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    for wrong in ("推理完释放", "仅在入库阶段", "双重缓存", "不常驻"):
        assert wrong not in txt, f"报告出现了代码不支持的表述：{wrong}"
    clip_row = next(ln for ln in txt.splitlines() if ln.startswith("| 显存管理"))
    assert "常驻" in clip_row and "查询" in clip_row
    cache_row = next(ln for ln in txt.splitlines() if ln.startswith("| 结果缓存"))
    assert "无结果缓存" in cache_row


def test_write_report_states_ok_definition_and_counts(tmp_path):
    """成功/拒答/服务端样本数三项口径都要落到表里与正文。"""
    r = _result(10, 10, 10, within=10, refused=3, server_samples=2)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    res = _row(txt, 10, n_cells=17)
    assert res[4] == "3" and res[14] == "2"         # 拒答列 / 服务端样本数列
    assert "成功 = HTTP 200 且响应体可解析" in txt
    assert "拒答仍计入成功与达标" in txt


def test_write_report_marks_unrecorded_budget_honestly(tmp_path):
    """手工构造（无逐请求样本）时不得编造达标数。"""
    r = _result(10, 100, 100, within=None)
    txt = write_report([r], tmp_path / "bench.md").read_text(encoding="utf-8")
    assert "未记录" in txt


def test_write_report_renders_gpu_dash_and_notes(tmp_path):
    r = _result(10, 4, 4, within=4)
    txt = write_report([r], tmp_path / "bench.md",
                       notes=["压测期间服务未重启，模型常驻"]).read_text(encoding="utf-8")
    assert "| - |" in txt                                   # 无显存数据用 - 而非 0
    assert "压测期间服务未重启" in txt
    txt2 = write_report([r], tmp_path / "b2.md").read_text(encoding="utf-8")
    assert "- 无" in txt2                                   # 无备注时显式写「无」


# ---------------------------------------------------------------- 实跑入口

def test_run_level_rejects_empty_questions(tmp_path):
    """空问题集在发第一个请求前就拒绝（无 socket，故可单测）。"""
    from loadtest import run_level
    with pytest.raises(ValueError):
        run_level("http://127.0.0.1:1", 1, 1, [])
