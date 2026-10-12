# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本测试职责：钉住响应时间的验收口径 —— /api/metrics 顶层字段一律指「首字延迟」，
#             完整答案时长只以 total_* 字段单独给出、不参与 3s 达标率。
#
# 为什么值得一条测试：
#   原来 /api/metrics 记录的是完整答案时长，恰好就是差 140ms 不达标的那个数。
#   文档写着「首字口径」，指标接口报着「完整时长」——下次复审拿 /api/metrics 一看就能翻案。
#   口径写死在文档里没用，得写死在代码里，并且有用例拦着不让它漂回去。

import pytest                        # 测试框架

from app import feedback as FB       # 被测模块


@pytest.fixture(autouse=True)        # 每个用例前后都清空样本，避免互相污染
def _clean_samples():
    FB._LATENCY.clear()              # 清首字样本
    FB._TOTAL.clear()                # 清完整时长样本
    yield                            # 执行用例
    FB._LATENCY.clear()              # 收尾再清一次
    FB._TOTAL.clear()                # 同上


def test_under_3s_rate_uses_first_token_not_total():
    """核心断言：首字达标、完整答案超时的情况下，达标率必须按首字算成 100%。"""
    FB.record(first_token=1.0, total=9.0)                                             # 首字达标、完整答案严重超时
    m = FB.metrics()                                                                  # 取指标
    assert m["basis"] == "first_token"                                                # 口径显式声明
    assert m["under_3s_rate"] == 1.0, "达标率被完整答案时长带偏了"                        # 必须全达标
    assert m["p50"] == 1.0                                                            # 顶层 P50 = 首字
    assert m["total_p50"] == 9.0                                                      # 完整答案单独给出


def test_metrics_reports_both_but_never_mixes_them():
    """混合样本：验证两套统计互不串味，且总量一致。"""
    for first, total in [(0.5, 4.0), (1.5, 5.0), (2.5, 6.0)]:                         # 三组：全部完整超时
        FB.record(first_token=first, total=total)                                     # 记录
    m = FB.metrics()                                                                  # 取指标
    assert m["n"] == 3                                                                # 样本数
    assert m["under_3s_rate"] == 1.0                                                  # 首字全部 ≤3s
    assert m["total_max"] == 6.0                                                      # 完整答案最大 6s（如实报出）
    assert m["max"] == 2.5, "顶层 max 不该是完整答案的 6.0"                             # 顶层只反映首字


def test_record_defaults_total_to_first_token():
    """非流式接口只传一个值：total 应回落为同一个数，不应出现 0。"""
    FB.record(2.0)                                                                    # 只传首字
    m = FB.metrics()                                                                  # 取指标
    assert m["p50"] == 2.0 and m["total_p50"] == 2.0                                  # 两套统计同值


def test_metrics_endpoint_shape():
    """接口层：/api/metrics 返回体里必须带 basis 与 total_* 字段。"""
    from fastapi.testclient import TestClient                                         # 测试客户端
    from app import main                                                              # 应用

    FB.record(first_token=1.0, total=8.0)                                             # 造一个样本
    data = TestClient(main.app).get("/api/metrics").json()                            # 调接口
    assert data["basis"] == "first_token"                                             # 口径字段在位
    for key in ("p50", "p95", "under_3s_rate", "total_p50", "total_p95", "total_max"):  # 必需字段
        assert key in data, f"缺少字段 {key}"                                          # 逐个检查
