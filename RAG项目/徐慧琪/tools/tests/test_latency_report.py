# 判据 8 的证据渲染器：钉**整行**而不是子串（P50/P95 对调后两个子串都还在），
# 并专测"历史行会不会被无声丢掉"——那正是这套机制存在的意义。
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest

from eval_qa import _percentile as eval_percentile
from latency_report import (LATENCY_HISTORY_END, LATENCY_HISTORY_HEADER, LATENCY_HISTORY_START,
                            _percentile as latency_percentile, latency_history,
                            render_latency_report, retrieval_seconds)

OUTLIER = "| 2026-09-28 13:05 | 100 | 1.004s | 4.620s | 23 | 手工补录：离群跑，原因未定位 |"
# 这一行的 P95 刻意**超过 2s**：小样本若刚好很快，计数是否按题数过滤就看不出差别，
# 那样变异 R2（计数不分题数）会活下来——夹具必须能区分两种口径
PARTIAL = "| 2026-09-28 14:30 | 5 | 3.000s | 4.000s | 3 | 部分运行：只量了前 5 题 |"


def _meta(**kwargs):
    """渲染器的必填字段集中在这里，免得每个用例各抄一遍（少写一个键要当场红）。"""
    base = {"name": "qa_eval_v0.jsonl", "md5": "abc", "n": 100, "command": "x"}
    base.update(kwargs)
    return base


def test_retrieval_seconds_times_each_item_once():
    # 判据 8 的读数靠这里逐题计时：注入假函数，单测不拉真模型。
    # 断言"每题恰好一次、按原序"——漏题或重复计都会改条数/顺序
    seen = []
    seconds = retrieval_seconds([{"query": "甲"}, {"query": "乙"}],
                                lambda question: seen.append(question))
    assert seen == ["甲", "乙"]
    assert len(seconds) == 2


def test_render_latency_report_pins_rows_not_just_substrings():
    # 判据 8 的证据要能被复核者重跑比对，所以它必须写明量的是**检索段**、
    # 挂哪一版评估集、以及预热单列。这里钉**整行**：P50 与 P95 对调后
    # "50.000s" 与 "95.000s" 两个子串都还在，只有按行断言才拦得住
    text = render_latency_report(
        [float(i) for i in range(1, 101)],
        _meta(n=100, command="python tools/eval_qa.py --retrieval-only"))
    assert "检索段" in text, "必须写明量的是检索段，否则会被当成端到端延迟引用"
    assert "abc" in text, "必须记评估集 MD5，否则不知道量的是哪一版题"
    assert "qa_eval_v0.jsonl" in text, "评估集文件名要进产物"
    assert "python tools/eval_qa.py --retrieval-only" in text, "重跑命令要写进产物，否则无从复核"
    assert "| 全部 100 条 | 50.000s | 95.000s | 1.000s | 100.000s |" in text
    # 首条含模型/CUDA 预热，必须剔除后再算一档：混进去会把一次性冷启动说成稳态延迟
    assert "| 剔除首条预热后 99 条 | 51.000s | 95.000s | 2.000s | 100.000s |" in text


def test_render_latency_report_warns_on_truncated_run():
    # 小样本的记录顶着判据 8 的名义被引用是这里最危险的失败方向：
    # 有 warning 就必须显眼地出现在产物里，没有就不能凭空多出一行
    assert "⚠️ 只量了前 5 题" in render_latency_report(
        [1.0] * 5, _meta(n=5, warning="⚠️ 只量了前 5 题"))
    assert "⚠️" not in render_latency_report([1.0] * 5, _meta(n=5))


def test_render_latency_report_survives_empty_run():
    # 空跑不该毁掉整轮产物（同 eval_qa._rate 的空分母规矩）：必须给"无观测"的诚实
    # 读数，而不是让 min() 抛 ValueError——空跑既不是 0s，也不能被读成达标
    text = render_latency_report([], _meta(n=0))
    assert "无观测" in text
    assert "不要读成 0s" in text


def test_render_latency_report_keeps_outlier_across_reruns():
    # 离群跑必须扛得住下一次干净跑：只读证据文件的人要能看到"曾有一次不达标"，
    # 否则那份记录传递出去的就是"无条件达标"
    text = render_latency_report([0.5] * 100, _meta(n=100), [OUTLIER])
    assert OUTLIER in text, "旧离群行被洗掉了"
    assert "不达标（检索段 P95 ≥ 2s）1 次" in text
    # 读回的那一步也要对：写盘再读回必须得到同一行，否则下次重跑就丢历史
    written = (LATENCY_HISTORY_START + "\n" + LATENCY_HISTORY_HEADER + "\n"
               + OUTLIER + "\n" + LATENCY_HISTORY_END)
    assert latency_history(written) == [OUTLIER]


def test_latency_history_refuses_to_lose_history_silently():
    # 标记文本一旦与旧文件对不上，旧离群行就会"无声消失"——正是这个机制存在的
    # 意义所在，所以必须炸掉而不是静默返回空表；只有"白纸"（首次运行）才允许空表
    with pytest.raises(ValueError):
        latency_history("# ③a 检索段延迟记录（旧版没有标记）\n\n内容\n")
    assert latency_history("") == []


def test_latency_history_raises_on_row_with_wrong_column_count():
    # 读侧也必须炸：这个文件**按设计就是手工编辑的**（历史里 4/5 行是手工补录），
    # 手工行多一个竖线若只是被过滤掉，它就在下一次运行时无声消失（只测写入侧的
    # 那条用例挡不住这个——它从没把畸形行喂给读取器）
    bad = "| 2026-09-28 | 5 | 0.5s | 0.6s | 0 | 部分运行 | 多出来的一格 |"
    with pytest.raises(ValueError):
        latency_history(f"{LATENCY_HISTORY_START}\n{LATENCY_HISTORY_HEADER}\n{bad}\n"
                        f"{LATENCY_HISTORY_END}")


def test_latency_history_keeps_a_note_that_contains_dashes():
    # 分隔行按"每格只有 - 和 :"识别：用"含 ---"判会把备注里写了 --- 的数据行
    # 当成分隔行丢掉——同一种无声丢失
    row = "| 2026-09-28 | 5 | 0.500s | 0.600s | 0 | 备注里写 --- 也不算分隔行 |"
    text = (f"{LATENCY_HISTORY_START}\n{LATENCY_HISTORY_HEADER}\n|---|---|---|---|---|---|\n"
            f"{row}\n{LATENCY_HISTORY_END}")
    assert latency_history(text) == [row]


def test_render_sanitizes_pipe_in_note_so_history_stays_parseable():
    # 写入侧的另一半：备注里的竖线会把行撑成 7 列（读侧现在会因此报错），
    # 所以写入时就要换掉，而不是等下一次跑的时候炸在别人手里
    text = render_latency_report([0.5] * 10, _meta(n=10, note="人工核对|复核过"), [])
    rows = latency_history(text)
    assert len(rows) == 1 and "人工核对/复核过" in rows[0]


def test_tally_counts_only_full_runs_but_keeps_every_row():
    # 判据 8 的门槛是 100 题的读数：把 --limit 的小样本混进"不达标次数"，会让这个
    # 计数既不是全量口径也不是小样本口径；但行本身一条不少地留在表里（可见）
    text = render_latency_report([0.5] * 100, _meta(n=100), [OUTLIER, PARTIAL])
    assert "全量 100 题" in text
    assert "不达标（检索段 P95 ≥ 2s）1 次" in text
    rows = latency_history(text)
    assert len(rows) == 3, "部分运行也要留在表里，只是不计数"
    assert PARTIAL in rows


def test_history_markers_are_pinned():
    # 标记是"下次重跑还认得出旧记录"的唯一依据：字面量被改，所有旧记录就失联
    # （守卫会在运行时炸，但那时已经晚了）——所以把字面量钉死，改名必须先红这条
    assert LATENCY_HISTORY_START == "<!-- 历史运行记录：脚本跨次重跑保留 -->"
    assert LATENCY_HISTORY_END == "<!-- 历史记录结束 -->"
    assert LATENCY_HISTORY_HEADER.startswith("| 运行时间 | 题数 |")


@pytest.mark.parametrize("n", [1, 5, 10, 101])
@pytest.mark.parametrize("pct", [50, 95])
def test_latency_percentile_agrees_with_eval_qa(n, pct):
    # 拆模块时复制了一份 _percentile：口径只能有一处来源，两份漂了必须当场红。
    # 多取几个 n：最近秩与插值两种定义在某些样本量上会分叉，只比 n=100 会漏
    sample = [float(i) for i in range(1, n + 1)]
    assert latency_percentile(sample, pct) == eval_percentile(sample, pct)
    assert latency_percentile([], pct) == eval_percentile([], pct) == 0.0
