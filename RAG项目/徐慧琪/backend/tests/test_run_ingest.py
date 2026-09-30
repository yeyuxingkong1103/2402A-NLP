# 端到端：三册区间必须严格不重叠，上册的边界共用页要被丢弃并记录
from app.ingest.run_ingest import VOLUMES, assign_ranges, drop_boundary


def test_volume_ranges_are_contiguous_and_disjoint():
    # 1260 = 462 + 577 + 221，三册不得重叠也不得留缝
    ranges = [(lo, hi) for _, lo, hi in VOLUMES]
    assert ranges == [(1, 462), (463, 1039), (1040, 1260)]
    for (_, _, hi), (_, lo_next, _) in zip(VOLUMES, VOLUMES[1:]):
        assert hi + 1 == lo_next


def test_covers_exactly_1260_articles():
    # 区间宽度之和必须等于官方条数合计，改区间时这条会立刻报警
    assert sum(hi - lo + 1 for _, lo, hi in VOLUMES) == 1260


def test_assign_ranges_keeps_only_own_range():
    numbers = [1, 2, 462, 463, 464, 466, 500]
    assert assign_ranges(numbers, 1, 462) == [1, 2, 462]


def test_drop_boundary_reports_discarded():
    # 上册 PDF 多出的 463~466 必须被明确报告，不能静默丢
    kept, dropped = drop_boundary([1, 2, 462, 463, 464, 465, 466], 1, 462)
    assert kept == [1, 2, 462]
    assert dropped == [463, 464, 465, 466]


def test_drop_boundary_empty_when_nothing_to_drop():
    # 中册/下册落在自身区间内，不应有丢弃项
    kept, dropped = drop_boundary([463, 464, 1039], 463, 1039)
    assert kept == [463, 464, 1039]
    assert dropped == []
