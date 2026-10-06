"""测试 RRF（Reciprocal Rank Fusion）融合与去重。

任务书 3-A 要求：
- 关键词与向量两路结果按 RRF 融合
- 按 chunk_key 去重
- 两路都命中的条目排序靠前
"""
import pytest
from app.retrieval.fusion import fuse_results, RankedItem


def test_rrf_single_list():
    """测试单路结果：直接返回，不改变顺序"""
    items = [
        RankedItem(chunk_key="chunk1", score=0.9, source="vector"),
        RankedItem(chunk_key="chunk2", score=0.8, source="vector"),
    ]

    result = fuse_results([items])

    assert len(result) == 2
    assert result[0].chunk_key == "chunk1"
    assert result[1].chunk_key == "chunk2"


def test_rrf_deduplication():
    """测试去重：相同 chunk_key 只出现一次"""
    list1 = [
        RankedItem(chunk_key="chunk1", score=0.9, source="vector"),
        RankedItem(chunk_key="chunk2", score=0.8, source="vector"),
    ]
    list2 = [
        RankedItem(chunk_key="chunk2", score=0.95, source="keyword"),  # 重复
        RankedItem(chunk_key="chunk3", score=0.85, source="keyword"),
    ]

    result = fuse_results([list1, list2])

    # 去重后只有 3 个
    assert len(result) == 3
    chunk_keys = {item.chunk_key for item in result}
    assert chunk_keys == {"chunk1", "chunk2", "chunk3"}


def test_rrf_both_hit_ranks_higher():
    """测试两路都命中的排序靠前"""
    list1 = [
        RankedItem(chunk_key="chunk1", score=0.9, source="vector"),  # 只在向量
        RankedItem(chunk_key="chunk2", score=0.8, source="vector"),  # 两路都有
    ]
    list2 = [
        RankedItem(chunk_key="chunk2", score=0.95, source="keyword"),  # 两路都有
        RankedItem(chunk_key="chunk3", score=0.85, source="keyword"),  # 只在关键词
    ]

    result = fuse_results([list1, list2])

    # chunk2 两路都命中，应该排第一
    assert result[0].chunk_key == "chunk2"


def test_rrf_formula():
    """测试 RRF 公式：1/(k + rank)，k=60（标准值）"""
    # 构造明确的排名顺序
    list1 = [
        RankedItem(chunk_key="A", score=0.9, source="vector"),  # rank=1
        RankedItem(chunk_key="B", score=0.5, source="vector"),  # rank=2
    ]
    list2 = [
        RankedItem(chunk_key="B", score=0.8, source="keyword"),  # rank=1
        RankedItem(chunk_key="C", score=0.7, source="keyword"),  # rank=2
    ]

    # RRF 分数：
    # A: 1/(60+1) = 0.0164
    # B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
    # C: 1/(60+2) = 0.0161
    # 排序：B > A > C

    result = fuse_results([list1, list2])

    assert result[0].chunk_key == "B"
    assert result[1].chunk_key == "A"
    assert result[2].chunk_key == "C"


def test_rrf_empty_lists():
    """测试空列表：返回空结果"""
    result = fuse_results([])
    assert len(result) == 0

    result = fuse_results([[], []])
    assert len(result) == 0


def test_rrf_preserves_metadata():
    """测试融合后保留元数据"""
    items = [
        RankedItem(
            chunk_key="chunk1",
            score=0.9,
            source="vector",
            content="第一条内容",
            document_title="劳动合同法",
        ),
    ]

    result = fuse_results([items])

    assert result[0].content == "第一条内容"
    assert result[0].document_title == "劳动合同法"
