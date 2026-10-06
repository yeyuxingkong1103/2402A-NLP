from app.core.rrf import rrf_fuse


def test_rrf_fuse_merges_and_scores():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    out = rrf_fuse(a, b)
    # x 在 a 排1、b 排2；y 在 a 排2、b 排1；两者得分相同，但都应排 w/z 之前
    assert out[0] in ("x", "y")
    assert out[1] in ("x", "y")
    assert set(out) == {"x", "y", "z", "w"}


def test_rrf_fuse_handles_empty():
    assert rrf_fuse([], []) == []


def test_rrf_fuse_dedups():
    a = ["x", "y"]
    b = ["x", "y"]
    out = rrf_fuse(a, b)
    assert out.count("x") == 1
    assert len(out) == 2
