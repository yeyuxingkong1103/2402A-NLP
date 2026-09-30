# 父子切分的验收点是"命中款能回填整条"（FR-2.2/FR-3.5），所以 parent_id 必须可反查
from app.ingest.chunk import build_chunks
from app.ingest.structure import Article

# 拿真实的第 1041 条当夹具（三款），比造数据更能暴露问题
ART_1041 = Article(
    number=1041, number_cn="一千零四十一",
    text="第一千零四十一条 婚姻家庭受国家保护。\n实行婚姻自由、一夫一妻、男女平等的婚姻制度。",
    path="第五编 婚姻家庭 > 第一章 一般规定",
    paragraphs=["第一千零四十一条 婚姻家庭受国家保护。",
                "实行婚姻自由、一夫一妻、男女平等的婚姻制度。"],
)


def test_every_article_produces_a_father_chunk():
    chunks = build_chunks([ART_1041])
    fathers = [c for c in chunks if c.chunk_type == "father"]
    assert len(fathers) == 1
    assert fathers[0].article_no == 1041
    assert fathers[0].parent_id is None


def test_each_paragraph_becomes_a_child_chunk():
    chunks = build_chunks([ART_1041])
    children = [c for c in chunks if c.chunk_type == "paragraph"]
    assert len(children) == 2, "两款必须切成两个子块"


def test_child_points_back_to_father():
    # 这是回填的根据：命中子块后按 parent_id 取回父块整条
    chunks = build_chunks([ART_1041])
    father = next(c for c in chunks if c.chunk_type == "father")
    for child in (c for c in chunks if c.chunk_type == "paragraph"):
        assert child.parent_id == father.chunk_id
        assert child.article_no == 1041


def test_chunk_ids_are_unique():
    chunks = build_chunks([ART_1041])
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_oversized_article_becomes_leaf_chunks():
    # 单条 > 800 字按句切，叶块也要能回指父块，否则回填断链
    long_article = Article(
        number=999, number_cn="九百九十九", text="第一千条 " + "甲。" * 500,
        path="测试编 > 测试章", paragraphs=["第一千条 " + "甲。" * 500],
    )
    chunks = build_chunks([long_article])
    leaves = [c for c in chunks if c.chunk_type == "leaf"]
    assert leaves, "超长条必须切叶块"
    father = next(c for c in chunks if c.chunk_type == "father")
    assert all(leaf.parent_id == father.chunk_id for leaf in leaves)


def test_item_paragraph_is_tagged_as_item():
    # 行首「（一）」式段落要标成 item，便于按项检索与展示
    art = Article(number=500, number_cn="五百", text="第五百条 …",
                  path="第三编 合同", paragraphs=["第五百条 …", "（一）第一项内容。", "（二）第二项内容。"])
    chunks = build_chunks([art])
    items = [c for c in chunks if c.chunk_type == "item"]
    assert len(items) == 2


def test_ids_are_stable_across_runs():
    # 重跑入库时 id 必须不变，否则增量 upsert 会重复写入
    first = {c.chunk_id for c in build_chunks([ART_1041])}
    second = {c.chunk_id for c in build_chunks([ART_1041])}
    assert first == second
