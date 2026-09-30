from backend.app.ingestion.structured_chunker import build_parent_child_chunks, flatten_chunks


def test_builds_parent_chunks_from_headings_and_articles():
    text = """# 婚姻法

第一章 总则

第一条 为了规范婚姻关系，制定本法。

第二条 婚姻自由。
"""

    parents = build_parent_child_chunks(text)

    assert [parent.structure_type for parent in parents] == ["heading", "section", "article", "article"]
    assert parents[2].title == "第一条 为了规范婚姻关系，制定本法。"
    assert parents[2].children[0].parent_id == parents[2].id


def test_long_paragraph_children_overlap_without_crossing_parent_boundary():
    first = "第一条 " + "甲" * 1500
    second = "第二条 乙的规定。"
    parents = build_parent_child_chunks(f"{first}\n\n{second}")

    assert len(parents) == 2
    assert len(parents[0].children) == 2
    assert parents[0].children[0].content[-160:] == parents[0].children[1].content[:160]
    assert all("乙的规定" not in child.content for child in parents[0].children)


def test_flatten_keeps_parent_and_child_metadata():
    parents = build_parent_child_chunks("标题\n\n正文")

    rows = flatten_chunks(parents)

    assert rows[0][4] == "parent"
    assert rows[1][4] == "child"
    assert rows[1][2] == rows[0][0]
