import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from parent_child_chunking import build_parent_child_chunks


class ParentChildChunkingTests(unittest.TestCase):
    def test_builds_parents_from_three_heading_levels_and_preserves_paths(self):
        markdown = """# 第一章
开头
## 1.1 二级
二级正文
### 1.1.1 三级
三级正文
#### 不参与切父块
四级正文
## 1.2 另一个二级
另一个正文
"""

        parents, children = build_parent_child_chunks(
            markdown,
            source_file="sample.md",
            min_chars=20,
            max_chars=40,
            overlap=5,
        )

        self.assertEqual(
            [parent["title"] for parent in parents],
            [
                "第一章",
                "第一章 > 1.1 二级",
                "第一章 > 1.1 二级 > 1.1.1 三级",
                "第一章 > 1.2 另一个二级",
            ],
        )
        self.assertEqual(parents[2]["content"], "三级正文\n#### 不参与切父块\n四级正文")
        self.assertTrue(all(child["parent_id"] in {parent["parent_id"] for parent in parents} for child in children))

    def test_keeps_html_table_as_single_table_child_without_overlap(self):
        markdown = """# 章节
表前正文
<table><tr><td>分类</td><td>定义</td></tr></table>
表后正文
"""

        parents, children = build_parent_child_chunks(
            markdown,
            source_file="sample.md",
            min_chars=20,
            max_chars=40,
            overlap=5,
        )

        self.assertEqual(len(parents), 1)
        table_children = [child for child in children if child["chunk_type"] == "table"]
        self.assertEqual(len(table_children), 1)
        self.assertEqual(table_children[0]["text"], "<table><tr><td>分类</td><td>定义</td></tr></table>")

    def test_keeps_contiguous_numbered_list_as_list_child(self):
        markdown = """# 诊疗关键点
（1）血压测量“三要点”：设备精准，安静放松，位置规范。

（2）诊断要点：诊室血压为主，140/90 mmHg为界。

（3）治疗“三原则”：达标、平稳、综合管理。
"""

        parents, children = build_parent_child_chunks(
            markdown,
            source_file="sample.md",
            min_chars=20,
            max_chars=80,
            overlap=5,
        )

        list_children = [child for child in children if child["chunk_type"] == "list"]
        self.assertEqual(len(list_children), 1)
        self.assertIn("血压测量“三要点”", list_children[0]["text"])
        self.assertIn("治疗“三原则”", list_children[0]["text"])

    def test_text_chunks_use_overlap_and_required_metadata(self):
        markdown = "# 章节\n" + "".join(
            [
                "一甲。",
                "二乙。",
                "三丙。",
                "四丁。",
                "五戊。",
                "六己。",
                "七庚。",
                "八辛。",
                "九壬。",
            ]
        )

        parents, children = build_parent_child_chunks(
            markdown,
            source_file="sample.md",
            min_chars=4,
            max_chars=18,
            overlap=5,
        )

        text_children = [child for child in children if child["chunk_type"] == "text"]
        self.assertGreater(len(text_children), 1)
        self.assertTrue(text_children[1]["text"].startswith("四丁。五戊。六己。"))
        for child in children:
            self.assertEqual(
                set(child),
                {
                    "child_id",
                    "parent_id",
                    "section_path",
                    "text",
                    "chunk_type",
                    "source_file",
                },
            )
            self.assertEqual(child["source_file"], "sample.md")
            self.assertEqual(child["section_path"], parents[0]["title"])

    def test_filters_non_knowledge_sections_before_chunking(self):
        markdown = """# 国家基层高血压防治管理指南2025版
中文正文
# National Clinical Practice Guidelines on the Management of Hypertension in Primary Health Care in China (2025)
## Abstract
English abstract
Key words: hypertension
## 1基层高血压管理基本要求
知识正文
## 国家基层高血压管理专家委员会第三届委员名单
委员名单
利益冲突：所有作者均声明不存在利益冲突
## 参考文献
[1] 引文
"""

        parents, children = build_parent_child_chunks(markdown, source_file="sample.md")

        self.assertEqual([parent["title"] for parent in parents], ["国家基层高血压防治管理指南2025版", "1基层高血压管理基本要求"])
        self.assertTrue(all("English abstract" not in child["text"] for child in children))
        self.assertTrue(all("委员名单" not in child["text"] for child in children))
        self.assertTrue(all("[1] 引文" not in child["text"] for child in children))

    def test_drops_empty_segments_but_keeps_short_text_children(self):
        markdown = """# 章节
（续表4）

这是一个短正文。
"""

        parents, children = build_parent_child_chunks(
            markdown,
            source_file="sample.md",
            min_chars=30,
            max_chars=80,
            overlap=5,
        )

        self.assertEqual(len(parents), 1)
        self.assertEqual(len(children), 1)
        self.assertIn("这是一个短正文。", children[0]["text"])

    def test_marks_large_tables_as_needing_split(self):
        table = "<table>" + ("<tr><td>很长的表格内容</td></tr>" * 120) + "</table>"
        markdown = "# 用药章节\n" + table

        parents, children = build_parent_child_chunks(markdown, source_file="sample.md")

        self.assertEqual(len(parents), 1)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["chunk_type"], "table")
        self.assertTrue(children[0]["needs_split"])

    def test_maintains_heading_stack_for_top_level_numbered_sections(self):
        markdown = """## 1基层高血压管理基本要求
## 1.1组建管理团队
团队正文
## 5 高血压治疗
## 5.4 降压药物治疗
### 5.4.2 降压药物选择
药物正文
"""

        parents, children = build_parent_child_chunks(markdown, source_file="sample.md", min_chars=4)

        self.assertEqual(
            [parent["title"] for parent in parents],
            [
                "1基层高血压管理基本要求",
                "1基层高血压管理基本要求 > 1.1组建管理团队",
                "5 高血压治疗",
                "5 高血压治疗 > 5.4 降压药物治疗",
                "5 高血压治疗 > 5.4 降压药物治疗 > 5.4.2 降压药物选择",
            ],
        )
        self.assertEqual(children[-1]["section_path"], "5 高血压治疗 > 5.4 降压药物治疗 > 5.4.2 降压药物选择")

    def test_filters_committee_member_titles(self):
        markdown = """## 10.2 教育内容
知识正文
## 常务委员
常务委员名单正文
## 委员
委员名单正文
## 秘书长
秘书长正文
"""

        parents, children = build_parent_child_chunks(markdown, source_file="sample.md")

        self.assertEqual([parent["title"] for parent in parents], ["10.2 教育内容"])
        self.assertTrue(all("委员名单正文" not in child["text"] for child in children))
        self.assertTrue(all("秘书长正文" not in child["text"] for child in children))

    def test_marks_empty_parents_without_children(self):
        markdown = """## 5 高血压治疗
## 5.1 治疗原则
治疗正文
"""

        parents, children = build_parent_child_chunks(markdown, source_file="sample.md", min_chars=4)

        self.assertTrue(parents[0]["is_empty"])
        self.assertFalse(parents[1].get("is_empty", False))
        self.assertEqual([child["section_path"] for child in children], ["5 高血压治疗 > 5.1 治疗原则"])

    def test_skips_figure_and_table_caption_headings_in_section_paths(self):
        markdown = """## 4高血压诊断与评估
## 4.1血压测量
测量总述
### 图1基层高血压诊疗管理流程图
### 4.1.1.2 测量方法
测量方法正文
## 5 高血压治疗
## 5.4降压药物治疗
治疗总述
### 表3生活方式干预目标及降压效果
<table><tr><td>表格内容</td></tr></table>
### 5.4.4用药注意事项
用药正文
"""

        parents, children = build_parent_child_chunks(markdown, source_file="sample.md", min_chars=4)

        self.assertEqual(
            [parent["title"] for parent in parents],
            [
                "4高血压诊断与评估",
                "4高血压诊断与评估 > 4.1血压测量",
                "4高血压诊断与评估 > 4.1血压测量 > 4.1.1.2 测量方法",
                "5 高血压治疗",
                "5 高血压治疗 > 5.4降压药物治疗",
                "5 高血压治疗 > 5.4降压药物治疗 > 5.4.4用药注意事项",
            ],
        )
        self.assertTrue(all("图1" not in child["section_path"] for child in children))
        self.assertTrue(all("表3" not in child["section_path"] for child in children))

    def test_text_chunks_split_on_sentence_boundaries_with_sentence_overlap(self):
        sentences = [
            "一甲。",
            "二乙。",
            "三丙。",
            "四丁。",
            "五戊。",
            "六己。",
            "七庚。",
            "八辛。",
            "九壬。",
        ]
        markdown = "# 章节\n" + "".join(sentences)

        parents, children = build_parent_child_chunks(
            markdown,
            source_file="sample.md",
            min_chars=4,
            max_chars=18,
            overlap=20,
        )

        text_chunks = [child["text"] for child in children if child["chunk_type"] == "text"]
        self.assertGreater(len(text_chunks), 1)
        for chunk in text_chunks:
            self.assertNotRegex(chunk, r"^[。！？；，、,.!?;]")
            self.assertRegex(chunk, r"[。！？；]$")
        self.assertTrue(text_chunks[0].endswith(sentences[5]))
        self.assertTrue(text_chunks[1].startswith(sentences[3]))
        self.assertIn(sentences[5], text_chunks[1])

    def test_split_list_segments_does_not_absorb_following_numbered_headings(self):
        markdown = """# 章节
（1）口服短效降压药物，门诊用药后观察。
（2）经上述处理，血压仍高，建议转诊。
5.4.6.2血压≥180/110 mmHg，伴心、脑、肾急性并发症的临床症状
急性症状正文用于确认新标题下会生成独立子块。
"""

        parents, children = build_parent_child_chunks(
            markdown,
            source_file="sample.md",
            min_chars=10,
            max_chars=120,
            overlap=20,
        )

        self.assertEqual(children[0]["chunk_type"], "list")
        self.assertNotIn("5.4.6.2血压", children[0]["text"])
        self.assertTrue(any(parent["title"].endswith("5.4.6.2血压≥180/110 mmHg，伴心、脑、肾急性并发症的临床症状") for parent in parents))
        self.assertTrue(any(child["section_path"].endswith("5.4.6.2血压≥180/110 mmHg，伴心、脑、肾急性并发症的临床症状") for child in children))

    def test_outputs_are_json_serializable(self):
        parents, children = build_parent_child_chunks("# 章节\n正文", source_file="sample.md")

        json.dumps(parents, ensure_ascii=False)
        json.dumps(children, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
