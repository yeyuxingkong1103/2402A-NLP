import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split_large_tables import clean_field_value, complete_group_fields, parse_html_table_rows, table_record_to_rows


class SplitLargeTablesTests(unittest.TestCase):
    def test_parse_html_table_rows_expands_rowspan_and_colspan(self):
        html = """<table>
<tr><td>分类</td><td>名称</td><td>每次剂量</td><td>服药频率</td><td>推荐常用起始用法</td><td>适应证</td><td>禁忌证</td><td>主要不良反应</td></tr>
<tr><td rowspan="2">A (ACEI)</td><td>卡托普利</td><td>12.5mg</td><td>2</td><td>12.5mg bid</td><td rowspan="2">心力衰竭</td><td colspan="2">妊娠相关禁忌和不良反应</td></tr>
<tr><td>依那普利</td><td>5mg</td><td>1</td><td>5mg qd</td><td>高钾血症</td><td>血管水肿</td></tr>
</table>"""

        rows = parse_html_table_rows(html)

        self.assertEqual(rows[1][0], "A (ACEI)")
        self.assertEqual(rows[2][0], "A (ACEI)")
        self.assertEqual(rows[2][1], "依那普利")
        self.assertEqual(rows[2][5], "心力衰竭")
        self.assertEqual(rows[1][6], "妊娠相关禁忌和不良反应")
        self.assertEqual(rows[1][7], "妊娠相关禁忌和不良反应")

    def test_complete_group_fields_backfills_later_rowspan_values(self):
        rows = [
            ["C(二氢吡啶类钙拮抗剂）", "氨氯地平", "2.5~10 mg", "1", "5mgqd", "", "", ""],
            ["C(二氢吡啶类钙拮抗剂）", "左旋氨氯地平", "2.5~5mg", "1", "2.5mg qd", "", "", ""],
            [
                "C(二氢吡啶类钙拮抗剂）",
                "硝苯地平控释片",
                "30~60mg",
                "1",
                "30mg qd",
                "老年单纯收缩期高血压；心绞痛",
                "相对禁忌：快速性心律失常",
                "头痛；面部潮红；踝部水肿",
            ],
        ]

        completed = complete_group_fields(rows)

        self.assertEqual(completed[0][5], "老年单纯收缩期高血压；心绞痛")
        self.assertEqual(completed[0][6], "相对禁忌：快速性心律失常")
        self.assertEqual(completed[0][7], "头痛；面部潮红；踝部水肿")

    def test_clean_field_value_merges_broken_words(self):
        value = "心力衰竭；心肌；梗死后；动脉粥样硬；化；血肌酐>3；mg/dI（265；umol/L）；头痛；面部潮红；踝部水肿"

        cleaned = clean_field_value(value)

        self.assertIn("心肌梗死后", cleaned)
        self.assertIn("动脉粥样硬化", cleaned)
        self.assertIn("血肌酐>3mg/dI（265umol/L）", cleaned)
        self.assertIn("头痛；面部潮红；踝部水肿", cleaned)
        self.assertIn("高钾血症；血管神经性水肿", clean_field_value("高；血管神经性水肿"))
        self.assertIn("需两种及以上药物治疗的高血压", clean_field_value("需两；种及以上药物治疗的高血压"))
        self.assertIn("心绞痛；动脉粥样硬化", clean_field_value("心绞痛；动脉粥样硬化"))

    def test_table_record_to_rows_preserves_generic_two_column_headers(self):
        record = {
            "child_id": "child_00074",
            "parent_id": "parent_0026",
            "section_path": "5 高血压治疗 > 要点 5D",
            "source_file": "guide.md",
            "text": """<table>
<tr><td>运动类型</td><td>运动推荐方式</td></tr>
<tr><td>有氧运动</td><td>散步、慢跑、游泳。</td></tr>
</table>""",
        }

        rows = table_record_to_rows(record)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chunk_type"], "table_row")
        self.assertIn("运动类型: 有氧运动", rows[0]["text"])
        self.assertIn("运动推荐方式: 散步、慢跑、游泳。", rows[0]["text"])
        self.assertNotIn("名称:", rows[0]["text"])

    def test_table_record_to_rows_preserves_generic_four_column_headers(self):
        record = {
            "child_id": "child_00093",
            "parent_id": "parent_0028",
            "section_path": "5 高血压治疗 > 要点 5E",
            "source_file": "guide.md",
            "text": """<table>
<tr><td>复方制剂每片含量</td><td>每日服药片数</td><td>每日服药次数</td><td>主要不良反应</td></tr>
<tr><td>缬沙坦 50 mg/氢氯噻嗪 12.5 mg</td><td>1~2</td><td>1</td><td>偶见血管性水肿</td></tr>
</table>""",
        }

        rows = table_record_to_rows(record)

        self.assertEqual(len(rows), 1)
        self.assertIn("复方制剂每片含量: 缬沙坦 50 mg/氢氯噻嗪 12.5 mg", rows[0]["text"])
        self.assertIn("每日服药片数: 1~2", rows[0]["text"])
        self.assertIn("主要不良反应: 偶见血管性水肿", rows[0]["text"])

        record = {
            "child_id": "child_00034",
            "parent_id": "parent_0028",
            "section_path": "5 高血压治疗 > 表4",
            "source_file": "guide.md",
            "text": """<table>
<tr><td>分类</td><td>名称</td><td>每次剂量</td><td>服药频率</td><td>推荐常用起始用法</td><td>适应证</td><td>禁忌证</td><td>主要不良反应</td></tr>
<tr><td>A (ACEI)</td><td>卡托普利</td><td>12.5mg</td><td>2</td><td>12.5mg bid</td><td>心力衰竭</td><td>妊娠</td><td>干咳</td></tr>
</table>""",
        }

        rows = table_record_to_rows(record)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["child_id"], "child_00034_row_001")
        self.assertEqual(rows[0]["chunk_type"], "table_row")
        self.assertEqual(rows[0]["row_index"], 1)
        self.assertIn("分类: A (ACEI)", rows[0]["text"])
        self.assertIn("名称: 卡托普利", rows[0]["text"])
        self.assertEqual(rows[0]["parent_id"], "parent_0028")


if __name__ == "__main__":
    unittest.main()
