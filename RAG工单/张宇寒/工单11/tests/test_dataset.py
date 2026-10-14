import unittest

from dataset import build_triplets, parse_questions


QUESTIONS = """# questions

## 1
- Question: 甲公司营业收入是多少？
- Reference: 甲公司营业收入100亿元。
- Stocks: 000001
- Pages: 000001:5
- Facts: 100

## 2
- Question: 乙公司净利润是多少？
- Reference: 乙公司净利润20亿元。
- Stocks: 000002
- Pages: 000002:8
- Facts: 20
"""

CHUNKS = [
    {"chunk_id": "a", "stock_code": "000001", "company": "甲公司", "page": 5, "text": "甲公司营业收入100亿元。"},
    {"chunk_id": "b", "stock_code": "000001", "company": "甲公司", "page": 7, "text": "甲公司员工人数500人。"},
    {"chunk_id": "c", "stock_code": "000002", "company": "乙公司", "page": 8, "text": "乙公司净利润20亿元。"},
]


class DatasetTests(unittest.TestCase):
    def test_parser_keeps_pages_and_facts(self):
        records = parse_questions(QUESTIONS)
        self.assertEqual(records[0]["pages"], {"000001": [5]})
        self.assertEqual(records[1]["facts"], [["20"]])

    def test_triplets_use_target_page_and_distinct_negative(self):
        result = build_triplets(CHUNKS, parse_questions(QUESTIONS), seed=42, train_ratio=0.5)
        all_rows = result["train"] + result["evaluation"]
        self.assertEqual(len(all_rows), 2)
        first = next(row for row in all_rows if row["question_id"] == 1)
        self.assertEqual(first["positive_id"], "a")
        self.assertNotEqual(first["positive_id"], first["negative_id"])
        self.assertIn("100亿元", first["positive"])
        self.assertEqual(first["answer"], "甲公司营业收入100亿元。")

    def test_split_is_repeatable_and_empty_corpus_is_rejected(self):
        one = build_triplets(CHUNKS, parse_questions(QUESTIONS), seed=42, train_ratio=0.5)
        two = build_triplets(CHUNKS, parse_questions(QUESTIONS), seed=42, train_ratio=0.5)
        self.assertEqual(one, two)
        with self.assertRaisesRegex(ValueError, "语料为空"):
            build_triplets([], parse_questions(QUESTIONS))


if __name__ == "__main__":
    unittest.main()
