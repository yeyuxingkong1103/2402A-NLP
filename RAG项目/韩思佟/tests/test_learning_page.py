"""四文件代码块学习资料一致性测试；不启动服务，也不调用模型。"""
import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "core-line-guide.json"
PAGE = ROOT / "app" / "static" / "learn.html"
EXPECTED = {
    "offline": "app/offline_pipeline.py",
    "online": "app/single_app.py",
    "evaluate": "app/evaluate.py",
    "pressure": "app/pressure_test.py",
}


class LearningPageTests(unittest.TestCase):
    def test_four_lessons_match_current_source(self):
        dataset = json.loads(GUIDE.read_text(encoding="utf-8"))
        self.assertEqual(3, dataset["version"])
        self.assertEqual("online", dataset["default_lesson"])
        self.assertEqual("block", dataset["default_mode"])
        lessons = {lesson["id"]: lesson for lesson in dataset["lessons"]}
        self.assertEqual(set(EXPECTED), set(lessons))

        for lesson_id, relative_path in EXPECTED.items():
            with self.subTest(lesson=lesson_id):
                source = ROOT / relative_path
                lines = source.read_text(encoding="utf-8").splitlines()
                lesson = lessons[lesson_id]
                self.assertLessEqual(len(lines), 300)
                self.assertEqual(relative_path, lesson["source"])
                self.assertEqual(len(lines), lesson["line_count"])
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(), lesson["sha256"]
                )
                entries = {entry["line"]: entry for entry in lesson["lines"]}
                self.assertEqual(sum(bool(line.strip()) for line in lines), len(entries))
                for number, code in enumerate(lines, 1):
                    if code.strip():
                        self.assertEqual(code, entries[number]["code"])
                        self.assertTrue(entries[number]["explanation"].strip())
                # 分块学习是默认模式；每块对应一段连续源码，不能越界或互相重叠。
                blocks = lesson.get("blocks")
                self.assertIsInstance(blocks, list)
                self.assertGreaterEqual(len(blocks), 1)
                previous_end = 0
                required = ("purpose", "input", "steps", "functions", "output",
                            "failure", "speech")
                for block in blocks:
                    with self.subTest(block=block.get("id")):
                        start = block.get("start_line")
                        end = block.get("end_line")
                        self.assertIsInstance(start, int)
                        self.assertIsInstance(end, int)
                        self.assertTrue(1 <= start <= end <= len(lines))
                        self.assertGreater(start, previous_end)
                        previous_end = end
                        self.assertTrue(block.get("id"))
                        self.assertTrue(block.get("label"))
                        for field in required:
                            self.assertTrue(block.get(field), f"代码块缺少{field}")
                        self.assertIsInstance(block["steps"], list)
                        self.assertIsInstance(block["functions"], list)
                        self.assertTrue(all(item.strip() for item in block["steps"]))
                        self.assertTrue(all(item.strip() for item in block["functions"]))
                        self.assertGreaterEqual(len(block["steps"]), 1)
                        self.assertGreaterEqual(len(block["functions"]), 1)
                        self.assertLessEqual(len(block["steps"]), 3)
                        self.assertLessEqual(len(block["functions"]), 5)

    def test_html_embeds_the_same_dataset_and_switcher(self):
        html = PAGE.read_text(encoding="utf-8")
        match = re.search(
            r'<script id="lesson-data" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        embedded = json.loads(match.group(1))
        self.assertEqual(json.loads(GUIDE.read_text(encoding="utf-8")), embedded)
        self.assertIn('id="fileSelect"', html)
        self.assertIn("switchLesson(dataset.default_lesson)", html)
        self.assertIn('id="blockMode"', html)
        self.assertIn('id="lineMode"', html)
        self.assertIn("showBlock", html)
        self.assertIn("lesson.blocks", html)
        self.assertIn("逐行查询（备用）", html)
        # 代码块用 JavaScript 转义换行拼接；若这里变成真实换行，浏览器会语法报错。
        self.assertIn(".join('\\n')", html)


if __name__ == "__main__":
    unittest.main()
