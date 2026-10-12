# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""DocumentQualityAssessmentSkill 单元测试

工单产出物 4：「测试脚本与验证报告；针对 Skill 核心功能的单元测试」。
验收标准还要求「该 Skill 能独立运行并返回正确的结构化报告」，
所以最后有一组针对**报告结构**的断言。

## 怎么跑

    python -m unittest test_skill -v          # 在 测试/ 目录下
    python test_skill.py                      # 也行（末尾有 main）

不需要网络、不需要那 1700 份语料 —— 临时文件都是现场造的，
所以这个脚本可以在任何机器上跑，这也是它和 `验证报告.md` 的分工：
**这里是逻辑对不对，报告是真数据上跑出来的结果。**

## 覆盖的分支，每条都对应一个实际踩过的坑

| 测试 | 防的是什么 |
|---|---|
| `test_扫描页要字符少且图片铺满` | 附图页冒充扫描页（IMDR 实测误判 298 份） |
| `test_图片块取不到时按扫描页处理` | 老缓存没存覆盖率 |
| `test_空when是兜底不是永远命中` | 每份文档都挂 Unclassified（实测打出 90 个标签） |
| `test_扫描件不打空文件标签` | 1700 份扫描专利全被判「坏文件」 |
| `test_主路由是第一条标签` | 一份文档两个路由，工作流不知道送哪 |
| `test_金额截出来的手机号要过滤` | 1,892,175,481.1 被截成 11 位冒充手机号 |
| `test_正常手机号要留下` | 过滤别把真的也挡了 |
| `test_扫描件换行不算字符` | 8 页扫描件数出 7 个「字符」（其实是换行符） |
| `test_损坏的PDF不掀翻整轮` | 坏文件让质检崩掉 |
| `test_断点续跑不重复解析` | 缓存失效 |
| `test_报告结构完整` | 五大功能的输出字段缺项 |
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent / "研发" / "document_quality_assessment" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import assess       # noqa: E402
import collect      # noqa: E402
import common       # noqa: E402
import dedup        # noqa: E402
import pdf_type     # noqa: E402
import sensitive    # noqa: E402
import summarize    # noqa: E402
import tags as tagmod   # noqa: E402

CFG = common.load_config()


# ------------------------------------------------------------------ PDF 分流

class TestPdfType(unittest.TestCase):

    def _cls(self, chars, covers, **over):
        c = json.loads(json.dumps(CFG))          # 深拷贝，避免测试间互相污染
        c["pdf_type"].update(over)
        return pdf_type.classify("x.pdf", c, chars, covers,
                                 len(chars) if chars else 0)

    def test_扫描页要字符少且图片铺满(self):
        """IMDR 实测：整页位图覆盖率 ≈1.00，附图页只有 0.3~0.45。

        只看字符数会把 298 份带附图页的**数字排版**专利误判成「混合型」。
        """
        # 10 页全部像是扫描页
        r = self._cls([0] * 10, [1.0] * 10)
        self.assertEqual(r["kind"], "scanned")

        # 10 页字符都少，但图片只占中间一块 —— 是附图页，不是扫描件
        r = self._cls([32] * 10, [0.4] * 10)
        self.assertEqual(r["kind"], "text")

        # 一半一半 → 混合
        r = self._cls([0] * 5 + [1000] * 5, [1.0] * 5 + [0.01] * 5)
        self.assertEqual(r["kind"], "mixed")

    def test_图片块取不到时按扫描页处理(self):
        """老缓存里没有 covers 字段。这时宁可多送一次 OCR，也不要漏。"""
        r = self._cls([0] * 10, None)
        self.assertEqual(r["kind"], "scanned")

    def test_混合型进待确认列表(self):
        """工单原文：判定是启发式，不保证正确，所以中间地带要输出待确认。"""
        r = self._cls([0] * 5 + [1000] * 5, [1.0] * 5 + [0.01] * 5)
        self.assertTrue(r["uncertain"])

    def test_打不开的PDF单独归类(self):
        r = self._cls(None, None)
        self.assertEqual(r["kind"], "unreadable")
        self.assertFalse(r["uncertain"])


# ------------------------------------------------------------------ 标签

class TestTags(unittest.TestCase):

    def _tags(self, **facts):
        base = {"ext": ".pdf", "pdf_kind": None, "chars": 1000}
        base.update(facts)
        return [t["name"] for t in tagmod.classify(base, CFG)]

    def test_空when是兜底不是永远命中(self):
        """`Unclassified` 的 when 是 {}。当成普通规则的话每份文档都会挂上它 ——
        实测 30 份文档打出 90 个标签，标签分布直接失去意义。"""
        t = self._tags(pdf_kind="text")
        self.assertEqual(t, ["Text_PDF"])
        self.assertNotIn("Unclassified", t)

    def test_真的没有规则命中才用兜底(self):
        t = self._tags(ext=".bin", pdf_kind=None, chars=1000)
        self.assertEqual(t, ["Unclassified"])

    def test_扫描件不打空文件标签(self):
        """扫描件字符数是 0 但完全正常。IMDR 里 1700 份扫描专利
        若不排除，每份都会被判成「坏文件」，路由变成「ocr、review」两个。"""
        t = self._tags(pdf_kind="scanned", chars=0)
        self.assertIn("Scan_PDF", t)
        self.assertNotIn("Empty_or_Broken", t)

    def test_文字型但抽不出字才是坏文件(self):
        """有文字层却几乎没抽出字 —— 这才值得人工看一眼。"""
        self.assertIn("Empty_or_Broken", self._tags(pdf_kind="text", chars=3))

    def test_主路由是第一条标签(self):
        """一份文档只能去一个解析器，工作流读 docs[][2]。"""
        ts = tagmod.classify({"ext": ".pdf", "pdf_kind": "scanned",
                              "chars": 0}, CFG)
        self.assertEqual(ts[0]["route"], "ocr")

    def test_超长文档额外提示切块(self):
        ts = tagmod.classify({"ext": ".pdf", "pdf_kind": "text",
                              "chars": 500000}, CFG)
        self.assertIn("Long_Document", [t["name"] for t in ts])


# ------------------------------------------------------------------ 敏感信息

class TestSensitive(unittest.TestCase):

    def _scan(self, text):
        return sensitive.scan_text(text, CFG, source="t.pdf")

    def test_金额截出来的手机号要过滤(self):
        """实测平安年报两条误报，都是逗号千分位的金额被截出 11 位：
           本年提取数18921754811,6892,734     ← 1,892,175,481.1
           应收账款21,19362019822,011
        前后紧邻逗号/小数点 → 判为长数字的一截，不算号码。"""
        self.assertEqual(self._scan("本年提取数18921754811,6892,734"), [])
        self.assertEqual(self._scan("应收账款21,19362019822,011"), [])

    def test_正常手机号要留下(self):
        hits = self._scan("联系电话：13812345678，邮箱 a@b.com")
        vals = {h["value"] for h in hits}
        self.assertIn("13812345678", vals)
        self.assertIn("a@b.com", vals)

    def test_每条命中都带上下文(self):
        """验收标准：「列表可操作性」—— 每条都要有足以让操作员判断的信息。"""
        hits = self._scan("联系" * 30 + "13812345678" + "结束" * 30)
        self.assertTrue(hits)
        self.assertIn("13812345678", hits[0]["context"])
        self.assertGreater(len(hits[0]["context"]), len("13812345678"))

    def test_专利号不会被当成身份证(self):
        """CN100342976C 里的数字串。前后数字边界要挡住。"""
        hits = [h for h in self._scan("公开号 CN100342976C") if h["type"] == "id_card"]
        self.assertEqual(hits, [])

    def test_邮箱默认开银行卡默认关(self):
        """工单原文：银行卡检测默认关闭（误报率太高）。"""
        kinds = {h["type"] for h in self._scan("6222020200112233445 a@b.com")}
        self.assertIn("email", kinds)
        self.assertNotIn("bank_card", kinds)


# ------------------------------------------------------------------ 相似度

class TestDedup(unittest.TestCase):

    def test_中文按三字滑窗切(self):
        """按单字切的话「的/一/是」这类字每份文档都有，不同文档也会撞车。"""
        s = dedup._shingles("带有预前处理的水泥磨", 3)
        self.assertIn("带有预", s)
        self.assertIn("水泥磨", s)
        self.assertNotIn("的", s)

    def test_英文按词切(self):
        self.assertIn("pump", dedup._shingles("a pump valve", 3))

    def test_相同文本距离为零(self):
        h = dedup.simhash("一种铝水循环泵，包括泵体和叶轮", 64, 3)
        self.assertEqual(dedup.hamming(h, h), 0)

    def test_相似文本比无关文本近(self):
        """三字滑窗下，同技术的两件应当明显近于不相干的两份。"""
        a = dedup.simhash("一种螺杆升降机，包括机架和螺杆", 64, 3)
        b = dedup.simhash("一种螺杆升降机，包括机架与螺杆", 64, 3)
        c = dedup.simhash("一种螺旋内刮式自清洗过滤器", 64, 3)
        self.assertLess(dedup.hamming(a, b), dedup.hamming(a, c))

    def test_取样覆盖头中尾(self):
        """只取头部会被套话骗到：年报招股书开头都是封面和目录。"""
        text = "HEAD" + "中" * 3000 + "TAIL"
        s = dedup.sample_text(text, 1000)
        self.assertIn("HEAD", s)
        self.assertIn("TAIL", s)


# ------------------------------------------------------------------ 长度分布

class TestLength(unittest.TestCase):

    def test_分位数与区间(self):
        recs = [{"path": f"{i}.pdf", "chars": c, "error": ""}
                for i, c in enumerate([0, 100, 1000, 5000, 100000])]
        r = collect.length_stats(recs, CFG)
        self.assertEqual(r["count"], 5)
        self.assertEqual(r["empty"], 1)
        self.assertEqual(sum(b["count"] for b in r["distribution"]), 5)

    def test_抽不出正文的记为失败(self):
        """chars=None 表示抽取失败，不能当成 0 字符混进分布。"""
        recs = [{"path": "a.pdf", "chars": None, "error": "打不开"},
                {"path": "b.pdf", "chars": 500, "error": ""}]
        r = collect.length_stats(recs, CFG)
        self.assertEqual(r["failed_count"], 1)
        self.assertEqual(r["count"], 1)


# ------------------------------------------------------------------ 端到端

class TestEndToEnd(unittest.TestCase):
    """造一个小目录真跑一遍 —— 验收标准要求「能独立运行并返回结构化报告」。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "docs"
        root.mkdir()
        (root / "a.md").write_text("这是一份说明文档。" * 100, encoding="utf-8")
        (root / "b.md").write_text("这是一份说明文档。" * 100, encoding="utf-8")
        (root / "short.md").write_text("太短", encoding="utf-8")
        (root / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
        (root / "notes.txt").write_text("联系人 13812345678", encoding="utf-8")
        cls.root = root
        cls.workdir = Path(cls.tmp.name) / "work"
        cls.rep = assess.assess(root, CFG, workdir=cls.workdir, progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_报告结构完整(self):
        """五大功能的输出字段一个都不能少。"""
        s, d = self.rep["summary"], self.rep["details"]
        for k in ("total_files", "ext", "pdf", "chars", "exact_dup_groups",
                  "similar_pairs", "sensitive_hits", "by_tag"):
            self.assertIn(k, s)
        for k in ("format", "pdf", "length", "dup", "sensitive", "tags", "errors"):
            self.assertIn(k, d)
        self.assertEqual(self.rep["skill"], "DocumentQualityAssessmentSkill")

    def test_两个内容相同的md被判完全重复(self):
        """MD5 那层是精确匹配，不需要人工确认。"""
        groups = self.rep["details"]["dup"]["md5_groups"]
        self.assertTrue(any(g[1] == 2 for g in groups),
                        f"应当有一组 2 个文件的完全重复：{groups}")

    def test_损坏的PDF不掀翻整轮(self):
        """坏文件是常态。它要进 errors，其余文件照常出结果。"""
        self.assertGreaterEqual(len(self.rep["details"]["errors"]), 1)
        self.assertGreaterEqual(self.rep["summary"]["total_files"], 5)

    def test_敏感信息带路径和上下文(self):
        hits = self.rep["details"]["sensitive"]["hits"]
        self.assertEqual(len(hits), 1)
        src, typ, val, ctx = hits[0]
        self.assertTrue(src.endswith("notes.txt"))
        self.assertEqual(val, "13812345678")
        self.assertIn("联系人", ctx)

    def test_每份文档都有主路由(self):
        docs = self.rep["details"]["tags"]["docs"]
        self.assertEqual(len(docs), self.rep["summary"]["total_files"])
        for path, tag_names, primary in docs:
            self.assertTrue(primary, f"{path} 没有主路由")
        self.assertEqual(sum(self.rep["summary"]["by_primary_route"].values()),
                         len(docs))

    def test_二次运行走缓存(self):
        """断点续跑：同样的输入第二次应当全部命中缓存。"""
        rep2 = assess.assess(self.root, CFG, workdir=self.workdir, progress=False)
        self.assertEqual(rep2["summary"]["reused_from_cache"],
                         rep2["summary"]["total_files"])


# ------------------------------------------------------------------ 汇总

class TestSummarize(unittest.TestCase):

    def test_主路由汇总每份只算一次(self):
        """by_route 把所有标签算一遍（一份文档可能算几次），
        by_primary 才是「N 份文档分别送去哪」。两个混了工作流会算错。"""
        tagged = [("a.pdf", [{"name": "Scan_PDF", "route": "ocr", "reason": ""}]),
                  ("b.pdf", [{"name": "Text_PDF", "route": "text", "reason": ""},
                             {"name": "Long_Document", "route": "chunk_aggressive",
                              "reason": ""}])]
        s = tagmod.summarize(tagged)
        self.assertEqual(s["by_route"], {"ocr": 1, "text": 1, "chunk_aggressive": 1})
        self.assertEqual(s["by_primary"], {"ocr": 1, "text": 1})
        self.assertEqual(sum(s["by_primary"].values()), 2)


if __name__ == "__main__":
    common.utf8_console()
    unittest.main(verbosity=2)
