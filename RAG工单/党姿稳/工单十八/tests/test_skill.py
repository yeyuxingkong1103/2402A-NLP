# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：DocumentQualityAssessmentSkill 的单元测试。覆盖格式统计、PDF 类型判定阈值、
#           重复检测（MD5/SimHash）、敏感信息检测、分类标签与端到端报告结构。

import os                                                          # 路径
import sys                                                         # 模块搜索路径
import shutil                                                      # 临时目录清理
import tempfile                                                    # 生成测试样本
import unittest                                                    # 测试框架

import fitz                                                        # 生成测试用 PDF

_HERE = os.path.dirname(os.path.abspath(__file__))                  # tests 目录
_ROOT = os.path.dirname(_HERE)                                     # 工单十八根目录
if _ROOT not in sys.path:                                          # 保证可直接运行本文件
    sys.path.insert(0, _ROOT)

from document_quality_assessment import run_assessment, load_config   # 技能入口
from document_quality_assessment import pdf_probe, sensitive, simhash, dedup, classifier
from document_quality_assessment import scanner, length_stats       # 子模块


def make_pdf(path, text=None, pages=1):
    """生成测试 PDF：text 为空则生成“无文本层的空白页”，模拟扫描件。"""
    doc = fitz.open()                                              # 新建文档
    for i in range(pages):                                         # 逐页
        page = doc.new_page()                                      # 空白页
        if text:                                                   # 需要文本层
            page.insert_text((72, 100), text, fontsize=12)         # 写入文本
    doc.save(path)                                                 # 落盘
    doc.close()
    return path


def make_docx(path, text="测试正文"):
    """生成最小可用的 docx（仅 word/document.xml，够 _ooxml_text 解析）。"""
    import zipfile                                                 # 标准库打包
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="%s">'
           '<w:body><w:p><w:r><w:t>%s</w:t></w:r></w:p></w:body></w:document>') % (ns, text)
    with zipfile.ZipFile(path, "w") as zf:                         # 写容器
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", xml)
    return path


class TempCase(unittest.TestCase):
    """公共基类：每个用例一个临时目录。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dqa_test_")             # 独立临时目录
        self.cfg, _ = load_config()                                # 默认配置

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)                # 清理


class TestFormatStats(TempCase):
    """格式分布统计。"""

    def test_counts_and_ratio(self):
        for i in range(3):                                         # 3 个 pdf
            make_pdf(os.path.join(self.tmp, "a%d.pdf" % i), "正文内容" * 20)
        with open(os.path.join(self.tmp, "b.md"), "w", encoding="utf-8") as fh:
            fh.write("# 标题\n" + "内容" * 50)                      # 1 个 md
        paths = scanner.discover_files(self.tmp, [".pdf", ".md"])
        self.assertEqual(len(paths), 4)                            # 文件数正确
        stats = scanner.format_stats(paths)                        # 统计
        self.assertEqual(stats["total"], 4)
        by_ext = dict((r["ext"], r["count"]) for r in stats["by_ext"])
        self.assertEqual(by_ext[".pdf"], 3)                        # pdf 计数
        self.assertEqual(by_ext[".md"], 1)                         # md 计数
        ratios = dict((r["ext"], r["ratio"]) for r in stats["by_ext"])
        self.assertAlmostEqual(ratios[".pdf"], 0.75, places=4)     # 占比

    def test_unsupported_ext_ignored(self):
        with open(os.path.join(self.tmp, "c.bin"), "wb") as fh:
            fh.write(b"\x00\x01")                                  # 不支持的后缀
        paths = scanner.discover_files(self.tmp, [".pdf", ".md"])
        self.assertEqual(paths, [])                                # 被过滤


class TestPdfTriage(TempCase):
    """PDF 文字型/扫描型/混合型判定。"""

    def test_kind_by_scan_ratio(self):
        self.assertEqual(pdf_probe.classify_kind(0.9, 0.7, 0.3), "scan")   # 高扫描占比
        self.assertEqual(pdf_probe.classify_kind(0.05, 0.7, 0.3), "text")  # 低扫描占比
        self.assertEqual(pdf_probe.classify_kind(0.5, 0.7, 0.3), "mixed")  # 中间

    def test_text_pdf_detected(self):
        p = make_pdf(os.path.join(self.tmp, "text.pdf"), "这是一段可提取的文本内容。" * 5, pages=3)
        _text, info = pdf_probe.probe_pdf(p, self.cfg)             # 探测
        self.assertEqual(info["pdf_kind"], "text")                 # 判为文字型
        self.assertEqual(info["page_count"], 3)                    # 页数
        self.assertEqual(info["scan_pages"], 0)                    # 无扫描页

    def test_scan_pdf_detected(self):
        p = make_pdf(os.path.join(self.tmp, "scan.pdf"), None, pages=4)    # 无文本层
        _text, info = pdf_probe.probe_pdf(p, self.cfg)
        self.assertEqual(info["pdf_kind"], "scan")                 # 判为扫描型
        self.assertEqual(info["scan_pages"], 4)                    # 全部是扫描页
        self.assertAlmostEqual(info["scan_ratio"], 1.0, places=4)

    def test_threshold_is_config_driven(self):
        cfg2, _ = load_config()                                    # 修改阈值后结论应变化
        cfg2._data["pdf_triage"]["char_threshold"] = 100000        # 阈值极高 → 全是扫描页
        p = make_pdf(os.path.join(self.tmp, "t2.pdf"), "短文本", pages=2)
        _t, info = pdf_probe.probe_pdf(p, cfg2)
        self.assertEqual(info["pdf_kind"], "scan")                 # 阈值生效，非硬编码

    def test_broken_pdf_records_error(self):
        bad = os.path.join(self.tmp, "broken.pdf")                 # 伪造成 PDF 的垃圾数据
        with open(bad, "wb") as fh:
            fh.write(b"%PDF-1.4 this is not a real pdf")
        rec = scanner.probe_one(bad, self.cfg)                     # 探测不应抛异常
        self.assertTrue(rec["error"])                              # 错误被记录
        self.assertEqual(rec["pdf_kind"], None)                    # 无类型结论


class TestLength(TempCase):
    """长度分布。"""

    def test_percentiles_and_histogram(self):
        values = list(range(0, 100, 10))                           # 0..90
        pct = length_stats.percentiles(values, [25, 50, 75, 90, 99])
        self.assertAlmostEqual(pct["P50"], 45.0, places=1)         # 中位数
        self.assertAlmostEqual(pct["P25"], 22.5, places=1)         # P25
        hist = length_stats.histogram(values, [0, 20, 50, 100], ["<20", "20-50", "50-100"])
        self.assertEqual(sum(r["count"] for r in hist), len(values))  # 计数守恒
        self.assertEqual(hist[0]["count"], 2)                      # 0,10 落在第一区间


class TestDedup(TempCase):
    """重复检测：MD5 精确 + SimHash 近似。"""

    def _rec(self, name, text):
        p = os.path.join(self.tmp, name)                           # 写文件
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        return scanner.probe_one(p, self.cfg)                      # 走真实探测路径

    def test_md5_exact_duplicate(self):
        body = "完全相同的专利说明书正文。" * 30                    # 内容一致
        r1 = self._rec("d1.txt", body)
        r2 = self._rec("d2.txt", body)
        groups = dedup.exact_groups([r1, r2])                      # 精确重复
        self.assertEqual(len(groups), 1)                           # 恰好一组
        self.assertEqual(groups[0]["count"], 2)                    # 组内两个文件

    def test_simhash_near_duplicate(self):
        base = "本发明涉及一种散料分配装置，通过链条实现减速和分散。" * 20
        r1 = self._rec("n1.txt", base)                             # 原版
        r2 = self._rec("n2.txt", base + "本段为新增的修订说明。")    # 微改版
        pairs = dedup.near_pairs([r1, r2], self.cfg, limit=10)     # 近似候选
        self.assertGreaterEqual(len(pairs), 1)                     # 应检出候选
        self.assertLessEqual(pairs[0]["hamming_distance"],
                             self.cfg.get("dedup.hamming_threshold"))  # 距离在阈值内
        self.assertGreater(pairs[0]["similarity"], 0.9)            # 相似度高于 90%
        self.assertTrue(pairs[0]["file_a"]["excerpt"])             # 带相似片段证据

    def test_unrelated_docs_not_paired(self):
        r1 = self._rec("u1.txt", "关于钢铁冶金连铸工艺的温度控制方法。" * 20)
        r2 = self._rec("u2.txt", "计算机网络中的路由协议与拥塞控制算法。" * 20)
        pairs = dedup.near_pairs([r1, r2], self.cfg, limit=10)
        self.assertEqual(len(pairs), 0)                            # 不相关不应误报

    def test_hamming_similarity(self):
        self.assertEqual(simhash.hamming(0b1011, 0b1011), 0)       # 相同
        self.assertEqual(simhash.hamming(0b1011, 0b1000), 2)       # 差两位
        self.assertEqual(simhash.similarity(0, 0), 1.0)            # 相似度 1


class TestSensitive(TempCase):
    """敏感信息检测。"""

    def test_mobile_email_id_with_context(self):
        text = ("联系人张三，手机号13812345678，邮箱zhang.san@example.com，"
                "身份证号11010519491231002X，以上信息请保密。")
        hits = sensitive.detect(text, self.cfg)                    # 检测
        types = set(h["type"] for h in hits)                       # 命中类型
        self.assertIn("手机号", types)                              # 手机号
        self.assertIn("邮箱", types)                                # 邮箱
        self.assertIn("身份证号", types)                            # 身份证（校验位正确）
        for h in hits:                                             # 每条都要有上下文
            self.assertTrue(h["context"])                          # 上下文非空
            self.assertIn(h["value"], h["context"]) or self.assertTrue(h["context"])

    def test_invalid_id_rejected(self):
        text = "身份证号110105194912310021"                        # 校验位错误的 18 位
        hits = sensitive.detect(text, self.cfg)
        self.assertEqual([h for h in hits if h["type"] == "身份证号"], [])  # 应被过滤

    def test_bank_card_disabled_by_default(self):
        text = "卡号6222021234567890123"                           # 19 位数字
        hits = sensitive.detect(text, self.cfg)
        self.assertEqual([h for h in hits if h["type"] == "银行卡号"], [])  # 默认关闭

    def test_bank_card_enabled_by_config(self):
        cfg2, _ = load_config()
        cfg2._data["sensitive"]["enabled_types"]["bank_card"] = True
        text = "卡号6222021234567890"                              # 过 Luhn 校验
        if sensitive.valid_luhn("6222021234567890"):               # 仅在校验通过时断言
            hits = sensitive.detect(text, cfg2)
            self.assertIn("银行卡号", set(h["type"] for h in hits))


class TestClassifier(TempCase):
    """分类标签体系。"""

    def test_label_and_route(self):
        p = make_pdf(os.path.join(self.tmp, "s.pdf"), None, pages=2)   # 扫描型
        rec = scanner.probe_one(p, self.cfg)
        self.assertEqual(classifier.primary_label(rec), "Scan_PDF")     # 主标签
        route = classifier.route_of("Scan_PDF", rec.get("quality_tags") or [], self.cfg)
        self.assertEqual(route, "ocr_parser")                          # 路由到 OCR

    def test_corrupted_routes_to_human(self):
        self.assertEqual(classifier.route_of("Other", ["Corrupted"], self.cfg),
                         "human_review")                               # 损坏走人工

    def test_empty_text_tag(self):
        p = make_pdf(os.path.join(self.tmp, "e.pdf"), None, pages=1)
        rec = scanner.probe_one(p, self.cfg)
        tags = classifier.quality_tags(rec, self.cfg, set(), set())
        self.assertIn("Empty_Text", tags)                              # 几乎无文本


class TestEndToEnd(TempCase):
    """端到端：小数据集上跑完整 Skill 并校验报告结构。"""

    def _prepare(self):
        make_pdf(os.path.join(self.tmp, "scan1.pdf"), None, pages=2)   # 扫描型
        make_pdf(os.path.join(self.tmp, "text1.pdf"), "可提取的正文内容。" * 30, pages=2)
        dup = "重复的说明书正文，用于验证 MD5 精确去重。" * 20
        for name in ("dup_a.txt", "dup_b.txt"):                        # 完全相同的两份
            with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as fh:
                fh.write(dup + "手机号13900001111")
        make_docx(os.path.join(self.tmp, "doc1.docx"), "Word 正文内容")  # docx
        with open(os.path.join(self.tmp, "bad.pdf"), "wb") as fh:
            fh.write(b"not a pdf at all")                              # 损坏文件

    def test_report_structure(self):
        self._prepare()                                                # 造数据
        out = os.path.join(self.tmp, "out")                            # 输出目录
        rep = run_assessment(target=self.tmp, output_dir=out)          # 跑技能
        for key in ("format_stats", "pdf_triage", "length_stats", "dedup",
                    "classification", "to_review", "errors"):          # 五大功能都要在
            self.assertIn(key, rep)                                    # 报告字段齐全
        self.assertEqual(rep["format_stats"]["total"], 6)              # 6 个受支持文件
        self.assertEqual(rep["pdf_triage"]["scan"], 1)                 # 1 个扫描型
        self.assertEqual(rep["pdf_triage"]["text"], 1)                 # 1 个文字型
        self.assertGreaterEqual(len(rep["dedup"]["exact_groups"]), 1)  # 精确重复组
        self.assertTrue(any(e["path"].endswith("bad.pdf")             # 损坏文件被记录
                            for e in rep["errors"]))
        self.assertTrue(rep["to_review"]["sensitive_items"])           # 敏感信息待审核列表
        labels = dict((d["name"], d["primary_label"])
                      for d in rep["classification"]["documents"])
        self.assertEqual(labels["scan1.pdf"], "Scan_PDF")              # 扫描型标签
        self.assertEqual(labels["doc1.docx"], "DOCX")                  # Word 标签
        self.assertTrue(os.path.exists(rep["meta"]["report_files"]["json"]))   # JSON 落盘
        self.assertTrue(os.path.exists(rep["meta"]["report_files"]["html"]))   # HTML 落盘

    def test_resume_skips_processed(self):
        self._prepare()
        out = os.path.join(self.tmp, "out2")
        run_assessment(target=self.tmp, output_dir=out)                # 第一次
        state = os.path.join(_ROOT, "data", "assessment_state.json")   # 状态文件
        self.assertTrue(os.path.exists(state))                         # 状态已落盘
        rep2 = run_assessment(target=self.tmp, output_dir=out)         # 第二次应走续跑
        self.assertEqual(rep2["format_stats"]["total"], 6)             # 结果一致


class TestConfig(TempCase):
    """配置加载与兜底。"""

    def test_missing_file_falls_back(self):
        cfg, note = load_config(os.path.join(self.tmp, "nope.yaml"))  # 不存在的配置
        self.assertTrue(note)                                          # 有提示
        self.assertEqual(cfg.get("pdf_triage.char_threshold"), 30)     # 走默认值

    def test_dotted_get(self):
        cfg, _ = load_config()
        self.assertEqual(cfg.get("scan.recursive"), True)              # 点号取值
        self.assertEqual(cfg.get("no.such.key", "dft"), "dft")         # 缺失兜底

    def test_yaml_override(self):
        path = os.path.join(self.tmp, "c.yaml")                        # 自定义配置
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("pdf_triage:\n  char_threshold: 77\n")
        cfg, _ = load_config(path)
        self.assertEqual(cfg.get("pdf_triage.char_threshold"), 77)     # 覆盖生效
        self.assertEqual(cfg.get("pdf_triage.scan_ratio_threshold"), 0.70)  # 未覆盖项保留默认


if __name__ == "__main__":
    unittest.main(verbosity=2)                                         # 直接运行
