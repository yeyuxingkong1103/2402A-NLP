# -*- coding: utf-8 -*-
"""CCPIA 附件下载器离线单测 + 已下载产物的落地校验（不联网）。"""
import csv
import importlib.util
from pathlib import Path

import fitz
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
PDF_NAME = "中国农药工业协会关于发布《水稻安全科学使用农药指南》等6项团体标准的公告.pdf"
PDF_PATHS = (EXT / PDF_NAME, RAW / PDF_NAME)
HREF = "/upload/default/20250308/7d6b88b71ee7ab38a158da8e776a76ba.pdf"
LABEL = "中国农药工业协会关于发布《水稻安全科学使用农药指南》等6项团体标准的公告.pdf"
BASE = "https://www.ccpia.com.cn/xiehuidongtai/77191.html"


def _load():
    """按文件路径动态加载被测模块（src/ingest/download_ccpia.py），避免依赖包安装方式。"""
    spec = importlib.util.spec_from_file_location("cc", ROOT / "src/ingest/download_ccpia.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load()
# 构造一段假的公告页 HTML：含一个相对链接的附件 + 一个普通站内链接（不应被当作附件）
HTML = f'<p>附件：<a href="{HREF}" target="_blank">{LABEL}</a></p><a href="/about.html">关于</a>'


def test_找附件_相对链接转绝对且只挑附件():
    """附件链接要从相对路径拼成绝对 URL，且只挑出附件链接、忽略「关于」这类站内导航。"""
    items = cc.find_attachments(HTML, BASE)
    assert items == [("https://www.ccpia.com.cn" + HREF, LABEL)]


def test_文件名用链接文字且不重复扩展名():
    """有链接文字时文件名取链接文字；若文字已带 .pdf，不得再追加一次扩展名变成 .pdf.pdf。"""
    assert cc.attachment_filename(HREF, LABEL) == LABEL
    assert not cc.attachment_filename(HREF, LABEL).endswith(".pdf.pdf")


def test_文件名回退到URL并清洗非法字符():
    """链接没文字时回退用 URL 末段；文件名里的 Windows 非法字符（: / * ? 等）替换成下划线，全空白则用兜底名。"""
    assert cc.attachment_filename("https://x.cn/a/b/c/d/7d6b88b71ee7.pdf", "") == "7d6b88b71ee7.pdf"
    assert cc.safe('公告:v1/终稿*?', "x") == "公告_v1_终稿__"
    assert cc.safe("   ", "fallback") == "fallback"


@pytest.mark.skipif(not any(p.exists() for p in PDF_PATHS), reason="公告 PDF 未下载")
def test_公告PDF内容与页数():
    """对真实下载的公告 PDF 做落地校验：共 2 页，且关键事实（标准编号、联系邮箱等）都能在正文里找到。"""
    path = next(p for p in PDF_PATHS if p.exists())
    d = fitz.open(path)
    assert d.page_count == 2
    text = "".join(d[p].get_text() for p in range(d.page_count))
    for kw in ("中农协", "15 号", "T/CCPIA 262-2025", "小麦安全科学使用农药指南", "ccpia_ttbz@163.com"):
        assert kw in text, kw


def test_公告PDF放在_external_不进语料目录():
    """公告是「缺口证据」不是语料：必须放 data/external，否则会污染 26 份/433 页的统计口径。"""
    assert not (RAW / PDF_NAME).exists(), "公告 PDF 混进了 data/raw"
    assert "data/external" in (ROOT / "claude.md").read_text(encoding="utf-8")


def test_清单分开写不覆盖国标批次清单():
    """CCPIA 批次的下载清单要单独成文件；若国标批次的 _下载清单.csv 已存在，其表头结构（8 列、含「标准号」）必须原样保留。"""
    ccpia = next((p for p in (EXT / "_下载清单_ccpia.csv", RAW / "_下载清单_ccpia.csv") if p.exists()), None)
    if ccpia is not None:                 # CCPIA 清单只登记公告这一份文件
        rows = list(csv.DictReader(open(ccpia, encoding="utf-8-sig")))
        assert len(rows) == 1 and rows[0]["文件"] == PDF_NAME
    gb = RAW / "_下载清单.csv"
    if gb.exists():                       # 国标批次清单必须仍是 verify_pdfs 的列结构，未被覆盖
        header = next(csv.reader(open(gb, encoding="utf-8-sig")))
        assert "标准号" in header and len(header) == 8
