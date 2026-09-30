# -*- coding: utf-8 -*-
"""RAG 组件测试：分块 / BM25 / PDF 解析 / 模型懒加载。"""

from app.rag.chunking import chunk_text, split_sentences
from app.rag.bm25 import BM25Encoder
import fitz
import pytest
from app.rag.pdf import extract_text
import threading
from app.rag.models import LazyModelProxy

# ---------- 分块器（原 tests/test_chunking.py） ----------

def test_split_sentences_splits_on_chinese_punctuation():
    text = "第一句话。第二句话！第三句？最后一句；"
    assert split_sentences(text) == ["第一句话。", "第二句话！", "第三句？", "最后一句；"]


def test_split_sentences_keeps_newlines_and_skips_whitespace():
    text = "甲段第一句。\n\n乙段第二句。  "
    assert split_sentences(text) == ["甲段第一句。", "\n\n乙段第二句。"]


def test_split_sentences_splits_english_sentences():
    text = "first sentence. second sentence. third."
    assert split_sentences(text) == ["first sentence.", " second sentence.", " third."]


def test_split_sentences_splits_on_newlines():
    text = "line one\nline two\nline three"
    assert split_sentences(text) == ["line one", "\nline two", "\nline three"]


def test_short_text_produces_single_chunk():
    assert chunk_text("高血压的预防。" ) == ["高血压的预防。"]


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_long_text_is_split_with_no_chunk_over_size():
    text = ("高血压是一种常见的慢性病。" * 100)  # 1300 字
    chunks = chunk_text(text, chunk_size=700, overlap=80)

    assert len(chunks) >= 2
    assert all(len(c) <= 700 for c in chunks)
    # 内容不丢失：每句话至少在分块中出现一次（重叠会导致重复出现，数量 ≥ 100）
    assert "".join(chunks).count("高血压是一种常见的慢性病。") >= 100


def test_adjacent_chunks_overlap():
    text = "句子A。" + "这是一个内容非常丰富的句子，用来填充分块内容。" * 30
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        # 下一块开头承接上一块结尾的重叠部分
        assert nxt.startswith(prev[-40:])


def test_single_oversized_sentence_is_hard_cut():
    text = "长" * 1500 + "。"
    chunks = chunk_text(text, chunk_size=700, overlap=80)

    assert len(chunks) >= 2
    assert all(len(c) <= 700 for c in chunks)

# ---------- BM25 编码器（原 tests/test_bm25.py） ----------

def fake_tokenizer(text):
    return text.split()  # 空格分词，测试可控


DOCS = ["高血压 预防 指南", "高血压 药物 治疗", "糖尿病 饮食 控制"]


def test_encode_documents_produces_sparse_dict_rows():
    encoder = BM25Encoder(tokenizer=fake_tokenizer)
    encoder.fit(DOCS)
    rows = encoder.encode_texts(DOCS)

    assert len(rows) == 3
    assert isinstance(rows[0], dict)
    # 每篇文档的 term 索引一致，值非零
    assert rows[0][encoder.vocab["高血压"]] > 0
    assert rows[1][encoder.vocab["药物"]] > 0
    assert rows[2][encoder.vocab["糖尿病"]] > 0


def test_query_encoding_uses_corpus_idf():
    encoder = BM25Encoder(tokenizer=fake_tokenizer)
    encoder.fit(DOCS)
    row = encoder.encode_query("高血压 怎么办")

    # 查询词必须存在（未登录词丢弃）
    assert encoder.vocab["高血压"] in row
    assert row[encoder.vocab["高血压"]] > 0
    assert "怎么办" not in encoder.vocab


def test_rare_term_gets_higher_weight_than_common_term():
    encoder = BM25Encoder(tokenizer=fake_tokenizer)
    encoder.fit(DOCS)
    row = encoder.encode_query("高血压 糖尿病")

    # 糖尿病只出现在 1 篇，高血压出现在 2 篇：罕见词权重应更高
    assert row[encoder.vocab["糖尿病"]] > row[encoder.vocab["高血压"]]


def test_unknown_terms_are_dropped_without_error():
    encoder = BM25Encoder(tokenizer=fake_tokenizer)
    encoder.fit(DOCS)
    assert encoder.encode_query("不存在的词") == {}


def test_fit_rebuilds_vocab_on_second_fit():
    encoder = BM25Encoder(tokenizer=fake_tokenizer)
    encoder.fit(DOCS)
    encoder.fit(["全新 语料"])
    assert "高血压" not in encoder.vocab
    assert "全新" in encoder.vocab

# ---------- PDF 解析（原 tests/test_pdf_parser.py） ----------

def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    return doc.tobytes()


def test_extract_text_returns_page_content():
    pdf = make_pdf(["salt diet for hypertension patients."])
    text = extract_text(pdf)
    assert "salt diet for hypertension patients" in text


def test_extract_text_joins_multiple_pages():
    pdf = make_pdf(["content of page one.", "content of page two."])
    text = extract_text(pdf)
    assert "content of page one" in text
    assert "content of page two" in text


def test_extract_text_raises_on_invalid_pdf():
    with pytest.raises(ValueError):
        extract_text("这不是一个PDF文件".encode("utf-8"))

# ---------- 模型懒加载代理（原 tests/test_lazy_model.py） ----------

def test_proxy_does_not_load_until_used():
    from types import SimpleNamespace

    calls = []

    def factory():
        calls.append(1)
        return SimpleNamespace(some_attribute=1)

    proxy = LazyModelProxy(factory)
    assert calls == []  # 构造时不加载

    assert proxy.some_attribute == 1  # 访问普通属性触发加载
    assert len(calls) == 1


def test_proxy_returns_same_instance_every_call():
    proxy = LazyModelProxy(lambda: object())
    assert proxy._get() is proxy._get()


def test_proxy_loads_only_once_under_concurrency():
    calls = []

    class Heavy:
        def __init__(self):
            calls.append(1)
            threading.Event().wait(0.01)  # 模拟慢加载

    proxy = LazyModelProxy(Heavy)
    threads = [threading.Thread(target=proxy._get) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1  # 20 线程并发也只加载一次


def test_proxy_delegates_method_calls():
    class Model:
        def predict(self, x):
            return x * 2

    proxy = LazyModelProxy(Model)
    assert proxy.predict(21) == 42


# ---------- PDF 去水印（原 app/rag/watermark.py） ----------

from app.rag.pdf import remove_watermark

_WATERMARK_PNG = None


def _logo_png() -> bytes:
    """10x10 灰色 PNG（模拟 Logo 水印），惰性生成。"""
    global _WATERMARK_PNG
    if _WATERMARK_PNG is None:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
        pix.set_rect(pix.irect, (200, 200, 200))
        _WATERMARK_PNG = pix.tobytes("png")
    return _WATERMARK_PNG


def _build_pdf(content: str, watermark_text: str = None, gray: bool = False,
               rotate: bool = False, image_pages: int = 0) -> bytes:
    """生成测试 PDF：ASCII 正文 + 可选水印字（灰/斜置）+ 可选每页图片水印。

    注意：base-14 字体不支持中文，正文与水印一律用 ASCII。
    """
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"main content page{i + 1}: {content}", fontsize=12)
        if watermark_text:
            color = (0.8, 0.8, 0.8) if gray else (0, 0, 0)
            if rotate:
                # insert_text 的 rotate 只支持 90° 倍数；斜置水印用 morph 旋转矩阵
                page.insert_text(
                    (200, 400), watermark_text, fontsize=24, color=color,
                    morph=(fitz.Point(200, 400), fitz.Matrix(45)),
                )
            else:
                page.insert_text((200, 400), watermark_text, fontsize=24, color=color)
        if image_pages and i < image_pages:
            page.insert_image(fitz.Rect(300, 100, 380, 180), stream=_logo_png())
    return doc.tobytes()


def test_remove_watermark_deletes_rotated_gray_watermark_keeps_content():
    pdf = _build_pdf("hypertension guideline", watermark_text="CONFIDENTIAL", gray=True, rotate=True)
    clean = remove_watermark(pdf)
    text = extract_text(clean)
    assert "CONFIDENTIAL" not in text
    for i in range(1, 4):
        assert f"main content page{i}" in text


def test_remove_watermark_deletes_keyword_watermark_even_if_black():
    # 黑字但含关键词、且多页重复 → 视为水印
    pdf = _build_pdf("main text", watermark_text="internal draft only")
    clean = remove_watermark(pdf)
    text = extract_text(clean)
    assert "internal draft" not in text
    for i in range(1, 4):
        assert f"main content page{i}" in text


def test_remove_watermark_deletes_repeated_image_watermark():
    pdf = _build_pdf("content", image_pages=3)
    clean = remove_watermark(pdf)
    clean_doc = fitz.open(stream=clean, filetype="pdf")
    # 三页的图片水印都被清除
    assert all(len(page.get_image_info()) == 0 for page in clean_doc)
    text = extract_text(clean)
    assert "main content page1" in text


def test_remove_watermark_keeps_single_page_image():
    # 只出现在 1 页的图片是真实插图，不误删
    pdf = _build_pdf("content", image_pages=1)
    clean_doc = fitz.open(stream=remove_watermark(pdf), filetype="pdf")
    assert len(clean_doc[0].get_image_info()) == 1
    assert len(clean_doc[1].get_image_info()) == 0


def test_remove_watermark_passthrough_clean_pdf():
    pdf = _build_pdf("clean document")
    clean = remove_watermark(pdf)
    text = extract_text(clean)
    for i in range(1, 4):
        assert f"main content page{i}" in text
    assert "clean document" in text


# ---------- 数据清洗（原 app/rag/cleaning.py） ----------

from app.rag.cleaning import clean_text


def test_clean_text_removes_control_characters():
    dirty = "高血压\x00患者\x1f应注意�低盐饮食。\r\n"
    assert clean_text(dirty) == "高血压患者应注意低盐饮食。"


def test_clean_text_collapses_blank_lines():
    dirty = "第一段。\n\n\n\n第二段。\n\n第三段。"
    assert clean_text(dirty) == "第一段。\n\n第二段。\n\n第三段。"


def test_clean_text_strips_garbage_lines():
    # 全符号/无中文无字母的行是垃圾
    dirty = "正文内容。\n====\n---\n下一段。"
    cleaned = clean_text(dirty)
    assert "====" not in cleaned
    assert "---" not in cleaned
    assert "正文内容。" in cleaned and "下一段。" in cleaned


def test_clean_text_returns_empty_for_blank_input():
    assert clean_text("") == ""
    assert clean_text("   \n \n") == ""


def test_clean_text_keeps_valid_punctuation_and_numbers():
    text = "收缩压≥140mmHg和（或）舒张压≥90mmHg，每日食盐≤5克。"
    assert clean_text(text) == text


# ---------- 后处理校验（原 app/rag/verification.py） ----------

from app.rag.verification import verify_answer


def test_verify_flags_numbers_not_in_contexts():
    answer = "收缩压降到150mmHg即可。"
    contexts = ["收缩压≥140mmHg和舒张压≥90mmHg可诊断为高血压。"]
    warnings = verify_answer(answer, contexts)
    assert any("150" in w for w in warnings)


def test_verify_no_warnings_when_numbers_covered():
    answer = "收缩压≥140mmHg即为高血压。"
    contexts = ["收缩压≥140mmHg可诊断为高血压。"]
    assert verify_answer(answer, contexts) == []


def test_verify_no_warnings_without_contexts():
    answer = "150mmHg是目标值。"
    assert verify_answer(answer, []) == []


def test_verify_handles_decimals_and_units():
    answer = "每日食盐3.5克。"
    contexts = ["每日食盐摄入量不超过5克。"]
    warnings = verify_answer(answer, contexts)
    assert any("3.5" in w for w in warnings)


def test_verify_dedupes_repeated_numbers():
    answer = "150、150、150。"
    contexts = ["140mmHg。"]
    warnings = verify_answer(answer, contexts)
    assert len(warnings) == 1


# ---------- 摘要（原 app/rag/cleaning.py make_summary） ----------

from app.rag.cleaning import make_summary


def test_make_summary_takes_first_chars_single_line():
    text = "高血压诊断标准：收缩压≥140mmHg。\n第二行内容。"
    s = make_summary(text)
    assert "高血压诊断标准" in s
    assert "\n" not in s


def test_make_summary_respects_max_len():
    text = "很长的文本内容。" * 50
    s = make_summary(text)
    assert len(s) <= 80


def test_make_summary_handles_empty():
    assert make_summary("") == ""


# ---------- 标题分块 / 语义分块（chunking 扩展） ----------

from app.rag.chunking import (
    chunk_by_headings,
    semantic_chunk,
    split_by_headings,
)


def test_split_by_headings_detects_chinese_and_numeric_headings():
    text = (
        "一、诊断标准\n收缩压≥140mmHg可诊断为高血压。\n"
        "二、生活方式\n每日食盐不超过5克。\n"
        "1. 药物治疗\n常用药物有五类。\n"
    )
    sections = split_by_headings(text)
    assert [h for h, _ in sections] == ["一、诊断标准", "二、生活方式", "1. 药物治疗"]
    assert "收缩压≥140" in sections[0][1]
    assert "每日食盐" in sections[1][1]


def test_split_by_headings_returns_single_section_without_headings():
    text = "普通正文没有标题。\n第二行。"
    sections = split_by_headings(text)
    assert len(sections) == 1
    assert sections[0][0] == ""


def test_chunk_by_headings_prepends_heading_to_chunks():
    text = "一、诊断标准\n" + ("收缩压≥140mmHg可诊断为高血压。" * 30) + "\n二、生活方式\n" + ("每日食盐不超过5克。" * 30)
    chunks = chunk_by_headings(text, chunk_size=200, overlap=20)
    assert len(chunks) >= 2
    assert any(c.startswith("一、诊断标准") for c in chunks)
    assert any(c.startswith("二、生活方式") for c in chunks)


def test_semantic_chunk_splits_on_low_similarity_boundary():
    """句子 A-B 相似度高、B-C 相似度低 → 在 B|C 之间切开。"""
    sentences = ["苹果是水果。", "苹果营养丰富。", "汽车需要加油。", "汽车保养很重要。"]

    class VecEmbedder:
        def embed_documents(self, texts):
            # 手工相似度：前两句向量接近，后两句向量接近，两组间正交
            return {
                "苹果是水果。": [1.0, 0.0],
                "苹果营养丰富。": [0.9, 0.1],
                "汽车需要加油。": [0.0, 1.0],
                "汽车保养很重要。": [0.1, 0.9],
            }[texts[0]] if False else [
                {"苹果是水果。": [1.0, 0.0], "苹果营养丰富。": [0.9, 0.1],
                 "汽车需要加油。": [0.0, 1.0], "汽车保养很重要。": [0.1, 0.9]}[t]
                for t in texts
            ]

    text = "".join(sentences)
    chunks = semantic_chunk(text, VecEmbedder(), max_chunk_size=1000, threshold=0.5)

    assert len(chunks) == 2
    assert "苹果" in chunks[0] and "汽车" not in chunks[0]
    assert "汽车" in chunks[1] and "苹果" not in chunks[1]


def test_semantic_chunk_respects_max_size():
    """相似度一直高时，按块大小兜底切分。"""
    sentences = ["同类话题的句子。" * 3] * 20

    class SameVec:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    text = "".join(sentences)
    chunks = semantic_chunk(text, SameVec(), max_chunk_size=300, threshold=0.5)

    assert len(chunks) >= 2
    assert all(len(c) <= 300 + 30 for c in chunks)  # 单句超长时允许少量超出


def test_semantic_chunk_single_sentence_falls_back():
    class SameVec:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    assert semantic_chunk("只有一句。", SameVec(), 300, 0.5) == ["只有一句。"]


# ---------- 父子块切分 ----------

from app.rag.chunking import parent_child_chunk


def test_parent_child_chunk_splits_into_parents_and_children():
    text = ("section one content. " * 60) + ("section two content. " * 60)
    pairs = parent_child_chunk(text, parent_size=300, child_size=100, overlap=20)

    assert len(pairs) >= 2  # 至少两个父块
    for parent_id, parent_text, children in pairs:
        assert parent_id  # 父块有唯一 ID
        assert children, "每个父块至少一个子块"
        assert all(len(c) <= 100 + 25 for c in children)
        assert all("section one" in c or "section two" in c for c in children)
        # 子块内容应能在父块中找到（去掉重叠截断的误差）
        assert any(c[:40] in parent_text for c in children)


def test_parent_child_chunk_ids_are_stable_and_unique():
    text = "content here. " * 80
    pairs1 = parent_child_chunk(text, parent_size=300, child_size=100, overlap=20)
    pairs2 = parent_child_chunk(text, parent_size=300, child_size=100, overlap=20)

    ids = [p[0] for p in pairs1]
    assert len(ids) == len(set(ids))  # 唯一
    assert ids == [p[0] for p in pairs2]  # 稳定（同内容同 ID）


def test_parent_child_chunk_small_text_single_parent():
    pairs = parent_child_chunk("short text.", parent_size=300, child_size=100, overlap=20)
    assert len(pairs) == 1
    assert pairs[0][2]  # 有子块


# ---------- OCR 与表格提取 ----------

from app.rag.ocr import RapidOCREngine, recognize_image


def _fake_png() -> bytes:
    """10x10 灰色 PNG 字节（OCR 测试用假图）。"""
    import fitz as _fitz

    pix = _fitz.Pixmap(_fitz.csRGB, _fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (255, 255, 255))
    return pix.tobytes("png")


def test_recognize_image_delegates_to_engine(monkeypatch):
    # 模拟引擎包装层：输入图片字节，返回排序拼接后的文本
    class FakeEngine:
        def __call__(self, image_bytes):
            return "文字A\n文字B"

    monkeypatch.setattr("app.rag.ocr.get_engine", lambda: FakeEngine())
    text = recognize_image(_fake_png())
    assert "文字A" in text and "文字B" in text


def test_rapid_ocr_engine_interface():
    # RapidOCR 引擎对象可直接调用；这里只验证包装类构造不需要安装引擎（懒加载）
    engine = RapidOCREngine()
    assert engine._engine is None


def test_extract_tables_from_pdf():
    """带表格线条的 PDF：pdfplumber 能识别出表格并转 markdown 文本。"""
    from app.rag.pdf import extract_tables

    doc = fitz.open()
    page = doc.new_page()
    # 画表格线 + 单元格文字（ASCII，base-14 字体）
    for x in (100, 200, 300):
        page.draw_line((x, 100), (x, 180))
    for y in (100, 140, 180):
        page.draw_line((100, y), (300, y))
    page.insert_text((110, 130), "name", fontsize=10)
    page.insert_text((210, 130), "value", fontsize=10)
    page.insert_text((110, 170), "salt", fontsize=10)
    page.insert_text((210, 170), "5g", fontsize=10)
    pdf = doc.tobytes()

    tables = extract_tables(pdf)

    assert tables, "应识别出至少一个表格"
    joined = "\n".join(tables)
    assert "name" in joined and "value" in joined
    assert "salt" in joined and "5g" in joined


def test_extract_tables_empty_without_tables():
    from app.rag.pdf import extract_tables

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "plain text no table.", fontsize=12)
    assert extract_tables(doc.tobytes()) == []
