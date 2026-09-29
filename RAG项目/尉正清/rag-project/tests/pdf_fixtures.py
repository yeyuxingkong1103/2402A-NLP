# tests/pdf_fixtures.py
"""合成 PDF 夹具。

`test_pdf_pipeline.py` 与 `test_knowledge.py` 共用这里的构造函数。

刻意**不用**真实出版物（如 WHO 报告）做夹具：现场合成的文档体积小、结构可控，
能精确断言「哪段该删、哪段该留」，也不会因为外部数据换版而失效——
何况那类文档有版权，本就不随仓库分发（见 `data/sources/README.md`）。
"""
import pymupdf

WM_TEXT = "内部资料 请勿外传"
PAGES = 12                       # 水印判定要求出现在半数以上页面，12 页给足余量
BODY_MARK = "儿童青少年与老年人"   # 正文尾句的标识串，用于断言未被误删


def blank_doc(pages: int = PAGES):
    """有正文的文档：每页一段会折行的中文段落。

    段落刻意长到折行——折出来的短续行正是水印误判的高发区。
    """
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(60, 60, 540, 300),
            "第 %d 页正文段落。这一段用于验证去水印不会误删正文内容。"
            "精神卫生服务应当覆盖全人群，包括%s。" % (i + 1, BODY_MARK),
            fontname="china-s", fontsize=12)
    return doc


def noise_pixmap(size: int = 120):
    """有噪声的图。

    纯色块压缩后不足 `MIN_IMAGE_BYTES`，指纹不可靠会被跳过，
    所以这里用噪声保证图片数据够大、能走指纹比对那条分支。
    """
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, size, size))
    for y in range(size):
        for x in range(size):
            pix.set_pixel(x, y, ((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256))
    return pix


def with_text_watermark(doc):
    """每页同一位置盖一处斜向浅灰短文本——最常见的文字水印形态。"""
    for page in doc:
        page.insert_textbox(pymupdf.Rect(60, 400, 540, 480), WM_TEXT,
                            fontname="china-s", fontsize=30,
                            color=(0.75, 0.75, 0.75), align=1)
    return doc


def with_image_watermark(doc):
    """每页同一位置盖一个重复铺版的 logo。"""
    pix = noise_pixmap()
    for page in doc:
        page.insert_image(pymupdf.Rect(460, 500, 540, 580), pixmap=pix)
    return doc


def save(doc, path) -> str:
    doc.save(str(path), garbage=3, deflate=True)
    doc.close()
    return str(path)


def text_of(path) -> str:
    """取出 PDF 的文本层全量内容（不做任何归一化）。"""
    doc = pymupdf.open(str(path))
    try:
        return "\n".join(doc[i].get_text("text") or "" for i in range(doc.page_count))
    finally:
        doc.close()


def watermarked_bytes() -> bytes:
    """带文字水印的 PDF 字节流，供接口上传用例使用。"""
    import io
    buf = io.BytesIO()
    doc = with_text_watermark(blank_doc())
    doc.save(buf, garbage=3, deflate=True)
    doc.close()
    return buf.getvalue()
