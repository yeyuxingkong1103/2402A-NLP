"""文档解析：PDF / txt / markdown / 图片。

PDF 解析策略（分层兜底）：
1. PyMuPDF(fitz) 逐页抽取文本（速度快、保留段落结构）
2. 单页文本过少（疑似扫描件）的页，渲染成图交给 OCR（RapidOCR）
3. 整篇仍过少时，文件路径再退回 pdfplumber（含表格抽取）

纯图片（png/jpg/...）直接走 OCR。OCR 受 settings.ocr_enabled 控制，关闭时
图片解析返回空文本，走调用方（ingest / upload）的「无有效文本」错误路径。
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from ..config import settings
from ..logging_config import get_logger
from .cleaner import clean_text
from .ocr import ocr_image

MIN_PDF_TEXT_LEN = 50   # 整篇：低于此值判为「基本没文本层」
MIN_PAGE_TEXT_LEN = 10  # 单页：低于此值判为「该页是扫描图，需 OCR」
OCR_PDF_DPI = 300       # 扫描页渲染分辨率（OCR 精度与速度的平衡）

# 图片没有文本层，只能走 OCR：settings.ocr_enabled=False 时这些格式一律解析出空串，
# 由调用方按「无有效文本」报错（见 app/api/knowledge.py 的 upload）。
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

log = get_logger("parser")


def _decode_text(data: bytes, name: str = "") -> str:
    """按常见编码解码文本文件。

    以前是 `decode("utf-8", errors="ignore")`：GBK/GB18030（中文 Windows 记事本的默认
    保存格式）或 UTF-16 的 txt 会被静默读成乱码——实测 108 字正文变 44 字乱码，垃圾
    chunk 照样入库、进 prompt，全程没有任何报错。用户只会发现"检索答非所问"。

    顺序：BOM 优先 → UTF-8 → gb18030（GBK 的超集，中文环境兜底）。gb18030 几乎不会
    解码失败，所以它后面还能再兜一层 replace 的情况很少见。
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "gb18030"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        if enc == "gb18030":
            log.info("文本 %s 不是 UTF-8，按 gb18030 解码（中文 Windows 常见）", name)
        return text
    log.warning("文本 %s 无法按常见编码解码，退化为 utf-8 + replace（内容可能有乱码）", name)
    return data.decode("utf-8", errors="replace")


class DocumentParser:
    SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".markdown"} | IMAGE_EXTS

    def parse_file(self, path: str | Path) -> dict:
        """磁盘文件 → {title, source, ext, text}，text 已过 clean_text。

        不支持的扩展名抛的是 **ValueError**（而不是别的异常类型）：CLI 批量入库靠
        这个类型把「这个文件跳过、继续下一个」和真正的故障区分开（见 scripts/ingest.py
        里 `except ValueError` 后只打一条 warning 就 continue）。
        """
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in self.SUPPORTED_EXTS:
            raise ValueError(f"不支持的文件类型: {ext}（支持 {sorted(self.SUPPORTED_EXTS)}）")

        if ext == ".pdf":
            raw = self._parse_pdf(p)
        elif ext in IMAGE_EXTS:
            raw = self._parse_image_file(p)
        else:
            raw = _decode_text(p.read_bytes(), p.name)

        text = clean_text(raw)
        # source 统一成正斜杠：它会进 Milvus 过滤表达式，而 quote_literal 拒绝反斜杠
        # （表达式里 `\` 能改写字符串边界）。原生 Windows 上 str(Path) 带 `\`，
        # 那份文档一入库就抛 ValueError，整轮 seed/ingest 直接中断。
        return {"title": p.name, "source": p.as_posix(), "ext": ext, "text": text}

    def parse_bytes(self, filename: str, data: bytes) -> dict:
        """内存字节 → 与 parse_file 同构的 dict，供 HTTP 上传用（没有磁盘路径）。

        source 固定为 ``upload://文件名``：它要进 Milvus 过滤表达式，必须是安全字面量
        （同样不能带反斜杠，理由见上面 parse_file），同时用前缀把「上传」与「磁盘路径」
        两种来源分开——文档级去重/重灌是按 source 认的，同名文件走两条入口互不覆盖。
        """
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            raw = self._parse_pdf_bytes(data)
        elif ext in IMAGE_EXTS:
            raw = self._parse_image_bytes(data)
        elif ext in {".txt", ".md", ".markdown"}:
            raw = _decode_text(data, filename)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

        text = clean_text(raw)
        return {"title": filename, "source": f"upload://{filename}", "ext": ext, "text": text}

    # ---- PDF ----

    def _parse_pdf(self, path: Path) -> str:
        """fitz 为主、pdfplumber 兜底。fitz 缺失或 PDF 损坏都被下面吞成空串，
        所以没装 PyMuPDF 的环境照样能跑，只是少了「扫描页按页 OCR」这一层。
        """
        try:
            import fitz  # PyMuPDF

            with fitz.open(path) as doc:
                text = self._parse_pdf_doc(doc)
        except Exception:  # noqa: BLE001
            text = ""
        if len(text.strip()) >= MIN_PDF_TEXT_LEN:
            return text
        fallback = self._parse_pdf_plumber(path)
        return fallback or text

    def _parse_pdf_bytes(self, data: bytes) -> str:
        """与 _parse_pdf（文件路径）保持同一套兜底策略。

        以前这条路径只有 PyMuPDF 一条路：抽不出文本就直接返回空串，而文件路径那条
        在文本过少时还会退回 pdfplumber（表格抽取）。同一个 PDF 走上传（bytes）
        与走 CLI（路径）能得到完全不同的结果，且没有任何提示。
        """
        try:
            import fitz  # PyMuPDF

            with fitz.open(stream=data, filetype="pdf") as doc:
                text = self._parse_pdf_doc(doc)
        except Exception:  # noqa: BLE001
            text = ""
        if len(text.strip()) >= MIN_PDF_TEXT_LEN:
            return text
        return self._parse_pdf_plumber(BytesIO(data)) or text

    def _parse_pdf_doc(self, doc) -> str:
        """逐页抽取；无文本层的页渲染成图走 OCR（覆盖纯扫描件 + 混合 PDF）。"""
        parts: list[str] = []
        for page in doc:
            page_text = (page.get_text() or "").strip()
            if len(page_text) >= MIN_PAGE_TEXT_LEN:
                parts.append(page_text)
                continue
            # 单页 OCR 失败不能连累整篇：OCR 引擎缺失/模型加载失败/ONNX 报错都会抛，
            # 而这里以前没有 per-page 兜底——异常穿透页循环后，_parse_pdf 的
            # `except: text = ""` 会把**已经抽出来的正文一起丢掉**，上传只报
            # 「无有效文本（扫描件 OCR 未识别成功）」，真因被掩盖。
            try:
                ocr = self._ocr_pdf_page(page)
            except Exception as exc:  # noqa: BLE001
                log.warning("第 %s 页 OCR 失败，跳过该页: %s", getattr(page, "number", "?"), exc)
                continue
            if ocr:
                parts.append(ocr)
        return "\n".join(parts)

    def _ocr_pdf_page(self, page) -> str:
        if not settings.ocr_enabled:
            return ""
        try:
            # alpha=False 是必须的：get_pixmap 默认可能带 alpha 通道，而下面按 "RGB"
            # 3 通道还原字节流，通道数不匹配时 frombytes 会直接抛。
            pix = page.get_pixmap(dpi=OCR_PDF_DPI, alpha=False)
            from PIL import Image

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception:  # noqa: BLE001
            return ""
        return ocr_image(img)

    # ---- 图片 ----
    def _parse_image_file(self, path: Path) -> str:
        # convert("RGB")：PNG 的 alpha、tiff / 调色板图的像素模式都不能直接喂 OCR。
        # 打不开（后缀是图片但内容不是、文件损坏）时返回空串——和「图里没字」在返回值
        # 上不可区分，上游只能统一报「无有效文本」，排障时得自己看图。
        if not settings.ocr_enabled:
            return ""
        try:
            from PIL import Image

            with Image.open(path) as img:
                return ocr_image(img.convert("RGB"))
        except Exception:  # noqa: BLE001
            return ""

    def _parse_image_bytes(self, data: bytes) -> str:
        if not settings.ocr_enabled:
            return ""
        try:
            from PIL import Image

            with Image.open(BytesIO(data)) as img:
                return ocr_image(img.convert("RGB"))
        except Exception:  # noqa: BLE001
            return ""

    # ---- pdfplumber 兜底（表格抽取）----
    @staticmethod
    def _parse_pdf_plumber(source) -> str:
        """source 可以是路径，也可以是字节流（pdfplumber 两者都收）。"""
        try:
            import pdfplumber

            parts = []
            with pdfplumber.open(source) as pdf:
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
            return "\n".join(parts)
        except Exception:  # noqa: BLE001
            return ""
