"""使用可选 PDF、OCR 和视觉模型能力解析图像。"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.models import ImageRecord


MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",
}


def validate_image(image_path: str | Path, max_bytes: int = MAX_IMAGE_BYTES, verify_format: bool = True) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"图像文件不存在：{path}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"图像文件大小不合理：{size} bytes")
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as image:
            image.verify()
            return image.get_format_mimetype() or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    except ImportError:
        header = path.read_bytes()[:12]
        for signature, mime in _IMAGE_SIGNATURES.items():
            if header.startswith(signature):
                return mime
        raise ValueError("无法校验图像格式：请安装 Pillow 或提供有效图像签名")
    except Exception as exc:
        raise ValueError(f"图像格式无效：{path}") from exc


def _image_text(record: ImageRecord) -> str:
    return record.content

def make_image_record(
    source: str,
    page: int,
    image_path: str | Path,
    ocr_text: str = "",
    description: str = "",
    image_error: str = "",
    index: Any | None = None,
) -> ImageRecord:
    """创建图像记录，并将 OCR/描述文本写入统一索引。"""
    path = Path(image_path)
    text = "\n".join(part.strip() for part in (ocr_text, description) if part.strip())
    metadata = {
        "image_path": str(path),
        "image_name": path.name,
        "ocr_text": ocr_text,
        "description": description,
        "modality": "image",
    }
    if image_error:
        metadata["image_error"] = image_error
    record = ImageRecord(source=source, page=page, content=text, metadata=metadata, path=str(path))
    if index is not None:
        if hasattr(index, "add_image"):
            index.add_image(record)
        elif hasattr(index, "add_text"):
            index.add_text(record.source, record.page, record.content, metadata={**record.metadata, "image_id": record.image_id})
        else:
            raise TypeError("index 必须支持 add_image 或 add_text")
    return record


def describe_image(image_path: str | Path, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> str:
    """调用兼容 OpenAI 的视觉接口；没有配置时不伪造识别结果。"""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return ""
    path = Path(image_path)
    try:
        validate_image(path)
    except ValueError:
        return ""
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        payload = {"model": model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages": [{"role": "user", "content": [{"type": "text", "text": "请用中文准确描述这张图像，无法确认的内容不要猜测。"}, {"type": "image_url", "image_url": {"url": f"data:image/{path.suffix.lstrip('.') or 'png'};base64,{encoded}"}}]}], "max_tokens": 300}
        request = Request(url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        return ""


def ocr_image(image_path: str | Path) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(image_path), lang="chi_sim+eng").strip()
    except Exception:
        return ""


def extract_images(pdf_path: str | Path, pages: Iterable[int] | None = None, output_dir: str | Path | None = None, index: Any | None = None) -> list[ImageRecord]:
    """从 PDF 提取图片并构建图像记录。"""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF 图像提取需要安装 pymupdf") from exc
    path = Path(pdf_path)
    target = Path(output_dir) if output_dir else path.parent / f"{path.stem}_images"
    target.mkdir(parents=True, exist_ok=True)
    selected = set(pages or [])
    records: list[ImageRecord] = []
    with fitz.open(str(path)) as document:
        for page_number, page in enumerate(document, 1):
            if selected and page_number not in selected:
                continue
            for image_number, image in enumerate(page.get_images(full=True), 1):
                extracted = document.extract_image(image[0])
                image_path = target / f"page-{page_number}-image-{image_number}.{extracted['ext']}"
                image_path.write_bytes(extracted["image"])
                ocr = ocr_image(image_path)
                description = ""
                image_error = ""
                try:
                    description = describe_image(image_path)
                except ValueError as exc:
                    image_error = str(exc)
                records.append(make_image_record(str(path), page_number, image_path, ocr, description, image_error, index))
    return records


def index_images(records: Iterable[ImageRecord], index: Any) -> None:
    for record in records:
        if hasattr(index, "add_image"):
            index.add_image(record)
        elif hasattr(index, "add_text"):
            index.add_text(record.source, record.page, record.content, metadata={**record.metadata, "image_id": record.image_id})
        else:
            raise TypeError("index 必须支持 add_image 或 add_text")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提取 PDF 图像并进行 OCR/视觉描述")
    parser.add_argument("path", type=Path)
    parser.add_argument("--page", type=int, action="append")
    args = parser.parse_args(argv)
    try:
        for record in extract_images(args.path, args.page):
            print(f"[{record.source} 第{record.page}页] {record.content}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
