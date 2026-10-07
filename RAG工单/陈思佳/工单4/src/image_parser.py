from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ImageAsset:
    source: str
    page: int
    index: int
    path: str
    ocr_text: str = ""


def extract_images(path: str | Path, output_dir: str | Path) -> list[ImageAsset]:
    """Extract embedded PDF images; OCR is optional and never blocks extraction."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("请先安装 pypdf：pip install pypdf") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(path)
    assets: list[ImageAsset] = []
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        for image_index, image in enumerate(page.images):
            target = output / f"{pdf_path.stem}_p{page_number}_{image_index}{Path(image.name).suffix or '.bin'}"
            target.write_bytes(image.data)
            assets.append(ImageAsset(str(pdf_path), page_number, image_index, str(target)))
    return assets


def ocr_image(asset: ImageAsset, language: str = "chi_sim+eng") -> ImageAsset:
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return asset
    text = pytesseract.image_to_string(Image.open(asset.path), lang=language).strip()
    return ImageAsset(asset.source, asset.page, asset.index, asset.path, text)


def image_documents(assets: Iterable[ImageAsset]):
    from rag_core import Document
    return [
        Document(
            text=asset.ocr_text or f"PDF 第 {asset.page} 页图像 {asset.index}",
            source=asset.source,
            page=asset.page,
            metadata={"content_type": "image", "image_path": asset.path},
        )
        for asset in assets
    ]
