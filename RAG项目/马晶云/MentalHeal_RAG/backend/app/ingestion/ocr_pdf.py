from pathlib import Path
from typing import Any

import fitz
import numpy as np

from .schemas import ExtractedPage
from .text_processing import normalize_text


class OcrPdfParser:
    name = "paddleocr"

    def __init__(self, language: str = "ch", use_gpu: bool = False) -> None:
        self.language = language
        self.use_gpu = use_gpu

    def parse(self, path: Path) -> dict[int, str]:
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError(
                "PaddleOCR is required. Install paddleocr and a compatible paddlepaddle package."
            ) from error

        device = "gpu" if self.use_gpu else "cpu"
        ocr = PaddleOCR(lang=self.language, device=device)
        results: dict[int, str] = {}
        with fitz.open(path) as document:
            for index, page in enumerate(document, start=1):
                image = self._page_image(page)
                result = ocr.predict(image)
                text = self._result_text(result)
                if text:
                    results[index] = text
        return results

    def apply(self, pages: list[ExtractedPage], ocr_by_page: dict[int, str]) -> None:
        for page in pages:
            page.ocr_text = normalize_text(ocr_by_page.get(page.page_number, ""))

    @staticmethod
    def _page_image(page: fitz.Page) -> Any:
        matrix = fitz.Matrix(2.0, 2.0)
        image = page.get_pixmap(matrix=matrix, alpha=False).pil_image()
        return np.asarray(image)

    @staticmethod
    def _result_text(result: Any) -> str:
        texts: list[str] = []
        items = result if isinstance(result, list) else [result]
        for item in items:
            payload = item if isinstance(item, dict) else getattr(item, "json", {})
            if not isinstance(payload, dict):
                continue
            for value in payload.get("rec_texts", []):
                if value:
                    texts.append(str(value))
        return "\n".join(texts)
