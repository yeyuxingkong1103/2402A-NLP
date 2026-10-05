import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz
import httpx
from openai import AsyncOpenAI

from .config import Settings


@dataclass
class ParsedChunk:
    content: str
    page_number: int
    section_title: str
    content_type: str = "text"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks_for_page(text: str, page_number: int, section_title: str) -> list[ParsedChunk]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks = []
    start = 0
    size, overlap = 850, 120
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            boundary = max(normalized.rfind("。", start, end), normalized.rfind("；", start, end))
            if boundary > start + 400:
                end = boundary + 1
        chunks.append(ParsedChunk(normalized[start:end], page_number, section_title))
        if end == len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def parse_with_pymupdf(path: Path) -> tuple[list[ParsedChunk], int]:
    chunks: list[ParsedChunk] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text")
            title = ""
            for line in text.splitlines():
                line = line.strip()
                if 2 <= len(line) <= 80 and (line.startswith(("第", "一、", "二、", "三、")) or "目录" in line):
                    title = line
                    break
            chunks.extend(_chunks_for_page(text, index, title))
        return chunks, len(document)


class DashScopeVisionClient:
    """Call Qwen-VL-Max through Alibaba Cloud Model Studio's OpenAI-compatible API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.dashscope_api_key or "missing",
            base_url=settings.dashscope_base_url,
        )

    async def describe_image(self, image_url: str, instruction: str = "提取图片中的文字和表格，保留表头、单位、年份及行列关系。") -> str:
        if not self.settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured")
        response = await self.client.chat.completions.create(
            model=self.settings.dashscope_vl_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是招股说明书图片和表格解析器，只输出忠实于图片的结构化文本。"},
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
        )
        return response.choices[0].message.content or ""


class MinerUClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def parse(self, path: Path) -> tuple[list[ParsedChunk], int]:
        endpoint = self.settings.mineru_api_url or (f"{self.settings.mineru_base_url.rstrip('/')}/parse" if self.settings.mineru_base_url else "")
        if not endpoint:
            raise RuntimeError("MINERU_API_URL or MINERU_BASE_URL is not configured")
        headers = {"Authorization": f"Bearer {self.settings.mineru_api_key}"} if self.settings.mineru_api_key else {}
        async with httpx.AsyncClient(timeout=self.settings.mineru_timeout_seconds) as client:
            with path.open("rb") as source:
                response = await client.post(endpoint, files={"file": (path.name, source, "application/pdf")}, headers=headers)
            response.raise_for_status()
            payload = response.json()
            task_id = payload.get("task_id") or payload.get("data", {}).get("task_id")
            if task_id:
                base_url = endpoint.rsplit("/", 1)[0]
                for _ in range(90):
                    await asyncio.sleep(2)
                    status = await client.get(f"{base_url}/extract/task/{task_id}", headers=headers)
                    status.raise_for_status()
                    payload = status.json()
                    state = str(payload.get("status") or payload.get("data", {}).get("state") or "").lower()
                    if state in {"completed", "success", "done", "succeeded"}:
                        break
                    if state in {"failed", "error", "cancelled"}:
                        raise RuntimeError(payload.get("message") or payload.get("data", {}).get("err_msg", "MinerU parsing failed"))
                else:
                    raise TimeoutError("MinerU parsing timed out")
            return self._payload_to_chunks(payload)

    @staticmethod
    def _payload_to_chunks(payload: dict) -> tuple[list[ParsedChunk], int]:
        chunks = []
        for item in payload.get("chunks", payload.get("pages", [])):
            content = item.get("content") or item.get("text") or item.get("markdown") or ""
            if content.strip():
                chunks.append(ParsedChunk(content.strip(), int(item.get("page_number", item.get("page", 1))), item.get("section_title", ""), item.get("content_type", "text")))
        return chunks, int(payload.get("pages", payload.get("page_count", 0)))


def write_parsed(chunks: list[ParsedChunk], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
