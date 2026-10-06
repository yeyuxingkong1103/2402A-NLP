from __future__ import annotations

"""教育资料网页爬虫，负责抓取网页、发现附件并保存原始资源。"""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import re

import httpx
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover - 可选依赖降级
    trafilatura = None

try:
    from slugify import slugify
except ImportError:  # pragma: no cover - 可选依赖降级
    def slugify(value: str, max_length: int = 80, lowercase: bool = False) -> str:
        text = re.sub(r"[^\w一-鿿-]+", "-", value).strip("-")
        if lowercase:
            text = text.lower()
        return text[:max_length]

from ..config.config import CrawlerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrawledResource:
    source_url: str
    local_path: Path
    content_type: str
    title: str
    kind: str


class EducationCrawler:
    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            headers={"User-Agent": config.user_agent},
            timeout=config.request_timeout_seconds,
            follow_redirects=True,
        )

    def crawl(self) -> list[CrawledResource]:
        visited: set[str] = set()
        queue = list(self.config.seed_urls)
        resources: list[CrawledResource] = []

        while queue and len(visited) < self.config.max_pages:
            url = queue.pop(0)
            if url in visited or not self._is_allowed(url):
                continue

            logger.info("开始抓取：%s", url)
            visited.add(url)
            try:
                response = self.client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as error:
                logger.warning("抓取失败：%s，原因：%s", url, error)
                continue

            resource = self._save_response(url, response)
            resources.append(resource)

            if self.config.follow_links and resource.kind == "html":
                queue.extend(self._extract_links(url, response.text, visited))

            resources.extend(self._download_linked_files(url, response.text))
            time.sleep(self.config.delay_seconds)

        return resources

    def close(self) -> None:
        self.client.close()

    def _save_response(self, url: str, response: httpx.Response) -> CrawledResource:
        content_type = response.headers.get("content-type", "").split(";")[0].lower()
        suffix = self._guess_suffix(url, content_type)
        title = self._extract_title(response.text) if "html" in content_type else Path(urlparse(url).path).name
        file_name = self._build_file_name(url, title, suffix)
        local_path = self.config.output_dir / file_name

        if suffix in {".html", ".htm"}:
            text = self._html_to_markdown(url, response.text)
            local_path = local_path.with_suffix(".md")
            local_path.write_text(text, encoding="utf-8")
            return CrawledResource(url, local_path, content_type, title, "markdown")

        local_path.write_bytes(response.content)
        return CrawledResource(url, local_path, content_type, title, suffix.lstrip("."))

    def _download_linked_files(self, base_url: str, html: str) -> list[CrawledResource]:
        soup = BeautifulSoup(html, "html.parser")
        found: list[CrawledResource] = []
        seen: set[str] = set()

        for tag in soup.find_all("a", href=True):
            href = str(tag["href"]).strip()
            url = urljoin(base_url, href)
            lower_path = urlparse(url).path.lower()
            if not any(lower_path.endswith(ext) for ext in self.config.download_file_types):
                continue
            if self.config.allowed_file_urls and url not in self.config.allowed_file_urls:
                continue
            if url in seen or not self._is_allowed(url):
                continue
            seen.add(url)
            logger.info("发现附件：%s", url)
            try:
                response = self.client.get(url)
                response.raise_for_status()
                found.append(self._save_response(url, response))
            except httpx.HTTPError as error:
                logger.warning("附件下载失败：%s，原因：%s", url, error)
        return found

    def _extract_links(self, base_url: str, html: str, visited: set[str]) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            url = urljoin(base_url, str(tag["href"]).strip())
            if url not in visited and self._is_allowed(url):
                links.append(url)
        return links

    def _is_allowed(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        return any(domain == item or domain.endswith(f".{item}") for item in self.config.allowed_domains)

    @staticmethod
    def _html_to_markdown(url: str, html: str) -> str:
        if trafilatura is not None:
            extracted = trafilatura.extract(html, url=url, output_format="markdown")
            if extracted:
                return extracted
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n", strip=True)

    @staticmethod
    def _extract_title(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        heading = soup.find(["h1", "h2"])
        return heading.get_text(strip=True) if heading else "untitled"

    @staticmethod
    def _guess_suffix(url: str, content_type: str) -> str:
        path_suffix = Path(urlparse(url).path).suffix.lower()
        if path_suffix:
            return path_suffix
        if "pdf" in content_type:
            return ".pdf"
        if "word" in content_type or "officedocument" in content_type:
            return ".docx"
        if "html" in content_type:
            return ".html"
        return ".bin"

    @staticmethod
    def _build_file_name(url: str, title: str, suffix: str) -> str:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        safe_title = slugify(title, max_length=80, lowercase=False) or "document"
        return f"{safe_title}-{digest}{suffix}"
