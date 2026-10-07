r"""
一次性建库：MinerU API 解析 PDF -> 切分 -> BGE 向量 -> FAISS 索引。

运行：
    C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe prepare.py
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag import CHUNKS_PATH, DATA_DIR, INDEX_PATH, MODEL_PATH, Chunk, save_faiss_index


PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = Path(r"C:\Users\lenovo\Desktop\RAG 工单 (1)\RAG 工单\附件\招股说明书1.pdf")
MINERU_DIR = DATA_DIR / "mineru"
MERGED_PATH = MINERU_DIR / "content_items.json"
PAGE_RANGES = ((1, "1-200"), (201, "201-400"), (401, "401-548"))
BASE_URL = "https://mineru.net/api/v4"
MAX_NETWORK_ATTEMPTS = 4


class MinerUClient:
    def __init__(self, token: str):
        if not token:
            raise ValueError("请先在 .env 中填写 MINERU_API_KEY")
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {token}"}

    def _json(self, response: requests.Response) -> dict:
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError("MinerU 返回了非 JSON 数据") from None
        if payload.get("code") not in (0, None):
            raise RuntimeError(f"MinerU 请求失败：{payload.get('msg', '未知错误')}")
        return payload

    @staticmethod
    def _can_retry(exc: requests.RequestException) -> bool:
        if not isinstance(exc, requests.HTTPError):
            return True
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500

    @staticmethod
    def _pause(attempt: int, action: str) -> None:
        seconds = min(2 ** attempt, 8)
        print(f"{action}连接中断，{seconds} 秒后重试...")
        time.sleep(seconds)

    def _upload(self, url: str, pdf_path: Path) -> None:
        last_error = None
        for attempt in range(1, MAX_NETWORK_ATTEMPTS + 1):
            try:
                with pdf_path.open("rb") as stream:
                    response = self.session.put(url, data=stream, timeout=(30, 180))
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                last_error = exc
                if attempt == MAX_NETWORK_ATTEMPTS or not self._can_retry(exc):
                    break
                self._pause(attempt, "上传")
        raise RuntimeError(f"MinerU 上传失败，已重试 {MAX_NETWORK_ATTEMPTS} 次：{last_error}") from last_error

    def submit(self, pdf_path: Path) -> str:
        files = [
            {
                "name": f"prospectus_part_{number}.pdf",
                "data_id": f"prospectus_part_{number}",
                "page_ranges": page_range,
            }
            for number, (_start, page_range) in enumerate(PAGE_RANGES, 1)
        ]
        body = {
            "files": files,
            "model_version": "vlm",
            "language": "ch",
            "enable_table": True,
            "enable_formula": False,
        }
        response = self.session.post(
            f"{BASE_URL}/file-urls/batch",
            headers={**self.headers, "Content-Type": "application/json"},
            json=body,
            timeout=30,
        )
        data = self._json(response).get("data", {})
        batch_id = data.get("batch_id")
        urls = data.get("file_urls", [])
        if not batch_id or len(urls) != 3:
            raise RuntimeError("MinerU 未返回完整的上传信息")
        for number, url in enumerate(urls, 1):
            print(f"上传第 {number}/{len(urls)} 个页码范围...")
            self._upload(url, pdf_path)
        return batch_id

    @staticmethod
    def _result_list(payload: dict) -> list[dict]:
        data = payload.get("data", {})
        if isinstance(data, list):
            return data
        for key in ("extract_result", "extract_results", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]
        return []

    def wait(self, batch_id: str, timeout_seconds: int = 3600) -> list[dict]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            response = self.session.get(
                f"{BASE_URL}/extract-results/batch/{batch_id}",
                headers=self.headers,
                timeout=30,
            )
            results = self._result_list(self._json(response))
            if results:
                states = [str(item.get("state", "")).lower() for item in results]
                progress = ", ".join(states)
                print(f"MinerU 状态：{progress}")
                failed = [item for item in results if str(item.get("state", "")).lower() == "failed"]
                if failed:
                    raise RuntimeError(f"MinerU 解析失败：{failed[0].get('err_msg', '未知错误')}")
                if len(results) == 3 and all(state == "done" for state in states):
                    return sorted(results, key=lambda item: str(item.get("data_id", item.get("file_name", ""))))
            time.sleep(5)
        raise TimeoutError("MinerU 解析超时")

    def _download_archive(self, url: str, archive: Path) -> None:
        partial = archive.with_suffix(archive.suffix + ".part")
        last_error = None
        for attempt in range(1, MAX_NETWORK_ATTEMPTS + 1):
            downloaded = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            try:
                with self.session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    resumed = downloaded > 0 and response.status_code == 206
                    mode = "ab" if resumed else "wb"
                    if not resumed:
                        downloaded = 0
                    expected_total = None
                    match = re.search(r"/(\d+)$", response.headers.get("Content-Range", ""))
                    if match:
                        expected_total = int(match.group(1))
                    elif response.headers.get("Content-Length"):
                        expected_total = downloaded + int(response.headers["Content-Length"])
                    with partial.open(mode) as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                actual_size = partial.stat().st_size
                if expected_total is not None and actual_size != expected_total:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"下载不完整：已收到 {actual_size} 字节，应为 {expected_total} 字节"
                    )
                partial.replace(archive)
                return
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                retryable = not isinstance(exc, requests.RequestException) or self._can_retry(exc)
                if attempt == MAX_NETWORK_ATTEMPTS or not retryable:
                    break
                self._pause(attempt, "下载")
        raise RuntimeError(f"MinerU 结果下载失败，已重试 {MAX_NETWORK_ATTEMPTS} 次：{last_error}") from last_error

    def download(self, results: list[dict]) -> list[Path]:
        paths = []
        for number, result in enumerate(results, 1):
            url = result.get("full_zip_url")
            if not url:
                raise RuntimeError("MinerU 结果中缺少 full_zip_url")
            part_dir = MINERU_DIR / f"part_{number}"
            part_dir.mkdir(parents=True, exist_ok=True)
            if list(part_dir.rglob("*_content_list.json")):
                print(f"复用已下载的第 {number}/{len(results)} 个解析结果。")
                paths.append(part_dir)
                continue
            archive = MINERU_DIR / f"part_{number}.zip"
            if not archive.exists() or not zipfile.is_zipfile(archive):
                print(f"下载第 {number}/{len(results)} 个解析结果...")
                self._download_archive(url, archive)
            with zipfile.ZipFile(archive) as package:
                package.extractall(part_dir)
            paths.append(part_dir)
        return paths


def get_or_create_batch(client: MinerUClient, pdf_path: Path) -> str:
    task_path = MINERU_DIR / "task.json"
    if task_path.exists():
        try:
            batch_id = json.loads(task_path.read_text(encoding="utf-8")).get("batch_id")
        except (OSError, ValueError):
            batch_id = None
        if batch_id:
            print(f"继续 MinerU 任务：{batch_id}")
            return str(batch_id)
    batch_id = client.submit(pdf_path)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"batch_id": batch_id}, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(task_path)
    return batch_id


def _item_text(item: dict) -> str:
    values = []
    for key in ("text", "table_body", "table", "content", "caption"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    captions = item.get("img_caption") or item.get("image_caption")
    if isinstance(captions, list):
        values.extend(str(value).strip() for value in captions if str(value).strip())
    return "\n".join(dict.fromkeys(values))


def merge_content_lists(part_dirs: list[Path]) -> list[dict]:
    merged = []
    for (start_page, _page_range), part_dir in zip(PAGE_RANGES, part_dirs):
        candidates = list(part_dir.rglob("*_content_list.json"))
        if not candidates:
            raise FileNotFoundError(f"{part_dir} 中没有 content_list.json")
        items = json.loads(candidates[0].read_text(encoding="utf-8"))
        raw_pages = [int(item.get("page_idx", 0)) for item in items if "page_idx" in item]
        pages_are_absolute = start_page > 1 and raw_pages and max(raw_pages) >= start_page - 1
        for item in items:
            text = _item_text(item)
            if not text:
                continue
            raw_page = int(item.get("page_idx", 0))
            page = raw_page + 1 if pages_are_absolute else start_page + raw_page
            merged.append(
                {
                    "page": page,
                    "type": str(item.get("type", "text")),
                    "text": text,
                }
            )
    return merged


def _split_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [piece for piece in pieces if piece]


def build_chunks(items: list[dict]) -> list[Chunk]:
    chunks = []
    title = ""
    for position, item in enumerate(items):
        item_type = item["type"].lower()
        text = item["text"].strip()
        if "title" in item_type:
            title = re.sub(r"\s+", " ", text)[:120]
            continue
        parts = [text] if "table" in item_type else _split_text(text)
        for part_number, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"p{item['page']}-{position}-{part_number}",
                    text=part,
                    title=title,
                    page=int(item["page"]),
                    content_type="table" if "table" in item_type else "text",
                )
            )
    return chunks


def build_index(chunks: list[Chunk]) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"本地 BGE 模型不存在：{MODEL_PATH}")
    encoder = SentenceTransformer(str(MODEL_PATH))
    texts = [f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text for chunk in chunks]
    vectors = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_faiss_index(index, INDEX_PATH)
    CHUNKS_PATH.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    import os

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF 不存在：{PDF_PATH}")
    MINERU_DIR.mkdir(parents=True, exist_ok=True)

    if MERGED_PATH.exists():
        print("复用已下载的 MinerU 解析结果。")
        items = json.loads(MERGED_PATH.read_text(encoding="utf-8"))
    else:
        client = MinerUClient(os.getenv("MINERU_API_KEY", "").strip())
        print("提交 MinerU 解析任务...")
        batch_id = get_or_create_batch(client, PDF_PATH)
        results = client.wait(batch_id)
        part_dirs = client.download(results)
        items = merge_content_lists(part_dirs)
        MERGED_PATH.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    chunks = build_chunks(items)
    if not chunks:
        raise RuntimeError("解析结果中没有可用文本")
    print(f"共生成 {len(chunks)} 个文本块，正在建立向量索引...")
    build_index(chunks)
    print(f"建库完成：{INDEX_PATH}")


if __name__ == "__main__":
    main()
