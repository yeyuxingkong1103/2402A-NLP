import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path


MAX_PDF_BYTES = 200 * 1024 * 1024
MAX_PDF_PAGES = 600
MINERU_API_BASE = "https://mineru.net/api/v4"
MINERU_POLL_INTERVAL_SECONDS = 5
MINERU_MAX_POLL_ATTEMPTS = 120


class ParsePdfError(RuntimeError):
    pass


def count_pdf_pages(pdf_path):
    errors = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except ImportError as exc:
        errors.append(f"pypdf: {exc}")
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except ImportError as exc:
        errors.append(f"PyPDF2: {exc}")
    except Exception as exc:
        errors.append(f"PyPDF2: {exc}")

    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        page_count = len(document)
        document.close()
        return page_count
    except ImportError as exc:
        errors.append(f"pypdfium2: {exc}")
    except Exception as exc:
        errors.append(f"pypdfium2: {exc}")

    raise ParsePdfError("无法统计 PDF 页数，请安装 pypdf、PyPDF2 或 pypdfium2；详情：" + " | ".join(errors))


def validate_pdf(pdf_path):
    if not pdf_path.exists():
        raise ParsePdfError(f"PDF 文件不存在：{pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ParsePdfError(f"输入文件不是 PDF：{pdf_path}")

    file_size = pdf_path.stat().st_size
    if file_size > MAX_PDF_BYTES:
        raise ParsePdfError(f"PDF 超过 200MB：{file_size / 1024 / 1024:.2f}MB")

    page_count = count_pdf_pages(pdf_path)
    if page_count > MAX_PDF_PAGES:
        raise ParsePdfError(f"PDF 超过 600 页：{page_count} 页")

    return file_size, page_count


def api_json_request(url, token, payload=None, method="POST"):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ParsePdfError(f"MinerU API 请求失败：HTTP {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise ParsePdfError(f"MinerU API 请求失败：{exc}") from exc


def download_api_file(url, token):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ParsePdfError(f"MinerU 文件下载失败：HTTP {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise ParsePdfError(f"MinerU 文件下载失败：{exc}") from exc


def unwrap_api_data(response):
    if not isinstance(response, dict):
        raise ParsePdfError(f"MinerU API 返回格式异常：{response}")
    if "code" in response and response.get("code") not in (0, "0", None):
        raise ParsePdfError(f"MinerU API 返回错误：{response}")
    data = response.get("data", response)
    if not isinstance(data, dict):
        raise ParsePdfError(f"MinerU API data 格式异常：{response}")
    return data


def extract_batch_id(response):
    data = unwrap_api_data(response)
    batch_id = data.get("batch_id")
    if not batch_id:
        raise ParsePdfError(f"MinerU API 未返回 batch_id：{response}")
    return batch_id


def upload_pdf_to_presigned_url(upload_url, pdf_path):
    parsed = urllib.parse.urlsplit(upload_url)
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_cls(parsed.netloc, timeout=300)
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    try:
        connection.request("PUT", path, body=pdf_path.read_bytes(), headers={})
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise ParsePdfError(f"上传 PDF 到 MinerU 失败：HTTP {response.status} {body.decode('utf-8', errors='replace')}")
    except OSError as exc:
        raise ParsePdfError(f"上传 PDF 到 MinerU 失败：{exc}") from exc
    finally:
        connection.close()


def submit_pdf_batch(pdf_path, token):
    response = api_json_request(
        f"{MINERU_API_BASE}/file-urls/batch",
        token,
        {
            "enable_formula": True,
            "enable_table": True,
            "language": "ch",
            "files": [{"name": pdf_path.name, "is_ocr": True}],
        },
    )
    data = unwrap_api_data(response)
    upload_urls = data.get("file_urls") or data.get("urls") or []
    if upload_urls:
        upload_pdf_to_presigned_url(upload_urls[0], pdf_path)
    return extract_batch_id(response)


def poll_extract_result(batch_id, token):
    url = f"{MINERU_API_BASE}/extract-results/batch/{batch_id}"
    for _ in range(MINERU_MAX_POLL_ATTEMPTS):
        response = api_json_request(url, token, method="GET")
        result = select_extract_result(response)
        status = str(result.get("status") or result.get("state") or result.get("extract_state") or "").lower()
        if status in {"done", "success", "completed"} or has_downloadable_result(result):
            return result
        if status in {"failed", "fail", "error"}:
            raise ParsePdfError(f"MinerU 解析失败：{result}")
        time.sleep(MINERU_POLL_INTERVAL_SECONDS)
    raise ParsePdfError(f"MinerU 解析超时：batch_id={batch_id}")


def select_extract_result(response):
    data = unwrap_api_data(response)
    results = data.get("extract_result") or data.get("extract_results") or data.get("results")
    if isinstance(results, list) and results:
        return results[0]
    return data


def has_downloadable_result(result):
    return any(result.get(key) for key in ("markdown_url", "md_url", "content_list_url", "full_zip_url", "zip_url"))


def write_api_outputs(result, pdf_path, output_dir):
    output_stem = pdf_path.stem.strip()
    markdown_path = output_dir / f"{output_stem}.md"
    content_list_path = output_dir / "content_list.json"
    zip_url = result.get("full_zip_url") or result.get("zip_url")
    if zip_url:
        write_outputs_from_zip(download_api_file(zip_url, os.environ["MINERU_TOKEN"]), markdown_path, content_list_path)
    else:
        markdown_url = result.get("markdown_url") or result.get("md_url")
        content_list_url = result.get("content_list_url")
        if not markdown_url or not content_list_url:
            raise ParsePdfError(f"MinerU 结果缺少 markdown/content_list 下载地址：{result}")
        markdown_path.write_bytes(download_api_file(markdown_url, os.environ["MINERU_TOKEN"]))
        content_list_path.write_bytes(download_api_file(content_list_url, os.environ["MINERU_TOKEN"]))
    return markdown_path, content_list_path


def write_outputs_from_zip(zip_bytes, markdown_path, content_list_path):
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        markdown_name = next((name for name in archive.namelist() if name.endswith(".md")), None)
        content_name = next((name for name in archive.namelist() if name.endswith("content_list.json")), None)
        if not markdown_name or not content_name:
            raise ParsePdfError("MinerU ZIP 中缺少 markdown 或 content_list.json")
        markdown_path.write_bytes(archive.read(markdown_name))
        content_list_path.write_bytes(archive.read(content_name))


def parse_pdf(pdf_path, output_dir):
    token = os.environ.get("MINERU_TOKEN")
    if not token:
        raise ParsePdfError("缺少环境变量 MINERU_TOKEN")

    validate_pdf(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_id = submit_pdf_batch(pdf_path, token)
    result = poll_extract_result(batch_id, token)
    markdown_path, content_list_path = write_api_outputs(result, pdf_path, output_dir)

    if not markdown_path.exists():
        raise ParsePdfError(f"MinerU 未生成 markdown：{markdown_path}")
    if not content_list_path.exists():
        raise ParsePdfError(f"MinerU 未生成 content_list.json：{content_list_path}")

    return markdown_path, content_list_path


def main():
    parser = argparse.ArgumentParser(description="Parse a PDF into markdown and content_list.json with MinerU precise OCR mode.")
    parser.add_argument("pdf_path", help="PDF 文件路径")
    parser.add_argument("output_dir", help="解析结果输出目录")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    output_dir = Path(args.output_dir)
    start = time.perf_counter()

    try:
        file_size, page_count = validate_pdf(pdf_path)
        markdown_path, content_list_path = parse_pdf(pdf_path, output_dir)
    except ParsePdfError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "pdf": pdf_path.as_posix(),
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "page_count": page_count,
                "markdown": markdown_path.as_posix(),
                "content_list": content_list_path.as_posix(),
                "elapsed_seconds": round(elapsed, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
