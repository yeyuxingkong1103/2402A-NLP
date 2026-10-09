from pathlib import Path
import pymupdf as fitz

def classify_pdf(path, cfg):
    char_th = cfg["scan_pdf"]["char_threshold_per_page"]
    ratio_th = cfg["scan_pdf"]["scan_page_ratio_threshold"]
    try:
        doc = fitz.open(path)
    except Exception as e:
        return {"type": "error", "error": str(e)}
    pages = doc.page_count
    scan_pages = 0
    total_chars = 0
    for page in doc:
        txt = page.get_text("text") or ""
        n = len(txt.strip())
        total_chars += n
        if n < char_th:
            scan_pages += 1
    doc.close()
    if pages == 0:
        return {"type": "error", "error": "empty pdf"}
    scan_ratio = scan_pages / pages
    if scan_ratio >= ratio_th:
        t = "scan"
    elif scan_ratio <= 1 - ratio_th:
        t = "text"
    else:
        t = "mixed"
    return {
        "type": t, "pages": pages, "scan_pages": scan_pages,
        "scan_ratio": round(scan_ratio, 4),
        "avg_chars": round(total_chars / pages, 2),
        "need_confirm": t == "mixed",
    }
