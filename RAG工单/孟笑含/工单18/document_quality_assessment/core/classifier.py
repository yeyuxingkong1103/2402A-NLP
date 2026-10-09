from pathlib import Path

def classify_document(path, pdf_info, cfg):
    suffix = path.suffix.lower()
    if pdf_info.get("type") == "error":
        return "Parse_Failed"
    if pdf_info.get("type") == "scan":
        return "Scan_PDF"
    if suffix in {".md", ".txt", ".html"}:
        return "Clean_Markdown"
    if suffix == ".pdf":
        if pdf_info.get("scan_ratio", 0) > 0.3:
            return "Scan_PDF"
        return "Clean_Markdown"
    return "Clean_Markdown"
