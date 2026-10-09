import re
from pathlib import Path
import pymupdf as fitz

PATTERNS = {
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "id_card": re.compile(r"(?<!\d)[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
}

def _read_text(path):
    try:
        if path.suffix.lower() == ".pdf":
            doc = fitz.open(path)
            t = "\n".join((p.get_text("text") or "") for p in doc)
            doc.close()
            return t
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def scan_sensitive(files, cfg):
    sc = cfg["sensitive"]
    ctx = sc["context_chars"]
    enabled = {k: sc.get("enable_" + k, False) for k in PATTERNS}
    results = []
    for f in files:
        text = _read_text(Path(f))
        if not text:
            continue
        for name, pat in PATTERNS.items():
            if not enabled.get(name):
                continue
            for m in pat.finditer(text):
                s, e = m.start(), m.end()
                results.append({
                    "file": str(f), "type": name, "match": m.group(),
                    "context": text[max(0, s-ctx): e+ctx],
                    "position": s, "status": "待审核",
                })
    return {"total": len(results), "items": results}
