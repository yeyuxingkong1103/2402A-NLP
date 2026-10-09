import statistics
from pathlib import Path
import pymupdf as fitz

def _doc_chars(path):
    try:
        if path.suffix.lower() == ".pdf":
            doc = fitz.open(path)
            n = sum(len((p.get_text("text") or "").strip()) for p in doc)
            doc.close()
            return n
        if path.suffix.lower() in {".md", ".txt", ".html"}:
            return len(path.read_text(encoding="utf-8", errors="ignore"))
        if path.suffix.lower() == ".docx":
            from docx import Document
            return sum(len(p.text) for p in Document(path).paragraphs)
    except Exception:
        return 0
    return 0

def length_distribution(files, cfg):
    lengths = [_doc_chars(Path(f)) for f in files]
    lengths = [x for x in lengths if x >= 0]
    if not lengths:
        return {"count": 0}
    qs = cfg["length"]["quantiles"]
    quantiles = {}
    if len(lengths) > 1:
        dec = statistics.quantiles(lengths, n=100)
        for q in qs:
            quantiles["P%d" % int(q*100)] = int(dec[int(q*100)-1])
    else:
        for q in qs:
            quantiles["P%d" % int(q*100)] = lengths[0]
    bins = cfg["length"]["bins"]
    bin_dist = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        c = sum(1 for x in lengths if lo <= x < hi)
        bin_dist.append({"range": "%s-%s" % (lo, hi), "count": c})
    return {
        "count": len(lengths), "min": min(lengths), "max": max(lengths),
        "mean": round(statistics.mean(lengths), 2),
        "quantiles": quantiles, "bins": bin_dist,
    }
