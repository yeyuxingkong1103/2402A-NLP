from collections import defaultdict
from pathlib import Path
import pymupdf as fitz
from utils.file_utils import file_md5

def md5_duplicates(files):
    groups = defaultdict(list)
    for f in files:
        try:
            groups[file_md5(Path(f))].append(str(f))
        except Exception:
            continue
    dup = {h: paths for h, paths in groups.items() if len(paths) > 1}
    return {
        "duplicate_groups": len(dup),
        "groups": [{"md5": h, "files": ps} for h, ps in dup.items()],
    }

def _read_text(path):
    p = Path(path)
    try:
        if p.suffix.lower() == ".pdf":
            doc = fitz.open(p)
            t = "\n".join((pg.get_text("text") or "") for pg in doc)
            doc.close()
            return t
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def simhash_similar(files, cfg):
    try:
        from simhash import Simhash
    except ImportError:
        return {"enabled": False, "reason": "simhash not installed"}
    shingle = cfg["duplicate"]["simhash_shingle_size"]
    th = cfg["duplicate"]["simhash_threshold"]
    def _shingles(text):
        words = text.split()
        return [" ".join(words[i:i+shingle]) for i in range(max(1, len(words)-shingle+1))]
    items = []
    for f in files:
        t = _read_text(f)
        if t.strip():
            items.append((str(f), Simhash(_shingles(t))))
    pairs = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            d = items[i][1].distance(items[j][1])
            if d <= th:
                pairs.append({"file_a": items[i][0], "file_b": items[j][0],
                              "hamming_distance": d, "status": "待确认"})
    return {"enabled": True, "pairs": pairs, "count": len(pairs)}
