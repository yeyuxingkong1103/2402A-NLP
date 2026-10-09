from collections import Counter
from utils.file_utils import iter_files

def format_distribution(folder_path):
    counter = Counter()
    total = 0
    for p in iter_files(folder_path):
        counter[p.suffix.lower()] += 1
        total += 1
    dist = {ext: {"count": c, "ratio": round(c / total, 4) if total else 0}
            for ext, c in counter.most_common()}
    return {"total": total, "distribution": dist}
