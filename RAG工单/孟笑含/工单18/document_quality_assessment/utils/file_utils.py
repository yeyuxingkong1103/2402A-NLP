import os, hashlib
from pathlib import Path

SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".txt", ".html", ".pptx", ".xlsx"}

def iter_files(folder_path, exts=None):
    exts = exts or SUPPORTED_EXTS
    for root, _, files in os.walk(folder_path):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                yield p

def file_md5(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
