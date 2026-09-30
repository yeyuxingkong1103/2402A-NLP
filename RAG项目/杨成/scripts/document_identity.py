import hashlib
import re


DOCUMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")


def normalize_document_id(document_id):
    value = str(document_id or "").strip()
    if not DOCUMENT_ID_RE.fullmatch(value):
        raise ValueError("document_id must be 3-128 characters of letters, digits, '_', '-', or '.'")
    return value


def make_primary_key(document_id, child_id):
    document_id = normalize_document_id(document_id)
    child_id = str(child_id or "").strip()
    if not child_id:
        raise ValueError("child_id must not be empty")
    return hashlib.sha256(f"{document_id}:{child_id}".encode("utf-8")).hexdigest()[:32]
