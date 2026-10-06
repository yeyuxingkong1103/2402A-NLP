# scripts/dataset_io.py
"""数据集清洗的通用 IO 工具：读写 JSONL、兼容 JSON/JSONL、生成稳定 ID。"""
import hashlib
import json
import os


def did(*parts) -> str:
    """由内容派生的稳定 ID —— 重复运行脚本得到同样的 ID。"""
    return hashlib.md5("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("  -> %-42s %6d 条  (%5d KB)" % (os.path.basename(path), len(records),
                                           os.path.getsize(path) // 1024))


def load_json(path: str):
    """兼容标准 JSON 与 JSONL 两种格式。"""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        if "Extra data" not in str(e):
            raise
        return [json.loads(l) for l in content.split("\n") if l.strip()]
