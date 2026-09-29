"""批量加载外部数据集到各角色 Milvus collection（配置驱动 + 通用解析）。

复用 ingest.py 的清洗/向量化/入库，换数据源只改 DOMAINS 配置。支持 CSV/JSON/JSONL/Parquet/txt。

用法（仓库根目录）：python scripts/load_data.py --domain 高血压医生 / --all
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import logging
import random
import time

from dotenv import load_dotenv

import ingest
import milvus_store

logger = logging.getLogger(__name__)

# 列名别名：配置里写规范化列名，这里做防御性探测，兼容常见变体。
COLUMN_ALIASES = {
    "ask": ["ask", "question", "提问", "问题"],
    "question": ["question", "ask", "提问", "问题"],
    "answer": ["answer", "reply", "回答", "答复", "回复"],
    "department": ["department", "dept", "科室", "部门"],
    "title": ["title", "标题"],
    "content": ["content", "text", "正文", "内容"],
    "text": ["text", "content", "对话", "内容", "正文"],
}

# 领域 → 加载配置。type: csv/json/txt；content_cols: (列名, 标签) 按序拼接；
# filter 可选 (筛选列, 取值集合)；encoding 默认 utf-8；collection 沿用 <英文>_guide 约定。
DOMAINS = {
    "高血压医生": {
        "type": "csv", "dir": "medical_dialogue",
        "content_cols": [("ask", "问"), ("answer", "答")],
        "filter": ("department", {"心血管科", "心血管内科", "心内科", "心外科", "心血管"}),
        "encoding": "gb18030", "source": "medical_dialogue.csv",
        "collection": "hypertension_guide", "limit": 1900,
    },
    "心理医生": {"type": "json", "dir": "psychology", "content_cols": [("text", "")],
                 "source": "smilechat", "collection": "psychology_guide", "limit": 1000},
    "律师": {"type": "txt", "dir": "law", "source": "chinese_laws.txt",
             "collection": "law_guide", "limit": 1000},
    "金融理财师": {"type": "csv", "dir": "finance", "content_cols": [("title", ""), ("content", "")],
                   "source": "cntrade.csv", "collection": "finance_guide", "limit": 500},
    "英语教师": {"type": "csv", "dir": "english", "content_cols": [("question", "问"), ("answer", "答")],
                 "source": "pelic.csv", "collection": "english_guide", "limit": 500},
}


def _find_col(norm_to_orig: dict, col: str) -> str:
    """按别名在归一化表头中找列，找不到则报错并列出实际列名。"""
    for cand in COLUMN_ALIASES.get(col, [col]):
        if cand in norm_to_orig:
            return norm_to_orig[cand]
    raise KeyError(f"缺少列 {col!r}，实际列名：{sorted(norm_to_orig.values())}")


def _resolve_columns(avail_keys, content_cols, filter):
    """把配置列名解析为实际列名。avail_keys 为实际表头/字段名。"""
    norm_to_orig = {str(k).strip().lower(): k for k in avail_keys}
    resolved = {col: _find_col(norm_to_orig, col) for col, _ in content_cols}
    if filter is not None:
        resolved[filter[0]] = _find_col(norm_to_orig, filter[0])
    return resolved


def _row_to_content(row: dict, resolved: dict, content_cols, filter) -> str | None:
    """把一行/一条记录拼成一段文本；filter 不命中或全空则返回 None。"""
    if filter is not None:
        fcol, fvalues = filter
        cell = str(row.get(resolved[fcol]) or "").strip()
        if not any(v in cell for v in fvalues):
            return None
    parts = []
    for col, label in content_cols:
        val = str(row.get(resolved[col]) or "").strip()
        if val:
            parts.append(f"{label}：{val}" if label else val)
    return "\n".join(parts) or None


def _sample_to_loaded(contents: list, source: str, domain: str, limit: int) -> list[dict]:
    """随机采样 limit 条（避免取前 N 条全是同一种病），包装成统一 loaded 记录。"""
    sampled = random.sample(contents, min(limit, len(contents)))
    return [
        {"content": c, "source": source, "domain": domain, "page": i}
        for i, c in enumerate(sampled)
    ]


def load_csv(dir_path, content_cols, source, domain, limit, filter=None, encoding="utf-8") -> list[dict]:
    """读 dir_path/*.csv，防御性探测列名 + 可选筛选，随机采样 limit 条。"""
    files = sorted(Path(dir_path).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"请先下载数据到 {dir_path}")
    fieldnames = None
    for f in files:
        with f.open(encoding=encoding, newline="") as fh:
            r = csv.DictReader(fh)
            if r.fieldnames:
                fieldnames = r.fieldnames
                break
    if fieldnames is None:
        return []
    resolved = _resolve_columns(fieldnames, content_cols, filter)
    contents = []
    for f in files:
        with f.open(encoding=encoding, newline="") as fh:
            for row in csv.DictReader(fh):
                content = _row_to_content(row, resolved, content_cols, filter)
                if content:
                    contents.append(content)
    return _sample_to_loaded(contents, source, domain, limit)


def _read_json_file(path: Path, rows: list) -> None:
    """读 JSON（数组/含数组字段的对象/JSONL）或 Parquet，追加 dict 到 rows。"""
    if path.suffix == ".parquet":
        import pandas as pd

        rows.extend(r for r in pd.read_parquet(path).to_dict("records") if isinstance(r, dict))
        return
    with path.open(encoding="utf-8") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        rows.extend(r for r in data if isinstance(r, dict))
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                rows.extend(r for r in v if isinstance(r, dict))
                break
    else:  # JSONL：每行一个对象
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue


def load_json(dir_path, content_cols, source, domain, limit, filter=None) -> list[dict]:
    """读 dir_path/*.json|*.parquet，防御性探测字段名，随机采样 limit 条。"""
    files = (
        sorted(Path(dir_path).glob("*.json"))
        + sorted(Path(dir_path).glob("*.jsonl"))
        + sorted(Path(dir_path).glob("*.parquet"))
    )
    if not files:
        raise FileNotFoundError(f"请先下载数据到 {dir_path}")
    rows: list[dict] = []
    for f in files:
        _read_json_file(f, rows)
    if not rows:
        return []
    avail_keys = list({k for r in rows for k in r})
    resolved = _resolve_columns(avail_keys, content_cols, filter)
    contents = [
        c for c in (_row_to_content(r, resolved, content_cols, filter) for r in rows) if c
    ]
    return _sample_to_loaded(contents, source, domain, limit)


def load_txt(dir_path, source, domain, limit, encoding="utf-8") -> list[dict]:
    """读 dir_path/*.txt，每行一条，随机采样 limit 条。"""
    files = sorted(Path(dir_path).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"请先下载数据到 {dir_path}")
    contents = []
    for f in files:
        with f.open(encoding=encoding) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    contents.append(line)
    return _sample_to_loaded(contents, source, domain, limit)


def _load_records(records: list[dict], source: str, domain: str, collection: str) -> int:
    """清洗 -> 向量化 -> 覆盖式入库（复用 ingest/milvus_store），返回入库 chunk 数。"""
    if not milvus_store.USE_MILVUS:
        logger.error("load_data 仅支持 Milvus（USE_MILVUS=true），当前 USE_MILVUS=false，跳过")
        return 0
    all_chunks = []
    for r in records:
        all_chunks.extend(
            ingest.split_chunks(ingest.normalize_whitespace(r["content"]), r["page"])
        )
    all_chunks, _ = ingest.clean_chunks(all_chunks)
    if not all_chunks:
        logger.warning("清洗后无有效 chunk，跳过入库（source=%s）", source)
        return 0

    embedder = ingest.load_embedder()
    documents = [c["content"] for c in all_chunks]
    logger.info("向量化 %d 个 chunk ...", len(documents))
    embeddings = embedder.encode(
        documents, batch_size=ingest.BATCH_SIZE, normalize_embeddings=True
    ).tolist()

    now = int(time.time())
    milvus_store.create_collection(collection)
    milvus_store.delete_by_source(collection, source)
    records_out = [
        {
            "id": c["id"], "embedding": emb, "content": c["content"], "page": c["page"],
            "source": source, "domain": domain,
            "created_at": now, "updated_at": now,
            "summary": ingest.make_summary(c["content"]),
            "parent_content": c.get("parent_content", ""),
        }
        for c, emb in zip(all_chunks, embeddings)
    ]
    milvus_store.insert(collection, records_out)
    return len(records_out)


def load_domain(domain: str, cfg: dict, data_dir: Path) -> int:
    """按配置派发到通用解析器，再入库，返回入库 chunk 数。"""
    dir_path = data_dir / cfg["dir"]
    typ = cfg["type"]
    content_cols = cfg.get("content_cols", [])
    filter = cfg.get("filter")
    encoding = cfg.get("encoding", "utf-8")
    source = cfg["source"]
    collection = cfg["collection"]
    limit = cfg["limit"]

    if typ == "csv":
        records = load_csv(dir_path, content_cols, source, domain, limit, filter, encoding)
    elif typ == "json":
        records = load_json(dir_path, content_cols, source, domain, limit, filter)
    elif typ == "txt":
        records = load_txt(dir_path, source, domain, limit, encoding)
    else:
        raise ValueError(f"未知 type：{typ}")

    if len(records) < limit:
        logger.warning("域「%s」实际记录 %d 条，少于 limit %d", domain, len(records), limit)
    count = _load_records(records, source, domain, collection)
    logger.info(
        "域「%s」完成：collection=%s source=%s 入库 %d chunk",
        domain, collection, source, count,
    )
    return count


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    random.seed(42)  # 采样可复现
    parser = argparse.ArgumentParser(description="批量加载外部数据集到各角色 collection")
    parser.add_argument("--domain", help="单个角色名（如 高血压医生）")
    parser.add_argument("--all", action="store_true", help="加载全部 5 个域")
    parser.add_argument("--limit", type=int, default=None, help="覆盖该域默认 limit")
    parser.add_argument("--data-dir", default=str(ROOT / "data/external"), help="外部数据根目录")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.all:
        for domain, cfg in DOMAINS.items():
            load_domain(domain, cfg, data_dir)
    elif args.domain:
        if args.domain not in DOMAINS:
            raise SystemExit(f"未知 domain：{args.domain}，可选：{list(DOMAINS)}")
        cfg = dict(DOMAINS[args.domain])
        if args.limit is not None:
            cfg["limit"] = args.limit
        load_domain(args.domain, cfg, data_dir)
    else:
        parser.error("必须指定 --domain 或 --all")


if __name__ == "__main__":
    main()
