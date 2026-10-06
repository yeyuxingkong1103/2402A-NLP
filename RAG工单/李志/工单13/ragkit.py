from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def chunks(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    return [clean[i:i + size] for i in range(0, len(clean), max(1, size - overlap))]


def read_pdf(path: Path) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要安装 pypdf：python -m pip install pypdf") from exc
    reader = PdfReader(str(path))
    return [{"source": path.name, "page": i + 1, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages)]


def load_documents(paths: list[str]) -> list[dict]:
    docs = []
    for raw in paths:
        path = Path(raw)
        files = list(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                continue
            try:
                if file.suffix.lower() == ".pdf":
                    pages = read_pdf(file)
                elif file.suffix.lower() in {".txt", ".md", ".json", ".jsonl", ".csv"}:
                    pages = [{"source": file.name, "page": 1,
                              "text": file.read_text(encoding="utf-8", errors="ignore")}]
                else:
                    continue
            except Exception as exc:
                print(f"跳过 {file}: {exc}")
                continue
            for page in pages:
                for index, text in enumerate(chunks(page["text"])):
                    docs.append({**page, "chunk": index, "text": text})
    return docs


def demo_documents() -> list[dict]:
    samples = [
        "武汉力源信息技术股份有限公司主营业务包括电子元器件代理分销、芯片解决方案及技术服务。",
        "公司组织结构中销售部包括大客户销售部、区域销售部和销售支持部。",
        "本次发行募集资金主要用于研发中心升级项目、营销网络建设项目和补充流动资金。",
        "公司法定代表人为赵佳生，注册资本及股本信息以招股说明书披露为准。",
        "风险因素包括供应链波动、市场竞争、汇率变化以及技术迭代风险。",
    ]
    return [{"source": "demo.txt", "page": 1, "chunk": i, "text": text}
            for i, text in enumerate(samples)]


def bm25(query: str, docs: list[dict]) -> list[float]:
    terms = tokens(query)
    tokenized = [tokens(doc["text"]) for doc in docs]
    average = sum(map(len, tokenized)) / max(1, len(tokenized))
    frequencies = Counter(term for doc in tokenized for term in set(doc))
    scores = []
    for doc in tokenized:
        counts = Counter(doc)
        score = 0.0
        for term in terms:
            tf = counts[term]
            inverse = math.log(1 + (len(docs) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            score += inverse * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(doc) / max(1, average)))
        scores.append(score)
    return scores


def vector(text: str, dimensions: int = 256, weights: dict[str, float] | None = None) -> list[float]:
    values = [0.0] * dimensions
    for term in tokens(text):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest, "big") % dimensions
        values[slot] += (weights or {}).get(term, 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def search(query: str, docs: list[dict], top_k: int = 3, mode: str = "hybrid",
           weights: dict[str, float] | None = None) -> list[dict]:
    if not docs:
        return []
    keyword = bm25(query, docs)
    query_vector = vector(query, weights=weights)
    semantic = [cosine(query_vector, vector(doc["text"], weights=weights)) for doc in docs]
    normalized = lambda values: [(value - min(values)) / ((max(values) - min(values)) or 1) for value in values]
    keyword, semantic = normalized(keyword), normalized(semantic)
    if mode == "bm25":
        scores = keyword
    elif mode == "vector":
        scores = semantic
    else:
        scores = [0.55 * a + 0.45 * b for a, b in zip(keyword, semantic)]
    ranked = sorted(range(len(docs)), key=lambda index: scores[index], reverse=True)[:top_k]
    return [{**docs[index], "score": round(scores[index], 4)} for index in ranked]


def answer(query: str, results: list[dict]) -> str:
    if not results or results[0]["score"] <= 0:
        return "知识库中未找到足够相关的信息。"
    sentences = []
    for result in results:
        pieces = re.split(r"(?<=[。！？.!?])", result["text"])
        sentences.extend(piece.strip() for piece in pieces if piece.strip())
    ranked = search(query, [{"text": text, "source": "context", "page": 0, "chunk": 0}
                            for text in sentences], 3)
    return "".join(item["text"] for item in ranked)


def graph_from_documents(docs: list[dict]) -> dict:
    nodes, edges = Counter(), Counter()
    pattern = re.compile(r"[\u4e00-\u9fff]{2,10}|[A-Za-z][A-Za-z0-9_+-]{2,}")
    stop = {"公司", "股份", "有限", "以及", "包括", "进行", "项目", "主要"}
    for doc in docs:
        entities = [item for item in pattern.findall(doc["text"]) if item not in stop][:20]
        for entity in set(entities):
            nodes[entity] += 1
        for left, right in zip(entities, entities[1:]):
            if left != right:
                edges[tuple(sorted((left, right)))] += 1
    return {"nodes": [{"id": key, "weight": value} for key, value in nodes.most_common(120)],
            "edges": [{"source": key[0], "target": key[1], "weight": value}
                      for key, value in edges.most_common(240)]}


def graph_expand(query: str, docs: list[dict], graph: dict, hops: int = 2) -> list[dict]:
    query_terms = set(tokens(query))
    adjacency = defaultdict(set)
    for edge in graph["edges"]:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])
    seeds = [node["id"] for node in graph["nodes"] if any(term in node["id"].lower() for term in query_terms)]
    queue = deque((seed, 0) for seed in seeds[:8])
    related = set(seeds)
    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for neighbor in adjacency[node]:
            if neighbor not in related:
                related.add(neighbor); queue.append((neighbor, depth + 1))
    expanded = query + " " + " ".join(sorted(related)[:30])
    return search(expanded, docs, 5)


def save(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def timed(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, (time.perf_counter() - started) * 1000
