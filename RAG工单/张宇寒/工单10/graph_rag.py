from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from corpus import Chunk, REPORTS
from evaluation import rewrite_query


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
METRICS = (
    "营业收入",
    "净利润",
    "总资产",
    "不良贷款率",
    "保费收入",
    "偿付能力",
    "营业利润",
    "基本每股收益",
)
SUMMARY_TITLES = ("主要会计数据", "主要财务数据", "财务摘要", "经营业绩")
GROUP_METRIC_PATTERN = re.compile(r"(?:本集团|集团|本公司|公司|本行|全行).{0,16}(?:全年)?实现营业收入")


def _add_node(nodes: dict[str, dict], node_id: str, label: str, node_type: str, **metadata) -> None:
    nodes.setdefault(node_id, {"id": node_id, "label": label, "type": node_type, "metadata": metadata})


def build_graph(chunks: list[Chunk]) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}

    def add_edge(source: str, target: str, relation: str) -> None:
        edges.setdefault((source, target, relation), {"source": source, "target": target, "relation": relation})

    for chunk in chunks:
        company_id = f"company:{chunk.stock_code}"
        report_id = f"report:{chunk.stock_code}:{chunk.year}"
        year_id = f"year:{chunk.year}"
        chunk_id = f"chunk:{chunk.chunk_id}"
        _add_node(nodes, company_id, chunk.company, "company", stock_code=chunk.stock_code)
        _add_node(nodes, report_id, chunk.document, "report", stock_code=chunk.stock_code, year=chunk.year)
        _add_node(nodes, year_id, str(chunk.year), "year", year=chunk.year)
        _add_node(
            nodes,
            chunk_id,
            f"{chunk.company} 第{chunk.page}页",
            "chunk",
            chunk_id=chunk.chunk_id,
            page=chunk.page,
            preview=chunk.text[:180],
        )
        add_edge(company_id, report_id, "HAS_REPORT")
        add_edge(report_id, year_id, "REPORT_YEAR")
        add_edge(report_id, chunk_id, "HAS_CHUNK")
        for metric in METRICS:
            if metric in chunk.text:
                metric_id = f"metric:{metric}"
                _add_node(nodes, metric_id, metric, "metric", name=metric)
                add_edge(chunk_id, metric_id, "MENTIONS")
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def understand_query(question: str) -> dict:
    stocks = [
        report.stock_code
        for report in REPORTS.values()
        if report.company in question
    ]
    stocks = list(dict.fromkeys(stocks))
    years = list(dict.fromkeys(int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)))
    metrics = [metric for metric in METRICS if metric in question]
    return {"stocks": stocks, "years": years, "metrics": metrics}


def _related_subgraph(graph: dict, hits: list[dict]) -> dict:
    chunk_ids = {f"chunk:{hit['chunk_id']}" for hit in hits}
    included = set(chunk_ids)
    first_edges = [edge for edge in graph["edges"] if edge["source"] in chunk_ids or edge["target"] in chunk_ids]
    for edge in first_edges:
        included.update((edge["source"], edge["target"]))
    report_ids = {node_id for node_id in included if node_id.startswith("report:")}
    second_edges = [
        edge
        for edge in graph["edges"]
        if edge["relation"] in {"HAS_REPORT", "REPORT_YEAR"}
        and (edge["source"] in report_ids or edge["target"] in report_ids)
    ]
    for edge in second_edges:
        included.update((edge["source"], edge["target"]))
    edges = [edge for edge in graph["edges"] if edge["source"] in included and edge["target"] in included]
    nodes = [node for node in graph["nodes"] if node["id"] in included]
    return {"nodes": nodes, "edges": edges}


class GraphRAG:
    def __init__(self, chunks: list[Chunk], graph: dict, vectorizer, matrix):
        self.chunks = chunks
        self.graph = graph
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def fit(cls, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("不能用空语料建立索引")
        texts = [f"{item.company} {item.year} {item.document} {item.title} {item.text}" for item in chunks]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=80000, sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        return cls(chunks, build_graph(chunks), vectorizer, matrix)

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR):
        chunk_path = data_dir / "chunks.json"
        graph_path = data_dir / "graph.json"
        index_path = data_dir / "tfidf.pkl"
        if not all(path.is_file() for path in (chunk_path, graph_path, index_path)):
            raise FileNotFoundError("尚未建立Graph RAG索引，请先运行prepare.py")
        chunks = [Chunk(**item) for item in json.loads(chunk_path.read_text(encoding="utf-8"))]
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        with index_path.open("rb") as stream:
            vectorizer, matrix = pickle.load(stream)
        return cls(chunks, graph, vectorizer, matrix)

    def search_baseline(self, question: str, top_k: int = 5) -> dict:
        question = re.sub(r"\s+", " ", question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if not 1 <= int(top_k) <= 10:
            raise ValueError("Top K必须在1到10之间")
        started = time.perf_counter()
        lexical = np.asarray((self.matrix @ self.vectorizer.transform([question]).T).toarray()).ravel()
        selected = np.argsort(-lexical)[:top_k].tolist()
        hits = [
            {
                **asdict(self.chunks[index]),
                "rank": rank,
                "score": round(float(lexical[index]), 6),
                "lexical_score": round(float(lexical[index]), 6),
                "matched_graph_nodes": [],
            }
            for rank, index in enumerate(selected, 1)
        ]
        return {"hits": hits, "retrieval_ms": round((time.perf_counter() - started) * 1000, 2)}

    def search_optimized(self, question: str, top_k: int = 5, preferred_query: str = "") -> dict:
        question = re.sub(r"\s+", " ", question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if not 1 <= int(top_k) <= 10:
            raise ValueError("Top K必须在1到10之间")
        started = time.perf_counter()
        rewritten, optimization = rewrite_query(question, preferred_query)
        intent = understand_query(rewritten)
        lexical = np.asarray((self.matrix @ self.vectorizer.transform([rewritten]).T).toarray()).ravel()
        scores = lexical.astype("float64")
        matched_nodes: list[list[str]] = [[] for _ in self.chunks]
        for index, chunk in enumerate(self.chunks):
            searchable = f"{chunk.title} {chunk.text}"
            if len(chunk.text.strip()) < 100:
                scores[index] -= 0.45
            if intent["stocks"] and chunk.stock_code in intent["stocks"]:
                scores[index] += 0.70
                matched_nodes[index].append(f"company:{chunk.stock_code}")
            if intent["years"] and chunk.year in intent["years"]:
                scores[index] += 0.18
                matched_nodes[index].append(f"year:{chunk.year}")
            for metric in intent["metrics"]:
                if metric in chunk.text:
                    scores[index] += 0.14
                    matched_nodes[index].append(f"metric:{metric}")
            if intent["metrics"] and any(term in f"{chunk.title} {chunk.text[:120]}" for term in SUMMARY_TITLES):
                scores[index] += 0.16
            if "营业收入" in intent["metrics"] and GROUP_METRIC_PATTERN.search(chunk.text):
                scores[index] += 0.30
            relation_matches = sum(1 for term in optimization["relations"] if term in searchable)
            scores[index] += min(relation_matches, 3) * 0.22

        def ranked(code: str | None = None) -> list[int]:
            candidates = [i for i, chunk in enumerate(self.chunks) if code is None or chunk.stock_code == code]
            return sorted(candidates, key=lambda i: (scores[i], lexical[i]), reverse=True)

        if intent["stocks"]:
            groups = {code: ranked(code) for code in intent["stocks"]}
            selected: list[int] = []
            depth = 0
            while len(selected) < top_k and any(depth < len(group) for group in groups.values()):
                for code in intent["stocks"]:
                    if depth < len(groups[code]) and len(selected) < top_k:
                        selected.append(groups[code][depth])
                depth += 1
        else:
            selected = ranked()[:top_k]
        hits = [
            {
                **asdict(self.chunks[index]),
                "rank": rank,
                "score": round(float(scores[index]), 6),
                "lexical_score": round(float(lexical[index]), 6),
                "matched_graph_nodes": matched_nodes[index],
            }
            for rank, index in enumerate(selected, 1)
        ]
        return {
            "hits": hits,
            "graph": _related_subgraph(self.graph, hits),
            "intent": intent,
            "query_optimization": optimization,
            "retrieval_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def search(self, question: str, top_k: int = 5) -> dict:
        return self.search_optimized(question, top_k)


def save_index(chunks: list[Chunk], data_dir: Path = DATA_DIR) -> GraphRAG:
    data_dir.mkdir(parents=True, exist_ok=True)
    rag = GraphRAG.fit(chunks)
    (data_dir / "chunks.json").write_text(
        json.dumps([asdict(item) for item in chunks], ensure_ascii=False), encoding="utf-8"
    )
    (data_dir / "graph.json").write_text(json.dumps(rag.graph, ensure_ascii=False), encoding="utf-8")
    with (data_dir / "tfidf.pkl").open("wb") as stream:
        pickle.dump((rag.vectorizer, rag.matrix), stream)
    return rag

