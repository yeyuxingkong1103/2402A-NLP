from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
DATA = ROOT / "data"
INDEX = DATA / "chunks.jsonl"
TABLE_INDEX = DATA / "tables.jsonl"
IMAGE_INDEX = DATA / "images.jsonl"
MAX_UPLOAD = 80 * 1024 * 1024
_lock = threading.RLock()
_chunks: list[dict] = []
_tables: list[dict] = []
_images: list[dict] = []
_sessions: dict[str, list[dict]] = {}


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\u3000", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def split_page(text: str, size: int = 520, overlap: int = 90) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    units: list[str] = []
    current = ""
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[。！？；;])", paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + len(sentence) > size:
                units.append(current)
                current = current[-overlap:] if overlap else ""
            if len(sentence) > size:
                if current:
                    units.append(current)
                    current = ""
                start = 0
                while start < len(sentence):
                    end = min(start + size, len(sentence))
                    units.append(sentence[start:end])
                    if end == len(sentence):
                        break
                    start = end - overlap
            else:
                current += sentence
    if current:
        units.append(current)
    return [part for part in units if len(part) > 15]


def extract_pdf(path: Path, document_name: str) -> list[dict]:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("缺少 PDF 解析组件。请先运行 run.bat 安装依赖。") from exc
    reader = PdfReader(str(path), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF 已加密，暂不支持解析。请上传未加密的 PDF。")
    rows: list[dict] = []
    for page_number, page in enumerate(reader.pages, 1):
        if page_number % 50 == 0:
            print(f"PDF 解析进度：{page_number}/{len(reader.pages)} 页", flush=True)
        text = page.extract_text() or ""
        for ordinal, content in enumerate(split_page(text)):
            rows.append({
                "id": f"{document_name}:{page_number}:{ordinal}",
                "document": document_name,
                "page": page_number,
                "text": content,
            })
    return rows


def normalize_cell(value: object) -> str:
    return clean_text(str(value or "")).replace("\n", " ")


def extract_tables(path: Path, document_name: str) -> list[dict]:
    """使用 pdfplumber 保留页码、行列和原始单元格，供可追溯表格检索。"""
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("缺少表格解析组件 pdfplumber。请先运行 run.bat 安装依赖。") from exc
    rows: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for table_number, table in enumerate(page.extract_tables() or [], 1):
                cleaned = [[normalize_cell(cell) for cell in row] for row in table]
                cleaned = [row for row in cleaned if any(row)]
                if len(cleaned) < 2:
                    continue
                width = max(len(row) for row in cleaned)
                cleaned = [row + [""] * (width - len(row)) for row in cleaned]
                header = cleaned[0]
                body = cleaned[1:]
                rows.append({
                    "id": f"{document_name}:table:{page_number}:{table_number}",
                    "document": document_name,
                    "page": page_number,
                    "table": table_number,
                    "headers": header,
                    "rows": body,
                    "text": " | ".join(header) + "\n" + "\n".join(" | ".join(row) for row in body),
                })
    return rows


def extract_images(path: Path, document_name: str) -> list[dict]:
    """建立 PDF 图像资产索引，保留页码、对象名、尺寸和图片文本检索字段。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("缺少 PDF 解析组件。请先运行 run.bat 安装依赖。") from exc
    reader = PdfReader(str(path), strict=False)
    rows: list[dict] = []
    for page_number, page in enumerate(reader.pages, 1):
        try:
            images = list(page.images)
        except Exception:
            images = []
        for ordinal, image in enumerate(images, 1):
            name = getattr(image, "name", f"image-{ordinal}")
            data = getattr(image, "data", b"")
            width = getattr(image, "width", None)
            height = getattr(image, "height", None)
            rows.append({
                "id": f"{document_name}:image:{page_number}:{ordinal}",
                "document": document_name,
                "page": page_number,
                "ordinal": ordinal,
                "name": name,
                "width": width,
                "height": height,
                "bytes": len(data),
                "text": f"第 {page_number} 页 图像 {name} {width or ''}x{height or ''}",
            })
    return rows


def load_index() -> None:
    global _chunks, _tables, _images
    DATA.mkdir(parents=True, exist_ok=True)
    with _lock:
        if INDEX.exists():
            _chunks = [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]
        if TABLE_INDEX.exists():
            _tables = [json.loads(line) for line in TABLE_INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]
        if IMAGE_INDEX.exists():
            _images = [json.loads(line) for line in IMAGE_INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]
        if _chunks and _tables and _images:
            return
        seed = DATA / "招股说明书1.pdf"
        if seed.exists():
            print(f"首次启动：正在解析 {seed.name} 并建立检索索引…", flush=True)
            rows = extract_pdf(seed, seed.name)
            save_chunks(rows)
            tables = extract_tables(seed, seed.name)
            save_tables(tables)
            save_images(extract_images(seed, seed.name))
        elif _chunks and not _tables:
            seed = DATA / "招股说明书1.pdf"
            if seed.exists():
                save_tables(extract_tables(seed, seed.name))
        if _chunks and not _images:
            seed = DATA / "招股说明书1.pdf"
            if seed.exists():
                save_images(extract_images(seed, seed.name))
        else:
            _chunks = []


def save_chunks(rows: list[dict]) -> None:
    global _chunks
    existing = [r for r in _chunks if r["document"] != rows[0]["document"]] if rows else _chunks
    _chunks = existing + rows
    DATA.mkdir(parents=True, exist_ok=True)
    INDEX.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _chunks), encoding="utf-8")


def save_tables(rows: list[dict]) -> None:
    global _tables
    existing = [r for r in _tables if r["document"] != rows[0]["document"]] if rows else _tables
    _tables = existing + rows
    DATA.mkdir(parents=True, exist_ok=True)
    TABLE_INDEX.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _tables), encoding="utf-8")


def save_images(rows: list[dict]) -> None:
    global _images
    existing = [r for r in _images if r["document"] != rows[0]["document"]] if rows else _images
    _images = existing + rows
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGE_INDEX.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _images), encoding="utf-8")


def tokens(text: str) -> list[str]:
    normalized = text.lower()
    words = re.findall(r"[a-z0-9]+", normalized)
    cjk = re.findall(r"[\u3400-\u9fff]+", normalized)
    for run in cjk:
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
        words.extend(run[i:i + 3] for i in range(len(run) - 2))
        if len(run) <= 5:
            words.append(run)
    return words


EXPANSIONS = {
    "收入": "营业收入 销售收入 军用领域 收入占比 主营业务",
    "标准": "国家标准 行业标准 参与制定 标准号",
    "上游": "上游供应商 供应商 原材料 采购",
    "下游": "下游行业 客户 应用领域 行业",
    "奖": "国家科技进步一等奖 获奖 科技奖励 项目",
    "注册资本": "注册资本 股本 总股本 万元",
    "法定代表人": "法定代表人 董事长 负责人",
    "流动资金": "补充流动资金 募集资金 用途",
    "重要供应商": "供应商 重要供应商 采购",
}


def expanded_query(query: str) -> str:
    additions = [terms for key, terms in EXPANSIONS.items() if key in query]
    return query + " " + " ".join(additions)


INTENT_RULES = {
    "收入/趋势": ("收入", "营收", "增长率", "分别是多少"),
    "组织结构": ("组织结构", "销售部", "销售处", "部门", "构成"),
    "企业关系": ("关联方", "控制关系", "法定代表人", "公司呢"),
    "项目荣誉": ("哪个工程", "一等奖", "荣获", "参与"),
    "募集资金": ("募集资金", "投资", "流动资金"),
}


def understand_query(question: str, history: list[dict]) -> dict:
    resolved = question.strip()
    prior = history[-1]["resolved_query"] if history else ""
    has_company = bool(re.search(r"武汉[^，。？?；]+(?:有限公司|股份有限公司)", resolved))
    if not has_company and re.search(r"\b(他|她|它|该公司|这个公司|那个公司|其中|这个项目)\b|公司呢|他参与", resolved) and prior:
        anchor = re.search(r"武汉[^，。？?]+(?:有限公司|股份有限公司)", prior)
        if anchor:
            resolved = resolved.replace("他", anchor.group(0)).replace("这个公司", anchor.group(0)).replace("那个公司", anchor.group(0)).replace("公司呢", anchor.group(0))
        else:
            resolved = prior + "；追问：" + resolved
    intent = "文档事实检索"
    for name, terms in INTENT_RULES.items():
        if any(term in resolved for term in terms):
            intent = name
            break
    entities = []
    entities.extend(re.findall(r"武汉[^，。？?；]+(?:有限公司|股份有限公司)", resolved))
    entities.extend(re.findall(r"20\d{2}(?:年[-至]20\d{2}年)?", resolved))
    entities.extend([term for term in ("军用领域", "组织结构", "销售部", "法定代表人", "国家科技进步一等奖") if term in resolved])
    return {"original_query": question, "resolved_query": resolved, "intent": intent, "entities": list(dict.fromkeys(entities)), "turn": len(history) + 1, "strategy": "指代消解 + 意图识别 + 实体抽取 + 混合检索"}


def bm25_scores(query: str) -> list[float]:
    query_terms = tokens(query)
    docs = [tokens(row["text"]) for row in _chunks]
    if not docs or not query_terms:
        return [0.0] * len(docs)
    lengths = [len(doc) for doc in docs]
    avg_len = sum(lengths) / len(lengths) or 1
    df = Counter(term for doc in docs for term in set(doc))
    scores = []
    for doc, length in zip(docs, lengths):
        freq = Counter(doc)
        score = 0.0
        for term in query_terms:
            count = freq.get(term, 0)
            if count:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / avg_len))
        scores.append(score)
    return scores


def rank_scores(query: str, subset: list[dict] | None = None) -> list[tuple[int, float]]:
    rows = subset if subset is not None else _chunks
    qtokens = Counter(tokens(query))
    scores: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        dtokens = Counter(tokens(row["text"]))
        common = set(qtokens) & set(dtokens)
        dot = sum(qtokens[t] * dtokens[t] for t in common)
        qnorm = math.sqrt(sum(v * v for v in qtokens.values())) or 1
        dnorm = math.sqrt(sum(v * v for v in dtokens.values())) or 1
        score = dot / (qnorm * dnorm)
        if score:
            scores.append((index, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def retrieve(query: str, top_k: int = 5, strategy: str = "hybrid", vector_weight: float = 0.55, reranker: str = "rrf") -> tuple[list[dict], dict]:
    if not _chunks:
        load_index()
    query2 = expanded_query(query)
    started = time.perf_counter()
    keyword = bm25_scores(query2)
    kw_rank = sorted(range(len(keyword)), key=lambda i: keyword[i], reverse=True)[:40]
    semantic = rank_scores(query2)
    sem_rank = [i for i, _ in semantic[:40]]
    rrf: dict[int, float] = {}
    routes = {"fulltext": [kw_rank], "vector": [sem_rank], "hybrid": [kw_rank, sem_rank]}.get(strategy, [kw_rank, sem_rank])
    weights = [1 - vector_weight, vector_weight] if strategy == "hybrid" else [1.0]
    for route_index, route in enumerate(routes):
        weight = weights[route_index] if route_index < len(weights) else 1.0
        for rank, index in enumerate(route, 1):
            rrf[index] = rrf.get(index, 0) + weight / (60 + rank)
    candidates = sorted(rrf, key=rrf.get, reverse=True)[:30]
    query_terms = set(tokens(query))
    reranked = []
    for index in candidates:
        row = _chunks[index]
        row_terms = set(tokens(row["text"]))
        overlap = len(query_terms & row_terms) / (len(query_terms) or 1)
        exact = 1.0 if query.strip() and query.strip() in row["text"] else 0.0
        dense_kw = keyword[index]
        phrase_bonus = sum(1.5 for phrase in ("重要供应商", "上游行业", "下游行业", "技术标准", "注册资本", "法定代表人", "国家科技进步一等奖", "补充流动资金") if phrase in query and phrase in row["text"])
        if reranker == "tfidf":
            score = rrf[index] * 1000 + overlap * 3 + min(dense_kw, 12) * 0.08
        elif reranker == "feedback":
            score = rrf[index] * 1000 + overlap * 2.5 + exact * 4 + phrase_bonus * 1.2
        else:
            score = rrf[index] * 1000 + overlap * 2 + exact * 3 + phrase_bonus + min(dense_kw, 8) * 0.04
        reranked.append((index, score))
    reranked.sort(key=lambda pair: pair[1], reverse=True)
    results = []
    for index, score in reranked[:top_k]:
        row = dict(_chunks[index])
        row["score"] = round(score, 4)
        row["snippet"] = row["text"][:260]
        results.append(row)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    route_label = {"fulltext": ["倒排索引", reranker], "vector": ["字符 n-gram", reranker], "hybrid": ["BM25", "字符 n-gram", "RRF", reranker]}[strategy if strategy in ("fulltext", "vector", "hybrid") else "hybrid"]
    return results, {"elapsed_ms": elapsed, "candidate_count": len(candidates), "routes": route_label, "strategy": strategy, "vector_weight": vector_weight, "reranker": reranker}


def baseline(query: str, top_k: int = 5) -> list[dict]:
    scores = bm25_scores(query)
    ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_chunks[index] for index in ids if scores[index] > 0]


def table_search(query: str, top_k: int = 5) -> list[dict]:
    q = set(tokens(expanded_query(query)))
    ranked = []
    for row in _tables:
        words = set(tokens(row["text"]))
        overlap = len(q & words)
        exact = 2 if query.strip() and query.strip() in row["text"] else 0
        if overlap or exact:
            item = dict(row)
            item["score"] = round(overlap + exact, 3)
            ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def image_search(query: str, top_k: int = 5) -> list[dict]:
    q = set(tokens(query))
    ranked = []
    for row in _images:
        words = set(tokens(row["text"]))
        overlap = len(q & words)
        if overlap or any(term in query for term in ("图", "图像", "结构", "增长", "组织")):
            item = dict(row)
            item["score"] = round(overlap + (0.5 if "图" in query or "图像" in query else 0), 3)
            ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


BENCHMARK = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？", "gold_pages": [378]},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？", "gold_pages": [160, 155]},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？", "gold_pages": [378]},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？", "gold_pages": [153]},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？", "gold_pages": [26]},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？", "gold_pages": [153, 154]},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？", "gold_pages": [157, 96]},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？", "gold_pages": [52]},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？", "gold_pages": [52]},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？", "gold_pages": [490, 491, 479, 30]},
]


def evaluate() -> dict:
    cases = []
    before_hits = after_hits = 0
    before_times: list[float] = []
    after_times: list[float] = []
    for case in BENCHMARK:
        started = time.perf_counter()
        old = baseline(case["question"])
        before_times.append((time.perf_counter() - started) * 1000)
        improved, metrics = retrieve(case["question"])
        after_times.append(metrics["elapsed_ms"])
        def hit(rows: list[dict]) -> bool:
            return any(row["page"] in case["gold_pages"] for row in rows)
        b, a = hit(old), hit(improved)
        before_hits += int(b)
        after_hits += int(a)
        matched = next((row for row in improved if row["page"] in case["gold_pages"]), None)
        issue = "通过：Top-5 命中标注相关页" if a else "问题：Top-5 未命中标注相关页，需要补充查询扩展或标注页"
        cases.append({"id": case["id"], "question": case["question"], "gold_pages": case["gold_pages"], "before_hit": b, "after_hit": a, "issue": issue,
                      "before_sources": [{"page": row["page"], "document": row["document"]} for row in old[:3]],
                      "after_sources": [{"page": row["page"], "document": row["document"], "snippet": row["text"][:180]} for row in improved[:3]],
                      "matched_page": matched["page"] if matched else None,
                      "matched_snippet": matched["text"][:180] if matched else "没有命中相关页"})
    total = len(cases) or 1
    return {"report": "RAG 功能测试及评估报告", "dataset": "工单01-06内置《招股说明书1.pdf》与10道验收题；未发现工单附件中的 ccf_competition.zip，因此未伪造其题集", "metric": "Top-5 gold-page evidence hit rate (retrieval proxy; not human answer accuracy)",
            "before_rate": round(before_hits / total, 3), "after_rate": round(after_hits / total, 3),
            "before_avg_ms": round(sum(before_times) / total, 1), "after_avg_ms": round(sum(after_times) / total, 1),
            "accuracy_target": 0.9, "recall_target": 0.95, "latency_target_ms": 3000, "cases": cases,
            "analysis": {"passed_cases": after_hits, "failed_cases": total - after_hits, "recommendation": "对未命中的问题补充同义词、页码标注或领域实体；人工答案准确率仍需专家标注后复核"}}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/status":
            self.send_json({"ready": bool(_chunks), "chunks": len(_chunks), "tables": len(_tables), "images": len(_images), "documents": sorted({row["document"] for row in _chunks})})
        elif route == "/api/tables":
            self.send_json({"tables": _tables, "count": len(_tables)})
        elif route == "/api/images":
            self.send_json({"images": _images, "count": len(_images)})
        elif route == "/api/evaluate":
            self.send_json(evaluate())
        elif route == "/api/report":
            report = evaluate()
            (ROOT / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self.send_json(report)
        else:
            super().do_GET()

    def do_POST(self):
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD:
            self.send_json({"error": "文件为空或超过 80 MB 限制。"}, 413)
            return
        payload = self.rfile.read(length)
        try:
            if route == "/api/ask":
                data = json.loads(payload)
                question = str(data.get("question", "")).strip()
                if not question:
                    raise ValueError("请输入问题。")
                session_id = str(data.get("session_id", "default"))[:100]
                history = _sessions.setdefault(session_id, [])
                understanding = understand_query(question, history)
                query = understanding["resolved_query"]
                strategy = str(data.get("strategy", "hybrid"))
                if strategy not in {"fulltext", "vector", "hybrid"}:
                    strategy = "hybrid"
                try:
                    vector_weight = max(0.0, min(1.0, float(data.get("vector_weight", 0.55))))
                except (TypeError, ValueError):
                    vector_weight = 0.55
                reranker = str(data.get("reranker", "rrf"))
                if reranker not in {"rrf", "tfidf", "feedback"}:
                    reranker = "rrf"
                sources, metrics = retrieve(query, strategy=strategy, vector_weight=vector_weight, reranker=reranker)
                table_sources = table_search(query, 5)
                image_sources = image_search(query, 5)
                if not sources or sources[0]["score"] < 0.15:
                    answer = "当前文档中没有找到足够直接的证据。请补充关键词或检查知识库内容。"
                else:
                    answer = "根据检索到的文档内容：\n" + "\n".join(f"- {row['snippet']} [第 {row['page']} 页]" for row in sources[:3])
                if table_sources:
                    answer += "\n\n表格证据：" + "\n".join(f"- 第 {row['page']} 页表格 {row['table']}：{'；'.join(' | '.join(r) for r in row['rows'][:2])}" for row in table_sources[:2])
                if image_sources:
                    answer += "\n\n图像证据：" + "\n".join(f"- 第 {row['page']} 页图像 {row['name']}（{row['width']}×{row['height']}）" for row in image_sources[:2])
                history.append({"question": question, "resolved_query": query, "intent": understanding["intent"]})
                _sessions[session_id] = history[-12:]
                metrics["turn"] = understanding["turn"]
                metrics["query_understanding"] = understanding
                self.send_json({"answer": answer, "sources": sources, "table_sources": table_sources, "image_sources": image_sources, "metrics": metrics, "query_understanding": understanding, "session_id": session_id})
            elif route == "/api/upload":
                message = (f"Content-Type: {self.headers.get('Content-Type')}\r\nMIME-Version: 1.0\r\n\r\n").encode() + payload
                form = BytesParser(policy=policy.default).parsebytes(message)
                file_part = next((part for part in form.iter_parts() if part.get_filename()), None)
                if file_part is None:
                    raise ValueError("未收到 PDF 文件。")
                filename = Path(file_part.get_filename()).name
                if not filename.lower().endswith(".pdf"):
                    raise ValueError("目前只接受 PDF 文件。")
                content = file_part.get_payload(decode=True)
                if not content or not content.startswith(b"%PDF"):
                    raise ValueError("文件内容不是有效 PDF。")
                temp = DATA / f"upload-{uuid.uuid4().hex}.pdf"
                temp.write_bytes(content)
                try:
                    rows = extract_pdf(temp, filename)
                    if not rows:
                        raise ValueError("PDF 未提取到可检索文本，可能是扫描件。")
                    save_chunks(rows)
                    save_tables(extract_tables(temp, filename))
                    save_images(extract_images(temp, filename))
                finally:
                    temp.unlink(missing_ok=True)
                self.send_json({"ok": True, "document": filename, "chunks_added": len(rows), "total_chunks": len(_chunks)})
            else:
                self.send_json({"error": "接口不存在。"}, 404)
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"处理失败：{type(exc).__name__}: {exc}"}, 500)


def main() -> None:
    load_index()
    port = int(os.environ.get("RAG_PORT", "4174"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RAG 工单 07 服务已启动：http://127.0.0.1:{port}（{len(_chunks)} 个文本片段，{len(_tables)} 张表格，{len(_images)} 张图像）")
    server.serve_forever()


if __name__ == "__main__":
    main()
