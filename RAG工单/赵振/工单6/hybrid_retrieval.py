"""工单编号：人工智能NLP-RAG-混合检索任务。"""

import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from rag import ROOT, table_group, understand


FIELDS = ("title", "summary", "body")
FIELD_WEIGHTS = {"title": 2.0, "summary": 1.2, "body": 1.0}


def tokens(text):
    result = []
    for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if "\u4e00" <= part[0] <= "\u9fff":
            result.extend(part[i:i + 2] for i in range(len(part) - 1))
            if len(part) == 1:
                result.append(part)
        else:
            result.append(part)
    return result


def prepare_fulltext(index):
    """为标题、摘要和正文建立轻量倒排索引。"""
    if "inverted" in index:
        return
    postings = defaultdict(dict)
    fields = []
    for number, chunk in enumerate(index["chunks"]):
        text = chunk["text"]
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), text[:80])
        record = {
            "title": chunk.get("title") or first_line[:80],
            "summary": chunk.get("summary") or text[:240],
            "body": text,
        }
        fields.append(record)
        for name in FIELDS:
            for term in set(tokens(record[name])):
                postings[term][number] = postings[term].get(number, 0) + FIELD_WEIGHTS[name]
    index["fields"] = fields
    index["inverted"] = dict(postings)


def fulltext_scores(query, index, field="全部字段", fuzzy=True):
    prepare_fulltext(index)
    names = FIELDS if field == "全部字段" else (field,)
    query = query.strip()
    expression = re.split(r"\s+(AND|OR|NOT)\s+", query, flags=re.I)
    if len(expression) == 3:
        left, operator, right = expression
        def contains(value, record):
            return any(value.lower() in record[name].lower() for name in names)
        left_hits = {i for i, record in enumerate(index["fields"]) if contains(left, record)}
        right_hits = {i for i, record in enumerate(index["fields"]) if contains(right, record)}
        hits = left_hits & right_hits if operator.upper() == "AND" else (
            left_hits | right_hits if operator.upper() == "OR" else left_hits - right_hits)
        return np.array([1.0 if i in hits else 0.0 for i in range(len(index["chunks"]))])

    phrase = re.findall(r'["“](.+?)["”]', query)
    if phrase:
        phrase = phrase[0]
        return np.array([1.0 if any(phrase.lower() in record[name].lower() for name in names) else 0.0
                         for record in index["fields"]])

    terms = [term for term in tokens(query) if term not in {"and", "or", "not"}]
    scores = np.zeros(len(index["chunks"]))
    vocab = index["inverted"]
    for term in terms:
        matches = [term] if term in vocab else []
        if not matches and fuzzy:
            matches = difflib.get_close_matches(term, vocab, n=2, cutoff=0.6)
        for match in matches:
            for number, weight in vocab[match].items():
                if any(match in tokens(index["fields"][number][name]) for name in names):
                    scores[number] += weight
    return scores


def _normalize(values):
    values = np.asarray(values, dtype=float)
    high = values.max(initial=0.0)
    return values / high if high else values


def _ranks(values):
    return np.argsort(np.argsort(-values))


def _llm_rerank(question, results, model_name, base_url, api_key):
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "ollama", base_url=base_url, timeout=4, max_retries=0)
    passages = [{"id": i, "text": item["text"][:500]} for i, item in enumerate(results[:8])]
    prompt = ("按问题相关度给片段打0到10分，只返回JSON，格式为"
              '{"scores":[{"id":0,"score":8}]}。\n'
              f"问题：{question}\n片段：{json.dumps(passages, ensure_ascii=False)}")
    reply = client.chat.completions.create(
        model=model_name, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content
    scores = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])["scores"]
    for item in scores:
        if 0 <= item["id"] < len(results):
            results[item["id"]]["score"] += float(item["score"]) / 10


def _feedback_rerank(results, path):
    counts = defaultdict(lambda: [0, 0])
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            counts[record.get("pages", [0])[0]][0 if record.get("rating") == "有帮助" else 1] += 1
    for item in results:
        good, bad = counts[item["page"]]
        if good + bad:
            item["score"] += 0.05 * (good - bad) / (good + bad)


def search(question, index, model, strategy="混合检索", vector_weight=0.6,
           fusion="加权平均", reranker="TF-IDF", field="全部字段", fuzzy=True,
           model_name="deepseek-r1:1.5b", base_url="http://127.0.0.1:11434/v1", api_key=""):
    """执行向量、全文或混合检索，并可选重排。"""
    info = understand(question)
    if "error" in info:
        return info, [], {"warning": ""}
    query = question.strip()
    vector = model.encode([info["search"]], normalize_embeddings=True)[0]
    dense = _normalize(index["vectors"] @ vector)
    text = _normalize(fulltext_scores(query, index, field, fuzzy))
    if strategy == "向量检索":
        scores = dense
    elif strategy == "全文检索":
        scores = text
    elif fusion == "RRF投票":
        scores = vector_weight / (60 + _ranks(dense)) + (1 - vector_weight) / (60 + _ranks(text))
    elif fusion == "排序投票":
        scores = (vector_weight * (_ranks(dense) < 20) + (1 - vector_weight) * (_ranks(text) < 20)).astype(float)
    else:
        scores = vector_weight * dense + (1 - vector_weight) * text

    group = table_group(question)
    image_group = "sales_org" if "销售部" in question or "组织结构图" in question else (
        "ic_growth" if "增长率" in question or "负增长" in question else None)
    for i, chunk in enumerate(index["chunks"]):
        if group and chunk.get("table_group") == group:
            scores[i] += 1.0
        if image_group and chunk.get("image_group") == image_group:
            scores[i] += 1.5

    results = []
    used = set()
    positions = {}
    for position in np.argsort(scores)[::-1]:
        chunk = index["chunks"][int(position)]
        key = (chunk["page"], chunk.get("table_id") if chunk.get("kind") == "table" else None)
        if key in used:
            continue
        results.append({**chunk, "score": float(scores[position])})
        positions[id(results[-1])] = int(position)
        used.add(key)
        if len(results) == 12:
            break

    warning = ""
    if reranker == "TF-IDF":
        words = index["tfidf"].transform([query])
        lexical = _normalize((index["matrix"] @ words.T).toarray().ravel())
        for item in results:
            pos = positions[id(item)]
            item["score"] = 0.75 * item["score"] + 0.25 * lexical[pos]
    elif reranker == "用户反馈":
        _feedback_rerank(results, ROOT / "data" / "feedback.jsonl")
    elif reranker == "LLM重排":
        try:
            _llm_rerank(query, results, model_name, base_url, api_key)
        except Exception as exc:
            warning = f"LLM重排不可用，保留原排序：{type(exc).__name__}"
    results.sort(key=lambda item: item["score"], reverse=True)
    return info, results, {"strategy": strategy, "fusion": fusion, "reranker": reranker,
                           "vector_candidates": int(np.count_nonzero(dense)),
                           "fulltext_candidates": int(np.count_nonzero(text)), "warning": warning}
