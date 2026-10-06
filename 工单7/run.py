"""工单编号：人工智能NLP-RAG-功能测试及评估。"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    cache = Path.home() / ".cache/huggingface/hub/models--moka-ai--m3e-small/snapshots"
    local = next((p for p in cache.glob("*") if (p / "modules.json").exists()), None)
    model = SentenceTransformer(str(local) if local else "moka-ai/m3e-small", device="cpu")
    model.max_seq_length = 256
    return model


def build(folder):
    """沿用工单01—06的逐页解析、切块和M3E向量索引。"""
    chunks, sources = [], []
    paths = list(Path(folder).glob("*.pdf")) if isinstance(folder, (str, Path)) else folder
    for path in sorted(paths):
        parts = path.stem.split("__")
        company = parts[3] if len(parts) > 4 else path.stem
        year = parts[4] if len(parts) > 4 else ""
        with pymupdf.open(path) as doc:
            sources.append({"file": path.name, "path": str(path.resolve()), "pages": len(doc)})
            for number, page in enumerate(doc, 1):
                text = re.sub(r"[ \t]+", " ", page.get_text(sort=True)).strip()
                for start in range(0, len(text), 800):
                    body = text[start:start + 1000]
                    if len(body) < 30:
                        continue
                    chunks.append({"id": len(chunks), "file": path.name, "company": company,
                                   "year": year, "page": number, "text": body})
    if not chunks:
        raise ValueError("目录里没有可解析的PDF，请检查文件夹。")
    DATA.mkdir(exist_ok=True)
    save_json(DATA / "chunks.json", chunks)
    save_json(DATA / "sources.json", sources)
    model = encoder()
    texts = [c["company"] + " " + c["year"] + " " + c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    np.save(DATA / "vectors.npy", vectors.astype("float32"))
    print(f"已建立 {len(sources)} 份PDF、{len(chunks)} 个文本块的索引。", flush=True)


def load():
    chunks = read_json(DATA / "chunks.json")
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2, max_features=60000)
    matrix = tfidf.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vectors": np.load(DATA / "vectors.npy"),
            "tfidf": tfidf, "matrix": matrix, "model": encoder()}


def search(question, index, top_k=5):
    """复用工单06的向量+字面加权检索思想，不加针对答案的规则。"""
    vector = index["model"].encode([question], normalize_embeddings=True)[0]
    dense = index["vectors"] @ vector
    words = index["tfidf"].transform([question])
    lexical = (index["matrix"] @ words.T).toarray().ravel()
    scores = 0.65 * dense + 0.35 * lexical
    return rank(scores, index["chunks"], top_k)


def rank(scores, chunks, top_k):
    found, used = [], set()
    for position in np.argsort(scores)[::-1]:
        chunk = chunks[int(position)]
        key = (chunk["file"], chunk["page"])
        if key in used:
            continue
        used.add(key)
        found.append({**chunk, "score": round(float(scores[position]), 5)})
        if len(found) == top_k:
            break
    return found


def evaluate(index, search_fn=search, output=ROOT / "评估结果.json"):
    results = []
    for case in read_json(ROOT / "questions.json"):
        started = time.perf_counter()
        hits = search_fn(case["question"], index)
        gold = {(g["company"], g["page"]) for g in case["gold"]}
        relevance = [int((h["company"], h["page"]) in gold) for h in hits]
        found = {(h["company"], h["page"]) for h in hits} & gold
        # NDCG按真实排名计算；无命中为0，理想序列含所有标注证据页。
        missing = len(gold) - len(found)
        truth = np.array([relevance + [1] * missing + [0]])
        predicted = np.array([list(range(len(hits), 0, -1)) + [-i - 1 for i in range(missing + 1)]])
        results.append({"id": case["id"], "question": case["question"],
                        "reference": case["reference"], "gold": case["gold"], "hits": hits,
                        "precision@5": sum(relevance) / 5,
                        "recall@5": len(found) / len(gold),
                        "mrr@5": next((1 / (i + 1) for i, v in enumerate(relevance) if v), 0),
                        "ndcg@5": float(ndcg_score(truth, predicted, k=5)),
                        "seconds": round(time.perf_counter() - started, 4)})
    metrics = ("precision@5", "recall@5", "mrr@5", "ndcg@5", "seconds")
    summary = {m: round(sum(r[m] for r in results) / len(results), 4) for m in metrics}
    save_json(output, {"summary": summary, "results": results,
                       "note": "仅对人工标注证据页评估检索；其他相关页未标注，精度可能偏低。未评估生成答案。"})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", help="首次运行时提供解压后的9份PDF目录")
    args = parser.parse_args()
    if args.pdf_dir:
        build(args.pdf_dir)
    evaluate(load())
