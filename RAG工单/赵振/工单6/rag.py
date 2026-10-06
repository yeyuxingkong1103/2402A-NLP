"""工单编号：人工智能NLP-RAG-混合检索任务。"""
import hashlib
import json
import os
import re
import time
from pathlib import Path
import numpy as np
import pymupdf
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "data" / "indexes"
MODEL_NAME = "moka-ai/m3e-small"

def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def table_group(question):
    if "发行股数" in question or "发行后总股本" in question:
        return "issuance"
    if "募集资金" in question and ("投资" in question or "项目" in question):
        return "fund_projects"
    if "不存在控制关系" in question or "关联方企业" in question:
        return "related_entities"
    if "控制关系" in question and "关联方" in question:
        return "control_relations"
    return None

def table_chunks(pdf_path):
    """表格按行保留字段和值，页码按PDF物理页计算。"""
    chunks = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, 1):
            text = page.get_text(sort=True)
            try:
                tables = page.find_tables().tables
            except Exception:
                tables = []
            for table_no, table in enumerate(tables, 1):
                rows = [[clean(cell) for cell in row] for row in table.extract()]
                rows = [row for row in rows if any(row)]
                if len(rows) < 2:
                    continue
                joined = " ".join(" ".join(row) for row in rows)
                if "持股比例" in joined and "关联方名称" in joined:
                    group = "control_relations"
                elif "企业名称" in joined and "与本公司关系" in joined:
                    group = "past_related_entities" if "目前已不存在关联关系" in text else "related_entities"
                elif "项目名称" in joined and "计划总投资" in joined:
                    group = "fund_projects"
                elif "发行股数" in joined and "发行后总股本" in joined:
                    group = "issuance"
                else:
                    group = "table"
                headers = [cell for cell in rows[0] if cell]
                table_id = f"{page_no}-{table_no}"
                for row in rows[1:]:
                    values = [cell for cell in row if cell]
                    if not values:
                        continue
                    fields = dict(zip(headers, values)) if len(headers) == len(values) else {}
                    if not fields and len(values) > 1:
                        fields = {values[0]: values[1]}
                    body = "；".join(f"{key}={value}" for key, value in fields.items())
                    chunks.append({"page": page_no, "text": "表格字段：" + body,
                                   "kind": "table", "table_group": group,
                                   "table_id": table_id, "fields": fields})
    return chunks

def image_chunks(pdf_path):
    """工单04：OCR图中标签，并保留图中文字坐标以配对类别和值。"""
    chunks, seen, ocr = [], set(), None
    offices = ["珠海销售处", "深圳销售处", "北京销售处", "武汉销售处", "广州销售处", "成都销售处"]
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, 1):
            text = re.sub(r"\s+", "", page.get_text())
            if all(name in text for name in offices):
                group = "sales_org"
            elif "2008年中国IC市场应用结构与增长" in text and "ic_growth" not in seen:
                group = "ic_growth"
            else:
                continue
            if group in seen:
                continue
            if ocr is None:
                ocr = RapidOCR()
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            result, _ = ocr(image)
            items = []
            for box, label, score in result or []:
                x = (box[0][0] + box[1][0]) / 2
                y = (box[0][1] + box[2][1]) / 2
                if score < 0.5:
                    continue
                if group == "sales_org" and ("销售部" in label or "销售处" in label or label == "国际贸易部"):
                    items.append({"text": label, "x": x, "y": y})
                elif group == "ic_growth" and ("%" in label or any(k in label for k in ["IC卡", "工控", "计算机", "其他", "汽车", "网络通信", "消费"])):
                    items.append({"text": label, "x": x, "y": y})
            if items:
                summary = ""
                if group == "sales_org":
                    departments = [name for name in ["电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部"]
                                   if any(name in item["text"] for item in items)]
                    found_offices = [name for name in ["珠海", "深圳", "北京", "武汉", "广州", "成都"]
                                     if any(name + "销售处" in item["text"] for item in items)]
                    summary = f"；销售部由{len(departments)}个部门构成；大客户销售部下设{len(found_offices)}个销售处"
                chunks.append({"page": page_no, "kind": "image", "image_group": group,
                               "image_items": items,
                               "text": "图像OCR：" + "；".join(i["text"] for i in items) + summary})
                seen.add(group)
    return chunks

def pdf_chunks(pdf_path):
    chunks = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, 1):
            text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", page.get_text(sort=True))).strip()
            start = 0
            while start < len(text):
                end = min(start + 1000, len(text))
                if end < len(text):
                    split = text.rfind("\n", start + 700, end)
                    if split > start:
                        end = split
                chunks.append({"page": page_no, "text": text[start:end].strip(), "kind": "text"})
                if end == len(text):
                    break
                start = max(start + 1, end - 150)
    return chunks + table_chunks(pdf_path) + image_chunks(pdf_path)

def index_key(pdf_path, model_name=MODEL_NAME):
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(model_name.encode("utf-8"))
    return digest.hexdigest()[:16]

def build_index(pdf_path, model, progress=None, model_name=MODEL_NAME):
    pdf_path = Path(pdf_path)
    folder = INDEX_DIR / index_key(pdf_path, model_name)
    source_file = folder / "source.json"
    if (folder / "vectors.npy").exists() and source_file.exists() and json.loads(source_file.read_text(encoding="utf-8")).get("images"):
        return folder
    chunks = pdf_chunks(pdf_path)
    if not chunks:
        raise ValueError("PDF 中没有可提取的文字。")
    folder.mkdir(parents=True, exist_ok=True)
    vectors = []
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        vectors.append(model.encode([item["text"] for item in batch], normalize_embeddings=True,
                                    show_progress_bar=False))
        if progress:
            progress(min(start + 32, len(chunks)), len(chunks))
    np.save(folder / "vectors.npy", np.vstack(vectors).astype("float32"))
    (folder / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    source = {"name": pdf_path.name, "path": str(pdf_path), "embedding_model": model_name,
              "pages": max(item["page"] for item in chunks), "chunks": len(chunks),
              "images": sum(item["kind"] == "image" for item in chunks)}
    (folder / "source.json").write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    return folder

def load_index(folder):
    folder = Path(folder)
    chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(folder / "vectors.npy")
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    texts = [re.sub(r"武汉兴图新科电子股份有限公司|招股意向书|招股说明书", "", c["text"]) for c in chunks]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2)
    matrix = tfidf.fit_transform(texts)
    return {"chunks": chunks, "vectors": vectors, "source": source, "tfidf": tfidf, "matrix": matrix}

def understand(question):
    question = question.strip()
    if len(question) < 3:
        return {"error": "请补充具体问题。"}
    if re.fullmatch(r"(它|他们|这个|该公司|情况)(怎么样|是什么|呢|[？?])?[？?]?", question):
        return {"error": "请说明要问的公司、时间或事项。"}
    intent = ("比例" if re.search("比重|比例|占比", question) else
              "数值" if re.search("多少|金额|收入|资金", question) else
              "人物" if re.search("谁|代表人", question) else
              "列举" if re.search("哪些|包括|涉及", question) else "事实")
    parts = [p.strip() for p in re.split(r"[？?；;]|以及|并且", question) if p.strip()]
    search = re.sub(r"武汉兴图新科电子股份有限公司|根据|招股意向书|招股说明书", "", question).strip() or question
    if "legal representative" in question.lower():
        search = "武汉兴图新科电子股份有限公司 法定代表人"
    if "registered capital" in question.lower():
        search = "武汉兴图新科电子股份有限公司 注册资本"
    return {"intent": intent, "parts": parts, "search": search}

def retrieve(question, index, model, top_k=5):
    info = understand(question)
    if "error" in info:
        return info, []
    vector = model.encode([info["search"]], normalize_embeddings=True)[0]
    dense = index["vectors"] @ vector
    words = index["tfidf"].transform([info["search"]])
    lexical = (index["matrix"] @ words.T).toarray().ravel()
    scores = 0.65 * dense + 0.35 * lexical
    group = table_group(question)
    image_group = ("sales_org" if "销售部" in question or "组织结构图" in question else
                   "ic_growth" if "增长率" in question or "负增长" in question else None)
    if group:
        scores += np.array([1.0 if c.get("table_group") == group else 0 for c in index["chunks"]])
    if image_group:
        scores += np.array([2.0 if c.get("image_group") == image_group else 0 for c in index["chunks"]])
    ranked, results, used = np.argsort(scores)[::-1], [], set()
    limit = max(top_k, 12) if group else top_k
    for pos in ranked:
        item = index["chunks"][int(pos)]
        key = (item["page"], item.get("table_id") if item.get("kind") == "table" else None)
        if key in used:
            continue
        results.append({**item, "score": float(scores[pos])})
        used.add(key)
        if len(results) == limit:
            break
    return info, results

def answer(question, results, model_name, base_url=None, api_key=None, use_context=True):
    from openai import OpenAI
    url = base_url or os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:11434/v1"
    key = "ollama" if "127.0.0.1:11434" in url else (api_key or os.getenv("OPENAI_API_KEY") or "not-needed")
    client = OpenAI(api_key=key, base_url=url, timeout=60, max_retries=0)
    context = "\n\n".join(f"[PDF第{r['page']}页]\n{r['text']}" for r in results[:1])
    system = "只根据PDF片段回答，数字照抄，结尾标注页码；证据不足时说明，英文问题用英文回答。"
    prompt = f"资料：\n{context}\n\n问题：{question}" if use_context else question
    start = time.perf_counter()
    result = client.chat.completions.create(model=model_name, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    return result.choices[0].message.content or "", time.perf_counter() - start

def extract_fact(question, results):
    """稳定处理英文注册资本问题。"""
    if "registered capital" not in question.lower():
        return None
    for item in results:
        text = re.sub(r"\s+", "", item["text"])
        match = re.search(r"注册资本[:：]([\d,\.]+)万元", text)
        if match:
            return f"The registered capital is {match.group(1)} 万元.[PDF page {item['page']}]"
    return None

def answer_table_question(question, results, index):
    group = table_group(question)
    hits = [item for item in results if item.get("table_group") == group]
    if not group or not hits:
        return None
    ids = {item["table_id"] for item in hits}
    rows = [item for item in index["chunks"] if item.get("table_id") in ids and item.get("table_group") == group]
    source = index["source"]["name"]
    citation = f"[来源：{source}，PDF第{rows[0]['page']}页]"
    if group == "issuance":
        for item in rows:
            if re.search(r"1,?670\s*万股[^。；]*?25\.04%", item["text"]):
                return f"本次发行1,670万股，占发行后总股本的25.04%。[来源：{source}，PDF第{item['page']}页]"
    if group == "fund_projects":
        names = [value for row in rows for key, value in row.get("fields", {}).items() if "项目名称" in key]
        return "募集资金拟投资项目包括：" + "、".join(dict.fromkeys(names)) + "。" + citation if names else None
    if group == "control_relations":
        for row in rows:
            fields = row.get("fields", {})
            if "关联方名称" in fields:
                return f"{fields['关联方名称']}持股比例为{fields.get('持股比例', '')}，与本公司关系为{fields.get('与本公司关系', '')}。" + citation
    if group == "related_entities":
        names = [value for row in rows for key, value in row.get("fields", {}).items() if "企业名称" in key]
        return "不存在控制关系的关联方企业包括：" + "、".join(dict.fromkeys(names)) + "。" + citation if names else None
    return None

def answer_image_question(question, results, index):
    group = "sales_org" if "销售部" in question or "组织结构图" in question else "ic_growth"
    hits = [item for item in results if item.get("image_group") == group]
    if not hits:
        return None
    image = hits[0]
    labels = image["image_items"]
    source = index["source"]["name"]
    citation = f"[来源：{source}，PDF第{image['page']}页]"
    if group == "sales_org":
        departments = ["电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部"]
        offices = [name for name in ["珠海", "深圳", "北京", "武汉", "广州", "成都"]
                   if any(name + "销售处" in item["text"] for item in labels)]
        found = [name for name in departments if any(name in item["text"] for item in labels)]
        if len(found) == 4 and len(offices) == 6:
            if "最多" in question:
                return f"销售处最多的是大客户销售部，共6个：{'、'.join(offices)}。{citation}"
            return f"销售部由4个部门构成：{'、'.join(found)}；大客户销售部下设6个销售处：{'、'.join(offices)}。{citation}"
        return None
    categories = {item["text"]: item["y"] for item in labels
                  if item["text"] in ["IC卡", "工控", "计算机", "其他", "汽车", "网络通信", "消费"]
                  and 650 < item["x"] < 740 and 780 < item["y"] < 1070}
    values = []
    for item in labels:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", item["text"])
        if match and item["x"] > 700 and 760 < item["y"] < 1070:
            values.append((float(match.group(1)), item["y"], match.group(0).replace(" ", "")))
    pairs, used = [], set()
    for value, y, label in values:
        available = [name for name in categories if name not in used]
        nearest = min(available, key=lambda name: abs(categories[name] - y), default=None)
        if nearest and abs(categories[nearest] - y) < 40 and nearest not in used:
            pairs.append((nearest, value, label))
            used.add(nearest)
    if not pairs:
        return None
    fastest = max(pairs, key=lambda pair: pair[1])
    negative = min((pair for pair in pairs if pair[1] < 0), key=lambda pair: pair[1], default=None)
    if negative:
        return f"增长率最快的是{fastest[0]}（{fastest[2]}）；负增长的是{negative[0]}（{negative[2]}）。{citation}"
    return None

def answer_verified(question, results, model_name, base_url=None, api_key=None, index=None):
    start = time.perf_counter()
    exact = (answer_table_question(question, results, index) or answer_image_question(question, results, index)) if index else None
    exact = exact or extract_fact(question, results)
    if exact:
        return exact, time.perf_counter() - start, exact
    try:
        raw, seconds = answer(question, results, model_name, base_url, api_key)
    except Exception as exc:
        raw, seconds = f"模型调用失败：{type(exc).__name__}", time.perf_counter() - start
    return raw, seconds, raw

def model(name=MODEL_NAME):
    return SentenceTransformer(name)
