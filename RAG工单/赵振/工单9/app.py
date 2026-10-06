"""工单编号：人工智能NLP-RAG-GraphRAG优化任务。"""
import json
import os
import re
import time
import urllib.request
import threading
from pathlib import Path

import numpy as np
import pymupdf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent
DATA = Path(os.getenv("DATA_DIR", ROOT / "data"))
COMPANIES = {"平安银行": "ping an bank", "招商银行": "china merchants bank",
             "邮储银行": "postal savings bank", "中国平安": "ping an insurance",
             "中国人寿": "china life", "中国太保": "china pacific insurance",
             "中信证券": "citic securities", "招商证券": "china merchants securities",
             "国泰君安": "guotai junan"}
TOPICS = {
    "盈利增长": ["盈利增长", "业务结构", "零售", "对公", "资产质量", "盈利", "profit growth"],
    "创新商业模式": ["创新商业模式", "高频生活", "共建", "逾越者联盟", "创新", "business model"],
    "拨备覆盖率": ["拨备覆盖率", "拨备", "provision coverage"],
    "资产质量": ["资产质量", "不良贷款", "逾期贷款", "asset quality"],
    "资本结构": ["资本充足率", "核心一级资本", "内生资本", "资本结构", "capital adequacy"],
    "风险管理": ["风险管理", "风险监测", "智慧风控", "高风险行业", "集中度", "经济下行", "信贷结构", "risk management"],
    "贷款结构": ["贷款结构", "行业结构", "信贷结构", "基础设施", "绿色信贷", "loan structure"],
    "绿色金融": ["绿色金融", "绿色投资", "绿色信贷", "新能源", "green finance"],
    "金融科技": ["金融科技", "科技金融", "数字化", "大数据", "人工智能", "fintech"],
    "投资收益": ["总投资收益率", "投资收益", "investment return"],
    "偿付能力": ["综合偿付能力", "偿付能力", "solvency"],
    "营业收入": ["营业收入", "revenue"],
    "责任投资": ["责任投资", "ESG", "responsible investment"],
}


def normalize(text):
    return re.sub(r"\s+", "", text).lower()


def load_index():
    pages = json.loads((DATA / "pages.json").read_text(encoding="utf-8"))
    texts = [normalize(p["text"]) for p in pages]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=60000)
    matrix = tfidf.fit_transform(texts)
    nodes, edges = {}, []
    for page, text in zip(pages, texts):
        report = page["company"] + "·" + page["year"]
        nodes[report] = {"id": report, "type": "年报"}
        page["topics"] = []
        for topic, words in TOPICS.items():
            if any(normalize(w) in text for w in words if not w.isascii()):
                page["topics"].append(topic)
                nodes[topic] = {"id": topic, "type": "金融主题"}
                edges.append({"source": report, "target": topic, "relation": "原文涉及",
                              "file": page["file"], "page": page["page"]})
    return {"pages": pages, "texts": texts, "tfidf": tfidf, "matrix": matrix,
            "graph": {"nodes": list(nodes.values()), "edges": edges}}


def query_info(question):
    query = question.lower()
    for name, english in COMPANIES.items():
        query = query.replace(english, name)
    for topic, words in TOPICS.items():
        for word in words:
            if word.isascii():
                query = query.replace(word, topic)
    companies = [c for c in COMPANIES if c in query]
    if not companies and "银行和保险" in query:
        companies = list(COMPANIES)[:6]
    topics = [t for t, words in TOPICS.items() if t in query or any(w in query for w in words)]
    return query, companies, topics, re.findall(r"20\d{2}", query)


def retrieve(question, index):
    query, companies, topics, years = query_info(question)
    words = index["tfidf"].transform([normalize(query)])
    scores = (index["matrix"] @ words.T).toarray().ravel()
    # 图谱限定机构/年份，按查询主题扩展；多机构各自取证，避免一家占满结果。
    selected = []
    groups = companies or [None]
    facets = topics if len(topics) > 2 else [None]
    for company in groups:
        for facet in facets:
            choices = []
            for i, page in enumerate(index["pages"]):
                if company and page["company"] != company:
                    continue
                if years and not any(y in page["year"] for y in years):
                    continue
                if topics and not set(topics) & set(page["topics"]):
                    continue
                if facet and facet not in page["topics"]:
                    continue
                text = index["texts"][i]
                terms = [w for t in ([facet] if facet else topics) for w in TOPICS[t] if not w.isascii()]
                score = float(scores[i]) + 0.12 * sum(normalize(w) in text for w in terms)
                fields = ["责任投资", "绿色信贷", "营业收入", "核心一级资本", "拨备覆盖率", "不良贷款率"]
                score += 0.4 * sum(w in query and w in text for w in fields)
                if "营业收入" in query and any(w in text for w in ["主要会计数据", "主要财务数据", "主要财务指标"]):
                    score += 0.8
                if "拨备覆盖率" in query and ("监管指标" in text or re.search(r"拨备覆盖率.{0,40}\d+\.\d+", text)):
                    score += 0.6
                if "董事长" in query and "董事长" in text:
                    score += 0.35
                choices.append((score, i))
            count = 2
            for score, i in sorted(choices, reverse=True)[:count]:
                if any(h["id"] == i for h in selected):
                    continue
                selected.append({**index["pages"][i], "id": i, "score": round(score, 4)})
    # 综合分析题补齐尚未覆盖的主题词，避免几个高分页面重复讲同一件事。
    if len(topics) > 2:
        for term in [w for t in topics for w in TOPICS[t] if not w.isascii() and len(w) >= 3]:
            if any(term in normalize(h["text"]) for h in selected):
                continue
            choices = [(float(scores[i]), i) for i, p in enumerate(index["pages"])
                       if (not companies or p["company"] in companies)
                       and (not years or any(y in p["year"] for y in years)) and term in index["texts"][i]]
            if choices and len(selected) < 48:
                score, i = max(choices)
                selected.append({**index["pages"][i], "id": i, "score": round(score, 4)})
    return selected


def generate(question, hits, model):
    context = "\n".join(f'[{i}] {h["company"]} {h["year"]} PDF第{h["page"]}页\n{h["text"]}'
                        for i, h in enumerate(hits, 1))
    prompt = ("只回答有原文支持的事实；逐条引用[编号]。核对机构、年份、数值、单位，"
              "勿混用收益与收益率。推论明确标为分析。缺证据就说明，不猜测。\n"
              + context[:22000] + "\n问题：" + question)
    if not re.search(r"[\u4e00-\u9fff]", question):
        prompt += "\nAnswer only in English."
    body = json.dumps({"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": 0, "num_predict": 900, "num_ctx": 16384}}).encode()
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat"
    with urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=90) as r:
        return json.loads(r.read())["message"]["content"]


def ask(question, index, model=""):
    started = time.perf_counter()
    if not isinstance(question, str) or len(question.strip()) < 3 or len(question) > 1000:
        raise ValueError("请输入3到1000字的具体问题。")
    hits = retrieve(question, index)
    if model:
        answer = generate(question, hits, model) if hits else "没有找到相符证据。"
        mode = "模型生成"
    else:
        english = not re.search(r"[\u4e00-\u9fff]", question)
        title = "Source extracts (original Chinese annual reports):" if english else "答案原文（摘录式回答）："
        answer = title + "\n\n" + "\n\n".join(f'[{i}] {h["company"]} {h["year"]} PDF第{h["page"]}页\n{h["text"]}'
                                                     for i, h in enumerate(hits, 1))
        mode = "原文摘录"
    return {"answer": answer, "hits": hits, "mode": mode,
            "seconds": round(time.perf_counter() - started, 4)}


app = FastAPI(title="金融图谱问答")
INDEX = None
UPLOAD_LOCK = threading.Lock()


@app.on_event("startup")
def startup():
    global INDEX
    INDEX = load_index()
    (DATA / "graph.json").write_text(json.dumps(INDEX["graph"], ensure_ascii=False), encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "pages": len(INDEX["pages"])}


@app.get("/graph")
def graph():
    return INDEX["graph"]


@app.post("/ask")
def question(body: dict):
    try:
        return ask(body.get("question", ""), INDEX, body.get("model", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(503, "服务暂不可用：" + type(e).__name__)


@app.post("/upload")
async def upload(request: Request, company: str, year: str, name: str = "文档.pdf"):
    global INDEX
    raw = await request.body()
    if not raw or len(raw) > 25 * 1024 * 1024 or not company.strip() or not re.fullmatch(r"20\d{2}", year):
        raise HTTPException(400, "请提供机构、四位年份及25MB以内的PDF。")
    name = Path(name).name
    if Path(name).suffix.lower() != ".pdf":
        raise HTTPException(400, "文件名必须以.pdf结尾。")
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as doc:
            pages = [{"file": name, "company": company, "year": year, "page": i,
                      "text": "\n\n".join(b[4] for b in p.get_text("blocks") if b[6] == 0)}
                     for i, p in enumerate(doc, 1)]
        if not any(p["text"].strip() for p in pages):
            raise ValueError("PDF没有可提取文字，扫描件需先OCR。")
    except Exception as e:
        raise HTTPException(400, "PDF解析失败：" + type(e).__name__)
    with UPLOAD_LOCK:
        old = json.loads((DATA / "pages.json").read_text(encoding="utf-8"))
        temp = DATA / "pages.tmp"
        temp.write_text(json.dumps(old + pages, ensure_ascii=False), encoding="utf-8")
        temp.replace(DATA / "pages.json")
        (DATA / name).write_bytes(raw)
        startup()
    return {"added_pages": len(pages), "total_pages": len(INDEX["pages"])}


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "index.html").read_text(encoding="utf-8")
