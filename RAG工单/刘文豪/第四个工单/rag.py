# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的图像内容解析及检索优化
RAG 核心：文本索引（工单03 全量集合，路径见 index_path.txt）+ 图像内容索引（本工单）双路检索
两路结果按相似度合并，提示词中说明图像块内容来自 OCR 与图像描述
LLM 与嵌入均通过本机 Ollama 官方 SDK（固定端点 127.0.0.1:11434）
"""
import re
from pathlib import Path

import chromadb
import ollama

IMG_CHROMA_DIR = "chroma_db"
IMG_COLLECTION = "zhaogu2_img_v4"
TEXT_COLLECTIONS = ["zhaogu1_v3", "zhaogu2_v3"]  # 工单03 按文档拆分的两个文本集合
TOP_K = 6
MIN_SCORE = 0.30
GEN_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

DOC_ALIAS = {"zhaogu1": "招股说明书1（兴图新科）", "zhaogu2": "招股说明书2（力源信息）"}

PROMPT_TMPL = """你是证券文档问答助手。规则：
1. 仅根据【检索上下文】回答【问题】，不得编造；
2. 只要上下文中出现了与问题对应的内容，就必须直接给出答案，禁止以"不够全面"为由拒答；
3. 上下文确实完全没有答案时，才回答"根据文档内容未找到"；
4. 标注【图像内容】的片段来自对文档图表页的 OCR 识别与图像描述，图表中的结构、数值是权威依据，
   组织结构图中上下层级关系按图中标签的从属顺序理解；
   统计"销售处"个数时，仅统计名称以"销售处"结尾的部门框（"XX分公司"不属于销售处）；
   OCR 常把 IC 误识别为 1C、把 O 误识别为 0，解读时自行纠正；
   比较增长率时，逐一核对图中所有行业的数值，增长率最快=数值最大的行业；
   若正文写明某行业"位列第二，仅次于X"，则增长率最快的就是X；
5. 表格中的数值是权威答案；
6. 问题包含多个小问时，必须逐一回答，不得只答其中一个；
7. 回答末尾标注出处，格式如（招股说明书2 第X页/图）。

【检索上下文】
{context}

【问题】{question}

【回答】"""


def get_collections():
    text_dir = Path("index_path.txt").read_text(encoding="utf-8").strip()
    client = chromadb.PersistentClient(path=text_dir)
    text_cols = [client.get_collection(c) for c in TEXT_COLLECTIONS]
    img_col = chromadb.PersistentClient(path=IMG_CHROMA_DIR).get_or_create_collection(
        IMG_COLLECTION, metadata={"hnsw:space": "cosine"})
    return text_cols, img_col


def clean_query(question: str) -> str:
    q = re.sub(r"根据?.{0,20}?招股(说明书|意向书)[，,]?", "", question)
    for name in ("武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司"):
        q = q.replace(name, "")
    q = q.replace("报告期内，", "")
    return q.strip("，, 。") or question


def embed_query(text: str):
    return ollama.embed(model=EMBED_MODEL, input=[text])["embeddings"][0]


def retrieve(text_cols, img_col, question: str, top_k: int = TOP_K):
    """三路检索：两份招股书文本集合 + 图像内容集合。
    图像块固定保留 Top-2（图像类问题的关键依据，不参与分数竞争），文本块按相似度取 Top-K。"""
    vec = embed_query(clean_query(question))
    hits = []
    for col in text_cols:
        res = col.query(query_embeddings=[vec], n_results=top_k)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = round(1 - dist, 4)
            if score >= MIN_SCORE:
                hits.append({"text": doc, "doc": meta["doc"], "page": meta["page"],
                             "kind": meta.get("kind", "text"), "score": score})
    hits.sort(key=lambda h: -h["score"])
    hits = hits[:top_k - 2]
    res = img_col.query(query_embeddings=[vec], n_results=3)
    img_hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = round(1 - dist, 4)
        if score >= 0.25:
            img_hits.append({"text": doc, "doc": meta["doc"], "page": meta["page"],
                             "kind": "image", "score": score})
    img_hits.sort(key=lambda h: -h["score"])
    return hits + img_hits[:2]


def llm_generate(prompt: str) -> str:
    resp = ollama.generate(model=GEN_MODEL, prompt=prompt, stream=False,
                           options={"temperature": 0.1, "seed": 42})
    return resp["response"].strip()


def answer(text_cols, img_col, question: str, top_k: int = TOP_K):
    hits = retrieve(text_cols, img_col, question, top_k)
    # 图像块放上下文最前：图像类问题的关键依据优先被模型看到
    hits.sort(key=lambda h: 0 if h["kind"] == "image" else 1)
    ctx = "\n\n".join(
        f"[{DOC_ALIAS[h['doc']]} 第{h['page']}页{' 图表' if h['kind']=='image' else ' 表格' if h['kind']=='table' else ''}] {h['text']}"
        for h in hits)
    prompt = PROMPT_TMPL.format(context=ctx, question=clean_query(question))
    ans = llm_generate(prompt)
    return ans, hits
