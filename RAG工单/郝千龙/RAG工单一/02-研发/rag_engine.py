# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【RAG主引擎 · rag_engine.py】编排"入库/混合检索/重排/生成/反馈/知识库管理"全流程
# 编写日期：2026-09-28   修订日期：2026-10-04
import asyncio
import copy
import json
import re
import time
import datetime as dt
from typing import List, Dict, Optional

import config
import pdf_parser
import text_splitter
from embedder import Embedder
from vector_store import VectorStore
from cache import Cache
from llm_client import LLMClient
from bm25_retriever import BM25Retriever
from reranker import Reranker

# 精选概念头（长词优先）：仅对这些招股书属性/概念做定义结构匹配，避免普通词噪声
_DEFINE_HEADS = [
    "补充流动资金", "法定代表人", "注册资本", "实收资本", "注册地址", "成立日期",
    "公司类型", "经营范围", "实际控制人", "控股股东", "主营业务", "募集资金",
    "总股本", "注册地", "上下游", "上游", "下游",
]
_DEFINE_BOOST = 0.02  # 强定义匹配加分（RRF 分数量级约 0.03，足以提升数名）


def _definition_boost(question: str, fused: List[Dict]) -> None:
    """
    定义枚举结构先验（确定性、无网络）
    仅对问题中出现的"精选概念头"匹配两类精确结构：
      ① 概念头 + 紧邻引导词(主要包括/主要涉及/为/是/：…) + 非平凡值或枚举；
      ② 概念头 + 紧邻数字（标签-数值型表格，如"补充流动资金 15,000.00"）。
    命中则给融合分加分，把字面定义块/被切块拆散的数值块提升进 Top-K，
    同时因概念头精选、结构收紧，避免普通"是/为"句带来的噪声与排序回退。
    """
    heads = [h for h in _DEFINE_HEADS if h in question]
    for c in fused:
        body = c["text"]
        for h in heads:
            esc = re.escape(h)
            # ① 定义/枚举：头后 0-8 字符内出现引导词，再接实质内容
            define_pat = (
                esc + r"[^。\n]{0,8}"
                + r"(?:主要包括|主要涉及|包括|涉及|是指|是|为|：|:)"
                + r"[^。\n]{2,80}"
            )
            # ② 标签紧邻数值：头后 0-6 字符内直接出现数字
            value_pat = esc + r"[^。\n]{0,6}\d"
            if re.search(define_pat, body) or re.search(value_pat, body):
                c["fusion_score"] = float(c.get("fusion_score", 0)) \
                    + _DEFINE_BOOST
                break  # 每块最多加一次


# 表格列/行标签：文本块末尾停在这些标签，说明行数据被切到了下一块
_TABLE_TAIL_LABELS = (
    "金额", "占比", "项目", "类型", "年度", "月份", "合计", "小计",
    "名称", "序号", "单价", "数量",
)
_TAIL_LABEL_RE = re.compile(r"(?:" + "|".join(_TABLE_TAIL_LABELS) + r")$")


def _complete_dangling_tables(selected: List[Dict]) -> List[Dict]:
    """
    表格悬空续写（确定性、无网络）
    当入选块末尾停在表格标签（如"金额/占比/项目"）时，判定该表被定长切块拆散，
    追加文档顺序上紧邻的下一块（要求含数字，确为数据行），补全被截断的数值。
    典型作用：把"国防领域 4,627.14 94.34% …"等金额续写块并入上下文，
    解决表格数值与表头分家导致的事实遗漏。
    """
    if not selected:
        return selected
    all_chunks = VectorStore.get_all_chunks()
    idx_of = {c["chunk_uid"]: i for i, c in enumerate(all_chunks)}
    out = list(selected)
    existing = {c["chunk_id"] for c in out}
    for c in selected:
        stripped = c["text"].strip()
        tail = stripped.splitlines()[-1].strip() if stripped else ""
        if not _TAIL_LABEL_RE.search(tail):
            continue  # 块尾不是悬空表格标签
        i = idx_of.get(c["chunk_id"])
        if i is None or i + 1 >= len(all_chunks):
            continue
        nxt = all_chunks[i + 1]
        if nxt["chunk_uid"] in existing or not re.search(r"\d", nxt["text"]):
            continue  # 已存在或续块不含数字则不追加
        out.append({
            "text": nxt["text"], "page": nxt["page"],
            "doc_id": nxt["doc_id"], "chunk_id": nxt["chunk_uid"],
            "score": c.get("score", 0),
        })
        existing.add(nxt["chunk_uid"])
    return out

# 文档级元信息文件（doc_id → 名称/页数/块数/入库时间）
DOC_META_FILE = config.DATA_DIR / "doc_meta.json"


# ====================================================================
# 一、离线入库
# ====================================================================
def ingest(pdf_path: Optional[str] = None, doc_id: str = "prospectus1",
           doc_name: str = None) -> Dict:
    """
    离线入库流水线：PDF → 解析 → 切分 → 向量化 → 向量库 → BM25 重建
    返回 {doc_id, pages, chunks}
    """
    pdf_path = pdf_path or str(config.SOURCE_PDF)
    doc_name = doc_name or pdf_path.split("/")[-1].split("\\")[-1]
    print(f"[STEP1] 解析 PDF: {pdf_path}")
    pages = pdf_parser.parse_pdf(pdf_path, cache=True)

    print(f"[STEP2] 切分文本")
    chunks = text_splitter.split_pages(pages)

    print(f"[STEP3] 向量化 {len(chunks)} 个文本块（{config.EMBED_MODEL_NAME}）")
    embed = Embedder()
    vecs = embed.encode([c["text"] for c in chunks])

    print(f"[STEP4] 写入向量库（{VectorStore.backend_name()}）")
    metas = [
        {"chunk_id": c["chunk_id"], "text": c["text"],
         "page": c["page"], "doc_id": doc_id}
        for c in chunks
    ]
    n = VectorStore.insert(vecs, metas)

    print(f"[STEP5] 重建 BM25 索引")
    BM25Retriever().rebuild_from_store()

    # 记录文档级元信息
    _save_doc_meta(doc_id, doc_name, len(pages), n)
    print(f"[DONE] 入库完成: {doc_name}, {len(pages)} 页, {n} 块")
    return {"doc_id": doc_id, "pages": len(pages), "chunks": n, "name": doc_name}


def _save_doc_meta(doc_id: str, name: str, pages: int, chunks: int) -> None:
    """写入/更新文档元信息"""
    metas = _load_doc_meta()
    now = dt.datetime.now().isoformat(timespec="seconds")
    item = {"doc_id": doc_id, "name": name, "pages": pages,
            "chunks": chunks, "ingest_at": now}
    metas = [m for m in metas if m["doc_id"] != doc_id]
    metas.append(item)
    DOC_META_FILE.write_text(
        json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_doc_meta() -> List[Dict]:
    """读取文档元信息"""
    if DOC_META_FILE.exists():
        try:
            return json.loads(DOC_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


# ====================================================================
# 二、混合检索（向量 + BM25 → RRF → Rerank）
# ====================================================================
def _rrf_fuse(list_a: List[Dict], list_b: List[Dict]) -> List[Dict]:
    """
    Reciprocal Rank Fusion：按两路排名倒数求和融合
    score = Σ 1/(RRF_K + rank)，rank 从 1 开始
    返回融合去重后的候选，按融合分降序
    """
    pool: Dict[str, Dict] = {}
    for rank, hits in enumerate((list_a, list_b), start=1):
        for i, h in enumerate(hits, start=1):
            uid = h["chunk_id"]
            if uid not in pool:
                pool[uid] = dict(h)
                pool[uid]["fusion_score"] = 0.0
            pool[uid]["fusion_score"] += 1.0 / (config.RRF_K + i)
    return sorted(pool.values(), key=lambda x: x["fusion_score"], reverse=True)


# 招股书章节锚点规则：问题命中关键词 → 检索时追加对应章节名以聚焦召回
_CHAPTER_RULES = [
    (["募集资金", "募投", "补充流动资金", "募集", "投资项目"], "募集资金运用"),
    (["法定代表人", "注册资本", "注册地址", "注册地", "成立日期",
      "公司类型", "经营范围", "控股股东", "实际控制人", "股本", "发起人",
      "统一社会信用"], "发行人基本情况"),
    (["风险"], "风险因素"),
    (["净利润", "营业收入", "毛利率", "现金流", "资产负债", "利润表",
      "财务", "审计", "净资产收益率", "偿债"], "财务会计信息"),
    (["上游", "下游", "技术标准", "科技进步", "供应商", "军用",
      "主营业务", "核心技术", "客户", "收入", "产品", "行业", "重要供应商"],
     "业务和技术"),
    (["股利", "分红", "利润分配"], "股利分配"),
]


def _chapter_anchor(question: str) -> str:
    """
    章节感知查询扩展：识别问题所属招股书章节，返回需追加的章节锚点词
    短问句（如"法定代表人是谁"）信息稀疏，追加章节锚点后可聚焦到
    "发行人基本情况"等章节，显著提升事实类问题的两路召回排名
    """
    for keywords, anchor in _CHAPTER_RULES:
        if any(k in question for k in keywords):
            return anchor
    return ""


# 查询同义词归一化：工单口语用词 → 招股书规范用词（追加到检索查询，提升词法命中）
_QUERY_SYNONYMS = {
    "军用领域": ["国防领域"],
    "军用": ["国防"],
}


def _query_synonyms(question: str) -> List[str]:
    """抽取问题中需补充的招股书同义词（如"军用"→追加"国防"）"""
    extras: List[str] = []
    for term, syns in _QUERY_SYNONYMS.items():
        if term in question:
            extras.extend(syns)
    return extras


def retrieve(question: str, recall_num: int = None,
             rerank: bool = None) -> List[Dict]:
    """
    混合检索：查询扩展 → 向量召回 + BM25 召回 → RRF 融合 → 可选 Rerank → Top-K
    返回 [{text,page,doc_id,chunk_id,score,...}, ...]，score 统一归一化到 [0,1]
    """
    recall_num = recall_num or config.RECALL_NUM
    embed = Embedder()

    # 查询扩展：同义词归一化 + 章节锚点（仅用于检索，不改变用户原始问题）
    parts = [question] + _query_synonyms(question)
    anchor = _chapter_anchor(question)
    if anchor:
        parts.append(anchor)
    search_q = " ".join(dict.fromkeys(parts))  # 去重保序
    q_vec = embed.encode_one(search_q)

    # 路径1：稠密向量语义召回
    vec_hits = VectorStore.search(q_vec, top_k=recall_num)
    # 路径2：BM25 关键词召回（公司名/年份/数字）
    bm_hits = []
    if config.ENABLE_BM25:
        bm_hits = BM25Retriever().search(search_q, top_n=config.BM25_TOP_N)

    # RRF 融合；若 BM25 不可用则直接用向量结果
    fused = _rrf_fuse(vec_hits, bm_hits) if bm_hits else vec_hits

    # 定义枚举结构先验：提升字面定义/枚举块（无网络、确定性），再按融合分重排
    if fused and "fusion_score" in fused[0]:
        _definition_boost(question, fused)
        fused.sort(key=lambda x: x["fusion_score"], reverse=True)

    # Rerank 精排（超时自动降级）
    if rerank is None:
        rerank = config.ENABLE_RERANK
    if rerank:
        fused = Reranker().rerank(question, fused)

    # 统一 score 字段：归一化到 [0,1]，避免泄漏无界的 BM25 原始分
    if fused:
        if rerank and "rerank_score" in fused[0]:
            # 重排启用：对 rerank 原始分做 min-max 归一化
            raw = [float(h["rerank_score"]) for h in fused]
            lo, hi = min(raw), max(raw)
            for h, v in zip(fused, raw):
                h["score"] = round((v - lo) / (hi - lo + 1e-9), 4)
        else:
            # 未重排：以最大 RRF 融合分为 1.0 归一化
            hi = max(float(h.get("fusion_score", h.get("score", 0))) for h in fused)
            for h in fused:
                h["score"] = round(
                    float(h.get("fusion_score", h.get("score", 0)))
                    / (hi + 1e-9), 4)
    top = fused[: config.RETRIEVE_TOP_K]
    # 表格悬空续写：补全被定长切块截断的数值行（可能使返回块数略多于 Top-K）
    return _complete_dangling_tables(top)


# ====================================================================
# 三、在线问答（同步/异步）
# ====================================================================
def ask(question: str, top_k: int = None, use_cache: bool = True,
        lang: str = "zh") -> Dict:
    """
    同步问答：返回 {answer, refs, latency_ms, cache_hit, stages}
    stages 记录各阶段毫秒耗时，便于性能分析
    """
    t0 = time.time()
    stages: Dict[str, int] = {}

    # 1. 缓存命中（深拷贝返回，避免修改缓存内存对象污染原始延迟/答案）
    if use_cache:
        cached = Cache.get_answer(question, top_k or config.RETRIEVE_TOP_K, lang)
        if cached:
            out = copy.deepcopy(cached)
            out["cache_hit"] = True
            out["latency_ms"] = int((time.time() - t0) * 1000)
            return out

    # 2. 混合检索
    t1 = time.time()
    hits = retrieve(question)
    stages["retrieve_ms"] = int((time.time() - t1) * 1000)

    # 3. LLM 生成
    t2 = time.time()
    contexts = [(h["text"], h["page"]) for h in hits]
    answer = LLMClient().answer_with_context(question, contexts, lang=lang)
    stages["llm_ms"] = int((time.time() - t2) * 1000)

    refs = [{"page": h["page"], "score": round(float(h.get("score", 0)), 4),
             "text": h["text"], "doc_id": h["doc_id"]} for h in hits]
    payload = {
        "answer": answer,
        "refs": refs,
        "cache_hit": False,
        "latency_ms": int((time.time() - t0) * 1000),
        "stages": stages,
        "lang": lang,
    }
    if use_cache:
        Cache.set_answer(question, top_k or config.RETRIEVE_TOP_K, payload, lang)
    return payload


async def ask_async(question: str, top_k: int = None, use_cache: bool = True,
                    lang: str = "zh") -> Dict:
    """异步问答：检索放线程池避免阻塞事件循环，LLM 走异步客户端"""
    t0 = time.time()

    if use_cache:
        cached = Cache.get_answer(question, top_k or config.RETRIEVE_TOP_K, lang)
        if cached:
            out = copy.deepcopy(cached)
            out["cache_hit"] = True
            out["latency_ms"] = int((time.time() - t0) * 1000)
            return out

    # CPU 密集的检索放线程池执行
    hits = await asyncio.to_thread(retrieve, question)
    contexts = [(h["text"], h["page"]) for h in hits]
    answer = await LLMClient().aanswer_with_context(question, contexts, lang=lang)

    refs = [{"page": h["page"], "score": round(float(h.get("score", 0)), 4),
             "text": h["text"], "doc_id": h["doc_id"]} for h in hits]
    payload = {
        "answer": answer,
        "refs": refs,
        "cache_hit": False,
        "latency_ms": int((time.time() - t0) * 1000),
        "lang": lang,
    }
    if use_cache:
        Cache.set_answer(question, top_k or config.RETRIEVE_TOP_K, payload, lang)
    return payload


def ask_baseline(question: str, lang: str = "zh") -> Dict:
    """同步基线：跳过检索，纯 LLM 回答"""
    t0 = time.time()
    answer = LLMClient().answer_baseline(question, lang=lang)
    return {"answer": answer, "refs": [], "cache_hit": False,
            "latency_ms": int((time.time() - t0) * 1000), "lang": lang}


async def ask_baseline_async(question: str, lang: str = "zh") -> Dict:
    """异步基线"""
    t0 = time.time()
    answer = await LLMClient().aanswer_baseline(question, lang=lang)
    return {"answer": answer, "refs": [], "cache_hit": False,
            "latency_ms": int((time.time() - t0) * 1000), "lang": lang}


# ====================================================================
# 四、反馈机制
# ====================================================================
def save_feedback(question: str, answer: str, rating: str,
                  comment: str = "") -> Dict:
    """
    保存用户反馈（👍 up / 👎 down + 文字）
    追加写入 feedback.jsonl，支持知识库持续改进
    """
    assert rating in ("up", "down"), "rating 必须为 up/down"
    record = {
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
    }
    with config.FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def list_feedback(limit: int = 100) -> List[Dict]:
    """读取反馈列表"""
    if not config.FEEDBACK_FILE.exists():
        return []
    lines = config.FEEDBACK_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines[-limit:]]


# ====================================================================
# 五、知识库管理
# ====================================================================
def list_documents() -> Dict:
    """列出全部已入库文档（元信息 + 实时块数）"""
    docs = _load_doc_meta()
    for d in docs:
        d["chunks_now"] = VectorStore.count()
    return {"documents": docs, "total_chunks": VectorStore.count(),
            "vector_backend": VectorStore.backend_name(),
            "cache_backend": Cache.backend_name()}


def delete_document(doc_id: str) -> Dict:
    """删除文档：向量库按 doc_id 删除 → BM25 重建 → 更新元信息"""
    VectorStore.delete_doc(doc_id)
    BM25Retriever().rebuild_from_store()
    metas = [m for m in _load_doc_meta() if m["doc_id"] != doc_id]
    DOC_META_FILE.write_text(
        json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"deleted": doc_id}


def health_check() -> Dict:
    """健康检查"""
    return {
        "status": "healthy",
        "chunks": VectorStore.count(),
        "cache": Cache.health(),
        "vector_backend": VectorStore.backend_name(),
    }


# ====================================================================
# 六、CLI 入口
# ====================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_engine.py [ingest|ask <q>|docs]")
        sys.exit(0)
    if sys.argv[1] == "ingest":
        ingest()
    elif sys.argv[1] == "ask":
        print(json.dumps(ask(sys.argv[2]), ensure_ascii=False, indent=2))
    elif sys.argv[1] == "docs":
        print(json.dumps(list_documents(), ensure_ascii=False, indent=2))

# ====================================================================
# 技术备注：
# 1. RAG：本模块编排 RAG 全链路——多路召回（语义+词法）→ RRF 融合
#    → CrossEncoder 重排 → LLM 生成 → 缓存，兼顾召回率与精度。
# 2. RRF 无需调参、对两路评分尺度不敏感，是工业界常用的结果融合算法。
# 3. 异步接口 + 线程池隔离 CPU 任务，显著提升高并发吞吐。
# 4. Transformer / Fine-tuning：检索与生成两端均为 Transformer，
#    任一侧微调后本引擎无需改动，仅需重新入库/部署。
# ====================================================================
