# -*- coding: utf-8 -*-
"""
工单02 公共工具库（评测基准 / 索引构建 / 指标计算 / 生成入口）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

本模块被 optimize_chunking.py / optimize_retrieval.py / optimize_prompt.py /
run_full_optimization.py / serve.py 共同引用，集中承载四类能力：

  1. 评分基准 ANSWER_KEYS
     —— 10 个兴图新科问题的「必备要点 / 加分要点 / 证据关键词 / 证据页码 /
        参考答案」，全部逐字核对《招股说明书1.pdf》原文得到，避免评测口径漂移。

  2. 索引构建 build_index
     —— 4 种分块策略各自建立独立的向量库 + BM25 索引，互不干扰；
        带清单（manifest）缓存，重复运行秒级复用，PDF 变更自动失效重建。

  3. 指标计算
     —— 检索层：证据命中率 / 证据 MRR / 证据 Recall@k / 页码命中率
        （单文档语料下 rag_core.evaluate.retrieval_metrics 的「文档级 Hit Rate」
         恒为 1 没有区分度，因此下沉到「证据片段级」统计）；
     —— 答案层：关键词准确率（口径与 rag_core.evaluate.keyword_accuracy 一致，
         但先做标点归一化，避免「15,000」与「15000」这类等价写法被误判）。

  4. 生成 Prompt 与统一问答入口
     —— PROMPT_NAIVE（工单01 基线风格）与 PROMPT_OPTIMIZED（工单02 优化版：
        先定位后作答 / 数值逐字核对 / 表格按行读 / 拒答机制 / 来源标注）；
        answer_question() 是四个实验脚本与 Web 服务共用的问答执行器。

说明：本模块只读取 rag_core，不修改 rag_core 下任何文件。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径准备：把「工单作业/」加入 sys.path，保证脚本可独立直接运行
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent          # 工单02-问答系统检索优化/src
_WO = _SRC.parent                               # 工单02-问答系统检索优化
ROOT = _WO.parent                               # 工单作业
for _p in (str(ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag_core import config, evaluate                        # noqa: E402
from rag_core.bm25 import BM25Retriever                      # noqa: E402
from rag_core.chunk import chunk_blocks                      # noqa: E402
from rag_core.evaluate import EvalRecord, format_summary     # noqa: E402
from rag_core.pdf_parse import ParsedDoc, parse_pdf          # noqa: E402
from rag_core.rerank import TFIDFReranker                    # noqa: E402
from rag_core.retriever import Retriever                     # noqa: E402
from rag_core.vectorstore import VectorStore                 # noqa: E402

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
WO_ID = "人工智能NLP-RAG-基于PDF文档的问答系统优化"       # 工单编号（注释规范要求）

SRC_DIR = _SRC
WO_DIR = _WO
RESULT_DIR = _WO / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = config.PDF_PROSPECTUS_1                # 招股说明书1.pdf（兴图新科）
DOC_NAME = "招股说明书1"
QUESTIONS = config.QUESTIONS_XINGTU               # 工单指定的 10 个问题

# 与 rag_core 保持一致的检索超参
CHUNK_SIZE = config.CHUNK_SIZE                    # 400 字
CHUNK_OVERLAP = config.CHUNK_OVERLAP              # 80 字
RECALL_K = config.TOP_K_RECALL                    # 召回 20 条
TOP_K = config.TOP_K_RERANK                       # 重排后取 5 条

CHUNK_STRATEGIES = ["fixed", "recursive", "semantic", "structure"]


# ---------------------------------------------------------------------------
# 一、评分基准（逐字核对《招股说明书1.pdf》原文后固化）
# ---------------------------------------------------------------------------
# 字段说明：
#   required    —— 答案必须全部命中的关键信息点（决定「准确率」判定）
#   detail      —— 加分信息点（用于覆盖率统计，不参与对错判定）
#   evidence    —— 检索层证据关键词：只要检索结果里出现，就认为「检索到了证据」
#   primary     —— 计算 MRR 用的首要证据词
#   pages       —— 该问题证据在 PDF 中的真实页码（1 起，用于页码命中率与溯源核对）
#   ground_truth—— 参考答案（照抄原文口径，供 RAGAS 类指标使用）
ANSWER_KEYS: dict[int, dict] = {
    260: {
        # 原文（第129页）：「报告期内，公司来自军用领域的收入分别为
        # 6,464.51万元、14,414.16万元、18,780.67万元和4,627.14万元」
        "required": ["6464.51", "14414.16", "18780.67", "4627.14"],
        "detail": ["军用领域的收入", "报告期"],
        "evidence": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"],
        "primary": "6,464.51",
        "pages": [129],
        "ground_truth": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别为"
                        "6,464.51万元、14,414.16万元、18,780.67万元和4,627.14万元。",
    },
    33: {
        # 原文（第129页）：「占主营业务收入比重分别为82.10%、97.31%、94.84%和94.34%」
        "required": ["82.10", "97.31", "94.84", "94.34"],
        "detail": ["主营业务收入", "比重"],
        "evidence": ["82.10%", "97.31%", "94.84%", "94.34%"],
        "primary": "82.10%",
        "pages": [129],
        "ground_truth": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占"
                        "主营业务收入的比重分别为82.10%、97.31%、94.84%和94.34%。",
    },
    95: {
        # 原文（第26/95页）：「参与制定了全军第一个视频指挥系统技术标准
        # （即《某视频技术规范1.0》）」
        "required": ["视频指挥系统技术标准"],
        "detail": ["某视频技术规范", "全军第一个"],
        "evidence": ["全军第一个视频指挥系统技术标准", "某视频技术规范"],
        "primary": "全军第一个视频指挥系统技术标准",
        "pages": [26, 95],
        "ground_truth": "武汉兴图新科电子股份有限公司参与制定了全军第一个视频指挥系统"
                        "技术标准，即《某视频技术规范1.0》。",
    },
    34: {
        # 原文（第152页）：「电子信息行业的上游涉及信息系统相关的电子元器件制造企业，
        # 以及机箱、机柜等金属壳体制造企业」
        "required": ["电子元器件"],
        "detail": ["金属壳体", "机箱", "机柜"],
        "evidence": ["电子元器件制造企业", "金属壳体制造企业"],
        "primary": "电子元器件制造企业",
        "pages": [152],
        "ground_truth": "电子信息行业的上游涉及信息系统相关的电子元器件制造企业，"
                        "以及机箱、机柜等金属壳体制造企业。",
    },
    957: {
        # 原文（第26/154页）：「兴图新科目前已经成为国防军队视频指挥领域的重要供应商」
        "required": ["视频指挥领域"],
        "detail": ["重要供应商", "国防军队"],
        "evidence": ["视频指挥领域的重要供应商"],
        "primary": "视频指挥领域的重要供应商",
        "pages": [26, 154],
        "ground_truth": "武汉兴图新科电子股份有限公司（兴图新科）目前已经成为国防军队"
                        "视频指挥领域的重要供应商。",
    },
    793: {
        # 原文（第152页）：「下游行业为各类终端用户，覆盖范围广泛，主要包括
        # 军队、政府机关、能源等行业企业」
        "required": ["政府机关"],
        "detail": ["军队", "能源", "终端用户"],
        "evidence": ["政府机关、能源等行业企业", "下游行业为各类终端用户"],
        "primary": "政府机关、能源等行业企业",
        "pages": [152],
        "ground_truth": "电子信息行业的下游为各类终端用户，覆盖范围广泛，主要包括"
                        "军队、政府机关、能源等行业企业。",
    },
    795: {
        # 原文（第94/96页）：「2014年12月，某大型研究所牵头承担的“某情报、指挥、
        # 控制与通信网络一体化工程”（即相当于美军的C4ISR系统）荣获国家科技进步一等奖」
        "required": ["一体化工程"],
        "detail": ["C4ISR", "情报、指挥、控制与通信网络", "2014年12月"],
        "evidence": ["一体化工程", "C4ISR"],
        "primary": "一体化工程",
        "pages": [94, 96],
        "ground_truth": "2014年12月，某大型研究所牵头承担的“某情报、指挥、控制与通信"
                        "网络一体化工程”（即相当于美军的C4ISR系统）荣获国家科技进步"
                        "一等奖，兴图新科是该工程网络化视频指挥系统的唯一参与者。",
    },
    543: {
        # 原文（第52页）：「注册资本：5,520万元」；摘要（第22页）：「注册资本5,520.00万元」
        "required": ["5520"],
        "detail": ["注册资本"],
        "evidence": ["注册资本：5,520万元", "注册资本5,520.00万元"],
        "primary": "注册资本：5,520万元",
        "pages": [22, 52],
        "ground_truth": "武汉兴图新科电子股份有限公司注册资本为5,520.00万元。",
    },
    531: {
        # 原文（第52页）：「法定代表人：程家明」
        "required": ["程家明"],
        "detail": ["法定代表人"],
        "evidence": ["法定代表人：程家明"],
        "primary": "法定代表人：程家明",
        "pages": [22, 52],
        "ground_truth": "武汉兴图新科电子股份有限公司法定代表人为程家明。",
    },
    207: {
        # 原文（第30/479页）：募集资金投资项目「3 补充流动资金 15,000.00」万元
        "required": ["15000"],
        "detail": ["补充流动资金"],
        "evidence": ["补充流动资金15,000.00", "补充流动资金15,000.0015,000.00"],
        "primary": "补充流动资金15,000.00",
        "pages": [30, 479],
        "ground_truth": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金"
                        "15,000.00万元用于补充流动资金。",
    },
}


# ---------------------------------------------------------------------------
# 二、文本归一化与答案判定
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """
    评测归一化：去掉千分位逗号、全角逗号、顿号、空白，统一小写。

    招股书里的数字常写成「15,000.00万元」，而模型可能输出「15000万元」；
    不归一化会把等价写法误判为错误答案，因此判分前统一处理。
    """
    if not text:
        return ""
    out = str(text).lower()
    for ch in (",", "，", "、", " ", "　", "\t", "\n", "\r", "（", "）", "(", ")"):
        out = out.replace(ch, "")
    return out


def check_answer(answer: str, qid: int) -> dict:
    """判定单个答案是否命中全部必备要点，返回明细。"""
    keys = ANSWER_KEYS[qid]
    norm_ans = normalize(answer)
    hits = [k for k in keys["required"] if normalize(k) in norm_ans]
    misses = [k for k in keys["required"] if normalize(k) not in norm_ans]
    detail_hits = [k for k in keys["detail"] if normalize(k) in norm_ans]
    return {
        "id": qid,
        "正确": len(misses) == 0,
        "命中": hits,
        "漏答": misses,
        "加分命中": detail_hits,
        "覆盖率": round(len(detail_hits) / max(len(keys["detail"]), 1), 3),
    }


def keyword_accuracy_norm(records: list[EvalRecord]) -> dict:
    """
    关键词准确率（归一化版）。

    复用 rag_core.evaluate.keyword_accuracy 的统计口径（全部要点命中才算对），
    只是先把答案与要点都做标点归一化后再比对。
    """
    copies: list[EvalRecord] = []
    for r in records:
        c = EvalRecord(**{**r.__dict__})          # 浅拷贝，避免污染原始记录
        c.answer = normalize(r.answer)
        copies.append(c)
    keys = {str(qid): [normalize(k) for k in v["required"]]
            for qid, v in ANSWER_KEYS.items()}
    return evaluate.keyword_accuracy(copies, keys)


def answer_details(records: list[EvalRecord]) -> list[dict]:
    """逐题给出判定明细（用于 Markdown 报告）。"""
    return [check_answer(r.answer, int(r.qid)) for r in records]


# ---------------------------------------------------------------------------
# 三、PDF 解析与索引构建
# ---------------------------------------------------------------------------
def _pdf_mtime() -> float:
    try:
        return round(Path(PDF_PATH).stat().st_mtime, 3)
    except OSError:
        return 0.0


def parse_document(with_tables: bool = False, with_images: bool = False,
                   use_cache: bool = True, verbose: bool = True) -> ParsedDoc:
    """
    解析《招股说明书1.pdf》，带两级容错：

      1. 优先命中 rag_core 的 JSON 解析缓存（稳定、快）；
      2. 首次解析失败（缓存损坏 / 环境异常）时清除缓存并重试一次，
         仍失败才抛出异常 —— 对应工单「文档解析失败」的容错要求。
    """
    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(
            f"未找到源 PDF：{PDF_PATH}\n"
            f"请确认工单附件《招股说明书1.pdf》已放在该路径下。"
        )
    if verbose:
        print(f"[1] 解析《{DOC_NAME}》…")
    try:
        parsed = parse_pdf(PDF_PATH, DOC_NAME, with_tables=with_tables,
                           with_images=with_images, use_cache=use_cache)
    except Exception as e:                                   # 容错：清缓存重试
        print(f"  [warn] 解析失败（{e}），清理解析缓存后重试一次…")
        for f in config.CACHE_DIR.glob(f"parsed_{DOC_NAME}_*.json"):
            try:
                f.unlink()
            except OSError:
                pass
        parsed = parse_pdf(PDF_PATH, DOC_NAME, with_tables=with_tables,
                           with_images=with_images, use_cache=False)

    if verbose:
        print(f"    {parsed.n_pages} 页，{len(parsed.blocks)} 个解析块")
    return parsed


def make_chunks(parsed: ParsedDoc, strategy: str,
                size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """按策略分块（semantic 策略的签名只有 size/threshold，单独处理）。"""
    if strategy == "semantic":
        # 语义分块按「相邻句向量相似度骤降点」断开，不使用重叠参数
        return chunk_blocks(parsed.blocks, strategy="semantic", size=size)
    return chunk_blocks(parsed.blocks, strategy=strategy, size=size, overlap=overlap)


def bm25_path(collection: str) -> Path:
    """每个 collection 独立一份 BM25 索引，避免互相覆盖。"""
    return config.INDEX_DIR / f"{collection}_bm25.pkl"


def index_manifest_path(collection: str) -> Path:
    return config.INDEX_DIR / f"{collection}.wo02.json"


def build_index(collection: str, strategy: str,
                size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                force: bool = False, verbose: bool = True,
                parsed: ParsedDoc | None = None) -> dict:
    """
    为指定分块策略建立「向量库 + BM25 索引」，返回索引统计信息。

    缓存策略：清单文件记录 (策略, size, overlap, PDF mtime, 块数)，
    与当前请求一致且向量库条数吻合时直接复用，实现重复运行秒级启动；
    任何一项不匹配（改了参数 / 换了 PDF / 索引被删）都会自动重建。
    """
    eff_overlap = None if strategy == "semantic" else overlap
    manifest_path = index_manifest_path(collection)

    if not force and manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            same = (m.get("strategy") == strategy and m.get("size") == size
                    and m.get("overlap") == eff_overlap
                    and m.get("pdf_mtime") == _pdf_mtime()
                    and m.get("n_chunks") == VectorStore(collection).count())
            if same:
                if verbose:
                    print(f"    [cache] 复用索引 {collection}：{m['n_chunks']} 块")
                return m
        except Exception as e:                       # 清单损坏 -> 重建，不影响主流程
            print(f"    [warn] 索引清单损坏（{e}），将重建 {collection}")

    t0 = time.perf_counter()
    if parsed is None:
        parsed = parse_document(verbose=verbose)
    chunks = make_chunks(parsed, strategy, size=size, overlap=overlap)

    # 向量库：先清空再写入，保证幂等（重跑不会重复累积）
    vs = VectorStore(collection)
    vs.reset()
    vs.add_chunks(chunks, show_progress=False)

    # BM25 倒排索引（混合检索与级联重排的 IDF 统计都依赖它）
    bm25 = BM25Retriever()
    bm25.build(chunks)
    bm25.save(bm25_path(collection))

    lens = sorted(len(c.text) for c in chunks) or [0]
    types: dict[str, int] = {}
    for c in chunks:
        types[c.type] = types.get(c.type, 0) + 1

    manifest = {
        "collection": collection,
        "strategy": strategy,
        "size": size,
        "overlap": eff_overlap,
        "n_chunks": len(chunks),
        "avg_chunk_chars": round(sum(lens) / len(lens), 1),
        "median_chunk_chars": lens[len(lens) // 2],
        "max_chunk_chars": lens[-1],
        "min_chunk_chars": lens[0],
        "chunk_types": types,
        "n_vectors": vs.count(),
        "build_seconds": round(time.perf_counter() - t0, 2),
        "pdf_mtime": _pdf_mtime(),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"    索引完成：{len(chunks)} 块，平均 {manifest['avg_chunk_chars']} 字，"
              f"耗时 {manifest['build_seconds']}s")
    return manifest


def get_retriever(collection: str, reranker: str = "none",
                  verbose: bool = True) -> Retriever:
    """
    构造检索器，并把重排器在「全量语料」上拟合好。

    注意：rag_core.retriever 只会自动为 CascadeReranker 拟合 IDF；
    TFIDFReranker 需要手动 fit，否则 IDF 恒为常数、数字/实体加权失效。
    """
    ret = Retriever(collection, bm25_path=bm25_path(collection))
    try:
        ret.load_bm25()
    except Exception as e:
        print(f"    [warn] BM25 装载失败（{e}），混合检索将退化为纯向量")
    rr = ret.get_reranker(reranker)
    if isinstance(rr, TFIDFReranker) and ret.bm25 is not None:
        rr.fit(ret.bm25.docs)
    if verbose:
        print(f"    检索器就绪：{collection} / 重排={reranker}"
              f"{'（BM25 未就绪）' if ret.bm25 is None else ''}")
    return ret


# ---------------------------------------------------------------------------
# 四、指标计算
# ---------------------------------------------------------------------------
def retrieval_evidence_metrics(results: list[dict], k: int = TOP_K) -> dict:
    """
    检索层指标（证据片段级，不依赖 LLM，纯计算）。

    Args:
        results: [{"qid": 260, "docs": [检索结果…]}, …]
        k: 只看前 k 条

    Returns:
        evidence_hit_rate  —— 前 k 条里出现过任一证据关键词的题目占比
        evidence_mrr       —— 首个命中证据的排名倒数均值（越大越靠前）
        evidence_recall    —— 前 k 条覆盖了多少比例的必备必备证据词（信息全不全）
        page_hit_rate      —— 前 k 条里有落在证据页（±1 页容差）的题目占比
    """
    n = 0
    hit = mrr = recall = page_hit = 0.0
    details: list[dict] = []

    for r in results:
        qid = int(r["qid"])
        keys = ANSWER_KEYS[qid]
        ev = [normalize(x) for x in keys["evidence"]]
        docs = (r.get("docs") or [])[:k]
        texts = [normalize(d.get("text", "")) for d in docs]
        pages = [int(d.get("page") or 0) for d in docs]
        blob = "\n".join(texts)

        covered = [e for e in ev if e in blob]
        first_rank = 0
        for i, t in enumerate(texts, 1):
            if any(e in t for e in ev):
                first_rank = i
                break
        pg_ok = any(abs(p - gp) <= 1 for p in pages for gp in keys["pages"])

        n += 1
        if covered:
            hit += 1
        if first_rank:
            mrr += 1.0 / first_rank
        recall += len(covered) / max(len(ev), 1)
        page_hit += 1.0 if pg_ok else 0.0

        details.append({
            "id": qid,
            "证据命中": bool(covered),
            "首个命中排名": first_rank or None,
            "证据覆盖": f"{len(covered)}/{len(ev)}",
            "命中页码": sorted(set(p for p in pages
                                   if any(abs(p - gp) <= 1 for gp in keys["pages"]))),
            "gold_pages": keys["pages"],
        })

    return {
        "n": n,
        "evidence_hit_rate": round(hit / n, 4) if n else 0.0,
        "evidence_mrr": round(mrr / n, 4) if n else 0.0,
        "evidence_recall_at_k": round(recall / n, 4) if n else 0.0,
        "page_hit_rate": round(page_hit / n, 4) if n else 0.0,
        "k": k,
        "details": details,
    }


def latency_stats(values: list[float]) -> dict:
    """耗时统计（秒）：均值 / P50 / P95 / 最大值。"""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}

    def _pct(q: float) -> float:
        idx = min(int(len(vals) * q), len(vals) - 1)
        return vals[idx]

    return {
        "avg": round(sum(vals) / len(vals), 3),
        "p50": round(_pct(0.50), 3),
        "p95": round(_pct(0.95), 3),
        "max": round(vals[-1], 3),
        "n": len(vals),
    }


# ---------------------------------------------------------------------------
# 五、生成 Prompt（朴素 vs 优化）与统一问答入口
# ---------------------------------------------------------------------------
# 基线（工单01）使用的朴素 Prompt：只说「根据资料回答」，没有任何约束
PROMPT_NAIVE_SYS = "你是一个问答助手，请根据用户提供的资料回答问题。"
PROMPT_NAIVE_USER = "【资料】\n{ctx}\n\n【问题】\n{q}\n\n请回答上面的问题。"

# 工单02 优化 Prompt：四条硬规则 —— 先定位后作答 / 数值逐字核对 /
# 表格按行读 / 上下文不足则拒答，并要求标注来源便于溯源
PROMPT_OPT_SYS = """你是一个严谨的金融文档问答助手，服务于招股说明书等专业文档的问答。

【作答规则】
1. 只依据【参考资料】作答，绝不使用参考资料之外的任何知识，不做常识性推测。
2. 先定位后作答：先确认所需信息出现在哪一个片段，再组织答案，不要凭印象作答。
3. 数值逐字核对：金额、比例、年份、股数必须与参考资料逐字一致，保留原始单位与口径；
   若资料给出多期数据，按期间分行/分项列出，绝不把不同年份或不同口径的数字混在一起。
4. 表格类资料按行读取，注意表头与数据行的对应关系，不要把列串行。
5. 若参考资料不足以回答问题，直接回复「根据提供的文档内容，未能找到该问题的答案。」，
   不要编造，也不要输出「可能」「大概」等猜测。
6. 回答简洁：先给结论，再列支撑细节，不要复述问题。
7. 结尾用【来源】标注引用的片段编号与页码，格式：来源：片段2（第58页）。"""

PROMPT_OPT_USER = "【参考资料】\n{ctx}\n\n【问题】\n{q}\n\n请严格依据参考资料作答。"

# 英文版（工单验收项「多语言支持」：英文提问用英文作答）
PROMPT_NAIVE_SYS_EN = "You are a QA assistant. Answer the question based on the reference material."
PROMPT_NAIVE_USER_EN = "【Reference】\n{ctx}\n\n【Question】\n{q}\n\nPlease answer the question."
PROMPT_OPT_SYS_EN = """You are a rigorous financial-document QA assistant (prospectuses, annual reports).

Rules:
1. Answer ONLY from the reference material. Never use outside knowledge, never guess.
2. Locate first: identify which excerpt contains the answer before writing it.
3. Verify every figure character by character; keep original units and periods;
   list multi-period data item by item instead of merging different years.
4. Read tables row by row; keep headers aligned with data rows.
5. If the references are insufficient, reply exactly:
   "Based on the provided documents, the answer to this question was not found."
   Never fabricate.
6. Be concise: conclusion first, then supporting details.
7. End with a 【Source】 line citing excerpt numbers and page numbers."""
PROMPT_OPT_USER_EN = "【Reference】\n{ctx}\n\n【Question】\n{q}\n\nAnswer strictly from the reference."

PROMPTS = {
    "naive": (PROMPT_NAIVE_SYS, PROMPT_NAIVE_USER, PROMPT_NAIVE_SYS_EN, PROMPT_NAIVE_USER_EN),
    "optimized": (PROMPT_OPT_SYS, PROMPT_OPT_USER, PROMPT_OPT_SYS_EN, PROMPT_OPT_USER_EN),
}


def is_english_question(text: str) -> bool:
    """中文占比低于 15% 视为英文提问（与 rag_core.generator 判定口径一致）。"""
    if not text:
        return False
    zh = sum(1 for c in text if "一" <= c <= "鿿")
    return zh / max(len(text), 1) < 0.15


def build_messages(question: str, context_text: str, style: str = "optimized"):
    """按风格与提问语言拼装消息列表，供 llm.chat 调用。"""
    sys_zh, user_zh, sys_en, user_en = PROMPTS.get(style, PROMPTS["optimized"])
    if is_english_question(question):
        sys_p, user_p = sys_en, user_en
    else:
        sys_p, user_p = sys_zh, user_zh
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_p.format(ctx=context_text, q=question)},
    ]


def generate_answer(question: str, context_text: str, style: str = "optimized",
                    model: str | None = None, max_tokens: int = 1500):
    """
    调用 LLM 生成答案，返回 (答案文本, 耗时秒)。

    容错：生成失败（网络异常 / 未配置 Key）时不抛异常打断整批实验，
    而是返回「服务繁忙」提示并记录耗时，保证批量评测可以跑完
    —— 对应工单「高可用性、容错机制」要求。
    """
    from rag_core import llm

    if not context_text.strip():
        return "根据提供的文档内容，未能找到该问题的答案。", 0.0

    t0 = time.perf_counter()
    try:
        answer = llm.chat(build_messages(question, context_text, style),
                          model=model, temperature=0.0,
                          max_tokens=max_tokens, tag=f"wo02_{style}")
        return answer.strip(), time.perf_counter() - t0
    except Exception as e:
        return f"（生成失败：{e}）", time.perf_counter() - t0


def require_llm_key() -> None:
    """需要调用 LLM 的实验脚本先做一次显式检查，给出可操作的提示。"""
    if not config.DEEPSEEK_API_KEY:
        raise SystemExit(
            "未检测到环境变量 DEEPSEEK_API_KEY，无法调用生成/重排模型。\n"
            "请先设置：set DEEPSEEK_API_KEY=sk-xxx  （Windows CMD）\n"
            "          $env:DEEPSEEK_API_KEY='sk-xxx'（PowerShell）"
        )


@dataclass
class QAConfig:
    """一次问答的完整配置（检索 + 重排 + 生成）。"""
    name: str
    collection: str
    strategy: str = "vector"          # vector | fulltext | hybrid
    fusion: str = "rrf"               # hybrid 时的融合算法
    reranker: str = "none"            # none | tfidf | llm | cascade
    prompt: str = "naive"             # naive | optimized
    recall_k: int = RECALL_K
    top_k: int = TOP_K
    alpha: float = config.HYBRID_ALPHA

    def to_dict(self) -> dict:
        return {
            "name": self.name, "collection": self.collection,
            "strategy": self.strategy, "fusion": self.fusion,
            "reranker": self.reranker, "prompt": self.prompt,
            "recall_k": self.recall_k, "top_k": self.top_k, "alpha": self.alpha,
        }


def answer_question(cfg: QAConfig, question: str, retriever: Retriever | None = None,
                    model: str | None = None, generate: bool = True) -> dict:
    """
    执行一次完整问答：检索 → 重排 → 生成，返回答案与分阶段耗时。

    Args:
        generate: False 时只跑检索链路（不调用生成模型），
                  用于「只评测检索质量、不消耗 token」的场景。

    返回字段：answer / docs / contexts / retrieval_seconds / generate_seconds /
              total_seconds / timings / refused / error
    """
    ret = retriever or get_retriever(cfg.collection, cfg.reranker, verbose=False)
    out = {
        "question": question, "answer": "", "docs": [], "contexts": [],
        "retrieval_seconds": 0.0, "generate_seconds": 0.0,
        "total_seconds": 0.0, "timings": {}, "refused": False, "error": "",
    }
    t_all = time.perf_counter()

    # ---- 1. 检索（含重排）----
    try:
        res = ret.retrieve(question, strategy=cfg.strategy, top_k=cfg.top_k,
                           recall_k=cfg.recall_k, reranker=cfg.reranker,
                           fusion=cfg.fusion, alpha=cfg.alpha)
        docs = res.docs
        out["timings"] = {k: round(v, 4) for k, v in res.timings.items()}
        out["retrieval_seconds"] = round(res.timings.get("total", 0.0), 4)
    except Exception as e:                       # 容错：检索异常 -> 空上下文 + 拒答
        out["error"] = f"检索失败：{e}"
        print(f"    [warn] {out['error']}（问题：{question[:24]}…）")
        docs = []

    out["docs"] = docs
    out["contexts"] = [d.get("text", "") for d in docs]

    # ---- 2. 生成 ----
    if generate:
        ctx = ret.format_context(docs) if docs else ""
        answer, gen_s = generate_answer(question, ctx, style=cfg.prompt, model=model)
        out["answer"] = answer
        out["generate_seconds"] = round(gen_s, 4)
        out["refused"] = ("未能找到该问题" in answer) or ("was not found" in answer)
    out["total_seconds"] = round(time.perf_counter() - t_all, 4)
    return out


def make_record(qid: int, question: str, result: dict, k: int = TOP_K) -> EvalRecord:
    """把一次问答结果转成 rag_core 的 EvalRecord（供评估与落盘）。"""
    docs = result.get("docs") or []
    keys = ANSWER_KEYS[int(qid)]
    rec = EvalRecord(
        qid=qid,
        question=question,
        answer=result.get("answer", ""),
        ground_truth=keys["ground_truth"],
        contexts=result.get("contexts", [])[:k],
        reference_doc=DOC_NAME,
        retrieved_docs=[d.get("doc", "") for d in docs[:k]],
        retrieved_pages=[int(d.get("page") or 0) for d in docs[:k]],
        latency=result.get("total_seconds", 0.0),
    )
    # 分阶段耗时写入 metrics，随 to_dict() 一起落盘（不参与均值汇总）
    rec.metrics["retrieval_seconds"] = result.get("retrieval_seconds", 0.0)
    rec.metrics["generate_seconds"] = result.get("generate_seconds", 0.0)
    return rec


# ---------------------------------------------------------------------------
# 六、结果落盘工具
# ---------------------------------------------------------------------------
def save_json(name: str, obj: dict) -> Path:
    """把实验结果写入 results/<name>.json。"""
    path = RESULT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    -> 已写出 {path}")
    return path


def _disp_width(s) -> int:
    """中文按 2 列宽计算，保证 Markdown 表格在等宽字体下对齐。"""
    w = 0
    for ch in str(s):
        w += 2 if ("一" <= ch <= "鿿" or ch in "（）【】《》：，、％·—") else 1
    return w


def md_table(headers: list, rows: list[list]) -> str:
    """生成对齐的 Markdown 表格。"""
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [max([_disp_width(h)] + [_disp_width(r[i]) for r in rows])
              for i, h in enumerate(headers)]
    def _line(cells):
        return "| " + " | ".join(
            c + " " * (widths[i] - _disp_width(c)) for i, c in enumerate(cells)) + " |"
    out = [_line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [_line(r) for r in rows]
    return "\n".join(out)


def save_md(name: str, text: str) -> Path:
    """把 Markdown 报告写入 results/<name>.md。"""
    path = RESULT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"    -> 已写出 {path}")
    return path


def update_doc_block(doc_path: Path, tag: str, content: str) -> bool:
    """
    把实测结果回填到文档中的标记块：
        <!-- AUTO:<tag>:BEGIN -->  ……  <!-- AUTO:<tag>:END -->
    这样《优化方案.md》里的对比表不必手抄，跑完脚本即是最新实测数据；
    文档不存在或没有标记时静默跳过，不影响实验流程。
    """
    doc_path = Path(doc_path)
    if not doc_path.exists():
        return False
    begin, end = f"<!-- AUTO:{tag}:BEGIN -->", f"<!-- AUTO:{tag}:END -->"
    text = doc_path.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        return False
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    doc_path.write_text(f"{head}{begin}\n{content}\n{end}{tail}", encoding="utf-8")
    print(f"    -> 已回填实测数据到 {doc_path.name} [{tag}]")
    return True


def summary_to_text(summary: dict) -> str:
    """rag_core 的汇总指标转人读文本。"""
    return format_summary(summary)


def print_stage(title: str) -> None:
    """统一的阶段标题输出。"""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
