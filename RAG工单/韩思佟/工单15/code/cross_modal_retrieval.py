# -*- coding: utf-8 -*-
"""
工单15 —— 跨模态检索优化模块（可独立运行，也可作为 RAGFlow 补丁落地）

实现四项优化（对应工单15「任务二」）：
  ① 查询理解：识别"第11页""图3""编号13的部件"等视觉引用
  ② 三路召回：原始查询 / 图像增强查询 / 结构化过滤  → RRF 融合
  ③ 跨模态 Re-Ranker：把图纸描述拼进候选文本后精排
  ④ 提示工程：含图纸上下文时，Prompt 明确指示"结合技术图纸描述分析"

同时给出对接 RAGFlow 的落地位置说明（见文件末尾 PATCH_GUIDE）。
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.llm import get_llm                                     # noqa: E402
from src.vlm import describe_with_vision_api                    # noqa: E402


# ================================================================ ① 查询理解
VISUAL_REF_PATTERNS = {
    "page": r"第\s*(\d+)\s*页",
    "figure": r"图\s*(\d+)",
    "part": r"(?:部件|编号|零件|元件)\s*([0-9]{1,3}(?:['′\"]|，)?)",
    "figref": r"(?:图示|图纸|图\s*中)",
}


@dataclass
class VisualRef:
    pages: List[int] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    parts: List[str] = field(default_factory=list)
    has_visual_ref: bool = False

    def to_dict(self):
        return self.__dict__.copy()


def extract_visual_ref(question: str) -> VisualRef:
    """从问题中抽取页码/图号/部件编号"""
    ref = VisualRef()
    ref.pages = [int(x) for x in re.findall(VISUAL_REF_PATTERNS["page"], question)]
    ref.figures = re.findall(VISUAL_REF_PATTERNS["figure"], question)
    ref.parts = [p.strip("，'′\"") for p in
                 re.findall(VISUAL_REF_PATTERNS["part"], question)]
    ref.has_visual_ref = bool(ref.pages or ref.figures
                              or re.search(VISUAL_REF_PATTERNS["figref"], question))
    return ref


def build_enhanced_query(question: str, ref: VisualRef,
                         figure_desc: str = "") -> str:
    """把视觉线索拼进检索查询"""
    parts = [question]
    clue = []
    if ref.pages:
        clue.append("第" + "、".join(str(p) for p in ref.pages) + "页")
    if ref.figures:
        clue.append("图" + "、".join(ref.figures))
    if ref.parts:
        clue.append("部件 " + "、".join(ref.parts))
    if clue:
        parts.append("[视觉线索] " + " ".join(clue))
    if figure_desc:
        parts.append("[图纸描述] " + figure_desc[:800])
    return "\n".join(parts)


# ================================================================ ② RRF 融合
def rrf_fuse(ranked_lists: List[List[str]], k: int = 60,
             weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion（RAGFlow 原生只有 weighted_sum，本补丁补上 RRF）"""
    weights = weights or [1.0] * len(ranked_lists)
    score: Dict[str, float] = {}
    for w, lst in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(lst, 1):
            score[doc_id] = score.get(doc_id, 0.0) + w * 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def weighted_fuse(score_lists: List[Dict[str, float]], weights: List[float]) -> List[Tuple[str, float]]:
    """加权分数融合（各自归一化后加权平均）"""
    agg: Dict[str, float] = {}
    for scores, w in zip(score_lists, weights):
        if not scores:
            continue
        mx = max(abs(v) for v in scores.values()) or 1.0
        for k_, v in scores.items():
            agg[k_] = agg.get(k_, 0.0) + w * (v / mx)
    return sorted(agg.items(), key=lambda x: -x[1])


# ================================================================ ③ 跨模态重排
class CrossModalReranker:
    """
    把「候选文本 + 图纸描述」拼成一个 pair 再送 rerank 模型。
    这样"第11页图3""编号13"这类视觉引用就能参与相关性打分。
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._m = None

    def _model(self):
        if self._m is None:
            from sentence_transformers import CrossEncoder
            from src import config
            self._m = CrossEncoder(self.model_path or config.RERANK_MODEL_PATH,
                                   max_length=512, device=config.EMBEDDING_DEVICE)
        return self._m

    def rerank(self, query: str, cands: List[dict], top_k: int = 5) -> List[dict]:
        try:
            m = self._model()
            import torch
            pairs = []
            for c in cands:
                txt = (c.get("text") or "")[:400]
                desc = (c.get("image_description") or "")[:300]
                pairs.append((query, f"{txt}\n[图纸] {desc}" if desc else txt))
            scores = m.predict(pairs, activation_fn=torch.nn.Identity())
            for c, s in zip(cands, scores):
                c["rerank_score"] = float(s)
        except Exception as e:                                   # noqa: BLE001
            for c in cands:
                c["rerank_score"] = float(c.get("score", 0.0))
            print("[warn] CrossEncoder 不可用，回退原分数：", e)
        cands.sort(key=lambda x: -x.get("rerank_score", 0))
        return cands[:top_k]


# ================================================================ ④ 提示工程
CROSS_MODAL_PROMPT = """你是一个工业技术文档问答助手。请只依据下面的资料回答问题。

【重要】资料中若包含以「[图纸]」开头的技术图纸解析描述，
请务必结合图纸中的部件编号、位置关系、气流/传动方向进行推理，
不要只依赖文字描述，也不要忽略图纸信息。

规则：
1. 材料中没有的信息回答"根据提供的资料无法确定"。
2. 涉及部件编号、位置关系时必须与图纸描述一致。
3. 分点作答，并在句末标注依据（页码/图号）。

===== 资料开始 =====
{context}
===== 资料结束 =====

问题：{question}
回答："""


# ================================================================ 跨模态检索器
def locate_figure_pages(question: str, store, max_pages: int = 3):
    """
    优化⑥：按**图题名称**定位图片页。

    很多图文题并不会写「第X页」，只会说「组织结构图中…」「2008年中国IC市场应用结构与增长图中…」。
    这时靠「首轮检索命中的图片块最多的一页」来猜页码非常不可靠（实测猜成了 117 / 83 页）。
    更稳的做法：把图题名（以「图」结尾的中文/英文/数字串）抽出来，回到库里去匹配
    图题文字（招股书的图题与正文都会出现图名），命中最多的那几页就是图片页。

    返回 (页码列表, 命中的图题关键词)
    """
    import re
    from collections import Counter
    kws = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,18}图", question)
    kws = [k for k in kws if len(k) >= 3]
    extra = []
    for k in kws:
        # 图题在正文里常写成「……增长（亿元）」这种不带「图」字的样子，
        # 所以同时准备「去掉尾巴的图字」与「最后 4 字」两个变体
        if len(k) >= 5:
            extra.append(k[:-1])
        if len(k) >= 6:
            extra.append(k[-4:])
    kws = list(dict.fromkeys(kws + extra))
    if not kws:
        return [], []
    cnt = Counter()
    # ★ 关键：PDF 抽出来的文本里会夹空格（"2008 年，中国IC 市场"），
    #   直接子串匹配必然失败，所以两边都先去空白再比。
    compact = lambda s: re.sub(r"\s+", "", s)                    # noqa: E731
    kws_c = [compact(k) for k in kws]
    for c in store.chunks:
        t = compact(c.text)
        if any(k in t for k in kws_c):
            cnt[c.page] += 1
    if not cnt:
        return [], kws
    pages = [p for p, _ in cnt.most_common(max_pages)]
    # 图文常常跨页（如第 38~39 页是同一张组织结构图），把命中的相邻页也带上
    neighbors = set()
    img_pages = {c.page for c in store.chunks if c.kind in ("image", "table")}
    for p in pages:
        for q in (p - 1, p + 1):
            if q in img_pages:
                neighbors.add(q)
    return sorted(set(pages) | neighbors), kws


class CrossModalRetriever:
    """在本地知识库上演示三路召回 + RRF + 跨模态重排"""

    def __init__(self, retriever, pdf_path: str | Path | None = None,
                 use_vision: bool = True, mode: str = "cross_modal"):
        """
        mode = "cross_modal"（优化后）：查询理解 + 三路召回 RRF + 跨模态重排
        mode = "baseline"   （优化前）：只用原始问题做一次混合检索，无视觉通道
        """
        self.r = retriever                     # src.retriever.Retriever
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.use_vision = use_vision
        self.mode = mode
        self.last_desc_source = ""
        self.reranker = CrossModalReranker()

    def _retrieve_baseline(self, question: str, top_k: int) -> dict:
        """优化前：普通混合检索（向量+BM25+RRF 粗排 + 文本重排），完全不看图"""
        hits = self.r.retrieve(question, strategy="hybrid", top_k=top_k,
                               do_rerank=True)
        cands = [{
            "cid": h.chunk.cid, "doc": h.chunk.doc, "page": h.chunk.page,
            "kind": h.chunk.kind, "text": h.chunk.text,
            "score": round(float(h.score), 5), "image_description": "",
        } for h in hits]
        return {"visual_ref": {"has_visual_ref": False, "pages": [], "markers": []},
                "enhanced_query": question,
                "figure_description": "",
                "candidates": cands}

    def _figure_description(self, page: int) -> str:
        """渲染该页 → 视觉模型读图（与 RAGFlow 的 VisionFigureParser 等价）"""
        if not (self.use_vision and self.pdf_path):
            return ""
        try:
            import fitz
            cache = ROOT / "cache" / "render"
            cache.mkdir(parents=True, exist_ok=True)
            img = cache / f"{self.pdf_path.stem}_p{page}_160.png"
            if not img.exists():
                doc = fitz.open(str(self.pdf_path))
                doc[page - 1].get_pixmap(dpi=160).save(str(img))
                doc.close()
            return describe_with_vision_api(
                img, "请按顺序转录图中所有可见文字（部件编号、尺寸标注、图题、箭头方向）。").text
        except Exception as e:                                   # noqa: BLE001
            # 视觉模型不可用（余额耗尽 / 断网）→ 降级：用该页正文文本作为图纸描述的代理。
            # 说明：招股书/专利的图文页正文本身就包含图题与部件名称，
            #      作为「无视觉模型时的等价输入」可让跨模态链路继续跑通。
            try:
                texts = [c.text for c in self.r.store.chunks
                         if c.page == page and c.kind == "text" and len(c.text) > 60]
                if texts:
                    self.last_desc_source = "page_text_fallback"
                    return max(texts, key=len)[:800]
            except Exception:                                    # noqa: BLE001
                pass
            self.last_desc_source = f"vision_unavailable: {e}"
            return ""

    def retrieve(self, question: str, top_k: int = 5) -> dict:
        if self.mode == "baseline":
            return self._retrieve_baseline(question, top_k)

        ref = extract_visual_ref(question)

        # --- 通道 A：原始查询（文本语义 + 全文）
        raw_hits = self.r.retrieve(question, strategy="hybrid", top_k=10,
                                   do_rerank=False)
        A = {h.chunk.cid: h for h in raw_hits}
        ranked_A = [h.chunk.cid for h in raw_hits]

        # --- 页码定位（分层：显式页码 > 图题名称匹配 > 图片块最多页兜底）
        page_source = "explicit" if ref.pages else ""
        if ref.has_visual_ref and not ref.pages:
            pages_found, kws = locate_figure_pages(question, self.r.store)
            if pages_found:
                ref.pages = pages_found
                page_source = "figure_name_match"
                print(f"[页码定位] 图题 {kws} → 第 {pages_found} 页")
        if ref.has_visual_ref and not ref.pages:
            # 兜底：首轮检索里图片/表格块最集中的页。
            # 注意：这个推断并不可靠（实测会猜错），所以只用来扩大候选，
            # **不参与下面的「页码命中保底」**。
            from collections import Counter
            img_pages = Counter(h.chunk.page for h in raw_hits
                                if h.chunk.kind in ("image", "table"))
            if img_pages:
                ref.pages = [img_pages.most_common(1)[0][0]]
                page_source = "inferred"
        self.last_page_source = page_source

        # --- 通道 B：图像增强查询（问题 + 图纸描述）
        fig_desc = ""
        if ref.has_visual_ref and ref.pages:
            fig_desc = self._figure_description(ref.pages[0])
        enhanced = build_enhanced_query(question, ref, fig_desc)
        b_hits = self.r.retrieve(enhanced, strategy="vector", top_k=10,
                                 do_rerank=False)
        # 只保留图像/表格类块，强化"图文块"召回
        b_hits_img = [h for h in b_hits if h.chunk.kind in ("image", "table")] or b_hits
        B = {h.chunk.cid: h for h in b_hits_img}
        ranked_B = [h.chunk.cid for h in b_hits_img]

        # --- 通道 C：结构化过滤（页码命中）
        C, ranked_C = {}, []
        if ref.pages:
            for c in self.r.store.chunks:
                if c.page in ref.pages:
                    C[c.cid] = c
            ranked_C = [c.cid for c in self.r.store.chunks if c.page in ref.pages][:20]

        # --- 融合
        pool = {**A, **B}
        fuse_ids = rrf_fuse([ranked_A, ranked_B, ranked_C], k=60, weights=[1.0, 1.2, 1.5])
        cands = []
        for cid, s in fuse_ids[:20]:
            ch = pool.get(cid) or C.get(cid)
            if ch is None:
                continue
            chunk = ch.chunk if hasattr(ch, "chunk") else ch
            cands.append({
                "cid": chunk.cid, "doc": chunk.doc, "page": chunk.page,
                "kind": chunk.kind, "text": chunk.text, "score": round(s, 5),
                "image_description": "",
            })

        # --- 给候选补上图纸描述（跨模态重排的输入）
        pages_with_img = {c["page"] for c in cands if c["kind"] in ("image", "table")}
        desc_cache = {}
        for pg in list(pages_with_img)[:3]:
            desc_cache[pg] = self._figure_description(pg)
        for c in cands:
            if c["page"] in desc_cache:
                c["image_description"] = desc_cache[c["page"]]
            elif fig_desc and c["page"] in ref.pages:
                c["image_description"] = fig_desc

        # --- 跨模态重排
        cands = self.reranker.rerank(enhanced, cands, top_k=top_k)

        # --- 优化⑦：页码命中保底（关键召回保障）
        # 只在页码**有可靠来源**时生效（问题里显式写了「第X页」，或图题名称在库里匹配到了图页）；
        # 用「图片块最多页」猜出来的页码不可靠，不参与保底，否则会把正确答案挤掉（实测过）。
        if ref.pages and getattr(self, "last_page_source", "") in ("explicit",
                                                                 "figure_name_match"):
            pinned = [c for c in cands
                      if c["page"] in ref.pages and c["kind"] in ("image", "table")]
            if not pinned:
                # 候选里没有该页图文块时，直接从库里补进来（最多 2 个）
                for ch in self.r.store.chunks:
                    if (ch.page in ref.pages and ch.kind in ("image", "table")
                            and len(pinned) < 2):
                        pinned.append({
                            "cid": ch.cid, "doc": ch.doc, "page": ch.page,
                            "kind": ch.kind, "text": ch.text, "score": 0.0,
                            "image_description": desc_cache.get(ch.page, ""),
                        })
            rest = [c for c in cands if c not in pinned]
            cands = (pinned + rest)[:top_k]

        return {"visual_ref": ref.to_dict(), "enhanced_query": enhanced,
                "figure_description": fig_desc[:1500],
                "desc_source": self.last_desc_source,
                "candidates": cands}


def answer(question: str, retriever: CrossModalRetriever, top_k: int = 5):
    res = retriever.retrieve(question, top_k)
    ctx_parts = []
    for i, c in enumerate(res["candidates"], 1):
        head = f"[{i}] {c['doc']} 第{c['page']}页（{c['kind']}）"
        body = c["text"][:800]
        if c["image_description"]:
            body += "\n[图纸] " + c["image_description"][:600]
        ctx_parts.append(f"{head}\n{body}")
    prompt = CROSS_MODAL_PROMPT.format(context="\n\n".join(ctx_parts), question=question)
    ans = get_llm().ask(prompt, temperature=0.0).text
    return ans, res


PATCH_GUIDE = """
===================== 落地到 RAGFlow 的补丁位置 =====================
1) 查询理解（视觉引用识别）
   新增 rag/nlp/visual_ref.py（本文件的 extract_visual_ref / build_enhanced_query）
   调用点：api/db/services/dialog_service.py::chat() 中
           retriever.retrieval(...) 之前，把 enhanced_query 作为检索输入。

2) 三路召回 + RRF
   新增 rag/nlp/fusion.py（本文件的 rrf_fuse / weighted_fuse）
   调用点：rag/nlp/search.py::Dealer.retrieval()，在原有 weighted_sum 结果之上
           再跑一次 RRF/加权融合。RAGFlow 原生 FusionExpr 只有 weighted_sum
           （见 rag/utils/es_conn.py:83），本补丁补齐 RRF。

3) 跨模态 Re-Ranker
   改写 rag/nlp/search.py::Dealer.rerank_by_model()
   把 pair 文本从 " ".join(tks) 改成 " ".join(tks) + "\\n[图纸]" + image_description。

4) 提示工程
   新增 rag/prompts/cross_modal_qa.md（本文件的 CROSS_MODAL_PROMPT）
   在 rag/prompts/generator.py 注册，对话配置里加开关。

5) 配置外部化
   新增 conf/cross_modal.json：
   {"enabled": true, "fusion": "rrf", "rrf_k": 60,
    "channel_weights": [1.0, 1.2, 1.5], "image_context_size": 300,
    "vision_model": "qwen-vl-max", "top_k": 5}
===================================================================
"""
