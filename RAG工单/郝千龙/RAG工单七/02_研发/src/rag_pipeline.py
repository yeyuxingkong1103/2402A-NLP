# -*- coding: utf-8 -*-
# 【RAG问答流水线 · rag_pipeline.py】PDF解析分块 + TF-IDF/BM25混合检索(RRF) + 抽取式作答 + 不可回答拒答
# 工单编号：人工智能NLP-RAG-功能测试及评估

"""RAG 流水线主体（沿用 01-06 工单“PDF解析→分块→双路召回→抽取作答”架构思路）。

链路：
1. 解析：PyMuPDF 逐页提取文本，清洗页眉页脚/页码/硬换行；
2. 分块：按页贪心打包为定长字符块（块间重叠），保留页码与来源文档；
3. 索引：jieba 中文切词 → sklearn TF-IDF 稠密打分 + 自研 BM25 稀疏打分；
4. 混合检索：两路 Top-N 经 RRF 倒数排名融合取并集，再以归一化线性分
   + 查询词覆盖度 + 题型结构信号（工商概况卡/枚举清单/金额邻近）精排；
5. 作答：证据块同时产出“整句（硬换行拼回）”与“表格行/卡片字段行”两类
   候选，按 IDF 覆盖度/长短语/类型吻合/父块锚点打分；数字型问题按问点
   数量合并最多 2 个互补数字证据；金融常见槽位（法定代表人/注册资本/
   保荐机构/补充流动资金）使用与具体公司无关的通用正则槽位；
6. 拒答：以“查询词加权覆盖度 + 高IDF专有词命中率”双信号判定（融合分
   经实测对越界问题无区分度，不参与拒答），枚举题额外允许“锚点+清单
   结构”挽救；不满足即不编造答案，返回统一拒答话术并置 ``refused=true``。

本模块纯 CPU、离线可运行，不依赖任何 LLM 接口。
"""
import math
import pickle
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset_loader import Document

# ---------------------------------------------------------------------------
# 一、配置项（阈值集中管理，04_优化阶段有前后指标对比）
# ---------------------------------------------------------------------------

CHUNK_SIZE = 360          # 单块目标字符数
CHUNK_OVERLAP = 80        # 相邻块重叠字符数
MIN_CHUNK_LEN = 12        # 过短块丢弃
RECALL_N = 20             # 单路召回数量
FINAL_TOP_K = 6           # 对外输出的证据块数量
ANSWER_POOL_K = 10        # 实际参与作答抽取的证据池（清单表可能排在第6位之后）
RRF_K = 60                # RRF 常数（越大名次差异越平滑）
# 融合权重：TF-IDF 与 BM25 为主，覆盖度与结构信号为辅
W_TFIDF, W_BM25, W_COVER = 0.42, 0.42, 0.08
# 拒答阈值（基于10题首轮实测信号在04_优化阶段标定）
REFUSAL_COVERAGE_MIN = 0.30   # 查询词加权覆盖度阈值
REFUSAL_RARE_TERM_MIN = 0.40  # 高IDF专有词命中率阈值（与覆盖度同时满足才作答）
# 统一拒答话术（不编造语料中不存在的信息）
REFUSAL_TEXT = "根据现有招股书资料无法回答该问题：未在资料中检索到足够的相关依据，为避免编造信息，拒绝作答。"

# 中文问答停用词（对定位无区分度的泛化词）
STOPWORDS = {
    "根据", "按照", "招股", "招股书", "招股说明书", "招股意向书", "意向书",
    "说明书", "股份有限公司", "有限公司", "公司", "企业", "本次", "报告期",
    "报告期内", "分别", "多少", "哪些", "哪个", "什么样", "请问", "一下",
    "以及", "对于", "关于", "相关", "主要", "发行人", "发行", "上市",
    "是", "的", "了", "在", "和", "与", "或", "为", "有", "不", "对",
    "中", "该", "其", "用", "于", "哪", "么", "怎", "何", "谁", "呢",
    "吗", "这", "那", "个", "项", "次", "将", "已", "均", "各类", "各种",
    "时", "后", "前", "本", "如何", "怎么", "怎样",
}

# 金融招股语料自定义词（避免被错误切分导致召回漂移）
USER_DEFINED_WORDS = [
    "法定代表人", "注册资本", "注册地址", "实收资本", "成立日期",
    "补充流动资金", "募集资金", "募集资金用途", "募集资金投资项目",
    "保荐人", "保荐机构", "主承销商", "发行股数", "发行后总股本",
    "总股本", "营业收入", "视频指挥控制系统", "视频预警控制系统",
    "视音频", "研发中心建设项目",
    "基于云联邦架构的军用视频指挥平台升级及产业化项目",
    "中泰证券股份有限公司", "武汉兴图新科电子股份有限公司",
    "在职员工", "员工人数", "员工总数", "研发人员", "前五名客户",
    "实际控制人", "控股股东", "高新技术", "科创板",
]
for _w in USER_DEFINED_WORDS:
    jieba.add_word(_w)

# 查询词项的“拆词别名”：长术语被PDF排版/同义字段拆散时，按别名组判定命中
# 键为查询中的长术语，值为若干“与关系”词组（检索时把别名扁平加入召回）
TERM_ALIASES = {
    "募集资金投资项目": (("募集资金",), ("项目", "用途", "投资")),
}


def _alias_constituents(terms: Sequence[str]) -> List[str]:
    """收集查询词项对应的拆词别名（扁平去重）。

    :param terms: 原始查询词项
    :return: 别名词项列表
    """
    extra: List[str] = []
    for t in dict.fromkeys(terms):
        for groups in TERM_ALIASES.get(t, ()):  # type: ignore[arg-type]
            for word in groups:
                if word not in extra:
                    extra.append(word)
    return extra

# 页眉/页脚噪声正则
_PAGE_MARK_RE = re.compile(r"1\s*-\s*\d+\s*-\s*\d+")
_PURE_DIGIT_RE = re.compile(r"^\d{1,3}$")
# 整句切分（先把硬换行拼回，按句读切；用于正文句子候选）
_SENT_SPLIT_RE = re.compile(r"[^。；;！？!?]+[。；;！？!?]?")
# 数字（含千分位、小数、百分号、年份）
_NUMBER_RE = re.compile(r"\d{4}\s*年|\d[\d,]*(?:\.\d+)?\s*(?:%|万元|亿元|万股|元|人|项|个|名)?")
_COMPANY_RE = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司)")
# 工商概况卡字段
_CARD_FIELDS = ("法定代表人", "注册地址", "实收资本", "成立日期", "控股股东",
                "实际控制人", "股份公司成立", "行业分类", "经营范围", "股票简称")
# 多问点数量题的问点标记
_QUANT_MARK_RE = re.compile(r"多少|几|比例|占比")


def compact(text: str) -> str:
    """去除全部空白（PDF表格与硬换行常把同一词拆开，如“视频预警控制 系统”）。

    :param text: 原始文本
    :return: 无空白文本
    """
    return re.sub(r"\s+", "", text)


def display_clean(text: str) -> str:
    """答案展示清洗：仅合并中文与中文/中文与数字之间的断词空格，保留正常间隔。

    :param text: 原始候选答案
    :return: 适合展示的答案文本
    """
    text = re.sub(r"(?<=[一-龥])\s+(?=[一-龥])", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """jieba 切词并过滤停用词与空白词。

    :param text: 原始文本
    :return: 词项列表（保留重复，用于词频统计）
    """
    return [t.strip() for t in jieba.lcut(text)
            if t.strip() and t.strip() not in STOPWORDS]


def term_present(term: str, compact_text: str) -> bool:
    """查询词项是否在块中命中（支持长术语的别名组兜底）。

    :param term: 查询词项
    :param compact_text: 已去空白的块文本
    :return: 命中返回 True
    """
    if term in compact_text:
        return True
    aliases = TERM_ALIASES.get(term)
    if aliases:
        # 每个别名组为“与”关系：如 募集资金 + (项目/用途/投资) 同时出现
        return all(any(a in compact_text for a in group) for group in aliases)
    return False


def _clean_page_lines(text: str) -> List[str]:
    """清洗单页文本：去页眉页码标记、纯页码行，返回有效行。

    :param text: PyMuPDF 提取的单页原文
    :return: 清洗后的行列表
    """
    lines = []
    for raw in text.splitlines():
        line = raw.replace("\u3000", " ").replace("\xa0", " ").strip()
        line = _PAGE_MARK_RE.sub("", line)
        if not line or _PURE_DIGIT_RE.match(line):
            continue
        lines.append(line)
    return lines


@dataclass
class Chunk:
    """检索块。"""

    chunk_id: int
    text: str
    doc_name: str
    page_no: int


def build_chunks(documents: Sequence[Document]) -> List[Chunk]:
    """把多份 PDF 的逐页文本切分为定长重叠字符块（保留换行作为行边界）。

    :param documents: Document 列表
    :return: Chunk 列表
    """
    chunks: List[Chunk] = []
    cid = 0
    for doc in documents:
        for page_idx, page_text in enumerate(doc.pages, start=1):
            flow = "\n".join(_clean_page_lines(page_text))
            if len(flow.strip()) < MIN_CHUNK_LEN:
                continue
            start, text_len = 0, len(flow)
            while start < text_len:
                end = min(start + CHUNK_SIZE, text_len)
                piece = flow[start:end].strip()
                if len(piece) >= MIN_CHUNK_LEN:
                    cid += 1
                    chunks.append(Chunk(
                        chunk_id=cid, text=piece,
                        doc_name=doc.file_name, page_no=page_idx))
                if end >= text_len:
                    break
                start = end - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# 二、BM25（Okapi 变体，自研实现，避免额外依赖）
# ---------------------------------------------------------------------------

class BM25:
    """轻量 BM25 倒排索引。"""

    def __init__(self, docs_tokens: List[List[str]],
                 k1: float = 1.5, b: float = 0.75) -> None:
        """构建词频/文档频率统计。

        :param docs_tokens: 每篇文档的词项列表
        :param k1: 词频饱和参数
        :param b: 文档长度归一化参数
        """
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        df: Dict[str, int] = {}
        # 倒排表：词项 -> [(文档下标, 词频)]，检索时只访问命中文档，避免全表扫描
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, tokens in enumerate(docs_tokens):
            freq: Dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            for t, f in freq.items():
                self.postings.setdefault(t, []).append((i, f))
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5))
                    for t, d in df.items()}
        # 长度归一化分母预算表（检索热路径直接索引）
        self._len_norm = (
            1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))

    def search(self, query_tokens: Sequence[str],
               top_n: int) -> List[Tuple[int, float]]:
        """对查询返回 BM25 Top-N（倒排表累加，非全表扫描）。

        :param query_tokens: 查询词项
        :param top_n: 返回数量
        :return: [(块下标, 分数)] 降序
        """
        scores: Dict[int, float] = {}
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, f in self.postings.get(term, ()):
                denom = f + self.k1 * self._len_norm[i]
                scores[i] = scores.get(i, 0.0) + idf * (f * (self.k1 + 1)) / denom
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:top_n] if s > 0]


# ---------------------------------------------------------------------------
# 三、RAG 流水线（索引 + 混合检索 + 作答 + 拒答）
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """单条检索证据。"""

    chunk_id: int
    score: float
    tfidf_rank: int
    bm25_rank: int
    page_no: int
    doc_name: str
    text: str
    hint: float = 0.0       # 题型结构信号加分（供答案阶段继承）
    anchor_hit: bool = False  # 块是否命中查询锚点词


@dataclass
class QAResult:
    """单题问答结果。"""

    question: str
    answer: str
    refused: bool
    refusal_reason: str
    evidences: List[Evidence]
    latency_s: float
    top_score: float
    coverage: float
    rare_term_hit: float
    structural_rescue: bool


class RAGPipeline:
    """TF-IDF + BM25 混合检索的抽取式 RAG 流水线。"""

    def __init__(self, chunks: List[Chunk]) -> None:
        """构建双路索引（纯 CPU，约数秒）。

        :param chunks: 分块列表
        """
        self.chunks = chunks
        corpus_text = [c.text for c in chunks]
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None,
            lowercase=False, sublinear_tf=True)
        self.tfidf_mat = self.vectorizer.fit_transform(corpus_text)
        self.bm25 = BM25([tokenize(t) for t in corpus_text])

    # ---------------- 查询信号 ----------------

    def _query_weighted_terms(self, query: str) -> List[Tuple[str, float]]:
        """抽取查询词项并以 IDF 加权（高区分度词权重更大）。

        拆词别名以 0.5 折权重并入，既缓解长术语词汇错位，又避免喧宾夺主。

        :param query: 原始问题
        :return: [(词项, 权重)] 权重降序
        """
        base = list(dict.fromkeys(tokenize(query)))
        terms = {t: self.bm25.idf.get(t, 8.0) for t in base}
        for alias in _alias_constituents(base):
            if alias not in terms:
                terms[alias] = 0.5 * self.bm25.idf.get(alias, 8.0)
        ordered = sorted(terms.items(), key=lambda x: x[1], reverse=True)
        return ordered

    def _coverage(self, query: str, text: str) -> Tuple[float, float]:
        """查询词在指定文本中的加权覆盖度与高IDF专有词命中率。

        :param query: 原始问题
        :param text: 候选块文本
        :return: (加权覆盖度, 专有词命中率)
        """
        ct = compact(text)
        weighted = self._query_weighted_terms(query)
        if not weighted:
            return 0.0, 0.0
        hit_w = sum(w for t, w in weighted if term_present(t, ct))
        total_w = sum(w for _, w in weighted) or 1.0
        rare = [(t, w) for t, w in weighted if w >= 1.5]
        rare_hit = sum(1 for t, _ in rare if term_present(t, ct))
        rare_ratio = rare_hit / len(rare) if rare else 1.0
        return hit_w / total_w, rare_ratio

    _CAP_NUM_CELL_RE = re.compile(r"^[\d,]{3,}(?:\.\d+)?$")

    @staticmethod
    def _capital_table_rows(text: str) -> List[Tuple[str, str, str]]:
        """重建 PyMuPDF 纵向展开的“募集资金用途表”单元格。

        PDF 表格按文本框输出时，单元格被拍成“名称（可跨行）/数字/数字”
        的纵向序列且序号列丢失（典型形态见招股书第30页）。本方法按
        “项目名称+项目总投资”表头定位，之后以 名称行…+两个数字单元格
        的模式重建每一行。

        :param text: 候选块文本
        :return: [(项目名称, 总投资, 募集资金投资额), ...]；无表返回空
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not any("项目名称" in l for l in lines) or \
                not any("总投资" in l for l in lines):
            return []
        # 从最后一个表头单元格之后开始解析数据区
        start = 0
        for i, line in enumerate(lines):
            if "募集资金投资" in line or "总投资" in line:
                start = i + 1
        # 表头残词与“建设期/占位符”单元格不计入项目名称
        header_words = {"拟投入募集资金", "建设期", "项目名称", "项目总投资",
                        "募集资金投资", "投资总额", "金额", "序号", "募集资金"}
        junk_cell = re.compile(r"^\d+个?月$|^[-—–_]+$")
        rows: List[Tuple[str, str, str]] = []
        pending: List[str] = []
        nums: List[str] = []

        def flush() -> None:
            cells = [c for c in pending
                     if c not in header_words and not junk_cell.match(c)]
            name = "".join(cells)
            if name and not re.search(r"合计|总计", name):
                rows.append((name, nums[0], nums[1]))

        for line in lines[start:]:
            cell = line.replace(" ", "")
            if RAGPipeline._CAP_NUM_CELL_RE.match(cell):
                nums.append(cell)
                if len(nums) == 2:
                    flush()
                    pending, nums = [], []
            else:
                pending.append(cell)
        return rows

    @staticmethod
    def _structural_hint(query: str, text: str,
                         weighted_terms: List[Tuple[str, float]]) -> Tuple[float, bool]:
        """题型结构信号：工商概况卡 / 枚举清单 / 金额邻近。

        解决纯词袋的两类失效：①发行人主体问被高频同名信息（子公司/中介
        机构）抢占；②枚举型问题的答案清单不含问题词（词汇错位）。

        :param query: 原始问题
        :param text: 候选块文本
        :param weighted_terms: 查询加权词项
        :return: (加分数, 是否命中查询锚点)
        """
        ct = compact(text)
        anchor_hit = any(term_present(t, ct) for t, _ in weighted_terms)
        bonus = 0.0

        # 信号一：工商概况卡（多个登记字段共现，发行人卡片特征）
        if re.search(r"法定代表人|注册资本|注册地址|成立日期|控股股东|实际控制人",
                     query):
            if sum(1 for f in _CARD_FIELDS if f in ct) >= 2:
                bonus += 0.28

        # 信号二：枚举清单。强信号要求“序号行≥2”（如募集资金用途表的
        # 1/2/3 行）或重建出的募投用途表（PyMuPDF 把表格单元格纵向展开，
        # 序号列丢失，按 表头+名称/数字单元格模式 重建），避免仅因多次
        # 提到锚点词的风险提示/制度段落抢占清单页；锚点+多数字但无表
        # 结构只给弱信号
        if re.search(r"哪些|哪几个|包括|什么项目|什么产品", query):
            list_rows = len(re.findall(r"(?:^|\s)[1-9]\s*[、.．\s]?\s*[一-龥A-Za-z]",
                                       text))
            n_num = len(_NUMBER_RE.findall(text))
            cap_rows = RAGPipeline._capital_table_rows(text)
            if anchor_hit and (list_rows >= 2 or len(cap_rows) >= 2):
                bonus += 0.30
            elif anchor_hit and n_num >= 3:
                bonus += 0.12
            if anchor_hit and "、" in ct:
                bonus += 0.06

        # 信号三：金额槽位词与大金额数字邻近（表格行特征）
        for anchor in ("补充流动资金", "注册资本", "募集资金总额"):
            if anchor in query and re.search(
                    anchor + r"[^。\n]{0,15}?\d[\d,]{3,}", ct):
                bonus += 0.25
                break
        return bonus, anchor_hit

    # ---------------- 检索 ----------------

    @staticmethod
    def _rrf(tf_hits: List[Tuple[int, float]],
             bm_hits: List[Tuple[int, float]]) -> Dict[int, float]:
        """RRF 倒数排名融合（只依赖名次，消除两路基带分差）。

        :param tf_hits: TF-IDF 召回
        :param bm_hits: BM25 召回
        :return: {块下标: rrf分}
        """
        fused: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(tf_hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (idx, _) in enumerate(bm_hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        return fused

    def retrieve(self, query: str,
                 top_k: int = FINAL_TOP_K) -> Tuple[List[Evidence], dict]:
        """双路召回→RRF融合→覆盖度/结构信号精排。

        :param query: 用户问题
        :param top_k: 返回证据数
        :return: (证据列表, 调试信号字典)
        """
        # 查询扩展：把长术语的拆词别名扁平并入两路召回
        base_tokens = list(dict.fromkeys(tokenize(query)))
        alias_tokens = _alias_constituents(base_tokens)
        retrieval_text = query + " " + " ".join(alias_tokens)
        q_vec = self.vectorizer.transform([retrieval_text])
        sim = cosine_similarity(q_vec, self.tfidf_mat)[0]
        tf_order = np.argsort(-sim)[:RECALL_N]
        tf_hits = [(int(i), float(sim[i])) for i in tf_order if sim[i] > 0]
        bm_hits = self.bm25.search(base_tokens + alias_tokens, RECALL_N)

        fused = self._rrf(tf_hits, bm_hits)
        tf_rank = {idx: r for r, (idx, _) in enumerate(tf_hits, 1)}
        bm_rank = {idx: r for r, (idx, _) in enumerate(bm_hits, 1)}
        tf_score = {idx: s for idx, s in tf_hits}
        bm_score = {idx: s for idx, s in bm_hits}
        tf_max = max(tf_score.values(), default=1.0) or 1.0
        bm_max = max(bm_score.values(), default=1.0) or 1.0
        weighted = self._query_weighted_terms(query)

        rows = []
        for idx in fused:
            cover, rare = self._coverage(query, self.chunks[idx].text)
            hint, anchor_hit = self._structural_hint(
                query, self.chunks[idx].text, weighted)
            n_tf = tf_score.get(idx, 0.0) / tf_max
            n_bm = bm_score.get(idx, 0.0) / bm_max
            final = W_TFIDF * n_tf + W_BM25 * n_bm + W_COVER * cover + hint
            rows.append((idx, final, hint, anchor_hit,
                         tf_rank.get(idx, -1), bm_rank.get(idx, -1)))
        rrf_rank = {idx: r for r, idx in enumerate(
            sorted(fused, key=fused.get, reverse=True), 1)}
        rows.sort(key=lambda r: (r[1], -rrf_rank[r[0]]), reverse=True)

        evidences: List[Evidence] = []
        for idx, final, hint, anchor_hit, tfr, bmr in rows[:top_k]:
            ch = self.chunks[idx]
            evidences.append(Evidence(
                chunk_id=ch.chunk_id, score=round(final, 4),
                tfidf_rank=tfr, bm25_rank=bmr, page_no=ch.page_no,
                doc_name=ch.doc_name, text=ch.text,
                hint=round(hint, 3), anchor_hit=anchor_hit))
        signals = {"tfidf_hits": len(tf_hits), "bm25_hits": len(bm_hits),
                   "candidate_union": len(fused)}
        return evidences, signals

    # ---------------- 候选答案生成与打分 ----------------

    def _iter_candidates(self, evidences: List[Evidence]):
        """从证据块生成候选答案：整句（硬换行拼回）+ 行单元（表格/卡片）。

        :param evidences: 证据列表
        :yield: (候选句, 证据名次, 父块结构加分, 父块锚点命中)
        """
        seen = set()

        header_re = re.compile(
            r"^[一-龥]{2,20}(?:股份有限公司|有限责任公司)\s*招股意向书\s*")

        def emit(unit: str, order: int, ev: Evidence, trim_at_period: bool):
            unit = display_clean(unit.strip(" ：:|"))
            unit = header_re.sub("", unit)
            # 表格行常被 PDF 折到下一句开头：保留首个句号前的内容
            if trim_at_period and "。" in unit:
                head, tail = unit.split("。", 1)
                unit = head + "。" if len(tail) <= 12 else unit
            if len(unit) < 6 and not re.search(r"\d", unit):
                return
            key = compact(unit)[:40]
            if key in seen:
                return
            seen.add(key)
            yield unit, order, ev.hint, ev.anchor_hit

        for order, ev in enumerate(evidences):
            # A. 行单元：表格行、概况卡字段（以换行分隔，每行可独立作答）
            for line in ev.text.split("\n"):
                yield from emit(line, order, ev, trim_at_period=True)
            # B. 整句：把硬换行拼回后按句读切分，覆盖正文叙述句
            for raw in _SENT_SPLIT_RE.findall(ev.text.replace("\n", "")):
                yield from emit(raw, order, ev, trim_at_period=False)

    def _sentence_score(self, query: str, sent: str, order: int,
                        hint: float, anchor_hit: bool) -> float:
        """候选答案句综合打分。

        组成：IDF 覆盖度（主）、长短语命中、数字/枚举类型吻合、证据名次
        先验、父块结构信号继承、长度惩罚。

        :param query: 原始问题
        :param sent: 候选句
        :param order: 来源证据名次
        :param hint: 父块结构加分
        :param anchor_hit: 父块是否命中查询锚点
        :return: 得分
        """
        cover, _ = self._coverage(query, sent)
        score = 2.0 * cover
        long_hits = sum(1 for t, _ in self._query_weighted_terms(query)
                        if len(t) >= 3 and term_present(t, compact(sent)))
        score += min(0.4, 0.2 * long_hits)
        is_number_q = bool(re.search(r"多少|数量|人数|金额|注册资本|股数|占比|比例",
                                     query))
        if is_number_q and _NUMBER_RE.search(sent):
            score += 0.2
            # 简短数字行（概况卡字段/表格行）优先于冗长拼接句，答案更凝练
            if len(sent) <= 80 and long_hits >= 1:
                score += 0.25
        if re.search(r"哪些|什么项目|什么产品|包括|产品有", query):
            if "、" in sent or re.search(r"[1-9][、.．]", sent):
                score += 0.2
        score += max(0.0, 0.15 - order * 0.03)
        score += hint  # 继承父块的结构信号（锚点清单行整体抬升）
        if anchor_hit:
            score += 0.05
        # 长度惩罚：过短多为标题碎片；过长（多为整块拼接）显著降权
        if len(sent) < 10 and not _NUMBER_RE.search(sent):
            score -= 0.4
        elif len(sent) > 160:
            score -= 0.5
        elif len(sent) > 110:
            score -= 0.1
        return score

    # ---------------- 通用金融槽位 ----------------

    def _slot_answer(self, query: str, evidences: List[Evidence]) -> str:
        """金融招股语料通用槽位：法定代表人/注册资本/保荐机构/补充流动资金。

        规则与具体公司无关，CCF 金融问答语料同样适用。

        :param query: 原始问题
        :param evidences: 证据列表
        :return: 模板化答案；未命中返回空串
        """
        # 1) 法定代表人：排除紧邻其他公司全称的匹配（中介卡片/对外投资），
        #    并用“发行人层级字段”（控股股东/实际控制人/股票种类等只在发行
        #    人概况卡出现的字段）区分发行人与子公司卡片
        if "法定代表人" in query:
            issuer_fields = ("控股股东", "实际控制人", "股份公司成立",
                             "行业分类", "股票种类", "设立方式", "招股意向书签署")
            issuer_name = ""
            m0 = _COMPANY_RE.search(compact(query))
            if m0:
                issuer_name = m0.group(0)
            pattern = re.compile(r"法定代表人[：:\s为是]{0,6}([一-龥]{2,4})")
            # 跨证据投票：发行人关键信息在概况卡、股东章节等多处重复，
            # 而子公司/中介卡片通常只出现一次，用融合分做权重累计，
            # 可稳定区分同名槽位；发行人层级字段另加权重
            votes: Dict[str, float] = {}
            for order, ev in enumerate(evidences):
                ct = compact(ev.text)
                issuer_bonus = 0.15 * sum(1 for f in issuer_fields if f in ct)
                for m in pattern.finditer(ct):
                    name = m.group(1)
                    while name and name[-1] in "注签盖名称住所电话分类类型成立":
                        name = name[:-1]
                    if len(name) < 2 or name in {"姓名", "名称"}:
                        continue
                    window = ct[max(0, m.start() - 24):m.start()]
                    others = [c for c in _COMPANY_RE.findall(window)
                              if issuer_name not in c]
                    if others:
                        continue  # 紧邻其他公司全称，属子公司/中介卡片
                    votes[name] = votes.get(name, 0.0) \
                        + ev.score + issuer_bonus
            if votes:
                name = max(votes, key=votes.get)
                return f"法定代表人是{name}。"
            return ""

        # 2) 注册资本：收集全部候选值，取最大值（注册资本经历次增资只增不减，
        #    当前注册资本必然是文档中的最大值；50万元设立值等小额被过滤）
        if "注册资本" in query:
            pattern = re.compile(
                r"注册资本[：:\s为是是]{0,8}?([0-9][0-9,]*(?:\.\d+)?)\s*(万?元)?")
            candidates: List[Tuple[float, str, float]] = []
            for order, ev in enumerate(evidences):
                ct = compact(ev.text)
                for m in pattern.finditer(ct):
                    raw_val = m.group(1).replace(",", "")
                    try:
                        val = float(raw_val)
                    except ValueError:
                        continue
                    if val < 100:
                        continue
                    window = ct[max(0, m.start() - 24):m.start()]
                    if _COMPANY_RE.findall(window) and "法定代表人" not in window:
                        # 紧邻其他公司且非工商概况卡：跳过
                        continue
                    sc = self._sentence_score(
                        query, ev.text, order, ev.hint, ev.anchor_hit)
                    candidates.append((val, m.group(1), sc))
            if candidates:
                # 最大值为主，同值时取句子分高者
                val, raw, _ = max(candidates, key=lambda x: (x[0], x[2]))
                text_val = f"{val:,.2f}".rstrip("0").rstrip(".")
                return f"注册资本为{text_val}万元。"
            return ""

        # 3) 保荐人/保荐机构（主承销商）：要求机构名紧随该称谓出现
        if re.search(r"保荐人|保荐机构|主承销商", query):
            pattern = re.compile(
                r"保荐(?:人|机构)（主承销商）(?:名称)?[^。\n]{0,10}?"
                r"([一-龥]{4,20}股份有限公司)")
            best = ("", -1.0)
            for order, ev in enumerate(evidences):
                ct = compact(ev.text)
                for m in pattern.finditer(ct):
                    name = m.group(1)
                    if "声明" in name or "核查" in name:
                        continue
                    sc = self._sentence_score(
                        query, ev.text, order, ev.hint, ev.anchor_hit)
                    if sc > best[1]:
                        best = (name, sc)
            if best[0]:
                return f"保荐人（主承销商）为{best[0]}。"
            return ""

        # 4) 补充流动资金金额：金额数字必须紧随“补充流动资金”出现。
        #    数字边界用负向断言，避免表格相邻两列“15,000.0015,000.00”
        #    被误连成 15,000.0015
        if "补充流动资金" in query and re.search(r"多少|金额|万元", query):
            money = re.compile(
                r"补充流动资金[^。0-9]{0,12}?"
                r"(\d[\d,]*\.\d{2}(?!\d)|\d[\d,]{3,}(?!\d))")
            best = (0.0, "", -1.0)
            for order, ev in enumerate(evidences):
                ct = compact(ev.text)
                for m in money.finditer(ct):
                    try:
                        val = float(m.group(1).replace(",", ""))
                    except ValueError:
                        continue
                    if val < 1000:  # 排除“不得超过12个月”等制度表述
                        continue
                    sc = self._sentence_score(
                        query, ev.text, order, ev.hint, ev.anchor_hit)
                    if val > best[0] or (val == best[0] and sc > best[2]):
                        best = (val, m.group(1), sc)
            if best[1]:
                return f"用于补充流动资金的募集资金为{best[1]}万元。"
            return ""
        return ""

    # ---------------- 抽取式作答 ----------------

    @staticmethod
    def _numbers(text: str) -> set:
        """候选答案中的归一化数字集合（用于多问点互补合并）。

        :param text: 候选文本
        :return: 数字字符串集合
        """
        nums = set()
        for raw in re.findall(r"\d{4}|\d[\d,]*(?:\.\d+)?", text):
            try:
                nums.add(str(float(raw.replace(",", ""))))
            except ValueError:
                nums.add(raw)
        return nums

    _LIST_ROW_RE = re.compile(r"^([1-9])\s*[、.．\s]+\s*(\S.{3,})$")

    def _enumeration_list_answer(self, query: str,
                                 evidences: List[Evidence]) -> str:
        """枚举题清单直取：从锚点证据块直接提取 1/2/3 序号行。

        枚举题的答案清单往往不含问题词（词汇错位），句级打分易被同锚点
        的制度/风险段落反超；序号清单行是表格的强结构信号，故在锚点
        命中的前三个证据块内直接识别序号行并按序号拼合。

        :param query: 原始问题
        :param evidences: 证据列表
        :return: 清单答案；未找到合规清单时返回空串
        """
        if not re.search(r"哪些|哪几个|包括|什么项目|什么产品", query):
            return ""
        # 路径一：募集资金用途表（纵向单元格重建），跨全部证据挑选行数
        # 最多、融合分最高的表；p30 即使只排在第6位也能被选中
        best_table: Optional[Tuple[int, float, List[Tuple[str, str, str]]]] = None
        for order, ev in enumerate(evidences):
            if not ev.anchor_hit:
                continue
            rows = self._capital_table_rows(ev.text)
            if len(rows) >= 2:
                key = (len(rows), ev.score - 0.01 * order)
                if best_table is None or key > best_table[0]:
                    best_table = (key, ev.score, rows)  # type: ignore
        if best_table is not None:
            rows = best_table[2]
            return "；".join(
                f"{name}（{raised}万元）" for name, _, raised in rows[:4])
        # 路径二：常规 1/2/3 序号行清单
        for ev in evidences:
            if not ev.anchor_hit:
                continue
            rows: List[Tuple[int, str]] = []
            for line in ev.text.split("\n"):
                line = display_clean(line.strip(" ：:|"))
                m = self._LIST_ROW_RE.match(line)
                if m and len(m.group(2)) >= 4:
                    rows.append((int(m.group(1)), line[:90]))
            if len(rows) < 2:
                continue
            nums = [n for n, _ in rows]
            adjacent = any(nums[i + 1] == nums[i] + 1
                           for i in range(len(nums) - 1))
            if 1 in nums or adjacent:
                rows.sort(key=lambda x: x[0])
                return "；".join(text for _, text in rows[:4])
        return ""

    def extract_answer(self, query: str, evidences: List[Evidence]) -> str:
        """在候选答案池中原地选取得分最高答案；多问点数字题合并互补证据。

        :param query: 原始问题
        :param evidences: 证据列表
        :return: 答案文本
        """
        slotted = self._slot_answer(query, evidences)
        if slotted:
            return slotted

        # 枚举题优先直取序号清单（防止制度/风险段落反超表格清单）
        enum_list = self._enumeration_list_answer(query, evidences)
        if enum_list:
            return enum_list

        candidates = list(self._iter_candidates(evidences))
        if not candidates:
            return ""
        number_question = bool(re.search(r"多少|数量|人数|金额|股数|占比|比例",
                                         query))
        if number_question:
            numbered = [c for c in candidates if _NUMBER_RE.search(c[0])]
            if numbered:
                candidates = numbered
        ranked = sorted(candidates,
                        key=lambda x: self._sentence_score(
                            query, x[0], x[1], x[2], x[3]),
                        reverse=True)
        if not ranked:
            return ""
        best_sent = ranked[0][0]
        best_score = self._sentence_score(
            query, best_sent, ranked[0][1], ranked[0][2], ranked[0][3])
        selected = [best_sent]

        # 多问点数字题（含两个及以上数量问点）：合并一条互补数字证据，
        # 要求互补句自身命中至少一个查询长短语；候选中取最短者，
        # 优先得到“发行后总股本 7,360 万股”这样的凝练字段行
        if number_question and len(_QUANT_MARK_RE.findall(query)) >= 2:
            known = self._numbers(best_sent)
            pool = []
            for sent, order, hint, anc in ranked[1:10]:
                sc = self._sentence_score(query, sent, order, hint, anc)
                if sc < best_score - 0.55:
                    break
                new_nums = self._numbers(sent) - known
                long_hit = any(len(t) >= 2 and term_present(t, compact(sent))
                               for t, _ in self._query_weighted_terms(query))
                if new_nums and long_hit:
                    pool.append((len(sent), sent))
            if pool and sum(map(len, selected)) < 200:
                selected.append(min(pool, key=lambda x: x[0])[1])
        elif re.search(r"哪些|什么项目|什么产品|包括|产品有", query):
            # 枚举题：优先收集同一清单的序号行（1/2/3），最多3行；
            # 或同证据块内的顿号枚举句。跨块且无序号的近似句不合并，
            # 避免把相邻的近似业务表述（干扰句）拼入答案
            best_order = ranked[0][1]
            for sent, order, hint, anc in ranked[1:10]:
                sc = self._sentence_score(query, sent, order, hint, anc)
                if sc < best_score - 0.60 or len(selected) >= 3:
                    break
                cs = compact(sent)
                if any(cs in compact(s) or compact(s) in cs for s in selected):
                    continue
                is_marker_line = bool(re.match(r"^\s*[1-9]\s*[、.．]?\s*[一-龥A-Za-z]",
                                               sent))
                is_sibling_list = (order == best_order and "、" in sent)
                if (is_marker_line or is_sibling_list) \
                        and sum(map(len, selected)) < 240:
                    selected.append(sent)
        return " ".join(selected)

    # ---------------- 单题问答（拒答判定 + 计时） ----------------

    def answer(self, query: str) -> QAResult:
        """执行一次完整问答：检索→拒答判定→抽取作答，全程打点计时。

        :param query: 用户问题
        :return: QAResult
        """
        t0 = time.perf_counter()
        pool, _ = self.retrieve(query, top_k=ANSWER_POOL_K)
        if not pool:
            return QAResult(
                question=query, answer=REFUSAL_TEXT, refused=True,
                refusal_reason="双路检索零召回，语料中不存在任何匹配内容",
                evidences=[], latency_s=round(time.perf_counter() - t0, 4),
                top_score=0.0, coverage=0.0, rare_term_hit=0.0,
                structural_rescue=False)

        top = pool[0]
        coverage, rare = self._coverage(query, top.text)
        # 枚举题结构挽救：锚点+清单结构信号足够强时，允许越过词面双阈值
        rescue = bool(re.search(r"哪些|哪几个|包括|什么项目|什么产品", query)) \
            and top.hint >= 0.30 and top.anchor_hit

        reasons = []
        if coverage < REFUSAL_COVERAGE_MIN:
            reasons.append(
                f"查询词覆盖度{coverage:.2f}低于阈值{REFUSAL_COVERAGE_MIN}")
        if rare < REFUSAL_RARE_TERM_MIN:
            reasons.append(
                f"专有词命中率{rare:.2f}低于阈值{REFUSAL_RARE_TERM_MIN}")
        should_refuse = bool(reasons) and not rescue

        latency = round(time.perf_counter() - t0, 4)
        # 作答在10块证据池上进行；对外只输出前 FINAL_TOP_K 块上下文
        if should_refuse:
            return QAResult(
                question=query, answer=REFUSAL_TEXT, refused=True,
                refusal_reason="；".join(reasons), evidences=pool[:FINAL_TOP_K],
                latency_s=latency, top_score=top.score,
                coverage=round(coverage, 4), rare_term_hit=round(rare, 4),
                structural_rescue=False)

        answer = self.extract_answer(query, pool)
        latency = round(time.perf_counter() - t0, 4)
        if not answer.strip():
            return QAResult(
                question=query, answer=REFUSAL_TEXT, refused=True,
                refusal_reason="证据块中未抽取出任何有效答案句",
                evidences=pool[:FINAL_TOP_K], latency_s=latency,
                top_score=top.score, coverage=round(coverage, 4),
                rare_term_hit=round(rare, 4), structural_rescue=rescue)
        return QAResult(
            question=query, answer=answer, refused=False, refusal_reason="",
            evidences=pool[:FINAL_TOP_K], latency_s=latency, top_score=top.score,
            coverage=round(coverage, 4), rare_term_hit=round(rare, 4),
            structural_rescue=rescue)


# ---------------------------------------------------------------------------
# 四、索引构建与磁盘缓存（语料不变时秒级重载，保证逐题 ≤3 秒）
# ---------------------------------------------------------------------------

def build_or_load_index(documents: Sequence[Document],
                        cache_path: Optional[str] = None) -> RAGPipeline:
    """构建流水线，可选按语料指纹缓存索引。

    :param documents: 语料文档列表
    :param cache_path: 索引缓存 pkl 路径；为 None 则不缓存
    :return: 已就绪的 RAGPipeline
    """
    fingerprint = sorted(
        (d.file_name, len(d.pages), sum(len(p) for p in d.pages))
        for d in documents)
    if cache_path:
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                pipe = RAGPipeline(cached["chunks"])
                pipe.vectorizer = cached["vectorizer"]
                pipe.tfidf_mat = cached["tfidf_mat"]
                pipe.bm25 = cached["bm25"]
                print(f"[索引] 命中缓存：{cache_path}（{len(pipe.chunks)} 块）")
                return pipe
        except (FileNotFoundError, pickle.PickleError, KeyError, EOFError):
            pass

    chunks = build_chunks(documents)
    print(f"[索引] 语料分块完成：{len(chunks)} 个检索块，开始构建 TF-IDF/BM25 索引…")
    t0 = time.perf_counter()
    pipe = RAGPipeline(chunks)
    print(f"[索引] 索引构建完成，用时 {time.perf_counter() - t0:.2f} 秒")
    if cache_path:
        import os
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "fingerprint": fingerprint,
                "chunks": chunks,
                "vectorizer": pipe.vectorizer,
                "tfidf_mat": pipe.tfidf_mat,
                "bm25": pipe.bm25,
            }, f)
        print(f"[索引] 已写入缓存：{cache_path}")
    return pipe
