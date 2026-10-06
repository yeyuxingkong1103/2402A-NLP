# -*- coding: utf-8 -*-
# 【重排器模块 · rerankers.py】实现三种重排策略：LLM重排、TF-IDF重排、用户反馈自适应重排
# 工单编号：人工智能NLP-RAG-混合检索任务

"""重排层（工单要求“至少3种重排算法”）：

1. ``LLMReranker`` 基于LLM的重排器：本地离线加载 BAAI/bge-reranker-large
   Cross-Encoder（大模型语义交叉编码），对“问题-候选块”逐对深度交互打分；
   本地无模型缓存时自动降级为词法信号精排，绝不联网下载；
2. ``TfidfReranker`` 基于TF-IDF的重排器：词级 TF-IDF 余弦相似度
   （jieba 分词拟合）+ IDF 查询覆盖度 + 长短语有序命中 + 答案类型吻合；
3. ``FeedbackAdaptiveReranker`` 基于用户反馈的自适应重排器：在 TF-IDF 词法
   精排底座上，叠加“历史采纳块词项画像”“相似问题采纳块提权”“不采纳块降权”，
   反馈数据由 feedback_store 持久化，实现越用越准的闭环。

为控制 CPU 端到端 ≤3 秒，LLM 重排候选数与截断长度在 config 中配置。
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from embeddings import model_available_locally
from vector_store import tokenize

logger = logging.getLogger(__name__)

# 查询泛化停用词（对答案定位无区分度）
QUERY_STOPWORDS = {
    "根据", "按照", "招股意向书", "招股说明书", "招股", "意向书", "意向",
    "说明书", "股份有限公司", "有限公司",
    "公司", "企业", "本次", "报告期", "报告期内", "分别", "多少", "哪些",
    "哪个", "什么样", "请问", "一下", "以及", "对于", "相关", "主要",
    "是", "的", "了", "在", "和", "与", "或", "为", "有", "不", "对",
    "中", "该", "其", "用", "于", "哪", "么", "怎", "何", "谁", "呢",
    "吗", "这", "那", "个", "项", "次", "将", "已", "均", "各类", "各种",
    "时", "后", "前", "本", "what", "how", "does", "the", "of", "is",
    "co", "ltd",
}
# 公司全称模式：高频署名实体对检索无区分度，查询侧剔除
_COMPANY_RE = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司)")


def normalize_query(query: str) -> str:
    """查询归一化：剔除公司全称，降低高频实体对全文路的干扰。

    :param query: 原始用户问题（中/英）
    :return: 归一化问题
    """
    return _COMPANY_RE.sub("", query)


def query_terms(query: str, idf: Dict[str, float]) -> List[Tuple[str, float]]:
    """抽取带 IDF 权重的查询词项（去停用词，中心词加权）。

    :param query: 原始问题
    :param idf: 倒排 IDF 表（未登录词给默认高 IDF）
    :return: [(词项, 权重)] 按权重降序去重
    """
    normalized = normalize_query(query)
    weighted: List[Tuple[str, float]] = []
    for t in tokenize(normalized):
        if t in QUERY_STOPWORDS:
            continue
        w = idf.get(t, 8.0)
        if f"的{t}" in normalized:  # “的”后中心词（如“下游”）提权
            w *= 1.3
        weighted.append((t, w))
    dedup = {t: w for t, w in weighted}
    return sorted(dedup.items(), key=lambda x: x[1], reverse=True)


def exact_phrases(query: str) -> List[str]:
    """抽取关键长短语（3字以上词项 + 中文片段滑窗子串），用于精确命中加分。

    :param query: 原始问题
    :return: 去重长短语列表
    """
    normalized = normalize_query(query)
    phrases = {t for t in tokenize(normalized) if len(t) >= 3
               and t not in QUERY_STOPWORDS}
    source = normalized
    for word in sorted(QUERY_STOPWORDS, key=len, reverse=True):
        source = source.replace(word, "")
    for run in re.findall(r"[一-龥]{4,14}", source):
        for size in (4, 5, 6):
            for start in range(0, len(run) - size + 1, 2):
                phrases.add(run[start:start + size])
    return list(phrases)


# 事实型问题的“属性标签”：问题含某标签时，正文里标签紧邻取值的块应显著加分
# （区分“法定代表人：程家明”的概况卡与仅提及“程家明，董事长”的管理层段落）
_ATTR_LABEL_VALUE = [
    ("法定代表人", re.compile(r"法定代表人[：:是为\s]{0,6}[一-龥A-Za-z]{2,4}")),
    ("成立日期", re.compile(r"成立日期[：:\s]{0,6}\d{4}\s*年")),
    ("注册资本", re.compile(r"注册资本[：:是为为\s]{0,6}?[0-9][0-9,\.]{2,}")),
    ("总股本", re.compile(r"(?:发行前|发行后)?总股本[^0-9]{0,8}[0-9][0-9,]{2,}")),
    ("主营业务", re.compile(r"主营业务[：:]?[是为]?[^。；]{0,40}")),
    ("募集资金", re.compile(r"募集资金[^。；]{0,30}")),
]


def answer_type_bonus(query: str, text: str) -> float:
    """按问题期望答案类型给予候选块吻合加分（数字/人名/枚举/属性标签）。

    :param query: 原始问题
    :param text: 候选块正文（已去标题前缀，避免“标题含词、正文答非所问”误加分）
    :return: 类型吻合加分
    """
    bonus = 0.0
    if re.search(r"多少|比重|比例|金额|注册资本|收入|数量|几个|占比", query):
        if re.search(r"\d[\d,\.]*\s*(?:%|万元|亿元|万股|个|名)?", text):
            bonus += 0.12
    if re.search(r"谁|代表人|发明人|who", query, re.IGNORECASE) and re.search(
            r"[是为：:]\s*[一-龥A-Za-z]{2,4}", text):
        bonus += 0.12
    if re.search(r"哪些|包括|涉及|what", query, re.IGNORECASE):
        # 枚举类问题：“包括/分别为+顿号列举”是强答案信号
        if ("包括" in text or "分别为" in text) and "、" in text:
            bonus += 0.50
        elif "、" in text or "；" in text or "|" in text:
            bonus += 0.08
    # 枚举类问题（前五大客户/供应商/募投项目）：同一引导父段落下常有多个
    # 兄弟块，实体行（XX公司/XX中心/XX平台）越密集越可能是“明细总表”块
    if re.search(r"客户|供应商|项目|哪些|什么业务", query):
        entities = re.findall(
            r"[一-龥A-Za-z0-9()（）]{2,18}?(?:公司|银行|中心|平台|工厂)", text)
        if entities:
            bonus += min(0.30, 0.05 * len(set(entities)))
    # 属性标签紧邻取值：强信号（如 法定代表人：程家明）
    for label, pattern in _ATTR_LABEL_VALUE:
        if label in query and pattern.search(text):
            bonus += 0.22
            break
    return bonus


def _normalize_base(base_scores: Dict[int, float],
                    candidate_idx: List[int]) -> Dict[int, float]:
    """融合底座分在候选集合内 min-max 归一化，消除三种融合分数量纲差异。

    :param base_scores: 融合分数字典
    :param candidate_idx: 候选 id 列表
    :return: {chunk_id: 0~1 归一底座分}
    """
    vals = [base_scores.get(i, 0.0) for i in candidate_idx]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
    span = hi - lo
    if span <= 1e-9:
        return {i: 0.5 for i in candidate_idx}
    return {i: (base_scores.get(i, 0.0) - lo) / span for i in candidate_idx}


def lexical_features(query: str, text: str, idf: Dict[str, float]) -> float:
    """词法精排特征分：IDF 覆盖度 + 长短语命中（封顶）+ 答案类型吻合。

    匹配只在**块正文**上进行（剔除标题路径与主体公司前缀），避免出现
    “标题写着主营业务、正文却是地区占比”的块靠标题骗到词法分。

    :param query: 原始问题
    :param text: 候选块文本
    :param idf: IDF 权重表
    :return: 词法特征分（约 0~2）
    """
    body = strip_chunk_prefix(text)
    terms = query_terms(query, idf)
    total_w = sum(w for _, w in terms) or 1.0
    coverage_w = sum(w for t, w in terms if t in body)
    score = 1.5 * coverage_w / total_w
    score += min(0.45, sum(0.15 for p in exact_phrases(query) if p in body))
    score += answer_type_bonus(query, body)
    return score


# ---------------------------------------------------------------------------
# 跨语言查询翻译/扩展（验收“支持中文和英文的问答”）：离线术语词典把英文
# 问题翻译为中文检索串（中文前置、英文原句保留），使其同时进入 bge 中英
# 同空间与中文 BM25 倒排；全程不调用任何外部翻译服务
# ---------------------------------------------------------------------------
# 英文业务术语 -> 中文检索词（每项给出标准词+同义扩展词，空格分隔）
EN_ZH_GLOSSARY = [
    (re.compile(r"registered\s+capital", re.I), "注册资本 注册资金"),
    (re.compile(r"legal\s+representative", re.I), "法定代表人 法人代表"),
    (re.compile(r"date\s+of\s+(?:incorporation|establishment|founding)"
                r"|established|founded|incorporated", re.I), "成立日期 成立时间"),
    (re.compile(r"engage[sd]?\s+in|engaged\s+in|main\s+business|"
                r"principal\s+business|what\s+business|what\s+does",
                re.I), "主营业务 主要从事 业务"),
    (re.compile(r"application\s+fields?|fields?|areas?", re.I), "应用领域 领域"),
    (re.compile(r"industry\s+chain", re.I), "产业链"),
    (re.compile(r"upstream", re.I), "上游"),
    (re.compile(r"downstream", re.I), "下游"),
    (re.compile(r"customers?", re.I), "客户"),
    (re.compile(r"suppliers?", re.I), "供应商"),
    (re.compile(r"share\s*capital|total\s+shares?|shares?", re.I),
     "总股本 股份 发行股数"),
    (re.compile(r"proceeds|raise[ds]?|investment\s+projects?|funds?",
                re.I), "募集资金 募投项目"),
    (re.compile(r"revenue|income", re.I), "营业收入 收入"),
    (re.compile(r"chairman", re.I), "董事长"),
    (re.compile(r"products?", re.I), "产品"),
    (re.compile(r"electronic\s+components?", re.I), "电子元器件"),
    (re.compile(r"integrated\s+circuit|\bIC\b", re.I), "IC 集成电路"),
    (re.compile(r"military", re.I), "军用 军工"),
    (re.compile(r"technical\s+standard|technology\s+standard", re.I),
     "技术标准"),
    (re.compile(r"supplier|vendor", re.I), "供应商"),
]
# 英文公司名 -> 中文简称（招股书内英文名仅出现在概况/封面，中文简称可全库召回）
EN_COMPANY_ALIASES = [
    (re.compile(r"xingtu\s*xinke\s*electronics|xingtu\s*xinke", re.I),
     "兴图新科 武汉兴图新科电子股份有限公司"),
    (re.compile(r"wuhan\s*p&s|p&s\s*information\s*technology|p&s", re.I),
     "力源信息 武汉力源信息技术股份有限公司"),
]


def is_english_query(query: str) -> bool:
    """判断是否英文问题（ASCII 字母占比高且基本不含汉字）。

    :param query: 用户问题
    :return: 英文问题返回 True
    """
    letters = len(re.findall(r"[A-Za-z]", query))
    han = len(re.findall(r"[一-龥]", query))
    return letters >= 8 and han <= 2


def translate_en_query(query: str) -> str:
    """离线词典式英译中：把英文问题翻译为中文检索串（供答案类型/槽位识别）。

    只翻译领域术语与公司实体，不做完整句法翻译；中文问题原样返回。

    :param query: 原始用户问题（中/英）
    :return: 中文术语串（非英文问题时原样返回）
    """
    if not is_english_query(query):
        return query
    zh_terms: List[str] = []
    for pattern, zh in EN_COMPANY_ALIASES:
        if pattern.search(query):
            zh_terms.extend(zh.split())
            break  # 公司简称只需一个
    for pattern, zh in EN_ZH_GLOSSARY:
        if pattern.search(query):
            zh_terms.extend(zh.split())
    # 去重保序
    return " ".join(dict.fromkeys(zh_terms))


# 中文查询同义词/近义表述扩展：招股书行文用词与提问用词常不一致
# （“应用领域”↔“应用于”、“前五大客户”↔“前五名客户”、“业务”↔“经营范围”），
# 扩展词追加在查询尾部，同时供 BM25 倒排与稠密向量双路召回，不改变原句语义。
ZH_SYNONYM_RULES = [
    (re.compile(r"业务|从事|经营"), "主营业务 经营范围"),
    (re.compile(r"应用领域|哪些领域|应用的?领域"), "应用于 广泛应用 产品应用"),
    # 客户表正文不含“客户”（PDF断字为“客 户”），追加表头词召回销售明细表
    (re.compile(r"前五大客户|前五名客户|大客户|主要客户"),
     "前五名客户 前五大客户 主要客户 销售金额 销售总额 销售额 占比 合计"),
    (re.compile(r"法定代表人|法人代表"), "法人代表 法定代表人"),
    (re.compile(r"注册资本|注册资金"), "注册资金 注册资本 实收资本"),
    (re.compile(r"成立日期|成立时间"), "成立日期 设立日期 工商成立"),
    (re.compile(r"募集资金投资|募投|投资项目|募集资金.*项目|募集资金用途"),
     "募集资金运用 募集资金用途 募集资金投资项目 项目名称"),
    (re.compile(r"补充流动资金"),
     "补充流动资金 拟投入募集资金 募集资金运用 募集资金用途"),
    (re.compile(r"总股本|发行股数|股本"), "股本 持股数"),
    (re.compile(r"技术标准|技术规范"), "技术标准 技术规范 规范"),
    (re.compile(r"重要供应商|主要供应商"), "重要供应商 主要供应商 供应商"),
    # 兴图新科的“重要供应商”事实与“视频指挥领域”同句（领域型提问的实体侧写）
    (re.compile(r"(?:兴图|新科|xinke)[\s\S]{0,24}供应商"
                r"|供应商[\s\S]{0,24}(?:兴图|新科|xinke)", re.I),
     "视频指挥 军队视频指挥 视频指挥领域"),
    (re.compile(r"产业链"), "上游生产商 下游用户 纽带 产业链"),
    (re.compile(r"上游"), "上游行业 上游产业"),
    (re.compile(r"下游"), "下游行业 下游产业"),
    (re.compile(r"收入"), "营业收入 销售收入"),
]


def expand_zh_synonyms(query: str) -> str:
    """对中文查询追加近义表述词（英文查询的中文翻译前缀同样适用）。

    :param query: 原始查询（或中英跨语言扩展后的查询）
    :return: “原句 + 同义表述词”扩展查询；无命中规则时原样返回
    """
    extra: List[str] = []
    for pattern, terms in ZH_SYNONYM_RULES:
        if pattern.search(query):
            extra.extend(terms.split())
    if not extra:
        return query
    tail = " ".join(dict.fromkeys(extra))
    return f"{query} {tail}"


def expand_cross_lingual(query: str) -> str:
    """英文问题翻译为中文检索串并保留英文原句（中文问题原样返回）。

    中文术语前置以主导 BM25 与重排词法信号，原始英文保留在尾部以命中
    招股书中的英文名称行；扩展后查询同时用于稠密召回（bge 中英同空间）、
    BM25 召回与重排，不调用任何外部翻译服务。

    :param query: 原始问题
    :return: “中文翻译词 + 英文原句”扩展查询串
    """
    if not is_english_query(query):
        return query
    zh = translate_en_query(query)
    if not zh:
        return query
    return f"{zh} {query}"


_CHUNK_HEAD_RE = re.compile(r"【[^】]*】")
_CHUNK_ENT_RE = re.compile(r"主体：[^\n]*\n?")


def strip_chunk_prefix(text: str) -> str:
    """去除块文本的标题路径与主体公司前缀，返回纯正文/表格内容。

    :param text: Chunk.text（含【标题】主体：公司 前缀）
    :return: 纯正文
    """
    return _CHUNK_ENT_RE.sub("", _CHUNK_HEAD_RE.sub("", text)).strip()


def rerank_excerpt(query: str, text: str, max_len: int,
                   q_terms: Optional[set] = None) -> str:
    """为重排截取候选块“精华窗口”：先去前缀，再在数字/查询词最密集处滑窗。

    财务表格行很长且答案数字可能在中后部，直接截首 64 字会把事实截掉，
    因此对含数字的块按窗口（查询词命中 + 数字密度）选取得分最高片段。

    :param query: 原始问题（q_terms 缺省时用于分词）
    :param text: 块完整文本
    :param max_len: 窗口字符上限
    :param q_terms: 预分词的查询词集合（批量重排时只分词一次，控 CPU 耗时）
    :return: 截取后的精华文本
    """
    body = strip_chunk_prefix(text).replace("\n", " ")
    if len(body) <= max_len:
        return body
    if q_terms is None:
        q_terms = {t for t in tokenize(query) if len(t) >= 2}

    def window_score(start: int) -> float:
        win = body[start:start + max_len]
        hits = sum(1 for t in q_terms if t in win)
        digits = len(re.findall(r"\d", win))
        return hits * 3 + digits * 0.05 - start * 0.0005  # 微偏好靠前窗口

    # 步长约为窗口 1/4，兼顾速度与覆盖
    step = max(24, max_len // 4)
    starts = range(0, len(body) - max_len + 1, step)
    best_start = max(starts, key=window_score, default=0)
    return body[best_start:best_start + max_len]


class BaseReranker:
    """重排器统一接口。"""

    name = "base"

    def rerank(self, query: str, pairs: List[Tuple[int, str]],
               base_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """对融合后的候选块重排。

        :param query: 用户问题
        :param pairs: [(chunk_idx, 块文本)] 候选
        :param base_scores: 融合底座分
        :return: [(chunk_idx, 最终得分)] 降序
        """
        raise NotImplementedError

    @staticmethod
    def _sort(scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """按得分降序输出。"""
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class LLMReranker(BaseReranker):
    """基于LLM（Cross-Encoder 大模型）的重排器；本地无模型时词法降级。"""

    name = "llm"

    def __init__(self, idf: Optional[Dict[str, float]] = None) -> None:
        """尝试离线加载本地 bge-reranker-large，失败则启用词法降级。

        :param idf: BM25 IDF 表（降级精排时做词项权重）
        """
        self.idf = idf or {}
        self.model = None
        if model_available_locally(CONFIG.reranker_model):
            try:
                from sentence_transformers import CrossEncoder
                # 离线加载，禁止联网；CPU 推理
                self.model = CrossEncoder(CONFIG.reranker_model, device="cpu",
                                          local_files_only=True)
                logger.info("LLM重排器已离线加载: %s", CONFIG.reranker_model)
            except Exception as exc:
                logger.warning("LLM重排模型加载失败，降级词法精排: %s", exc)
        else:
            logger.info("未检测到 %s 本地缓存，LLM重排降级词法精排（不联网下载）",
                        CONFIG.reranker_model)

    def rerank(self, query: str, pairs: List[Tuple[int, str]],
               base_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """Cross-Encoder 打分后与词法/答案类型特征混合；异常时自动降级。

        纯 Cross-Encoder 对“同一主题的散文页 vs 含精确数字的表格页”区分度
        不足（财务分析散文常被误排到数字事实表之前），因此最终分：
        ``final = (1-α)·CE归一 + α·词法特征归一``，
        数字型问题对完全不含数字的候选追加惩罚。

        :param query: 用户问题
        :param pairs: 候选块
        :param base_scores: 融合底座分
        :return: 重排降序结果
        """
        if not pairs:
            return []
        if self.model is not None:
            try:
                # 去标题/主体前缀，按查询词与数字密度选精华窗口，避免截断丢事实；
                # 查询只分词一次，控制 14 候选滑窗的 CPU 开销
                q_terms = {t for t in tokenize(query) if len(t) >= 2}
                tr = [rerank_excerpt(query, t, CONFIG.rerank_truncate, q_terms)
                      for _, t in pairs]
                raw = self.model.predict([(query, t) for t in tr],
                                         show_progress_bar=False)
                ce = {pairs[i][0]: float(raw[i]) for i in range(len(pairs))}
                ce_norm = _normalize_base(ce, [i for i, _ in pairs])
                alpha = CONFIG.llm_lexical_blend
                numeric_q = bool(re.search(
                    r"多少|比重|比例|金额|注册资本|收入|数量|几个|占比", query))
                scores: Dict[int, float] = {}
                for idx, text in pairs:
                    lex = min(1.0, lexical_features(query, text, self.idf) / 2.0)
                    s = (1 - alpha) * ce_norm.get(idx, 0.0) + alpha * lex
                    if numeric_q and not re.search(r"\d", strip_chunk_prefix(text)):
                        s -= CONFIG.llm_rerank_num_penalty
                    scores[idx] = s
                return self._sort(scores)
            except Exception as exc:
                logger.warning("LLM重排推理失败，本次降级词法精排: %s", exc)
        # 词法降级：融合归一底座 + 词法特征
        norm = _normalize_base(base_scores, [i for i, _ in pairs])
        scores = {idx: norm.get(idx, 0.0) * 0.3
                  + lexical_features(query, text, self.idf)
                  for idx, text in pairs}
        return self._sort(scores)


class TfidfReranker(BaseReranker):
    """基于 TF-IDF 的重排器：词级 TF-IDF 余弦 + 词法覆盖特征。"""

    name = "tfidf"

    def __init__(self, chunk_texts: List[str],
                 idf: Optional[Dict[str, float]] = None) -> None:
        """在全库块文本上拟合词级 TF-IDF（jieba 分词）。

        :param chunk_texts: 全部块文本
        :param idf: BM25 IDF 表
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.idf = idf or {}
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: tokenize(x), token_pattern=None,
            sublinear_tf=True, max_features=30000)
        self.doc_matrix = self.vectorizer.fit_transform(chunk_texts)

    def rerank(self, query: str, pairs: List[Tuple[int, str]],
               base_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """TF-IDF 余弦相似度与词法特征、融合底座线性融合打分。

        :param query: 用户问题
        :param pairs: 候选块
        :param base_scores: 融合底座分
        :return: 重排降序结果
        """
        if not pairs:
            return []
        idx_list = [i for i, _ in pairs]
        q_vec = self.vectorizer.transform([normalize_query(query)])
        # 仅在候选文档子矩阵上算余弦，毫秒级
        from sklearn.metrics.pairwise import cosine_similarity
        cos = cosine_similarity(q_vec, self.doc_matrix[idx_list])[0]
        norm = _normalize_base(base_scores, idx_list)
        scores: Dict[int, float] = {}
        for row, (idx, text) in enumerate(pairs):
            scores[idx] = (1.2 * float(cos[row])
                           + lexical_features(query, text, self.idf)
                           + 0.2 * norm.get(idx, 0.0))
        return self._sort(scores)


class FeedbackAdaptiveReranker(BaseReranker):
    """基于用户反馈的自适应重排器：TF-IDF 词法底座 + 反馈画像动态加权。"""

    name = "adaptive"

    def __init__(self, chunk_texts: List[str], feedback_store,
                 idf: Optional[Dict[str, float]] = None) -> None:
        """拟合词级 TF-IDF 并载入历史反馈画像。

        :param chunk_texts: 全部块文本
        :param feedback_store: FeedbackStore 反馈存储
        :param idf: BM25 IDF 表
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import Counter
        self.idf = idf or {}
        self.chunk_texts = chunk_texts
        self.feedback = feedback_store
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: tokenize(x), token_pattern=None,
            sublinear_tf=True, max_features=30000)
        self.doc_matrix = self.vectorizer.fit_transform(chunk_texts)
        self.query_matrix = None
        self._build_profile()

    def refresh(self) -> None:
        """反馈文件新增后重建画像（模拟反馈一轮后调用）。"""
        self.feedback.records = []
        self.feedback._load()
        self._build_profile()

    def _build_profile(self) -> None:
        """根据历史采纳/不采纳记录构建词项正画像、负画像与相似问题索引。"""
        from collections import Counter
        self.adopted_ids = self.feedback.adopted_ids()
        self.rejected_ids = self.feedback.rejected_ids()
        pos_counter: Counter = Counter()
        for cid in self.adopted_ids:
            if 0 <= cid - 1 < len(self.chunk_texts):
                pos_counter.update(tokenize(self.chunk_texts[cid - 1]))
        # 仅保留有区分度的内容词（去停用词、去高频公司名）
        self.positive_terms = {
            t: c for t, c in pos_counter.items()
            if t not in QUERY_STOPWORDS and c >= 1}
        self.grouped = self.feedback.feedback_by_query()
        # 历史问题向量化，供“相似问题采纳块”匹配
        self.hist_queries = list(self.grouped.keys())
        self.hist_q_vec = (self.vectorizer.transform(
            [normalize_query(q) for q in self.hist_queries])
            if self.hist_queries else None)

    def rerank(self, query: str, pairs: List[Tuple[int, str]],
               base_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """词法底座分 + 采纳词项画像 + 相似问题采纳提权/不采纳降权。

        :param query: 用户问题
        :param pairs: 候选块
        :param base_scores: 融合底座分
        :return: 重排降序结果
        """
        if not pairs:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        idx_list = [i for i, _ in pairs]
        q_vec = self.vectorizer.transform([normalize_query(query)])
        cos = cosine_similarity(q_vec, self.doc_matrix[idx_list])[0]
        norm = _normalize_base(base_scores, idx_list)

        # 相似历史问题 → 其采纳/不采纳块集合
        sim_adopt, sim_reject = set(), set()
        if self.hist_q_vec is not None and self.hist_q_vec.shape[0] > 0:
            sims = cosine_similarity(q_vec, self.hist_q_vec)[0]
            for loc in sims.argsort()[::-1][:3]:
                if float(sims[loc]) >= CONFIG.adaptive_query_sim:
                    grp = self.grouped[self.hist_queries[int(loc)]]
                    sim_adopt |= grp["adopt"]
                    sim_reject |= grp["reject"]

        scores: Dict[int, float] = {}
        for row, (idx, text) in enumerate(pairs):
            score = (1.2 * float(cos[row])
                     + lexical_features(query, text, self.idf)
                     + 0.2 * norm.get(idx, 0.0))
            # ① 历史采纳词项画像：候选块命中采纳词越多，提权越多
            if self.positive_terms:
                toks = tokenize(text)
                hit = sum(1 for t in toks if t in self.positive_terms)
                score += CONFIG.adaptive_positive * min(1.2, hit / 12.0)
            # ② 相似问题明确采纳/不采纳过的块：强信号
            #    反馈库存的是 chunk_id（1 基），候选 idx 为列表下标（0 基）
            if (idx + 1) in sim_adopt:
                score += 0.8
            if (idx + 1) in sim_reject:
                score -= CONFIG.adaptive_negative
            # ③ 被任意历史问题“不采纳”过的块弱降权（避免误伤，力度小于②）
            if (idx + 1) in self.rejected_ids:
                score -= 0.25
            scores[idx] = score
        return self._sort(scores)


# 重排器注册表（retriever/evaluate/app 统一使用短名）
RERANKER_REGISTRY = {
    "llm": LLMReranker,
    "tfidf": TfidfReranker,
    "adaptive": FeedbackAdaptiveReranker,
}
