# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""全文检索：倒排索引 + 布尔/短语/模糊查询 + 多字段

工单6 要求「使用倒排索引技术实现全文检索，支持布尔查询、短语匹配和模糊匹配，
支持多字段检索（如标题、正文、摘要）」。

中文分词不引入额外依赖：用**字符二元组（bigram）**建索引。中文里「军用领域」
切成 军用/用领/领域，检索时同样切，能匹配上；ASCII 词按非字母数字切分成整词。
这样无需 jieba 之类的分词库，对招股书这类术语密集的文本效果够用。
"""
import re
from collections import defaultdict

# 一个「词」：连续的汉字串，或连续的字母数字串
_TOKEN = re.compile(r"[一-龥]+|[A-Za-z0-9]+")


def tokenize(text):
    """中文字符二元组 + 英文整词。"""
    tokens = []
    for piece in _TOKEN.findall(text or ""):
        if piece[0].isascii():                 # 英文/数字：整词，另加小写形式
            tokens.append(piece.lower())
        elif len(piece) == 1:                  # 单个汉字
            tokens.append(piece)
        else:                                  # 汉字串切成二元组
            tokens.extend(piece[i:i + 2] for i in range(len(piece) - 1))
    return tokens


def _edit_distance_within(a, b, limit):
    """编辑距离是否 <= limit（带提前退出，比完整 DP 快）。"""
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


class FullTextIndex:
    """倒排索引。字段分「标题」与「正文」两路，可按字段加权。"""

    def __init__(self, title_weight=2.0, body_weight=1.0, summary_weight=1.3):
        self.title_weight = title_weight
        self.body_weight = body_weight
        self.summary_weight = summary_weight      # 摘要比正文略高
        self.docs = []                          # 原始块
        self.postings = defaultdict(dict)        # token -> {doc_id: 加权词频}
        self.title_text = []                    # 标题（短语匹配用）
        self.body_text = []                     # 正文
        self.summary_text = []                  # 摘要

    # ---------- 建索引 ----------
    SUMMARY_LEN = 120

    @classmethod
    def split_fields(cls, chunk):
        """把一块拆成（标题, 正文, 摘要）三个字段。

        工单要求「支持多字段检索（如标题、正文、摘要）」，三个字段都建索引：

          标题  表格/图像块取自带的题注（`【…】`），文本块取首行 ——
                招股书的首行往往就是所在小节标题
          正文  整块内容
          摘要  正文开头的一段 —— 开篇通常最能代表这块在讲什么，
                检索时给比正文略高的权重
        """
        text = chunk.get("text", "")
        title, body = "", text
        if chunk.get("type") in ("table", "image") and text.startswith("【"):
            end = text.find("】")
            if end > 0:
                title, body = text[1:end], text[end + 1:]
        else:
            first_line = text.split("\n", 1)[0].strip()
            if len(first_line) < 40:
                title = first_line
        return title, body, body[:cls.SUMMARY_LEN]

    def build(self, chunks):
        self.docs = list(chunks)
        self.postings.clear()
        self.title_text, self.body_text, self.summary_text = [], [], []
        for i, chunk in enumerate(self.docs):
            title, body, summary = self.split_fields(chunk)
            self.title_text.append(title)
            self.body_text.append(body)
            self.summary_text.append(summary)
            counts = defaultdict(float)
            for tok in tokenize(title):          # 三个字段都进倒排表
                counts[tok] += self.title_weight
            for tok in tokenize(summary):
                counts[tok] += self.summary_weight
            for tok in tokenize(body):
                counts[tok] += self.body_weight
            for tok, weight in counts.items():
                self.postings[tok][i] = weight
        return self

    # ---------- 查询解析 ----------
    @staticmethod
    def _terms(query):
        """把查询切成 (是否排除, 词) 列表，支持 NOT/- 前缀。"""
        out = []
        for raw in re.split(r"\s+", (query or "").strip()):
            if not raw:
                continue
            neg = raw.startswith(("NOT ", "-", "!"))
            raw = re.sub(r"^(NOT|not)", "", raw).lstrip("-!")
            if raw:
                out.append((neg, raw.strip('"“”')))
        return out

    def _match_token(self, term, fuzzy):
        """把查询词展开成实际索引词：精确 / 模糊（~n 或编辑距离 1）。"""
        direct = tokenize(term)
        if not fuzzy:
            return direct
        expanded = set(direct)
        for tok in list(self.postings):
            if _edit_distance_within(tok, term.lower(), fuzzy):
                expanded.add(tok)
        return expanded

    # ---------- 检索 ----------
    def search(self, query, top_k=10, mode="and", fuzzy=0, phrase=True):
        """执行一次全文检索。

        mode: and（全部词命中）/ or（任一词命中）
        fuzzy: 编辑距离容错，0 表示关闭
        phrase: 是否对引号内的短语做整体匹配加分
        """
        terms = self._terms(query)
        if not terms:
            return []

        scores, hit_count = defaultdict(float), defaultdict(int)
        for neg, term in terms:
            matched = self._match_token(term, fuzzy)
            doc_ids = set()
            for tok in matched:
                doc_ids.update(self.postings.get(tok, {}))
                for doc_id, weight in self.postings.get(tok, {}).items():
                    scores[doc_id] += weight
            if neg:                                   # NOT：整条命中集里剔除
                for doc_id in list(scores):
                    if doc_id in doc_ids:
                        scores.pop(doc_id)
                        hit_count.pop(doc_id, None)
            else:
                for doc_id in doc_ids:
                    hit_count[doc_id] += 1

        # and：要求每个正向词都命中
        positive = sum(1 for neg, _ in terms if not neg)
        candidates = [d for d in scores
                      if mode != "and" or hit_count[d] >= positive]

        # 短语匹配加分：引号里的短语整体出现，额外加权
        if phrase:
            for quoted in re.findall(r'"([^"]+)"|“([^”]+)”', query or ""):
                p = (quoted[0] or quoted[1]).strip()
                if len(p) < 2:
                    continue
                for doc_id in candidates:
                    if p in self.body_text[doc_id] or p in self.title_text[doc_id]:
                        scores[doc_id] += 5.0 * self.title_weight

        ranked = sorted(candidates, key=lambda d: -scores[d])[:top_k]
        return [(self.docs[d], float(scores[d])) for d in ranked]
