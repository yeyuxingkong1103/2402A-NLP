# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单06 - 混合检索（向量检索 / 全文检索 / 混合检索）
"""
文本分析：**全项目唯一的切词实现**。

【为什么必须收敛成一处】工单06 一下有三处要用"同一套切词"：
  1. Milvus 稀疏向量 —— 入库时写、查询时算（`vectorstore.lexical_sparse`）
  2. 应用层倒排索引 —— BM25F、位置、短语匹配（`app/core/fulltext.py`）
  3. 查询解析器 —— 把用户输入切成 term 再去匹配上面两个
规则一旦漂移，就会出现"稀疏路召回了这块、倒排路却没召回"，而这种分歧
**不报错**，只会让混合检索的对照实验变成无头案。所以：切词只此一家。

【数字保护】`5,520.00`、`1-1-128` 这类 token 必须**整体**保留 —— 招股书的核心答案
大量是数字与页码，被 jieba 切成 `5` `,` `520` `.` `00` 之后就废了。
"""

from __future__ import annotations

import re
import zlib

import jieba

# 页码形态（招股书1 的页脚是 `1-1-N`）
_CN_NUM = re.compile(r"1-1-\d+")
# 一般数字：含千分位逗号、小数点、负号、百分号
_NUM_TOKEN = re.compile(r"\d[\d,.\-]*%?")


def term_id(term: str) -> int:
    """token → 稀疏向量的维度。**内容稳定**，与进程无关。

    【为什么不能用内置 hash()】Python 3 对 str 的 hash 是**逐进程随机化**的
    （PYTHONHASHSEED 未固定时）。而入库是一个进程、查询是另一个进程 ——
    两边算出的维度对不上，Milvus 稀疏检索就会**静默失效**：不报错、召回率≈0，
    表现为"关键词怎么搜都搜不到"。
    实测（同一台机器、连续两个进程）：`hash("万元")` 分别是
    -8503448058700183264 与 4762181532052463439。

    改用 crc32（内容寻址）并只取 31 位：既跨进程稳定，又与旧实现占用的
    维度空间一致（不超出 Milvus 稀疏向量的维度上限）。22424 个词映射到
    2^31 个桶，碰撞概率可忽略。
    """
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def tokenize_positions(text: str) -> list[tuple[str, int]]:
    """切成 `(token, 位置)` 序列。位置 = 在本次切词结果里的下标。

    【为什么要位置】工单06 的"短语匹配"要求 token 相邻（`"军用领域的收入"`），
    没有位置就只能退化成"这几个词都出现过"，那是布尔 AND，不是短语。

    【为什么按 span 排序并去重叠】旧实现是"先把所有数字抠出来、再把 jieba 的词
    补在后面"—— 于是同一个数字区域会被 jieba 重复贡献碎片（`520`、`00` 都成了
    独立 token），而且**顺序是乱的**（数字全排在中文前面）。顺序乱就没法做位置。
    现在改成一次左到右扫描、按 span 排序，并丢掉与数字区域重叠的 jieba 碎片。
    """
    spans: dict[tuple[int, int], str] = {}
    for rx in (_CN_NUM, _NUM_TOKEN):
        for m in rx.finditer(text):
            tok = m.group().rstrip(".,")
            if len(tok) >= 2:
                spans.setdefault((m.start(), m.end()), tok)
    num_ranges = list(spans)

    for word, s, e in jieba.tokenize(text):
        w = word.strip()
        if len(w) < 2 and not w.isdigit():
            continue
        if any(s < ne and ns < e for ns, ne in num_ranges):
            continue                      # 与数字区域重叠 → 是 jieba 切出的碎片
        spans.setdefault((s, e), w)

    return [(spans[k], i) for i, k in enumerate(sorted(spans))]


def tokenize(text: str) -> list[str]:
    """切成 token 序列（保留顺序与重复）。入库与倒排索引用它。"""
    return [t for t, _ in tokenize_positions(text)]


def is_numeric_token(token: str) -> bool:
    """token 是不是"数字类"（含千分位/小数点/百分号）。

    【为什么模糊匹配要问这个】招股书里数字**就是答案** —— 给 `5,520` 做编辑距离
    会命中 `5,530`，等于把答案改错。所以模糊匹配只作用于中文词，数字只做
    "归一化后的精确等价"。
    """
    return bool(_NUM_TOKEN.fullmatch(token) or _CN_NUM.fullmatch(token))


# ----------------------------------------------------------------------
# 查询词集合（工单02 的词法重排 / 工单06 的 TF-IDF 重排共用）
# ----------------------------------------------------------------------
# 停用词：这些词在招股书里处处出现，对区分度没有贡献，留在查询词集合里
# 会稀释词法召回率。
STOPWORDS: set[str] = set("""的 了 吗 呢 是 和 与 及 或 在 有 为 对 不 都 很 就 也 还 这 那
哪些 哪个 多少 如何 什么 请问 根据 报告期内 公司 招股意向书 分别 主要 包括""".split())


def query_tokens(text: str) -> set[str]:
    """查询侧的词集合：jieba 切词，去掉 1 字词与停用词。

    【注意与 tokenize() 的区别 —— 两者服务的目的不同，**不要合并**】
      · `tokenize()`   —— 建索引用：保留数字整体、保留位置、**不过滤停用词**
        （倒排索引要能按任意词查，包括"公司"）。
      · `query_tokens()` —— 算"查询词覆盖率"用：必须过滤停用词，否则
        「哪些/多少/公司」这些到处都有的词会把覆盖率顶满，指标失真。
    历史上它叫 `retriever._tokens`（工单02 写的），工单06 挪到这里共享给
    重排器 —— 行为逐字节不变，测试里有回归闸门。
    """
    return {t for t in jieba.cut(text) if len(t) >= 2 and t not in STOPWORDS}


# ----------------------------------------------------------------------
# Query 抽象（工单02 写的，工单06 挪来共享给重排器）
# ----------------------------------------------------------------------
def entity_names() -> list[str]:
    """本库的「实体专名」——查询时抽象掉它们（按长度降序，最长匹配优先）。

    来源以 `doc_profiles` 为准（已配置文档的专名是唯一事实来源），
    `settings.query_entity_names` 作为兼容性补充项。
    """
    from app.config import settings
    from app.core.doc_profiles import all_entity_names
    names = list(all_entity_names())
    for n in getattr(settings, "query_entity_names", []) or []:
        if n and n not in names:
            names.append(n)
    return sorted(names, key=len, reverse=True)


def abstract_query(question: str) -> str:
    """抽象化：去掉文档自身的实体名与「根据…招股意向书，」这类引导语。

    【为什么这一步是决定性的】工单01 的题全写成「武汉兴图新科电子股份有限公司…」，
    而招股书正文**从不自称全称**（一律用「公司/发行人」）。这 16 个字会主导查询，
    也会稀释「查询词覆盖率」—— 公司全称那些 token 在任何片段里都不出现，
    于是 lex 被整体压低、区分度下降。实测去掉后 recall@5 从 7/10 升到 9/10。
    """
    import re
    q = question
    for name in entity_names():
        q = q.replace(name, "")
    q = re.sub(r"根据\s*[，,]?\s*", "", q)
    q = re.sub(r"^[，,、\s]+", "", q)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q or question
