"""Query 改写 / 扩写：把用户问题改写成多个检索查询，提升召回。

动机（RAG 优化里的 Query 改写/扩写）：用户的自然语言问题往往不是好的检索查询——
口语化、指代多、措辞与知识库原文不一致。改写让语义更贴近知识库表述，扩写给出
多个角度的查询（细化 / 泛化 / 同义），多查询各自召回后 RRF 融合，召回更稳。

LLM 依赖：改写用 LLM（qwen3）。QUERY_REWRITE_ENABLED 关闭或 LLM 失败时退回原问题，
单查询路径与原先完全一致。
"""
from __future__ import annotations

import re

from ..config import Settings
from ..llm import LLMClient
from ..logging_config import get_logger

log = get_logger("query_rewrite")

# 这段提示词的措辞和下面 rewrite() 的解析是**配套的**，改动前先看解析侧：
#   · 「只输出查询，每行一个」是 splitlines() 能用的前提；模型多说一句解释，那行就会被
#     当成一条检索查询（去吃一次 embedding 和一次 BM25）——错了不报错，只是召回悄悄变差。
#   · 「不要编号」是尽量少触发 _LIST_MARKER 的剥离逻辑（正文本身可能以数字开头，见那里）。
_REWRITE_SYSTEM = (
    "你是检索查询改写器。把用户的问题改写成 2~4 个更适合做向量/关键词检索的查询："
    "一个语义等价的原意改写，其余从不同角度扩写（细化具体点、泛化上位词、补关键术语）。"
    "每个查询独立、简洁、贴近知识库可能的措辞。只输出查询，每行一个，不要编号、不要解释。"
)

# 最多用于检索的查询数（含原问题），改写得再多也截断，避免检索与向量化成本失控
MAX_QUERIES = 4

# 行首的「列表编号」标记：项目符号，或 数字+分隔符（1. / 2、/ 3)）。
#
# 这里刻意**只认带分隔符的编号**，不认「数字 + 空格」：模型偶尔不听话写成编号（提示词已说
# 不要编号），但「2 型糖尿病的诊断标准」这种正文本身就以数字开头。以前的写法是
# `lstrip("-·*0123456789.、 ")`——它按**字符集**剥离，于是把内容开头的数字一并吃掉：
# 实测「2 型糖尿病的诊断标准」→「型糖尿病的诊断标准」、「120/80mmHg 是多少」→「/80mmHg
# 是多少」。这两条被削过的查询会当作独立一路参与 RRF 融合，等于用错查询去召回。
# 数字后没有 `.、)` 分隔符的一律保留，宁可留个多余的编号，也不能改坏查询语义。
_LIST_MARKER = re.compile(r"^\s*(?:[-·*•]|\d{1,2}\s*[.、)．])\s*")


class QueryRewriter:
    def __init__(self, settings: Settings, llm: LLMClient | None = None):
        self.settings = settings
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        """懒建 LLM 客户端，并把它做成可注入的（构造参数 llm=…）。

        两个用处：一是改写默认关闭（QUERY_REWRITE_ENABLED=False），不该为没开的开关
        付构造代价；二是测试要能塞假 LLM（tests/test_query_rewrite.py 就是这么做的），
        注入的路径不能绕开——直接在这里 new 一个就废掉了。
        没上锁：并发首次访问最多各建一个客户端（无副作用），为此加锁不划算。
        """
        if self._llm is None:
            self._llm = LLMClient(self.settings)
        return self._llm

    def rewrite(self, question: str) -> list[str]:
        """改写 + 扩写：返回 [原问题, 改写1, 改写2, ...]；失败/关闭时返回 [原问题]。

        契约：**原问题恒在首位、且永不抛异常**——检索链路靠"至少有一条查询"活着，
        LLM 挂了、返回空、返回垃圾，最差也只是退化成单查询（改写前的行为），不能反过来
        让整个 /chat 失败。

        成本提醒（打开开关前先知道）：每条查询在 HybridRetriever 里都要各自跑一次
        embedding + Milvus 检索 + BM25 检索，条数就是倍数；而这里的一次 LLM 调用是**同步**
        挡在检索前面的。所以开启后延迟和条数一起涨，融合下来候选数也会翻几倍——
        pipeline 的池宽截断（pipeline.py）就是为这个加的。
        """
        if not self.settings.query_rewrite_enabled:
            return [question]
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user", "content": question},
                ]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("query 改写失败，退回原问题: %s", exc)
            return [question]

        queries: list[str] = [question]
        # 逐行一条查询。丢掉三类行：空行、模型把原问题又抄了一遍（否则同一份语料在 RRF 里
        # 被记两次分、虚高）、以及与前面重复的——重复查询同样是给同一批 chunk 重复计分。
        for q in raw.splitlines():
            q = _LIST_MARKER.sub("", q.strip()).strip()
            if q and q != question and q not in queries:
                queries.append(q)
        # 截断放在去重**之后**：先去重再截，模型前几行重复输出时才不会白占名额；
        # 原问题已在首位，所以截断永远不会把原问题截掉。
        return queries[:MAX_QUERIES]
