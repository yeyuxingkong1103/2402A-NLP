"""检索的数据模型。

三条边界：

1. **`RetrievedPassage` / `RetrievalResult` 的字段名 MUST 与 `docs/05` §4.2
   逐字段一致**，MUST NOT 增删改。前端 `frontend/js/transcript.js` 按这些字段名
   渲染引用卡片，服务端产出的 `citation` 对象也由它们映射而来（FR-015）。
   改这里 = 同时改两处消费方，而其中一处（前端）在本特性里 MUST 保持零改动。

2. **`CorpusChunk` / `Candidate` 是内部结构，MUST NOT 出现在任何响应中。**
   它们存在的意义是让「两路各自的原始名次」在融合后仍然可查 —— 融合规则的有效性
   只能靠回看两路名次来验证（data-model.md §2.1）。

3. **`Candidate` 的 `rrf_score` MUST NOT 泄漏到 `RetrievedPassage.score`。**
   后者恒为余弦相似度（research R6）。理由见 `fuse.py` 的模块文档。
"""

from pydantic import BaseModel

__all__ = ["CorpusChunk", "Candidate", "RetrievedPassage", "RetrievalResult"]


class CorpusChunk(BaseModel):
    """内存语料条目。**两路共用的唯一语料源**（D1 / FR-009）。

    `tokens` 与 `tokens_len` 在**启动期**算好并缓存：查询期的每一次 BM25 打分
    都要用文档长度，重算分词会把查询期成本乘以语料规模。语料是只读的，
    缓存没有失效问题。

    `tokens` 不参与序列化（`exclude=True`）—— 它只服务于 BM25，出现在 CLI 的
    `--json` 输出里只会让输出膨胀几十倍，却没有任何消费方。
    """

    chunk_id: str
    text: str
    file_name: str
    page_start: int
    page_end: int
    section: str | None = None
    block_type: str = "text"

    tokens: list[str] = []
    tokens_len: int = 0


class Candidate(BaseModel):
    """单条候选在两路中的名次。融合前的中间结构，仅内部分析用。

    不变量（data-model.md §2.1）：
        semantic_rank is not None  ⟺  cosine is not None
        lexical_rank  is not None  ⟺  bm25   is not None
        二者至少有一个非 None —— 否则该候选不该存在。
    """

    chunk: CorpusChunk
    cosine: float | None = None
    semantic_rank: int | None = None
    bm25: float | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0

    # 该候选命中的查询词占全部查询词的比例（0.0–1.0）。
    #
    # 由 `service` 在融合后回填 —— `fuse.merge` 不认识查询，只认识两路名次，
    # 这是刻意的：融合是纯函数，不该知道"用户问了什么"。
    #
    # ⚠️ 它是关键词路准入的**必要条件之一**（2026-09-28 裁决）。原因：BM25 名次
    # 单独不可用 —— 它对任何查询都必然返回前 N 名，包括与知识库完全无关的提问。
    # 实测「今天晚饭吃什么好」的 top-3 覆盖率只有 0.20，而「氢氯噻嗪」是 1.00。
    matched_ratio: float = 0.0

    @property
    def hits_both(self) -> bool:
        """两路都命中。这类候选应当排在仅单路命中的之前 —— RRF 的核心效果。"""
        return self.semantic_rank is not None and self.lexical_rank is not None

    @property
    def source_label(self) -> str:
        """CLI 报告里的「来源」列。"""
        if self.hits_both:
            return "两路"
        return "语义" if self.semantic_rank is not None else "关键词"


class RetrievedPassage(BaseModel):
    """一条可展示、可溯源的原文片段。`docs/05` §4.2 的契约，逐字段一致。

    ⚠️ `score` **恒为余弦相似度**（`[-1, 1]`），MUST NOT 写入 RRF 融合分。
    两条理由都是硬的：
      · 前端 `transcript.js` 会 `score.toFixed(2)` 直接显示给用户 ——
        塞入 RRF 分会让用户看到一个与"像不像"无关的数字；
      · 阈值判定用的是相似度（Q4 裁决），`score` 与阈值必须是同一把尺子。

    仅被关键词路命中的片段取 `0.0`（不是 `None`）：契约里 `score: float`
    非可选，且 `0.0` 诚实表达"语义上完全没命中"。用 `None` 会让前端的
    `toFixed` 抛异常。
    """

    chunk_id: str
    file_name: str
    page_start: int
    page_end: int
    section: str | None = None
    block_type: str
    text: str
    score: float

    @classmethod
    def from_candidate(cls, cand: Candidate) -> "RetrievedPassage":
        """从中间结构构造契约模型。**这是两者之间唯一的转换点。**"""

        chunk = cand.chunk
        return cls(
            chunk_id=chunk.chunk_id,
            file_name=chunk.file_name,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section=chunk.section,
            block_type=chunk.block_type,
            text=chunk.text,
            score=float(cand.cosine) if cand.cosine is not None else 0.0,
        )


class RetrievalResult(BaseModel):
    """一次检索的完整结论。`docs/05` §4.2 的契约，逐字段一致。

    三字段语义见 contracts/retrieval.md §2（Q4 裁决的扩展）：
        top_score        最终结果中最大的余弦；全为关键词命中时 0.0；空集时 None
        is_empty         最终结果为空
        below_threshold  结果非空，且所有余弦都低于阈值 —— 即"全部靠关键词救回来的"

    ⚠️ `below_threshold` 在本特性里**改了含义**。原文是"命中但不够像"，现在是
    "知识库里有这个词，但语义模型没理解它"。新定义更可操作：这个信号若随时间
    上升，说明关键词路在承担主要工作，是模型/语料不匹配的早期征兆。
    """

    passages: list[RetrievedPassage] = []
    top_score: float | None = None
    is_empty: bool = True
    below_threshold: bool = False
