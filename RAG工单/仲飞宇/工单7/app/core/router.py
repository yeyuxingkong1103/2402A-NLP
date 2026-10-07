# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单03 - PDF文档的表格解析及检索优化
"""
实体路由：从问题里认出公司在说哪份文档，把检索限定到那一份。

【为什么必须做】两份 PDF 都是招股意向书，章节结构雷同，实测这些关键词**两边都有**：

    关键词              招股书1   招股书2
    发行股数              3 页     3 页     ← 两边的股数完全不同！
    占发行后总股本         3 页     3 页
    募集资金             21 页    29 页

不区分就会张冠李戴。而两家公司的名字是**完美区分器**：

    赵马克（力源实控人）   0 页    52 页
    程家明（兴图实控人）  73 页     0 页

【为什么是硬过滤而不是"加权提权"】上表那三组词两边都有，靠排序**无法可靠区分**
（相关性分数会因为上下文措辞而上下浮动）。而问题里点名了公司，匹配是确定的。
所以直接按 `doc_name` 过滤，不赌排序。

【为什么过滤用 doc_name 而不是 doc_id】doc_id 是入库时按「文件名:大小:mtime」
算的 sha256（见 pipeline.doc_id_of）—— 文件被重新保存一次（mtime 变）就对不上了。
doc_name 是文件名，稳定且可读，且已经在 Milvus 里存着。邻块查询仍用 doc_id
（那是从命中行里带出来的，天然正确）。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.doc_profiles import DOC_PROFILES, doc_ids_by_entity


@dataclass(frozen=True)
class RouteResult:
    """路由结果。"""

    doc_key: str | None = None      # 命中的文档 key；None = 认不出，不过滤
    doc_name: str | None = None     # 命中的文件名
    matched: str = ""               # 命中的实体专名（便于排查）
    reason: str = ""

    @property
    def routed(self) -> bool:
        return self.doc_name is not None

    @property
    def expr(self) -> str:
        """给 Milvus `search(filter=...)` 的表达式。空串 = 不过滤。"""
        return f'doc_name == "{self.doc_name}"' if self.doc_name else ""


def route(question: str) -> RouteResult:
    """按问题里出现的公司专名路由。

    【最长匹配优先】`武汉力源信息技术股份有限公司` 必须先于 `武汉力源` 命中 ——
    否则短名先匹配，虽然结果一样（同一个文档），但 `matched` 会记成短名，
    排查时看不出到底匹配到了什么。doc_profiles 的 all_entity_names() 已按长度降序。

    【工单07 补的次级判据：等长时谁先出现谁赢】
    加工单07 的 9 份年报后出现了**等长专名**：`平安银行`（平安银行2019年报）与
    `中国平安`（中国平安2019年报）都是 4 个字。问题里同时出现两者时，
    只按长度排序会**按 dict 的插入顺序**决定胜负 —— 那是「配置写在前面的赢」，
    与问题在问谁毫无关系，还会随 DOC_PROFILES 的书写顺序静默漂移。
    改为「等长时取问题中**位置更靠前**的那个」：一句话里先点名的公司就是主题，
    这与人的读法一致，也把不确定性钉死了。
    """
    if not question:
        return RouteResult(reason="空问题")

    mapping = doc_ids_by_entity()
    hits = [(name, question.find(name)) for name in mapping if name in question]
    if not hits:
        return RouteResult(reason="未识别到公司专名，全库检索")
    name, _pos = min(hits, key=lambda kv: (-len(kv[0]), kv[1]))
    key = mapping[name]
    prof = DOC_PROFILES[key]
    return RouteResult(doc_key=key, doc_name=prof.doc_name,
                       matched=name, reason=f"命中实体「{name}」")


def route_multi(question: str) -> list[str]:
    """命中的**所有**文档（用于检测"一句话里提了两家公司"）。

    这种情况不做过滤：问题同时涉及两份文档，限定任何一份都会漏。
    """
    mapping = doc_ids_by_entity()
    hits = {mapping[n] for n in mapping if n in question}
    return sorted(hits)
