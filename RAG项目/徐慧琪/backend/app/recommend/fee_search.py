"""收费口径语料的检索面：集合身份 + 混合检索。

存在的理由（HTTP 接口设计 §三 定调第 3 条）：费用检索原先住在 tools/ingest_fee.py，
而 tools/ 不是生产包的成员 —— 装配一旦搬进 app/core/factory.py，服务进程里
`import ingest_fee` 会当场 ModuleNotFoundError（tools/ 不在 sys.path 上），
正是「CLI 能跑、服务不能」的标本。故检索面与集合身份一起搬进本包；入库面
（schema、切分、写库）留在 tools/ingest_fee.py，那边**从本模块取**集合名与字段口径 ——
方向是 tools → backend（既有的允许方向），写侧与读侧也就不会各存一份。

本模块只调注入进来的 client 与 encoder，不加载任何模型：它的测试因此不必等 2.3GB。
"""
from __future__ import annotations

from pymilvus import AnnSearchRequest, RRFRanker

# 编码与检索参数**全部**从既有实现取，一个数值都不抄第二份 ——
# 抄了就会改一处漏一处（这正是本模块从 tools/ 搬来时保留的既有口径）
from app.db.milvus import DEFAULT_EF, DENSE_TOP_K, RRF_K, SPARSE_TOP_K
from app.ingest.embed import encode_texts
from app.recommend.fees import STATUS_VALID
from app.retrieval.rerank import RERANK_INPUT_TOPK, top_blocks

FEE_COLLECTION = "fee_corpus"

# 检索取回给上层的字段：fees.estimate 读 text（生成要整段、回查要数字）、
# source_doc / source_no（「依据」溯源）与 status（效力闸门：非现行不采用，
# 见 fees.estimate 的效力分支）。前四者都必须是 str —— 渲染方对
# source_doc / source_no 做 " ".join(...)，非 str 会在答案算完之后抛 TypeError
# （见下方 search 的返回）；status 缺了会被归一成空串，编排层按「证明不了现行」
# 判拒（fail-closed），而不是当成缺省放行
FEE_OUTPUT_FIELDS = ["text", "source_doc", "source_no", "status"]

# 效力过滤表达式：与法条侧同口径（app/db/milvus.build_filter_expr 的 status 项）。
# fee_corpus 没有 chunk_type 字段，故不能复用那边整条表达式；这里只留 status 一项。
# 字面量取自 fees.STATUS_VALID 而不是再写一遍「现行有效」：它与编排层的效力闸门
# 必须同源，否则两处各写一份，改一处漏一处
FEE_FILTER_EXPR = f'status == "{STATUS_VALID}"'

# 取回条数。estimate 只取 hits[0]（最相关的一段），多取两条是为了真跑时能看见
# 「同时命中了谁」，人工核对档位时有用
FEE_TOP_K = 3


def search(client, encoder, query: str, top_k: int = FEE_TOP_K,
           scorer=None) -> list[dict]:
    """fee_corpus 混合检索：dense + sparse 双路召回，RRF 融合 + 可选精排。

    为什么不复用 app/db/milvus.hybrid_search：那个函数的集合名与过滤表达式
    （status + chunk_type）都按 law_chunks 的 schema 写死，而 fee_corpus 没有
    chunk_type 字段，传进去会被 Milvus 拒（字段不存在）；改它又会牵动生产集合
    law_chunks 的既有行为。故在这里自己发请求，但**索引参数一个数值都不抄**，
    全从那边取（DENSE_TOP_K / SPARSE_TOP_K / RRF_K / DEFAULT_EF）。

    效力过滤（expr）是**主防线**：设计 §七 要求「命中片段的 status 非现行 →
    不采用该片段」，而语料今天虽然 39 段全是「现行有效」，将来收进失效文件时
    这里若不滤，检索不会报错、只会静默引用一段过期价位。fees.estimate 里还有
    同一判据的兜底（search_fn 可注入，编排层不假定检索方一定过滤了），两层都
    在 —— 只留一层的话，绕开那一层（换检索实现 / 改 output_fields）就静默失效。
    expr 挂在每个 AnnSearchRequest 上（与 app/db/milvus.hybrid_search 同写法）：
    两路召回都要滤，挂一路等于另一路仍然漏。

    `scorer` 是可注入的**精排钩子**（bge-reranker 的 `predict`，与法条侧
    `pipeline.retrieve` 交给 `top_blocks` 的是同一个东西）：技术方案 6.4 明写
    费用检索是「混合检索 **+ 精排**（同 5.2）」。默认 None 时退回纯 RRF ——
    本函数的调用方与用例行为一字不变（精排要加载 2.2GB 模型，不能默认开）。

    返回行**恒有** FEE_OUTPUT_FIELDS 那四个 str 字段；走精排时还多一个
    `rerank_score`（浮点，来自 top_blocks）—— 调用方按需取用，fees.estimate
    只读那四个，多一键不影响任何判据。
    """
    # 空查询返回空，不改判成异常。这里是**退化输入**的护栏而不是正常路径：编码器
    # （app/ingest/embed.py）对空串会抛「sparse 转换后为空」（2026-09-29 真跑实测），
    # 放它抛会被 attach 记成「费用信息暂时不可用」——把「输入不合法」说成「服务故障」
    # 的假警报。正常路径已由调用方兜住：案由认不出时 app.recommend.extras 的
    # search_fn 回退到用户原问句（设计 §七「费用照常走检索」），所以真正走到这里的
    # 只剩「问句本身也是空串」这一种退化。返回空列表由 estimate 归到 no_corpus：
    # 没有查询串就没有证据，不猜数是这里唯一说得通的口径
    if not query.strip():
        return []
    dense, sparse = encode_texts(encoder, [query])[0]
    reqs = [
        AnnSearchRequest(data=[dense], anns_field="dense",
                         param={"metric_type": "COSINE", "params": {"ef": DEFAULT_EF}},
                         limit=DENSE_TOP_K, expr=FEE_FILTER_EXPR),
        AnnSearchRequest(data=[sparse], anns_field="sparse",
                         param={"metric_type": "IP"}, limit=SPARSE_TOP_K,
                         expr=FEE_FILTER_EXPR),
    ]
    # 精排在场时粗排多召回：5.2 的精排输入是「粗排 top 20~50」。只对 RRF 的前
    # top_k 条再排序，精排能做的仅「在这几条里换个序」——候选集本身漏掉了正确
    # 片段时无从纠正，而终审判的正是「top1 取错片段」这一类。无 scorer 时限额
    # 保持 top_k：纯 RRF 路径（含既有用例）一字不变
    recall = RERANK_INPUT_TOPK if scorer is not None else top_k
    res = client.hybrid_search(FEE_COLLECTION, reqs, ranker=RRFRanker(k=RRF_K),
                               limit=recall, output_fields=FEE_OUTPUT_FIELDS)
    # str(... or "") 把「三个契约字段恒为 str」显式钉在这一层：缺字段或值为 None
    # 时退化成空串（渲染方按空值跳过），而不是留一个 None 到渲染期才炸
    hits = [{name: str(dict(hit["entity"]).get(name) or "")
             for name in FEE_OUTPUT_FIELDS} for hit in res[0]]
    if scorer is None:
        return hits
    # 精排后 hits[0] 才是 fees.estimate 取用的那一条。top_blocks 返回新字典
    # （多一个 rerank_score 键、不原地改输入），四个契约字段原样带过去 ——
    # estimate 只读 text / status / source_doc / source_no，多一键不改变判据
    return top_blocks(query, hits, scorer, input_topk=RERANK_INPUT_TOPK,
                      output_topk=top_k)
