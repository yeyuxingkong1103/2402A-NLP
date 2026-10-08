"""S9 混合检索（语义 + BM25 关键词）的实现模块。

对外有**两个**入口：
    · `backend/retrieve_search.py` —— CLI（单条检索 / 纯函数自检 / 语料自检）
    · `backend/api/stream.py`      —— 运行时服务在提问链路上调用

本包同时被这两条路径引用，这是刻意的（FR-023）：**服务端与脚本共用同一份
实现**，两条路径不可能漂移。这与 S8 的 `backend/query/` 是同一取向。

模块划分（按「什么时机可能出错」切分）：

    store     唯一碰 Milvus：连接 + 全量语料 + 向量检索  —— 启动期（库不可达）
    lexical   jieba 分词 + BM25 建索引与检索            —— 启动期（语料为空）
    fuse      RRF 融合 + 去重 + 确定性排序               —— 纯计算
    service   search() 编排 + 阈值判定 + 索引单例        —— 请求期
    report    CLI 报告渲染                              —— 纯输出
    models    契约模型（RetrievedPassage / RetrievalResult）

---

⚠️ **本包 MUST NOT 写向量库。**

`store.py` 是本包唯一持有 Milvus 连接的模块，而它只读。这样切分让「检索不写库」
成为**结构上的事实**，而不是一句约定 —— 与 `backend/index/` 的「只有 store
持有连接」是同一取向，只是方向相反（那个包只写，这个包只读）。

⚠️ **本包 MUST NOT 自行编码问题。**

编码口径由 `backend/embed/model.py` 独占（specs/004 的 D3 裁决）。查询向量由
调用方传入（S8 已算好），本包只消费它。再引入一个编码入口，就是 S8 整篇规格
在防的那件事。
"""

from pathlib import Path

# ---- 路径 ----

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_MANIFEST = PROJECT_ROOT / "data" / "index_manifest.json"

# ---- collection 契约（docs/04 §9.1，与 backend/index/__init__.py 一致）----
#
# ⚠️ 这些常量与 backend/index/ 的**重复声明**是刻意的，不是抄漏。
#    复用 backend.index 会把「写库能力」带进只读路径（那个包含 insert/delete），
#    而本包的结构保证恰恰是"碰不到写操作"。重复的是 4 个字符串，换来的是
#    一条不可能被绕过的边界。

DEFAULT_COLLECTION = "med_rag_v1"
VECTOR_FIELD = "vector"
PRIMARY_FIELD = "chunk_id"

DIM = 1024
METRIC_TYPE = "COSINE"

DEFAULT_URI = "http://localhost:19530"
TOKEN_ENV = "MILVUS_TOKEN"

# 拉全量语料时的上限。**显式给出，MUST NOT 依赖 Milvus 的默认上限** ——
# 与 backend/index/store.py 的 QUERY_LIMIT 同一理由：默认值会让语料被**静默截断**，
# 关键词路悄悄少掉一部分文档，而语义路不受影响。两路从此不一致，且没有报错。
QUERY_LIMIT = 16384

# 从库里拉回来用于展示与建索引的字段。**不含 vector** ——
# 把 1024 维向量拉回来毫无用处，只会让响应体大出两个数量级。
CORPUS_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "text",
    "file_name",
    "page_start",
    "page_end",
    "section",
    "block_type",
)

# 这些字段为空即视为语料有问题（`corpus` 子命令会检查）。
REQUIRED_CORPUS_FIELDS: tuple[str, ...] = ("chunk_id", "text", "file_name", "block_type")

# ---- RRF ----

# 平滑常数（research R5）。60 是原论文（Cormack et al., 2009）的经验值，
# 也是 Elasticsearch / Vespa 等系统的默认取值。**它是平滑常数，不是调参旋钮** ——
# 写进配置只为可追溯，不为日常调整。
DEFAULT_RRF_K = 60

# 关键词路名次 ≤ 此值即可绕过语义阈值纳入（Q4 裁决 / research R7）。
#
# 取 3 而非 1：允许关键词路的前几名都过 —— 只放第 1 名会让"原文在两段里
# 各说一半"这种情况只召回其中一半。
DEFAULT_ADMIT_RANK = 3

# 关键词路准入的**查询词覆盖率**下限（2026-09-28 裁决，Q4 的补充）。
#
# 含义：一个候选要凭关键词路准入，除了 BM25 名次够靠前，还必须**命中了查询词
# 的一半以上**。BM25 名次单独不可用 —— 它对任何查询都必然返回前 N 名，包括
# "今天晚饭吃什么好"这种与知识库无关的提问。
#
# 为什么覆盖率能分开（实测，语料 65 chunks）：
#     氢氯噻嗪（术语照抄）  → 2/2 = 1.00  → 准入
#     高血压               → 1/1 = 1.00  → 准入
#     今天晚饭吃什么好（无关）→ 1/5 = 0.20  → 拦下（正确拒答）
#
# 直觉：关键词准入存在的理由是"用户照抄术语"——照抄时查询词基本都在库里；
# 闲聊式提问的大部分词库里根本没有。**这与语料规模无关**，因此优于
# "BM25 绝对分阈值"（那个换语料就废且废得无声无息）。
#
# ⚠️ 另外两个候选判据都被实测否掉了，记在这里省得后人再试一遍：
#   · IDF 门槛 —— 分不开。「晚饭」idf=3.78、「好」idf=2.69，
#     比「高血压」的 0.42 还高。
#   · 余弦下限 —— 分不开。无关问题最高余弦 0.5198，
#     高于术语问题的 0.5023。
DEFAULT_LEXICAL_MIN_COVERAGE = 0.5

# 每路取多少候选再融合。
DEFAULT_CANDIDATES = 20

# ---- BM25 ----

# Okapi BM25 的标准取值。**写在这里而不是从库里取默认值** ——
# 我们自实现（research R1），没有库可依赖，参数因此必须显式。
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

# ---- 退出码（contracts/cli.md §3；沿用 docs/05 §5 的分工）----

EXIT_OK = 0
EXIT_ARGS = 1
EXIT_DATA = 2
EXIT_DEP = 3

EXIT_DESCRIPTION = {
    EXIT_OK: "成功",
    EXIT_ARGS: "参数错误",
    EXIT_DATA: "数据/校验问题",
    EXIT_DEP: "外部依赖不可用",
}

# ---- 检索状态（data-model.md §6）----
#
# 六种取值，用于日志与 CLI 报告。前四种是**正常结论**，后两种是**故障**。
# 区分二者的意义：把"检索坏了"记成"知识库里没有"，会让一次故障看起来
# 像一次正常的空结果，而用户会换个问法继续试。

STATE_OK = "ok"
STATE_LEXICAL_ONLY = "lexical_only"
STATE_BELOW_THRESHOLD = "below_threshold"
STATE_NO_CANDIDATES = "no_candidates"
STATE_VECTOR_UNAVAILABLE = "vector_unavailable"
STATE_STORE_UNAVAILABLE = "store_unavailable"


class RetrievalError(Exception):
    """检索基础设施失败。**不是「检索为空」。**

    ⚠️ 刻意**不继承** `ValueError` —— 与 `backend/query/QueryError`、
    `backend/index/IngestError` 同一取向：继承内建异常会让 `except ValueError`
    意外捕获到它，而检索链路上游（Pydantic 校验）正在大量抛 `ValueError`。

    ⚠️ 「检索为空」MUST NOT 用本异常表达（docs/05 §4.2 约束 2）。
    空结果返回 `RetrievalResult(is_empty=True)`，是正常返回值。
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def assert_dim_matches_index_package() -> None:
    """启动期自检：本包的 DIM 与 `backend.index` 一致。

    与 `backend/query/__init__.py` 的同名断言是同一取向：DIM 在本包里是
    **重复声明**的（为了不引入 backend.index 的写库能力），重复声明就有漂移的
    可能，而漂移的后果（维度对不上却不报错）正是本特性要防的那一类。

    MUST NOT 在导入期执行 —— 那会让 `import backend.retrieve` 触发
    `backend.index` 的导入链。它由 CLI 与服务端的启动序列显式调用。
    """

    from backend.index import DIM as INDEX_DIM

    if DIM != INDEX_DIM:
        raise RetrievalError(
            EXIT_DATA,
            "维度声明不一致：backend/retrieve 说 %d，backend/index 说 %d" % (DIM, INDEX_DIM),
        )
