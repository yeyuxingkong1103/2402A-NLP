"""S6 入库的实现模块。对外唯一入口是 backend/index_milvus.py。

模块划分（按「是否会写库」切分，见 specs/005 的 plan.md）：

    inputs    读四份产物 + V1–V12 校验   —— 纯读，不写
    versions  pipeline_config_hash       —— 纯计算
    store     Milvus 交互                —— **唯一会写库的模块**
    manifest  index_manifest.json        —— 纯文件读写
    report    终端报告渲染                —— 纯输出

这样切分是为了让 spec 的 FR-005「一切校验先于任何写操作」在**结构上可见**：
只有 store 持有 Milvus 连接，前面几个模块想写也写不了。
"""

# ---- collection 契约（docs/04 §9.1，V1 锁定）----

COLLECTION_NAME = "med_rag_v1"
VECTOR_FIELD = "vector"
PRIMARY_FIELD = "chunk_id"

DIM = 1024
INDEX_TYPE = "FLAT"
METRIC_TYPE = "COSINE"

# VARCHAR 的 max_length 按 **UTF-8 字节** 计（Milvus 语义，上限 65535）。
# 实测 text 最长 7,794 字节 —— 注意字节/字符比是 1.9 不是 3（表格含 ASCII 表线）。
MAX_LENGTHS = {
    "chunk_id": 64,
    "doc_id": 64,
    "text": 16384,
    "file_name": 512,
    "section": 512,
    "block_type": 32,
    "source_hash": 64,
    "pipeline_config_hash": 64,
}

# `chunk_meta` 用 JSON 字段装 S4 记录里**没有单独成列**的那部分。这样：
#   · 「这条 1024 维向量是从哪串文本算出来的」有据可查（text_for_embedding）
#   · S4 将来新增字段时，S6 一行都不用改、不用重建 collection
META_FIELD = "chunk_meta"

# Milvus 单个 JSON 字段上限 65536 字节（官方文档）。实测估算最大约 8 KB。
META_MAX_BYTES = 65536

# 键名只允许字母、数字、下划线 —— Milvus 对 JSON 键的硬约束，非法键会让查询解析出错。
META_KEY_PATTERN = r"^[A-Za-z0-9_]+$"

# `chunk_meta` 至少要装这些键。用「至少」而非「恰好」：
# S4 将来加字段会自动进 meta，不该因此报错；但这 6 个少了就是产物出问题了。
META_REQUIRED_KEYS = (
    "text_for_embedding",
    "heading_path",
    "block_ids",
    "sub_index",
    "char_len",
    "chunk_rule_version",
)

# 落库字段全清单（不含 vector）。查回来核对时用它，避免 output_fields=["*"] 把向量也拉回来。
# 这个元组同时驱动三处，改一处即三处生效：schema 核对、回滚备份的取回字段、`chunk_meta` 的排除集。
SCALAR_FIELDS = (
    "chunk_id",
    "text",
    "doc_id",
    "file_name",
    "page_start",
    "page_end",
    "section",
    "block_type",
    "source_hash",
    "pipeline_config_hash",
    META_FIELD,
)

# ---- 连接 ----

DEFAULT_URI = "http://localhost:19530"
TOKEN_ENV = "MILVUS_TOKEN"

# 写后立刻读自己写的数据，必须强一致；否则 count(*) 可能读到旧值，
# 把「读到旧值」误判成「少写了」并触发一次不必要的回滚。在建表时设定。
CONSISTENCY = "Strong"

# ---- 退出码（contracts/cli.md §3；0–3 沿用 docs/05 §5）----

EXIT_OK = 0
EXIT_ARGS = 1
EXIT_VALIDATION = 2
EXIT_DEP = 3
EXIT_GATE = 4
EXIT_ROLLBACK = 5

SEVERITY = (EXIT_ROLLBACK, EXIT_GATE, EXIT_DEP, EXIT_VALIDATION, EXIT_ARGS, EXIT_OK)

# ---- 测试钩子（仅 quickstart §7 用，正常路径不受影响）----

FAULT_ENV = "MEDRAG_TEST_FAULT"
FAULT_INSERT_COUNT = "insert_count"


class IngestError(Exception):
    """带退出码的入库错误。

    注意：**不叫 IndexError** —— 那会遮蔽内建名，让本包内所有 `except IndexError`
    和 numpy 抛出的下标错误都跟着变形。
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message
