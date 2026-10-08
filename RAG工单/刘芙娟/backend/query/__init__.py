"""S8 提问落盘与问题向量化的实现模块。

对外有**两个**入口：
    · `backend/query_embed.py`  —— CLI（批量 / 自检 / 清理 / 查看）
    · `backend/api/capture.py`  —— 运行时服务在提问路径上调用

本包同时被这两条路径引用，这是刻意的（FR-032）：**服务端与脚本共用同一份
编码实现**，两条路径不可能漂移。既有的 `backend/embed/` 与 `backend/index/`
都只服务离线管线，本包是第一个跨"离线 / 在线"边界的实现包。

模块划分（按「什么时机可能出错」切分）：

    gate      指纹门禁          —— 读别人的产物判断自己能不能干活
    store     文件读写与清理     —— 唯一碰文件系统的地方
    service   常驻 Encoder + 组装 —— 服务端与 CLI 的共用实现

⚠️ 本包 MUST NOT 自行实现编码。编码口径由 `backend/embed/model.py` 独占
（specs/004 的 D3 裁决）。这里只做「调用它、组装记录、写文件」。
"""

from pathlib import Path

# ---- 路径 ----

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_DIR = PROJECT_ROOT / "data" / "questions"
INDEX_MANIFEST = PROJECT_ROOT / "data" / "index_manifest.json"

FILE_SUFFIX = ".jsonl"
DATE_FORMAT = "%Y%m%d"

# ---- JSONL 字段 ----
#
# ⚠️ 字段名只此一处定义。写入方与读取方引用同一份常量 —— 改名不会只改一半。
#
# FIELD_ORDER 是**约定**：JSON 对象本无序，我们约定它有序。
# 好处是 `grep -o '"question":"[^"]*"'` 这类排障手法可预期（contracts/store.md §2）。

F_ANSWER_ID = "answer_id"
F_QUESTION = "question"
F_ASKED_AT = "asked_at"
F_EMBEDDING = "embedding"
F_VECTOR = "vector"
F_STATUS = "status"
F_ERROR = "error"

FIELD_ORDER: tuple[str, ...] = (
    F_ANSWER_ID,
    F_QUESTION,
    F_ASKED_AT,
    F_EMBEDDING,
    F_VECTOR,
    F_STATUS,
    F_ERROR,
)

REQUIRED_FIELDS: frozenset[str] = frozenset(FIELD_ORDER)

# ---- 状态取值 ----

STATUS_OK = "ok"
STATUS_FAILED = "failed"

# 未向量化时的占位原因。US2 落地后这条路径被 service 取代，
# 保留它是为了让"记录已写入但向量还没算"这个中间态可辨识。
ERR_NOT_EMBEDDED = "尚未向量化"

# ---- 数值契约 ----

# 与 backend.embed.DIM 必须一致。这里重复声明是为了让本包能在**不加载模型**
# 的前提下校验向量维度（门禁与 verify 都跑在模型之前）。一致性由
# tests 之外的断言守着 —— 见 `assert_dim_matches_embedding_package()`。
DIM = 1024

# L2 范数与 1 的允许偏差，沿用 backend/embed/__init__.py 的 NORM_TOLERANCE。
NORM_TOLERANCE = 1e-5

# ---- 退出码（contracts/cli.md §2）----
#
# 沿用 docs/05 §5 的分工：1 参数 / 2 校验 / 3 外部依赖，4 为数据问题
# （backend/index/ 已有同类语义）。

EXIT_OK = 0
EXIT_ARGS = 1
EXIT_GATE = 2
EXIT_MODEL = 3
EXIT_DATA = 4


class QueryError(Exception):
    """带退出码的提问链路错误。CLI 直接用它映射进程退出码。

    刻意**不继承** `ValueError` —— 既有 `IngestError` 的注释里记过这个坑：
    继承内建异常会让 `except ValueError` 意外捕获到它。
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def assert_dim_matches_embedding_package() -> None:
    """启动期自检：本包的 DIM 与 `backend.embed` 一致。

    为什么值得一条断言：DIM 在本包里是**重复声明**的，为的是让门禁与 verify
    能在模型加载之前跑。重复声明就有漂移的可能，而漂移的后果（维度对不上却
    不报错）正是本特性要防的那一类。断言比注释可靠。

    MUST NOT 在导入期执行 —— 那会让 `import backend.query` 触发
    `backend.embed` 的导入，而后者会拉起 torch 的重量级导入链。
    """

    from backend.embed import DIM as EMBED_DIM

    if DIM != EMBED_DIM:
        raise QueryError(
            EXIT_DATA,
            "维度声明不一致：backend/query 说 %d，backend/embed 说 %d" % (DIM, EMBED_DIM),
        )
