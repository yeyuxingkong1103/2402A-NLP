"""长期记忆的 Milvus 集合管理（自 long_term.py 拆出）。

这里只管"集合长什么样"：集合常量、行的内存表示（MemoryRecord）、
存储异常类型、建集合（幂等）与字段清单。
记忆的读写业务（写入去重、检索、软删除）在 app/memory/long_term.py。

设计红线（用户裁决 + 架构文档 9.3）：
- **独立集合** legal_long_term_memory，严禁与知识库集合 legal_documents 混用——
  从存储层杜绝"过滤条件写漏导致跨用户泄漏"。

字段（docs/技术栈与架构文档.md 9.3 全量落位）：
user_id / character_id / 记忆文本 content / 摘要 summary / 向量 dense_vector /
重要性 importance / 创建时间 created_at / 更新时间 updated_at /
过期时间 expire_at（0=永不过期）/ 来源会话 source_session_id / 删除状态 deleted
"""
from __future__ import annotations

from dataclasses import dataclass

# 默认集合名（独立于知识库集合 legal_documents，见 docs 9.3）
DEFAULT_COLLECTION_NAME = "legal_long_term_memory"


@dataclass(frozen=True)
class MemoryRecord:
    """一条长期记忆（Milvus 行的内存表示）。

    字段：memory_id 主键；user_id 归属用户；character_id 角色（可空）；
    content 记忆正文；summary 摘要；importance 重要性权重；
    created_at/updated_at 秒级时间戳；expire_at 过期时刻（**0 = 永不过期**，
    不用 1970 表达"未知"）；source_session_id 来源会话；deleted 软删除标记。

    为什么 frozen：本类是"从 Milvus 读出来的一行"的快照，业务上不允许就地改
    再写回（改字段必须走 long_term.py 的显式更新方法），冻结能在编码期拦住
    "以为改了内存就等于改了库"这类静默 bug。
    """

    memory_id: str
    user_id: str
    character_id: str
    content: str
    summary: str
    importance: float
    created_at: int
    updated_at: int
    expire_at: int
    source_session_id: str
    deleted: bool


class LongTermMemoryError(RuntimeError):
    """长期记忆存储异常（集合创建失败、向量写入失败等）。

    单独定义类型而不是裸抛 RuntimeError：调用方需要把它与"LLM/检索"的
    RuntimeError 区分开，前者可降级为"本轮不读写记忆"，后者要向上暴露。
    """


class MemoryCollectionMixin:
    """LongTermMemoryStore 的集合管理部分：建集合（幂等）与字段清单。

    为什么用 Mixin 而不是独立类：这两个方法要直接读写宿主对象的
    client / collection_name / dimension / schema_fields，拆成关联类需要
    额外转发层；Mixin 让宿主只做 `class LongTermMemoryStore(MemoryCollectionMixin)`
    就能拿到，同时把"集合结构"与"读写业务"分成两个文件（≤300 行约束）。
    """

    def ensure_collection(self) -> None:
        """创建独立记忆集合（幂等）并加载。字段按 docs 9.3 全量落位。

        参数：无（集合名 / 维度取自宿主对象属性）。
        返回：None。

        幂等：`has_collection` 为真时跳过建表，因此在装配阶段无条件调用是安全的
        （每次进程启动都会调一次，允许重复执行）。
        末尾的 load_collection 不能省：Milvus 新建的集合默认不驻内存，
        不 load 则紧接着的检索会直接失败。
        """
        if not self.client.has_collection(self.collection_name):
            from pymilvus import DataType

            # auto_id=False：主键由业务侧生成，便于"同一事实重复写入"时按 id 去重
            schema = self.client.create_schema(auto_id=False)
            schema.add_field("memory_id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("user_id", DataType.VARCHAR, max_length=64)
            schema.add_field("character_id", DataType.VARCHAR, max_length=64, nullable=True)
            schema.add_field("content", DataType.VARCHAR, max_length=4096)
            schema.add_field("summary", DataType.VARCHAR, max_length=1024)
            schema.add_field("importance", DataType.FLOAT)
            schema.add_field("created_at", DataType.INT64)
            schema.add_field("updated_at", DataType.INT64)
            schema.add_field("expire_at", DataType.INT64)  # 0 = 永不过期
            schema.add_field("source_session_id", DataType.VARCHAR, max_length=64, nullable=True)
            schema.add_field("deleted", DataType.BOOL)

            # 向量维度取自宿主对象属性（与所用 embedding 模型一致），不能写死
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dimension)
            index_params = self.client.prepare_index_params()
            # COSINE 与法条集合保持同一度量口径，两边相似度阈值可直接比较
            index_params.add_index(
                "dense_vector", index_type="AUTOINDEX", metric_type="COSINE"
            )
            self.client.create_collection(
                self.collection_name, schema=schema, index_params=index_params
            )
        self.schema_fields = self._expected_fields()
        self.client.load_collection(self.collection_name)

    @staticmethod
    def _expected_fields() -> list[str]:
        """集合字段清单：写入/查询前用来核对 schema 是否按 9.3 落位。

        参数：无。
        返回：字段名列表（含向量字段 dense_vector）。

        为什么把清单写成字面量而不是从 schema 对象动态取：它同时充当
        "我们期望库里有什么"的断言基准，从实际 schema 反推就失去了校验意义。
        """
        return [
            "memory_id",
            "user_id",
            "character_id",
            "content",
            "summary",
            "importance",
            "created_at",
            "updated_at",
            "expire_at",
            "source_session_id",
            "deleted",
            "dense_vector",
        ]
