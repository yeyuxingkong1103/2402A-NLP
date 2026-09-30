import logging  # 导入日志模块，用于记录运行日志和异常
import math  # 导入数学模块，后面 sigmoid 需要用到 math.exp
from dataclasses import dataclass, field  # 导入 dataclass 和 field，用于定义数据类

from ..config import get_settings  # 从上层 config 模块导入配置获取函数

logger = logging.getLogger(__name__)  # 获取当前模块的 logger 实例


def _sigmoid(x: float) -> float:  # 定义 sigmoid 函数，输入 float，返回 float
    """rerank 原始 logit 归一为 (0,1) 的可读相关度（非真概率）。"""  # 文档字符串：说明用途
    return 1.0 / (1.0 + math.exp(-x))  # sigmoid 公式，把任意实数压到 (0,1)


@dataclass  # 装饰器：自动生成 __init__ 等方法
class Source:  # 定义 Source 数据类，表示单个检索来源片段
    type: str            # "setting" | "memory" | "document"  # 来源类型
    text: str            # 片段原文  # 片段正文
    label: str = ""      # 展示名：文档名 / setting_type / memory_type  # 展示名，默认空
    doc_id: int | None = None  # 文档 ID，默认 None
    chunk_index: int | None = None  # 文档分块序号，默认 None
    score: float | None = None  # 相关度（sigmoid 归一 0~1；回退时 None）  # 相关度分数

    def to_dict(self) -> dict:  # 定义转字典方法
        return {  # 返回一个 dict
            "type": self.type,  # 来源类型
            "label": self.label,  # 展示名
            "text": self.text,  # 片段正文
            "doc_id": self.doc_id,  # 文档 ID
            "chunk_index": self.chunk_index,  # 分块序号
            "score": self.score,  # 相关度
        }


@dataclass  # 装饰器：自动生成 __init__ 等方法
class RetrievalResult:  # 定义检索结果数据类
    settings: list[Source] = field(default_factory=list)  # 角色设定结果列表，默认空列表
    memories: list[Source] = field(default_factory=list)  # 长期记忆结果列表，默认空列表
    documents: list[Source] = field(default_factory=list)  # 文档结果列表，默认空列表


def _ranked_sources(rows: list[dict], ranked: list[tuple[int, float]], *, type: str, label_key: str = "", text_key: str = "text") -> list[Source]:  # 定义按 rerank 顺序组装 Source 的辅助函数
    """按 rerank 返回的 (index, score) 顺序，从原始检索行切出带元数据的 Source。"""  # 文档字符串：说明用途
    out = []  # 初始化输出列表
    for i, score in ranked:  # 遍历 rerank 返回的 (下标, 分数)
        row = rows[i]  # 按下标取出原始检索行
        out.append(Source(  # 构造 Source 并加入输出列表
            type=type,  # 来源类型由调用方传入
            text=row.get(text_key, ""),  # 取正文，默认空字符串
            label=row.get(label_key, "") if label_key else "",  # 取展示名，label_key 为空则用空字符串
            doc_id=row.get("doc_id"),  # 取 doc_id
            chunk_index=row.get("chunk_index"),  # 取 chunk_index
            score=_sigmoid(score) if score is not None else None,  # 分数不为 None 时做 sigmoid 归一，否则保持 None
        ))
    return out  # 返回组装好的 Source 列表


class RAGPipeline:  # 定义 RAG 检索流水线类
    def __init__(self, embedding, rerank, milvus):  # 构造函数，注入三个依赖
        self.embedding = embedding  # 保存 embedding 组件
        self.rerank = rerank  # 保存 rerank 组件
        self.milvus = milvus  # 保存 milvus 客户端
        self.top_k = get_settings().retrieval_top_k  # 读取召回条数配置
        self.top_m = get_settings().rerank_top_m  # 读取 rerank 保留条数配置
        self.doc_top_k = get_settings().doc_top_k  # 读取文档召回条数配置

    async def retrieve(self, character_id: int, user_id: int, query: str) -> RetrievalResult:  # 异步检索入口
        result = RetrievalResult()  # 初始化结果对象

        try:  # 尝试检索角色设定
            settings = await self.milvus.hybrid_search(  # 调用 milvus 混合检索
                "character_settings", query, f"character_id == {character_id}", self.top_k)  # 指定集合、query、过滤条件、top_k
        except Exception:  # 捕获异常
            logger.exception("hybrid_search(character_settings) failed")  # 记录异常日志
            settings = []  # 降级为空列表

        passages = [s["text"] for s in settings]  # 提取待重排文本
        if passages:  # 如果有文本
            ranked = await self._rerank_safe(query, passages, "character_settings")  # 安全 rerank
            result.settings = _ranked_sources(settings, ranked, type="setting", label_key="setting_type")  # 组装结果

        try:  # 尝试检索长期记忆
            mems = await self.milvus.hybrid_search(  # 调用 milvus 混合检索
                "long_term_memory", query,  # 指定集合和 query
                f"user_id == {user_id} && character_id == {character_id}", self.top_k)  # 过滤条件
        except Exception:  # 捕获异常
            logger.exception("hybrid_search(long_term_memory) failed")  # 记录异常日志
            mems = []  # 降级为空列表

        passages = [m["content"] for m in mems]  # 提取待重排文本（记忆用 content 字段）
        if passages:  # 如果有文本
            ranked = await self._rerank_safe(query, passages, "long_term_memory")  # 安全 rerank
            result.memories = _ranked_sources(mems, ranked, type="memory", label_key="memory_type", text_key="content")  # 组装结果

        try:  # 尝试检索文档
            docs = await self.milvus.hybrid_search(  # 调用 milvus 混合检索
                "documents", query, "id > 0", self.doc_top_k)  # 指定集合、query、过滤条件、top_k
        except Exception:  # 捕获异常
            logger.exception("hybrid_search(documents) failed")  # 记录异常日志
            docs = []  # 降级为空列表

        passages = [d["text"] for d in docs]  # 提取待重排文本
        if passages:  # 如果有文本
            ranked = await self._rerank_safe(query, passages, "documents")  # 安全 rerank
            result.documents = _ranked_sources(docs, ranked, type="document", label_key="source")  # 组装结果

        return result  # 返回聚合检索结果

    async def _rerank_safe(self, query: str, passages: list[str], label: str) -> list[tuple[int, float | None]]:  # 定义安全 rerank 方法
        """rerank 失败时不静默返回空，回退到未重排的原顺序（相关度置 None，前端显示「—」）。"""  # 文档字符串：说明降级策略
        try:  # 尝试正常 rerank
            return await self.rerank.rerank(query, passages, self.top_m)  # 调用 rerank 组件
        except Exception:  # 捕获异常
            logger.exception("rerank(%s) failed, falling back to un-ranked order", label)  # 记录异常日志
            return [(i, None) for i in range(len(passages))]  # 回退：保留原顺序，分数置 None