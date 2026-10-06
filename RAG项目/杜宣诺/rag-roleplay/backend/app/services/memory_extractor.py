import json  # 导入 json 模块，用于解析 LLM 返回的 JSON

from pydantic import BaseModel  # 导入 pydantic 的 BaseModel，用于定义数据模型

from ..config import get_settings  # 从上层 config 模块导入配置获取函数


class Memory(BaseModel):  # 定义记忆数据模型
    type: str  # 记忆类型：fact / event / preference
    content: str  # 记忆正文
    importance: int  # 重要度，0-10


EXTRACT_PROMPT = """从以下对话中抽取值得长期记住的信息，只抽取事实、事件、用户偏好三类。  # 抽取记忆的提示词模板
以 JSON 返回，格式：{"memories": [{"type": "fact|event|preference", "content": "...", "importance": 0-10}]}  # 指定返回的 JSON 结构
没有则返回 {"memories": []}。  # 没有可抽取内容时的返回约定

对话：
{conversation}
"""


def _cosine(a: list[float], b: list[float]) -> float:  # 定义余弦相似度函数
    dot = sum(x * y for x, y in zip(a, b))  # 计算向量点积
    na = sum(x * x for x in a) ** 0.5  # 计算 a 的模长
    nb = sum(x * x for x in b) ** 0.5  # 计算 b 的模长
    return dot / (na * nb) if na and nb else 0.0  # 模长非零时返回余弦相似度，否则返回 0.0


class MemoryExtractor:  # 定义记忆抽取器类
    def __init__(self, llm, embedding, milvus):  # 构造函数，注入三个依赖
        self.llm = llm  # 保存 LLM 组件
        self.embedding = embedding  # 保存 embedding 组件
        self.milvus = milvus  # 保存 milvus 客户端
        self.threshold = get_settings().memory_dedup_threshold  # 读取去重相似度阈值配置

    async def extract(self, conversation: list[dict]) -> list[Memory]:  # 异步从对话中抽取记忆
        text = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)  # 把对话拼成 "role: content" 的文本
        raw = await self.llm.chat([  # 调用 LLM 抽取记忆
            {"role": "system", "content": "你是记忆抽取器，只输出 JSON。"},  # system 提示：要求只输出 JSON
            {"role": "user", "content": EXTRACT_PROMPT.replace("{conversation}", text)},  # user 提示：填入对话文本
        ])
        try:  # 尝试解析 LLM 返回
            data = json.loads(raw)  # 把返回字符串解析成 dict
        except json.JSONDecodeError:  # 解析失败
            return []  # 返回空列表
        return [Memory(**m) for m in data.get("memories", [])]  # 把每条记忆构造成 Memory 对象

    async def dedupe(self, memories: list[Memory], character_id: int, user_id: int) -> list[Memory]:  # 异步去重
        if not memories:  # 没有记忆
            return []  # 直接返回空列表
        existing = await self.milvus.hybrid_search(  # 从 milvus 检索已有记忆
            "long_term_memory", " ".join(m.content for m in memories),  # 集合名 + 用新记忆内容拼成的查询
            f"user_id == {user_id} && character_id == {character_id}", top_k=10,  # 过滤条件 + top_k
        )
        kept = []  # 初始化保留列表
        for m in memories:  # 遍历每条新记忆
            emb = (await self.embedding.encode_dense([m.content]))[0]  # 编码该记忆为稠密向量
            too_similar = any(  # 判断是否与已有记忆过于相似
                _cosine(emb, e.get("dense_vector", [])) > self.threshold for e in existing  # 与每条已有记忆算余弦相似度并比阈值
            )
            if not too_similar:  # 如果不相似
                kept.append(m)  # 保留该记忆
        return kept  # 返回去重后的记忆列表

    async def run(self, session_id: int, user_id: int, character_id: int, conversation: list[dict]) -> int:  # 异步执行完整流程
        memories = await self.extract(conversation)  # 先抽取记忆
        memories = await self.dedupe(memories, character_id, user_id)  # 再去重
        if memories:  # 如果还有记忆
            rows = [{  # 组装待写入 milvus 的行
                "user_id": user_id, "character_id": character_id,  # 用户/角色 ID
                "memory_type": m.type, "content": m.content,  # 记忆类型和正文
                "importance": m.importance, "source_msg_ids": str(session_id),  # 重要度和来源会话 ID
                "created_at": 0,  # 创建时间，暂固定为 0
            } for m in memories]
            await self.milvus.upsert_memories(rows)  # 写入 milvus
        return len(memories)  # 返回最终写入的记忆条数