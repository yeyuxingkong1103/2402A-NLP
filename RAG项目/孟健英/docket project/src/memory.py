# -*- coding: utf-8 -*-
"""记忆层：短期（Redis 会话历史，TTL 过期）+ 长期（Milvus 摘要存储与召回）。"""
import json  # 消息序列化：Redis 只存字符串，dict 要靠 json 互转
import logging  # 长期记忆沉淀日志：摘要生成/跳过/入库全程可追溯
import re  # 数值+单位正则（150/95、5mg、2片）：个人医疗数据单独触发记忆沉淀
import time  # updated_at 毫秒时间戳：长期记忆按更新时间倒序召回
from typing import List  # 类型标注，让返回值结构可读

import numpy as np  # 向量计算：embedding 转 float32 数组
import redis  # 短期记忆依赖：Redis 内存数据库客户端，key-value 存取 O(1)
from pymilvus import MilvusClient, DataType  # 长期记忆依赖：Milvus 向量数据库客户端 + 字段类型枚举
from sentence_transformers import SentenceTransformer  # 文本向量化模型：把摘要/问题编码成稠密向量

from src.config import settings  # 统一配置中心：Redis 地址、Milvus 集合名、TTL 都从 .env 读

logger = logging.getLogger(__name__)  # 长期记忆日志器，沉淀/跳过/异常都打到 logs/app.log

class ShortTermMemory:  # 短期记忆：Redis List 存最近 20 轮对话，面向"会话级"高频读写
    """基于 Redis List 的多轮对话短期记忆。"""

    def __init__(self, ttl_seconds: int | None = None):  # 初始化连接与过期时间；传 None 则用全局配置
        self.redis = redis.Redis(  # 建立 Redis 连接（内存数据库，读写微秒级）
            host=settings.redis_host,  # 主机地址
            port=settings.redis_port,  # 端口
            db=settings.redis_db,  # 逻辑库编号：按业务隔离 key 空间
            decode_responses=True,  # 返回 str 而非 bytes，省去手动 decode
        )
        self.ttl = ttl_seconds or settings.session_ttl  # 会话过期时间（默认 1800s）：30 分钟不活跃自动释放内存
        self.max_turns = 20  # 最多保留 20 条消息（10 轮）

    def _key(self, session_id: str) -> str:  # 统一 key 命名，避免散落硬编码
        return f"session:{session_id}"  # 带业务前缀的 key：session:xxx，便于按前缀排查和清理

    def add(self, session_id: str, role: str, content: str) -> None:  # 追加一条消息到会话尾部
        """追加一条消息，超过上限时裁剪最旧的。"""
        key = self._key(session_id)  # 先算好 key，下面三条命令复用
        msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)  # 序列化成 JSON 字符串；ensure_ascii=False 保留中文原文
        self.redis.rpush(key, msg)  # RPUSH 尾部追加：List 天然按时间有序，O(1)
        self.redis.expire(key, self.ttl)  # 每次写入刷新 TTL（滑动过期）：持续对话不过期，停下 30 分钟自动清

        while self.redis.llen(key) > self.max_turns:  # LLEN 查长度 O(1)，超出上限就裁；滑动窗口裁剪
            self.redis.lpop(key)  # LPOP 弹掉最旧一条，保持只留最近 20 条

    def get_history(self, session_id: str) -> List[dict]:  # 读取完整会话历史
        """获取完整会话历史（LangChain 格式）。"""
        key = self._key(session_id)  # 算 key
        raw = self.redis.lrange(key, 0, -1)  # LRANGE 0 -1 取全部元素，一次命令拿整段历史
        return [json.loads(m) for m in raw]  # 反序列化回 dict 列表，供拼提示词用

    def clear(self, session_id: str) -> None:  # 清空会话
        """清空会话（结束会话时调用）。"""
        self.redis.delete(self._key(session_id))  # DEL 整个 key：结束会话时释放内存

def _to_python_floats(raw) -> list:  # 向量类型转换工具函数
    """把 numpy 向量强制转成纯 Python float 列表（pymilvus struct.pack 硬性要求）。"""
    return np.asarray(raw, dtype=np.float32).reshape(-1).astype(float).tolist()  # numpy float32 → 原生 float 列表，否则写入会报类型错

_FIRST_PERSON = ("我", "本人", "自己")  # 第一人称词：区分"高血压吃什么药"（科普）与"我在吃降压药"（个人）
_TIME_STATE = ("最近", "现在", "已经", "停了", "换了", "这几天", "昨天", "今天", "一直", "老是")  # 时间/状态词：描述自身近况
_MED_WORDS = ("血压", "血糖", "用药", "药", "剂量", "过敏", "检查", "报告", "症状", "头晕", "失眠",  # 个人医疗关键词（含心理类症状）
              "心慌", "胸闷", "病史", "诊断", "难受", "不舒服", "尿", "焦虑", "抑郁", "压力",
              "心情", "情绪", "睡不好", "睡不着", "紧张", "害怕", "头疼", "头痛", "乏力")
_NUM_UNIT = re.compile(r"\d{2,3}\s*/\s*\d{2,3}|\d+\s*(?:mg|毫克|片|ml|毫升|mmol)", re.I)  # 150/95、5mg、2片 等个人数值

def is_personal(text: str) -> bool:  # 纯知识问答轮跳过 LLM；规则：医疗词+（第一人称或时间状态词），数值+单位单独命中
    if not text:  # 空输入不触发
        return False
    if _NUM_UNIT.search(text):  # 带医疗数值/单位（血压、药量）必属个人陈述
        return True
    return any(w in text for w in _MED_WORDS) and any(w in text for w in _FIRST_PERSON + _TIME_STATE)  # 两类词组合防误伤科普题

class LongTermMemory:  # 长期记忆：Milvus 按 用户×角色×会话 三维隔离；同会话增量合并1行，同角色跨会话聚合召回
    """每段会话一条不断增量更新的健康摘要；读取时按角色聚合该用户全部会话。"""

    def __init__(self, embedder: SentenceTransformer):  # 注入共享向量化模型，与知识库同模型保证向量空间一致
        self.client = MilvusClient(uri=settings.milvus_uri, timeout=10)  # 连接 Milvus，超时 10s 快速失败
        self.embedder = embedder  # 保存向量化模型引用
        self.collection = settings.milvus_memory_collection  # 记忆专用 collection 名，与知识库分开
        self._ensure_collection()  # 幂等建表：不存在才建，重复启动不报错

    def _ensure_collection(self) -> None:  # 确保 collection 存在（Schema：三维隔离字段 + 时间戳）
        """确保 memory collection 存在。"""
        if self.client.has_collection(self.collection):  # 已存在直接返回
            return  # 幂等退出
        dim = len(_to_python_floats(self.embedder.encode(["test"])[0]))  # 试编码拿向量维度（BGE-m3 为 1024 维）
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)  # 主键自增，禁止动态字段保证结构可控
        schema.add_field("id", DataType.INT64, is_primary=True)  # 主键 id
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)  # 摘要向量：维度必须与嵌入模型一致
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)  # 用户维度：当前固定 default
        schema.add_field("role_id", DataType.VARCHAR, max_length=32)  # 角色维度：doctor/tcm/psychologist 互不覆盖
        schema.add_field("session_id", DataType.VARCHAR, max_length=32)  # 会话维度：同角色下 N 段会话各自独立
        schema.add_field("summary", DataType.VARCHAR, max_length=2000)  # 摘要正文（≤300字）
        schema.add_field("updated_at", DataType.INT64)  # 更新毫秒戳：召回按时间倒序，也便于排查
        index = self.client.prepare_index_params()  # 准备索引参数
        index.add_index(field_name="vector", index_type="FLAT", metric_type="IP")  # FLAT 暴力精确（记忆量小）+ 内积（向量归一化后等价余弦）
        self.client.create_collection(self.collection, schema=schema, index_params=index)  # 建表带索引一步到位

    @staticmethod
    def _filter(user_id: str, role_id: str, session_id: str) -> str:  # 统一三维过滤表达式，避免硬编码散落
        return f'user_id == "{user_id}" and role_id == "{role_id}" and session_id == "{session_id}"'

    def _get_old(self, user_id: str, role_id: str, session_id: str):  # 查本会话已有摘要（严格 0 或 1 条）
        rows = self.client.query(self.collection, filter=self._filter(user_id, role_id, session_id),  # 强一致：刚写入立刻能查到
            output_fields=["id", "summary"], consistency_level="Strong", limit=1)
        return rows[0] if rows else None  # 返回含 id 的旧记录，没有返回 None

    def _insert(self, user_id: str, role_id: str, session_id: str, summary: str) -> None:  # 插入一条新摘要（三维+摘要+时间戳）
        vec = _to_python_floats(self.embedder.encode(summary, normalize_embeddings=True))  # 摘要向量化并归一化
        self.client.insert(self.collection, [{"vector": vec, "user_id": str(user_id),
            "role_id": role_id, "session_id": session_id, "summary": summary,
            "updated_at": int(time.time() * 1000)}])  # 毫秒时间戳

    def remember(self, llm, user_id: str, role_id: str, session_id: str, turn: list) -> bool:  # 旧摘要+本轮2条→LLM增量合并，只更新本会话那1行
        from langchain_core.messages import HumanMessage  # 局部导入：记忆层不硬依赖 LangChain
        user_text = next((m["content"] for m in turn if m["role"] == "user"), "")  # 取本轮用户原话（含 OCR 拼接，药名才不丢）
        if not is_personal(user_text):  # 纯知识科普轮不调 LLM 省耗时
            logger.info("长期记忆跳过（非个人陈述）：%s", user_text[:30]); return False  # 跳过留痕
        old = self._get_old(user_id, role_id, session_id)  # 先读旧摘要：绝不无条件删除
        dialog = "\n".join(f"{'患者' if m['role'] == 'user' else '助手'}：{m['content'][:300]}" for m in turn)  # 本轮一问一答，每条截300字
        if old:  # 增量合并：保留仍有效/补充新信息/修正冲突（先说在吃后说停药以最新为准），早期信息由旧摘要承载不依赖 Redis 窗口
            prompt = (f"已有健康摘要：\n{old['summary']}\n\n本轮新对话：\n{dialog}\n\n请更新为一份摘要（300字以内）："
                      "保留仍有效信息、补充新信息、修正冲突；按 用药>过敏>慢性病>近期异常>症状>生活习惯 优先级精简。")
        else:  # 本会话首条：直接总结本轮
            prompt = ("请把以下医患对话总结成患者健康摘要（300字以内），只记患者本人的症状/诊断/用药/过敏/检查指标/血压血糖/生活习惯，"
                      "不写建议；按 用药>过敏>慢性病>近期异常>症状>生活习惯 精简。\n\n" + dialog)
        try:
            summary = llm.invoke([HumanMessage(content=prompt)], timeout=15).content.strip()  # 短超时：记忆沉淀不能拖垮主问答
        except Exception as exc:
            logger.warning("长期记忆摘要失败（不影响问答）：%s", exc); return False  # 失败静默跳过，绝不影响主流程
        if not summary or len(summary) < 5: return False  # 空串/过短不写（"青霉素过敏。"仅6字是有效摘要，阈值不能太高）
        if len(summary) > 300: summary = summary[:300]  # prompt 已要求按优先级精简，这里兜底硬截断
        if old: self.client.delete(self.collection, filter=f'id in [{old["id"]}]')  # 按主键精确删旧（只删本会话1条），不波及其它会话/角色
        self._insert(user_id, role_id, session_id, summary)  # 写入合并后摘要
        logger.info("长期记忆已更新[%s/%s]：%s", role_id, session_id, summary[:60]); return True  # grep 此日志确认写入闭环

    def recall_by_role(self, user_id: str, role_id: str, limit: int = 20) -> List[str]:  # 读：同用户同角色全部会话摘要，最近在前，封顶20段防 prompt 超长
        rows = self.client.query(self.collection, filter=f'user_id == "{user_id}" and role_id == "{role_id}"',  # 不走向量检索：按维度精确取全量一条不漏
            output_fields=["summary", "updated_at"], order_by="updated_at desc", limit=limit, consistency_level="Strong")  # 强一致+时间倒序
        return [r["summary"] for r in rows]  # 摘要原文列表，由 prompts.format_summary 拼背景

    def delete_session(self, user_id: str, role_id: str, session_id: str) -> None:  # 删会话级联：侧栏删对话时同步清理该会话长期记忆
        self.client.delete(self.collection, filter=self._filter(user_id, role_id, session_id))  # 只删该三维坐标下 0/1 条
        logger.info("长期记忆已删除会话[%s/%s]", role_id, session_id)  # 留痕便于核对
