"""长期记忆：从对话抽取客观事实 -> 向量化存入 Milvus user_memory -> 按 user_id 召回。

所有操作 try/except 降级（失败仅 warning、返回空），绝不阻断主问答流程。
抽取由 rag.py 每 5 轮触发一次，走 fire-and-forget 线程，不阻塞请求。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading

from ingest import load_embedder
from milvus_store import (
    MEMORY_COLLECTION,
    create_memory_collection,
    insert,
    search_memory,
)

logger = logging.getLogger(__name__)

MAX_FACTS = 5       # 每次抽取最多保留的事实条数
MEMORY_TOP_K = 3    # 每次召回的事实条数

EXTRACT_PROMPT = """你是信息抽取助手。从以下对话中抽取关于【用户】的客观、稳定、长期有效的事实，供医生后续了解患者背景。

只抽取以下类型（客观且不易随时间改变）：
- 健康史 / 家族史：如「用户有高血压」「用户父亲有糖尿病」「用户对青霉素过敏」
- 长期用药或治疗：如「用户正在长期服用氨氯地平」
- 基本身份与长期属性：年龄、职业、身高体重、是否吸烟饮酒等

严禁抽取：
- 情绪、感受、临时症状（如「今天头晕」「有点紧张」）
- 单次问答的瞬时内容（如某次血压读数、某个具体提问）
- 医生/助手说的话（那不是关于用户的事实）
- 推测、不确定、假设的内容

如果对话中没有符合条件的客观事实，输出空数组 []。

只输出 JSON 数组，不要输出任何其他文字或解释，最多 5 条：
["事实1", "事实2"]"""


def _history_text(history: list[dict]) -> str:
    """把 [{role, content}] 格式化成「用户：/医生：」逐行文本。"""
    lines = []
    for msg in history:
        speaker = "用户" if msg.get("role") == "user" else "医生"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{speaker}：{content}")
    return "\n".join(lines)


def _fact_id(user_id: str, content: str) -> str:
    """MD5 去重主键：同 user_id 下同 content 生成同 id，Milvus insert 天然覆盖。"""
    return hashlib.md5(f"{user_id}\x00{content}".encode("utf-8")).hexdigest()


def extract_facts(history: list[dict]) -> list[str]:
    """DeepSeek 抽取客观事实，返回最多 MAX_FACTS 条；失败返回 []。"""
    from rag import DEEPSEEK_MODEL, _get_client  # 延迟 import 复用客户端，避免循环 import

    if not history:
        return []
    text = _history_text(history)
    if not text:
        return []
    prompt = EXTRACT_PROMPT + "\n\n对话内容：\n" + text
    try:
        resp = _get_client().chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip()).strip()
        arr = json.loads(raw)
        if not isinstance(arr, list):
            return []
        facts = [str(f).strip() for f in arr if isinstance(f, str) and f.strip()]
        return facts[:MAX_FACTS]
    except Exception as e:  # noqa: BLE001
        logger.warning("事实抽取失败，返回空: %s", e)
        return []


def save_facts(user_id: str, facts: list[str]) -> None:
    """向量化 + 写入 user_memory（MD5 去重）；失败仅告警不抛错。"""
    if not facts:
        return
    try:
        create_memory_collection()
        embedder = load_embedder()
        vecs = embedder.encode(facts, normalize_embeddings=True).tolist()
        records = [
            {"id": _fact_id(user_id, f), "user_id": user_id, "content": f, "embedding": v}
            for f, v in zip(facts, vecs)
        ]
        insert(MEMORY_COLLECTION, records)
        logger.info("长期记忆写入 %d 条（user_id=%s）", len(records), user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("长期记忆写入失败: %s", e)


def recall_facts(user_id: str, query: str, top_k: int = MEMORY_TOP_K) -> list[str]:
    """按 user_id 召回与 query 相关的事实文本；失败返回 []。"""
    try:
        embedder = load_embedder()
        vec = embedder.encode([query], normalize_embeddings=True).tolist()[0]
        return search_memory(user_id, vec, top_k)
    except Exception as e:  # noqa: BLE001
        logger.warning("长期记忆召回失败: %s", e)
        return []


def schedule_extract(user_id: str, history: list[dict]) -> None:
    """fire-and-forget：后台线程抽取 + 保存，不阻塞主请求。"""
    threading.Thread(
        target=_extract_and_save, args=(user_id, history), daemon=True
    ).start()


def _extract_and_save(user_id: str, history: list[dict]) -> None:
    """后台线程体：抽取事实，有则写入。"""
    facts = extract_facts(history)
    if facts:
        save_facts(user_id, facts)
"""
memory 长期记忆模块的作用：从对话中提取用户稳定客观信息，存入独立记忆向量库，后续提问时召回，让 AI 记住用户长期背景。第一部分，`schedule_extract()`开启后台守护线程，执行`_extract_and_save()`，采用 fire-and-forget 模式。之所以放后台线程，是因为调用大模型抽取事实会消耗时间；如果在主请求线程同步执行，用户要等待抽取完成才能收到回答，响应变慢。`extract_facts()`中`_history_text()`把对话整理成固定格式文本，方便大模型读取。代码延迟导入 rag 里面的 LLM 客户端，防止 memory 模块和 rag 模块互相引用造成循环导入报错。提示词`EXTRACT_PROMPT`严格限定抽取范围，只保留长期稳定事实，排除临时症状、瞬时读数、AI 说的话；如果不加约束，模型会抽取大量临时信息，记忆库里堆满无效内容，召回的时候会混入无关信息，干扰回答。同时限定输出 JSON 数组，还会清除 ``` 代码标记；如果模型自由输出自然语言，JSON 解析直接失败，拿不到事实。设置最多提取 5 条，控制写入数量；不限制条数，单次可能提取大量事实，增加向量库存储和写入开销。抽取遇到任何异常都捕获，返回空数组；如果异常直接抛出，后台线程崩溃，还可能影响主链路。第二部分，`save_facts()`用来保存事实。先调用`create_memory_collection()`，保证记忆集合存在，避免集合还没创建就写入而报错。`_fact_id()`把 user_id 和事实文本一起做 MD5，生成唯一主键。同一个用户、完全一样的事实会得到相同 id，插入 Milvus 时自动覆盖，实现去重；不做这个去重逻辑，同一条事实反复抽取、反复入库，记忆库里面大量重复记录，召回结果冗余。加载 embedding 模型，将文本转为向量再写入。写入 Milvus 的过程加异常捕获，只告警不抛错；记忆只是增强能力，不是问答必需，写入失败不能打断主业务。第三部分，`recall_facts()`用于召回记忆。把用户当前问题编码成向量，调用`search_memory()`并且带上 user_id 过滤，**只能取出当前用户自己的记忆**。如果不带 user_id 过滤，会把别的用户的记忆召回，造成用户之间数据泄露。每次召回最多取 3 条，控制记忆上下文长度；召回太多事实，会占用大量 prompt 的 token，拉高成本，还可能引入无关记忆造成干扰。召回过程异常捕获，失败返回空列表；一旦 Milvus 故障，只是不加载记忆，问答仍然正常跑。整体设计思路：长期记忆属于增强功能，不是必须；抽取动作异步后台执行，不影响用户响应；严格控制抽取内容、数量，做去重；所有环节异常都降级，保证主问答流程不受记忆模块的故障影响。
"""