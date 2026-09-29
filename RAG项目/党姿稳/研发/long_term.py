"""
long_term.py — 长期记忆（Milvus 集合 long_term_memory）

按 user_id + domain 过滤，两类记忆靠既有的 fact_type 字段区分：

    summary  每轮存"用户问题 + 回答要点前 200 字"的摘要
    profile  每 PROFILE_EXTRACT_EVERY 轮用大模型提炼的偏好与已确认事实

检索流程：向量召回 → 原始余弦阈值过滤 → 时间衰减重排 → 取 top_k。
**时间衰减只影响排序**：久远的记忆会排到后面，但不会被阈值剔除，
否则长期记忆会退化成短期记忆。

LOCAL_MODE=True 时走 local_store.LocalStore，无需 Milvus。
"""

from __future__ import annotations

import math
import time

import config
import embeddings
import session_memory
from vector_store import get_store

_logger = None


def _log():
    """延迟初始化日志，避免模块导入时就写日志文件。"""
    global _logger
    if _logger is None:
        from loguru import logger

        _logger = logger
    return _logger


# 长期记忆的事实类型，与 vector_store.MEMORY_FIELDS 里的 fact_type 对应
FACT_TYPES = ("dialogue", "preference", "profile", "summary", "correction")

# 关系记忆提取提示词。短任务必须关掉思维链，否则推理模型会把 max_tokens 全用在
# 思考上、正文返回为空。
_PROFILE_PROMPT = """你是一个用户画像提取助手。请从下面的对话中提取「用户偏好」与「已确认事实」。

要求：
1. 只提取对话里明确出现过的信息，不要推测，不要编造。
2. 没有任何可提取内容时，只回复 NONE 四个字母，不要输出别的。
3. 有内容时每行一条、最多 3 条，严格按下面的格式：
偏好：<用户表达过的喜好、要求或习惯>
事实：<用户提到过的关于自己的确定信息>
"""


def _now() -> int:
    return int(time.time())


def _decay_score(hit: dict) -> float:
    """时间衰减后的排序分：越新近得分越高。只用于排序，不参与阈值过滤。"""
    age_days = max(0.0, (time.time() - float(hit.get("created_at") or 0)) / 86400.0)
    tau = config.MEMORY_DECAY_TAU_DAYS or 30.0
    return float(hit.get("score") or 0.0) * math.exp(-age_days / tau)


def store_long_term(
    user_id: str,
    text: str,
    role: str = "",
    fact_type: str = "summary",
    source: str = "",
    summary: str = "",
    domain: str = "",
) -> int:
    """写入一条长期记忆。

    先按向量相似度查重：命中 DUPLICATE_THRESHOLD 的旧记录会被删除后重写。
    返回写入条数（0 表示入参为空）。
    """
    text = (text or "").strip()
    if not text:
        return 0

    store = get_store()
    vector = embeddings.encode_query(text)

    try:
        existing = store.search(
            config.LONG_TERM_COLLECTION, vector, top_k=1, filters={"user_id": user_id}
        )
    except Exception as exc:
        _log().warning(f"长期记忆查重失败，改为直接写入：{exc}")
        existing = []

    if existing and existing[0].get("score", 0.0) >= config.DUPLICATE_THRESHOLD:
        old_id = existing[0].get("id")
        if old_id is not None:
            try:
                store.delete(config.LONG_TERM_COLLECTION, {"id": old_id})
            except Exception:
                pass

    record = {
        "text": text,
        "summary": summary or text[:120],
        "source": source,
        "user_id": user_id,
        "domain": domain,
        "role": role,
        "fact_type": fact_type if fact_type in FACT_TYPES else "summary",
        "embedding": vector,
        "created_at": _now(),
        "updated_at": _now(),
    }
    return len(store.upsert(config.LONG_TERM_COLLECTION, [record]))


def search_long_term(
    user_id: str,
    query: str,
    domain: str | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """按语义检索该用户的长期记忆，按时间衰减后的分数降序返回。

    min_score 为空时按当前向量后端取阈值（bge-m3 用 0.7，hash 降级用 0.3）。
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = config.LONG_TERM_TOP_K if top_k is None else top_k
    threshold = config.long_term_threshold() if min_score is None else min_score

    filters: dict = {"user_id": user_id}
    if domain:
        filters["domain"] = domain

    try:
        # 多召回一倍，给时间衰减重排留出腾挪空间
        hits = get_store().search(
            config.LONG_TERM_COLLECTION,
            embeddings.encode_query(query),
            top_k=top_k * 2,
            filters=filters,
        )
    except Exception as exc:
        _log().warning(f"长期记忆检索失败：{exc}")
        return []

    # 先用原始余弦过滤，再按衰减分数排序：阈值不受时间影响
    kept = [h for h in hits if float(h.get("score") or 0.0) >= threshold]
    kept.sort(key=_decay_score, reverse=True)
    return kept[:top_k]


def format_hits(hits: list[dict]) -> str:
    """把召回的长期记忆格式化成提示词片段，供【历史对话参考】段使用。"""
    lines: list[str] = []
    for hit in hits:
        text = (hit.get("text") or "").strip()
        if not text:
            continue
        stamp = time.strftime("%Y-%m-%d", time.localtime(float(hit.get("created_at") or 0)))
        kind = "用户偏好" if hit.get("fact_type") == "profile" else "摘要"
        lines.append(f"- [{stamp} {kind}] {text}")
    return "\n".join(lines)


def list_long_term(user_id: str, limit: int = 50) -> list[dict]:
    """按时间倒序列出该用户的长期记忆。"""
    try:
        return get_store().query(
            config.LONG_TERM_COLLECTION, filters={"user_id": user_id}, limit=limit
        )
    except Exception as exc:
        _log().warning(f"列出长期记忆失败：{exc}")
        return []


def clear_long_term(user_id: str) -> int:
    """删除该用户的全部长期记忆。"""
    try:
        return get_store().delete(config.LONG_TERM_COLLECTION, {"user_id": user_id})
    except Exception as exc:
        _log().warning(f"清空长期记忆失败：{exc}")
        return 0


# ------------------------------------------------------------ 关系记忆提取


def _extract_from_history(history: list[dict]) -> str:
    """用大模型从最近若干轮对话里提炼偏好与已确认事实。"""
    from llm_client import get_client

    lines = [
        f"{'用户' if item.get('role') == 'user' else '助手'}：{item.get('content', '')}"
        for item in history
        if item.get("content")
    ]
    if not lines:
        return ""

    try:
        raw = get_client().chat(
            _PROFILE_PROMPT,
            "\n".join(lines),
            history=None,
            temperature=0.0,
            max_tokens=256,
            reasoning_effort=config.LLM_REASONING_EFFORT,
        )
    except Exception as exc:
        _log().warning(f"提取用户画像失败：{exc}")
        return ""

    text = (raw or "").strip()
    if not text or text.upper().startswith("NONE"):
        return ""
    # 模型偶尔会把提示词里的示例也抄回来，这里做一次兜底
    if "偏好：" not in text and "事实：" not in text:
        return ""
    return text


def maybe_extract_profile(user_id: str, domain: str = "") -> None:
    """每 PROFILE_EXTRACT_EVERY 轮提炼一次关系记忆。

    调用方应放进后台线程：这是一次额外的模型往返（数秒），
    放在请求路径上会让每第 N 轮明显变慢。
    """
    every = config.PROFILE_EXTRACT_EVERY
    if every <= 0:
        return
    if session_memory.bump_round(user_id) % every != 0:
        return

    history = session_memory.get_short_term(user_id)
    if len(history) < 2:
        return

    text = _extract_from_history(history)
    if not text:
        return

    store_long_term(user_id, text, fact_type="profile", source=domain, domain=domain)
    _log().info(f"已为用户 {user_id} 提取关系记忆：{text[:60]}")


if __name__ == "__main__":
    import vector_store

    config.LOCAL_MODE = True
    vector_store.reset_store()

    uid = "_selftest_lt"
    clear_long_term(uid)

    store_long_term(uid, "用户问题：试用期一般多久\n回答要点：三个月到六个月", domain="legal")
    store_long_term(uid, "偏好：希望回答简洁\n事实：用户是法学生", fact_type="profile", domain="legal")
    store_long_term(uid, "用户问题：试用期一般多久\n回答要点：这是医疗领域的记录", domain="medical")

    assert len(list_long_term(uid)) == 3, "三条记忆没有全部写进去"

    legal_hits = search_long_term(uid, "试用期一般多久", domain="legal", top_k=5)
    print(f"限 legal 召回 {len(legal_hits)} 条：")
    print(format_hits(legal_hits))

    assert legal_hits, "domain 过滤把该领域的记忆也滤掉了"
    assert all(h.get("domain") == "legal" for h in legal_hits), "domain 过滤没有生效"

    clear_long_term(uid)
    assert list_long_term(uid) == []
    print("long_term 自检通过。")
