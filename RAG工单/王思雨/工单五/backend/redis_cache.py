# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：缓存与会话层。Redis 同时承担三件事：多轮会话历史、检索/答案缓存、用户反馈留存。
#           坑（实测）：本机 Redis 5.0.14 不支持 RESP3(HELLO)，redis-py 8 必须显式 protocol=2，否则连不上。
import hashlib                                                                  # 生成缓存键与结构指纹
import json                                                                     # 会话与反馈的序列化
import time                                                                     # 反馈时间戳
from pathlib import Path                                                        # 读取源码文件算指纹
from typing import List, Dict, Optional                                         # 类型标注

from . import config as C                                                       # 复用集中配置

_CLIENT = None                                                                  # Redis 客户端单例
_FAILED = False                                                                 # 本次会话内是否发生过 Redis 故障（用于把降级如实上报到响应里）


def mark_failed() -> None:                                                      # 标记发生过故障
    """任何 Redis 操作失败后调用；在 ./pipeline.py 里汇总进响应的 degraded 字段。"""
    global _FAILED                                                              # 声明全局
    _FAILED = True                                                              # 置位


def mark_ok() -> None:                                                          # 标记恢复正常
    """任一 Redis 操作成功后调用，避免一次瞬时故障把整个进程标记成永久降级。"""
    global _FAILED                                                              # 声明全局
    _FAILED = False                                                             # 复位


def degraded() -> bool:                                                         # 查询降级状态
    """True 表示最近一次操作失败过；供 /chat 上报 degraded=["redis"]。"""
    return _FAILED                                                              # 返回
_FINGERPRINT = ""                                                               # 缓存结构指纹（进程内只算一次）
_INDEX_FP = ""                                                                  # 知识库指纹（进程内只算一次）


def _redis():                                                                   # 懒加载 Redis 客户端
    """连接 Redis；失败返回 None，由上层退化为"无缓存/无历史"模式而不是报错。"""
    global _CLIENT                                                              # 声明单例
    if _CLIENT is None:                                                         # 尚未连接
        import redis                                                            # 延迟导入
        _CLIENT = redis.Redis.from_url(C.REDIS_URL, protocol=C.REDIS_PROTOCOL,  # 显式 RESP2，兼容老版本 Redis
                                       decode_responses=True,                   # 直接返回 str，省去手工解码
                                       socket_timeout=3,                        # 超时要短，缓存不该拖慢主流程
                                       socket_connect_timeout=3)                # 连接超时同理
    return _CLIENT                                                              # 返回


def available() -> bool:                                                        # 健康检查与降级判断
    """探测 Redis 是否可用；用于 /health 与"缓存不可用时照常作答"。"""
    try:                                                                        # ping 可能抛异常
        ok = bool(_redis().ping())                                              # 探活
        mark_ok() if ok else mark_failed()                                      # 刷新降级状态
        return ok                                                               # 返回布尔
    except Exception as exc:                                                    # 连接失败
        print(f"[缓存降级] Redis 不可用：{type(exc).__name__}: {exc}")           # 明确报因
        mark_failed()                                                           # 如实上报降级
        return False                                                            # 不可用


def fingerprint() -> str:                                                       # 缓存结构指纹
    """用关键源码的哈希自动生成结构指纹。

    坑（实测）：旧缓存会静默吞掉新字段——曾出现 degraded 字段被旧缓存吞掉，导致降级无提示。
    把影响响应结构的模块源码一起哈希，改代码即自动换键，不需要手工升版本号。
    """
    global _FINGERPRINT                                                         # 声明全局
    if _FINGERPRINT:                                                            # 已算过
        return _FINGERPRINT                                                     # 直接返回
    parts = []                                                                  # 待哈希内容
    for name in ("config.py", "schemas.py", "deepseek_llm.py",                  # 影响响应结构的模块清单
                 "hybrid_retriever.py", "reranker.py", "query_rewriter.py"):    # 同上
        p = Path(__file__).resolve().parent / name                              # 定位文件
        try:                                                                    # 读取可能失败
            parts.append(f"{name}:{hashlib.sha256(p.read_bytes()).hexdigest()}")  # 文件内容哈希
        except Exception:                                                       # 文件缺失
            parts.append(f"{name}:missing")                                     # 以缺失占位，保证指纹仍可算
    _FINGERPRINT = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]  # 截断成 12 位
    return _FINGERPRINT                                                         # 返回


def index_fingerprint() -> str:                                                 # 知识库指纹
    """对分块文件与 BM25 索引做内容哈希，标识"当前这版知识库"。

    坑（实测，必须带）：缓存键只带**代码**指纹是不够的。曾出现：改了分块逻辑并重建索引后，
    旧缓存仍命中原先的错误答案（问"力源信息法定代表人"却回"胡戎"），因为响应只跟"问题+历史+代码"有关，
    跟知识库本身无关。带上知识库指纹后，重新入库即自动换键，历史错误答案不会再被复用。
    哈希只在进程内算一次；索引文件约数 MB，一次性开销可忽略。
    """
    global _INDEX_FP                                                            # 声明全局
    if _INDEX_FP:                                                               # 已算过
        return _INDEX_FP                                                        # 直接返回
    parts = []                                                                  # 待哈希内容
    for name in ("index.json", "bm25.pkl"):                                     # 知识库的两个产物
        p = C.ROOT / "data" / name                                              # 定位文件
        try:                                                                    # 文件可能尚未生成
            h = hashlib.sha256()                                                # 流式哈希，避免一次性读入内存
            with open(p, "rb") as f:                                            # 二进制读取
                for block in iter(lambda: f.read(1 << 20), b""):                # 每次 1MB
                    h.update(block)                                             # 累加
            parts.append(f"{name}:{h.hexdigest()}")                             # 记录
        except Exception:                                                       # 文件缺失
            parts.append(f"{name}:missing")                                     # 占位，保证仍可算
    _INDEX_FP = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]  # 截断成 12 位
    return _INDEX_FP                                                            # 返回


def _key(kind: str, raw: str) -> str:                                           # 统一拼键
    """缓存键 = 前缀 + 代码结构指纹 + 知识库指纹 + 内容哈希；两个指纹缺一不可，见各自说明。"""
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]                    # 内容哈希
    return f"gd5:{kind}:{fingerprint()}:{index_fingerprint()}:{h}"              # 最终键


# ==================== 一、多轮会话历史 ====================
def append_history(session_id: str, role: str, content: str) -> None:           # 追加一轮消息
    """按 chat:{sid}:history 存列表；只留最近 HISTORY_TURNS 轮，防止上下文无限膨胀。"""
    if not session_id:                                                          # 无会话号
        return                                                                  # 不记录
    r = _redis()                                                                # 取客户端
    key = f"chat:{session_id}:history"                                          # 历史键（工单指定形态）
    text = (content or "")[:C.SESSION_MAX_CHARS]                                # 截断，防止超长正文撑爆历史
    try:                                                                        # Redis 异常不能影响作答
        r.rpush(key, json.dumps({"role": role, "content": text}, ensure_ascii=False))  # 追加
        r.ltrim(key, -C.HISTORY_TURNS * 2, -1)                                  # 只保留最近 N 轮（一问一答算两条）
        r.expire(key, C.HISTORY_TTL)                                            # 续期
    except Exception as exc:                                                    # 写失败
        print(f"[缓存降级] 写会话历史失败：{type(exc).__name__}: {exc}")         # 记录后继续


def get_history(session_id: str) -> List[Dict]:                                 # 读取会话历史
    """返回 [{"role": "user"/"assistant", "content": ...}]；不可用时返回空列表。"""
    if not session_id:                                                          # 无会话号
        return []                                                               # 空历史
    try:                                                                        # 读可能失败
        raw = _redis().lrange(f"chat:{session_id}:history", 0, -1)              # 取全部
        return [json.loads(x) for x in raw]                                     # 反序列化
    except Exception as exc:                                                    # 失败
        print(f"[缓存降级] 读会话历史失败：{type(exc).__name__}: {exc}")         # 记录
        mark_failed()                                                           # 如实上报降级
        return []                                                               # 返回空


def clear_history(session_id: str) -> None:                                     # 清空会话
    """开新会话时调用，避免上一轮实体被继承。"""
    try:                                                                        # 删除可能失败
        _redis().delete(f"chat:{session_id}:history", f"chat:{session_id}:entity")  # 同时清历史与实体
    except Exception as exc:                                                    # 失败
        print(f"[缓存降级] 清会话失败：{type(exc).__name__}: {exc}")             # 记录


def history_digest(session_id: str) -> str:                                     # 历史摘要
    """把最近若干轮压成一个短摘要串，参与缓存键计算（工单要求 key 含"历史摘要"）。"""
    hist = get_history(session_id)                                              # 取历史
    tail = hist[-C.REWRITE_MAX_HISTORY * 2:] if hist else []                    # 只取最近数轮
    return " / ".join(f"{h.get('role')}:{str(h.get('content', ''))[:40]}" for h in tail)  # 压缩成短串


# ==================== 二、实体追踪（多轮指代消解用）====================
def set_entity(session_id: str, name: str) -> None:                             # 记录当前讨论的公司
    """用户切换公司后，"他/这个公司"应指向新公司；故实体要随会话持久化。"""
    if not session_id or not name:                                              # 参数不全
        return                                                                  # 不记录
    try:                                                                        # 写可能失败
        r = _redis()                                                            # 取客户端
        r.set(f"chat:{session_id}:entity", name[:80])                           # 存最新实体
        r.expire(f"chat:{session_id}:entity", C.HISTORY_TTL)                    # 续期
    except Exception as exc:                                                    # 失败
        print(f"[缓存降级] 写实体失败：{type(exc).__name__}: {exc}")             # 记录


def get_entity(session_id: str) -> str:                                         # 读取当前实体
    """返回当前会话正在讨论的公司名；没有则返回空串。"""
    try:                                                                        # 读可能失败
        v = _redis().get(f"chat:{session_id}:entity")                           # 取实体
        return v or ""                                                          # 兜底成空串
    except Exception as exc:                                                    # 失败
        print(f"[缓存降级] 读实体失败：{type(exc).__name__}: {exc}")             # 记录
        mark_failed()                                                           # 如实上报降级
        return ""                                                               # 返回空


# ==================== 三、检索与答案缓存 ====================
def get_json(kind: str, raw_key: str) -> Optional[dict]:                        # 读缓存
    """"raw_key" 由调用方拼装（问题 + 历史摘要 + 召回参数），本函数只负责加结构指纹。"""
    try:                                                                        # 读可能失败
        v = _redis().get(_key(kind, raw_key))                                   # 取缓存
        return json.loads(v) if v else None                                     # 反序列化
    except Exception as exc:                                                    # 失败
        print(f"[缓存降级] 读缓存失败：{type(exc).__name__}: {exc}")             # 记录
        mark_failed()                                                           # 如实上报降级
        return None                                                             # 视作未命中


def set_json(kind: str, raw_key: str, value: dict, ttl: int = None) -> None:    # 写缓存
    """写缓存失败不影响本次作答，只是下次仍要重算。"""
    try:                                                                        # 写可能失败
        _redis().setex(_key(kind, raw_key), ttl or C.CACHE_TTL,                 # 带过期写入
                       json.dumps(value, ensure_ascii=False))                   # 序列化
    except Exception as exc:                                                    # 失败
        print(f"[缓存降级] 写缓存失败：{type(exc).__name__}: {exc}")             # 记录


# ==================== 四、用户反馈 ====================
def save_feedback(record: Dict) -> bool:                                        # 保存反馈
    """工单要求"支持用户反馈"：把点赞/点踩与文字意见落 Redis 列表，供后续复盘。"""
    record = dict(record)                                                       # 复制，避免改到调用方对象
    record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")                           # 补时间戳
    try:                                                                        # 写可能失败
        r = _redis()                                                            # 取客户端
        sid = str(record.get("session_id") or "default")                        # 会话号
        r.rpush(f"chat:{sid}:feedback", json.dumps(record, ensure_ascii=False))  # 追加到该会话的反馈列表
        r.expire(f"chat:{sid}:feedback", C.CACHE_TTL)                           # 续期
        return True                                                             # 成功
    except Exception as exc:                                                    # 失败
        print(f"[缓存降级] 反馈保存失败：{type(exc).__name__}: {exc}")           # 记录
        mark_failed()                                                           # 如实上报降级
        return False                                                            # 返回失败，让接口如实告知


def list_feedback(session_id: str, limit: int = 50) -> List[Dict]:              # 读反馈
    """用于自检与演示时回看用户的评价。"""
    try:                                                                        # 读可能失败
        raw = _redis().lrange(f"chat:{session_id}:feedback", -limit, -1)         # 取最近若干条
        return [json.loads(x) for x in raw]                                     # 反序列化
    except Exception as exc:                                                    # 失败
        print(f"[缓存降级] 反馈读取失败：{type(exc).__name__}: {exc}")           # 记录
        mark_failed()                                                           # 如实上报降级
        return []                                                               # 返回空
