"""Redis 客户端与 Key 设计。

Redis 在本项目里承担 5 类职责，互不干扰（靠 Key 前缀区分）：
1. 短期记忆：最近若干轮对话，让模型有上下文；
2. 缓存：用户资料 / 角色配置 / 检索结果，降低 MySQL 与 Milvus 压力；
3. 限流：按分钟计数，防刷；
4. 并发锁：同一会话同时只允许一个请求在生成，避免消息错乱；
5. JWT 黑名单：支持“登出即刻失效”，弥补 JWT 本身无法撤销的缺陷。

**全局设计原则：可用性优先（fail-open）**。
下面几乎所有函数在捕获 RedisError 后都会返回“安全默认值”而不是抛异常，
因为 Redis 只是加速层与暂存层，不是真相来源：
- 读缓存失败 -> 回源 MySQL/Milvus 即可，功能不受影响；
- 限流失败 -> 放行（宁可被刷也不能让正常用户全部 500）；
- 锁失败 -> 放行（宁可偶尔并发也不能让会话彻底卡死）；
- 黑名单查询失败 -> 放行（可接受的安全权衡，换取高可用）。
唯一例外是写入类操作会返回 False/0，让调用方知道“这次没成功”，但不阻断主流程。

Key 规范：
  session:{user_id}:{persona_id}:{conversation_id}:messages  List          最近聊天记录 TTL 86400
  user:{user_id}:profile                                     String(JSON)  用户缓存 TTL 3600
  persona:{persona_id}                                       String(JSON)  角色缓存 TTL 3600
  rate_limit:{user_id}:{minute}                              String        限流 EXPIRE 60
  lock:conversation:{conversation_id}                        String        并发锁 NX EX 30
  memory_summary:{user_id}:{persona_id}:{conversation_id}    String        会话摘要缓存
  token_blacklist:{jti}                                      String        JWT 吊销黑名单 EXPIRE=token 剩余有效期

命名约定说明：多级用冒号分隔（Redis 社区惯例，可视化工具会自动折叠成树），
把「维度」按 从大到小 排列，如 session:用户:角色:会话，
这样既便于人工排查，也方便用 scan 按前缀批量清理（如 persona 变更时清检索缓存）。
"""
import json
from typing import Any, Dict, List, Optional

import redis
from redis.exceptions import RedisError

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger("db.redis")

# 进程级单例。redis-py 的 Redis 对象内部自带连接池，
# 因此全局共用一个即可，不必也不能每次请求新建（否则会不停建连接）。
_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """返回全局唯一的 Redis 客户端（懒加载）。"""
    global _client
    if _client is None:
        _client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            # 逻辑库编号（0-15），不同用途可分库隔离，避免误删。
            db=settings.redis_db,
            # 空密码必须传 None 而不是空字符串：传 "" 时 redis-py 依旧会发送
            # AUTH 命令，对未设密码的实例会直接报错。
            password=settings.redis_password or None,
            # 自动把 bytes 解码成 str。否则每个调用点都要写 .decode()，
            # 且 json.loads 也能直接吃 str，代码更干净。
            decode_responses=True,
            # 读写超时 5 秒：避免 Redis 卡住时把整个 HTTP 请求一起拖死。
            socket_timeout=5,
            # 建连超时单独设置：防止 Redis 不可达时在这里长时间阻塞。
            socket_connect_timeout=5,
            # 每 30 秒对空闲连接做一次健康检查，自动剔除被服务端关闭的连接，
            # 免去手动处理 “Connection reset by peer”。
            health_check_interval=30,
        )
        logger.info("Redis 客户端已创建：%s:%s/%s", settings.redis_host, settings.redis_port, settings.redis_db)
    return _client


# ---------- Key ----------
# 下面一组函数是「Key 的唯一生成入口」。
# 为什么不直接在业务代码里写 f-string？因为一旦某处拼错一个冒号或顺序，
# 就会写进另一个 Key，表现为“数据莫名丢失”且极难排查。
# 集中在这里定义，读写双方天然一致。
def session_messages_key(user_id: int, persona_id: int, conversation_id: int) -> str:
    """短期记忆 List 的 Key。带 persona_id 是因为同一会话在不同角色下上下文应隔离。"""
    return f"session:{user_id}:{persona_id}:{conversation_id}:messages"


def user_profile_key(user_id: int) -> str:
    """用户资料缓存 Key。"""
    return f"user:{user_id}:profile"


def persona_key(persona_id: int) -> str:
    """角色（心理医生人设）缓存 Key。"""
    return f"persona:{persona_id}"


def rate_limit_key(user_id: int, minute: str) -> str:
    """用户级限流 Key。带 minute 后缀是为了天然按分钟分桶——
    新的一分钟自动是新 Key，计数从 0 开始，不需要定时任务去清零。"""
    return f"rate_limit:{user_id}:{minute}"


def conversation_lock_key(conversation_id: int) -> str:
    """会话级并发锁 Key。粒度选“会话”而非“用户”：
    同一用户在不同会话里可以并行，同一会话内则必须串行。"""
    return f"lock:conversation:{conversation_id}"


def memory_summary_key(user_id: int, persona_id: int, conversation_id: int) -> str:
    """会话摘要缓存 Key（把长对话压缩成摘要后入 Milvus，本地也留一份）。"""
    return f"memory_summary:{user_id}:{persona_id}:{conversation_id}"


def token_blacklist_key(jti: str) -> str:
    """JWT 黑名单 Key：jti 是 token 的唯一 ID，比用整个 token 串做 Key 更短更安全。"""
    return f"token_blacklist:{jti}"


# ---------- 通用 ----------
def set_json(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """写入 JSON 值。ttl 为 None 时表示不过期。

    返回 bool 而非抛异常：调用方据此决定要不要记日志，但不影响主流程。
    """
    try:
        # ensure_ascii=False 保证中文以原字符存储（可读、也更省空间）。
        data = json.dumps(value, ensure_ascii=False)
        if ttl:
            # setex = SET + EXPIRE，一条命令原子完成「赋值并设置过期」。
            # 若写成先 set 再 expire，两条命令之间进程崩溃就会留下永不过期的
            # Key，形成内存里的“僵尸数据”缓慢泄漏。
            get_redis().setex(key, ttl, data)
        else:
            get_redis().set(key, data)
        return True
    except RedisError as exc:
        logger.error("Redis set 失败 key=%s：%s", key, exc)
        return False


def get_json(key: str) -> Any:
    """读取 JSON 值；Key 不存在或 Redis 异常时返回 None。"""
    try:
        raw = get_redis().get(key)
        # raw 为 None 表示 Key 不存在（含已过期），此时不要调 json.loads，
        # 否则会因 None 抛 TypeError。
        return json.loads(raw) if raw else None
    except (RedisError, json.JSONDecodeError) as exc:
        # 同时捕获 JSONDecodeError：极端情况下 Key 可能被其他写入方写成了
        # 非 JSON 内容，缓存脏数据只应导致“缓存未命中”，不该让接口 500。
        logger.error("Redis get 失败 key=%s：%s", key, exc)
        return None


def delete(*keys: str) -> int:
    """删除一个或多个 Key，返回实际删除数量。"""
    try:
        # 空参数时不调用 delete：DEL 无参数会抛错；用 if 短路返回 0。
        return get_redis().delete(*keys) if keys else 0
    except RedisError as exc:
        logger.error("Redis delete 失败：%s", exc)
        return 0


def acquire_lock(key: str, ttl: int = 30) -> bool:
    """尝试获取分布式锁：成功返回 True，已被别人持有返回 False。

    :param ttl: 锁的自动过期秒数，防止持锁进程崩溃后锁永远无法释放（死锁）。
    """
    try:
        # nx=True 等价于 SET ... NX：仅当 Key 不存在时才写入，写入成功即抢到锁。
        # 这一步是原子的，多个进程同时抢也只有一个能成功。
        # ex=ttl 给锁设过期时间，即“租约”——宁可锁提前失效（后续请求会重新抢），
        # 也不能出现永久死锁导致该会话再也无法回复。
        return bool(get_redis().set(key, "1", nx=True, ex=ttl))
    except RedisError as exc:
        logger.error("Redis 加锁失败 key=%s：%s", key, exc)
        return True  # 降级：Redis 异常时不阻塞业务
        # 为什么失败反而返回 True？因为锁的目的是“防止消息错乱”这种软约束，
        # 而 Redis 挂掉时如果返回 False，所有用户都会收到“系统繁忙”，
        # 可用性损失远大于偶发并发的损失——这是有意的取舍。


def release_lock(key: str) -> None:
    """释放锁。直接 DEL 即可（当前实现未做“只有持有者才能删”的校验）。"""
    delete(key)


def health_check() -> bool:
    """健康检查：PING 通即认为可用。"""
    try:
        return bool(get_redis().ping())
    except RedisError as exc:
        logger.error("Redis 健康检查失败：%s", exc)
        return False


# ---------- JWT 吊销黑名单 ----------
def blacklist_token(jti: str, ttl_seconds: int) -> bool:
    """把 token 的 jti 加入黑名单，TTL 与 token 剩余有效期一致。

    为什么 TTL 必须等于剩余有效期？因为 token 到期后本身就失效了，
    黑名单再留着它毫无意义，只会持续占用内存。用“剩余有效期”做 TTL，
    黑名单大小天然有上界，不用额外的清理任务。
    """
    if ttl_seconds <= 0:
        # 已过期或即将过期的 token 无需拉黑，直接返回 False 表示“没做这件事”。
        return False
    try:
        get_redis().set(token_blacklist_key(jti), "1", ex=ttl_seconds)
        return True
    except RedisError as exc:
        logger.error("写入 JWT 黑名单失败 jti=%s：%s", jti, exc)
        return False


def is_token_blacklisted(jti: str) -> bool:
    """True 表示已吊销；Redis 异常时放行（与限流一致的可 用性优先策略）。"""
    if not jti:
        # 老版本签发的 token 可能没有 jti 声明，此时无从查黑名单，视为未吊销。
        return False
    try:
        # exists 返回 0/1，比 get 再判断内容更省一次数据读取。
        return bool(get_redis().exists(token_blacklist_key(jti)))
    except RedisError as exc:
        # 用 warning 而非 error：这是“降级放行”，属于已知的权衡行为。
        logger.warning("查询 JWT 黑名单失败（放行）：%s", exc)
        return False


# ---------- 短期记忆 ----------
def append_message(user_id: int, persona_id: int, conversation_id: int,
                   role: str, content: str,
                   max_turns: Optional[int] = None, ttl: Optional[int] = None) -> None:
    """向短期记忆 List 追加一条消息，并裁剪到最近 N 轮（N 轮 ≈ 2N 条）。"""
    # 参数默认值取配置而不是写死，便于通过 .env 调整上下文长度而不改代码。
    max_turns = max_turns or settings.short_term_max_turns
    ttl = ttl or settings.short_term_ttl
    key = session_messages_key(user_id, persona_id, conversation_id)
    item = json.dumps({"role": role, "content": content}, ensure_ascii=False)
    try:
        # 用 pipeline 把 3 条命令打包成一次往返（默认非事务模式）。
        # 关键点是「原子性」：如果分 3 次发送，中间可能被其他请求插进来，
        # 导致 LTRIM 裁掉了别人刚写的消息，或 EXPIRE 用了错误的值。
        pipe = get_redis().pipeline()
        # 从右侧追加，保证 List 内顺序 = 对话时间顺序。
        pipe.rpush(key, item)
        # 只保留最后 2*max_turns 个元素（-2N 到 -1）。
        # 用 LTRIM 而不是定期全量重写，是为了让裁剪成本恒定（O(N) 但 N 很小），
        # 且列表永远不会无限增长，避免长会话把 Redis 内存吃满。
        pipe.ltrim(key, -2 * max_turns, -1)
        # 每次追加都续期 TTL，实现“滑动窗口”：只要用户还在聊，
        # 上下文就不会被清理；停止聊天 24 小时后自然释放内存。
        pipe.expire(key, ttl)
        pipe.execute()
    except RedisError as exc:
        # 写短期记忆失败只丢上下文，不影响本轮回复的生成，
        # 因此记录日志后静默返回，不向上抛（可用性优先）。
        logger.error("写入短期记忆失败 key=%s：%s", key, exc)


def get_recent_messages(user_id: int, persona_id: int, conversation_id: int,
                        limit: Optional[int] = None) -> List[Dict[str, str]]:
    """读取最近 limit 条消息，返回按时间升序的 [{role, content}]。"""
    limit = limit or settings.short_term_max_turns * 2
    key = session_messages_key(user_id, persona_id, conversation_id)
    try:
        # lrange(key, -limit, -1)：从尾部往前取，正好是“最近 limit 条”。
        # 注意负索引在 count 超过列表长度时会自动从 0 开始，不会报错。
        raw = get_redis().lrange(key, -limit, -1)
        # 逐条反序列化。若某条 JSON 损坏会抛 JSONDecodeError，
        # 由下面的 except 兜住并返回空列表（整段上下文丢失总好过接口报错）。
        return [json.loads(x) for x in raw]
    except (RedisError, json.JSONDecodeError) as exc:
        logger.error("读取短期记忆失败 key=%s：%s", key, exc)
        return []


def clear_short_term(user_id: int, persona_id: int, conversation_id: int) -> None:
    """清空某个会话的短期记忆（用户点“新会话”或“清除上下文”时调用）。"""
    delete(session_messages_key(user_id, persona_id, conversation_id))


def touch_ttl(user_id: int, persona_id: int, conversation_id: int) -> None:
    """刷新短期记忆的过期时间，但不动内容。

    场景：本轮只读历史、没有产生新消息（如用户只发了一个表情被拦截），
    仍然应该让“用户还在活跃”这个事实把上下文续期。
    """
    try:
        get_redis().expire(
            session_messages_key(user_id, persona_id, conversation_id), settings.short_term_ttl
        )
    except RedisError as exc:
        # 续期失败无伤大雅，下次 append_message 还会再续，所以只记 warning。
        logger.warning("刷新短期记忆 TTL 失败：%s", exc)


# ---------- 缓存 ----------
def cache_user_profile(user_id: int, data: Dict[str, Any], ttl: int = 3600) -> None:
    """缓存用户资料，默认 1 小时过期（兜底：即使忘了主动失效也会自动纠正）。"""
    set_json(user_profile_key(user_id), data, ttl)


def get_cached_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """读取用户资料缓存，None 表示未命中（调用方应回源 MySQL）。"""
    return get_json(user_profile_key(user_id))


def invalidate_user_profile(user_id: int) -> None:
    """用户资料变更后主动删缓存（Cache-Aside 模式的“写后失效”）。"""
    delete(user_profile_key(user_id))


def cache_persona(persona_id: int, data: Dict[str, Any], ttl: int = 3600) -> None:
    """缓存角色人设。角色是读多写少的配置类数据，非常适合缓存。"""
    set_json(persona_key(persona_id), data, ttl)


def get_cached_persona(persona_id: int) -> Optional[Dict[str, Any]]:
    """读取角色缓存。"""
    return get_json(persona_key(persona_id))


def invalidate_persona(persona_id: int) -> None:
    """角色配置被管理员修改后清缓存，保证立即生效。"""
    delete(persona_key(persona_id))


# ---------- 限流 ----------
def check_rate_limit_by_key(key: str, limit: int) -> bool:
    """通用滑动分钟窗口限流：返回 True 表示允许通过。"""
    try:
        # INCR 是原子自增，返回自增后的值。并发下不会出现两个请求读到同一个计数，
        # 这是“用 Redis 限流”相比“应用内存计数”的核心优势（多实例部署也准确）。
        count = get_redis().incr(key)
        if count == 1:
            # 只给“第一次自增”的这个 Key 设置过期。
            # 每次都 expire 是多余的；而如果忘了设置，Key 会永久存在导致
            # 用户被永久限流——所以这里必须设，且只设一次最省命令。
            get_redis().expire(key, 60)
        return count <= limit
    except RedisError as exc:
        # fail-open：限流器故障时放行。
        # 理由：限流是保护性措施，不应成为单点故障；
        # 若这里 return False，Redis 一挂全站用户都会收到 429。
        logger.warning("限流检查失败（放行）：%s", exc)
        return True


def login_rate_limit_key(ip: str, minute: str) -> str:
    """登录接口按 IP 限流（未登录时没有 user_id，只能用 IP 作为维度）。"""
    return f"rate_limit:login:{ip}:{minute}"


def check_rate_limit(user_id: int, limit: Optional[int] = None) -> bool:
    """返回 True 表示允许通过。"""
    limit = limit or settings.rate_limit_per_minute
    # 取“到分钟”的时间戳作为分桶后缀，例如 202609211430。
    # 这样做的好处是无需任何清理任务：过了这一分钟，旧 Key 自然过期、
    # 新 Key 从 0 开始计数。
    minute = __import__("datetime").datetime.now().strftime("%Y%m%d%H%M")
    return check_rate_limit_by_key(rate_limit_key(user_id, minute), limit)


# ---------- 检索结果缓存（阶段 4：命中可跳过向量化 + Milvus + 重排） ----------
# 一次 RAG 检索的代价 = 一次 Embedding（几百毫秒）+ Milvus 查询 + Reranker。
# 相同角色、相同问题的重复提问（用户改一个字再问、或反复问同一句）
# 极其常见，命中缓存可直接省掉整条链路，是本项目最大的性能优化点。
_RETRIEVAL_CACHE_PREFIX = "retrieval_cache"


def retrieval_cache_key(persona_id: int, fingerprint: str) -> str:
    """检索缓存 Key。

    为什么把 persona_id 放在前缀里（而不是和 fingerprint 拼在一起）？
    因为清理时需要按角色批量失效（该角色知识库更新后），
    `retrieval_cache:{persona_id}:*` 这个模式能直接命中该角色全部缓存，
    若把 persona_id 混进指纹就无法这样批量扫描了。
    """
    return f"{_RETRIEVAL_CACHE_PREFIX}:{persona_id}:{fingerprint}"


def get_retrieval_cache(persona_id: int, fingerprint: str):
    """读取检索结果缓存，未命中或异常时返回 None。"""
    try:
        raw = get_redis().get(retrieval_cache_key(persona_id, fingerprint))
        return json.loads(raw) if raw else None
    except RedisError as exc:
        # 缓存读取失败只意味着“这次走完整检索”，对用户完全无感，故 warning。
        logger.warning("检索缓存读取失败（跳过）：%s", exc)
        return None
    except (TypeError, ValueError):
        # 反序列化失败同样视为未命中；这里不记日志，避免脏数据刷日志。
        return None


def cache_retrieval(persona_id: int, fingerprint: str, payload, ttl: Optional[int] = None) -> None:
    """写入检索结果缓存。"""
    try:
        # 同样用 setex 一条命令完成“写值 + 过期”，避免中间态残留永久 Key。
        get_redis().setex(
            retrieval_cache_key(persona_id, fingerprint),
            ttl or settings.retrieval_cache_ttl,
            json.dumps(payload, ensure_ascii=False),
        )
    except RedisError as exc:
        logger.warning("检索缓存写入失败（跳过）：%s", exc)


def clear_retrieval_cache(persona_id: Optional[int] = None) -> None:
    """知识库变更后失效缓存；persona_id 为空时清全部。

    必须在知识库增删改后调用，否则用户会拿着过期的检索结果继续聊天，
    表现为“新上传的文档不生效”，是这类系统里最常见的线上问题。
    """
    try:
        pattern = (f"{_RETRIEVAL_CACHE_PREFIX}:{persona_id}:*" if persona_id
                   else f"{_RETRIEVAL_CACHE_PREFIX}:*")
        # 用 scan_iter 而不是 KEYS：KEYS 会一次性遍历整个键空间并阻塞 Redis 主线程，
        # 在线上可能造成秒级卡顿；scan 是游标式分批返回，count=200 控制每批大小，
        # 对线上服务影响可控。
        keys = list(get_redis().scan_iter(match=pattern, count=200))
        if keys:
            # delete 支持一次传多个 Key，减少往返次数。
            get_redis().delete(*keys)
            logger.info("已失效检索缓存 %d 条（persona=%s）", len(keys), persona_id or "*")
    except RedisError as exc:
        # 清理失败不影响正确性（旧缓存有 TTL 兜底会自动过期），只记 warning。
        logger.warning("检索缓存清理失败（忽略）：%s", exc)