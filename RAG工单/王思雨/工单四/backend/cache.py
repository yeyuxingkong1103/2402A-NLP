# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：Redis 缓存层。相同问题直接命中缓存；Redis 不可用时静默降级，绝不阻断问答主流程。
import hashlib, json                                                        # 缓存键摘要与序列化
from pathlib import Path                                                    # 定位 rag.py 以计算结构指纹
import config as C                                                          # 集中配置

_CLIENT, _FP = None, None                                                   # 客户端与结构指纹单例


def client():                                                               # 建立连接
    """Redis 5.x 不支持 RESP3，必须显式用 protocol=2，否则连接直接报错。"""
    global _CLIENT                                                          # 声明单例
    if _CLIENT is None:                                                     # 首次才连
        import redis                                                        # 延迟导入，加快启动
        _CLIENT = redis.Redis.from_url(C.REDIS_URL, protocol=2, socket_timeout=0.5)  # 0.5 秒超时，Redis 挂了也不拖慢问答
    return _CLIENT                                                          # 返回单例


def fingerprint() -> str:                                                   # 返回体结构指纹
    """取 rag.py 源码哈希前 8 位——返回体由它构造，源码一变指纹就变。

    为什么不用手写版本号：**手写的版本号迟早会忘**。实测踩过一次——给返回体新增
    degraded 降级标记后忘了升版本，旧缓存把没有该字段的结构原样返回，
    结果是降级结果不带任何提示地被静默返回。交给源码哈希就不需要谁记得做这件事。
    """
    global _FP                                                              # 声明单例
    if _FP is None:                                                         # 只算一次
        try: _FP = hashlib.md5((Path(__file__).resolve().parent / "rag.py").read_bytes()).hexdigest()[:8]  # rag 构造返回体
        except Exception: _FP = "nosrc"                                     # 源码读不到也不能让缓存整体失效
    return _FP                                                              # 返回指纹


def key_of(question: str, lang: str) -> str:                                # 生成缓存键
    """键 = 前缀 + 结构指纹 + 语言 + 问题摘要；语言区分中英文问同一件事。"""
    return f"{C.CACHE_PREFIX}{fingerprint()}:{lang}:{hashlib.md5(question.encode('utf-8')).hexdigest()}"  # 固定长度键


def get(key: str):                                                          # 读缓存
    """命中且结构一致才返回；Redis 故障或未命中一律返回 None。"""
    try:                                                                    # 缓存故障不影响问答
        raw = client().get(key)                                             # 取值
        if not raw: return None                                             # 未命中
        data = json.loads(raw.decode("utf-8"))                              # 反序列化
        return None if data.pop("_fp", None) != fingerprint() else data      # 条目自带写入时指纹，对不上按未命中
    except Exception: return None                                           # 任何异常都按未命中处理


def set(key: str, value: dict, ttl: int = None) -> bool:                     # 写缓存
    """带过期时间写入并记下结构指纹；写失败只返回 False，不抛异常。"""
    try:                                                                    # 同上，静默降级
        client().setex(key, ttl or C.CACHE_TTL,                              # 默认 24 小时过期
                       json.dumps(dict(value, _fp=fingerprint()), ensure_ascii=False))  # 附写入时指纹
        return True                                                         # 写成功
    except Exception: return False                                          # 写失败


def ping() -> str:                                                          # 健康探测
    """返回 ok 或错误类型，供 /api/health 使用，不抛栈。"""
    try: client().ping(); return "ok"                                       # 连通
    except Exception as e: return f"error: {type(e).__name__}"               # 返回错误类型便于排障
