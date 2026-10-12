# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：Redis 缓存层。相同问题直接命中缓存；Redis 不可用时静默降级，绝不阻断问答主流程。
import hashlib, json                                                              # 缓存键摘要与序列化
from pathlib import Path                                                          # 定位 rag_core.py 以计算结构指纹
from . import config as C                                                         # 复用集中配置

_CLIENT = None                                                                    # Redis 客户端单例
_FP = None                                                                        # 结构指纹单例


def client():                                                                     # 建立连接
    """Redis 5.x 不支持 RESP3，必须显式用 protocol=2，否则连接直接报错。"""
    global _CLIENT                                                                # 声明单例
    if _CLIENT is None:                                                           # 首次调用才连
        import redis                                                              # 延迟导入，加快启动
        _CLIENT = redis.Redis.from_url(C.REDIS_URL, protocol=2, socket_timeout=0.5)  # 0.5 秒超时，Redis 挂了也不拖慢问答
    return _CLIENT                                                                # 返回单例


def schema_fingerprint() -> str:                                                  # 返回体结构指纹
    """取 rag_core.py 源码的哈希前 8 位——返回体由它构造，源码一变指纹就变。

    为什么不用手写的版本号：**手动的版本号迟早会忘**。实测踩过一次——
    给返回体新增 retrieval_mode / degraded 后忘了升版本，旧的缓存条目把老结构原样返回，
    结果是**降级结果不带任何提示地被静默返回**，前端永远显示不出警告。
    交给源码哈希就不需要任何人记得做这件事。
    """
    global _FP                                                                    # 声明单例
    if _FP is None:                                                               # 只算一次
        try:                                                                      # 源码读不到时不能让缓存整个失效
            src = (Path(__file__).resolve().parent / "rag_core.py").read_bytes()   # rag_core 是构造返回体的模块
            _FP = hashlib.md5(src).hexdigest()[:8]                                 # 取前 8 位足够区分
        except Exception: _FP = "nosrc"                                            # 退回固定值，缓存仍可用
    return _FP                                                                    # 返回指纹


def key_of(question: str, lang: str) -> str:                                      # 生成缓存键
    """键 = 前缀 + 结构指纹 + 语言 + 问题摘要；语言区分中英文问同一件事，摘要避免中文键过长。"""
    digest = hashlib.md5((question or "").encode("utf-8")).hexdigest()             # md5 摘要固定长度
    return "gd3:ask:%s:%s:%s" % (schema_fingerprint(), lang, digest)               # 带结构指纹的缓存键


def get(key: str):                                                                # 读缓存
    """命中且结构一致才返回字典；结构不一致按未命中处理。Redis 故障或未命中都返回 None。"""
    try:                                                                          # 缓存故障不影响问答
        raw = client().get(key)                                                   # 取值
        if not raw: return None                                                   # 未命中
        data = json.loads(raw.decode("utf-8"))                                     # 反序列化
        # 双保险：条目自带写入时的结构指纹，对不上就当未命中。
        # 好处不只是防错——运维直接用 redis-cli 看键值就能知道这条是哪个版本写进去的。
        if data.pop("_fp", None) != schema_fingerprint(): return None               # 结构不一致，丢弃
        return data                                                               # 结构一致，返回
    except Exception: return None                                                 # 任何异常都按未命中处理


def set(key: str, value: dict, ttl: int = None) -> bool:                           # 写缓存
    """带过期时间写入，并在条目里记下结构指纹；写失败只返回 False，不抛异常。"""
    try:                                                                          # 同上，静默降级
        payload = dict(value, _fp=schema_fingerprint())                            # 附上写入时的结构指纹
        client().setex(key, ttl or C.CACHE_TTL, json.dumps(payload, ensure_ascii=False))  # 默认 24 小时过期
        return True                                                               # 写成功
    except Exception: return False                                                # 写失败


def ping() -> str:                                                                # 健康探测
    """返回 ok 或错误类型，用于 /health 接口，不抛栈。"""
    try: client().ping(); return "ok"                                              # 连通
    except Exception as exc: return f"error: {type(exc).__name__}"                 # 返回错误类型便于排障
