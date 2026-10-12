# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 检索与生成核心：Milvus + BM25 混合检索 → 重排 → DeepSeek，附 Redis 缓存
import hashlib, json, pickle, time                                            # 指纹/序列化/计时
import numpy as np, jieba                                                     # 向量运算 / 查询端分词
from pymilvus import MilvusClient                                             # 向量库客户端
import redis as _redis                                                        # 缓存客户端
from openai import OpenAI                                                     # DeepSeek 走 OpenAI 兼容协议
from . import config as C                                                     # 配置
jieba.setLogLevel(60)                                                         # 静音 jieba
_M = {}                                                                       # 进程内单例表：模型与客户端只建一次
def embed(texts):                                                             # 文本 → 1024 维单位向量
    if "embed" not in _M:                                                     # 模型只加载一次（每次约 10 秒）
        from sentence_transformers import SentenceTransformer                 # 延迟导入，导入本身慢
        _M["embed"] = SentenceTransformer(C.EMBED_MODEL_PATH, device=C.DEVICE)  # bge-m3，走 GPU
    return _M["embed"].encode(list(texts), batch_size=C.EMBED_BATCH, normalize_embeddings=True, show_progress_bar=False)  # 归一化后可直接用余弦
def milvus():                                                                 # Milvus 客户端单例（复用本机已运行容器，勿再起 compose）
    if "milvus" not in _M: _M["milvus"] = MilvusClient(uri=C.MILVUS_URI)    # 只建一次连接
    return _M["milvus"]                                                       # 返回客户端
def rds():                                                                    # Redis 客户端单例
    if "redis" not in _M:                                                     # 只建一次连接
        _M["redis"] = _redis.Redis.from_url(C.REDIS_URL, protocol=C.REDIS_PROTOCOL, decode_responses=True)  # 本机 Redis 5 必须 protocol=2，否则连接直接被拒
    return _M["redis"]                                                        # 返回客户端
def llm():                                                                    # DeepSeek 客户端单例
    if "llm" not in _M:                                                       # 每请求新建会多付约 1 秒 TLS 握手
        _M["llm"] = OpenAI(api_key=C.LLM_API_KEY, base_url=C.LLM_BASE_URL, timeout=C.LLM_TIMEOUT, max_retries=0)  # 超时即返回不重试，避免长尾叠加
    return _M["llm"]                                                          # 返回客户端
def vector_search(q, k=None):                                                 # 通道一：向量召回
    r = milvus().search(C.MILVUS_COLLECTION, data=[list(embed([q])[0])], limit=k or C.VEC_TOP_K, output_fields=["text", "company", "year", "page"], search_params={"metric_type": "COSINE"}, consistency_level="Strong")  # 默认 Bounded 会让刚写入的数据读不到
    return [dict(h["entity"], score=float(h["distance"]), src="vec") for h in r[0]]  # 统一成字典，来源标记 vec
def bm25_search(q, k=None):                                                   # 通道二：关键词召回（BM25 索引进程内缓存，只读一次磁盘）
    if "bm25" not in _M:                                                      # 首次调用才载入索引
        _M["bm25"] = pickle.loads((C.DATA / "bm25.pkl").read_bytes())          # 含算法实例与全部块记录
    idx = _M["bm25"]; sc = idx["bm25"].get_scores([w for w in jieba.lcut(q) if w.strip()])  # BM25 打分
    return [dict(idx["recs"][i], score=float(sc[i]), src="bm25") for i in np.argsort(-np.asarray(sc))[: k or C.BM25_TOP_K] if sc[i] > 0]  # 取前 k 并过滤 0 分噪声
def fuse(vh, bh, k=None):                                                     # 混合融合：先做通道内归一化再加权相加（不归一化会让弱权通道被架空）
    for hs in (vh, bh):                                                       # 两个通道各自做通道内 min-max 归一化
        s = np.array([h["score"] for h in hs], dtype=float) if hs else np.zeros(0)  # 取出该通道分数
        lo, hi = (s.min(), s.max()) if len(s) else (0.0, 0.0)                   # 极值；空通道不会用到
        for h, v in zip(hs, (s - lo) / (hi - lo) if hi > lo else np.ones_like(s)):  # 全同分时给常数，避免除零
            h["nscore"] = float(v)                                             # 写入归一化分数
    pool = {}                                                                 # 以正文前缀为键合并两通道
    for h, w in [(x, C.VEC_WEIGHT) for x in vh] + [(x, C.BM25_WEIGHT) for x in bh]:  # 逐条带权重
        key = h["text"][:80]                                                   # 去重键
        if key in pool:                                                       # 两通道都召回
            pool[key]["score"] += w * h["nscore"]; pool[key]["src"] += "+" + h["src"]  # 分数相加、来源合并
        else:                                                                 # 首次出现
            pool[key] = dict(h, score=w * h["nscore"])                        # 记加权分
    return sorted(pool.values(), key=lambda x: -x["score"])[: k or C.FUSION_TOP_K]  # 取融合后前 k 条
def rerank(q, hits, n=None):                                                  # 重排：bge-reranker-v2-m3 精排
    if not hits:                                                              # 无候选
        return hits                                                           # 直接返回
    if "rerank" not in _M:                                                    # 重排模型单例，只加载一次
        from FlagEmbedding import FlagReranker                                # 官方封装
        _M["rerank"] = FlagReranker(C.RERANK_MODEL_PATH, use_fp16=True)        # 半精度省显存
    pairs = [[q, h["text"][: C.RERANK_MAX_CHARS]] for h in hits]                # 输入截断，不截断会按最长序列 padding 拖慢
    for h, s in zip(hits, _M["rerank"].compute_score(pairs, normalize=True)):   # 逐个打分
        h["rscore"] = float(s)                                                # 记录重排分
    return sorted(hits, key=lambda x: -x["rscore"])[: n or C.RERANK_TOP_N]      # 取前 N 条喂给大模型
def _fp():                                                                    # 缓存键必须带结构指纹，否则改动会被旧缓存静默吞掉
    idx = (C.DATA / "bm25.pkl").stat().st_mtime if (C.DATA / "bm25.pkl").exists() else 0  # 索引变 → 缓存失效
    src = b"".join((C.ROOT / "app" / n).read_bytes() for n in ("rag_core.py", "config.py", "ingest.py")) + str(idx).encode()  # 检索/分块/配置任一改动或索引重建 → 缓存失效
    return hashlib.sha256(src).hexdigest()[:16]                                 # 取前 16 位足够
def retrieve(q, mode="hybrid"):                                               # 检索总入口
    vh = vector_search(q) if mode in ("hybrid", "vector") else []              # 按模式决定是否走向量通道
    bh = bm25_search(q) if mode in ("hybrid", "bm25") else []                  # 按模式决定是否走关键词通道
    hits = rerank(q, fuse(vh, bh) if mode == "hybrid" else (vh or bh))         # 混合则先融合；单通道直接重排
    return hits, {"vec": len(vh), "bm25": len(bh), "kept": len(hits)}          # 返回结果与通道计数，供问题分析
def build_context(hits):                                                      # 拼接上下文，带可核验引用
    return "\n\n".join(f"【片段{i+1}｜{h['company']}{h['year']}年报 第{h['page']}页】\n{h['text']}" for i, h in enumerate(hits))  # 每段标明公司与页码
def generate(q, ctx):                                                         # 调 DeepSeek 生成答案
    sys_p = "你是上市公司年报问答助手。只能依据【检索片段】作答，不得编造；每个结论句末用【来源：公司 第X页】标注出处；片段中没有的信息就回答“年报中未提及”。"  # 约束防幻觉
    kw = {"reasoning_effort": C.LLM_REASONING_EFFORT} if C.LLM_REASONING_EFFORT else {}  # 推理型模型必须显式传 none，否则返回空
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": f"【检索片段】\n{ctx}\n\n【用户问题】\n{q}"}]  # 系统约束 + 片段 + 问题
    t0 = time.time()                                                          # 计时
    r = llm().chat.completions.create(model=C.LLM_MODEL, temperature=0.0, max_tokens=C.LLM_MAX_TOKEN, messages=msgs, **kw)  # 温度 0 保证评测可复现
    return (r.choices[0].message.content or "").strip(), round((time.time() - t0) * 1000)  # 返回答案与耗时毫秒
def answer(q, mode="hybrid", use_cache=True):                                 # 完整问答链路（评估与前端都调它）
    raw = f"{q}|{mode}|{C.VEC_WEIGHT}|{C.BM25_WEIGHT}|{C.RERANK_TOP_N}"         # 缓存键原始串：问题 + 所有影响结果的参数
    ckey = "gd7:ans:" + _fp() + ":" + hashlib.sha256(raw.encode()).hexdigest()[:32]  # 前缀 + 结构指纹 + 参数哈希
    t0 = time.time()                                                          # 计时起点放在缓存查询之前，命中时才能报出真实耗时
    if use_cache:                                                             # 允许关缓存（评估要测真实链路）
        try:                                                                  # 缓存故障不能影响主流程
            v = rds().get(ckey)                                                # 查缓存
        except Exception:                                                     # Redis 挂了
            v = None                                                          # 当作未命中
        if v:                                                                 # 命中
            return dict(json.loads(v), cached=True, latency_ms=round((time.time() - t0) * 1000))  # 报真实命中耗时（亚毫秒），不沿用上次生成耗时
    hits, stat = retrieve(q, mode)                                             # 混合检索 + 重排
    ans, ms = generate(q, build_context(hits))                                 # 生成答案
    out = {"question": q, "answer": ans, "mode": mode, "retrieval_stat": stat, "latency_ms": round((time.time() - t0) * 1000), "llm_ms": ms, "cached": False,  # 时延与通道计数
           "contexts": [{"text": h["text"], "company": h["company"], "year": h["year"], "page": h["page"], "src": h["src"], "rscore": round(h.get("rscore", 0), 4)} for h in hits]}  # 供 RAGAS 与问题分析
    if use_cache:                                                             # 需要缓存才写
        try:                                                                  # 写失败只记不抛
            rds().setex(ckey, C.CACHE_TTL, json.dumps(out, ensure_ascii=False))  # 带 TTL 写入
        except Exception:                                                     # 写缓存失败
            print("[cache] 写入失败，忽略")                                      # 不打断问答
    return out                                                                # 返回结果
