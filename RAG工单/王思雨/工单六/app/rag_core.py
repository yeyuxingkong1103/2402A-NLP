# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：核心层。向量检索(召回+重排) / BM25 全文检索(布尔+短语+模糊+多字段) / 混合融合(加权平均+投票+RRF)
#           / 三种重排器(bge、TF-IDF、用户反馈自适应) / Redis 缓存与会话 / DeepSeek 中英文多轮问答。
import hashlib, json, math, re, time                                               # 标准库
from typing import Dict, List, Optional, Tuple                                      # 类型标注
from . import config as C                                                           # 全局配置

_EMBED = None                                                                       # 向量模型单例
_RERANK = None                                                                      # 重排模型单例
_RERANK_ERR = ""                                                                    # 重排加载失败原因（记住后不再重试）
_MILVUS = None                                                                      # Milvus 客户端单例
_REDIS = None                                                                       # Redis 客户端单例
_REDIS_BAD = False                                                                  # Redis 是否处于降级状态
_JIEBA = None                                                                       # jieba 模块单例
_LLM = None                                                                         # DeepSeek 客户端单例
_STORE = {"fields": {}, "chunks": [], "vocab": []}                                  # BM25 多字段索引 + 块表 + 词表
_STORE_LOADED = False                                                               # 索引是否已尝试从磁盘载入
_LATIN = re.compile(r"[a-zA-Z]+")                                                   # 连续拉丁字母
_NUM = re.compile(r"\d[\d,\.]*")                                                    # 数字（含千分位/小数点）
_PHRASE = re.compile(r"\"([^\"]+)\"")                                               # 短语语法 "……"
_OPMAP = {"AND": "must", "OR": "should", "NOT": "neg", "与": "must", "或": "should", "非": "neg"}  # 布尔运算符
_FIELDS = ["chunk_id", "doc", "title", "text", "abstract", "source", "page_no", "page_end", "block_type", "locator"]  # Milvus 业务字段
_CTX = re.compile(r"【[^】]*】")                                                      # 引用标记（评测前必须剥离）

# ==================== 一、向量嵌入（默认 bge-m3，预留多模型配置）====================
def embed_model(name: str = "bge-m3"):                                              # 懒加载向量模型
    """首次调用加载并常驻；EMBED_MODELS 里登记了哪个名字就加载哪个本地路径。"""
    global _EMBED                                                                   # 声明单例
    if _EMBED is None:                                                              # 尚未加载
        from sentence_transformers import SentenceTransformer                        # 延迟导入，缩短启动时间
        _EMBED = SentenceTransformer(C.EMBED_MODELS.get(name, name), device=C.DEVICE)  # 本地路径加载，不联网
    return _EMBED                                                                   # 返回单例

def embed(texts: List[str], name: str = "bge-m3"):                                  # 批量向量化
    """L2 归一化后内积等价于余弦相似度，因此 Milvus 侧使用 IP 度量。"""
    import numpy as np                                                              # 延迟导入
    v = embed_model(name).encode(texts, normalize_embeddings=True, batch_size=C.EMBED_BATCH)  # 归一化编码
    return np.asarray(v, dtype=np.float32)                                          # 统一 float32，Milvus 要求

# ==================== 二、中文分词（BM25 的索引与查询单元）====================
def _jieba():                                                                       # 懒加载 jieba
    """首次调用加载词典并关日志，避免 import 期付出几百毫秒和刷屏。"""
    global _JIEBA                                                                   # 声明模块级单例
    if _JIEBA is None:                                                              # 尚未加载
        import jieba                                                                # 延迟导入
        jieba.setLogLevel(60)                                                       # 关闭 jieba 日志噪声
        _JIEBA = jieba                                                              # 常驻
    return _JIEBA                                                                   # 返回模块

def tokenize(text: str) -> List[str]:                                               # 中英混合分词
    """中文走 jieba，英文与数字单独补抽；数字同时收录带/去千分位两种形态，提高金额匹配率。"""
    text = text or ""                                                               # 防空
    toks = [w.strip().lower() for w in _jieba().lcut(text)                          # jieba 切词
            if w.strip() and w.strip() not in " \t\n　|，,。.；;：:（）()【】[]《》<>%￥$"]  # 丢弃纯标点与空白
    toks += [w.lower() for w in _LATIN.findall(text)]                               # 补抽英文（jieba 对长英文串切得差）
    for w in _NUM.findall(text):                                                    # 补抽数字，财报问答的关键线索
        toks += [w.lower(), w.replace(",", "").lower()]                             # 同时收录 "14,414.16" 与 "14414.16"
    return toks                                                                     # 返回词表

def _normtext(text: str) -> str:                                                    # 归一化文本，供短语匹配
    """去掉一切非字母/数字/汉字并转小写，"营业收入（万元）" 与 "营业收入万元" 视为同一串。"""
    return re.sub(r"[^0-9a-z一-鿿]+", "", (text or "").lower())             # 仅保留可比较字符

# ==================== 三、BM25 全文检索（多字段 + 布尔 + 短语 + 模糊）====================
def bm25_build(chunks: List[Dict]) -> Dict:                                         # 构建索引
    """标题/正文/摘要各建一棵 BM25，检索时按 FIELD_WEIGHTS 加权求和，即工单要求的多字段检索。"""
    from rank_bm25 import BM25Okapi                                                 # 延迟导入
    fields = {}                                                                     # 字段名 → BM25
    for key in ("title", "body", "abstract"):                                       # 三个字段分别建索引
        corpus = [tokenize(c.get({"title": "title", "body": "text", "abstract": "abstract"}[key], ""))  # 该字段分词
                  for c in chunks]                                                  # 覆盖全部块
        fields[key] = BM25Okapi(corpus) if any(corpus) else None                    # 全空字段不建索引，避免除零
    for c in chunks:                                                                # 为每个块预计算检索辅助结构
        c["_tok"] = set(tokenize(f"{c.get('title','')} {c.get('text','')}"))         # 词集合，供布尔筛选
        c["_norm"] = _normtext(f"{c.get('title','')}{c.get('text','')}")             # 归一化串，供短语匹配
    return {"fields": fields, "chunks": chunks}                                      # 打包成索引对象

def bm25_save(store: Dict) -> None:                                                  # 索引落盘
    """重建需数分钟，落盘后重启免于重跑分词。"""
    import pickle                                                                   # 用 pickle 存对象
    C.BM25_FILE.parent.mkdir(parents=True, exist_ok=True)                            # 确保目录存在
    with open(C.BM25_FILE, "wb") as f:                                               # 二进制写入
        pickle.dump(store, f)                                                        # 一次写入

def bm25_load() -> Dict:                                                             # 索引读盘
    """文件缺失或损坏时返回空结构，由调用方提示先执行 POST /ingest。"""
    global _STORE, _STORE_LOADED                                                     # 写入模块级缓存
    import pickle                                                                    # 反序列化
    _STORE_LOADED = True                                                             # 标记已尝试载入
    try:                                                                             # 读盘可能失败
        if C.BM25_FILE.exists():                                                     # 有索引文件
            _STORE = pickle.load(open(C.BM25_FILE, "rb"))                            # 载入
    except Exception as exc:                                                         # 版本不符/文件损坏
        print(f"[BM25] 索引读取失败：{type(exc).__name__}: {exc}")                     # 明确报因
    return _STORE                                                                    # 返回（可能为空）

def query_parse(q: str) -> Dict:                                                     # 解析查询语法
    """把查询拆成 必须/可选/排除 词项 + 短语 + 模糊词，支持 AND|OR|NOT(与|或|非)、"短语"、词~。"""
    phrases = [p.strip() for p in _PHRASE.findall(q or "") if p.strip()]              # 抽出引号内短语
    rest = _PHRASE.sub(" ", q or "")                                                 # 余下部分
    acc = {"must": [], "should": [], "neg": [], "phrases": phrases, "fuzzy": []}      # 结果容器
    state = "should"                                                                 # 默认 OR 语义（BM25 本身按命中打分）
    for raw in rest.split():                                                         # 按空白切词
        up = raw.upper()                                                             # 大小写不敏感
        if up in _OPMAP or raw in _OPMAP:                                            # 命中布尔运算符
            state = _OPMAP.get(up) or _OPMAP[raw]                                    # 切换后续词项的归属
            continue                                                                 # 运算符本身不入词项
        if raw.endswith("~"):                                                        # 词~ 表示模糊匹配
            acc["fuzzy"].append(raw[:-1])                                            # 记入模糊表
            continue                                                                 # 不再当普通词
        acc[state] += tokenize(raw)                                                  # 常规词项按当前状态归位
    return acc                                                                       # 返回解析结果

def _fuzzy_hits(terms: List[str], limit: int) -> List[Tuple[int, float]]:            # 模糊匹配
    """在**块文本**上做近似子串搜索（partial_ratio），容忍错别字，如 "程加明~" 命中 "程家明"。

    早先的实现是拿查询词去比对 2.7 万条的分词词表，行不通：jieba 会把中文人名切开
    （"程家明" 根本不在词表里），而且 3 字人名错 1 字的 ratio 只有 66.7，落在阈值 80 以下。
    改成对块正文做部分匹配后，"程加明" 能直接对齐 "程家明"。
    """
    chunks = _STORE["chunks"]                                                        # 块表
    if not terms or not chunks:                                                      # 无模糊词或索引为空
        return []                                                                    # 返回空
    from rapidfuzz import process, fuzz                                              # 延迟导入
    norms = [c.get("_norm", "") for c in chunks]                                     # 块的归一化正文
    hit: Dict[int, float] = {}                                                       # 块号 → 最高相似度
    for t in terms:                                                                  # 逐个模糊词
        tn = _normtext(t)                                                            # 归一化
        if len(tn) < 2:                                                              # 单字太短，误伤面太大
            continue                                                                 # 跳过
        for _, score, idx in process.extract(tn, norms, scorer=fuzz.partial_ratio,   # 近似子串搜索
                                             limit=limit, score_cutoff=C.FUZZY_CUTOFF):  # 取前 limit 且过阈值
            hit[int(idx)] = max(hit.get(int(idx), 0.0), float(score))                 # 保留最高分
    return sorted(hit.items(), key=lambda kv: -kv[1])                                # 按相似度降序

def bm25_search(query: str, top_k: int = None, text: str = None) -> List[Dict]:      # 关键词检索入口
    """返回带 bm25_score 的 top_k 块；索引不可用或无命中时返回空，由上层降级到纯向量。"""
    top_k = top_k or C.BM25_TOP_K                                                    # 默认召回条数
    if not _STORE["chunks"] and not _STORE_LOADED:                                   # 首次使用且未载入
        bm25_load()                                                                  # 惰性载入磁盘索引
    chunks = _STORE["chunks"]                                                        # 块表
    if not chunks:                                                                   # 索引未建
        print("[检索降级] BM25 索引不可用，关键词通道返回空")                            # 明确报因
        return []                                                                    # 返回空
    parsed = query_parse(query)                                                      # 解析布尔/短语/模糊语法
    q = (tokenize(text) if text else parsed["must"] + parsed["should"]               # 只用解析出的词项，避免把 AND/NOT/~ 当检索词
         + tokenize(" ".join(parsed["phrases"])))                                    # 短语内容也要参与打分，否则纯短语查询会因"无词项"提前返回空
    if not q and not parsed["fuzzy"]:                                                # 既无词项也无模糊词（如纯 NOT 查询）
        return []                                                                    # 返回空
    scores = None                                                                    # 多字段加权得分
    if not q:                                                                        # 纯模糊查询（如 "程加明~"）：没有词项可打分
        import numpy as np                                                           # 延迟导入
        scores = np.zeros(len(chunks), dtype=float)                                  # 从零分起步，完全靠下面的模糊命中给分
    for key, w in C.FIELD_WEIGHTS.items():                                           # 逐字段打分加权
        idx = _STORE["fields"].get(key)                                              # 该字段索引
        if idx is None or not w:                                                     # 字段未建或权重为 0
            continue                                                                 # 跳过
        try:                                                                         # 打分可能因词表不符失败
            s = idx.get_scores(q)                                                    # 该字段得分向量
            scores = s * w if scores is None else scores + s * w                     # 累加到总分
        except Exception as exc:                                                     # 索引与查询不匹配
            print(f"[检索降级] BM25 字段 {key} 打分失败：{type(exc).__name__}")          # 明确报因
    if scores is None:                                                               # 三个字段都不可用
        return []                                                                    # 降级
    if parsed["fuzzy"]:                                                              # 有模糊词：叠加近似子串命中
        import numpy as np                                                           # 延迟导入
        mx = float(scores.max()) or 1.0                                              # 以最高 BM25 为尺子
        for i, sc in _fuzzy_hits(parsed["fuzzy"], top_k):                            # 逐条近似命中
            scores[i] += (sc / 100.0) * mx                                           # 折算到与 BM25 同量级后叠加
    masked = _apply_mask(chunks, parsed, scores)                                     # 施加布尔与短语筛选
    import numpy as np                                                               # 延迟导入
    order = np.argsort(masked)[::-1][:top_k]                                         # 取分数最高的若干
    out = []                                                                         # 结果
    for i in order:                                                                  # 逐个组装
        s = float(masked[i])                                                         # 该块得分
        if s <= 0:                                                                   # 零分说明一个词都没命中
            continue                                                                 # 丢弃，避免噪声进重排
        out.append({k: v for k, v in chunks[int(i)].items() if not k.startswith("_")} | {"bm25_score": s})  # 剥离内部字段
    return out                                                                       # 返回命中

def _apply_mask(chunks: List[Dict], parsed: Dict, scores) -> "object":                # 布尔/短语筛选
    """AND 要求词项全在、NOT 要求全不在、短语要求归一化子串存在；无约束时原样返回。"""
    must, neg = parsed["must"], parsed["neg"]                                        # 必含、必不含
    phrases = [_normtext(p) for p in parsed["phrases"]]                              # 归一化短语
    if not must and not neg and not phrases:                                         # 无任何约束
        return scores                                                                # 跳过筛选，省一次全表扫描
    import numpy as np                                                               # 延迟导入
    mask = np.ones(len(chunks), dtype=bool)                                          # 初始全通过
    for i, c in enumerate(chunks):                                                   # 逐块判定
        toks, norm = c.get("_tok", set()), c.get("_norm", "")                        # 预计算的词集合与归一化串
        if must and not set(must) <= toks:                                           # AND：必须全部出现
            mask[i] = False                                                          # 淘汰
        elif neg and (set(neg) & toks):                                              # NOT：一个都不能出现
            mask[i] = False                                                          # 淘汰
        elif phrases and not all(p in norm for p in phrases):                        # 短语：全部要作为连续子串出现
            mask[i] = False                                                          # 淘汰
    return scores * mask                                                             # 约束外的块直接归零

# ==================== 四、Milvus 向量检索（召回）====================
def milvus_client():                                                                 # 建客户端
    """连接 Milvus；失败由上层捕获并如实上报降级，不让异常冒到接口外。"""
    global _MILVUS                                                                   # 单例
    if _MILVUS is None:                                                              # 尚未连接
        import logging                                                               # 抑制 pymilvus 的 DEBUG 刷屏
        logging.getLogger("pymilvus").setLevel(logging.WARNING)                      # 只留警告以上
        from pymilvus import MilvusClient                                            # 延迟导入
        _MILVUS = MilvusClient(uri=C.MILVUS_URI, timeout=C.MILVUS_TIMEOUT)           # 带超时，Milvus 挂了要快速失败
    return _MILVUS                                                                   # 返回单例

def vector_search(question: str, top_k: int = None) -> List[Dict]:                   # 向量召回
    """问题编码后在 Milvus 取余弦最相近的 top_k；强一致读规避"刚写入读不到"。"""
    top_k = top_k or C.VEC_TOP_K                                                     # 默认召回条数
    try:                                                                             # 向量化与检索分开兜底：故障源不同
        qvec = embed([question])[0].tolist()                                          # 问题向量
    except Exception as exc:                                                         # 模型路径错/显存不足
        print(f"[检索降级] 向量模型不可用：{type(exc).__name__}: {exc}")               # 明确指向模型
        return []                                                                    # 返回空
    try:                                                                             # 再检索
        hits = milvus_client().search(collection_name=C.MILVUS_COLLECTION, data=[qvec],  # 单向量查询
                                      limit=top_k, output_fields=_FIELDS,             # 回带业务字段
                                      consistency_level="Strong")                    # 强一致，规避刚写入读不到
    except Exception as exc:                                                         # 连接失败/超时/集合不存在
        print(f"[检索降级] Milvus 检索失败：{type(exc).__name__}: {exc}")              # 明确指向 Milvus
        return []                                                                    # 返回空
    out = []                                                                         # 结果
    for h in (hits[0] if hits else []):                                              # 展开第一组
        ent = dict(h.get("entity", {}))                                              # 实体字段
        ent["vec_score"] = float(h.get("distance", 0.0))                             # 附上向量得分
        out.append(ent)                                                              # 收集
    return out                                                                       # 返回

# ==================== 五、混合检索融合（加权平均 / 投票 / RRF）====================
def _minmax(vals: List[float]) -> List[float]:                                       # 通道内归一化
    """把一路分数线性压到 0~1。必须先归一化再加权，否则两路数量级不同会直接架空弱权通道。"""
    if not vals:                                                                     # 空
        return []                                                                    # 返回空
    lo, hi = min(vals), max(vals)                                                    # 极值
    return [(v - lo) / (hi - lo) if hi > lo else 1.0 for v in vals]                  # 全等时统一给 1

def fuse(vec: List[Dict], kw: List[Dict], method: str, vw: float, bw: float, top_k: int) -> List[Dict]:  # 融合入口
    """按 chunk_id 归并两路结果；weighted=加权平均、rrf=倒数排名融合、vote=投票机制。"""
    pool = {}                                                                        # chunk_id → 记录
    for tag, hits, w in (("vec", vec, vw), ("bm25", kw, bw)):                         # 两路分别归一化
        norms = _minmax([float(h.get("vec_score", h.get("bm25_score", 0.0))) for h in hits])  # 通道内归一化
        for rank, (h, nv) in enumerate(zip(hits, norms)):                            # 逐条
            cid = h.get("chunk_id") or h.get("id")                                   # 块号（无则用主键）
            item = pool.setdefault(cid, {k: v for k, v in h.items() if not k.startswith("_")} |  # 首见时建记录
                                   {"chunk_id": cid, "_ch": {}, "fused": 0.0})       # 初始化通道明细
            item["_ch"][tag] = {"rank": rank, "norm": nv, "w": w}                    # 记录该通道的排名与归一化分
    for item in pool.values():                                                       # 计算最终融合分
        ch = item["_ch"]                                                             # 通道明细
        if method == "rrf":                                                          # RRF：Σ w/(k+rank)
            item["fused"] = sum(c["w"] / (C.RRF_K + c["rank"] + 1) for c in ch.values())  # 排名越前分越高
        elif method == "vote":                                                       # 投票：被几路召回 + 排名微调
            item["fused"] = sum(c["w"] for c in ch.values()) - 0.001 * min(c["rank"] for c in ch.values())  # 票数优先
        else:                                                                        # 加权平均（默认）
            item["fused"] = sum(c["w"] * c["norm"] for c in ch.values())             # 归一化分加权求和
        item["channels"] = ",".join(sorted(ch))                                      # 命中的通道，便于前端展示
        item["rank_detail"] = {t: c["rank"] + 1 for t, c in ch.items()}              # 两路名次，便于对比排序变化
    ranked = sorted(pool.values(), key=lambda x: x["fused"], reverse=True)           # 融合分降序
    return ranked[:top_k]                                                            # 截断

# ==================== 六、三种重排算法 + 级联 ====================
def rerank_model():                                                                  # 懒加载 bge-reranker
    """加载失败返回 None 并记住原因，由调用方回退到融合排序并如实标记 degraded。"""
    global _RERANK, _RERANK_ERR                                                      # 全局状态
    if _RERANK is not None or _RERANK_ERR:                                           # 已就绪或已确认失败
        return _RERANK                                                               # 不重复尝试，避免每请求卡住
    try:                                                                             # 加载可能因路径/显存失败
        from FlagEmbedding import FlagReranker                                       # 延迟导入
        _RERANK = FlagReranker(C.RERANK_MODEL_PATH, use_fp16=(C.DEVICE == "cuda"))   # 仅 GPU 开 fp16
    except Exception as exc:                                                         # 加载失败
        _RERANK_ERR = f"{type(exc).__name__}: {exc}"                                 # 记住原因
        print(f"[重排降级] 重排模型加载失败：{_RERANK_ERR}")                            # 明确报因
    return _RERANK                                                                   # 返回

def rerank_bge(query: str, cands: List[Dict], top_n: int) -> Tuple[List[Dict], List[str]]:  # 交叉编码重排
    """用 bge-reranker-v2-m3 成对精排；打分输入截断到 RERANK_MAX_CHARS 以控制 padding 开销。"""
    model = rerank_model()                                                           # 取模型
    if not cands:                                                                    # 无候选
        return [], []                                                                # 直接返回
    if model is None:                                                                # 模型不可用
        return cands[:top_n], ["rerank"]                                             # 回退到融合排序并如实上报降级
    pairs = [[query, (c.get("text") or "")[:C.RERANK_MAX_CHARS]] for c in cands]      # 查询-片段配对
    try:                                                                             # 打分可能因显存失败
        s = model.compute_score(pairs, normalize=True)                                # 归一化到 0~1
    except Exception as exc:                                                         # 打分失败
        print(f"[重排降级] bge 打分失败：{type(exc).__name__}: {exc}")                 # 明确报因
        return cands[:top_n], ["rerank"]                                             # 回退
    s = [float(s)] if isinstance(s, (int, float)) else [float(x) for x in s]          # 单条时返回标量，统一成列表
    ranked = sorted(zip(cands, s), key=lambda t: t[1], reverse=True)                  # 按重排分降序
    return [dict(c, rerank_bge=sc) for c, sc in ranked[:top_n]], []                  # 返回带分数的候选

def rerank_tfidf(query: str, cands: List[Dict], top_n: int) -> Tuple[List[Dict], List[str]]:  # TF-IDF 重排
    """在候选集上现建词频矩阵，用余弦相似度精排；纯 CPU、毫秒级，适合当级联的第一级。"""
    texts = [(c.get("text") or "") for c in cands]                                   # 候选文本
    if not cands or not any(texts):                                                  # 无候选或全空
        return cands[:top_n], []                                                     # 原样返回
    try:                                                                             # 建矩阵可能因空词表失败
        from sklearn.feature_extraction.text import TfidfVectorizer                   # 延迟导入
        from sklearn.metrics.pairwise import cosine_similarity                        # 延迟导入
        vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)  # 用同一套分词，保证口径一致
        m = vec.fit_transform(texts + [query])                                        # 候选 + 查询一起建矩阵
        s = cosine_similarity(m[-1], m[:-1])[0]                                       # 查询与各候选的余弦
    except Exception as exc:                                                         # 词表为空等
        print(f"[重排降级] TF-IDF 打分失败：{type(exc).__name__}: {exc}")              # 明确报因
        return cands[:top_n], ["rerank_tfidf"]                                        # 回退
    ranked = sorted(zip(cands, [float(x) for x in s]), key=lambda t: t[1], reverse=True)  # 降序
    return [dict(c, rerank_tfidf=sc) for c, sc in ranked[:top_n]], []                # 返回

def rerank_adaptive(query: str, cands: List[Dict], top_n: int) -> Tuple[List[Dict], List[str]]:  # 反馈自适应重排
    """用 Redis 中的用户反馈调整片段/文档权重：score = 基础分 × (1 + α·反馈净票)。"""
    if not cands:                                                                    # 无候选
        return [], []                                                                # 返回空
    base = _minmax([float(c.get("fused", c.get("vec_score", 0.0))) for c in cands])    # 基础分归一化
    out = []                                                                         # 结果
    for c, b in zip(cands, base):                                                     # 逐条加权
        up, down = fb_stats(c.get("chunk_id", ""), c.get("doc", ""))                  # 取该片段与该文档的净反馈
        adj = (up - down) / (up + down + 1.0)                                         # 平滑后的反馈倾向（-1~1）
        out.append(dict(c, adapt_score=round(float(b) * (1 + C.ADAPTIVE_ALPHA * adj), 6),  # 反馈加权后的分数
                        feedback={"up": up, "down": down}))                            # 同时回带原始票数，便于核查
    out.sort(key=lambda x: x["adapt_score"], reverse=True)                            # 降序
    return out[:top_n], []                                                            # 截断返回

_RERANKERS = {"bge": rerank_bge, "tfidf": rerank_tfidf, "adaptive": rerank_adaptive}  # 重排器注册表

def rerank_pipeline(query: str, cands: List[Dict], names: List[str], top_n: int) -> Tuple[List[Dict], List[str]]:  # 级联重排
    """按顺序级联：便宜的 TF-IDF 先截断到 CASCADE_MID，再由 bge 精排，顺序不同耗时不同。"""
    deg, cur = [], list(cands)                                                        # 降级记录与当前候选
    names = [n for n in (names or []) if n in _RERANKERS]                             # 过滤未知重排器名
    for i, name in enumerate(names):                                                  # 逐级执行
        keep = top_n if i == len(names) - 1 else max(top_n, C.CASCADE_MID)            # 末级才收到 top_n，中间级留宽
        cur, d = _RERANKERS[name](query, cur, keep)                                   # 执行该级
        deg += d                                                                      # 汇总降级
    return cur[:top_n], sorted(set(deg))                                              # 返回最终候选与降级原因

# ==================== 七、Redis：缓存 / 反馈 / 会话 ====================
def redis_client():                                                                  # Redis 单例
    """显式 protocol=2：本机 Redis 5.0.14 不支持 RESP3(HELLO)，redis-py 8 默认协商会直接失败。"""
    global _REDIS                                                                     # 单例
    if _REDIS is None:                                                               # 尚未连接
        import redis                                                                 # 延迟导入
        _REDIS = redis.Redis.from_url(C.REDIS_URL, protocol=C.REDIS_PROTOCOL,        # 强制 RESP2
                                      decode_responses=True, socket_timeout=3,       # 直接返回 str，超时要短
                                      socket_connect_timeout=3)                      # 缓存不该拖慢主流程
    return _REDIS                                                                    # 返回

def _fingerprint() -> str:                                                           # 代码结构指纹
    """对核心源码取哈希：改了代码就自动换缓存键，避免旧缓存静默吞掉新字段（如 degraded）。"""
    h = hashlib.sha256()                                                             # 累积哈希
    for f in ("config.py", "rag_core.py", "ingest.py"):                              # 参与指纹的源码
        p = C.ROOT / "app" / f                                                        # 文件路径
        h.update(p.read_bytes() if p.exists() else b"")                               # 累积内容
    return h.hexdigest()[:12]                                                        # 取前 12 位

def _index_fingerprint() -> str:                                                     # 知识库指纹
    """索引文件变了（重新分块/换 PDF）键就变，避免复用旧知识库的答案。"""
    h = hashlib.sha256()                                                             # 累积哈希
    for p in (C.BM25_FILE,):                                                         # 索引文件
        if p.exists():                                                               # 存在才计
            h.update(str(p.stat().st_mtime_ns).encode())                             # 用修改时间，避免每次读大文件
    return h.hexdigest()[:12]                                                        # 取前 12 位

def _ckey(kind: str, raw: str) -> str:                                               # 组装缓存键
    """键 = 工单前缀 + 代码指纹 + 知识库指纹 + 内容哈希，四段缺一不可。"""
    return f"gd6:{kind}:{_fingerprint()}:{_index_fingerprint()}:{hashlib.sha256(raw.encode()).hexdigest()[:32]}"  # 内容哈希防长键

def cache_get(kind: str, raw: str):                                                  # 读缓存
    """读失败视作未命中并标记降级，绝不因缓存故障中断主流程。"""
    global _REDIS_BAD                                                                 # 降级标志
    try:                                                                             # 网络/服务异常
        v = redis_client().get(_ckey(kind, raw))                                     # 取值
        _REDIS_BAD = False                                                           # 成功则复位降级标志
        return json.loads(v) if v else None                                          # 反序列化
    except Exception as exc:                                                         # 读失败
        _REDIS_BAD = True                                                            # 记录降级
        print(f"[缓存降级] 读缓存失败：{type(exc).__name__}: {exc}")                   # 明确报因
        return None                                                                  # 视作未命中

def cache_set(kind: str, raw: str, value, ttl: int = None) -> None:                  # 写缓存
    """写失败只影响下次重算，不标记降级（不影响本次结果的正确性）。"""
    try:                                                                             # 网络/服务异常
        redis_client().setex(_ckey(kind, raw), ttl or C.CACHE_TTL,                    # 带过期时间
                             json.dumps(value, ensure_ascii=False))                   # 中文不转义
    except Exception as exc:                                                         # 写失败
        print(f"[缓存降级] 写缓存失败：{type(exc).__name__}: {exc}")                   # 明确报因

def fb_add(chunk_id: str, rating: int, doc: str = "") -> None:                       # 写用户反馈
    """片段级与文档级各记一票；自适应重排器读它来调整权重。"""
    try:                                                                             # Redis 可能不可用
        r = redis_client()                                                           # 客户端
        for key in (f"gd6:fb:chunk:{chunk_id}" if chunk_id else None,                 # 片段级票箱
                    f"gd6:fb:doc:{doc}" if doc else None):                            # 文档级票箱
            if key:                                                                   # 键有效才写
                r.hincrby(key, "up" if rating > 0 else "down", 1)                     # 点赞加到 up，点踩加到 down
                r.expire(key, C.CACHE_TTL)                                            # 反馈也带有效期
    except Exception as exc:                                                         # 写失败
        print(f"[反馈降级] 反馈写入失败：{type(exc).__name__}: {exc}")                  # 明确报因

def fb_stats(chunk_id: str, doc: str = "") -> Tuple[int, int]:                       # 读反馈票数
    """返回 (点赞数, 点踩数)；两处票箱相加，Redis 不可用时按 0 票处理。"""
    try:                                                                             # Redis 可能不可用
        r, up, down = redis_client(), 0, 0                                           # 客户端与计数器
        for key in (f"gd6:fb:chunk:{chunk_id}" if chunk_id else None,                 # 片段级
                    f"gd6:fb:doc:{doc}" if doc else None):                            # 文档级
            if key:                                                                   # 键有效才读
                up += int(r.hget(key, "up") or 0)                                     # 累加点赞
                down += int(r.hget(key, "down") or 0)                                 # 累加点踩
        return up, down                                                               # 返回
    except Exception:                                                                # 读失败
        return 0, 0                                                                   # 按无反馈处理

def hist_get(session_id: str) -> List[Dict]:                                          # 读会话历史
    """返回最近 HISTORY_TURNS 轮；Redis 不可用时返回空，问答退化为单轮。"""
    try:                                                                             # Redis 可能不可用
        return [json.loads(x) for x in redis_client().lrange(f"gd6:hist:{session_id}", -C.HISTORY_TURNS * 2, -1)]  # 取尾部
    except Exception:                                                                # 读失败
        return []                                                                     # 按无历史处理

def hist_add(session_id: str, role: str, content: str) -> None:                       # 追加会话历史
    """用 rpush + ltrim + expire 维持定长滑动窗口，防止历史无限膨胀。"""
    try:                                                                             # Redis 可能不可用
        r = redis_client()                                                           # 客户端
        key = f"gd6:hist:{session_id}"                                                # 会话键
        r.rpush(key, json.dumps({"role": role, "content": (content or "")[:C.SESSION_MAX_CHARS]}, ensure_ascii=False))  # 截断后入队
        r.ltrim(key, -C.HISTORY_TURNS * 2, -1)                                        # 只留最近若干条
        r.expire(key, C.HISTORY_TTL)                                                  # 续期
    except Exception as exc:                                                         # 写失败
        print(f"[会话降级] 历史写入失败：{type(exc).__name__}: {exc}")                  # 明确报因

# ==================== 八、DeepSeek 问答（中英文）====================
def llm_client():                                                                     # DeepSeek 客户端单例
    """必须复用同一个客户端：每次新建都要重付 DNS+TLS+连接池构建，本机实测单次多花约 1 秒，
    足以把"检索+生成"的端到端耗时从 ~1.2 秒推到 2.5 秒以上，直接顶穿 3 秒指标。"""
    global _LLM                                                                       # 单例
    if _LLM is None:                                                                  # 尚未创建
        if not C.DEEPSEEK_API_KEY:                                                    # 未配密钥
            raise RuntimeError("未配置 DEEPSEEK_API_KEY 环境变量")                     # 明确报因
        from openai import OpenAI                                                     # 延迟导入
        _LLM = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL,            # OpenAI 兼容客户端
                      timeout=C.LLM_TIMEOUT, max_retries=0)                           # 重试自己控制，关掉 SDK 内建重试
    return _LLM                                                                       # 返回单例

def llm_chat(messages: List[Dict], timeout: int = None, max_tokens: int = None) -> str:  # 调大模型
    """空内容视为失败并重试：推理型模型的思考过程会吃光 max_tokens 后返回空串。"""
    cli = llm_client()                                                                # 复用单例客户端
    kwargs = {"timeout": timeout} if timeout else {}                                  # 仅在显式传超时时覆盖
    last = None                                                                       # 最后一次异常
    for attempt in range(C.LLM_RETRIES + 1):                                          # 首次 + 重试
        kw = {"model": C.LLM_MODEL, "messages": messages,                              # 基础参数
              "max_tokens": max_tokens or C.LLM_MAX_TOKENS,                           # 生成长度
              "temperature": C.LLM_TEMPERATURE}                                       # 温度 0
        if C.LLM_REASONING_EFFORT:                                                    # 显式配置了才传
            kw["reasoning_effort"] = C.LLM_REASONING_EFFORT                            # 推理型模型必须传 none
        try:                                                                          # 调用可能超时/限流
            content = (cli.chat.completions.create(**kw, **kwargs).choices[0].message.content or "").strip()  # 取正文
            if content:                                                               # 有内容即成功
                return content                                                        # 返回
            last = RuntimeError("大模型返回空内容")                                     # 空内容当失败，触发重试
        except Exception as exc:                                                      # 网络/限流/402
            last = exc                                                                # 记录
        if "Timeout" in type(last).__name__:                                          # 超时重试只会把耗时翻倍
            break                                                                    # 直接放弃，交给上层降级
        if attempt < C.LLM_RETRIES:                                                   # 还没到末次
            print(f"[大模型] 第 {attempt + 1} 次失败（{type(last).__name__}: {str(last)[:80]}），准备重试", flush=True)  # 重试必须留痕，否则尾延迟无法归因
            time.sleep(1.0 * (attempt + 1))                                           # 线性退避
    raise RuntimeError(f"大模型调用失败：{type(last).__name__}: {last}")                # 重试耗尽

def _looks_english(text: str) -> bool:                                                # 判断问题语种
    """ASCII 字母数多于汉字数即视为英文提问，用来切换回答语言。"""
    ascii_n = len(re.findall(r"[a-zA-Z]", text or ""))                                # 英文字母数
    cjk = len(re.findall(r"[一-鿿]", text or ""))                             # 汉字数
    return ascii_n > cjk                                                              # 英文占优

def retrieve(query: str, mode: str = "hybrid", top_k: int = None, vw: float = None,  # 统一检索入口
             bw: float = None, fusion: str = "weighted", rerankers: List[str] = None) -> Tuple[List[Dict], List[str]]:  # 返回候选与降级
    """mode 取 vector/全文(bm25)/hybrid；hybrid 下并行两路再融合；最后按重排器顺序级联精排。"""
    top_k = top_k or C.RERANK_TOP_N                                                   # top_k = 最终返回条数（不是融合池大小）
    vw = C.VEC_WEIGHT if vw is None else vw                                            # 向量权重
    bw = C.BM25_WEIGHT if bw is None else bw                                           # BM25 权重
    deg = []                                                                          # 降级记录
    vec = vector_search(query, C.VEC_TOP_K) if mode in ("vector", "hybrid") else []    # 向量通道召回
    kw = bm25_search(query, C.BM25_TOP_K) if mode in ("bm25", "hybrid") else []        # 全文通道召回
    if mode == "vector" and not vec:                                                  # 向量通道空
        deg.append("vector")                                                          # 如实上报
    if mode == "bm25" and not kw:                                                     # 全文通道空
        deg.append("bm25")                                                            # 如实上报
    if mode == "hybrid":                                                              # 混合：两路融合
        pool = fuse(vec, kw, fusion, vw, bw, C.FUSION_TOP_K)                          # 融合池固定为 30，绝不用 top_k 截断
    else:                                                                             # 单路：直接归一化
        src = vec if mode == "vector" else kw                                         # 选通道
        norm = _minmax([float(x.get("vec_score", x.get("bm25_score", 0.0))) for x in src])  # 单通道内归一化
        pool = [dict(h) | {"fused": nv, "channels": mode}                             # 补融合分字段，保持下游一致
                for h, nv in zip(src, norm)][:C.FUSION_TOP_K]                         # 同样只截到融合池
    if not pool:                                                                      # 两路都空
        return [], sorted(set(deg + ["no_hit"]))                                       # 明确上报无命中
    cands, d = rerank_pipeline(query, pool, C.DEFAULT_RERANKERS if rerankers is None else rerankers, top_k)  # 显式传 [] 表示不重排，不能被兜回默认
    return cands, sorted(set(deg + d))                                                # 返回

def build_context(cands: List[Dict]) -> str:                                          # 拼检索上下文
    """每条片段带来源标记，便于答案引用与人工核对。"""
    blocks = []                                                                       # 片段块
    for i, c in enumerate(cands, 1):                                                  # 逐条编号
        blocks.append(f"[片段{i}] 来源：{c.get('doc','')} 第{c.get('page_no','?')}页 "  # 出处
                      f"章节：{c.get('title','') or '（无）'}\n{(c.get('text') or '').strip()}")  # 正文
    return "\n\n".join(blocks)                                                        # 拼接

def answer(question: str, session_id: str = "default", mode: str = "hybrid",        # 问答入口
           top_k: int = None, vw: float = None, bw: float = None, fusion: str = "weighted",  # 检索参数
           rerankers: List[str] = None, use_cache: bool = True) -> Dict:              # 是否用缓存
    """检索 → 拼上下文 → DeepSeek 生成 → 写缓存与会话；返回值含答案、引用、上下文与降级原因。"""
    t0 = time.time()                                                                  # 计时起点
    hist = hist_get(session_id)                                                        # 取会话历史
    q_ret = question                                                                   # 实际用于检索的问句
    if hist and (len(question) <= 12 or re.search(r"他|她|它|该|这|那|其|这个|那个|上述", question)):  # 短问句或含指代
        q_ret = f"{hist[-1]['content']}　{question}"                                    # 拼上一轮用户问句，完成指代消解
    raw_key = "|".join([question, mode, str(top_k or C.FUSION_TOP_K),                  # 缓存键内容：问题
                        f"{vw if vw is not None else C.VEC_WEIGHT}-{bw if bw is not None else C.BM25_WEIGHT}",  # 权重
                        fusion, ",".join(C.DEFAULT_RERANKERS if rerankers is None else rerankers),  # 融合与重排配置
                        hashlib.sha256("".join(h["content"][:40] for h in hist[-4:]).encode()).hexdigest()[:8]])  # 历史摘要
    if use_cache:                                                                      # 允许走缓存
        hit = cache_get("answer", raw_key)                                             # 读缓存
        if hit:                                                                        # 命中
            return dict(hit, cached=True, latency_ms=int((time.time() - t0) * 1000))   # 回填真实耗时并标记命中
    cands, deg = retrieve(q_ret, mode, top_k, vw, bw, fusion, rerankers)               # 检索
    if not cands:                                                                      # 无命中
        ans = "未在招股说明书知识库中检索到与问题相关的内容，无法作答。" if not _looks_english(question) else "No relevant content was found in the knowledge base."  # 双语兜底
    else:                                                                              # 有命中
        ctx = build_context(cands)                                                     # 上下文
        lang = "请用英文作答。" if _looks_english(question) else "请用中文作答。"         # 语言指令
        sys_p = (f"你是招股说明书问答助手。只能依据给定的【检索片段】作答，不得编造；{lang}"       # 系统提示
                 "关键数字与结论必须能在片段中找到，并在句末用【来源：文档 第X页】标注；"          # 引用要求
                 "若片段不足以回答，直接说明无法回答，不要臆测。")                            # 兜底要求
        budget = (C.LLM_DEADLINE_S - (time.time() - t0)) if C.LLM_DEADLINE_S > 0 else None  # 总预算扣掉检索已用时，剩下的给生成
        if budget is not None and budget <= 0.3:                                        # 预算已耗尽
            ans = "响应预算已用尽，以下为检索到的原文片段：\n" + build_context(cands[:3])   # 直接降级，不再等大模型
            deg = sorted(set(deg + ["llm_deadline"]))                                   # 如实上报降级原因
        else:                                                                           # 还有预算
            try:                                                                        # 生成可能失败或超时
                ans = llm_chat([{"role": "system", "content": sys_p}] + hist[-C.HISTORY_TURNS * 2:] +  # 带历史实现多轮
                               [{"role": "user", "content": f"【检索片段】\n{ctx}\n\n【用户问题】\n{question}"}],  # 本轮提问
                               timeout=budget)                                          # 预算即本次调用的超时上限
            except Exception as exc:                                                    # 大模型不可用/超预算
                print(f"[问答降级] {type(exc).__name__}: {exc}")                         # 明确报因
                ans = "大模型调用失败或超时，以下为检索到的原文片段：\n" + build_context(cands[:3])  # 降级为只列原文，不编造
                deg = sorted(set(deg + ["llm"]))                                        # 如实上报
    refs = [{"doc": c.get("doc", ""), "page": c.get("page_no", 0), "chunk_id": c.get("chunk_id", ""),  # 引用来源
             "score": round(float(c.get("fused", 0.0)), 4), "channel": c.get("channels", ""),           # 融合分与通道
             "snippet": (c.get("text") or "")[:180]} for c in cands]                    # 摘要片段
    doc = {"answer": ans, "references": refs, "cached": False, "degraded": deg,        # 响应主体
           "session_id": session_id, "retrieval_query": q_ret,                         # 会话与消解后的问句
           "contexts": [(c.get("text") or "") for c in cands]}                         # 供 RAGAS 评估使用
    if use_cache:                                                                      # 写缓存
        cache_set("answer", raw_key, {k: v for k, v in doc.items() if k != "latency_ms"})  # 缓存不含耗时
        cache_set("retrieval", raw_key, [{"chunk_id": c.get("chunk_id"), "fused": c.get("fused"),  # 检索结果单独缓存
                                          "channel": c.get("channels"), "rank": i + 1}  # 名次与通道
                                         for i, c in enumerate(cands)])                 # 逐条
    hist_add(session_id, "user", question)                                              # 记用户问
    hist_add(session_id, "assistant", _CTX.sub("", ans)[:C.SESSION_MAX_CHARS])           # 记助手答（剥离引用标记）
    return dict(doc, latency_ms=int((time.time() - t0) * 1000))                          # 回填耗时
