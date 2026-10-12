# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本模块职责（检索与生成侧）：双路召回 → RRF 融合 → 重排 → 组装提示词 → LLM 生成 → 缓存。
# 上游（LLM 调用、缓存、Query 理解）见 app/llm.py；存储层见 app/store.py；解析分块见 app/rag.py。

import re                             # 近重复片段判定用的字符 n-gram 清洗
import time                           # 各阶段耗时统计
from concurrent.futures import ThreadPoolExecutor   # 线程池：让 LLM 改写与首轮召回并行

import numpy as np                    # 排序与向量运算
from . import config as C             # 全局配置
from . import rag                     # 向量化与分词
from . import store                   # Milvus 与 BM25 存储层
from . import llm as L                # 大模型调用、缓存、Query 理解


# ============================ 混合召回 ============================
def search_dense(query: str, top_k: int = None, extra_queries: list[str] = None) -> list[dict]:  # 向量召回
    """用 bge-m3 编码查询（含子问题）在 Milvus 做 HNSW 检索，合并去重后返回。"""
    client = store.get_milvus()                                        # Milvus 客户端
    if not client.has_collection(C.MILVUS_COLLECTION):                 # 尚未建索引
        return []                                                      # 返回空，由上层提示先建库
    queries = [query] + [q for q in (extra_queries or []) if q and q != query]  # 主问题 + 子问题
    vecs = rag.embed_texts(queries)                                    # 批量编码，一次 GPU 前向
    best = {}                                                          # pk → 命中的最好分数
    for v in vecs:                                                     # 逐个查询向量检索
        hits = client.search(collection_name=C.MILVUS_COLLECTION, data=[v.tolist()],
                             limit=top_k or C.RETRIEVE_TOP_K, consistency_level="Strong",
                             output_fields=["content", "doc_id", "doc_name", "chunk_index",
                                            "page_no", "page_end", "chunk_type"])[0]  # 单查询结果
        for h in hits:                                                 # 遍历命中
            pk = h["id"]                                               # 主键
            if pk not in best or h["distance"] > best[pk]["score"]:    # 保留最高分
                best[pk] = {"pk": pk, "score": float(h["distance"]),   # COSINE 距离即相似度
                            **{k: h["entity"].get(k) for k in ("content", "doc_id", "doc_name",
                               "chunk_index", "page_no", "page_end", "chunk_type")}}  # 携带标量字段
    return sorted(best.values(), key=lambda x: -x["score"])            # 按相似度降序返回


def search_bm25(query: str, top_k: int = None) -> list[dict]:          # 关键词召回
    """用 jieba 分词后走 rank_bm25 打分，返回 top_k 命中。"""
    bm25, corpus = store.get_bm25()                                    # 取 BM25 索引与语料
    if bm25 is None:                                                   # 语料为空
        return []                                                      # 无结果
    scores = bm25.get_scores(rag.tokenize(query))                      # 计算全语料 BM25 分数
    idx = np.argsort(scores)[::-1][:top_k or C.RETRIEVE_TOP_K]         # 取分数最高的若干条
    out = []                                                           # 结果列表
    for i in idx:                                                      # 遍历候选下标
        if scores[i] <= 0:                                             # 分数为 0 表示无词命中
            continue                                                   # 跳过
        c = corpus[int(i)]                                             # 对应语料
        out.append({"pk": f'{c["doc_id"]}-{c["chunk_index"]}', "score": float(scores[i]),  # 伪主键
                    "content": c["content"], "doc_id": c["doc_id"], "doc_name": c["doc_name"],
                    "chunk_index": c["chunk_index"], "page_no": c["page_no"],
                    "page_end": c["page_end"], "chunk_type": c["type"]})  # 与向量结果同构
    return out                                                         # 返回关键词召回结果


def rrf_fuse(result_lists: list[list[dict]], k: int = 60, top_n: int = None) -> list[dict]:  # RRF 融合
    """Reciprocal Rank Fusion：按名次倒数加权融合多路召回，避免不同量纲分数直接相加。"""
    merged = {}                                                        # pk → 融合结果
    for results in result_lists:                                       # 遍历每一路召回
        for rank, item in enumerate(results):                          # 名次从 0 开始
            key = f'{item["doc_id"]}-{item["chunk_index"]}'            # 用文档+分块序号做稳定标识
            if key not in merged:                                      # 首次出现
                merged[key] = {**item, "rrf": 0.0, "sources": []}      # 初始化融合项
            merged[key]["rrf"] += 1.0 / (k + rank + 1)                 # 累加 RRF 分值
            merged[key]["sources"].append(item.get("source", ""))      # 记录来源通道
            merged[key]["score"] = max(merged[key]["score"], item["score"])  # 保留原始最高分
    out = sorted(merged.values(), key=lambda x: -x["rrf"])             # 按 RRF 分值降序
    return out[:top_n or C.FUSION_TOP_N]                               # 截断到进入重排的数量


# ============================ 重排 ============================
_RERANKER = None                      # 重排模型单例
_RERANK_TOK = None                    # 重排分词器单例


def _get_reranker():                                                   # 重排模型单例
    """加载 bge-reranker-v2-m3（FP16 + GPU），失败返回 (None, None) 由上层跳过重排。"""
    global _RERANKER, _RERANK_TOK                                      # 模块级单例
    if _RERANKER is None and _RERANK_TOK is None:                      # 尚未尝试加载
        try:                                                           # 加载可能因显存/路径失败
            import torch                                               # 延迟导入
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 延迟导入
            _RERANK_TOK = AutoTokenizer.from_pretrained(C.RERANK_MODEL_PATH)  # 本地分词器
            model = AutoModelForSequenceClassification.from_pretrained(  # 本地模型
                C.RERANK_MODEL_PATH, torch_dtype=torch.float16 if C.EMBED_DEVICE == "cuda" else torch.float32)
            _RERANKER = model.eval().to(C.EMBED_DEVICE)                # 切评估模式并上设备
        except Exception as exc:                                       # 加载失败
            print(f"[重排降级] reranker 加载失败，跳过重排：{type(exc).__name__}: {exc}")  # 记录
            _RERANKER, _RERANK_TOK = None, None                        # 保持 None，上层跳过
    return _RERANKER, _RERANK_TOK                                      # 返回模型与分词器


def rerank(question: str, candidates: list[dict], top_k: int = None) -> list[dict]:  # 精排
    """用交叉编码器对候选片段打分并按阈值过滤；模型不可用时原样返回前 top_k 条。"""
    top_k = top_k or C.RERANK_TOP_K                                    # 默认取 5 条
    if not candidates:                                                 # 无候选
        return []                                                      # 直接返回
    model, tok = _get_reranker()                                       # 取模型
    if model is None:                                                  # 降级：跳过重排
        return candidates[:top_k]                                      # 直接用融合结果
    import torch                                                       # 延迟导入
    pairs = [[question, c["content"][:1200]] for c in candidates]      # 问题-片段对
    # 性能要点（各配置重复 5 次取中位数）：14 对候选一次前向约 0.67s。
    # 缩池子到 10 对、或按长度分桶（3/5/14），耗时都在 0.667~0.701s 噪声内 —— 均为无效优化，故不引入。
    # max_length 必须保持 512：降到 384 会截掉分块尾部的关键数字（如"军用领域收入 6,464.51 万元…"恰在段尾），
    # 实测关键事实命中率从 100% 掉到 80%。真正的取舍是"要不要重排"本身（省 0.67s），见《优化记录》§10.4。
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))   # 按片段长度升序，便于批内长度接近
    scores = [0.0] * len(pairs)                                        # 分数数组（按原始下标）
    with torch.no_grad():                                              # 关闭梯度，省显存
        inputs = tok([pairs[j] for j in order], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to(C.EMBED_DEVICE)   # 按长度序一次编码
        logits = model(**inputs).logits.view(-1).float()               # 取 logits
        raw = torch.sigmoid(logits).cpu().numpy().tolist()             # 转 0~1 相关度分数
    for k, j in enumerate(order):                                      # 桶内位置 → 原始下标
        scores[j] = raw[k]                                             # 回填分数
    for c, s in zip(candidates, scores):                               # 回填分数
        c["rerank_score"] = float(s)                                   # 记录重排分
    ranked = sorted(candidates, key=lambda x: -x["rerank_score"])      # 按相关度降序
    kept = [c for c in ranked if c["rerank_score"] >= C.RERANK_MIN_SCORE]  # 阈值过滤
    return (kept or ranked)[:top_k]                                    # 全被过滤时兜底返回最高分几条


# ============================ 检索总入口与生成 ============================
def _recall(query: str, extra: list[str] | None, top_k: int | None):   # 一趟双路召回
    """对给定查询同时跑向量与 BM25 召回，返回 (dense, sparse)。"""
    return search_dense(query, top_k, extra), search_bm25(query, top_k)   # 两路并返回


def _merge_hits(a: list[dict], b: list[dict]) -> list[dict]:           # 合并两轮召回结果
    """按「文档+分块序号」去重，同一条保留较高分数，避免重复片段挤占重排名额。"""
    best = {}                                                          # 稳定标识 → 命中最优项
    for item in a + b:                                                 # 遍历两轮结果
        key = f'{item["doc_id"]}-{item["chunk_index"]}'                # 稳定标识
        if key not in best or item["score"] > best[key]["score"]:      # 首次出现或分数更高
            best[key] = item                                           # 保留更优项
    return sorted(best.values(), key=lambda x: -x["score"])            # 按分数降序


def _shingles(text: str, n: int = 4) -> set:                           # 取字符 n-gram 集合
    t = re.sub(r"\s+", "", text)                                       # 去掉空白，避免排版差异干扰
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}          # 返回 n-gram 集合


def dedup_near(after_rerank: list[dict], thresh: float = 0.75) -> list[dict]:   # 近重复片段去重
    """按字符 4-gram 的 Jaccard 相似度去掉近重复片段，保留排序靠前的那条。

    动机：招股说明书会把同一段内容在"重大事项提示""风险因素""业务与技术"等章节反复陈述，
    实测中同一个问题召回的前 6 名里常有两三条是这种重复段落，白占上下文预算与重排名额，
    把真正含答案的表格/正文挤到第 7 名之外。去重后既提精度又不增加上下文长度。
    """
    kept = []                                                          # 已保留的片段
    for item in after_rerank:                                          # 按重排分数从高到低遍历
        s = _shingles(item["content"])                                 # 当前片段的 n-gram
        dup = False                                                    # 是否与已保留项重复
        for k in kept:                                                 # 与已保留片段逐一比较
            kk = k["_sh"]                                              # 复用已算好的 n-gram 集合
            inter = len(s & kk)                                        # 交集
            if inter and inter / len(s | kk) >= thresh:                 # Jaccard 超阈值即判为重复
                dup = True                                             # 标记重复
                break                                                  # 跳出内层循环
        if not dup:                                                    # 不是重复
            item["_sh"] = s                                            # 缓存 n-gram，供后续比较
            kept.append(item)                                          # 保留
    for k in kept:                                                     # 清理临时字段
        k.pop("_sh", None)                                             # 不污染返回结构
    return kept                                                        # 返回去重后的片段


def retrieve(question: str, top_k: int = None, use_rewrite: str = "auto",
             use_rerank: bool = True) -> dict:                          # 检索总入口
    """串联 Query 理解 → 双路召回 → RRF → 重排，返回上下文与各阶段耗时。

    关键优化：**LLM 改写与首轮召回并行**。先用原始问题立刻召回（几十毫秒），
    同时等 LLM 改写；改写结果与原问题不同则再补一轮并按分数合并。
    这样 LLM 改写只会增加召回，永远不会让结果变差，也不把改写延迟串到检索前面。
    """
    t0 = time.time()                                                   # 计时起点
    with ThreadPoolExecutor(max_workers=1) as ex:                      # 起一个线程跑首轮召回
        fut = ex.submit(_recall, question, None, top_k)                # 先用原问题召回
        qu = L.understand_query(question, use_rewrite)                 # 同时做 Query 理解
        dense, sparse = fut.result()                                   # 取回首轮召回结果
    t1 = time.time()                                                   # Query 理解 + 首轮召回耗时
    if qu.get("rewritten") and (qu["rewritten"] != question or qu.get("sub_questions")):
        d2, s2 = _recall(qu["rewritten"], qu.get("sub_questions"), top_k)   # 用改写式补一轮
        dense, sparse = _merge_hits(dense, d2), _merge_hits(sparse, s2)     # 合并去重
    for d in dense:                                                    # 标注来源通道
        d["source"] = "dense"                                          # 向量
    for s in sparse:                                                   # 标注来源通道
        s["source"] = "bm25"                                           # 关键词
    t2 = time.time()                                                   # 召回结束时刻
    fused = rrf_fuse([dense, sparse])                                  # RRF 融合
    if use_rerank:                                                     # 启用重排
        pool = rerank(qu["rewritten"], fused, top_k=C.RERANK_POOL)     # 先重排出一个更大的候选池
        ctx = dedup_near(pool)[:C.RERANK_TOP_K]                        # 去近重复后再截断到最终条数
    else:                                                              # 未启用重排
        ctx = dedup_near(fused)[:C.RERANK_TOP_K]                       # 直接用融合结果并去重
    t3 = time.time()                                                   # 重排耗时
    total, contexts = 0, []                                            # 上下文累计长度与被选片段
    for c in ctx:                                                      # 逐条累加，控制总长
        if total + len(c["content"]) > C.CONTEXT_MAX_CHARS:            # 超过上限
            break                                                      # 停止累加
        total += len(c["content"])                                     # 累计
        contexts.append(c)                                             # 收集
    return {"query_understanding": qu, "contexts": contexts,          # 返回结果
            "n_dense": len(dense), "n_sparse": len(sparse), "n_fused": len(fused),  # 各阶段数量
            "timing": {"understand": round(t1 - t0, 3), "recall": round(t2 - t1, 3),  # 耗时明细
                       "rerank": round(t3 - t2, 3), "retrieve": round(t3 - t0, 3)}}


def format_contexts(contexts: list[dict]) -> list[str]:                 # 给片段加出处标注
    """返回「【资料i｜第N页｜类型】正文」形式的片段文本。

    与 build_prompt 使用同一套标注，供 RAGAS 评估复用——评测必须喂给判官**模型实际看到的**
    同一份上下文，否则模型回答里的页码引用会被判成"无依据"，faithfulness 被系统性低估
    （实测：直接传裸文本，"法定代表人是程家明（依据：第51页）"只得 0.5；传带标注文本得 1.0）。
    """
    return [f'【资料{i}｜第{c["page_no"]}页｜{c["chunk_type"]}】\n{c["content"]}'
            for i, c in enumerate(contexts, 1)]                        # 逐条编号并标注出处


def build_prompt(question: str, contexts: list[dict]) -> list[dict]:    # 组装提示词
    """把检索片段连同页码标注拼进上下文；要求只依据资料作答并说明资料不足。"""
    if contexts:                                                       # 有检索结果
        ctx = "\n\n".join(format_contexts(contexts))                    # 拼接带出处标注的上下文
        sys = ("你是一个严谨的文档问答助手。只能依据下方【资料】回答问题，禁止编造。\n"
               "作答要求：\n"
               "1) 通读全部资料后再作答，不要只看了前几条就下结论；\n"
               "2) 资料中的数字（金额、比例、年份）必须原样完整列出，涉及多个期间的不得遗漏；\n"
               "3) 只有在通读全部资料后仍确实找不到，才回答“资料中未找到”；\n"
               "4) 回答准确、简洁，并注明依据的页码。\n"
               f"用与提问相同的语言作答。\n\n===资料开始===\n{ctx}\n===资料结束===")  # 系统提示
    else:                                                              # 没有检索结果
        sys = "你是一个严谨的文档问答助手。当前没有检索到相关资料，请说明无法回答。"  # 明确拒答
    return [{"role": "system", "content": sys}, {"role": "user", "content": question}]  # 返回消息列表


def answer(question: str, top_k: int = None, use_rewrite: str = "auto",
           use_rerank: bool = True, use_cache: bool = True) -> dict:   # 问答主入口（非流式）
    """完整问答链路，返回答案、引用片段、是否命中缓存与各阶段耗时。"""
    t0 = time.time()                                                   # 计时起点
    key = L.cache_key("ans", f"{question}|{top_k}|{use_rewrite}|{use_rerank}")  # 缓存键
    if use_cache:                                                      # 允许用缓存
        hit = L.cache_get(key)                                         # 查缓存
        if hit:                                                        # 命中
            hit["cached"] = True                                       # 标记命中
            hit["latency"] = {"total": round(time.time() - t0, 3)}     # 记录极短耗时
            return hit                                                 # 直接返回
    ret = retrieve(question, top_k, use_rewrite, use_rerank)           # 执行检索
    t1 = time.time()                                                   # 检索完成时刻
    try:                                                               # 生成阶段单独容错
        text = L.llm_chat(build_prompt(question, ret["contexts"]), max_tokens=450,
                          temperature=0.0)  # 事实型问答用 0 温度：消除随机性，避免漏列数字
        degraded = False                                               # 未降级
    except Exception as exc:                                           # LLM 不可用
        text = _fallback_answer(question, ret["contexts"], exc)        # 用检索原文拼兜底答案
        degraded = True                                                # 标记降级
    latency = {**ret["timing"], "generate": round(time.time() - t1, 3),  # 生成耗时
               "total": round(time.time() - t0, 3)}                    # 端到端总耗时
    result = {"question": question, "answer": text,                    # 组装结果
              "contexts": [_slim(c) for c in ret["contexts"]],         # 引用片段（精简字段）
              "query_understanding": ret["query_understanding"],       # Query 理解结果
              "cached": False, "degraded": degraded,                   # 缓存/降级标记
              "stats": {k: ret[k] for k in ("n_dense", "n_sparse", "n_fused")},  # 召回统计
              "latency": latency}                                      # 耗时
    if use_cache and not degraded:                                     # 只缓存正常结果
        L.cache_set(key, {**result, "cached": False}, ttl=1800)        # 缓存 30 分钟
    return result                                                      # 返回


def _fallback_answer(question: str, contexts: list[dict], exc: Exception) -> str:  # 生成兜底
    """DeepSeek 不可用时，直接摘录检索到的原文并明确标注，不编造内容。"""
    if not contexts:                                                   # 连资料都没有
        return f"生成服务暂不可用（{type(exc).__name__}），且未检索到相关资料。"  # 如实说明
    head = "\n\n".join(f'【第{c["page_no"]}页·{c["chunk_type"]}】{c["content"][:300]}' for c in contexts[:3])  # 摘录
    return f"（生成服务暂不可用：{type(exc).__name__}，以下为检索到的原文摘录）\n{head}"  # 返回摘录


def slim(c: dict) -> dict:                                             # 精简引用片段字段
    """只保留前端展示与反馈需要的字段，避免响应体过大。"""
    return {k: c.get(k) for k in ("content", "page_no", "page_end", "chunk_type", "doc_name")} | \
           {"score": round(float(c.get("rerank_score", c.get("score", 0))), 4),
            "sources": c.get("sources", [])}                           # 附上相似度与来源通道


_slim = slim                                                           # 兼容内部旧调用名


def answer_llm_only(question: str) -> dict:                            # 纯 LLM 对照组
    """不传任何检索上下文，直接问 DeepSeek，用于工单要求的 RAG vs 纯 LLM 对比。"""
    t0 = time.time()                                                   # 计时
    try:                                                               # 容错
        text = L.llm_chat([{"role": "system", "content": "回答用户问题，简洁准确。"},
                           {"role": "user", "content": question}],
                          max_tokens=450, temperature=0.0)  # 无上下文；与 RAG 路径同参数，保证对比公平
    except Exception as exc:                                           # 调用失败
        text = f"（纯 LLM 调用失败：{type(exc).__name__}）"             # 如实报告
    return {"question": question, "answer": text, "contexts": [],      # 无引用
            "latency": {"total": round(time.time() - t0, 3)}}          # 耗时
