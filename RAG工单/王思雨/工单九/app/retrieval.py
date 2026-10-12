# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# 检索层 + 生成层：before=直调 08 原链路；after=工单九优化链路（2 跳图扩展 + 社区摘要 + 重标定融合/重排 + 引用生成 + Redis 缓存 + 降级）
import hashlib, json, time                                                      # 缓存键 / 序列化 / 计时
from . import config as C; from . import graph_rag as G                          # 本单配置 / 图谱层（图约束、图扩展、社区摘要）
import 工单八.app.rag_core as R8                                                # 复用 08 的向量/BM25/重排/客户端/Redis 原语，本单不重复实现
def _ck(q, tag):                                                                # 缓存键：结构指纹 + 全部影响结果的参数 + 问题；任一模块或索引改动即失效
    return "gd9:ans:" + R8._fp() + ":" + hashlib.sha256(f"{tag}|{q}|{C.HOPS}|{C.EXPAND_TERMS}|{C.VEC_W}|{C.BM25_W}|{C.FUSION_K}|{C.RERANK_N}|{C.USE_COMM}".encode()).hexdigest()[:32]  # 复用 08 结构指纹（源码哈希 + 索引 mtime）
def fuse_after(vh, bh):                                                         # 【优化·检索层】通道内 min-max 归一化后加权（直接加权会让区间不相交的弱权通道被架空）
    import numpy as np                                                          # 仅此处需数值运算，就近导入
    for hs in (vh, bh):                                                         # 两通道分别归一化
        s = np.array([h["score"] for h in hs], float) if hs else np.zeros(0)     # 取该通道分数
        lo, hi = (s.min(), s.max()) if len(s) else (0.0, 0.0)                    # 通道内极值
        for h, v in zip(hs, (s - lo) / (hi - lo) if hi > lo else np.ones_like(s)): h["nscore"] = float(v)  # 全同分给常数，避免除零
    pool = {}                                                                   # 以正文前缀为键合并两通道
    for h, w in [(x, C.VEC_W) for x in vh] + [(x, C.BM25_W) for x in bh]:        # 逐条带权重
        if (k := h["text"][:80]) in pool: pool[k]["score"] += w * h["nscore"]; pool[k]["src"] += "+" + h["src"]  # 双通道命中：加权分相加、来源合并
        else: pool[k] = dict(h, score=w * h["nscore"])                           # 首现：记加权分
    return sorted(pool.values(), key=lambda x: -x["score"])[: C.FUSION_K]        # 取融合后前 FUSION_K 条
def retrieve_after(q):                                                          # 工单九优化检索链路
    gc = G.context(q)                                                           # 图上下文：实体链接 + HOPS 跳邻域扩展 + 社区摘要 + 图约束代码
    qb = (q + " " + " ".join(gc["expand"])) if gc["expand"] else q               # 图扩展词注入关键词查询（只喂 BM25，向量语义已被原句占满）
    vh = R8.vector_search(q, k=C.C8.VEC_TOP_K, codes=gc["codes"]); bh = R8.bm25_search(qb, k=C.C8.BM25_TOP_K, codes=gc["codes"])  # 向量通道 + 关键词通道，**两通道都下推图约束**（只锁向量通道则"锁到目标公司"名存实亡）
    hits = R8.rerank(q, fuse_after(vh, bh), n=C.RERANK_N)                        # 融合 → bge-reranker-v2-m3 精排到 RERANK_N 条
    return hits + gc.get("summaries", []), {"vec": len(vh), "bm25": len(bh), "kept": len(hits) + len(gc.get("summaries", [])), "graph": gc}  # 追加社区摘要（global 通道）并回报各通道计数
def generate_after(q, hits, lang="auto"):                                       # 【优化·生成层】结论→证据→来源三段式，句末可核验引用，中英文自适应
    ctx = "\n\n".join(f"【片段{i+1}｜{h.get('company','社区摘要')}{h.get('year','')} 第{h.get('page',0)}页】\n{h['text']}" for i, h in enumerate(hits))  # 每段标明出处，便于逐句引用与人工核查
    sys_p = ("你是金融年报问答助手。规则：1) 只依据【检索片段】作答，绝不编造；2) 先给结论，再给证据；"        # 【优化·生成层】相比 08 增加分点与证据链要求
             "3) 每个结论句末用【来源：公司 第X页】标注；4) 片段不足时明确回答“年报中未提及”，不要猜测；"      # 防幻觉的两条硬约束
             "5) 用中文提问就用中文作答，用英文提问就用英文作答。")                                     # 中英文自适应（工单验收项）
    kw = {"reasoning_effort": C.C8.LLM_REASONING_EFFORT} if C.C8.LLM_REASONING_EFFORT else {}  # 推理型模型必须显式传 none，否则思考吃光 max_tokens 返回空
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": f"【检索片段】\n{ctx}\n\n【用户问题】\n{q}"}]  # 组装消息
    t0 = time.time()                                                            # 生成计时
    r = R8.llm().chat.completions.create(model=C.C8.LLM_MODEL, temperature=C.C8.LLM_TEMPERATURE, max_tokens=C.C8.LLM_MAX_TOKEN, messages=msgs, **kw)  # 温度 0 保证评测可复现，reasoning_effort 由 kw 按模型需要注入
    return (r.choices[0].message.content or "").strip(), round((time.time() - t0) * 1000)  # 返回答案与生成耗时（毫秒）
def answer_after(q, use_cache=True, lang="auto"):                               # 工单九完整问答链路（前端与评估都调它）
    t0 = time.time(); key = _ck(q, "after")                                     # 计时起点放在缓存查询之前（命中时才能报出真实耗时）+ 缓存键
    if use_cache:                                                               # 允许命中缓存
        try: v = R8.rds().get(key)                                              # 容错 E4：缓存故障不能影响主流程
        except Exception: v = None                                              # Redis 挂了当作未命中
        if v: return dict(json.loads(v), cached=True, latency_ms=round((time.time() - t0) * 1000))  # 命中：报亚毫秒级真实命中耗时
    hits, stat = retrieve_after(q)                                              # 图约束/扩展检索 + 重排
    if (time.time() - t0) > C.DEADLINE_S:                                       # 【优化·生成层】端到端超 3 秒预算
        ans, ms = "【降级】检索耗时超 3 秒预算，以下为检索原文（未经生成）：\n" + "\n\n".join(h["text"] for h in hits)[:1200], 0  # 降级：不再赌一次大模型调用，保证响应有上界
    else:                                                                       # 正常路径
        try: ans, ms = generate_after(q, hits, lang)                            # 生成
        except Exception as e: ans, ms = f"【降级】生成不可用（{type(e).__name__}），以下为检索原文：\n" + "\n\n".join(h["text"] for h in hits)[:1200], 0  # 容错 E3：超时/402 同一降级形态，HTTP 仍 200
    out = {"question": q, "answer": ans, "mode": "graph-opt", "lang": lang, "cached": False, "llm_ms": ms, "retrieval_stat": stat,  # 组装结果
           "graph": {"entities": stat["graph"]["entities"][:12], "codes": stat["graph"]["codes"], "expand": stat["graph"]["expand"], "n_communities_used": len(stat["graph"].get("summaries", []))}, "contexts": [{"text": h["text"], "company": h.get("company", ""), "page": h.get("page", 0), "src": h.get("src", ""), "rscore": round(h.get("rscore", 0), 4)} for h in hits]}  # 图信息供前端展示 / 上下文供 RAGAS 与人工核查
    out["latency_ms"] = round((time.time() - t0) * 1000)                        # 含生成的总耗时
    if use_cache:                                                               # 需要缓存才写
        try: R8.rds().setex(key, C.C8.CACHE_TTL, json.dumps(out, ensure_ascii=False))  # 带 TTL 写入
        except Exception: print("[cache] 写入失败，忽略")                        # 写失败只记不抛（容错 E4）
    return out                                                                  # 返回
def answer_before(q, use_cache=False):                                          # before 基线：直调 08 原 GraphRAG 链路（同语料同问题，只有链路不同）
    r = R8.answer(q, mode="graph", use_cache=use_cache)                         # 08 自带缓存键前缀 gd8:，与本单 gd9: 互不干扰
    r["mode"] = "graph-08-baseline"                                             # 标明来源，避免前端与报告混淆
    return r                                                                    # 原样返回，字段与 after 对齐（answer/contexts/latency_ms）
