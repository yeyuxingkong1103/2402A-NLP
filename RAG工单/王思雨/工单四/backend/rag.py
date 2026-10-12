# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：问答流水线。混合检索（向量+BM25 经 RRF 融合）→ bge-reranker-v2-m3 精排 → DeepSeek 生成，
#           配套 Redis 缓存、检索精确度计算与依赖故障降级。索引构建与入库在 scripts/ingest.py。
import json, pickle, re, time                                               # 序列化、索引持久化、语言判定、计时
from backend import cache as K                                              # Redis 缓存
from backend import chunking as G                                           # 语义分块
from backend import store as V                                              # 向量库与重排
import config as C                                                          # 集中配置

STATE = {"bm25": None, "chunks": [], "client": None, "stats": {}, "tried": 0.0}  # 进程内索引状态，常驻内存避免每次重载
REL_THRESHOLD = 0.5                                                         # ★检索精确度口径：重排分 ≥ 该值即视为相关块


def detect_lang(question: str) -> str:                                      # 语言判定
    """中文字符占比超 15% 视为中文提问，否则英文；供多语言问答（工单验收项）使用。"""
    q = question or ""                                                      # 防空
    return "zh" if len(re.findall(r"[一-鿿]", q)) >= max(1, len(q) * 0.15) else "en"  # 超阈值判中文


def tokenize(text: str) -> list:                                            # BM25 分词器
    """中文 jieba 切分，英文与数字整体保留——保证 1,670、25.04%、-2.0% 这类串能精确命中。"""
    import jieba                                                            # 延迟导入
    return [t.lower() for t in jieba.lcut(text or "") if t.strip()]         # 切分后统一小写


def build_bm25(chunks: list):                                               # 构建 BM25 索引
    """BM25Okapi 对数字、比例、专有名词敏感，正好补向量检索对精确串不敏感的短板。"""
    from rank_bm25 import BM25Okapi                                        # 延迟导入
    return BM25Okapi([tokenize(c["text"]) for c in chunks])                 # 逐块分词建索引


def bm25_search(bm25, chunks: list, question: str, top_k: int) -> list:      # 关键词检索
    """按 BM25 得分降序取 top_k，只保留得分大于 0 的块。"""
    import numpy as np                                                      # 延迟导入
    if bm25 is None: return []                                              # 索引未就绪
    s = np.asarray(bm25.get_scores(tokenize(question)))                     # 全部块得分
    order = np.argsort(s)[::-1][:top_k]                                     # 降序取前 K
    return [dict(chunks[i], bm25_score=float(s[i])) for i in order if s[i] > 0]  # 非零分块


_LATIN = re.compile(r"[a-zA-Z]{2,}")                                        # 拉丁字母串：水印被 OCR 打碎后残留的形态


def noisy(text: str) -> bool:                                               # ★水印噪声判定
    """判断块是否被 PDF 斜向水印污染。

    招股说明书的斜向水印不在文本层，MinerU 读图时必然读进单元格，OCR 会把水印切成
    `hzong`、`inen2017年度7`、`ons:` 这类字母碎片混在中文数字之间。实测干净的正文与表格块
    拉丁字母串数为 0，而被污染的表格行有 2~3 个。被污染的表格数字与年份的对应关系会错位——
    "军用领域收入"一题就因此从这张表里读出了错误的年度金额，必须压下去。
    """
    if any(w.lower() in text.lower() for w in C.WATERMARK_TOKENS): return True   # 直接命中水印词
    return len(_LATIN.findall(text)) >= 2                                   # 多个字母碎片 = OCR 残渣


_FRAG = re.compile(r"(?<=[一-鿿])([a-z]{2,8})(?=[一-鿿]|[\W_]|$)")            # 紧贴中文的小写拉丁残渣


def clean(text: str) -> str:                                                # ★水印残渣清洗
    """删掉紧贴中文的全小写拉丁碎片——它们是斜向水印"人工智能刘敏"的拼音被 OCR 打碎后的残渣。

    实测：不清理时 id=3 的答案会写成"赵马克gzh"（真名 赵马克，全库干净出现 66 次、带 gzh 仅 2 次），
    实体名被污染，而关键词判分照样给过——正是判分器抓不到的那类错。全库 970 个块共 1336 处。
    判据用"全小写且紧贴中文"：正规术语是 IC / ERP / GDP / BOM 这类全大写，实测一个都没被误删；
    已知误伤仅 logo 与 min 两词，在中文招股书里不承载信息，取舍划算。
    """
    return _FRAG.sub("", text or "")                                        # 删碎片，保留其余原文


def rrf(vec_hits: list, bm25_hits: list) -> list:                           # ★RRF 融合
    """Reciprocal Rank Fusion：score = Σ 1/(k+rank)。

    比"归一化后加权"更稳：各路分数量纲不同（向量内积 vs BM25 无界分），归一化会被单路极值带偏；
    RRF 只用排名，权重天然均衡、无需调参。实测在本题集上 RRF 的召回质量优于加权融合（见 docs/optimization.md）。
    """
    merged = {}                                                             # chunk_id → 候选块
    for key, hits in (("vec_rank", vec_hits), ("bm25_rank", bm25_hits)):     # 两路
        for rank, h in enumerate(hits):                                     # 逐条按排名计分
            slot = merged.setdefault(h["chunk_id"], dict(h, score=0.0))     # 首次命中登记，保留定位字段
            slot["score"] += 1.0 / (C.RRF_K + rank + 1)                     # RRF 累加
            slot[key] = rank + 1                                            # 记下各路排名，便于排障
    # 图像语义块权重加成：图块数量远少于正文块，但信息密度高（一张组织结构图含 30 个实体），
    # 不加成时会被大量正文块用排名压下去，图文类问题就永远挤不进重排候选。
    for s in merged.values():                                               # 逐条
        if s.get("block_type") == "image": s["score"] *= C.IMAGE_BOOST      # 图像块加成
        if noisy(s.get("text", "")): s["score"] *= C.NOISE_PENALTY          # ★水印污染块降权：其数字与年份的对应关系不可信
    return sorted(merged.values(), key=lambda d: d["score"], reverse=True)[:C.FUSION_TOP_K]  # 降序取待重排候选


def hybrid_search(client, bm25, chunks: list, question: str) -> tuple:       # 混合检索总入口
    """向量与 BM25 两路并行召回再 RRF 融合；同时回报**实际生效的通道**，供上层标注降级。"""
    from concurrent.futures import ThreadPoolExecutor                       # 线程池：编码走 GPU、BM25 走 CPU，可真正并行
    with ThreadPoolExecutor(max_workers=2) as pool:                         # 两路同时开跑
        fv = pool.submit(V.vector_search, client, question, C.VEC_TOP_K)     # 向量通道
        fb = pool.submit(bm25_search, bm25, chunks, question, C.BM25_TOP_K)  # 关键词通道
        vec, bm = fv.result(), fb.result()                                  # 等两路完成
    if not vec and not bm: raise RuntimeError("检索服务暂时不可用，请稍后重试")  # 两路皆空说明依赖不可用
    # 少了一路就是降级：不能静默返回低质量结果，否则验收方会以为降级与正常等价
    mode = "hybrid" if (vec and bm) else ("bm25_only" if bm else "vector_only")  # 实际生效的通道
    return rrf(vec, bm), mode                                               # 融合候选与检索模式


def load() -> None:                                                         # 懒加载索引
    """首次提问时载入块列表、BM25 索引并连接 Milvus，之后常驻内存。"""
    # 坑：Milvus 不可用时 MilvusClient() 在**构造阶段**就抛异常；不加保护，异常会从 load 直接冒到最外层，
    # 检索内部的兜底分支根本走不到，结果是整个问答失败——而不是退回 BM25 继续作答。
    if STATE["client"] is None and time.time() - STATE["tried"] > 60:        # 每 60 秒最多重连一次，避免每次提问都白等超时
        STATE["tried"] = time.time()                                        # 记下尝试时间
        try: STATE["client"] = V.milvus_client()                            # 尝试连接
        except Exception as e: print(f"[检索降级] Milvus 不可用（{type(e).__name__}），改用 BM25 关键词检索")  # 服务继续可用
    if STATE["chunks"]: return                                              # 已载入
    # 索引文件损坏要给出可操作提示：实测 bm25.pkl 被写坏时，用户完全无从知道该重建索引
    try:                                                                    # 读落盘的块与 BM25
        d = json.loads(C.INDEX_FILE.read_text(encoding="utf-8"))            # 块列表
        STATE["chunks"], STATE["stats"] = d["chunks"], d.get("stats") or {}  # 恢复块与统计
        STATE["bm25"] = pickle.load(open(C.BM25_FILE, "rb"))                # 反序列化 BM25
    except FileNotFoundError: raise                                         # 交给上层报"尚未构建"
    except Exception as e:                                                  # 内容损坏（JSON 截断、pickle 损坏等）
        raise RuntimeError(f"索引文件已损坏（{type(e).__name__}），请执行 python scripts/ingest.py 重建索引")


def _client():                                                              # DeepSeek 客户端
    """OpenAI 兼容接口；超时受限以保证快速失败，而不是长时间挂住接口。"""
    from openai import OpenAI                                               # 延迟导入
    if not C.DEEPSEEK_API_KEY: raise RuntimeError("未配置 DEEPSEEK_API_KEY，请参考 .env.example 配置后重试")  # 缺密钥明确提示
    return OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL, timeout=C.LLM_TIMEOUT, max_retries=1)  # 受限超时


def prompt(question: str, contexts: list, lang: str) -> str:                # 构造提示词
    """严格依据检索资料作答、用与提问相同的语言（多语言验收项）；资料无据时明确拒答以抑制编造。"""
    def lab(i, c):                                                          # 给每条资料打出处标签
        """标签写明页码与图像类型，让模型能分辨"范围相近但不同"的表/图，避免跨表合并条目。"""
        src = c.get("locator") or f"{c.get('doc', '')} P{c.get('page_no')}"  # 可读出处
        tag = f"（{c['image_type']}）" if c.get("image_type") else ""         # 图像块标注语义类型
        return f"[资料{i + 1}｜来源：{src}{tag}]\n{c['text']}"                # 标签 + 片段全文
    body = "\n\n".join(lab(i, c) for i, c in enumerate(contexts))           # 拼接资料
    asked = "用中文作答" if lang == "zh" else "Answer in English"             # 语言约束
    return ("你是招股说明书问答助手，请严格依据下列资料回答，不要编造。\n"          # 角色与首要约束
            f"要求：1) {asked}；2) 数字、比例、名称必须与资料完全一致，不要四舍五入或改写；"  # 准确性
            "3) 作答要完整，涉及企业名称、项目名称、奖项、标准时给出资料中的完整名称；"       # 完整性
            "4) 每份资料都标了来源；若多份资料内容相近但范围不同，只采用与问题问法最匹配的一份；"  # 防跨表合并
            "5) 标为【图像语义】【图内层级】的资料来自文档中的插图，可直接作为依据；"          # ★图像块可作答
            # 实测坑：问"在哪个领域已经成为重要供应商"，模型只回了"国防军队视频指挥领域。"——
            # 把问题里的"在哪个领域"回声式地截断成短语，丢掉了"已经成为重要供应商"这个论断。
            "6) 答案要写成完整通顺的句子，不要只回一个短语片段；"                            # 完整性下限
            "7) 资料中找不到答案时回复\"资料中未找到相关信息\"。\n\n"                      # 拒答策略
            f"{body}\n\n问题：{question}\n答案：")                                # 注入资料与问题


def generate(question: str, contexts: list, lang: str) -> str:              # 生成答案
    """温度 0 保证可复现；推理型模型必须显式传 reasoning_effort，否则思考过程吃光 max_tokens 返回空内容。"""
    args = dict(model=C.LLM_MODEL, temperature=0, max_tokens=800,           # 模型与采样参数
                messages=[{"role": "user", "content": prompt(question, contexts, lang)}])  # 单轮提示词
    if C.LLM_REASONING_EFFORT: args["reasoning_effort"] = C.LLM_REASONING_EFFORT  # 推理型模型开关
    return (_client().chat.completions.create(**args).choices[0].message.content or "").strip()  # 取答案文本


def precision(contexts: list) -> float:                                     # ★检索精确度
    """口径：返回的 N 条上下文中，重排分 ≥ REL_THRESHOLD 的相关块所占比例（0~1）。

    这是**无标注在线口径**——线上没有标准答案可比。带标注的口径（命中出处页/关键词）由
    scripts/test_questions.py 离线计算，两者都写进 README，避免把在线代理指标说成准确率。
    """
    if not contexts: return 0.0                                             # 空则 0
    ok = sum(1 for c in contexts if float(c.get("score", 0.0)) >= REL_THRESHOLD)  # 相关块计数
    return round(ok / len(contexts), 3)                                     # 占比


def expand_table_rows(hits: list, chunks: list) -> list:                     # ★表格行补全
    """命中某张表的任一行时，把该表其余行一并带进上下文。

    为什么必须做：行级块让检索打得准，但**答不全**。实测"不存在控制关系的关联方企业有哪些"——
    7 家企业各占一行，重排 Top-8 里只进得来 4 行，答案就少写 3 家；而把 Top-N 调到 12 又会带进
    隔壁"曾为关联方"表的无关企业，触发另一种错答。按 table_id 取回同表全部行，两个毛病一起解决。
    """
    by_id = {}                                                              # table_id → 该表全部块（按文档顺序）
    for c in chunks:                                                        # 遍历全部块
        if c.get("table_id"): by_id.setdefault(c["table_id"], []).append(c)   # 只收表格行
    # 第一段：重排命中的块**原样排在前面**，保持重排给出的优先级。
    # 曾经把补全行插在触发行后面，结果"军用领域收入"题被一张 10 行的竞争表灌满上下文，
    # 模型改用了那张表的数字（4,508.18），而正确答案（正文的 6,464.51）被挤到一边——补全不能反客为主。
    out = [h for h in hits]                                                 # 保住精排结果
    seen = {h["chunk_id"] for h in hits}                                    # 去重集合
    # 第二段：把同表其余行**追加**到末尾；只补小表。
    # 大表（行数超过补全上限）整表跳过：这类表多是"主营业务收入按行业列示"之类的全量数据表，
    # 补进来只会挤占高分块的位置，而用户问的通常是表里的某一行。
    # 只让排进前 N 名的行触发补全：实测排序靠后的行往往只是"擦边命中"，
    # 把它所属的整张表拖进来会灌入大量无关行（见 config.py 的 TABLE_EXPAND_TRIGGER_RANK 说明）。
    for h in hits[:C.TABLE_EXPAND_TRIGGER_RANK]:                            # 按重排顺序，仅取前 N 名
        tid = h.get("table_id")                                             # 所属表
        rows = by_id.get(tid) if tid else None                              # 同表全部行
        if not rows or len(rows) > C.TABLE_EXPAND_MAX: continue              # 非表格行或大表 → 不补
        for r in sorted(rows, key=lambda c: c.get("row_index", 0)):          # 按行号还原表内顺序
            if r["chunk_id"] in seen: continue                              # 已收过则跳过
            seen.add(r["chunk_id"])                                         # 登记
            out.append(dict(r, rerank_score=float(h.get("rerank_score", 0.0)),  # 沿用触发行分数供前端展示
                            _expanded=True))                                # 标记为补全行
    return out[:C.TABLE_EXPAND_MAX + C.RERANK_TOP_N]                        # 限制总条数，防止撑爆提示词


def merge_table_rows(hits: list) -> list:                                   # ★表级呈现
    """把同一张表的命中行合并成一条上下文：**行级检索，表级呈现**。

    表是以「行」为单位入库的——行级块让检索打得准（问"注册资本"就精确命中那一行）。
    但喂给大模型时不该是碎片：实测 id=4（问"不存在控制关系的关联方企业有哪些"）返回 12 条上下文，
    同一张表的多行各自成条、彼此不连贯，裁判逐条判相关性时大量判为不相关，上下文精确度只有 0.0~0.33；
    模型也更难从碎片里数清到底有几家。
    故检索仍在行上做，**呈现前把同表行拼回一张表**。
    """
    groups, pos, singles = {}, {}, []                                        # 表分组、出现位置、非表格块
    for i, h in enumerate(hits):                                             # 逐条
        tid = h.get("table_id")                                              # 所属表
        if not tid: singles.append((i, h)); continue                          # 非表格块原样保留
        if tid not in groups: groups[tid], pos[tid] = [], i                   # 首次出现记下位置
        groups[tid].append(h)                                                # 归组
    out = {}                                                                # 位置 → 块
    for tid, rows in groups.items():                                         # 逐表
        if len(rows) == 1: out[pos[tid]] = rows[0]; continue                   # 单行不必合并
        rows = sorted(rows, key=lambda c: c.get("row_index", 0))              # 按行号还原表内顺序
        top = max(rows, key=lambda c: float(c.get("rerank_score", c.get("score", 0.0))))  # 以得分最高那行为准，保住它在上下文里的优先级
        out[pos[tid]] = dict(top, _merged=len(rows),                          # 合并块
                             text="\n".join(f"第{r.get('row_index', 0)}行 {r.get('text', '')}" for r in rows))  # 拼成整表
    for i, h in singles: out[i] = h                                          # 非表格块
    return [out[i] for i in sorted(out)]                                     # 按原优先级顺序返回


def _ctx(hits: list) -> list:                                               # 检索结果 → 接口输出
    """把块转为前端需要的结构：片段全文、出处、图像类型、逐条相关度。"""
    return [dict(doc=h.get("doc", ""), page_no=h.get("page_no"),           # 文档与页码
                 text=clean(h.get("text", "")), block_type=h.get("block_type", "text"),  # ★片段先清水印残渣，喂给大模型与前端都干净
                 image_type=h.get("image_type", ""), image_path=h.get("image_path", ""),  # ★图像元数据，前端回显
                 locator=h.get("locator", ""),                               # 可读定位串
                 score=round(float(h.get("rerank_score", h.get("score", 0.0))), 4))  # 相关度
            for h in hits]                                                  # 逐条转换


def _reason(mode: str, llm_err: str) -> str:                                # 汇总降级说明
    """少一路检索或生成不可用都会实打实降低答案质量，必须显式标注而不是静默返回。"""
    r = {"bm25_only": "向量检索不可用，已降级为 BM25 关键词检索，实体名相近的问题准确率会下降",  # 实测单通道会把
         "vector_only": "关键词检索不可用，已降级为向量检索，数字与比例类问题的准确率会下降"}.get(mode, "")  # "兴图新科"与"兴图投资"混起来
    return "；".join(x for x in (r, f"生成服务不可用（{llm_err}）" if llm_err else "") if x)  # 两类降级可叠加


def prepare(question: str, lang: str = "auto", use_cache: bool = True) -> dict:  # 问答前半程
    """校验输入 → 查缓存 → 混合检索 + 重排。返回上下文与耗时，或带回友好提示的错误结果。"""
    t0 = time.time()                                                        # 计时起点
    q = (question or "").strip()                                            # 去首尾空白
    base = dict(contexts=[], retrieval_mode=None, degraded=False, cached=False)  # 公共字段
    if not q: return dict(base, answer="请输入需要查询的问题。", error="empty_question", elapsed=round(time.time() - t0, 3))  # 容错：空输入
    if len(q) > 500: return dict(base, answer="问题过长，请控制在 500 字以内。", error="query_too_long", elapsed=round(time.time() - t0, 3))  # 容错：超长
    lang = detect_lang(q) if lang in ("auto", "", None) else lang            # 自动判定语言（多语言支持）
    key = K.key_of(q, lang)                                                 # 缓存键
    if use_cache:                                                           # 允许缓存时
        hit = K.get(key)                                                    # 读缓存
        if hit: return dict(hit, cached=True, elapsed=round(time.time() - t0, 3))  # 命中即返回完整答案
    try:                                                                    # 检索阶段统一容错
        load()                                                              # 懒加载索引
        t = time.time()                                                     # 检索计时
        cand, mode = hybrid_search(STATE["client"], STATE["bm25"], STATE["chunks"], q)  # 混合召回，并回报生效通道
        hits = V.rerank(q, cand, C.RERANK_TOP_N)                             # 精排
        if not hits: raise RuntimeError("未检索到相关内容，请换一种问法或确认文档已解析")  # 召回为空明确提示
        # 检索精确度在**补全之前**算：它衡量的是检索/重排这一步的质量，
        # 补全行是"同表顺带取回"，拿它们充数会把指标虚高，口径就不干净了。
        prec = precision(_ctx(hits))                                        # ★检索精确度
        hits = expand_table_rows(hits, STATE["chunks"])                       # ★同表行补全，保证表格题答得全
        # ★低分尾巴裁剪：重排分低于相关性阈值的块，按本项目自己的口径就是"不相关"，
        # 留在上下文里只会给大模型添噪声、拉低上下文精确度。下限保底，避免全被裁空。
        keep = [h for h in hits if float(h.get("rerank_score", h.get("score", 0.0))) >= REL_THRESHOLD]  # 达标块
        if len(keep) >= C.MIN_CONTEXT: hits = keep                            # 够用才裁
        hits = merge_table_rows(hits)                                        # ★同表行拼回整表，再喂给大模型
        cs = _ctx(hits)                                                     # 转成接口结构
        return dict(question=q, lang=lang, contexts=cs, error=None,          # 上下文与出处
                    retrieval_mode=mode, degraded="only" in mode,            # 是否降级
                    degraded_reason=_reason(mode, ""),                       # 降级说明
                    retrieval_precision=prec,                                # ★检索精确度
                    retrieval_ms=round((time.time() - t) * 1000, 1),         # 检索+重排耗时
                    elapsed=round(time.time() - t0, 3), cached=False, _key=key)  # 计时与内部缓存键
    except FileNotFoundError:                                               # 索引文件缺失
        return dict(base, answer="知识库尚未构建，请先执行 python scripts/ingest.py。", error="index_missing",  # 友好提示
                    elapsed=round(time.time() - t0, 3))                     # 计时
    except Exception as e:                                                  # 其余异常统一兜底
        print(f"[问答异常] {type(e).__name__}: {e}")                          # 打印便于排障
        msg = str(e) if isinstance(e, (RuntimeError, FileNotFoundError)) else "系统繁忙或依赖服务暂时不可用，请稍后重试。"  # 自己抛的中文提示直接透出
        return dict(base, answer=msg, error=type(e).__name__, detail=str(e)[:200],  # 友好提示 + 错误类型
                    elapsed=round(time.time() - t0, 3))                     # 计时


def answer(question: str, lang: str = "auto", use_cache: bool = True) -> dict:  # 问答主流程
    """prepare + 生成，一次性返回完整 JSON；评测、压测与接口调用都走这条。"""
    t0 = time.time()                                                        # 总计时
    p = prepare(question, lang, use_cache)                                  # 前半程
    if p.get("error") or p.get("cached"): return dict(p, elapsed=round(time.time() - t0, 3))  # 出错或命中缓存，无需生成
    tg = time.time()                                                        # 生成计时
    try: text, err = generate(p["question"], p["contexts"], p["lang"]), None  # 调大模型
    except Exception as e:                                                  # 断网、限流、Key 失效、余额不足(402)、超时
        print(f"[生成降级] {type(e).__name__}: {e}")                          # 打印原因
        text, err = "生成服务暂不可用，以下为检索到的原文片段，请自行查阅。", type(e).__name__  # 把检索结果交回用户，而不是只回一句报错
    reason = _reason(p.get("retrieval_mode", ""), err)                       # 检索降级 + 生成降级
    r = dict(answer=text, lang=p.get("lang"), contexts=p.get("contexts", []),  # 答案与出处
             error=None, cached=False, retrieval_mode=p.get("retrieval_mode"),  # 检索通道
             retrieval_precision=p.get("retrieval_precision", 0.0),           # ★检索精确度
             degraded=bool(reason), degraded_reason=reason,                   # 降级标记，前端据此提示
             retrieval_ms=p.get("retrieval_ms"),                             # 检索耗时（毫秒）
             generate_ms=round((time.time() - tg) * 1000, 1),                 # 生成耗时（毫秒）
             elapsed=round(time.time() - t0, 3))                             # 端到端耗时
    if use_cache: K.set(p["_key"], {k: v for k, v in r.items() if k != "elapsed"})  # 写缓存；耗时是变量不入缓存
    return r                                                                # 返回


def warmup() -> dict:                                                        # 服务启动预热
    """启动即载入索引与本地模型；否则首个请求要多等十几秒加载模型，会被当成系统卡住。"""
    try:                                                                    # 预热失败不阻断启动
        load()                                                              # 载入块、BM25 并连 Milvus
        V.embed(["预热"])                                                     # 触发 bge-m3 加载
        V.rerank("预热", [dict(text="预热", chunk_id="warm", page_no=0)], 1)   # 触发重排模型加载
        # 还要空跑一次完整检索：Milvus 首次 search 会触发段加载，实测不预热的话
        # 第一个真实请求要等 24 秒（后续请求才回到 2 秒量级），会被当成系统卡死。
        hybrid_search(STATE["client"], STATE["bm25"], STATE["chunks"], "预热")  # 触发 Milvus 段加载与 jieba 词典构建
        return dict(ok=True, chunks=len(STATE["chunks"]))                    # 回报已载入块数
    except Exception as e:                                                  # 依赖未就绪
        print(f"[预热失败] {type(e).__name__}: {e}")                          # 打印原因
        return dict(ok=False, error=f"{type(e).__name__}: {e}")               # 返回失败信息，首个请求会再试一次
