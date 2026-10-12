# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：问答流水线。索引构建（解析→表格结构化→语义分块→向量化→Milvus）、DeepSeek 生成、缓存与容错主流程。
import json, pickle, re, time                                                      # 序列化、索引持久化、语言判定、计时
from . import cache as K                                                           # Redis 缓存
from . import chunking as G                                                        # 语义分块
from . import config as C                                                          # 集中配置
from . import retriever as R                                                       # 检索与重排
from . import table_struct as T                                                    # 表格结构化
from . import vector_store as V                                                    # 向量库

STATE = {"bm25": None, "chunks": [], "client": None, "stats": {}, "milvus_tried": 0.0}  # 进程内索引状态，常驻内存避免每次重载


def detect_lang(question: str) -> str:                                              # 语言判定
    """中文按字符占比判定：中文字符超过 15% 视为中文提问，否则英文。"""
    cjk = len(re.findall(r"[一-鿿]", question or ""))                                # 中文字符数
    return "zh" if cjk >= max(1, len(question or "") * 0.15) else "en"               # 超阈值判定中文


def build_index(use_mineru: bool = True, only: str = None) -> dict:                  # 构建索引（离线动作）
    """逐篇解析 PDF → 表格结构化 → 语义分块 → 向量入库 → 建 BM25，并把块与索引落盘供服务复用。"""
    from . import pdf_parser as P                                                   # 延迟导入，避免循环依赖
    import shutil                                                                   # 清理单篇解析产物
    pdfs = C.docs()                                                                 # 知识库内的全部 PDF
    if not pdfs: raise FileNotFoundError(f"未在 {C.RAW_DIR} 找到任何 PDF，请先上传")    # 目录为空时明确报错
    if only and not any(p.stem == only for p in pdfs):                                # 指定了不存在的文档
        raise FileNotFoundError(f"知识库中没有文档：{only}，当前有 {[p.stem for p in pdfs]}")  # 明确报错而不是静默产出空索引
    all_blocks, report = [], []                                                     # 汇总裁剪后的块与逐篇解析报告
    for pdf in pdfs:                                                                # 逐个文档处理（串行：本机 GPU 只有 8G，并行会被系统杀）
        t0 = time.time()                                                            # 单篇计时
        # only 的语义是"强制重新解析该篇"，而不是"索引里只留该篇"——
        # 若只索引指定文档，另一篇会被挤出索引，后续提问直接查不到，这是必须避免的静默数据丢失。
        if only == pdf.stem: shutil.rmtree(C.PARSED_DIR / pdf.stem, ignore_errors=True)  # 清掉旧产物，逼 MinerU 重跑
        blocks, parser = P.parse_pdf(pdf, use_mineru)                                # 解析，失败自动逐级降级
        blocks = T.structure_tables(blocks)                                          # ★ 表格结构化与定位
        tables = len({b["table_id"] for b in blocks if b.get("table_id")})            # 统计表格张数
        all_blocks += blocks                                                         # 汇总
        report.append(dict(doc=pdf.stem, parser=parser, tables=tables, blocks=len(blocks),  # 逐篇报告
                           elapsed=round(time.time() - t0, 1)))                      # 解析耗时
    chunks = G.semantic_chunks(all_blocks, V.embed)                                  # 语义分块（表格行/标题/段落分别处理）
    client = V.milvus_client()                                                       # 连接 Milvus
    V.recreate_collection(client)                                                    # 重建集合，保证干净状态
    V.insert_chunks(client, chunks)                                                  # 向量入库
    bm25 = R.build_bm25(chunks)                                                      # 建 BM25 索引
    stats = dict(docs=report, chunks=len(chunks),                                     # 汇总规模信息：逐篇报告与总块数
                 tables=sum(r["tables"] for r in report),                            # 表格总张数
                 table_rows=sum(1 for c in chunks if c["block_type"] == "table_row"))  # 表格行块数（工单核心产物）
    C.INDEX_FILE.write_text(json.dumps(dict(chunks=chunks, stats=stats), ensure_ascii=False),  # 块与统计落盘
                            encoding="utf-8")                                        # 中文可读
    pickle.dump(bm25, open(C.BM25_FILE, "wb"))                                       # BM25 索引落盘，重启免重建
    STATE.update(bm25=bm25, chunks=chunks, client=client, stats=stats)                 # 同步内存状态，立即可用
    return stats                                                                     # 返回构建报告


def _load() -> None:                                                                # 懒加载索引
    """首次提问时载入块列表、BM25 索引并连接 Milvus，之后常驻内存。"""
    # 坑：Milvus 不可用时 MilvusClient() 会在构造阶段就抛异常。若不加保护，异常会从 _load 直接冒到
    # 最外层，向量检索内部的兜底分支根本走不到，结果是整个问答失败——而不是退回 BM25 继续作答。
    # 所以这里必须自己兜住：连不上就把 client 置空，让 BM25 通道把问答撑起来。
    if STATE["client"] is None and time.time() - STATE["milvus_tried"] > 60:           # 每 60 秒最多重连一次，避免每次提问都白等超时
        STATE["milvus_tried"] = time.time()                                            # 记下本次尝试时间
        try: STATE["client"] = V.milvus_client()                                       # 尝试连接 Milvus
        except Exception as exc:                                                       # 连不上或超时
            print(f"[检索降级] Milvus 不可用（{type(exc).__name__}），改用 BM25 关键词检索")  # 打印原因，服务继续可用
    if STATE["chunks"]: return                                                       # 已载入则直接返回
    # 索引文件损坏要给出可操作的提示，而不是让它冒到最外层变成一句"系统繁忙"——
    # 实测 bm25.pkl 被写坏时，用户完全无从知道该重建索引。
    try:                                                                             # 载入分块结果与 BM25 索引
        data = json.loads(C.INDEX_FILE.read_text(encoding="utf-8"))                   # 读落盘的块列表
        STATE["chunks"], STATE["stats"] = data["chunks"], data.get("stats") or {}      # 恢复块与统计
        STATE["bm25"] = pickle.load(open(C.BM25_FILE, "rb"))                          # 反序列化 BM25
    except FileNotFoundError: raise                                                  # 文件缺失交给上层报"尚未构建"
    except Exception as exc:                                                         # 内容损坏（JSON 截断、pickle 损坏等）
        raise RuntimeError(f"索引文件已损坏（{type(exc).__name__}），请执行 python -m app.rag_core 重建索引")


def warmup() -> dict:                                                                # 服务启动预热
    """启动即载入索引与两个本地模型；否则第一个用户请求要多等 20 秒加载模型，会被当成系统卡住。"""
    try:                                                                             # 预热失败不阻断启动
        _load()                                                                      # 载入块列表、BM25 索引并连接 Milvus
        V.embed(["预热"])                                                             # 触发 bge-m3 加载
        R.rerank("预热", [dict(text="预热", chunk_id="warm", page_no=0)], 1)           # 触发重排模型加载
        return dict(ok=True, chunks=len(STATE["chunks"]))                             # 回报已载入的块数
    except Exception as exc:                                                         # 依赖未就绪等情况
        print(f"[预热失败] {type(exc).__name__}: {exc}")                               # 打印原因
        return dict(ok=False, error=f"{type(exc).__name__}: {exc}")                    # 返回失败信息，首个请求会再试一次


def _prompt(question: str, contexts: list, lang: str) -> str:                         # 构造提示词
    """严格依据检索资料作答、用与提问相同的语言；资料中无据时明确拒答以抑制编造。"""
    def _label(i, c):                                                              # 给每条资料打出处标签
        """标签写明页码与表格定位，让模型能分辨"范围相近但不同"的表，避免跨表合并条目。"""
        return f"[资料{i + 1}｜来源：{c.get('locator') or (c.get('doc', '') + ' P' + str(c.get('page_no')))}]\n{c['text']}"  # 标签 + 片段全文
    material = "\n\n".join(_label(i, c) for i, c in enumerate(contexts))             # 拼接检索到的资料
    asked = "用中文作答" if lang == "zh" else "Answer in English"                       # 语言约束
    return ("你是招股说明书问答助手，请严格依据下列资料回答，不要编造。\n"                    # 角色与首要约束
            f"要求：1) {asked}；2) 数字、比例、名称必须与资料完全一致，不要四舍五入或改写；"       # 准确性要求
            "3) 作答要完整，涉及企业名称、项目名称、奖项、标准时给出资料中的完整名称；"            # 完整性要求
            "4) 每份资料都标注了来源；若多份资料内容相近但范围不同，只采用与问题问法最匹配的一份，"    # 防跨表合并：实测"曾为关联方"表会
            "不要把不同资料的条目合并作答；"                                              # 被并入"不存在控制关系的关联方"的答案
            "5) 资料中找不到答案时回复\"资料中未找到相关信息\"。\n\n"                        # 拒答策略
            f"{material}\n\n问题：{question}\n答案：")                                  # 注入资料与问题


def _llm_args(question, contexts, lang, stream: bool) -> dict:                        # 组装大模型请求参数
    """温度 0 保证答案可复现；推理型模型必须显式传 reasoning_effort，否则思考过程吃光 max_tokens 返回空内容。"""
    args = dict(model=C.LLM_MODEL, messages=[{"role": "user", "content": _prompt(question, contexts, lang)}],  # 模型与提示词
                temperature=0, max_tokens=800, stream=stream)                        # 温度、上限、是否流式
    if C.LLM_REASONING_EFFORT: args["reasoning_effort"] = C.LLM_REASONING_EFFORT       # 推理型模型的开关
    return args                                                                      # 返回参数


def _client():                                                                        # 大模型客户端
    """DeepSeek 走 OpenAI 兼容接口；超时受限以保证快速失败而不是长时间挂住。"""
    from openai import OpenAI                                                        # 延迟导入
    if not C.DEEPSEEK_API_KEY: raise RuntimeError("未配置 DEEPSEEK_API_KEY，请参考 .env.example 配置后重试")  # 缺密钥明确提示
    return OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL, timeout=20, max_retries=1)  # 20 秒超时


def generate(question: str, contexts: list, lang: str) -> str:                        # 非流式生成
    """一次性取回完整答案，供评测、压测等不需要流式的调用方使用。"""
    resp = _client().chat.completions.create(**_llm_args(question, contexts, lang, False))  # 一次调用
    return (resp.choices[0].message.content or "").strip()                            # 取答案文本


def generate_stream(question: str, contexts: list, lang: str):                         # 流式生成
    """逐段吐出 token：前端可以边生成边显示，把"用户看到第一个字"的等待压到检索完成那一刻。"""
    for chunk in _client().chat.completions.create(**_llm_args(question, contexts, lang, True)):  # 流式调用
        delta = chunk.choices[0].delta.content                                        # 本次增量
        if delta: yield delta                                                         # 只吐有内容的增量


def _contexts(hits: list) -> list:                                                   # 整理检索结果为接口输出
    """把块转为前端需要的结构：片段全文、页码、表格定位、逐条相关度。"""
    return [dict(doc=h.get("doc", ""), page_no=h.get("page_no"), page_end=h.get("page_end", h.get("page_no")),  # 文档与页区间
                 text=h.get("text", ""), block_type=h.get("block_type", "text"),     # 片段全文与块类型
                 table_id=h.get("table_id", ""), table_caption=h.get("caption", ""),  # 表格定位：表标识与表题
                 row_index=h.get("row_index", 0), locator=h.get("locator", ""),       # 行号与可读定位串
                 score=round(float(h.get("rerank_score", h.get("score", 0.0))), 4))    # 相关度得分
            for h in hits]                                                           # 逐条转换


def _retrieval_reason(mode: str) -> str:                                              # 检索降级说明
    """少了一路检索通道，答案质量会实打实下降，必须显式标注而不是静默返回。"""
    return {"bm25_only": "向量检索不可用，已降级为 BM25 关键词检索，实体名相近的问题准确率会下降",  # 实测 BM25 单通道会把
            "vector_only": "关键词检索不可用，已降级为向量检索，数字与比例类问题的准确率会下降"}.get(mode, "")  # "兴图新科"与"兴图投资"混起来
def _join_reason(reason: str, llm_error: str) -> str:                                 # 拼接降级原因
    """检索降级与生成降级可以叠加，一起告诉用户。"""
    return "；".join(x for x in (reason, f"生成服务不可用（{llm_error}）" if llm_error else "") if x)


def prepare(question: str, lang: str = "auto", use_cache: bool = True, top_n: int = None) -> dict:  # 问答前半程
    """校验输入 → 查缓存 → 混合检索 + 重排。返回上下文与检索耗时，或错误结果。

    之所以从 answer 里拆出来：流式接口要在**检索一完成**就把片段推给前端，
    不能等生成结束——否则首字延迟等于整个端到端耗时，流式就白做了。
    """
    t0 = time.time()                                                                 # 计时起点
    question = (question or "").strip()                                              # 去首尾空白
    if not question: return dict(answer="请输入需要查询的问题。", contexts=[], error="empty_question",  # 容错：空输入
                                  retrieval_mode=None, degraded=False, elapsed=round(time.time() - t0, 3), cached=False)  # 友好提示
    if len(question) > 500: return dict(answer="问题过长，请控制在 500 字以内。", contexts=[], error="query_too_long",  # 容错：超长输入
                                         retrieval_mode=None, degraded=False, elapsed=round(time.time() - t0, 3), cached=False)  # 直接返回
    lang = detect_lang(question) if lang in ("auto", "", None) else lang               # 自动判定语言
    key = K.key_of(question, lang)                                                    # 缓存键
    if use_cache:                                                                    # 允许缓存时
        hit = K.get(key)                                                             # 尝试读缓存
        if hit: return dict(hit, cached=True, elapsed=round(time.time() - t0, 3), _key=key)  # 命中即返回完整答案
    try:                                                                             # 检索阶段统一容错
        _load()                                                                      # 懒加载索引
        t_ret = time.time()                                                          # 检索计时起点
        cand, mode = R.hybrid_search(STATE["client"], STATE["bm25"], STATE["chunks"], question)  # 混合召回，并回报实际生效的通道
        hits = R.rerank(question, cand, top_n or C.RERANK_TOP_N)                       # 重排，取交给大模型的上下文条数
        if not hits: raise RuntimeError("未检索到相关内容，请换一种问法或确认文档已解析")      # 召回为空时明确提示
        reason = _retrieval_reason(mode)                                              # 检索降级说明
        return dict(question=question, lang=lang, contexts=_contexts(hits), error=None,  # 上下文与来源
                    retrieval_mode=mode, degraded=bool(reason), degraded_reason=reason,  # 降级标记
                    retrieval_ms=round((time.time() - t_ret) * 1000, 1),               # 检索 + 重排耗时
                    elapsed=round(time.time() - t0, 3), cached=False, _key=key)        # 计时与内部缓存键
    except FileNotFoundError:                                                        # 索引文件缺失
        return dict(answer="知识库尚未构建，请先调用 /parse 或执行 python -m app.rag_core。", contexts=[],  # 友好提示
                    error="index_missing", retrieval_mode=None, degraded=False, elapsed=round(time.time() - t0, 3), cached=False)
    except Exception as exc:                                                         # 其余异常统一兜底
        print(f"[问答异常] {type(exc).__name__}: {exc}")                                # 打印错误便于排障
        friendly = str(exc) if isinstance(exc, (RuntimeError, FileNotFoundError)) else "系统繁忙或依赖服务暂时不可用，请稍后重试。"  # 自己抛的中文提示直接透出
        return dict(answer=friendly, contexts=[], error=type(exc).__name__, detail=str(exc)[:200],  # 友好提示
                    retrieval_mode=None, degraded=False, elapsed=round(time.time() - t0, 3), cached=False)  # 附错误类型


def answer(question: str, lang: str = "auto", use_cache: bool = True, top_n: int = None) -> dict:  # 问答主流程（非流式）
    """prepare + 生成，一次性返回完整 JSON；评测、压测与普通接口调用都走这条。"""
    t0 = time.time()                                                                 # 总计时起点
    prep = prepare(question, lang, use_cache, top_n)                                  # 前半程：校验 + 缓存 + 检索
    if prep.get("error") or prep.get("cached"):                                       # 出错或缓存命中，无需生成
        return dict(prep, elapsed=round(time.time() - t0, 3))                          # 直接返回
    t_gen = time.time()                                                              # 生成计时起点
    try: text, llm_error = generate(prep["question"], prep["contexts"], prep["lang"]), None  # 调大模型生成（带出处标签）
    except Exception as exc:                                                         # 断网、限流、Key 失效、超时
        print(f"[生成降级] {type(exc).__name__}: {exc}")                                # 打印原因便于排障
        text = "生成服务暂不可用，以下为检索到的原文片段，请自行查阅。"                      # 把检索结果交回用户，而不是只回一句报错
        llm_error = type(exc).__name__                                                # 记录错误类型
    result = _finish(prep, text, llm_error, t_gen, t0)                                 # 组装返回体
    if use_cache: K.set(prep["_key"], {k: v for k, v in result.items() if k != "elapsed"})  # 写缓存，耗时是变量不入缓存
    return result                                                                    # 返回结果


def _finish(prep: dict, text: str, llm_error: str, t_gen: float, t0: float) -> dict:   # 组装返回体
    """把检索阶段的上下文与生成阶段的结果合并成统一返回体，并叠加降级说明。"""
    reason = _join_reason(prep.get("degraded_reason", ""), llm_error)                  # 检索降级 + 生成降级
    return dict(answer=text, lang=prep.get("lang"), contexts=prep.get("contexts", []),  # 答案与来源片段
                error=None, cached=False, retrieval_mode=prep.get("retrieval_mode"),   # 未命中缓存，带回检索通道
                degraded=bool(reason), degraded_reason=reason,                        # 降级标记，前端据此提示
                retrieval_ms=prep.get("retrieval_ms"),                                # 检索 + 重排耗时（毫秒）
                generate_ms=round((time.time() - t_gen) * 1000, 1),                   # 生成耗时（毫秒）
                elapsed=round(time.time() - t0, 3))                                   # 端到端耗时


if __name__ == "__main__":                                                           # 支持命令行直接构建索引
    print(json.dumps(build_index(), ensure_ascii=False, indent=1))                    # 输出解析器、表格数与块规模
