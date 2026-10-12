# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""问答流水线：语义分块、索引构建、Redis 缓存、DeepSeek 生成，并保留优化前基线以做对比。"""
import hashlib, json, pickle, re, time                                                            # 缓存键、序列化、索引持久化、切句、计时
from . import config as C                                                                         # 复用集中配置
from . import retrieval as R                                                                      # 复用检索能力

_SENT = re.compile(r"(?<=[。！？；])|(?<=[!?])\s+|(?<=[.])\s+")                                      # 中英文句末切分，小数点后无空格不会被切断
STATE = {"bm25": None, "chunks": {}, "client": None}                                              # 进程内索引状态：BM25、两套块列表、Milvus 客户端
_INDEX, _BM25 = C.ROOT / "data" / "index.json", C.ROOT / "data" / "bm25.pkl"                       # 索引落盘路径，重启免于重建


def split_sentences(text: str) -> list:                                                           # 句子切分
    """按中英文句末标点切句，过短片段向后合并，保证每个语义单元自带上下文。"""
    out, buf = [], ""                                                                             # out 存结果，buf 暂存碎片
    for part in _SENT.split(text or ""):                                                          # 逐片段遍历
        buf += part                                                                               # 累积到缓冲区
        if len(buf.strip()) >= 8: out.append(buf.strip()); buf = ""                                # 够长即作为一句收集
    return out + ([buf.strip()] if buf.strip() else [])                                           # 收尾，余下片段也成句


def _flush(buf: list, chunks: list) -> None:                                                      # 把缓冲区落成一个块
    """缓冲区拼接为块文本；页码记录起始页与结束页，便于按页统计检索命中率。"""
    text = "\n".join(x["content"] for x in buf).strip() if buf else ""                             # 拼接块文本
    if len(text) < 10: return                                                                     # 空或过短碎片直接丢弃
    pages = [x["page_no"] for x in buf]                                                           # 收集涉及页码
    chunks.append(dict(text=text, page_no=pages[0], page_end=max(pages), chunk_id=f"opt-{len(chunks)}"))  # 记录块与唯一 ID


def semantic_chunks(blocks: list) -> list:                                                        # 语义分块（优化后方案核心）
    """相邻单元向量相似度低于阈值即切块，超长兜底；标题与表格单独成界，标题作章节语境。"""
    units = []                                                                                    # 待分块的语义单元
    for b in blocks:                                                                              # 遍历解析出的块
        if b["type"] == "table" or len(b["content"]) <= C.SEM_CHUNK_MAX:                           # 表格与短块整体作为一个单元
            units.append(dict(content=b["content"], page_no=b["page_no"], kind=b["type"]))         # 直接登记
        else:                                                                                     # 长段落按句拆开
            units += [dict(content=s, page_no=b["page_no"], kind="text") for s in split_sentences(b["content"])]  # 逐句登记并保留页码
    if not units: return []                                                                       # 没有可用单元则返回空
    vecs, chunks = R.embed([u["content"] for u in units]), []                                      # 一次性编码全部单元；chunks 收集结果
    buf, bvec, title = [], [], ""                                                                 # 滑动缓冲：单元、对应向量、当前章节名
    for unit, vec in zip(units, vecs):                                                             # 顺序扫描语义单元
        break_here = unit["kind"] in ("title", "table")                                            # 标题与表格必须单独成界
        if buf and (break_here or sum(len(x["content"]) for x in buf) + len(unit["content"]) > C.SEM_CHUNK_MAX  # 边界、超长
                    or float(sum(bvec) / len(bvec) @ vec) < C.SEM_THRESHOLD):                       # 或与当前块语义中心相似度低于阈值
            _flush(buf, chunks); buf, bvec = [], []                                                # 在此处切块并清空缓冲
        if unit["kind"] == "title": title = unit["content"][:60]; continue                          # 标题只作章节语境，截断后不入块
        if not buf and title: buf.append(dict(content=title, page_no=unit["page_no"])); bvec.append(vec)  # 新块开头带上章节名
        buf.append(unit); bvec.append(vec)                                                          # 当前单元入缓冲
        if unit["kind"] == "table": _flush(buf, chunks); buf, bvec = [], []                          # 表格内容独立成块
    _flush(buf, chunks)                                                                           # 处理尾部缓冲
    merged = []                                                                                   # 合并过短块：实测"章节名+公司名"这类 20~60 字块会挤占检索前排却无信息
    for c in chunks:                                                                              # 逐个检查块长度
        if merged and (len(c["text"]) < 150 or len(merged[-1]["text"]) < 150):                     # 当前块或前一块过短
            merged[-1]["text"] += "\n" + c["text"]                                                 # 并入前一块，保持上下文连贯
            merged[-1]["page_end"] = max(merged[-1]["page_end"], c["page_end"])                    # 同步扩展页区间
        else: merged.append(c)                                                                     # 长度合适则独立成块
    for i, c in enumerate(merged): c["chunk_id"] = f"opt-{i}"                                       # 重新编号，保证块 ID 连续
    return merged                                                                                 # 返回语义块


def baseline_chunks(pdf_path) -> list:                                                            # 优化前基线：PyMuPDF + 固定长度分块
    """纯 PyMuPDF 逐页取文本后按固定字符硬切，不含表格还原与语义边界，用以形成对照。"""
    import pymupdf                                                                                # 延迟导入
    doc, chunks = pymupdf.open(str(pdf_path)), []                                                  # 打开 PDF 并准备结果
    for i in range(doc.page_count):                                                                # 逐页处理
        text = re.sub(r"\s+", " ", doc[i].get_text()).strip()                                       # 压平空白，去掉换行噪声
        for j in range(0, len(text), C.BASE_CHUNK_SIZE):                                            # 按固定步长切分
            piece = text[j:j + C.BASE_CHUNK_SIZE].strip()                                           # 取一段
            if len(piece) >= 30: chunks.append(dict(text=piece, page_no=i + 1, page_end=i + 1, chunk_id=f"base-{len(chunks)}"))  # 过滤过短片段
    doc.close()                                                                                   # 释放句柄
    return chunks                                                                                 # 返回固定长度块


def build_indexes(use_mineru: bool = True) -> dict:                                                # 构建优化前后两套索引
    """分别建基线索引与优化索引并写入 Milvus，把块列表与 BM25 索引落盘供后续复用。"""
    from . import pdf_parser as P                                                                  # 延迟导入，避免循环依赖
    client = R.milvus_client()                                                                     # 连接 Milvus
    base_chunks = baseline_chunks(C.PDF_PATH)                                                      # 优化前：固定长度分块
    blocks, parser = P.parse_pdf(C.PDF_PATH, use_mineru)                                           # 优化后：MinerU 解析，失败自动降级
    opt_chunks = semantic_chunks(blocks)                                                           # 优化后：语义分块
    for name, chunks in ((C.MILVUS_COLLECTION_BASE, base_chunks), (C.MILVUS_COLLECTION, opt_chunks)):  # 两套索引分别入库
        R.recreate_collection(client, name); R.insert_chunks(client, name, chunks)                  # 重建集合后写入向量
    _INDEX.write_text(json.dumps({"base_chunks": base_chunks, "opt_chunks": opt_chunks, "parser": parser},  # 块列表与解析器名落盘
                                 ensure_ascii=False), encoding="utf-8")                             # 保证中文可读
    pickle.dump(R.build_bm25(opt_chunks), open(_BM25, "wb"))                                        # 只有优化方案用 BM25，持久化免重建
    STATE.update({"bm25": None, "chunks": {}, "client": client})                                    # 复位内存状态，下次重新加载
    return {"parser": parser, "baseline_chunks": len(base_chunks), "optimized_chunks": len(opt_chunks)}  # 返回规模信息


_REDIS = None                                                                                     # Redis 客户端单例占位


def _cache(key: str, value: dict = None):                                                          # 读写缓存二合一
    """value 为空表示读缓存，否则写缓存；Redis 不可用时静默降级，绝不阻断问答主流程。"""
    global _REDIS                                                                                  # 声明模块级单例
    try:                                                                                           # 缓存故障不应影响问答
        if _REDIS is None:                                                                         # 首次调用建立连接
            import redis                                                                           # 延迟导入
            _REDIS = redis.Redis.from_url(C.REDIS_URL, protocol=2, socket_timeout=0.5)             # 本机 Redis 5.x 不支持 RESP3，必须用 RESP2
        if value is None:                                                                          # 读分支
            raw = _REDIS.get(key)                                                                  # 取值
            return json.loads(raw.decode("utf-8")) if raw else None                                 # 反序列化返回，未命中给 None
        _REDIS.setex(key, 86400, json.dumps(value, ensure_ascii=False)); return None                # 写分支：带一天过期时间
    except Exception: return None                                                                   # 连接失败一律按未命中处理


def generate(question: str, contexts: list) -> str:                                                # 调用 DeepSeek 生成答案
    """严格依据检索资料作答，用与提问相同的语言回答，找不到依据时明确拒答以抑制幻觉。"""
    from openai import OpenAI                                                                      # 延迟导入，加快启动
    if not C.DEEPSEEK_API_KEY: raise RuntimeError("未配置 DEEPSEEK_API_KEY，请参考 .env.example 配置后重试")  # 缺密钥时明确提示
    client = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL, timeout=20, max_retries=1)  # 指向 DeepSeek 兼容接口，超时受限保证快速失败
    material = "\n\n".join(f"[资料{i + 1}]\n{c}" for i, c in enumerate(contexts))                    # 拼接检索资料
    prompt = ("你是招股说明书问答助手。请严格依据下列资料回答，不要编造。\n"                                   # 角色与首要约束
              "要求：1) 用与问题相同的语言作答；2) 数字、比例、名称必须与资料完全一致；"                         # 中英文支持与准确性要求
              f"3) 作答要完整：涉及名称、标准、工程时给出资料中的完整名称，不要只给简称或文号；"                            # 避免答案不完整导致要点缺失
              f"4) 资料中找不到答案时回复\"资料中未找到相关信息\"。\n\n{material}\n\n问题：{question}\n答案：")      # 拒答策略与资料注入
    resp = client.chat.completions.create(model=C.LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=512)  # 温度 0 保证答案稳定可复现
    return (resp.choices[0].message.content or "").strip()                                          # 取出答案文本


def answer(question: str, mode: str = "optimized", use_cache: bool = True) -> dict:                 # 问答主流程
    """校验输入 → 查缓存 → 检索（基线仅向量 / 优化走混合+重排）→ 生成，返回答案与各阶段耗时。"""
    t0 = time.time()                                                                               # 整体计时起点
    question = (question or "").strip()                                                            # 去除首尾空白
    if not question: return dict(answer="请输入需要查询的问题。", error="empty_question", elapsed=round(time.time() - t0, 3), cached=False)  # 容错：空输入
    if len(question) > 500: return dict(answer="问题过长，请控制在 500 字以内。", error="query_too_long", elapsed=round(time.time() - t0, 3), cached=False)  # 容错：超长输入
    key = "rag:%s:%s" % (mode, hashlib.md5(question.encode("utf-8")).hexdigest())                   # 缓存键：模式 + 问题摘要
    if use_cache:                                                                                  # 允许用缓存时
        hit = _cache(key)                                                                          # 尝试读缓存
        if hit: return dict(hit, cached=True, elapsed=round(time.time() - t0, 3))                  # 命中直接返回，耗时仅为缓存读取
    try:                                                                                           # 检索与生成阶段统一容错
        if STATE["client"] is None: STATE["client"] = R.milvus_client()                             # 懒加载 Milvus 连接
        if not STATE["chunks"]:                                                                     # 首次提问载入块列表与 BM25，之后常驻内存
            data = json.loads(_INDEX.read_text(encoding="utf-8"))                                    # 读取落盘索引
            STATE["chunks"] = {"base": data["base_chunks"], "opt": data["opt_chunks"]}               # 分别保存两套块
            STATE["bm25"] = pickle.load(open(_BM25, "rb"))                                           # 反序列化 BM25
        t_ret = time.time()                                                                         # 检索计时起点
        if mode == "baseline":                                                                      # 优化前基线：仅向量检索、不重排
            contexts = R.vector_search(STATE["client"], C.MILVUS_COLLECTION_BASE, question, C.RERANK_TOP_N)  # 只查基线集合
        else:                                                                                        # 优化后：混合检索 + 重排
            contexts = R.rerank(question, R.hybrid_search(STATE["client"], C.MILVUS_COLLECTION, STATE["bm25"], STATE["chunks"]["opt"], question), C.RERANK_TOP_N)  # 召回后精排
        t_gen = time.time()                                                                         # 生成计时起点
        text = generate(question, [c["text"] for c in contexts])                                     # 调用大模型生成答案
        result = dict(answer=text, mode=mode, elapsed=round(time.time() - t0, 3),                    # 答案与总耗时
                      retrieval_ms=round((t_gen - t_ret) * 1000, 1),                                 # 检索+重排耗时（毫秒）
                      generate_ms=round((time.time() - t_gen) * 1000, 1),                            # 生成耗时（毫秒）
                      contexts=[dict(page_no=c["page_no"], page_end=c.get("page_end", c["page_no"]), text=c["text"],  # 返回命中片段全文（界面自行截断展示，评测按全文判定）
                                     score=round(c.get("rerank_score", c.get("vec_score", c.get("score", 0.0))), 4)) for c in contexts],  # 相关度得分
                      cached=False, error=None)                                                      # 未命中缓存
        if use_cache: _cache(key, {k: v for k, v in result.items() if k != "elapsed"})                # 结果写缓存，耗时是变量不入缓存
        return result                                                                                # 返回结果
    except FileNotFoundError:                                                                        # 索引文件缺失
        return dict(answer="知识库尚未构建，请先执行 python -m src.rag_pipeline 构建索引。", error="index_missing", elapsed=round(time.time() - t0, 3), cached=False)  # 友好提示
    except Exception as exc:                                                                         # 其余异常统一兜底
        print(f"[问答异常] {type(exc).__name__}: {exc}")                                              # 打印错误类型便于排障
        friendly = str(exc) if isinstance(exc, RuntimeError) else "系统繁忙或依赖服务暂时不可用，请稍后重试。"  # RuntimeError 是本模块自己抛的中文提示，直接透出
        return dict(answer=friendly, error=type(exc).__name__, detail=str(exc)[:200], elapsed=round(time.time() - t0, 3), cached=False)  # 返回友好提示


if __name__ == "__main__":                                                                         # 支持直接执行以构建索引
    print(json.dumps(build_indexes(), ensure_ascii=False))                                          # 输出解析器与两套索引的块规模
