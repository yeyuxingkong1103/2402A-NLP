# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：语义分块。图像块、表格行、标题、段落四类分别处理——相邻语义单元相似度低于阈值处切块。
import re                                                                   # 中英文切句
import config as C                                                          # 集中配置

_SENT = re.compile(r"(?<=[。！？；])|(?<=[!?])\s+|(?<=[.])\s+")                # 句末标点切分；小数点后无空格不会被误切


def split_sentences(text: str) -> list:                                     # 句子切分
    """按中英文句末标点切句；过短碎片向后合并，保证每个语义单元自带上下文。"""
    out, buf = [], ""                                                       # 结果与暂存缓冲
    for part in _SENT.split(text or ""):                                    # 逐片段
        buf += part                                                         # 累积
        if len(buf.strip()) >= 8: out.append(buf.strip()); buf = ""          # 够长即成句
    return out + ([buf.strip()] if buf.strip() else [])                     # 收尾


def _flush(buf: list, chunks: list, n: int) -> None:                        # 缓冲区落块
    """拼接为块文本；页码取起止页，图像块与表格行额外带出定位字段。"""
    if not buf: return                                                      # 空缓冲
    text = "\n".join(x["content"] for x in buf).strip()                     # 拼接
    if len(text) < 10: return                                               # 过短碎片丢弃
    pages = [x["page_no"] for x in buf]                                     # 涉及页码
    h = buf[0]                                                              # 以首单元决定定位属性
    chunks.append(dict(doc=h.get("doc", ""), text=text, page_no=min(pages),  # 文档、文本、起始页
                       page_end=max(pages), block_type=h.get("type", "text"),  # 结束页与块类型
                       table_id=h.get("table_id", ""), row_index=h.get("row_index", 0),  # 表格行定位：供同表补全
                       image_type=h.get("image_type", ""), image_path=h.get("image_path", ""),  # ★图像语义块元数据
                       locator=h.get("locator", ""), chunk_id=f"gd4-{n + len(chunks)}"))  # 定位串与全局唯一块号


def semantic_chunks(blocks: list, embed) -> list:                           # 语义分块主流程
    """图像块与表格行是硬边界（各自独立成块，且不与段落混合），其余按语义相似度切分。"""
    HARD = ("image", "table_row", "title")                                  # 不允许被合并、合并时作边界的类型
    units = []                                                              # 待分块的语义单元
    for b in blocks:                                                        # 遍历解析出的块
        if b.get("type") in HARD or len(b["content"]) <= C.SEM_CHUNK_MAX:    # 图像/表格行/标题/短块：整体作为一单元
            units.append(dict(b, content=b["content"]))                     # 直接登记
        else: units += [dict(b, content=s, type="text") for s in split_sentences(b["content"])]  # 长段落按句拆开
    if not units: return []                                                 # 无单元
    # 只为需要语义比较的单元编码；图像与表格行是硬边界，向量不参与切块判定，可省下大量编码时间
    need = [i for i, u in enumerate(units) if u["type"] not in ("image", "table_row")]  # 需要向量的下标
    vecs = dict(zip(need, embed([units[i]["content"] for i in need]))) if need else {}  # 批量编码
    chunks, buf, bvec, title = [], [], [], ""                               # 结果、滑动缓冲、缓冲向量、当前章节名
    for i, unit in enumerate(units):                                        # 顺序扫描
        vec = vecs.get(i)                                                   # 本单元向量（图像/表格行无）
        hard = unit["type"] in HARD                                         # 硬边界
        long = sum(len(x["content"]) for x in buf) + len(unit["content"]) > C.SEM_CHUNK_MAX  # 超长
        # 图像/表格行没有向量，必须先短路，否则 None 参与矩阵运算会报错
        far = vec is not None and buf and bvec and float(sum(bvec) / len(bvec) @ vec) < C.SEM_THRESHOLD  # 与块中心语义距离过远
        if buf and (hard or long or far): _flush(buf, chunks, n=0); buf, bvec = [], []  # 三种边界处切块
        if unit["type"] == "title": title = unit["content"][:60]; continue   # 标题只作章节语境，本身不入块
        if not buf and title and vec is not None:                           # 段落新块开头补章节名，避免"张冠李戴"
            buf.append(dict(unit, content=title)); bvec.append(vec)          # 借标题补上下文
        buf.append(unit)                                                    # 入缓冲
        if vec is not None: bvec.append(vec)                                # 登记向量
        if unit["type"] in ("image", "table_row"):                          # 图像与表格行立即独立成块
            _flush(buf, chunks, n=0); buf, bvec = [], []                     # 不与正文混合，保证定位精确
    _flush(buf, chunks, n=0)                                                # 尾部缓冲
    merged = []                                                             # 合并过短正文块："章节名+公司名"这类碎块会挤占前排却无信息
    for c in chunks:                                                        # 逐块
        if merged and c["block_type"] not in ("image", "table_row") and (len(c["text"]) < C.MERGE_MIN  # 图像与表格行不参与合并
                                                                        or len(merged[-1]["text"]) < C.MERGE_MIN):
            merged[-1]["text"] += "\n" + c["text"]                          # 并入前一块
            merged[-1]["page_end"] = max(merged[-1]["page_end"], c["page_end"])  # 扩展页区间
        else: merged.append(c)                                              # 独立成块
    for i, c in enumerate(merged): c["chunk_id"] = f"gd4-{i}"                # 重编号，保证块 ID 连续
    return merged                                                           # 返回语义块
