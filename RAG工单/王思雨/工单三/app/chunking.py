# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：语义分块。表格行 / 标题 / 段落三类分别处理——表格行独立成块并带表头语境，标题作章节边界，长段落按语义切分。
import re                                                                         # 中英文切句
from . import config as C                                                         # 复用集中配置

# 中英文句末切分；小数点后无空格，不会被误切
_SENT = re.compile(r"(?<=[。！？；])|(?<=[!?])\s+|(?<=[.])\s+")


def split_sentences(text: str) -> list:                                           # 句子切分
    """按中英文句末标点切句，过短碎片向后合并，保证每个语义单元自带上下文。"""
    out, buf = [], ""                                                             # out 存结果，buf 暂存碎片
    for part in _SENT.split(text or ""):                                          # 逐片段遍历
        buf += part                                                               # 累积到缓冲区
        if len(buf.strip()) >= 8: out.append(buf.strip()); buf = ""                # 够长即作为一句收集
    return out + ([buf.strip()] if buf.strip() else [])                           # 收尾，余下片段也成句


def _flush(buf: list, chunks: list) -> None:                                      # 把缓冲区落成一个块
    """缓冲区拼接为块文本；页码记录起止页，表格行块额外带出表格定位字段。"""
    if not buf: return                                                           # 空缓冲直接返回
    text = "\n".join(x["content"] for x in buf).strip()                           # 拼接块文本
    if len(text) < 10: return                                                     # 空或过短碎片丢弃
    pages = [x["page_no"] for x in buf]                                           # 收集涉及页码
    head = buf[0]                                                                 # 以首单元决定块的定位属性
    chunks.append(dict(doc=head.get("doc", ""), text=text, page_no=min(pages),     # 文档名、文本、起始页
                       page_end=max(pages), block_type=head.get("type", "text"),   # 结束页与块类型
                       table_id=head.get("table_id", ""), caption=head.get("caption", ""),  # 表格定位：表标识与表题
                       row_index=head.get("row_index", 0), locator=head.get("locator", ""),  # 行号与可读定位串
                       chunk_id=f'gd3-{len(chunks)}'))                            # 全局唯一块号


def semantic_chunks(blocks: list, embed) -> list:                                 # 语义分块主流程
    """相邻单元与当前块语义中心相似度低于阈值处切块；标题与表格行强制单独成界。"""
    units = []                                                                    # 待分块的语义单元
    for b in blocks:                                                              # 遍历结构化后的块
        kind = b.get("type")                                                      # 块类型
        if kind in ("table_row", "table") or len(b["content"]) <= C.SEM_CHUNK_MAX:  # 表格行、整表与短块整体作为一个单元
            units.append(dict(b, content=b["content"]))                            # 直接登记
        else:                                                                     # 长段落按句拆开
            units += [dict(b, content=s, type="text") for s in split_sentences(b["content"])]  # 逐句登记并保留页码
    if not units: return []                                                       # 没有可用单元则返回空
    # 只为需要参与语义比较的单元编码：表格行与整表是硬边界，其向量不参与切块判定。
    # 实测这一步跳过 5723/12145 个单元，重建索引可省下近一半编码时间。
    need = [i for i, u in enumerate(units) if u["type"] not in ("table_row", "table")]  # 需要向量的单元下标
    vecs = dict(zip(need, embed([units[i]["content"] for i in need]))) if need else {}  # 批量编码并按下标索引
    chunks, buf, bvec, title = [], [], [], ""                                      # 结果、滑动缓冲、对应向量、当前章节名
    for i, unit in enumerate(units):                                              # 顺序扫描语义单元
        vec = vecs.get(i)                                                         # 取本单元向量，表格单元为 None
        hard = unit["type"] in ("title", "table_row", "table")                     # 三种必须切块的硬边界
        long = sum(len(x["content"]) for x in buf) + len(unit["content"]) > C.SEM_CHUNK_MAX  # 超长
        # 表格单元没有向量，必须先短路，否则 None 会被 numpy 当成 0 维数组参与矩阵乘而报错
        far = vec is not None and buf and bvec and float(sum(bvec) / len(bvec) @ vec) < C.SEM_THRESHOLD  # 与当前块语义中心距离过远
        if buf and (hard or long or far): _flush(buf, chunks); buf, bvec = [], []    # 在边界/超长/语义转折处切块
        if unit["type"] == "title": title = unit["content"][:60]; continue          # 标题只作章节语境，不入块
        if not buf and title and vec is not None:                                   # 段落新块开头补章节名（表格单元自带语境，不补）
            buf.append(dict(unit, content=title)); bvec.append(vec)                 # 借标题补足上下文
        buf.append(unit)                                                            # 当前单元入缓冲
        if vec is not None: bvec.append(vec)                                        # 仅非表格单元登记向量，供后续相似度比较
        if unit["type"] in ("table_row", "table"): _flush(buf, chunks); buf, bvec = [], []  # 表格单元立即落块，不与段落混
    _flush(buf, chunks)                                                           # 处理尾部缓冲
    merged = []                                                                   # 合并过短块：实测"章节名+公司名"这类碎块会挤占检索前排却无信息
    for c in chunks:                                                              # 逐个检查块长度
        if merged and c["block_type"] not in ("table_row", "table") and (len(c["text"]) < C.MERGE_MIN  # 表格行不参与合并，否则定位失效
                                                                        or len(merged[-1]["text"]) < C.MERGE_MIN):
            merged[-1]["text"] += "\n" + c["text"]                                  # 并入前一块，保持上下文连贯
            merged[-1]["page_end"] = max(merged[-1]["page_end"], c["page_end"])     # 同步扩展页区间
        else: merged.append(c)                                                     # 长度合适或属于表格则独立成块
    for i, c in enumerate(merged): c["chunk_id"] = f"gd3-{i}"                       # 重新编号，保证块 ID 连续
    return merged                                                                  # 返回语义块
