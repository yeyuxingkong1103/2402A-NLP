# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：语义分块。工单要求"按标题、段落、句子语义相似度分块，chunk_size=512，overlap=80"。
#           做法：块内按句切分 → 用 bge-m3 逐句编码 → 与当前块中心的余弦相似度低于阈值或超过长度上限即切块。
import re                                                                       # 正则：断句与标题识别
from typing import List, Dict, Tuple, Optional                                  # 类型标注

import numpy as np                                                              # 向量运算

from . import config as C                                                       # 复用集中配置

_SENT_END = re.compile(r"(?<=[。！？；!?;])")                                     # 中文与英文的句末标点（保留标点）
_TITLE_LIKE = re.compile(r"^(第[一二三四五六七八九十百]+[章节节]|[（(][一二三四五六七八九十]+[）)]|\d+(\.\d+)*[、.]?\s*)")  # 章节/条号标题形态
# 坑（实测）：招股书里"4、法定代表人：赵马克"这类"条号+键值"的一行，形态上完全符合标题正则，
#   若按标题丢弃，就会把"某公司的法定代表人是某人"这种关键事实从索引里抹掉，
#   导致问"某公司法定代表人是谁"时检索不到正确依据，只能召回其他公司的同名条目而答错。
#   故：含冒号的短行不当作标题，一律保留为可检索内容。
_KEY_VALUE = re.compile(r"[：:]")                                                # 键值分隔符
_BLANK = re.compile(r"\n{2,}")                                                  # 连续空行


def split_sentences(text: str, max_len: int = 200) -> List[str]:                # 断句
    """先按句末标点切，再对超长句按逗号二次切分；保证送入编码器的单元不会过长。"""
    parts: List[str] = []                                                       # 结果
    for seg in _SENT_END.split(text or ""):                                     # 按句末标点粗切
        seg = seg.strip()                                                       # 去首尾空白
        if not seg:                                                             # 空段跳过
            continue                                                            # 继续
        if len(seg) <= max_len:                                                 # 长度合适
            parts.append(seg)                                                   # 直接收下
            continue                                                            # 下一个
        buf = ""                                                                # 超长句先按逗号攒
        for piece in re.split(r"(?<=[，,])", seg):                              # 逗号后切分
            if len(buf) + len(piece) > max_len and buf:                         # 攒够了
                parts.append(buf)                                               # 收下一段
                buf = piece                                                     # 另起一段
            else:                                                               # 还没攒够
                buf += piece                                                    # 继续攒
        if buf:                                                                 # 收尾
            parts.append(buf)                                                   # 收下剩余
    return parts                                                                # 返回句列表


def _cos(a: np.ndarray, b: np.ndarray) -> float:                                # 余弦相似度
    """bge-m3 输出已归一化，内积即余弦；此处仍做一次防零除保护。"""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))                        # 模长乘积
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0                 # 防零除


def semantic_split(text: str, encode) -> List[Tuple[str, float]]:               # 单块文本 → 语义子块
    """返回 [(子块文本, 该块与相邻块的最小相似度)]；相似度用作切块依据，便于事后复盘阈值。"""
    sents = split_sentences(text)                                               # 断句
    if not sents:                                                               # 没有内容
        return []                                                               # 返回空
    if len(text) <= C.CHUNK_SIZE and len(sents) <= 3:                           # 短文本不必拆
        return [(text.strip(), 1.0)]                                            # 整块返回，相似度记 1
    vecs = encode(sents)                                                        # 批量编码所有句子
    chunks: List[Tuple[str, float]] = []                                        # 结果
    cur: List[str] = [sents[0]]                                                 # 当前块
    cur_len = len(sents[0])                                                     # 当前块长度
    sims: List[float] = []                                                      # 当前块内相邻句相似度
    for i in range(1, len(sents)):                                              # 从第二句开始
        s = sents[i]                                                            # 当前句
        center = np.mean(vecs[max(0, i - len(cur)):i], axis=0)                  # 当前块的语义中心（用已入块的句向量均值）
        sim = _cos(vecs[i], center)                                             # 新句与中心的相似度
        too_long = cur_len + len(s) > C.SEM_MAX_CHUNK                           # 触到硬上限
        too_big = cur_len + len(s) > C.CHUNK_SIZE                               # 触到目标长度
        off_topic = sim < C.SEM_THRESHOLD                                       # 语义漂移
        if (too_long or (too_big and off_topic)) and cur:                       # 满足切块条件
            body = "".join(cur).strip()                                         # 生成本块文本
            if body:                                                            # 非空才收
                chunks.append((body, min(sims) if sims else 1.0))               # 记录块与其最小句间相似度
            tail = body[-C.CHUNK_OVERLAP:] if C.CHUNK_OVERLAP > 0 else ""       # 取上一个块尾部作为重叠前缀
            cur = [tail + s] if tail else [s]                                   # 新块以重叠开头
            cur_len = len(cur[0])                                               # 重置长度
            sims = []                                                           # 重置相似度统计
        else:                                                                   # 不切
            cur.append(s)                                                       # 并入当前块
            cur_len += len(s)                                                   # 累加长度
            sims.append(sim)                                                    # 记录相似度
    body = "".join(cur).strip()                                                 # 收尾块
    if body:                                                                    # 非空才收
        chunks.append((body, min(sims) if sims else 1.0))                       # 记录
    return _merge_small(chunks)                                                 # 过短的块向后合并


def _merge_small(chunks: List[Tuple[str, float]]) -> List[Tuple[str, float]]:   # 合并碎片块
    """短于 SEM_MIN_CHUNK 的块向后合并，避免碎片挤占检索前排（实测碎片块极易被误召回）。"""
    out: List[Tuple[str, float]] = []                                           # 结果
    for body, sim in chunks:                                                    # 逐块
        if out and len(out[-1][0]) < C.SEM_MIN_CHUNK and len(out[-1][0]) + len(body) <= C.SEM_MAX_CHUNK:  # 上一块过短且合并后不超限
            prev_body, prev_sim = out[-1]                                       # 取上一块
            out[-1] = (prev_body + body, min(prev_sim, sim))                    # 合并并保留较小的相似度
        else:                                                                   # 否则
            out.append((body, sim))                                             # 独立成块
    return out                                                                  # 返回


def is_heading(text: str) -> bool:                                              # 判断是否为章节标题行
    """标题 = 章节/条号形态 + 够短 + 不含冒号。

    含冒号的短行（如"4、法定代表人：赵马克"）是键值内容，绝不能当标题丢弃；
    这条规则是实测踩坑后加的：丢掉它会让"某公司法定代表人是某人"这类事实从索引里消失。
    """
    t = (text or "").strip()                                                    # 去空白
    return bool(t) and len(t) < 60 and bool(_TITLE_LIKE.match(t)) and not _KEY_VALUE.search(t)  # 三个条件同时满足


def merge_short_blocks(blocks: List[Dict], min_len: int = None,
                       max_len: int = None) -> List[Dict]:
    """把**连续的同页短文本块**合并成一段。

    为什么要合并：招股书的"公司概况""本次发行概况"是以多个极短行给出的，
    例如 `1、公司名称：武汉力源信息技术股份有限公司` / `4、法定代表人：赵马克`。
    这些行各自只有 10~30 字：
      · 单独成块时，向量被稀释、BM25 也只有一两个词命中，重排得分很低；
      · 且行内不含公司全称的块（如"4、法定代表人：赵马克"）更无法与"某公司的法定代表人是谁"对齐。
    实测后果：问"武汉力源信息技术股份有限公司的法定代表人是谁"时，含正确依据的块排不进重排前 8，
    系统只能召回**其他公司**的同名条目（某关联公司的"法定代表人胡戎"）而答错。
    合并后，这些行连成一段带公司全称的完整概况，即可被稳定召回。
    """
    min_len = min_len or C.SEM_MIN_CHUNK                                        # 判定"短块"的门槛
    max_len = max_len or C.CHUNK_SIZE                                           # 合并后的上限
    out: List[Dict] = []                                                        # 结果
    buf: Optional[Dict] = None                                                  # 正在累积的块
    for b in blocks:                                                            # 逐块扫描
        # 标题行必须单独成块，不能被并入短块——否则 chunk_blocks 里的"章节语境"就再也更新不到，
        # 图片块会挂着几页之前的陈旧章节名（实测：组织结构图被标成"(五)机构独立"，重排因此把它排到第 10 名开外）。
        short_text = (b.get("block_type") == "text"                            # 只合并正文类
                      and len((b.get("text") or "").strip()) < min_len          # 且本身很短
                      and not is_heading(b.get("text")))                        # 且不是标题行
        if not short_text:                                                      # 不是短正文：先把缓冲冲刷掉
            if buf:                                                             # 有缓冲
                out.append(buf); buf = None                                     # 收下
            out.append(b)                                                       # 原样保留该块（表格/图片/长正文）
            continue                                                            # 下一个
        same_page = buf is not None and buf.get("page_no") == b.get("page_no")   # 必须同页，避免跨页捏造归属
        fits = buf is not None and len(buf["text"]) + len(b["text"]) + 1 <= max_len  # 合并后不超上限
        if buf is not None and same_page and fits:                              # 可以并入缓冲
            buf["text"] = buf["text"].rstrip() + "\n" + b["text"].strip()        # 追加一行
        else:                                                                   # 不能并入
            if buf:                                                             # 冲刷旧缓冲
                out.append(buf)                                                 # 收下
            buf = dict(b)                                                       # 以当前块开新缓冲
            buf["text"] = (b.get("text") or "").strip()                         # 复制文本
    if buf:                                                                     # 收尾
        out.append(buf)                                                         # 收下最后一个缓冲
    return out                                                                  # 返回合并后的块列表


def split_atomic(text: str, max_len: int = None) -> List[str]:                  # 拆分超长的表格/图像块
    """按行把超长原子块切成多段，并为每段补上首行（表头）作为语境。

    为什么要拆：整张长表作为一个块有两个代价——
      1) 检索粒度差：问的是表中某一行，却只能召回整张表；
      2) 向量化慢：bge-m3 的最长序列是 8192 token，超长块会让成批编码严重变慢（实测长表块是入库耗时的主要来源）。
    保留首行作前缀，是为了让每一段都还能"自证"它属于哪张表。
    """
    max_len = max_len or C.SEM_MAX_CHUNK                                        # 单段上限
    if len(text) <= max_len:                                                    # 没超限
        return [text]                                                           # 原样返回
    lines = [l for l in (text or "").split("\n") if l.strip()]                  # 按行切分并去掉空行
    if len(lines) <= 1:                                                         # 只有一行（超长单行）
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]      # 按字符硬切
    header = lines[0][:200]                                                     # 首行（通常是表头）截断后作语境
    out: List[str] = []                                                         # 结果
    buf = ""                                                                    # 当前段
    for line in lines[1:]:                                                      # 从第二行开始
        piece = line[:max_len]                                                  # 单行本身过长时也要截断
        if buf and len(buf) + len(piece) + 1 > max_len:                         # 再加就超限
            out.append(buf)                                                     # 收下当前段
            buf = ""                                                            # 另起一段
        buf = f"{buf}\n{piece}" if buf else piece                               # 拼入当前段
    if buf:                                                                     # 收尾
        out.append(buf)                                                         # 收下剩余
    return [f"{header}\n{seg}" if header and not seg.startswith(header) else seg  # 补表头前缀
            for seg in out]                                                     # 逐段处理


def chunk_blocks(blocks: List[Dict], encode) -> List[Dict]:                     # 块列表 → 检索用分块
    """把解析出的原始块转成带定位信息的分块；表格与图片文本保持整块，不做语义切分。"""
    chunks: List[Dict] = []                                                     # 结果
    section = ""                                                                # 当前章节标题，作为语境前缀
    counter = 0                                                                 # 全局块号计数器
    # 先合并连续的同页短文本行，避免"4、法定代表人：赵马克"这类短行单独成块而被重排淹没。
    for b in merge_short_blocks(blocks):                                        # 遍历合并后的块
        btype = b.get("block_type", "text")                                     # 块类型
        text = (b.get("text") or "").strip()                                    # 块文本
        page = int(b.get("page_no", 0))                                         # 页码
        if btype in ("header", "page_number") and text:                         # 页眉页脚
            continue                                                            # 直接丢弃，纯噪声
        if btype == "text" and is_heading(text):                                # 是标题行（判定口径见 is_heading）
            section = text                                                      # 更新章节语境
            continue                                                            # 标题本身不进检索（避免只命中标题答不出内容）
        if not text:                                                            # 无文本（空图等）
            continue                                                            # 跳过
        prefix = f"【{b.get('doc', '')}·{section}】" if section else f"【{b.get('doc', '')}】"  # 语境前缀：把章节名带进块
        if btype in ("table", "image", "chart"):                                # 表格与图像文本按整块处理
            pieces = [(seg, 1.0) for seg in split_atomic(text)]                 # 不做语义切分，但超长时按行拆分
            atomic = True                                                       # 标记为原子块
        else:                                                                   # 正文
            pieces = semantic_split(text, encode)                               # 语义切分
            atomic = False                                                      # 非原子
        for body, sim in pieces:                                                # 逐个子块
            if not body.strip():                                                # 空块
                continue                                                        # 跳过
            counter += 1                                                        # 递增块号
            chunks.append({                                                     # 组装分块
                "chunk_id": f"{b.get('doc', 'doc')}-{counter:05d}",             # 全局唯一块号
                "doc": b.get("doc", ""),                                        # 来源文档名
                "text": f"{prefix}{body}",                                      # 带语境前缀的正文
                "source": b.get("doc", ""),                                     # 工单要求字段：来源
                "page_no": page,                                                # 起始页
                "page_end": page,                                               # 结束页（块内不跨页）
                "block_type": btype,                                            # 块类型
                "section": section,                                             # 所属章节
                "atomic": atomic,                                               # 是否为不可再切的原子块
                "sem_sim": round(float(sim), 4),                                # 切块时的最小句间相似度（调参依据）
                "locator": f"{b.get('doc', '')} P{page}",                       # 可读定位串，直接给前端展示
                "image_path": b.get("image_path", ""),                          # 图片来源（若为图像块）
            })                                                                  # 组装结束
    return chunks                                                               # 返回全部分块
