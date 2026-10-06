# -*- coding: utf-8 -*-
"""离线数据增强模块：删低质、语义去重、生成摘要、构建父子块，产出 enriched_chunks.json。"""

import json                                   # 导入 json，用于读写解析结果
import re                                     # 导入 re，用于分句与字符占比统计
import time                                   # 导入 time，用于统计耗时
from pathlib import Path                      # 导入 Path，用于路径处理

import config                                 # 导入配置模块，各阈值从这里读
import vector_store                           # 导入向量库模块，复用 BGE-M3 向量化
from logger import get_logger                 # 导入日志工具，用于记录增强过程

logger = get_logger("enrich")                 # 创建本模块的 logger 实例

DATA_DIR = Path(__file__).resolve().parent / "data"          # 数据目录
IN_JSON = DATA_DIR / "parsed_chunks.json"                    # 输入：第 2 步的解析结果
OUT_JSON = DATA_DIR / "enriched_chunks.json"                 # 输出：增强后的结果
MINHASH_THRESHOLD_COUNT = 1000                               # 超过该条数改用 MinHash 近似去重
MINHASH_SIGNATURE = 32                                       # MinHash 签名长度
SHINGLE_SIZE = 4                                             # MinHash 的 shingle 字符数
SENTENCE_SPLIT = re.compile(r"[。！？；]")                    # 中文分句符
KEY_SENTENCE_TOP = 1                                         # 摘要里取频次最高的前几句


# ===================== 一、摘要生成 =====================

def _split_sentences(text: str) -> list:      # 内部函数：中文分句
    """按中文句末标点把文本切成句子，保留标点，去掉空白句。"""
    parts = SENTENCE_SPLIT.split(text)         # 按标点切分
    return [p.strip() for p in parts if p and p.strip()]   # 去掉空句后返回


def generate_summary(text: str, max_len: int = 120) -> str:
    """生成摘要（规则版，不调 LLM）：第一句 + 出现频次最高的关键句，截断到 max_len。"""
    if not text or not text.strip():           # 空文本
        return ""                              # 返回空串
    content = text.strip()                     # 去空白
    if len(content) <= max_len:                # 本身就很短
        return content                         # 直接当摘要
    sentences = _split_sentences(content)      # 分句
    if not sentences:                          # 分不出句子
        return content[:max_len]               # 直接截断
    first = sentences[0]                       # 第一句通常点明主题，必取
    counts = {}                                # 统计词频，用于挑关键句
    for sentence in sentences:                 # 逐句统计
        for token in re.findall(r"[一-龥]{2,4}", sentence):   # 取 2~4 字中文片段当词
            counts[token] = counts.get(token, 0) + 1   # 累计出现次数
    ranked = []                                # 保存带分数的句子
    for idx, sentence in enumerate(sentences[1:], start=1):   # 从第二句开始挑关键句
        score = sum(counts.get(t, 0) for t in re.findall(r"[一-龥]{2,4}", sentence))   # 句内词频和
        ranked.append((score / (idx + 1), sentence))          # 除以位置，避免偏向靠后的长句
    picks = [first]                            # 摘要以第一句开头
    for _, sentence in sorted(ranked, reverse=True)[:KEY_SENTENCE_TOP]:   # 取频次最高的关键句
        if sentence not in picks:              # 避免与第一句重复
            picks.append(sentence)             # 加入摘要
    summary = "。".join(picks).strip("。")      # 用句号拼接
    if len(summary) > max_len:                 # 超过长度上限
        summary = summary[:max_len]            # 截断
    return summary                             # 返回摘要


# ===================== 二、删除低质量块 =====================

def _digit_ratio(text: str) -> float:          # 内部函数：数字占比
    """统计文本中数字字符的占比，用于识别纯表格数字块。"""
    if not text:                               # 空文本
        return 0.0                             # 占比为 0
    digits = sum(1 for ch in text if ch.isdigit())   # 数字字符数
    return digits / len(text)                  # 返回占比


def _space_ratio(text: str) -> float:          # 内部函数：空白占比
    """统计文本中空白字符的占比，用于识别 PDF 排版残留。"""
    if not text:                               # 空文本
        return 0.0                             # 占比为 0
    spaces = sum(1 for ch in text if ch.isspace())   # 空白字符数
    return spaces / len(text)                  # 返回占比


def remove_low_quality(chunks: list, min_len: int = None, max_len: int = None) -> tuple:
    """删除低质量块：过短、过长、空白占比过高、数字占比过高；返回（保留列表, 丢弃数）。"""
    low = config.ENRICH_MIN_LEN if min_len is None else min_len      # 长度下限
    high = config.ENRICH_MAX_LEN if max_len is None else max_len     # 长度上限
    kept = []                                  # 保存保留的块
    dropped = 0                                # 丢弃计数
    for chunk in chunks:                       # 逐块检查
        text = (chunk.get("text") or "").strip()   # 取出正文
        if not text:                           # 空文本
            dropped += 1                       # 计入丢弃
            continue                           # 跳过
        if len(text) < low:                    # 过短，多为断句碎片
            dropped += 1                       # 计入丢弃
            continue                           # 跳过
        if len(text) > high:                   # 过长，异常块
            dropped += 1                       # 计入丢弃
            continue                           # 跳过
        if _space_ratio(text) > 0.5:           # 空白占比超过一半
            dropped += 1                       # 计入丢弃
            continue                           # 跳过
        if _digit_ratio(text) > 0.7:           # 数字占比超过七成
            dropped += 1                       # 计入丢弃
            continue                           # 跳过
        kept.append(chunk)                     # 通过全部检查，保留
    logger.info("删除低质块：%d → %d，丢弃 %d", len(chunks), len(kept), dropped)   # 记录日志
    return kept, dropped                       # 返回保留列表与丢弃数


# ===================== 三、语义去重 =====================

def _minhash_signature(text: str) -> list:     # 内部函数：MinHash 签名
    """用字符 shingle 的哈希最小值构造 MinHash 签名，用于大规模近似去重。"""
    shingles = set()                           # 保存 shingle 的哈希值
    for i in range(max(len(text) - SHINGLE_SIZE + 1, 1)):   # 滑动窗口
        piece = text[i:i + SHINGLE_SIZE]       # 取一个 shingle
        shingles.add(hash(piece))              # 记录哈希
    if not shingles:                           # 极短文本
        return [0] * MINHASH_SIGNATURE         # 返回全零签名
    signature = []                             # 保存签名
    for seed in range(MINHASH_SIGNATURE):      # 用不同种子生成多个最小值
        signature.append(min((h ^ seed) for h in shingles))   # 取异或后的最小值
    return signature                           # 返回签名列表


def _signature_similarity(sig_a: list, sig_b: list) -> float:   # 内部函数：签名相似度
    """比较两个 MinHash 签名的相同位置比例，作为 Jaccard 相似度的估计。"""
    same = sum(1 for a, b in zip(sig_a, sig_b) if a == b)   # 相同位置计数
    return same / len(sig_a) if sig_a else 0.0   # 返回比例


def dedup_chunks(chunks: list, threshold: float = None) -> tuple:
    """语义去重：短文本优先保留，相似度超过阈值的只留一条；返回（去重列表, 丢弃数）。"""
    limit = config.DEDUP_THRESHOLD if threshold is None else threshold   # 相似度阈值
    ordered = sorted(chunks, key=lambda c: len(c.get("text") or ""))     # 按长度升序，短的优先
    if len(ordered) > MINHASH_THRESHOLD_COUNT:   # 条数过多时改用近似算法
        logger.info("块数 %d 超过 %d，改用 MinHash 近似去重", len(ordered), MINHASH_THRESHOLD_COUNT)   # 日志
        return _dedup_by_minhash(ordered, limit)   # 走近似分支
    texts = [(c.get("text") or "").strip() for c in ordered]   # 取出全部正文
    vectors = vector_store.embed_texts(texts)[0] if texts else []   # 批量向量化（只取 dense）
    kept = []                                  # 保留的块
    kept_vectors = []                          # 保留块的向量
    dropped = 0                                # 丢弃计数
    for chunk, vector in zip(ordered, vectors):   # 逐块判重
        duplicate = False                      # 是否判定为重复
        for kept_vec in kept_vectors:          # 与已保留的逐一比较
            if _cosine(vector, kept_vec) > limit:   # 相似度超过阈值
                duplicate = True               # 判定重复
                break                          # 不再比后面的
        if duplicate:                          # 是重复
            dropped += 1                       # 计入丢弃
            continue                           # 跳过
        kept.append(chunk)                     # 保留
        kept_vectors.append(vector)            # 同步记录向量
    logger.info("语义去重：%d → %d，丢弃 %d（阈值 %.2f）", len(chunks), len(kept), dropped, limit)   # 日志
    return kept, dropped                       # 返回结果


def _cosine(vec_a: list, vec_b: list) -> float:   # 内部函数：余弦相似度
    """计算两个向量的余弦相似度，零向量返回 0。"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))          # 点积
    norm_a = sum(a * a for a in vec_a) ** 0.5               # A 的模
    norm_b = sum(b * b for b in vec_b) ** 0.5               # B 的模
    if norm_a == 0 or norm_b == 0:             # 有零向量
        return 0.0                             # 相似度为 0
    return dot / (norm_a * norm_b)             # 返回余弦值


def _dedup_by_minhash(chunks: list, threshold: float) -> tuple:   # 内部函数：近似去重
    """用 MinHash 签名做大规模近似去重，返回（去重列表, 丢弃数）。"""
    kept = []                                  # 保留的块
    signatures = []                            # 保留块的签名
    dropped = 0                                # 丢弃计数
    for chunk in chunks:                       # 逐块处理
        text = (chunk.get("text") or "").strip()   # 取出正文
        sig = _minhash_signature(text)         # 计算签名
        if any(_signature_similarity(sig, s) > threshold for s in signatures):   # 与已保留的比较
            dropped += 1                       # 判定重复
            continue                           # 跳过
        kept.append(chunk)                     # 保留
        signatures.append(sig)                 # 记录签名
    logger.info("MinHash 近似去重：%d → %d，丢弃 %d", len(chunks), len(kept), dropped)   # 记录日志
    return kept, dropped                       # 返回结果


# ===================== 四、父子块 =====================

def build_parent_child(chunks: list, parent_size: int = None) -> list:
    """构造父子块：每 parent_size 个子块合成一个父块，返回父块与子块混合的列表。"""
    size = config.PARENT_SIZE if parent_size is None else parent_size   # 每个父块包含的子块数
    result = []                                # 保存最终结果
    parent_index = 0                           # 父块在结果列表中的索引
    for start in range(0, len(chunks), size):  # 按 size 分组
        group = chunks[start:start + size]     # 取出这一组子块
        child_ids = []                         # 该父块包含的子块编号
        for child in group:                    # 遍历组内子块
            child_id = len(result) + 1         # 子块将放在父块之后，先占位计算
            child["is_parent"] = False         # 标记为子块
            child["parent_id"] = parent_index  # 记录所属父块索引
            child["child_id"] = child_id       # 子块自身编号
            child_ids.append(child_id)         # 收进父块的子块列表
            result.append(child)               # 子块加进结果
        parent = {                             # 组装父块
            "chunk_id": parent_index,          # 父块编号
            "text": "".join((c.get("text") or "") for c in group),   # 父块文本为子块拼接
            "source": group[0].get("source", "") if group else "",   # 沿用第一个子块的来源
            "page": group[0].get("page", 0) if group else 0,         # 沿用第一个子块的页码
            "is_parent": True,                 # 标记为父块
            "child_ids": child_ids,            # 包含的子块编号
            "parent_id": -1,                   # 父块自身没有父块
            "summary": "",                     # 父块不单独生成摘要
        }                                      # 父块组装结束
        result.insert(parent_index, parent)    # 把父块插到它那组子块之前
        parent_index += len(group) + 1         # 下一组父块的索引越过本组父块与子块
    logger.info("父子块构建：%d 个子块 → %d 个父块，共 %d 条记录",
                len(chunks), (len(chunks) + size - 1) // size, len(result))   # 记录日志
    return result                              # 返回父块与子块混合列表


# ===================== 五、组合流程与命令行入口 =====================

def _flatten(json_path: Path) -> list:         # 内部函数：摊平解析结果
    """读取解析结果 JSON，把所有 PDF 的块摊平成一个列表。"""
    payload = json.loads(json_path.read_text(encoding="utf-8"))   # 读取并解析
    chunks = []                                # 保存摊平结果
    for result in payload.get("results", []):  # 遍历每个 PDF
        for item in result.get("chunks", []):  # 遍历每个块
            item.setdefault("source", result.get("source", ""))   # 补齐来源字段
            chunks.append(item)                # 收进列表
    return chunks                              # 返回摊平结果


def enrich_all(json_path: str = None, out_path: str = None) -> dict:
    """组合流程：删低质 → 去重 → 生成摘要 → 构建父子块 → 写出 JSON，返回各步数量。"""
    src = Path(json_path) if json_path else IN_JSON      # 输入路径
    dst = Path(out_path) if out_path else OUT_JSON       # 输出路径
    chunks = _flatten(src)                     # 读取并摊平
    before = len(chunks)                       # 原始数量
    kept, low_dropped = remove_low_quality(chunks)        # 第一步：删低质
    after_low = len(kept)                      # 删低质后数量
    kept, dup_dropped = dedup_chunks(kept)     # 第二步：语义去重
    after_dedup = len(kept)                    # 去重后数量
    for chunk in kept:                         # 第三步：逐块生成摘要
        chunk["summary"] = generate_summary(chunk.get("text") or "", 120)   # 写入 summary 字段
    combined = build_parent_child(kept)        # 第四步：构建父子块
    payload = {                                # 组装输出内容
        "generated_at": int(time.time()),      # 生成时间戳
        "stats": {"before": before, "after_low_quality": after_low,
                  "after_dedup": after_dedup, "final": len(combined),
                  "low_quality_dropped": low_dropped, "duplicate_dropped": dup_dropped},   # 各步统计
        "chunks": combined,                    # 父子块混合列表
    }                                          # 输出内容组装结束
    dst.parent.mkdir(parents=True, exist_ok=True)   # 确保输出目录存在
    dst.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")   # 写出 JSON
    logger.info("增强结果已写入：%s", dst)      # 记录日志
    return payload["stats"]                    # 返回各步数量


def main() -> None:                            # 命令行入口
    """命令行入口：执行离线增强并打印每一步的数量变化。"""
    start = time.time()                        # 记录耗时起点
    print("=" * 70)                            # 打印分隔线
    print("离线数据增强（enrich）")              # 打印标题
    print("-" * 70)                            # 打印分隔线
    stats = enrich_all()                       # 执行增强
    print(f"  原始块数　　　：{stats['before']}")            # 打印原始数量
    print(f"  删低质后　　　：{stats['after_low_quality']}（丢弃 {stats['low_quality_dropped']}）")   # 打印
    print(f"  语义去重后　　：{stats['after_dedup']}（丢弃 {stats['duplicate_dropped']}）")        # 打印
    print(f"  父子块最终条数：{stats['final']}")             # 打印最终数量
    print("-" * 70)                            # 打印分隔线
    print(f"输出文件：{OUT_JSON}")              # 打印输出路径
    print(f"耗时：{time.time() - start:.1f} 秒")   # 打印耗时
    print("=" * 70)                            # 打印分隔线


if __name__ == "__main__":                     # 支持 python -m enrich 直接运行
    main()                                     # 执行命令行入口
