# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：关键词检索层。BM25 与向量检索互补——向量负责语义，BM25 负责"专有名词、数字、股票代码"这类精确匹配。
import pickle                                                                   # 索引持久化
import re                                                                       # 分词前清洗
from typing import List, Dict, Optional, Tuple                                  # 类型标注

from . import config as C                                                       # 复用集中配置

_TOKEN_LATIN = re.compile(r"[a-zA-Z]+")                                         # 连续拉丁字母（英文词）
_TOKEN_NUM = re.compile(r"\d[\d,\.]*")                                          # 数字（含千分位与小数点）
_JIEBA = None                                                                   # jieba 单例


def _jieba():                                                                   # 懒加载 jieba
    """首次调用加载词典，避免 import 期就付出几百毫秒。"""
    global _JIEBA                                                               # 声明单例
    if _JIEBA is None:                                                          # 尚未加载
        import jieba                                                            # 延迟导入
        jieba.setLogLevel(60)                                                   # 关掉 jieba 的日志噪声
        _JIEBA = jieba                                                          # 常驻
    return _JIEBA                                                               # 返回


def tokenize(text: str) -> List[str]:                                           # 中英混合分词
    """中文走 jieba，英文与数字单独抽出并统一小写；三者合并后既是索引单元也是查询单元。"""
    text = text or ""                                                           # 防空
    toks: List[str] = []                                                        # 结果
    for w in _jieba().lcut(text):                                               # jieba 分词
        w = w.strip()                                                           # 去空白
        if not w or w in " \t\n　|，,。.；;：:（）()【】[]《》<>%￥$":                # 丢掉纯标点与空白
            continue                                                            # 跳过
        toks.append(w.lower())                                                  # 统一小写，保证大小写不敏感
    for w in _TOKEN_LATIN.findall(text):                                        # 补抽英文词（jieba 对长英文串切得不好）
        toks.append(w.lower())                                                  # 追加小写词
    for w in _TOKEN_NUM.findall(text):                                          # 补抽数字
        toks.append(w.lower())                                                  # 追加；数字是财报类问题的关键线索
        toks.append(w.replace(",", "").lower())                                 # 同时收录去千分位形态，提高"14414.16"与"14,414.16"的匹配率
    return toks                                                                 # 返回


def build(chunks: List[Dict]):                                                  # 构建 BM25 索引
    """用块文本建索引；返回 (BM25 对象, 块列表) 二元组，两者必须同序一一对应。"""
    from rank_bm25 import BM25Okapi                                             # 延迟导入
    corpus = [tokenize(c.get("text", "")) for c in chunks]                      # 逐块分词
    return BM25Okapi(corpus), chunks                                            # 建索引


def save(index, chunks: List[Dict], path=None) -> None:                         # 落盘
    """索引持久化，重启免于重建（重建要先跑一遍分词，数千块约需数秒）。"""
    path = path or C.BM25_FILE                                                  # 默认路径
    path.parent.mkdir(parents=True, exist_ok=True)                              # 确保目录存在
    with open(path, "wb") as f:                                                 # 二进制写入
        pickle.dump({"index": index, "chunks": chunks}, f)                      # 一起存，保证同序


def load(path=None) -> Tuple[Optional[object], List[Dict]]:                     # 读盘
    """读取索引；文件缺失或损坏时返回 (None, [])，由调用方回退到重建。"""
    path = path or C.BM25_FILE                                                  # 默认路径
    if not path.exists():                                                       # 没有索引文件
        return None, []                                                         # 通知调用方重建
    try:                                                                        # 读盘可能因版本不符失败
        with open(path, "rb") as f:                                             # 二进制读取
            obj = pickle.load(f)                                                # 反序列化
        return obj.get("index"), obj.get("chunks", [])                          # 返回
    except Exception as exc:                                                    # 文件损坏等
        print(f"[BM25] 索引读取失败：{type(exc).__name__}: {exc}")               # 记录
        return None, []                                                         # 回退重建


def search(index, chunks: List[Dict], query: str, top_k: int) -> List[Dict]:    # 关键词检索
    """返回命中的 top_k 个块（附 bm25_score）；索引缺失时返回空列表，由上层降级到纯向量。"""
    if index is None or not chunks:                                             # 索引不可用
        print("[检索降级] BM25 索引不可用，关键词通道返回空")                      # 明确报因
        return []                                                               # 返回空
    q = tokenize(query)                                                         # 查询分词
    if not q:                                                                   # 查询无有效词
        return []                                                               # 返回空
    try:                                                                        # 打分可能因维度不符失败
        scores = index.get_scores(q)                                            # BM25 打分
    except Exception as exc:                                                    # 索引与查询不匹配
        print(f"[检索降级] BM25 打分失败：{type(exc).__name__}: {exc}")          # 明确报因
        return []                                                               # 返回空
    import numpy as np                                                          # 延迟导入
    order = np.argsort(scores)[::-1][:top_k]                                    # 取分数最高的若干
    out: List[Dict] = []                                                        # 结果
    for i in order:                                                             # 逐个组装
        s = float(scores[i])                                                    # 该块得分
        if s <= 0:                                                              # 零分说明一个词都没命中
            continue                                                            # 丢弃，避免噪声进重排
        item = dict(chunks[int(i)])                                             # 复制块字段
        item["bm25_score"] = s                                                  # 附上关键词得分
        out.append(item)                                                        # 收集
    return out                                                                  # 返回
