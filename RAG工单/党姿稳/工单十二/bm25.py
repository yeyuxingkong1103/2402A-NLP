# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：中文分词与 BM25 全文检索实现（不依赖第三方 BM25 库，保证环境兼容）

import math                                                     # 导入数学库用于对数运算
import re                                                       # 导入正则用于标点清洗
import jieba                                                    # 导入结巴分词用于中文切词
import config                                                   # 导入本工单配置模块
from logger import LOGGER                                       # 导入统一日志对象

jieba.setLogLevel(20)                                           # 关闭 jieba 的 DEBUG 日志，保持输出整洁

_PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）《》【】〈〉·,.;:?!\"'()\[\]{}<>|/\\~`@#$%^&*+=_-]+")  # 标点与空白正则


def tokenize(text):                                             # 定义分词函数
    """中英混合分词：英文数字保留原形小写，中文用 jieba 切分，过滤标点与单字噪音。"""
    text = text.lower()                                         # 统一转小写（对英文与数字有效）
    text = _PUNCT.sub(" ", text)                                # 将标点与空白替换为空格
    tokens = []                                                 # 初始化词元列表
    for tok in jieba.lcut(text):                                # 使用 jieba 精确模式切词
        tok = tok.strip()                                       # 去除词元两侧空白
        if not tok:                                             # 跳过空词元
            continue                                            # 继续下一个
        if len(tok) == 1 and not re.match(r"[a-z0-9]", tok):    # 过滤单个中文汉字（信息量过低）
            continue                                            # 继续下一个
        tokens.append(tok)                                      # 保留有效词元
    return tokens                                               # 返回词元列表


class BM25(object):                                             # 定义 BM25 检索器类
    """标准 BM25（Okapi）实现：构建倒排索引并返回 top-k 结果，支持限定候选集合。"""

    def __init__(self, k1=1.5, b=0.75):                         # 构造方法：设置 BM25 调节参数
        self.k1 = k1                                            # 词频饱和参数 k1
        self.b = b                                              # 长度归一化参数 b
        self.df = {}                                            # 词 -> 文档频次
        self.doc_len = []                                       # 每个文档的词元数量
        self.avg_len = 0.0                                      # 平均文档长度
        self.n_docs = 0                                         # 文档总数
        self.inverted = {}                                      # 词 -> {文档下标: 词频}

    def fit(self, texts):                                       # 定义索引构建方法
        """对文本列表建立倒排索引与统计量。"""
        self.n_docs = len(texts)                                # 记录文档总数
        for i, text in enumerate(texts):                        # 遍历每个文档
            toks = tokenize(text)                               # 分词
            self.doc_len.append(len(toks))                      # 记录文档长度
            seen = set()                                        # 用于统计文档频次（同文档只计一次）
            for t in toks:                                      # 遍历该文档所有词元
                self.inverted.setdefault(t, {})                 # 初始化倒排项
                self.inverted[t][i] = self.inverted[t].get(i, 0) + 1  # 累计该词在本文档的词频
                seen.add(t)                                     # 记录已见词
            for t in seen:                                      # 遍历去重后的词
                self.df[t] = self.df.get(t, 0) + 1              # 文档频次加一
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0  # 计算平均长度
        LOGGER.info("BM25 索引构建完成：%d 文档，%d 词", self.n_docs, len(self.df))  # 输出统计日志

    def _idf(self, term):                                       # 定义逆文档频率计算
        """IDF 采用带平滑的 BM25 变体，避免出现负值。"""
        df = self.df.get(term, 0)                               # 取该词的文档频次
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))  # 平滑 IDF 公式

    def search(self, query, top_k=10, allowed_idx=None):        # 定义检索方法
        """返回 [(文档下标, 分数), ...] 按分数降序；allowed_idx 可限定候选集合。"""
        q_toks = tokenize(query)                                # 查询分词
        scores = {}                                             # 文档下标 -> 累计分数
        for t in q_toks:                                        # 遍历查询词
            if t not in self.inverted:                          # 词语不在词表中则跳过
                continue                                        # 继续下一个词
            idf = self._idf(t)                                  # 计算该词的 IDF
            for idx, tf in self.inverted[t].items():            # 遍历包含该词的文档
                if allowed_idx is not None and idx not in allowed_idx:  # 限定候选时跳过无关文档
                    continue                                    # 继续下一个文档
                dl = self.doc_len[idx] or 1                     # 文档长度（防除零）
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1.0))  # BM25 分母
                scores[idx] = scores.get(idx, 0.0) + idf * tf * (self.k1 + 1) / denom  # 累加该词得分
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 按分数降序排序
        return ranked[:top_k]                                   # 返回前 top_k 条

    def save(self, path):                                       # 定义索引缓存方法
        """把检索所需的统计量序列化到磁盘，避免每次启动都重新分词。"""
        import pickle                                           # 延迟导入 pickle（仅保存时需要）
        state = {"k1": self.k1, "b": self.b, "df": self.df,     # 保存调节参数与文档频次
                 "doc_len": self.doc_len, "avg_len": self.avg_len,  # 保存文档长度与平均长度
                 "n_docs": self.n_docs, "inverted": self.inverted}  # 保存文档总数与倒排索引
        with open(path, "wb") as f:                             # 以二进制写模式打开
            pickle.dump(state, f, protocol=4)                   # 使用协议 4 兼容 Python 3.8
        LOGGER.info("BM25 索引已缓存：%s", path)                 # 输出完成日志

    @classmethod                                                # 声明类方法
    def load(cls, path):                                        # 定义索引加载方法
        """从磁盘恢复 BM25 索引；失败时返回 None 由上层重建。"""
        import pickle                                           # 延迟导入 pickle
        try:                                                    # 尝试读取缓存
            with open(path, "rb") as f:                         # 以二进制读模式打开
                state = pickle.load(f)                          # 反序列化状态字典
            obj = cls(k1=state["k1"], b=state["b"])             # 用保存的参数创建实例
            obj.df = state["df"]                                # 恢复文档频次
            obj.doc_len = state["doc_len"]                      # 恢复文档长度
            obj.avg_len = state["avg_len"]                      # 恢复平均长度
            obj.n_docs = state["n_docs"]                        # 恢复文档总数
            obj.inverted = state["inverted"]                    # 恢复倒排索引
            LOGGER.info("命中 BM25 索引缓存：%d 文档", obj.n_docs)  # 输出缓存命中日志
            return obj                                          # 返回恢复后的实例
        except Exception as e:                                  # 缓存损坏或不存在
            LOGGER.warning("BM25 索引缓存不可用，将重建：%r", e)   # 输出告警
            return None                                         # 返回 None


def build_or_load_index(chunks, force=False):                   # 定义索引构建或加载函数
    """构建（或从缓存加载）语料块的 BM25 索引。"""
    if not force:                                               # 非强制重建时先尝试读缓存
        idx = BM25.load(config.BM25_CACHE_FILE)                 # 尝试加载
        if idx is not None and idx.n_docs == len(chunks):       # 文档数一致才可用
            return idx                                          # 返回缓存的索引
    idx = BM25()                                                # 创建新的索引实例
    idx.fit([c["text"] for c in chunks])                        # 用块正文构建索引
    config.ensure_dirs()                                        # 确保数据目录存在
    idx.save(config.BM25_CACHE_FILE)                            # 落盘缓存
    return idx                                                  # 返回索引
