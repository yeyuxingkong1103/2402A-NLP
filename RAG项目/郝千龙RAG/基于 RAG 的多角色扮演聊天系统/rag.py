# -*- coding: utf-8 -*-  # 声明文件编码为 UTF-8，保证中文字符正常解析
"""【混合检索 · rag.py】RAG 检索环节：BM25 稀疏关键词召回（jieba 中文分词）+ Milvus 稠密向量余弦召回，按「BM25 + 余弦×10」分数融合；含句对模型、查询分词、索引 pickle 持久化与知识库动态更新。"""  # 模块文档字符串：中文名 + 文件名 + 一句话作用
from __future__ import annotations  # 启用 PEP 563 延迟注解，让 list[SentencePair] 等类型注解在旧版本也能用

import pickle  # 用于把 BM25 索引/句对序列化到缓存文件，加速冷启动
import re  # 正则表达式，用于英文分词 token 提取
from dataclasses import dataclass  # 数据类装饰器，自动生成 __init__ 等方法，简化句对模型
from pathlib import Path  # 面向对象的路径操作，跨平台兼容

import jieba  # 中文分词库，把中文句子切成词，供 BM25 建索引
from rank_bm25 import BM25Okapi  # BM25 算法实现，做稀疏关键词检索打分

from config import CACHE_DIR, SCORE_THRESHOLD, TOP_K, tsv_path  # 导入缓存目录、得分阈值、top_k 默认值与数据集路径解析函数
from logger import log  # 统一日志器，便于调试与运行追踪

# 英文分词正则：匹配单词（含 don't 这类撇号缩写）或纯数字，避免把标点当 token
_EN_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")  # 预编译正则，提升重复调用性能


@dataclass  # 数据类装饰器：自动生成构造函数与 __repr__，简化模型定义
class SentencePair:
    """知识库最小单元：一条中英句对（或一个文档分块）。"""  # 类文档：说明本类的语义角色

    en_id: str  # 英文句子的原始 ID（来自 TSV 第一列），用于溯源
    english: str  # 英文句子正文
    zh_id: str  # 中文句子的原始 ID（来自 TSV 第三列），用于溯源
    chinese: str  # 中文句子正文
    source: str = "tsv"     # 文档来源（数据集路径 / 上传文件路径）  # 标记本条数据出处，便于区分主数据集与动态上传
    summary: str = ""       # 摘要（入库 Milvus / 展示用）  # 简短描述，前端展示与向量库元数据共用

    def as_context(self) -> str:
        """拼装成提示词里的上下文格式。"""  # 说明：返回会被塞进大模型 prompt 的文本
        return f"English: {self.english}\nChinese: {self.chinese}"  # 固定格式输出，便于大模型对齐中英对照

    def as_text(self) -> str:
        """BM25 语料文本：文档分块取正文，句对取中英拼接。"""  # 说明：BM25 建索引时的语料字符串
        if self.english.startswith("[DOC]"):  # 约定 [DOC] 前缀表示文档分块，正文在 chinese 字段
            return self.chinese  # 文档分块：只取中文正文做 BM25 语料
        return f"{self.english} <-> {self.chinese}"  # 句对：中英拼接建索引，双向都能命中


def tokenize(text: str) -> list[str]:
    """双语分词：英文小写词/数字 + 中文 jieba 切词，供 BM25 使用。"""  # 函数职责：把查询/语料切成 token 列表
    text = text.strip()  # 先去首尾空白，避免空格干扰分词
    if not text:  # 空串直接返回空列表，避免后续无意义计算
        return []
    en = [t.lower() for t in _EN_TOKEN.findall(text)]       # 英文 token  # 正则提取英文/数字 token 并统一小写，做大小写无关匹配
    zh_source = _EN_TOKEN.sub(" ", text)                     # 先移除英文，避免中英粘连  # 把英文部分替换成空格，防止 jieba 把英文切碎
    zh = [t.strip() for t in jieba.lcut(zh_source) if t.strip() and not t.isspace()]  # jieba 全切分，过滤掉纯空白 token
    return en + zh  # 英文 token 在前、中文 token 在后，合并成统一 token 流喂给 BM25


def load_pairs(path: Path | None = None) -> list[SentencePair]:
    """加载 TSV 数据集（四列：英文ID、英文、中文ID、中文）。"""  # 函数职责：从 TSV 读全部句对
    path = path or tsv_path()  # 未传路径时用配置中的默认数据集路径
    if not path.exists():  # 数据集不存在直接抛错，避免下游空跑
        raise FileNotFoundError(f"找不到数据集: {path}")  # 明确报错信息，便于排错
    pairs: list[SentencePair] = []  # 收集解析出的句对
    with path.open(encoding="utf-8") as f:  # 用 UTF-8 打开，支持中文，with 自动关流
        for line in f:  # 逐行读取，避免大文件一次性占内存
            parts = line.rstrip("\n").split("\t")  # 去行尾换行后按制表符切四列
            if len(parts) != 4:  # 跳过格式不完整的行  # 列数不对就跳过，保证语料干净
                continue
            en_id, english, zh_id, chinese = parts  # 解构四列：英ID、英文、中ID、中文
            if english.strip() and chinese.strip():  # 中英都非空才入库，过滤空值噪声
                pairs.append(
                    SentencePair(
                        en_id,  # 英文 ID 原样保留，用于回溯
                        english.strip(),  # 英文正文去首尾空白
                        zh_id,  # 中文 ID 原样保留
                        chinese.strip(),  # 中文正文去首尾空白
                        source=str(path),  # 记录来源路径，便于区分主数据集与上传文档
                        summary=english.strip()[:80],  # 摘要取英文前 80 字符  # 截断做摘要，前端列表展示用
                    )
                )
    if not pairs:  # 一条都没解析到，说明数据集异常，主动报错
        raise ValueError(f"数据集为空: {path}")  # 抛错避免后续建空索引
    log.info("loaded %s pairs from %s", len(pairs), path)  # 记录日志，便于核对加载量
    return pairs  # 返回句对列表给上层建索引


class TranslationRetriever:
    """BM25 检索器：持有全部句对与 BM25 索引，支持增量更新与混合检索。"""  # 类职责：封装召回逻辑

    def __init__(self, pairs: list[SentencePair], bm25: BM25Okapi):
        """构造检索器：注入全部句对与已建好的 BM25 索引，并预计算每条语料的分词结果备用。"""
        self.pairs = pairs  # 持有全部句对（含主数据集 + 动态上传）
        self.bm25 = bm25  # 持有已建好的 BM25 索引，可直接检索
        self._tokenized = [tokenize(p.as_text()) for p in pairs]  # 预存每条语料的 token 列表，便于重建索引

    def _rebuild(self) -> None:
        """重建 BM25 索引（增量加文档后调用）。"""  # 说明：增量更新后索引失效，需重建
        self._tokenized = [tokenize(p.as_text()) for p in self.pairs]  # 对当前全部句对重新分词
        self.bm25 = BM25Okapi(self._tokenized)  # 用新语料重建 BM25，使新加文档可被检索

    @classmethod
    def build(cls, path: Path | None = None) -> "TranslationRetriever":
        """构建检索器：优先用 pickle 缓存（按数据集 mtime 失效），否则全量建索引。"""  # 工厂方法：缓存优先，避免冷启动重建索引
        path = path or tsv_path()  # 默认数据集路径
        cache_file = CACHE_DIR / "teacher_bm25.pkl"  # 主索引缓存文件路径
        extra_file = CACHE_DIR / "extra_docs.pkl"  # 动态文档缓存文件路径（独立持久化）
        mtime = path.stat().st_mtime if path.exists() else 0  # 取数据集修改时间，用作缓存失效判据
        # 动态文档：用户上传入库的额外句对（与主数据集分开持久化）  # 分开存避免覆盖主数据集缓存
        extra_pairs: list[SentencePair] = []  # 收集动态上传的句对
        if extra_file.exists():  # 存在动态文档缓存则反序列化加载
            with extra_file.open("rb") as f:  # 二进制读 pickle
                extra_pairs = pickle.load(f)  # 反序列化得到动态句对列表
        # 命中缓存：数据集未修改且路径一致  # 用 mtime + 路径双校验，防止脏缓存
        if cache_file.exists():  # 主缓存存在时尝试复用
            with cache_file.open("rb") as f:  # 二进制读
                payload = pickle.load(f)  # 反序列化缓存内容：含 mtime/tsv/pairs/bm25
            if payload.get("mtime") == mtime and payload.get("tsv") == str(path):  # 双校验通过才算命中
                retriever = cls(payload["pairs"], payload["bm25"])  # 直接用缓存中的句对与索引构造检索器
                if extra_pairs:  # 有动态文档则合并并重建索引
                    retriever.pairs.extend(extra_pairs)  # 追加动态句对
                    retriever._rebuild()  # 重建索引让动态文档可被检索
                return retriever  # 缓存命中路径：直接返回，跳过全量建索引

        # 缓存未命中：全量加载 + 建 BM25 索引 + 写缓存  # 冷启动分支
        pairs = load_pairs(path)  # 从 TSV 全量加载句对
        corpus = [tokenize(p.as_text()) for p in pairs]  # 逐条分词生成语料
        bm25 = BM25Okapi(corpus)  # 用语料建 BM25 索引
        payload = {"mtime": mtime, "tsv": str(path), "pairs": pairs, "bm25": bm25}  # 组装缓存字典
        with cache_file.open("wb") as f:  # 二进制写
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)  # 用最高协议序列化，加速下次启动
        retriever = cls(pairs, bm25)  # 构造检索器
        if extra_pairs:  # 同样合并动态文档
            retriever.pairs.extend(extra_pairs)  # 追加动态句对
            retriever._rebuild()  # 重建索引
        return retriever  # 返回构建好的检索器

    def add_pairs(self, new_pairs: list[SentencePair]) -> int:
        """知识库动态更新：增量追加句对 → 重建索引 → 持久化动态文档。"""  # 说明：实现"边聊边入库"
        if not new_pairs:  # 空入参直接返回 0，避免无意义重建
            return 0
        self.pairs.extend(new_pairs)  # 把新句对追加到内存列表
        self._rebuild()  # 重建 BM25 索引，让新文档立刻可检索
        extra_file = CACHE_DIR / "extra_docs.pkl"  # 动态文档持久化路径
        extra = [p for p in self.pairs if p.source != str(tsv_path())]  # 筛出非主数据集来源的句对（即上传的动态文档）
        with extra_file.open("wb") as f:  # 二进制写
            pickle.dump(extra, f, protocol=pickle.HIGHEST_PROTOCOL)  # 持久化动态文档，下次启动可恢复
        log.info("knowledge updated, extra docs=%s", len(extra))  # 日志记录动态文档数量
        return len(new_pairs)  # 返回本次新增条数，供上层提示用户

    def search(self, query: str, top_k: int | None = None) -> list[tuple[SentencePair, float]]:
        """BM25 稀疏检索：打分 → 排序 → 阈值过滤 → 取 top_k。"""  # 说明：纯 BM25 单路召回
        top_k = top_k or TOP_K  # 未指定 top_k 用配置默认值
        tokens = tokenize(query)  # 对查询分词，得到 BM25 检索 token
        if not tokens:  # 查询分词后为空（如纯标点），直接返回空结果
            return []
        scores = self.bm25.get_scores(tokens)  # 用 BM25 对全部语料打分，返回分数数组
        # 先取 top_k*3 候选再过滤，避免阈值过滤后结果不足  # 放大候选池，防止阈值过滤后不够 top_k
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(top_k * 3, top_k)]  # 按分数降序取前 top_k*3 个候选索引
        results = []  # 收集最终结果
        for i in ranked:  # 遍历候选索引
            score = float(scores[i])  # 取该候选的 BM25 分数，转 float 便于后续计算
            if score <= SCORE_THRESHOLD:  # 低分噪声过滤  # 低于阈值的视为噪声跳过
                continue
            results.append((self.pairs[i], score))  # 命中结果加入返回列表
            if len(results) >= top_k:  # 够 top_k 就提前结束，节省遍历
                break
        return results  # 返回 (句对, 分数) 列表

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        role_code: str | None = None,
    ) -> list[tuple[SentencePair, float]]:
        """BM25 + 可选 Milvus 向量召回，再简单融合。

        role_code 非空时，向量路只在该角色知识块内检索（多角色隔离）；
        BM25 路保持全量（关键词精确匹配不受角色限制，融合时角色专属向量分占主导）。
        """  # 方法职责：多路召回 + 分数融合，互补 BM25 关键词命中与向量语义命中
        from embeddings import embed_texts  # 延迟导入，避免循环依赖与冷启动开销
        from vector_store import search_vectors  # 延迟导入向量检索，未启用 Milvus 时不报错

        bm25_hits = self.search(query, top_k=top_k)  # 第一路：BM25 稀疏召回，拿关键词精确匹配结果
        fused: dict[str, tuple[SentencePair, float]] = {}  # 融合桶：key=语料文本，value=(句对, 累加分数)
        for pair, score in bm25_hits:  # 把 BM25 命中灌入融合桶
            fused[pair.as_text()] = (pair, score)  # 用语料文本作 key，便于和向量路去重合并
        vectors = embed_texts([query])  # 未启用向量化时返回 None，跳过向量路  # 第二路：把查询转向量
        if vectors:  # 向量路启用时执行稠密召回
            for hit in search_vectors(vectors[0], top_k=top_k or TOP_K, role_code=role_code):  # 在 Milvus 中按角色隔离检索
                text = hit["text"]  # 向量库返回的文档正文
                pair = SentencePair(
                    "vec", "[DOC]", "vec", text,  # 用 [DOC] 前缀标记为文档分块类型
                    source=hit.get("source") or "milvus",  # 来源默认 milvus
                    summary=hit.get("summary") or "",  # 摘要可空
                )
                prev = fused.get(text)  # 查融合桶是否已有同文本结果（两路都命中的情况）
                fused_score = float(hit["score"]) * 10  # 余弦分数放大，与 BM25 分数量级对齐  # 余弦 0~1 放大 10 倍，与 BM25 分数同量级
                if prev:  # 两路都命中同一文档
                    fused[text] = (prev[0], prev[1] + fused_score)  # 两路都命中：分数相加  # 同一文档分数相加，提升双命中权重
                else:  # 仅向量路命中
                    fused[text] = (pair, fused_score)  # 仅向量命中：直接写入
        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)  # 按融合后总分降序排序
        return ranked[: top_k or TOP_K]  # 截取前 top_k 条作为最终召回结果


# =====================================================================
# 知识点说明（RAG：检索环节）
# ---------------------------------------------------------------------
# 1. RAG 三段式：索引（分块+向量化，见 ingest.py / embeddings.py）→
#    检索（本文件：召回+融合）→ 生成（generation.py 拼提示词给大模型）。
# 2. 稀疏检索 BM25：词频-逆文档频率打分的经典算法，对专有名词、编号等
#    "精确词"命中强；中文用 jieba 分词 + 英文正则分词（tokenize）。
# 3. 混合检索（Hybrid Search）：BM25 稀疏召回 + Milvus 稠密向量召回
#    （hybrid_search），两路按分数融合，互补长短：BM25 补关键词命中，
#    向量补语义同义改写（Milvus 亦原生支持 BM25 稀疏向量的混合检索）。
# 4. 多路召回：从不同数据源（本地 BM25、Milvus，后续可扩 MySQL、
#    Neo4j、MongoDB、互联网等）各查一路再合并，提高整体召回率。
# 5. 知识库动态更新：add_pairs 增量加句对并重建 BM25 索引，extra_docs.pkl
#    持久化动态文档，实现"边聊边入库"。
# 6. 得分过滤：SCORE_THRESHOLD 过滤低分噪声（余弦/BM25 得分过滤），
#    避免无关资料污染提示词。
# =====================================================================
