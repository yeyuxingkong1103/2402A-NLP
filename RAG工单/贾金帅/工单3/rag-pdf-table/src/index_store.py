"""
知识库（索引）存储与管理
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

对应工单产出物「知识库管理：支持对解析后的文档内容进行存储和管理」。

落盘三件套（都在 data/index/ 下，纯本地文件，无外部服务依赖）：
  chunks.jsonl      分块正文 + 元数据（页码/章节/类型）
  embeddings.npy    归一化向量矩阵
  index_meta.json   索引元信息（模型、维度、条数、构建时间、源文件指纹）

为什么用本地文件而不是向量数据库：本工单是单文档、量级 ~3k 块，
暴力余弦在 CPU 上 < 5ms，完全满足 3 秒响应要求；
引入 Milvus/ES 只会多一个必须常驻的进程和一堆端口冲突（本机 19530 已被占用）。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunker import Chunk
from .config import (
    CHUNKS_JSONL,
    EMBEDDINGS_NPY,
    INDEX_DIR,
    INDEX_META_JSON,
    INDEX_NOTABLE_DIR,
    WORK_ORDER_NO,  # noqa: F401
    WORK_ORDER_NO_TABLE,  # noqa: F401  工单03 编号
)


def index_paths(dir_path: Path | None = None) -> tuple[Path, Path, Path]:
    """
    返回某个索引目录下的三件套路径。

    工单03 起同时存在两份索引（主线 = 表格结构化，对照组 = 表格不结构化），
    因此路径不能写死；默认仍是主线目录。
    """
    d = Path(dir_path) if dir_path else INDEX_DIR
    return d / CHUNKS_JSONL.name, d / EMBEDDINGS_NPY.name, d / INDEX_META_JSON.name


# ------------------------------------------------------------------ 写索引


def file_fingerprint(path: Path) -> str:
    """源文件指纹：大小 + mtime，用来判断索引是否与源文档一致。"""
    st = path.stat()
    raw = f"{path.name}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def save_index(chunks: list[Chunk], embeddings: np.ndarray,
               source_paths: Path | list[Path], model_path: str,
               out_dir: Path | None = None) -> dict:
    """
    把分块与向量写入磁盘，返回索引元信息。

    工单03 起语料是**两份文档**，因此入参是路径列表（单份也接受，
    内部统一成列表）—— 元信息里同时记下两个源文件各自的指纹，
    只要任意一份换了版本，指纹对不上就能立刻发现索引过期。

    out_dir：写到指定目录（用于工单03 的「表格不结构化」对照索引，见 config.INDEX_NOTABLE_DIR）。
    """
    d = Path(out_dir) if out_dir else INDEX_DIR
    d.mkdir(parents=True, exist_ok=True)
    chunks_path, emb_path, meta_path = index_paths(d)

    if isinstance(source_paths, (str, Path)):
        source_paths = [Path(source_paths)]
    source_paths = [Path(p) for p in source_paths]

    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    np.save(emb_path, embeddings.astype(np.float32))

    docs: dict[str, int] = {}
    for c in chunks:
        key = c.doc_key or c.doc or "?"
        docs[key] = docs.get(key, 0) + 1

    meta = {
        "work_order_no": WORK_ORDER_NO_TABLE,
        "source_file": source_paths[0].name if len(source_paths) == 1 else
                       " + ".join(p.name for p in source_paths),
        "source_files": [
            {"name": p.name, "fingerprint": file_fingerprint(p)} for p in source_paths
        ],
        "source_fingerprint": "|".join(file_fingerprint(p) for p in source_paths),
        "embedding_model": Path(model_path).name,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "num_chunks": len(chunks),
        "num_table_chunks": sum(1 for c in chunks if c.type == "table"),
        "chunks_per_doc": docs,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_chunk_len": max((len(c.text) for c in chunks), default=0),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ------------------------------------------------------------------ 读索引


class KnowledgeBase:
    """
    内存态知识库：分块 + 向量 + BM25 词表。

    线程安全：`get()` 用双重检查锁保证只加载一次。
    """

    _instance: "KnowledgeBase | None" = None
    _variants: dict[str, "KnowledgeBase"] = {}
    _lock = threading.Lock()

    def __init__(self, chunks: list[dict], embeddings: np.ndarray, meta: dict):
        self.chunks = chunks
        self.embeddings = embeddings
        self.meta = meta
        self._bm25 = None
        self._tokens: list[list[str]] = []
        self._ubiquitous: set[str] | None = None
        self._phrases: list[str] | None = None

    # -------------------------------------------------- 单例
    @classmethod
    def get(cls, dir_path: Path | None = None) -> "KnowledgeBase":
        """
        取知识库单例。

        dir_path 为空 → 主线索引（表格结构化）；
        指定目录 → 对应的那份（工单03 用它取「表格不结构化」的对照索引），
        同样只加载一次并缓存。
        """
        if dir_path is not None:
            key = str(Path(dir_path))
            if key not in cls._variants:
                with cls._lock:
                    if key not in cls._variants:
                        cls._variants[key] = cls.load(dir_path)
            return cls._variants[key]

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.load()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """索引重建后调用，强制下次访问重新加载。"""
        with cls._lock:
            cls._instance = None
            cls._variants.clear()

    @classmethod
    def get_notable(cls) -> "KnowledgeBase":
        """工单03 的对照索引：同一套语料，但**表格不做结构化**。"""
        return cls.get(INDEX_NOTABLE_DIR)

    @classmethod
    def load(cls, dir_path: Path | None = None) -> "KnowledgeBase":
        chunks_path, emb_path, meta_path = index_paths(dir_path)
        if not chunks_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                "索引不存在。请先执行：python scripts/build_index.py"
                + (f" --no-tables（对照索引 {chunks_path.parent}）" if dir_path else "")
            )
        chunks = [
            json.loads(ln)
            for ln in chunks_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        embeddings = np.load(emb_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"索引不一致：分块 {len(chunks)} 条，向量 {embeddings.shape[0]} 条。请重建索引。"
            )
        return cls(chunks, embeddings, meta)

    # -------------------------------------------------- BM25
    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        中文分词。必须用 jieba：默认按空格切会把整句当成一个 token，
        BM25 对中文长句恒为空，混合检索会**静默退化成纯向量检索**。
        """
        import jieba

        return [t for t in jieba.lcut(text) if t.strip() and not t.isspace()]

    @property
    def bm25(self):
        """按需构建 BM25 索引（约 1~2 秒），只建一次。"""
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._tokens = [self.tokenize(c["text"]) for c in self.chunks]
            self._bm25 = BM25Okapi(self._tokens)
        return self._bm25

    # -------------------------------------------------- 语料级"无区分度词"
    # 单文档语料里，主体名称（"武汉兴图新科电子股份有限公司"及其分词碎片）几乎出现在
    # 每一个块中，对检索没有任何区分力，却是 BM25 的高频命中源。
    # 实测后果：问「报告期内来自军用领域的收入分别是多少」能正确召回第342页；
    # 而同样的问题只要带上公司全称，召回第一名就变成一张贷款合同表格（第532页，
    # 因为那块里反复出现公司全称）——正确答案被挤出上下文。
    # 做法：按**文档频率**自动识别这类词并在检索式中剔除，等价于让语料自己生成停用词表。
    #
    # 阈值 0.2 是实测扫出来的（同一份 10 题 + 3 个离题问题）：
    #   不滤 → 9/10 命中（公司全称把贷款合同表格顶到第一，正确答案被挤出）
    #   0.30 → 9/10      0.20 → **10/10**      0.15 → 9/10（把有效词也滤掉了）
    # 且 0.20 下离题问题最高依据分仍是 0.7410，低于 0.80 闸门，没有副作用。
    UBIQUITOUS_DF_RATIO = 0.2

    @property
    def ubiquitous_terms(self) -> set[str]:
        """出现在超过半数块中的词（长度>1）。惰性计算，依赖 BM25 的分词结果。"""
        if self._ubiquitous is None:
            _ = self.bm25  # 确保 _tokens 已就绪
            n = len(self._tokens)
            if n == 0:
                self._ubiquitous = set()
            else:
                df: dict[str, int] = {}
                for toks in self._tokens:
                    for t in set(toks):
                        df[t] = df.get(t, 0) + 1
                self._ubiquitous = {
                    t for t, c in df.items() if len(t) > 1 and c / n > self.UBIQUITOUS_DF_RATIO
                }
        return self._ubiquitous

    def filter_query_terms(self, query: str) -> str:
        """
        从检索式中剔除「无区分度成分」。两级过滤，缺一不可：

        **第一级：短语级**（见 ubiquitous_phrases）。
        词级过滤有个结构性缺陷：中文分词会把「武汉兴图新科电子股份有限公司」
        切成 武汉/兴图/新科/电子/股份/有限公司 六个碎片，每个碎片的 DF 只有
        0.06~0.21 —— 谁都够不着 0.20 的阈值，于是整串公司名只被剔掉一个「兴图」。
        实测后果：问「……在哪个领域已经成为重要供应商」，加了公司全称时召回
        全是公司基本信息页（第22/52/61页），去掉公司全称后立刻回到正确的
        行业分析章节。**问题不在块切得好不好，而在检索器分不清哪些词有区分度。**

        **第二级：词级**（见 ubiquitous_terms）。

        最后有长度兜底：滤完不足 4 字就退回原串，避免查询被清空导致检索失去条件。
        """
        out = query
        # 长短语优先替换（列表已按长度降序），避免短短语把长短语切碎
        for p in self.ubiquitous_phrases:
            if p in out:
                out = out.replace(p, "")

        tokens = self.tokenize(out)
        if tokens:
            kept = [t for t in tokens if t not in self.ubiquitous_terms]
            if kept:
                out = "".join(kept)

        out = out.strip()
        return out if len(out) >= 4 else query

    # -------------------------------------------------- 语料级"无区分度短语"
    # 词级 DF 的补充。中文分词会把专有名称切碎，碎片各自都不高频，
    # 于是「武汉兴图新科电子股份有限公司」这种贯穿全文的主体名反而漏网 ——
    # 实测只剔掉了「兴图」一个词，其余原样送进检索器。
    #
    # 做法是**迭代式短语合并**（BPE 的思路，但用于"找噪音"而不是"建词表"）：
    #   第 1 轮：找出高频相邻二元组 → 就地合并成新 token
    #   第 2..N 轮：在合并结果上重复，短语会逐轮变长
    # 实测 1 轮只能得到「兴图新科」，3 轮就能长成「兴图新科电子股份有限公司」。
    # 每轮的成本是 O(全部 token) ≈ 30 万次操作，一轮约 0.3s；
    # 构建只在索引加载后做一次，且被 warm_up 提前触发，不影响单次请求延迟。
    PHRASE_COMMON_DF = 0.05    # 够格参与拼接的「常见词」
    PHRASE_HOT_DF = 0.10       # 高频相邻二元组（高于此值才认为两词总是一起出现）
    PHRASE_KEEP_DF = 0.10      # 整条短语的最小 DF（短语天然比单词稀有，所以低于词级阈值）
    PHRASE_MIN_CHARS = 4
    PHRASE_ROUNDS = 3

    @property
    def ubiquitous_phrases(self) -> list[str]:
        """语料级重复短语（主体名称、固定表述），**按长度降序**，便于先长后短地替换。"""
        if self._phrases is None:
            self._phrases = self._build_phrases()
        return self._phrases

    def _build_phrases(self) -> list[str]:
        from collections import Counter

        _ = self.bm25  # 确保 _tokens 已就绪
        n = len(self._tokens)
        if n == 0:
            return []

        toks_list: list[list[str]] = [list(t) for t in self._tokens]
        found: dict[str, int] = {}

        for _ in range(self.PHRASE_ROUNDS):
            df: Counter = Counter()
            for tk in toks_list:
                df.update(set(tk))
            common = {t for t, c in df.items() if len(t) > 1 and c / n > self.PHRASE_COMMON_DF}
            if not common:
                break

            big: Counter = Counter()
            for tk in toks_list:
                big.update({a + b for a, b in zip(tk, tk[1:]) if a in common and b in common})
            hot = {p for p, c in big.items() if c / n > self.PHRASE_HOT_DF}
            if not hot:
                break

            merged: list[list[str]] = []
            for tk in toks_list:
                out: list[str] = []
                i, L = 0, len(tk)
                while i < L:
                    if i < L - 1 and tk[i] + tk[i + 1] in hot:
                        out.append(tk[i] + tk[i + 1])
                        i += 2
                    else:
                        out.append(tk[i])
                        i += 1
                merged.append(out)
            toks_list = merged

            # 记录这一轮新长出来的短语及其 DF
            df2: Counter = Counter()
            for tk in toks_list:
                df2.update({t for t in set(tk) if len(t) >= self.PHRASE_MIN_CHARS})
            for p, c in df2.items():
                found[p] = max(found.get(p, 0), c)

        return sorted(
            {p for p, c in found.items() if c / n > self.PHRASE_KEEP_DF},
            key=len,
            reverse=True,
        )

    def bm25_scores(self, query: str) -> np.ndarray:
        """返回全量块的 BM25 原始分（无上界，注意与余弦分不同量纲）。"""
        bm25 = self.bm25
        tokens = self.tokenize(query)
        if not tokens:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(bm25.get_scores(tokens), dtype=np.float32)

    # -------------------------------------------------- 多文档消歧（工单03）
    @property
    def doc_keys(self) -> set[str]:
        """本索引里出现的文档短键集合。"""
        return {c.get("doc_key", "") for c in self.chunks if c.get("doc_key")}

    def detect_docs(self, query: str) -> list[str]:
        """
        从**原问题**里识别它问的是哪一份文档。

        为什么要单独做这一步（实测踩到的坑）：
          DF 过滤会把公司全称从检索式里剔掉（工单2 的优化，单文档下是纯收益）。
          两份文档并存时这招反过来咬人 ——
            「武汉兴图新科电子股份有限公司注册资本是多少」
            → 剔成「注册资本是多少」→ 力源那边的验资章节被召回，兴图的答案被挤出候选。
          公司名在两文档语料里是**最强的消歧信号**，必须用它，只是用法要换：
          不是「留在 BM25 里当关键词」，而是「把候选池限定到那份文档」。

        返回命中的文档短键列表；一句里同时提到两家时返回两家（不做限定，交给检索打分）。
        """
        from .config import DOC_ALIASES

        hits: list[str] = []
        present = self.doc_keys
        for key, aliases in DOC_ALIASES.items():
            if present and key not in present:
                continue
            if any(a in query for a in aliases):
                hits.append(key)
        return hits

    def stats(self) -> dict:
        return {
            **{k: v for k, v in self.meta.items()},
            "loaded_chunks": len(self.chunks),
            "embedding_shape": list(self.embeddings.shape),
        }
