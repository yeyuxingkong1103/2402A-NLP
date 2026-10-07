"""
嵌入模型注册表（多模型支持）
工单编号：人工智能NLP-RAG-混合检索任务

对应工单技术要求：「支持多种嵌入模型（如 bge、m3e 以及其他嵌入模型）」。

设计取舍：把「模型」提升为**一等配置**，而不是散落在代码里的一个路径常量。

    模型 key ──┬─→ 编码器（进程内单例，按 key 缓存）
               ├─→ 索引目录 data/index_<key>（向量维度/语义空间不同，必须分开存）
               └─→ 元信息（维度、是否已下载、索引是否已构建）

为什么必须按 key 分目录存索引：
    bge-small-zh 是 512 维、m3e-base 是 768 维，同一份 `embeddings.npy` 根本装不下两种；
    即便是同维度的两个模型，向量空间也完全不同 —— 用 A 建的向量去和 B 编码的查询
    做点积，得到的是**没有意义的数**（不会报错，只会悄悄答错，这是最危险的一种 bug）。
    所以「换模型」= 「换一套索引」，本模块用 `index_dir_for()` 把这件事变成强约束。

默认 key 是 bge-small-zh，它直接复用 `data/index`（工单1~5 已建好的那份），
因此旧索引与新功能天然兼容，不需要重建。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from .config import (
    DATA_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_KEY,
    EMBEDDING_QUERY_INSTRUCTION,
    EMBEDDING_REGISTRY,
    WORK_ORDER_NOS,  # noqa: F401
)

logger = logging.getLogger(__name__)

_encoders: dict[str, object] = {}
_lock = threading.Lock()

#: 默认 key。它对应工单1~5 的 `data/index`（不额外建目录），保持向后兼容。
DEFAULT_KEY = "bge-small-zh"


def list_models() -> list[dict]:
    """
    列出全部注册模型及其**实际可用性**。

    `downloaded` 看本地目录在不在；`index_built` 看该模型的索引三件套齐不齐。
    两个标志分开，是因为它们对应两种完全不同的用户动作：
    「去下模型」vs「去跑 build_index」。
    """
    out = []
    for key, spec in EMBEDDING_REGISTRY.items():
        p = Path(spec["path"])
        d = index_dir_for(key)
        downloaded = p.is_dir() and any(p.glob("*.safetensors") or p.glob("*.bin"))
        built = (d / "chunks.jsonl").exists() and (d / "embeddings.npy").exists()
        out.append({
            "key": key,
            "name": spec["name"],
            "path": str(p),
            "dim": spec["dim"],
            "note": spec.get("note", ""),
            "downloaded": downloaded,
            "index_dir": str(d),
            "index_built": built,
            # 只有「模型下载好了」且「该模型的索引建好了」才真的能用 ——
            # 两者缺一都会在检索时降级，所以可用性必须由两者共同决定。
            "available": bool(downloaded and built),
            "is_default": key == DEFAULT_KEY,
        })
    return out


def index_dir_for(key: str) -> Path:
    """
    某个模型对应的索引目录。

    默认模型用 `data/index`（兼容工单1~5），其余用 `data/index_<key>`。
    目录名里的 `-` 换成 `_`，避免在某些工具链里被当成参数分隔符。
    """
    if key in ("", None) or key == DEFAULT_KEY:
        return DATA_DIR / "index"
    return DATA_DIR / f"index_{key.replace('-', '_')}"


def model_status(key: str) -> dict:
    """单个模型的可用性详情（含给前端的提示语）。"""
    spec = EMBEDDING_REGISTRY.get(key)
    if spec is None:
        return {"key": key, "available": False, "note": f"未注册的模型 key：{key}"}
    p = Path(spec["path"])
    d = index_dir_for(key)
    downloaded = p.is_dir() and any(p.glob("*.safetensors") or p.glob("*.bin"))
    built = (d / "chunks.jsonl").exists() and (d / "embeddings.npy").exists()
    note = spec.get("note", "")
    if not downloaded:
        note = (f"模型未下载。请把 {spec['name']} 放到 {p}，"
                f"或改 .env 里的路径。可用：git clone 或 huggingface-cli download。")
    elif not built:
        note = (f"该模型的索引尚未构建。执行："
                f"python scripts/build_index.py --embed-model {key}")
    return {
        "key": key, "name": spec["name"], "path": str(p), "dim": spec["dim"],
        "downloaded": downloaded, "index_dir": str(d), "index_built": built,
        "available": bool(downloaded and built), "note": note,
    }


def get_encoder(key: str | None = None):
    """
    取编码器（进程内单例）。

    默认模型直接复用 `embedder.get_model()` —— 工单1~5 的主链路已经在用它，
    再加载一份会白白多占几百 MB 内存（本机内存本就紧张）。
    """
    k = key or EMBEDDING_MODEL_KEY
    if k == DEFAULT_KEY:
        from .embedder import get_model

        return get_model()

    if k in _encoders:
        return _encoders[k]
    with _lock:
        if k in _encoders:
            return _encoders[k]
        spec = EMBEDDING_REGISTRY.get(k)
        if spec is None:
            raise KeyError(f"未注册的嵌入模型 key：{k}")
        p = Path(spec["path"])
        if not p.is_dir():
            raise FileNotFoundError(
                f"模型目录不存在：{p}（{spec['name']}）。"
                f"请先下载模型，或改用其它 --embed-model。"
            )
        from sentence_transformers import SentenceTransformer

        logger.info("加载嵌入模型 %s（%s）", k, p)
        _encoders[k] = SentenceTransformer(str(p), device="cpu")
        return _encoders[k]


def embed_texts(texts: list[str], key: str | None = None) -> np.ndarray:
    """把文本编码成 L2 归一化向量。与 embedder.embed_texts 同样的契约。"""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_encoder(key)
    vecs = model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE,
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str, key: str | None = None) -> np.ndarray:
    """
    单条查询编码，返回 (1, D)。

    `EMBEDDING_QUERY_INSTRUCTION` 非空时会加在查询前面 ——
    bge 系列官方给的用法（短查询加指令能略微提升召回）。
    注意**文档侧不加**，所以索引不受该配置影响，可以随时切换。
    """
    q = text
    if EMBEDDING_QUERY_INSTRUCTION:
        q = f"{EMBEDDING_QUERY_INSTRUCTION}{text}"
    return embed_texts([q], key)


def cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """全量余弦（两边已归一 → 点积）。"""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ query_vec.reshape(-1).astype(np.float32)


def reset() -> None:
    """释放已加载的非默认模型（测试/切换时用）。"""
    _encoders.clear()
