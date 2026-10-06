# -*- coding: utf-8 -*-
"""
BGE-M3 向量模型封装

职责：
    提供统一的文本向量化能力，供**离线建库脚本**与**在线问答链路**共用，
    避免两处分别加载模型造成显存/内存翻倍。

V1 范围：
    只使用 BGE-M3 的**稠密向量**（dense，1024 维）做朴素向量检索。

V2 范围（当前）：
    BGE-M3 同一份权重还能输出**稀疏权重**（lexical weights）。
    V2 的混合检索（稠密 + 稀疏 + RRF）使用 encode_sparse()。
    为降低 CPU 环境下的推理开销，稀疏与稠密由
    encode_dense_and_sparse() 在**一次 forward** 中同时产出 ——
    两者共用同一个编码器，只有输出头不同。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 稀疏编码（V2 混合检索）
# ---------------------------------------------------------------------------

# BGE-M3 基于 XLM-Roberta，其特殊 token 为：<s>=0、<pad>=1、</s>=2。
# 实测这些 token 也会获得 0.16~0.20 的权重，但它们在**任何**文本中都出现，
# 只贡献常量偏置、无区分度，属纯噪声，必须显式排除（见 ADR-015）。
_SPECIAL_TOKEN_IDS = frozenset({0, 1, 2})


def _to_sparse_dict(
    weights: Any,
    input_ids: Any,
    attention_mask: Any,
) -> Dict[int, float]:
    """
    把 token 级权重压缩为 {token_id: weight} 稀疏字典。

    规则：
        1. padding 位置（attention_mask=0）跳过
        2. 权重 <= 0 的 token 跳过（ReLU 后仍可能为 0）
        3. 特殊 token 跳过（见 _SPECIAL_TOKEN_IDS）
        4. 同一 token_id 多次出现时取最大权重

    这是纯函数，不依赖模型，便于单元测试。
    """
    # 三个张量按位置一一对应，所以直接 zip 起来逐 token 处理：
    # tok_id 是这个 token 在词表里的编号（稀疏向量正是以它为维度下标的），
    # weight 是模型给它的重要性权重，mask 标记它是不是补齐用的填充位。
    result: Dict[int, float] = {}
    for tok_id, weight, mask in zip(
        input_ids.tolist(), weights.tolist(), attention_mask.tolist()
    ):
        # mask=0 是补齐出来的空位；weight<=0 表示模型认为这个词不重要 —— 都跳过。
        if mask == 0 or weight <= 0:
            continue
        # 特殊 token 在每个文本里都出现，带不出区分度，跳过。
        if tok_id in _SPECIAL_TOKEN_IDS:
            continue
        # 同一个词可能在一句话里出现多次，只保留权重最大的那一次：
        # 稀疏向量里一个词只能有一个权重。
        if weight > result.get(tok_id, 0.0):
            result[tok_id] = float(weight)
    return result


# 向量模型不可用时的统一异常：离线建库脚本据此判断"这台机器模型没配好"，
# 在线链路的 V2/V3 也据此走降级分支。
class EmbedderNotAvailableError(RuntimeError):
    """向量模型不可用（未安装依赖或模型路径不存在）"""


class BGEM3Embedder:
    """
    BGE-M3 向量模型封装。

    线程安全的懒加载单例：首次调用时才加载权重，
    避免服务启动阶段耗时过长。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        # 模型权重目录（默认取配置 bge_m3_path）、跑在 CPU 还是 GPU（embed_device）、
        # 单条文本最多截到多少 token（embed_max_length）、一次编多少条（embed_batch_size）。
        self.model_path = model_path or settings.bge_m3_path
        self.device = device or settings.embed_device
        self.max_length = max_length or settings.embed_max_length
        self.batch_size = batch_size or settings.embed_batch_size

        # 懒加载用的三个字段：
        #   _model         BGE-M3 主体，负责产出稠密向量
        #   _sparse_linear 稀疏权重头，负责产出稀疏权重（与主体共用一个编码器）
        #   _lock          保证并发请求同时触发加载时只加载一次
        self._model: Any = None
        self._sparse_linear: Any = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------------

    def _load(self) -> Any:
        """加载模型（懒加载 + 加锁，保证多线程下只加载一次）"""
        # 加锁前先查一次：模型已经加载好就直接返回，让常用路径不必去抢锁。
        if self._model is not None:
            return self._model

        # 拿到锁之后再查一次（双重检查）：等锁的这段时间里别人可能已经加载完了。
        with self._lock:
            if self._model is not None:
                return self._model

            from pathlib import Path
            # 模型目录不存在就直接判为"向量模型不可用"。这是部署时最常见的问题：
            # .env 里的 BGE_M3_PATH 配错，或权重根本没下载下来。
            if not Path(self.model_path).exists():
                raise EmbedderNotAvailableError(
                    f"BGE-M3 模型路径不存在：{self.model_path}\n"
                    f"请检查 .env 中的 BGE_M3_PATH 配置"
                )

            # 依赖没装同样归一化成"向量模型不可用"，上层才好写统一的降级分支。
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderNotAvailableError(
                    "加载 BGE-M3 需要 sentence-transformers，请先安装：\n"
                    "    pip install sentence-transformers"
                ) from exc

            logger.info("正在加载 BGE-M3 向量模型：%s（设备=%s）",
                        self.model_path, self.device)
            # 真正把权重载入内存。这一步比较慢，加载完由 self._model 缓存住，
            # 后续所有请求都不再重复加载。
            # trust_remote_code=True 是因为 BGE-M3 自带的 pooling 配置需要执行
            # 模型目录里的代码才能生效。
            model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True,
            )
            # 长文本场景下放开序列长度限制
            try:
                model.max_seq_length = self.max_length
            except Exception:
                pass

            self._model = model
            logger.info("BGE-M3 加载完成，向量维度：%d", self.dimension)
            return self._model

    @property
    def dimension(self) -> int:
        """向量维度（BGE-M3 固定为 1024）"""
        model = self._model
        # 模型还没加载时直接返回配置里的维度。建 Milvus 集合、建索引都要用到维度，
        # 不该为了拿一个数字就把整个模型加载起来。
        if model is None:
            return settings.milvus_dim
        try:
            # 正常路径：直接问模型要它输出的向量维度。
            return int(model.get_sentence_embedding_dimension())
        except Exception:
            # 万一取不到，退回配置值兜底 —— 关键是保证"Milvus 集合定义的维度"
            # 与"实际写进去的向量维度"永远一致，否则写入时就会报错。
            return settings.milvus_dim

    # ----------------------------------------------------------------
    # 稀疏权重头
    # ----------------------------------------------------------------

    def _load_sparse_linear(self) -> Any:
        """
        加载 BGE-M3 的稀疏权重头 sparse_linear.pt。

        该文件仅约 3.5KB，是 BGE-M3 官方的 lexical weights 线性层
        （weight shape (1, 1024) + bias）。
        直接加载可避免引入 FlagEmbedding 依赖 —— 后者会连带触发
        sentence-transformers 的版本降级，影响本项目既有的稠密编码
        （见 ADR-014）。
        """
        # 同样用"加锁前查一次 + 加锁后再查一次"的懒加载写法。
        if self._sparse_linear is not None:
            return self._sparse_linear

        with self._lock:
            if self._sparse_linear is not None:
                return self._sparse_linear

            from pathlib import Path

            import torch

            # 稀疏权重头是 BGE-M3 官方随模型一起发布的一个小文件，就放在模型目录里。
            path = Path(self.model_path) / "sparse_linear.pt"
            if not path.exists():
                raise EmbedderNotAvailableError(
                    f"BGE-M3 稀疏权重文件不存在：{path}\n"
                    f"V2 混合检索需要该文件（sparse_linear.pt）"
                )

            # 先把权重文件读进来，先放 CPU —— 万一 GPU 不可用也不会直接失败。
            state = torch.load(str(path), map_location="cpu")
            # 按权重矩阵的形状反推线性层的输入/输出维度，再建一个同形状的空层。
            linear = torch.nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
            # 把读到的权重灌进去，这一步之后得到的就是 BGE-M3 的稀疏输出头。
            linear.load_state_dict(state)
            # 切到推理模式：关掉 dropout 之类的训练期行为，
            # 保证同样的输入每次得到同样的输出（评测可复现）。
            linear.eval()
            # 挪到和主体模型同一个设备上，否则前向计算时张量设备不匹配会报错。
            linear.to(self.device)

            self._sparse_linear = linear
            logger.info("BGE-M3 稀疏权重头加载完成：%s", path)
            return linear

    # ----------------------------------------------------------------
    # 稠密向量
    # ----------------------------------------------------------------

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        批量编码为稠密向量。

        参数：
            texts        : 待编码文本列表
            batch_size   : 批大小，CPU 环境建议 8-16
            show_progress: 是否显示进度条（离线建库时建议开启）

        返回：与输入等长的向量列表，每个向量为 1024 维浮点列表。
        """
        # 空输入直接返回，没必要把空列表喂给模型。
        if not texts:
            return []

        model = self._load()
        # 交给 sentence-transformers 批量编码。
        vectors = model.encode(
            list(texts),
            # 一次编多少条：CPU 环境下批太大反而更慢，所以默认只开 8~16。
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            # 归一化后内积等价于余弦相似度，与 Milvus 的 COSINE 度量一致
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # numpy 数组转成普通 Python 列表：这样才能写进 Milvus，也方便序列化。
        return [v.tolist() for v in vectors]

    def encode_query(self, text: str) -> List[float]:
        """
        编码单条查询文本。

        在线链路只用这一次编码，开销可忽略；
        BGE-M3 官方建议查询侧可加指令前缀，但中英文通用检索场景
        不加前缀效果同样良好，此处保持与建库侧一致。
        """
        # 包一层只是为了把"单条文本"统一成批量接口，再取出唯一那个结果。
        # 空输入时 encode 返回空列表，这里也返回空，不做 [0] 越界取值。
        vectors = self.encode([text], show_progress=False)
        return vectors[0] if vectors else []

    # ----------------------------------------------------------------
    # 稀疏向量（V2 混合检索使用）
    # ----------------------------------------------------------------

    def encode_dense_and_sparse(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        一次 forward 同时产出稠密向量与稀疏权重。

        为什么要合并为一次 forward：
            BGE-M3 的稠密与稀疏共用同一个编码器，只有输出头不同。
            分两次调用会让 CPU 环境下的推理耗时翻倍。
            已验证手动 CLS 池化的结果与 sentence-transformers 完全一致
            （见 tests/test_sparse_encode.py 的一致性回归测试）。

        返回：(稠密向量列表, 稀疏字典列表)，两者与输入等长且一一对应。
        """
        # 空输入时两个返回值都要跟着空，保持"等长且一一对应"的约定。
        if not texts:
            return [], []

        import torch

        model = self._load()
        # 直接从 sentence-transformers 的封装里掏出底层的 XLMRoberta 模型：
        # 稠密和稀疏两个输出头都要拿它输出的 hidden state，所以必须绕过封装直接调。
        auto_model = model[0].auto_model          # 底层 XLMRobertaModel
        tokenizer = model.tokenizer
        sparse_linear = self._load_sparse_linear()

        bs = batch_size or self.batch_size
        # 两个输出头都切到推理模式，保证结果稳定可复现。
        auto_model.eval()
        sparse_linear.eval()

        dense_all: List[List[float]] = []
        sparse_all: List[Dict[int, float]] = []

        # 分批处理，避免一次性把大量文本塞进张量导致内存爆掉。
        for start in range(0, len(texts), bs):
            batch = list(texts[start : start + bs])
            # 分词成模型能吃的 token id：
            # padding=True 把一批里的句子补齐到同一长度（张量必须等长），
            # truncation + max_length 截断过长句子，防止超长输入拖慢推理。
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # 把分词结果搬到模型所在设备（CPU 或 GPU），否则无法做前向计算。
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # 关掉梯度记录：这里只做推理不训练，能省内存也更快。
            with torch.no_grad():
                # ★ 整个方法只跑这一次 forward，稠密和稀疏都从它上面取 ★
                # 这就是"两个输出头共用一个编码器"的含义，也是性能优化的关键：
                # 如果分两次调用，CPU 上的推理耗时会直接翻倍。
                hidden = auto_model(**encoded).last_hidden_state   # (B, L, 1024)
                # 稠密：CLS 池化 + L2 归一化（与 1_Pooling/config.json 一致）
                # 取每句第 0 个位置（即 CLS）的向量作为整句表示，
                # 归一化之后两向量的内积就等于余弦相似度。
                cls = hidden[:, 0]
                dense = torch.nn.functional.normalize(cls, p=2, dim=1)
                # 稀疏：ReLU 后的 token 级权重，屏蔽 padding
                # 用隐藏状态过一遍稀疏线性层，得到每个 token 的重要性分数。
                raw = sparse_linear(hidden).squeeze(-1)            # (B, L)
                # ReLU 把负权重压成 0（稀疏向量里不该出现负数），
                # 再乘 attention_mask 把补齐位置的权重清零。
                weights = torch.relu(raw * encoded["attention_mask"])

            # 稠密结果搬回 CPU 并转成普通列表，可以直接写库。
            dense_all.extend(dense.cpu().numpy().tolist())
            # 稀疏结果逐条压缩成 {token_id: 权重} 字典。
            for i in range(len(batch)):
                sparse_all.append(
                    _to_sparse_dict(
                        weights[i], encoded["input_ids"][i], encoded["attention_mask"][i]
                    )
                )

        if show_progress:
            logger.info("稠密+稀疏编码完成：%d 条文本", len(dense_all))

        # 两个列表严格按输入顺序一一对应，调用方可以放心按下标配对使用。
        return dense_all, sparse_all

    def encode_sparse(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
    ) -> List[Dict[int, float]]:
        """
        编码为稀疏权重（BGE-M3 lexical weights，V2 混合检索使用）。

        返回：[{token_id: weight}, ...]，与输入等长。

        实际计算走 encode_dense_and_sparse，此处只取稀疏部分；
        调用方若同时需要稠密向量，应直接调用 encode_dense_and_sparse
        以避免重复推理。
        """
        # 稠密部分用不上就用下划线接收丢弃。注意这里**整个前向计算照样跑满了** ——
        # 如果调用方其实稠密也要，就应该直接调 encode_dense_and_sparse，
        # 别再调一次 encode，那等于把推理白跑两遍。
        _, sparse = self.encode_dense_and_sparse(texts, batch_size=batch_size)
        return sparse


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

# 进程级单例。BGE-M3 权重不小，而且离线建库脚本和在线问答链路都要用它，
# 全进程必须只加载一份，否则内存直接翻倍。
_embedder: Optional[BGEM3Embedder] = None
# 保护单例创建的锁，避免并发请求同时触发加载。
_embedder_lock = threading.Lock()


def get_embedder() -> BGEM3Embedder:
    """获取向量模型单例（进程内只加载一次权重）"""
    # 双重检查加锁：实例已建好时就不再去抢锁。
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = BGEM3Embedder()
    return _embedder
