# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""重排：bge-reranker-base（CrossEncoder），不可用时降级为启发式重排。

降级链：模型缺失/加载失败 → 关键词重合度 + 原分数 的启发式重排。
功能不中断，只是精度下降 —— 符合硬性要求 7 的容错原则。
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from rag04.config import Settings
from rag04.schema import Hit

logger = logging.getLogger("rag04.rerank")

_STOP = set("的了吗呢是有在和与及对为上中下什么多少哪些哪个请问请根据")


def _terms(text: str) -> set[str]:
    """粗粒度词元：中文字符二元组 + 英文数字词。"""
    if not text:
        return set()
    toks = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    zh = re.sub(r"[^一-鿿]", "", text)
    for i in range(len(zh) - 1):
        bg = zh[i:i + 2]
        if not (bg[0] in _STOP and bg[1] in _STOP):
            toks.add(bg)
    return toks


def heuristic_rerank(hits: list[Hit], question: str, top_k: int) -> list[Hit]:
    """启发式降级重排：关键词重合 + 原分数，稳定可复现。"""
    if not hits:
        return []
    qt = _terms(question)
    scored = []
    for idx, h in enumerate(hits):
        ht = _terms(h.text)
        overlap = len(qt & ht) / max(len(qt), 1)
        scored.append((overlap * 0.7 + h.score * 0.3, -idx, h))
    # 主键分数降序；次键 -idx 升序 = 同分时按输入序的固定逆序，稳定可复现
    # （保证确定性，不随 dict/set 迭代顺序漂移）
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [h for _, _, h in scored[:top_k]]


def _ensure_safe_import_order() -> None:
    """Windows 本机环境修复（必修，否则 reranker 首次加载段错误）。

    实测（本机 Win11 + torch 2.13 + transformers 5.15）：torch 加载后首次
    ``import transformers`` 会触发 transformers→sklearn→pandas→pyarrow 的循环
    导入，递归 1500+ 层后栈溢出（access violation，与 vlparser 记录的
    pyarrow 破坏 Windows 堆是同一环境问题）。两条已实测（各 2/2 通过）的规避：
      1) torch 尚未加载：先 import sklearn（vlparser 同款预热）；
      2) torch 已加载：必须先调 torch.cuda.is_available() 初始化 CUDA 上下文，
         再 import sklearn —— 若直接 import sklearn 同为本机必崩路径（2/2 崩）。
    非 Windows 平台无此问题，直接返回。
    """
    if sys.platform != "win32":  # pragma: no cover - 本机为 win32
        return
    try:
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.cuda.is_available()      # 先于 sklearn/transformers，缺一不可
        import sklearn  # noqa: F401
    except Exception:  # pragma: no cover - 预热尽力而为，失败不阻断主链路
        logger.debug("脆弱依赖预热失败", exc_info=True)


def _extract_scores(out, batch: int) -> list[float]:
    """模型输出 → 一维 float 分数列表。

    兼容 transformers 4.x（.logits 直接是张量）与 5.x（部分方法改为返回包装
    对象，Task 6 的 CLIP 已实测）。若取不到与样本数匹配的真张量则抛错，
    交由上层降级 —— 绝不静默产出错值。
    """
    import torch

    logits = getattr(out, "logits", out)
    if not torch.is_tensor(logits):
        for attr in ("logits", "scores", "pooler_output"):
            v = getattr(logits, attr, None)
            if torch.is_tensor(v):
                logits = v
                break
        else:  # pragma: no cover - 版本不兼容时给出可诊断错误而非静默错值
            raise TypeError(f"无法从 {type(out).__name__} 取出分数张量")

    flat = logits.reshape(-1)
    if flat.numel() != batch:
        raise ValueError(
            f"reranker 分数张量形状异常：{tuple(logits.shape)} → {flat.numel()} 个分数，"
            f"期望 {batch}（num_labels 应为 1）"
        )
    return flat.float().cpu().tolist()


class CrossEncoderReranker:
    """bge-reranker-base 封装。惰性加载，加载失败即置 unavailable。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._failed = False

    @property
    def model_dir(self) -> Path:
        return Path(self.settings.models_dir) / "bge-reranker-base"

    @property
    def available(self) -> bool:
        if self._failed:
            return False
        return (self.model_dir / "pytorch_model.bin").exists()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.available:
            raise FileNotFoundError(
                f"reranker 未就位：{self.model_dir}。请运行 python scripts/download_models.py"
            )
        try:
            _ensure_safe_import_order()
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
        except Exception:
            # 加载尝试失败：置 _failed，available 转 False，后续调用直接走启发式，
            # 不再每次重试（模型 1.1GB / 约 8s，重试会拖垮每条问答的时延）。
            self._failed = True
            raise
        self._model = (model, tok, device)
        return self._model

    def rerank(self, hits: list[Hit], question: str, top_k: int) -> list[Hit]:
        import torch

        model, tok, device = self._load()
        pairs = [(question, h.text[:1024]) for h in hits]
        with torch.no_grad():
            inp = tok(pairs, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(device)
            out = model(**inp)
            scores = _extract_scores(out, len(hits))

        out_hits = []
        for h, sc in zip(hits, scores):
            out_hits.append(Hit(
                chunk_id=h.chunk_id, doc_id=h.doc_id, page=h.page,
                block_type=h.block_type, source_id=h.source_id, text=h.text,
                score=float(sc), channel="rerank", image_path=h.image_path,
            ))
        out_hits.sort(key=lambda x: -x.score)
        return out_hits[:top_k]


# 模型加载约 8s / 1.1GB，必须跨查询复用（否则每条问答都付一次加载成本，
# 无法满足端到端 <3s 的硬性验收）。键含 models_dir 与 rerank_model：
# 不同 Settings 指向不同模型目录时不得串用实例。
_RERANKER_CACHE: dict[tuple[str, str], CrossEncoderReranker] = {}


def get_reranker(settings: Settings) -> CrossEncoderReranker:
    """按模型目录惰性构造并缓存 CrossEncoderReranker（模式同 vlparser.get_clip）。"""
    key = (str(Path(settings.models_dir).resolve()), settings.rerank_model)
    rr = _RERANKER_CACHE.get(key)
    if rr is None:
        rr = CrossEncoderReranker(settings)
        _RERANKER_CACHE[key] = rr
    return rr


def rerank(hits: list[Hit], question: str, s: Settings,
           top_k: int | None = None) -> list[Hit]:
    """重排入口。模型不可用时静默降级为启发式，绝不让主链路失败。"""
    if not hits:
        return []
    k = top_k or s.final_top_k
    if not s.use_rerank:
        return hits[:k]

    rr = get_reranker(s)
    if not rr.available:
        logger.info("reranker 不可用，降级为启发式重排")
        return heuristic_rerank(hits, question, k)

    try:
        return rr.rerank(hits, question, k)
    except Exception as e:
        logger.warning("重排失败，降级启发式：%s: %s", type(e).__name__, e)
        return heuristic_rerank(hits, question, k)
