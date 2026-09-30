# -*- coding: utf-8 -*-
"""BGE-m3 真实向量化后端。

权重走 ModelScope（本机 huggingface.co 不可达，modelscope.cn 可达）。
依赖为重量级（torch + sentence-transformers），因此：
  * import 延迟到真正加载模型时；
  * 缺依赖时抛出带修复命令的 RuntimeError，而不是 ImportError 堆栈；
  * 加载失败不阻塞离线兜底链路（调用方按配置降级即可）。
"""
from __future__ import annotations

import os

from .base import Embedder

DEFAULT_MODEL = "BAAI/bge-m3"


def resolve_model_path(model_name: str) -> str:
    """把模型名解析成**本地目录**：本地目录短路 → ModelScope 缓存/下载 → 原样返回。

    ★ **本地目录短路**（2026-09-21 加，云端实测驱动）：若传进来的是**已存在的目录**
    （例如云端预下好的
    ``/root/autodl-tmp/modelscope/models/BAAI--bge-m3/snapshots/master``），
    **直接用它、绝不联网**。理由有两条，都是实测踩出来的：

    1. 省掉每次启动的 ModelScope 校验（无卡模式下 0.5 核 + 2 GB 内存，
       第一次 embed 有 ~60s 花在这上面）；
    2. 避免"为省磁盘删掉的冗余文件被自动补回"——``snapshot_download`` 会把
       快照补全（bge-m3 的 onnx 副本就是这样被拉回来，缓存 2.2G ↔ 4.3G 反复横跳）。
    """
    candidate = os.path.expanduser(str(model_name or "").strip())
    if candidate and os.path.isdir(candidate):
        return candidate

    try:
        from modelscope import snapshot_download  # type: ignore
    except ImportError:
        return model_name

    try:
        return snapshot_download(model_name)
    except Exception as exc:  # noqa: BLE001 - 需要把原始原因带给用户
        raise RuntimeError(
            f"从 ModelScope 获取 {model_name} 失败：{exc}\n"
            f"提示：本机 huggingface.co 不可达，请确认 modelscope 可用，"
            f"或设置 HF_ENDPOINT=https://hf-mirror.com 后重试，"
            f"或改用离线兜底后端 EMBEDDING_PROVIDER=offline。"
        ) from exc


class BgeM3Embedder(Embedder):
    name = "bge-m3"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 8, use_fp16: bool = False, max_length: int = 8192) -> None:
        self.model_name = model_name
        self.device = device or os.environ.get("EMBEDDING_DEVICE") or None
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.dim = 1024
        self._model = None

    # ---------- 懒加载 ----------
    def _load(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers，无法使用 BGE-m3。\n"
                "请先安装重依赖：\n"
                "    .venv\\Scripts\\python.exe -m pip install -r requirements-full.txt\n"
                "或改用离线兜底后端：EMBEDDING_PROVIDER=offline"
            ) from exc

        path = resolve_model_path(self.model_name)
        model = SentenceTransformer(path, device=self.device)
        model.max_seq_length = self.max_length
        if self.use_fp16:
            try:
                model.half()
            except Exception:  # noqa: BLE001 - CPU 上不支持 half
                pass
        self._model = model
        return model

    # ---------- Embedder 接口 ----------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
