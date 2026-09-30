# -*- coding: utf-8 -*-
"""
BGE-reranker 重排封装

职责：
    对检索召回的候选做**精排**，把真正相关的块提到前面。

与 RRF 的分工（见 ADR-023）：
    RRF       负责「把多路候选合并成一个候选池」—— 解决**召回**
    本模块    负责「在这个池子里排出最终顺序」—— 解决**排序**
    重排**替代** RRF 作为最终排序，而非叠加：两者分数尺度完全不同，
    混合无依据。

降级约定（见 ADR-024）：
    模型加载失败、推理异常、**推理超时**三种情形统一抛
    RerankerNotAvailableError（即：本模块对外**只抛这一种异常**），
    **调用方必须捕获并退回 RRF 原顺序**，绝不允许重排故障中断问答。

    为什么必须把「加载失败」也归一化：调用方按契约只捕
    RerankerNotAvailableError。若加载阶段的原始异常（目录存在但权重
    损坏时 transformers 抛的是 ValueError，不是 RuntimeError）直接逃逸，
    任何按文档编写的调用方都会在「模型坏了」这条路径上崩掉问答 ——
    正是 ADR-024 禁止的场景。

    为什么要超时：重排是本链路唯一的阻塞式重推理（实测 10 候选约 2.6s）。
    调用方的 except 只能接住**抛出的异常**，接不住**挂起**；一旦推理卡死，
    请求永不返回，N3（首次提问 ≤8 秒）直接被违反，RERANK_TIMEOUT 变成死配置。

为什么不需要 FlagEmbedding：
    bge-reranker-v2-m3 是标准的 CrossEncoder 结构（model.safetensors），
    sentence-transformers 可直接加载本地路径（同 ADR-014 的取舍）。
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


# 重排环节出任何问题，对外都只抛这一个异常类型。
# 好处是调用方（在线链路）写一个 except 就能兜住全部故障情形，
# 然后退回 RRF 的原始顺序继续答题 —— 而不是让一次重排故障把整个问答打断。
class RerankerNotAvailableError(RuntimeError):
    """重排模型不可用（路径不存在或依赖缺失）"""


class BGEReranker:
    """
    BGE-reranker 封装。

    线程安全的懒加载单例：首次调用时才加载权重（约 2.3GB），
    避免服务启动阶段耗时过长与无谓的内存占用。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        # 重排模型权重在磁盘上的位置（默认取配置 bge_reranker_path）；
        # device 是推理跑在 CPU 还是 GPU 上，与向量模型共用 embed_device 配置。
        self.model_path = model_path or settings.bge_reranker_path
        self.device = device or settings.embed_device
        # rerank_max_length / rerank_batch_size / rerank_timeout 均已在
        # backend/config.py（十三、重排）中定义，并可由 .env 覆盖，
        # 此处省略参数时即取配置值。getattr 的兜底默认值仅为防御性写法：
        # 与其他配置项保持一致，避免 settings 被替换成裁剪版对象时直接报错。
        # max_length：「问题 + 候选块」拼在一起后最多保留多少 token，
        # 超出部分直接截掉。这是重排耗时的主要来源，调小能明显提速。
        self.max_length = max_length or getattr(settings, "rerank_max_length", 256)
        # batch_size：一次推理同时送入多少对「问题-候选块」。
        self.batch_size = batch_size or getattr(settings, "rerank_batch_size", 8)
        # timeout：单次重排推理最多等多少秒，超时就降级退回 RRF 顺序。
        self.timeout = timeout or getattr(settings, "rerank_timeout", 30)
        # 下面两个字段服务于"懒加载"：_model 先留空，等第一次真要用时才去加载权重；
        # _lock 保证并发请求同时触发加载时，只有一个人真去加载。
        self._model: Any = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------------

    def _load(self) -> Any:
        # 第一次检查故意放在加锁之前：模型已经加载好时直接返回，
        # 不必去抢锁，让绝大多数请求都走无锁的快路径。
        if self._model is not None:
            return self._model

        # 拿到锁后再检查一次（双重检查）：等锁的这段时间里，
        # 可能已经有别的请求把模型加载完了，这里就不必重复加载。
        with self._lock:
            if self._model is not None:
                return self._model

            from pathlib import Path

            # 路径就不存在的话直接判为"重排不可用"，不必再往下走加载流程。
            # 这是最常见的部署问题（.env 里 BGE_RERANKER_PATH 写错，或权重没下载）。
            if not Path(self.model_path).exists():
                raise RerankerNotAvailableError(
                    f"重排模型路径不存在：{self.model_path}\n"
                    f"请检查 .env 中的 BGE_RERANKER_PATH 配置"
                )

            # 依赖没装（缺 sentence-transformers）也归一化成"重排不可用"，
            # 而不是把 ImportError 抛出去：运行环境没装重排依赖时，
            # 系统仍然应该能用 RRF 的顺序正常答题。
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerNotAvailableError(
                    "加载重排模型需要 sentence-transformers，请先安装：\n"
                    "    pip install sentence-transformers"
                ) from exc

            logger.info(
                "正在加载 BGE-reranker：%s（设备=%s，max_length=%d）",
                self.model_path, self.device, self.max_length,
            )
            # 真正把权重载入内存（约 2.3GB，所以首次调用这里会明显慢一下，
            # 之后就由 self._model 缓存住，不再重复加载）。
            try:
                model = CrossEncoder(
                    self.model_path,
                    device=self.device,
                    max_length=self.max_length,
                )
            except Exception as exc:
                # 归一化：目录存在但权重缺失/损坏、下载不完整、架构不兼容、
                # device 不可用时，transformers 抛的是 ValueError / OSError /
                # RuntimeError 等各不相同的原始异常，一律收敛为本模块契约的
                # RerankerNotAvailableError，保证调用方的降级分支一定生效。
                raise RerankerNotAvailableError(
                    f"重排模型加载失败：{self.model_path}\n{exc}"
                ) from exc

            self._model = model
            logger.info("BGE-reranker 加载完成")
            return self._model

    # ----------------------------------------------------------------
    # 重排
    # ----------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        对候选重排。

        参数：
            query      : 用户**原始问题**（不是改写后的关键词串）。
                         理由：重排模型在自然语言问答数据上训练，用完整问句
                         更契合其训练分布；改写的关键词串是为字面检索优化的。
            candidates : 候选列表，每项至少含 content
            top_k      : 返回条数，None 表示全部返回

        返回：
            新列表，按 rerank_score 降序；**保留候选原有全部字段**，
            并新增 rerank_score。原列表不被修改。

        抛出：
            RerankerNotAvailableError —— 模型不可用、加载失败、推理失败或
            推理超时（> ``self.timeout`` 秒）。**这是本方法对外抛出的唯一
            异常类型**，调用方捕获它并退回 RRF 原顺序即可完成降级。
        """
        # 一个候选都没有就直接返回空，连模型都不用去加载。
        # 上游检索全空时走的就是这条路径。
        if not candidates:
            return []

        model = self._load()
        # ★ 重排的核心动作：把「用户问题」和每个候选块的正文配成一个"对" ★
        # 这正是 CrossEncoder（交叉编码器）与向量检索（双编码器）的根本差别：
        # 向量检索是问题和块**分开**编码、最后算距离，块与问题之间无从交互；
        # 而这里是把问题和块**一起**喂给模型，让模型直接看到两者的字面交互，
        # 因此判断"这一块到底能不能回答这个问题"要准得多 —— 代价是慢，
        # 所以只对已经召回的少量候选做，不能用来扫全库。
        pairs = [(query, str(c.get("content", ""))) for c in candidates]

        # 交给带超时保护的推理。无论超时还是推理失败，都会抛
        # RerankerNotAvailableError，由调用方接住并退回 RRF 顺序。
        scores = self._predict_with_timeout(model, pairs)

        # 把模型吐出来的裸分数贴回各个候选块上。整段包在 try 里，
        # 保证无论解析出什么岔子，对外都只抛那一种契约异常。
        try:
            # num_labels>1 的权重会返回向量而非标量，float() 会抛 TypeError，
            # 同样必须收敛为 RerankerNotAvailableError 而非逃逸。
            # 条数对不上说明模型输出和输入没对齐，后面的 zip 会静默错配，
            # 把 A 块的分数贴到 B 块头上 —— 这种错误必须当场发现。
            if len(scores) != len(candidates):
                raise ValueError(
                    f"打分条数与候选数不一致：{len(scores)} != {len(candidates)}"
                )
            scored: List[Dict[str, Any]] = []
            # 逐个候选贴上它的重排分。转成 float 是为了后面排序时数值可比。
            for cand, score in zip(candidates, scores):
                enriched = dict(cand)                # 不修改调用方的字典
                enriched["rerank_score"] = float(score)
                scored.append(enriched)
        except Exception as exc:
            raise RerankerNotAvailableError(f"重排打分结果解析失败：{exc}") from exc

        # 按重排分从高到低排（加负号即为降序）。
        # 这一排序结果就是最终送进大模型的上下文顺序：排得越准，
        # 真正能回答问题的块越可能排在前面、在截断之前被留下，
        # 大模型也越不容易被排在中间的不相关块带偏。
        scored.sort(key=lambda c: -c["rerank_score"])
        # top_k 为 None 表示不截断，把全部候选都交出去。
        return scored[:top_k] if top_k else scored

    # ----------------------------------------------------------------
    # 带超时的推理
    # ----------------------------------------------------------------

    def _predict_with_timeout(
        self, model: Any, pairs: List[Tuple[str, str]]
    ) -> Sequence[float]:
        """
        带超时的推理调用。所有异常（含超时）归一化为 RerankerNotAvailableError。

        为什么需要超时：调用方只能捕获抛出的异常，捕不住**挂起**。重排是本
        链路唯一的阻塞式重推理（实测 10 候选约 2.6~3.0s），一旦卡死，请求
        永不返回，N3（≤8 秒）直接被违反，RERANK_TIMEOUT 也变成死配置。

        为什么用 daemon 线程 + 队列，而**不用** ThreadPoolExecutor
        （含「显式 shutdown(wait=False)」的写法）：
            CPython 3.9+ 的 ThreadPoolExecutor 工作线程是**非 daemon** 线程，
            concurrent.futures.thread 在 atexit 注册的 _python_exit 会 join
            所有遗留线程。因此 shutdown(wait=False) 只是让**调用方**在超时点
            拿回控制权，**并不减少进程墙钟时间**。
            本模块修复轮实测（任务本身 3s、timeout=1，bash time 量一次性进程）：
                ThreadPoolExecutor 版：rerank() 1.01s 返回，进程 real 3.403s
                （退出时被 _python_exit join，多等约 2.4s）；
                本实现（daemon）：rerank() 约 1.0s 返回，进程 real 约 1.0s。
            长驻服务不受影响，但 Task 9 的对照实验脚本是**一次性进程**：一旦
            触发重排超时（默认 30s），脚本会在结束处假死到推理自然结束（最坏
            拖 30s），污染对照实验的墙钟统计，并可能被误判为死锁。
            daemon 线程**不会**被 _python_exit join，进程可立即退出。

        取舍：线程无法强杀。超时后该 daemon 线程仍会把这次推理跑完（白耗一次
            推理、占一份内存），只是不再阻塞任何退出路径。代价可接受：宁可
            浪费一次推理，也不能拖住问答或污染评测墙钟。
        """
        # 子线程把推理结果（或异常）回传给主线程的通道，只放得下一条。
        result_q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=1)

        # 子线程里真正干活的推理：model.predict 一次性给所有 pairs 打分，
        # 一次送多少条由 batch_size 控制。这里捕获 BaseException 是为了让
        # 子线程任何形式的失败都能回传，否则主线程只能一路干等到超时。
        def _worker() -> None:
            try:
                result_q.put((
                    "ok",
                    model.predict(
                        pairs,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                    ),
                ))
            except BaseException as exc:  # noqa: BLE001 - 需覆盖一切
                result_q.put(("err", exc))

        try:
            # daemon=True 是重点：进程退出时不会被 atexit 的 _python_exit join。
            #
            # ★ 线程的创建与启动必须留在 try 之内 ★
            #   线程/FD 耗尽时 start() 会抛 RuntimeError("can't start new thread")。
            #   该异常必须被下面的 except Exception 接住并归一化，否则会击穿本模块
            #   「对外只抛 RerankerNotAvailableError」的契约 —— 长驻服务里超时
            #   泄漏的 daemon 线程会累积，使其并非纯理论。
            threading.Thread(target=_worker, daemon=True).start()

            # 等推理结果，最多等 timeout 秒。等不到会抛 queue.Empty，
            # 由紧跟着的分支转成"重排不可用"。
            kind, value = result_q.get(timeout=self.timeout)
        except queue.Empty as exc:
            # 注意：queue.Empty 是 Exception 的子类，本分支必须排在
            # except Exception 之前，否则会被后者吞掉、丢失「超时」语义。
            logger.warning("重排推理超时（>%ss），降级退回 RRF 顺序", self.timeout)
            raise RerankerNotAvailableError(
                f"重排推理超时（超过 {self.timeout} 秒）"
            ) from exc
        except Exception as exc:
            raise RerankerNotAvailableError(f"重排推理失败：{exc}") from exc

        # 子线程推理报错了：统一转成契约异常往上报，绝不让原始异常逃出去。
        if kind == "err":
            # 工作线程的异常经队列回传后在此重抛。只有 Exception 才重抛并保留
            # 异常链；BaseException 中不属于 Exception 的部分（KeyboardInterrupt
            # / SystemExit）若在主线程重抛，会击穿「对外只抛一种异常」的契约，
            # 故就地降级为本模块的契约异常。
            if isinstance(value, Exception):
                raise RerankerNotAvailableError(f"重排推理失败：{value}") from value
            logger.warning("重排推理被中断，降级退回 RRF 顺序：%r", value)
            raise RerankerNotAvailableError(f"重排推理被中断：{value!r}")

        return value


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

# 进程级单例。重排权重有 2.3GB，全进程必须只加载一份。
_reranker: Optional[BGEReranker] = None
# 保护单例创建的锁，避免多个请求同时进来时重复加载模型。
_reranker_lock = threading.Lock()


def get_reranker() -> BGEReranker:
    """获取重排模型单例（进程内只加载一次权重）"""
    # 同样是双重检查加锁：已经建好实例时就不再去抢锁。
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = BGEReranker()
    return _reranker
