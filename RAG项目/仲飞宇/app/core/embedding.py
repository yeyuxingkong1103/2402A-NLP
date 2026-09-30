"""向量化抽象层：默认 Ollama bge-m3，OpenAI 兼容。

向量统一 L2 归一化，配合 Milvus 的 IP（内积）度量等价于余弦相似度。
支持 EMBED_PROVIDER=dummy 做离线冒烟测试（确定性哈希向量）。
"""
from __future__ import annotations

import hashlib
import math
import random
import time

from openai import OpenAI

from .config import Settings
from .logging_config import get_logger

log = get_logger("embedding")


class EmbeddingClient:
    def __init__(self, settings: Settings):
        # 构造期不碰网络（OpenAI() 只是建客户端，首次请求才连），所以启动与 /health
        # 都不会被 embedding 服务拖住；dummy 下干脆不建客户端，离线跑链路时才不会
        # 在构造处就抛连接错误。
        self.settings = settings
        self._client: OpenAI | None = None
        if settings.embed_provider != "dummy":
            self._client = OpenAI(
                base_url=settings.embed_effective_base_url,
                api_key=settings.embed_api_key or "ollama",
            )

    def ping(self) -> bool:
        # dummy 直接算「可达」：没有外部依赖可探，返回 False 只会让 /health 平白多一档
        # degraded，把真正的故障淹掉。
        if self.settings.embed_provider == "dummy":
            return True
        try:
            # 探活打的是一次真实的 embedding 请求，而不是像 llm.ping 那样列模型：
            # 这里要确认的是 /embeddings 这条路径真能算，模型没拉下来时列模型也照样成功。
            self.embed_texts(["ping"])
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Embedding 不可达（%s）: %s", self.settings.embed_effective_base_url, exc)
            return False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # 空批量直接返回，不发起请求：没有要算的东西，而各家服务端对 input=[] 的
        # 回应并不统一（400 / 空 data 都有），不该让一次空调用变成一次错误。
        if not texts:
            return []
        if self.settings.embed_provider == "dummy":
            return [self._dummy_embed(t) for t in texts]

        # 分批发，不整批发：整批在语料大一点时**必挂**——本机实测一次发 838 条，Ollama 直接
        # 400（它内部 tokenize 的子进程连接被拒），512 条正常。这不是性能优化，是可用性修复：
        # 分批前跑 scripts/seed.py 灌数据集语料（单个文件几千 chunk）必然中断在第一个文件上。
        # 批大小走 settings.embed_batch_size（默认 256），换 provider 时按对方上限调小。
        batch = max(1, self.settings.embed_batch_size)
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            part = texts[i : i + batch]
            resp = self._embed_one_batch(part)
            # 按 index 排序，保证输入输出顺序一致
            ordered = sorted(resp.data, key=lambda d: d.index)
            if len(ordered) != len(part):
                # 上游按批量上限截断、或丢项时，下游 zip(texts, vectors) 会静默丢尾块——
                # 回调方还会报「成功 N 块」（N 也跟着变小），日志和前端全都看不出少了。
                raise RuntimeError(f"Embedding 返回条数不符：请求 {len(part)} 条，实际返回 {len(ordered)} 条")
            out.extend(self._normalize(d.embedding) for d in ordered)
        return out

    def _embed_one_batch(self, texts: list[str], retries: int = 3):
        """单个批次调一次 /embeddings，失败重试。

        重试不是「以防万一」，本机实测 Ollama 会**奇偶交替**地失败（第 1 次成、第 2 次败、
        第 3 次成……），报错是它自己内部 tokenize 子进程的连接被拒——与请求内容无关，重试
        一次就过。它返回的是 HTTP 400，而 400 属于「不重试」类错误码，openai SDK 的
        max_retries（本项目的 LLM 侧更是显式设成 0）都不会管它，只能在这里自己重试。
        不加这一段，灌一次数据集语料必然中断在半路（实测三次里断两次）。
        """
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self._client.embeddings.create(model=self.settings.embed_model, input=texts)
            except Exception as exc:  # noqa: BLE001 —— 上游抖动是常态，见方法注释
                last = exc
                if attempt == retries:
                    break
                log.warning("Embedding 第 %d/%d 次失败（%s），重试", attempt, retries, exc)
                time.sleep(0.5 * attempt)  # 递增退避：连打太快只会在同一个竞态上反复撞
        raise RuntimeError(f"Embedding 连续 {retries} 次失败：{last}") from last

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        # 零向量（理论上不该出现，上游给空/全零时会发生）原样返回，避免除零崩在检索链路里；
        # 它在 IP 度量下与任何向量的内积都是 0，也就是相似度最低，不会凭空排到前面去。
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    def _dummy_embed(self, text: str) -> list[float]:
        """确定性伪向量：相同文本得到相同向量，仅用于离线链路联调。"""
        # 种子取自文本摘要，而不是让 random 用进程级种子：跨进程、重跑一次 ingest 之后
        # 同一句话必须还是同一个向量，否则新查询向量与库里旧向量对不上（同文本本应同向量）。
        digest = hashlib.md5(text.encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        # 维度跟真模型保持一致（embed_dim 默认 1024，与 bge-m3 对齐）：维度不同会导致
        # 切 provider 时 Milvus 直接拒绝写入，等于每次换后端都要重建集合。
        # 但要注意这些伪向量彼此之间没有语义相似性，不同文本只是不同的随机方向——
        # dummy 下的检索排序没有意义，只够验证「链路会不会断」，别拿它评效果。
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.settings.embed_dim)]
        return self._normalize(vec)
