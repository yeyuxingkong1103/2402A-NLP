"""重排适配器：调用 BGE-reranker（硅基流动 API）。

为什么需要它：向量召回只保证"相关的不被漏掉"，不保证"最相关的排在最前"。
召回 20 条之后用重排模型精排出前 5 条，能把真正对得上问题的法条顶上去，
这是回答质量提升最明显的一步。
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# 传输函数签名：地址、请求头、请求体字节、超时 → 解析后的 JSON 字典
Transport = Callable[[str, dict[str, str], bytes, float], dict]


class RerankerApiError(RuntimeError):
    """重排服务调用失败（超时、HTTP 错误、返回结构无效）。"""


@dataclass(frozen=True)
class RankedCandidate:
    """重排结果：候选在输入列表中的位置，以及相关度分数。"""

    index: int
    score: float


class SiliconFlowRerankerClient:
    """调用硅基流动 OpenAI 风格的重排接口。"""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.transport = transport or self._request

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RankedCandidate]:
        """按相关度重排候选文本，返回降序结果。

        参数：
        - query：用户问题（原始表述即可）
        - documents：待重排的候选文本
        - top_n：只保留前 N 条；不传则返回全部

        返回：
        - RankedCandidate 列表，score 降序；index 指向入参 documents 的位置
        """
        if not query.strip():
            raise RerankerApiError("重排的查询文本不能为空")
        if not documents:
            return []
        if not self.api_base_url or not self.api_key or not self.model:
            raise RerankerApiError(
                "重排配置不完整，请检查 RERANKER_API_BASE_URL / RERANKER_API_KEY / RERANKER_MODEL"
            )

        request_body: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
        }
        if top_n is not None:
            request_body["top_n"] = top_n

        payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.transport(
                f"{self.api_base_url}", headers, payload, self.timeout
            )
        except TimeoutError as error:
            raise RerankerApiError("重排请求超时") from error
        except RerankerApiError:
            raise
        except Exception as error:
            raise RerankerApiError(f"重排请求失败：{type(error).__name__}") from error

        return self._extract_ranked_candidates(response, len(documents))

    @staticmethod
    def _extract_ranked_candidates(
        response: dict, document_count: int
    ) -> list[RankedCandidate]:
        """解析重排结果，并校验索引范围与分数有效性。"""
        items = response.get("results") if isinstance(response, dict) else None
        if not isinstance(items, list) or not items:
            raise RerankerApiError("重排返回结构无效")

        candidates: list[RankedCandidate] = []
        for item in items:
            try:
                index = int(item["index"])
                score = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError) as error:
                raise RerankerApiError("重排返回条目格式无效") from error
            # 索引越界说明服务端返回与请求不匹配，不能带着错位数据继续
            if index < 0 or index >= document_count:
                raise RerankerApiError("重排返回的索引超出候选范围")
            candidates.append(RankedCandidate(index=index, score=score))

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates

    def _request(
        self,
        url: str,
        headers: dict[str, str],
        payload: bytes,
        timeout: float,
    ) -> dict:
        """真实 HTTP 调用；错误信息只保留状态码，不回显请求内容。"""
        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RerankerApiError(f"重排 HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise RerankerApiError(f"重排网络请求失败：{error.reason}") from error
        except TimeoutError:
            raise
        except json.JSONDecodeError as error:
            raise RerankerApiError("重排返回 JSON 无效") from error


def build_reranker_from_settings() -> SiliconFlowRerankerClient:
    """按应用配置创建重排客户端，保证"配置从哪来"只有一个出处。"""
    from app.core.config import settings

    return SiliconFlowRerankerClient(
        api_base_url=settings.reranker_api_base_url,
        api_key=settings.reranker_api_key,
        model=settings.reranker_model,
        timeout=settings.reranker_timeout_seconds,
    )
