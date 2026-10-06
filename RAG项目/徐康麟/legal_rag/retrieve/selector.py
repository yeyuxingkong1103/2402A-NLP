"""LLM 列表式选择器：把「该法内部的宽候选」交给大模型挑条（P8 的最后一条未验证杠杆）。

为什么走到这一步（四轮实测的结论链）：
1. 候选池放大到 30 **更差**（v3c）；
2. 换重排器（large/base）条号级 Recall@5 只 +2.6% 且未达门槛；
3. 法内检索 + 名额保留两版实现**完全没改善**（cosine 反降）；
4. 术语映射（口语→立法术语）也没改善（噪声内偏负）。
⇒ 词面与通用向量/重排都抓不住"问题问的是**哪一条**"这层语义，需要一个**读得懂条文的判官**。
这里用现成的大模型做**列表式重排**（listwise rerank）：候选只有一部法的条文（几十到两百条），
让它直接挑"最能回答该问题的条"。**它是选择器，不是生成器** —— 不产生新文本，只给序号，
所以不会引入编造（选中的条文原文来自语料）。

工程口径：
* 候选只取「法内宽候选」，每条只给 条号 + 前 N 字（控 prompt 成本）；
* 输出严格 JSON（`{"picks": [...], "reason": "..."}`），**解析失败/越界/调用失败都必须看得见**
  （日志 + 返回值），绝不静默当成"没选出来"。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: 提示词：只让它**挑条**，不许它写答案（避免把"选择"变成"生成"）。这是**实测最好**的一版
#: （78 题、k=5、条号级）：便宜口径下 cosine Recall@5 0.5513、m3 0.5385。
SELECT_PROMPT = """你是中国法律检索专家。下面给出**用户问题**与**同一部法律的若干条文**。

要求：
1. 从候选里挑出**最能直接回答该问题**的条文，最多 {top_k} 条；
2. **只依据候选原文判断**，不要引入外部知识、不要补充法条；
3. 严格输出 JSON，不要解释：{{"picks": [序号1, 序号2], "reason": "不超过 20 字"}}
4. 若候选里确实没有一条能回答，输出 {{"picks": []}}。

用户问题：{question}

候选条文（序号从 1 开始）：
{numbered}
"""

#: **带版本标记**的提示词（`LAW_SELECTOR_VERSION_LABELS=true` 时才用）。
#: 实测（2026-09-24，78 题条号级）：m3 0.5385→**0.5256**、cosine 0.5513→**0.5385**（各 −1 题）
#: ⇒ **默认不用**。而且当初立这个改动的理由（"选择器挑了未生效新版的条号"）是**过期挑条缓存**
#: 造成的假象：候选池经版本过滤后重新挑条时，现行有效版的第五十七条本来就被排在第一位。
#: 代码留着：语料换成"新旧版条号差异更大"的场景时，标记可能才有价值 —— 到时要重新实测。
SELECT_PROMPT_WITH_VERSIONS = """你是中国法律检索专家。下面给出**用户问题**与**同一部法律的若干条文**。

要求：
1. 从候选里挑出**最能直接回答该问题**的条文，最多 {top_k} 条；
2. **只依据候选原文判断**，不要引入外部知识、不要补充法条；
3. 候选方括号里是**版本状态**：`现行有效` 优先选；`尚未施行` 的条文**只在问题明确问新规时**
   才选（例如"新法怎么规定""明年生效的版本"）；没有标记的按普通候选对待；
4. 严格输出 JSON，不要解释：{{"picks": [序号1, 序号2], "reason": "不超过 20 字"}}
5. 若候选里确实没有一条能回答，输出 {{"picks": []}}。

用户问题：{question}

候选条文（序号从 1 开始）：
{numbered}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def prompt_version(version_labels: dict | None = None) -> str:
    """当前提示词版本的标记（**挑条缓存按它作废**：喂给模型的东西变了，旧选择就不算数）。

    ``v1`` = 不带版本标记（默认，实测最好）；``v2`` = 候选带"现行有效/尚未施行"
    （实测各 −1 题，见 `SELECT_PROMPT_WITH_VERSIONS` 的注释）。
    """
    return "v2-versionlabelled" if version_labels else "v1"
#: 条号（与 prompt.first_article 同口径，单测里也用到）
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零〇两0-9]+条")


def article_of(text: str) -> str:
    """取条文正文里的第一个条号（没有就返回空串）。"""
    match = _ARTICLE_RE.search(text or "")
    return match.group(0) if match else ""


def build_candidates_block(hits: Sequence[Any], *, snippet_chars: int = 120,
                           max_candidates: int = 60,
                           version_labels: dict[str, str] | None = None) -> str:
    """把候选拼成编号清单（1-based）。纯函数，便于单测。

    ``version_labels`` 是 ``{来源 basename: 版本标记}``（如 ``现行有效`` / ``尚未施行(2027-01-01)``），
    没给或没命中就不带标记 —— 模型据此优先现行有效版（用户口径：现行有效版为准，涉新规要提示）。
    """
    labels = version_labels or {}
    lines: list[str] = []
    for index, hit in enumerate(hits[:max_candidates], start=1):
        chunk = getattr(hit, "chunk", None)
        source = str(getattr(chunk, "source", "") or "")
        body = str(getattr(chunk, "text", "") or "").strip().replace("\n", " ")
        article = article_of(body)
        head = f"[{index}]"
        label = labels.get(Path(source).name) or labels.get(source)
        if label:
            head += f"[{label}]"
        head += f" {source}"
        if article:
            head += f" {article}"
        lines.append(f"{head}\n    {body[:snippet_chars]}")
    return "\n".join(lines)


def parse_picks(text: str, *, count: int) -> tuple[list[int], str]:
    """从模型回复里抠出 picks（**0-based**），返回 ``(序号, 原因/错误)``。

    越界、非整数、重复一律丢弃并说明；解析不出来就返回空 + 原因（**不当成"没选出来"**）。
    """
    raw = str(text or "").strip()
    if not raw:
        return [], "空回复"
    match = _JSON_RE.search(raw)
    if not match:
        return [], "回复里没有 JSON"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], f"JSON 解析失败：{exc}"
    picks = payload.get("picks")
    if not isinstance(picks, list):
        return [], "JSON 里没有 picks 数组"
    reason = str(payload.get("reason") or "")
    chosen: list[int] = []
    for item in picks:
        try:
            number = int(item)
        except (TypeError, ValueError):
            logger.warning("[列表式选择器] 丢弃非整数序号：%r", item)
            continue
        if not 1 <= number <= count:
            logger.warning("[列表式选择器] 丢弃越界序号 %s（候选 %d 条）", number, count)
            continue
        if number - 1 not in chosen:
            chosen.append(number - 1)
    return chosen, reason


@dataclass
class ListwiseSelector:
    """用大模型从"法内宽候选"里挑条。``client`` 需要实现 ``chat(messages) -> str``。"""

    client: Any
    top_k: int = 3
    max_candidates: int = 60
    snippet_chars: int = 120
    #: 版本标记（``{来源 basename: "现行有效" | "尚未施行(2027-01-01)"}``）：由检索侧从
    #: 版本过滤器灌进来。空 dict = 候选不带版本标记（老行为）。
    version_labels: dict[str, str] = field(default_factory=dict)
    calls: int = 0
    last_reason: str = ""
    last_error: str = ""
    #: 最近一次模型原始回复（排障用：JSON 没解析出来时，**必须能看见模型到底说了什么**）
    last_reply: str = ""
    #: 累计调用耗时（秒）与累计 prompt 字符数 —— 用来算"每题多花多少秒"这个折中账
    seconds: float = 0.0
    prompt_chars: int = 0
    #: 连续失败次数（达到 ``max_fail_streak`` 就熔断：不再调用，避免大模型端点抖动时
    #: 每题都卡一次超时 —— 用户口径是"速度也不能太慢"）
    fail_streak: int = 0
    max_fail_streak: int = 3
    tripped: bool = False
    history: list[dict] = field(default_factory=list)

    def select(self, question: str, hits: Sequence[Any]) -> dict:
        """返回 ``{"picked": [hit...], "reason", "error", "candidates"}``；**绝不抛异常**。"""
        candidates = list(hits)[: self.max_candidates]
        outcome: dict = {"picked": [], "reason": "", "error": "", "candidates": len(candidates)}
        if not candidates:
            outcome["error"] = "没有候选"
            return outcome
        if self.tripped:
            # 已经熔断：直接放弃本轮（**不是静默**：计数与首次熔断日志都在）
            outcome["error"] = f"已熔断（连续失败 {self.fail_streak} 次），本轮跳过"
            return outcome
        prompt = (SELECT_PROMPT_WITH_VERSIONS if self.version_labels else SELECT_PROMPT).format(
            top_k=self.top_k, question=question,
            numbered=build_candidates_block(
                candidates, snippet_chars=self.snippet_chars,
                max_candidates=self.max_candidates,
                version_labels=self.version_labels))
        self.calls += 1
        self.prompt_chars += len(prompt)
        started = time.perf_counter()
        try:
            reply = self.client.chat([{"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001 - 选择器失败不能拖垮检索
            self.seconds += time.perf_counter() - started
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            self.last_error = outcome["error"]
            self.fail_streak += 1
            if self.fail_streak >= self.max_fail_streak and not self.tripped:
                # 熔断：大模型端点看起来不可用 -> 后面**不再调用**（每题多一次超时会把问答拖垮）
                self.tripped = True
                logger.warning("[列表式选择器] 连续失败 %d 次 -> **熔断**，后续请求不再调用选择器"
                               "（检索照常，只是少了「读一遍挑条」这一步）：最后一次原因=%s",
                               self.fail_streak, outcome["error"])
            else:
                logger.warning("[列表式选择器] 调用失败（本轮按原排序返回）：%s", outcome["error"])
            return outcome
        indices, reason = parse_picks(reply, count=len(candidates))
        self.seconds += time.perf_counter() - started
        self.fail_streak = 0                      # 成功即复位熔断计数
        self.last_reply = str(reply or "")
        self.last_reason, self.last_error = reason, ""
        if not indices:
            # 解析不出序号时，原始回复必须落日志（否则"模型为什么没挑出来"无从查证）
            logger.warning("[列表式选择器] 没解析出条号（%s）：原始回复前 200 字=%r",
                           reason, self.last_reply[:200])
        outcome["reason"] = reason
        outcome["picked"] = [candidates[i] for i in indices[: self.top_k]]
        outcome["indices"] = indices[: self.top_k]
        self.history.append({"question": question, "indices": outcome["indices"],
                             "reason": reason, "candidates": len(candidates),
                             "reply": self.last_reply[:400]})
        logger.info("[列表式选择器] 候选 %d 条 -> 选中 %s（%s）",
                    len(candidates), outcome["indices"], reason[:40])
        return outcome
